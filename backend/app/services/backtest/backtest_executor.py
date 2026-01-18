"""
回测执行器 - 完整的回测流程执行和结果分析
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
from functools import partial

from .backtest_engine import (
    BaseStrategy, StrategyFactory, PortfolioManager, BacktestConfig,
    TradingSignal, Trade, Position
)
from .strategies import AdvancedStrategyFactory
from .backtest_progress_monitor import backtest_progress_monitor
from app.core.error_handler import TaskError, ErrorSeverity, ErrorContext
from app.models.task_models import BacktestResult


class DataLoader:
    """数据加载器"""
    
    def __init__(self, data_dir: str = "backend/data", max_workers: Optional[int] = None):
        self.data_dir = Path(data_dir)
        self.max_workers = max_workers  # 用于并行加载数据
    
    def load_stock_data(self, stock_code: str, start_date: datetime, 
                       end_date: datetime) -> pd.DataFrame:
        """加载股票历史数据"""
        try:
            # 使用统一的数据加载器
            from app.services.data.stock_data_loader import StockDataLoader
            loader = StockDataLoader(data_root=str(self.data_dir))
            
            # 加载数据
            data = loader.load_stock_data(stock_code, start_date=start_date, end_date=end_date)
            
            if data.empty:
                raise TaskError(
                    message=f"未找到股票数据文件: {stock_code}",
                    severity=ErrorSeverity.HIGH,
                    context=ErrorContext(stock_code=stock_code)
                )
            
            if len(data) == 0:
                raise TaskError(
                    message=f"指定日期范围内无数据: {stock_code}, {start_date} - {end_date}",
                    severity=ErrorSeverity.MEDIUM,
                    context=ErrorContext(stock_code=stock_code)
                )
            
            # 验证必需的列
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                raise TaskError(
                    message=f"数据缺少必需列: {missing_columns}",
                    severity=ErrorSeverity.HIGH,
                    context=ErrorContext(stock_code=stock_code)
                )
            
            # 添加股票代码属性
            data.attrs['stock_code'] = stock_code
            
            logger.info(f"加载股票数据成功: {stock_code}, 数据量: {len(data)}, 日期范围: {data.index[0]} - {data.index[-1]}")
            return data
            
        except TaskError:
            raise
        except Exception as e:
            raise TaskError(
                message=f"加载股票数据失败: {str(e)}",
                severity=ErrorSeverity.HIGH,
                context=ErrorContext(stock_code=stock_code),
                original_exception=e
            )
    
    def load_multiple_stocks(self, stock_codes: List[str], start_date: datetime,
                           end_date: datetime, parallel: bool = True) -> Dict[str, pd.DataFrame]:
        """
        加载多只股票数据
        
        Args:
            stock_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            parallel: 是否并行加载（默认True）
        """
        stock_data = {}
        failed_stocks = []
        
        if parallel and len(stock_codes) > 1 and self.max_workers:
            # 并行加载多只股票数据
            max_workers = min(self.max_workers, len(stock_codes))
            logger.info(f"并行加载 {len(stock_codes)} 只股票数据，使用 {max_workers} 个线程")
            
            def load_single_stock(stock_code: str) -> Tuple[str, Optional[pd.DataFrame], Optional[str]]:
                """加载单只股票数据，返回 (stock_code, data, error)"""
                try:
                    data = self.load_stock_data(stock_code, start_date, end_date)
                    return (stock_code, data, None)
                except Exception as e:
                    error_msg = str(e)
                    logger.error(f"加载股票数据失败: {stock_code}, 错误: {error_msg}")
                    return (stock_code, None, error_msg)
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(load_single_stock, code): code for code in stock_codes}
                
                for future in as_completed(futures):
                    stock_code, data, error = future.result()
                    if data is not None:
                        stock_data[stock_code] = data
                    else:
                        failed_stocks.append(stock_code)
        else:
            # 顺序加载（兼容旧逻辑）
            for stock_code in stock_codes:
                try:
                    data = self.load_stock_data(stock_code, start_date, end_date)
                    stock_data[stock_code] = data
                except Exception as e:
                    logger.error(f"加载股票数据失败: {stock_code}, 错误: {e}")
                    failed_stocks.append(stock_code)
                    continue
        
        if failed_stocks:
            logger.warning(f"部分股票数据加载失败: {failed_stocks}")
        
        if not stock_data:
            raise TaskError(
                message="所有股票数据加载失败",
                severity=ErrorSeverity.HIGH
            )
        
        return stock_data


class BacktestExecutor:
    """回测执行器"""
    
    def __init__(self, data_dir: str = "backend/data", enable_parallel: bool = True, 
                 max_workers: Optional[int] = None):
        """
        初始化回测执行器
        
        Args:
            data_dir: 数据目录
            enable_parallel: 是否启用并行化（默认True）
            max_workers: 最大工作线程数，默认使用CPU核心数
        """
        import os
        if max_workers is None:
            max_workers = min(os.cpu_count() or 4, 8)  # 最多8个线程，避免过多线程导致开销
        
        self.enable_parallel = enable_parallel
        self.max_workers = max_workers
        self.data_loader = DataLoader(data_dir, max_workers=max_workers if enable_parallel else None)
        self.execution_stats = {
            "total_backtests": 0,
            "successful_backtests": 0,
            "failed_backtests": 0
        }
        
        if enable_parallel:
            logger.info(f"回测执行器已启用并行化，最大工作线程数: {max_workers}")
    
    async def run_backtest(self, strategy_name: str, stock_codes: List[str],
                    start_date: datetime, end_date: datetime,
                    strategy_config: Dict[str, Any],
                    backtest_config: Optional[BacktestConfig] = None,
                    task_id: str = None) -> Dict[str, Any]:
        """运行回测"""
        try:
            self.execution_stats["total_backtests"] += 1
            
            # 生成回测ID
            backtest_id = f"bt_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(str(stock_codes))}"
            
            # 使用默认配置
            if backtest_config is None:
                backtest_config = BacktestConfig()
            
            # 开始进度监控
            if task_id:
                await backtest_progress_monitor.start_backtest_monitoring(
                    task_id=task_id,
                    backtest_id=backtest_id
                )
                await backtest_progress_monitor.update_stage(
                    task_id, "initialization", progress=100, status="completed"
                )
            
            # 创建策略
            if task_id:
                await backtest_progress_monitor.update_stage(
                    task_id, "strategy_setup", status="running"
                )
            
            # 优先使用高级策略工厂
            try:
                strategy = AdvancedStrategyFactory.create_strategy(strategy_name, strategy_config)
            except Exception:
                # 如果高级策略工厂没有该策略，回退到基础策略工厂
                strategy = StrategyFactory.create_strategy(strategy_name, strategy_config)
            
            if task_id:
                await backtest_progress_monitor.update_stage(
                    task_id, "strategy_setup", progress=100, status="completed"
                )
            
            # 创建组合管理器
            portfolio_manager = PortfolioManager(backtest_config)
            
            # 加载数据
            if task_id:
                await backtest_progress_monitor.update_stage(
                    task_id, "data_loading", status="running"
                )
            
            logger.info(f"开始回测: {strategy_name}, 股票: {stock_codes}, 期间: {start_date} - {end_date}")
            stock_data = self.data_loader.load_multiple_stocks(stock_codes, start_date, end_date)
            
            if task_id:
                await backtest_progress_monitor.update_stage(
                    task_id, "data_loading", progress=100, status="completed"
                )
            
            # 获取交易日历
            trading_dates = self._get_trading_calendar(stock_data, start_date, end_date)
            
            if len(trading_dates) < 20:
                error_msg = f"交易日数量不足: {len(trading_dates)}，至少需要20个交易日"
                if task_id:
                    await backtest_progress_monitor.set_error(task_id, error_msg)
                raise TaskError(
                    message=error_msg,
                    severity=ErrorSeverity.MEDIUM
                )
            
            # 更新总交易日数（同时写入数据库）
            if task_id:
                progress_data = backtest_progress_monitor.get_progress_data(task_id)
                if progress_data:
                    progress_data.total_trading_days = len(trading_dates)
                
                # 将总交易日数写入数据库
                try:
                    from app.core.database import SessionLocal
                    from app.repositories.task_repository import TaskRepository
                    from app.models.task_models import TaskStatus
                    
                    session = SessionLocal()
                    try:
                        task_repo = TaskRepository(session)
                        existing_task = task_repo.get_task_by_id(task_id)
                        if existing_task:
                            result_data = existing_task.result or {}
                            progress_data_db = result_data.get('progress_data', {})
                            progress_data_db['total_days'] = len(trading_dates)
                            result_data['progress_data'] = progress_data_db
                            
                            task_repo.update_task_status(
                                task_id=task_id,
                                status=TaskStatus.RUNNING,
                                result=result_data
                            )
                    finally:
                        session.close()
                except Exception as e:
                    logger.warning(f"更新总交易日数失败: {e}")
            
            # 执行回测
            if task_id:
                await backtest_progress_monitor.update_stage(
                    task_id, "backtest_execution", status="running"
                )
            
            backtest_results = await self._execute_backtest_loop(
                strategy, portfolio_manager, stock_data, trading_dates, task_id, backtest_id
            )
            
            if task_id:
                await backtest_progress_monitor.update_stage(
                    task_id, "backtest_execution", progress=100, status="completed"
                )
            
            # 计算绩效指标
            if task_id:
                await backtest_progress_monitor.update_stage(
                    task_id, "metrics_calculation", status="running"
                )
            
            performance_metrics = portfolio_manager.get_performance_metrics()
            
            if task_id:
                await backtest_progress_monitor.update_stage(
                    task_id, "metrics_calculation", progress=100, status="completed"
                )
            
            # 生成回测报告
            if task_id:
                await backtest_progress_monitor.update_stage(
                    task_id, "report_generation", status="running"
                )
            
            backtest_report = self._generate_backtest_report(
                strategy_name, stock_codes, start_date, end_date,
                backtest_config, portfolio_manager, performance_metrics
            )
            
            if task_id:
                await backtest_progress_monitor.update_stage(
                    task_id, "report_generation", progress=100, status="completed"
                )
                await backtest_progress_monitor.update_stage(
                    task_id, "data_storage", progress=100, status="completed"
                )
            
            self.execution_stats["successful_backtests"] += 1
            logger.info(f"回测完成: {strategy_name}, 总收益: {performance_metrics.get('total_return', 0):.2%}")
            
            # 完成监控
            if task_id:
                await backtest_progress_monitor.complete_backtest(
                    task_id, {"total_return": performance_metrics.get('total_return', 0)}
                )
            
            return backtest_report
            
        except Exception as e:
            self.execution_stats["failed_backtests"] += 1
            error_msg = f"回测执行失败: {str(e)}"
            
            if task_id:
                await backtest_progress_monitor.set_error(task_id, error_msg)
            
            raise TaskError(
                message=error_msg,
                severity=ErrorSeverity.HIGH,
                original_exception=e
            )
    
    def _get_trading_calendar(self, stock_data: Dict[str, pd.DataFrame],
                            start_date: datetime, end_date: datetime) -> List[datetime]:
        """获取交易日历"""
        # 合并所有股票的交易日期
        all_dates = set()
        for data in stock_data.values():
            all_dates.update(data.index.tolist())
        
        # 过滤日期范围并排序
        trading_dates = sorted([
            date for date in all_dates 
            if start_date <= date <= end_date
        ])
        
        return trading_dates
    
    async def _execute_backtest_loop(self, strategy: BaseStrategy, 
                             portfolio_manager: PortfolioManager,
                             stock_data: Dict[str, pd.DataFrame],
                             trading_dates: List[datetime],
                             task_id: str = None,
                             backtest_id: str = None) -> Dict[str, Any]:
        """执行回测主循环"""
        total_signals = 0
        executed_trades = 0
        
        for i, current_date in enumerate(trading_dates):
            try:
                # 获取当前价格
                current_prices = {}
                for stock_code, data in stock_data.items():
                    if current_date in data.index:
                        current_prices[stock_code] = data.loc[current_date, 'close']
                
                if not current_prices:
                    continue
                
                # 生成交易信号（支持并行生成多股票信号）
                all_signals = []
                
                if self.enable_parallel and len(stock_data) > 3:
                    # 并行生成多股票信号
                    def generate_stock_signals(stock_code: str, data: pd.DataFrame) -> List[TradingSignal]:
                        """为单只股票生成信号（用于并行执行）"""
                        if current_date in data.index:
                            historical_data = data[data.index <= current_date]
                            if len(historical_data) >= 20:  # 确保有足够的历史数据
                                try:
                                    return strategy.generate_signals(historical_data, current_date)
                                except Exception as e:
                                    logger.warning(f"生成信号失败 {stock_code}: {e}")
                                    return []
                        return []
                    
                    # 使用线程池并行生成信号
                    with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                        futures = {
                            executor.submit(generate_stock_signals, code, data): code 
                            for code, data in stock_data.items()
                        }
                        
                        for future in as_completed(futures):
                            try:
                                signals = future.result()
                                all_signals.extend(signals)
                            except Exception as e:
                                stock_code = futures[future]
                                logger.error(f"并行生成信号失败 {stock_code}: {e}")
                else:
                    # 顺序生成信号（股票数量少或禁用并行）
                    for stock_code, data in stock_data.items():
                        if current_date in data.index:
                            # 获取到当前日期的历史数据
                            historical_data = data[data.index <= current_date]
                            if len(historical_data) >= 20:  # 确保有足够的历史数据
                                try:
                                    signals = strategy.generate_signals(historical_data, current_date)
                                    all_signals.extend(signals)
                                except Exception as e:
                                    logger.warning(f"生成信号失败 {stock_code}: {e}")
                                    continue
                
                total_signals += len(all_signals)
                
                # 保存信号记录到数据库
                if task_id and all_signals:
                    try:
                        from app.core.database import get_async_session
                        from app.repositories.backtest_detailed_repository import BacktestDetailedRepository
                        import uuid
                        
                        # 使用传入的backtest_id或生成一个
                        current_backtest_id = backtest_id or (f"bt_{task_id[:8]}" if task_id else f"bt_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                        
                        # 批量保存信号记录
                        signals_data = []
                        for signal in all_signals:
                            signal_data = {
                                'signal_id': f"sig_{uuid.uuid4().hex[:12]}",
                                'stock_code': signal.stock_code,
                                'stock_name': None,  # 可以从股票数据中获取
                                'signal_type': signal.signal_type.name,
                                'timestamp': signal.timestamp,
                                'price': signal.price,
                                'strength': signal.strength,
                                'reason': signal.reason,
                                'metadata': signal.metadata,
                                'executed': False
                            }
                            signals_data.append(signal_data)
                        
                        # 异步保存信号记录
                        async for session in get_async_session():
                            try:
                                repository = BacktestDetailedRepository(session)
                                await repository.batch_save_signal_records(
                                    task_id=task_id,
                                    backtest_id=current_backtest_id,
                                    signals_data=signals_data
                                )
                                await session.commit()
                                break
                            except Exception as e:
                                await session.rollback()
                                logger.warning(f"保存信号记录失败: {e}")
                                break
                    except Exception as e:
                        logger.warning(f"保存信号记录时出错: {e}")
                
                # 执行交易信号
                trades_this_day = 0
                executed_trade_signals = []  # 记录已执行的交易对应的信号
                for signal in all_signals:
                    if strategy.validate_signal(signal, 
                                              portfolio_manager.get_portfolio_value(current_prices),
                                              portfolio_manager.positions):
                        trade = portfolio_manager.execute_signal(signal, current_prices)
                        if trade:
                            executed_trades += 1
                            trades_this_day += 1
                            # 记录已执行的信号，用于后续标记
                            executed_trade_signals.append({
                                'stock_code': signal.stock_code,
                                'timestamp': signal.timestamp,
                                'signal_type': signal.signal_type.name
                            })
                
                # 标记已执行的信号
                if task_id and executed_trade_signals:
                    try:
                        from app.core.database import get_async_session
                        from app.repositories.backtest_detailed_repository import BacktestDetailedRepository
                        
                        async for session in get_async_session():
                            try:
                                repository = BacktestDetailedRepository(session)
                                for executed_signal in executed_trade_signals:
                                    await repository.mark_signal_as_executed(
                                        task_id=task_id,
                                        stock_code=executed_signal['stock_code'],
                                        timestamp=executed_signal['timestamp'],
                                        signal_type=executed_signal['signal_type']
                                    )
                                await session.commit()
                                break
                            except Exception as e:
                                await session.rollback()
                                logger.warning(f"标记信号为已执行失败: {e}")
                                break
                    except Exception as e:
                        logger.warning(f"标记信号为已执行时出错: {e}")
                
                # 记录组合快照
                portfolio_manager.record_portfolio_snapshot(current_date, current_prices)
                
                # 更新进度监控（同时更新数据库）
                if task_id and i % 5 == 0:  # 每5天更新一次进度
                    portfolio_value = portfolio_manager.get_portfolio_value(current_prices)
                    logger.debug(f"准备更新进度: task_id={task_id}, i={i}, total_days={len(trading_dates)}, signals={len(all_signals)}, trades={trades_this_day}, total_signals={total_signals}, total_trades={executed_trades}")
                    
                    # 计算进度百分比（回测执行阶段占30-90%，即60%的进度范围）
                    execution_progress = (i + 1) / len(trading_dates) * 100
                    overall_progress = 30 + (execution_progress / 100) * 60  # 30%到90%
                    
                    # 更新数据库中的任务进度（包含详细数据）
                    try:
                        from app.core.database import SessionLocal
                        from app.repositories.task_repository import TaskRepository
                        from app.models.task_models import TaskStatus
                        from datetime import datetime
                        
                        session = SessionLocal()
                        try:
                            task_repo = TaskRepository(session)
                            
                            # 读取现有的 result 数据
                            existing_task = task_repo.get_task_by_id(task_id)
                            if not existing_task:
                                logger.warning(f"任务不存在，无法更新进度: {task_id}")
                            else:
                                result_data = existing_task.result or {}
                                if not isinstance(result_data, dict):
                                    result_data = {}
                                progress_data = result_data.get('progress_data', {})
                                if not isinstance(progress_data, dict):
                                    progress_data = {}
                                
                                # 更新进度数据
                                progress_data.update({
                                    'processed_days': i + 1,
                                    'total_days': len(trading_dates),
                                    'current_date': current_date.strftime('%Y-%m-%d'),
                                    'signals_generated': len(all_signals),
                                    'trades_executed': trades_this_day,
                                    'total_signals': total_signals,
                                    'total_trades': executed_trades,
                                    'portfolio_value': portfolio_value,
                                    'last_updated': datetime.utcnow().isoformat()
                                })
                                
                                result_data['progress_data'] = progress_data
                                
                                # 记录日志以便调试
                                logger.info(f"更新回测进度数据: task_id={task_id}, processed_days={i+1}, total_days={len(trading_dates)}, signals={total_signals}, trades={executed_trades}, portfolio={portfolio_value:.2f}, progress_data_keys={list(progress_data.keys())}")
                                
                                task_repo.update_task_status(
                                    task_id=task_id,
                                    status=TaskStatus.RUNNING,
                                    progress=overall_progress,
                                    result=result_data  # 包含详细进度数据
                                )
                                
                                # 确保 result 字段被标记为已修改并提交
                                session.commit()
                                logger.info(f"进度数据已提交到数据库: task_id={task_id}, result_data_keys={list(result_data.keys())}, progress_data={progress_data}")
                        except Exception as inner_error:
                            session.rollback()
                            logger.error(f"更新任务进度到数据库失败（内部错误）: {inner_error}", exc_info=True)
                            raise
                        finally:
                            session.close()
                    except Exception as db_error:
                        logger.error(f"更新任务进度到数据库失败: {db_error}", exc_info=True)
                    
                    # 更新进程内的进度监控（虽然主进程看不到，但保持一致性）
                    await backtest_progress_monitor.update_execution_progress(
                        task_id=task_id,
                        processed_days=i + 1,
                        current_date=current_date.strftime('%Y-%m-%d'),
                        signals_generated=len(all_signals),
                        trades_executed=trades_this_day,
                        portfolio_value=portfolio_value
                    )
                
                # 定期输出进度日志
                if i % 50 == 0:
                    progress = (i + 1) / len(trading_dates) * 100
                    portfolio_value = portfolio_manager.get_portfolio_value(current_prices)
                    logger.debug(f"回测进度: {progress:.1f}%, 组合价值: {portfolio_value:.2f}")
                
            except Exception as e:
                error_msg = f"回测循环错误，日期: {current_date}, 错误: {e}"
                logger.error(error_msg)
                
                # 添加警告到进度监控
                if task_id:
                    await backtest_progress_monitor.add_warning(task_id, error_msg)
                
                continue
        
        # 最终进度更新
        if task_id:
            final_portfolio_value = portfolio_manager.get_portfolio_value({})
            await backtest_progress_monitor.update_execution_progress(
                task_id=task_id,
                processed_days=len(trading_dates),
                current_date=trading_dates[-1].strftime('%Y-%m-%d') if trading_dates else None,
                signals_generated=0,
                trades_executed=0,
                portfolio_value=final_portfolio_value
            )
        
        return {
            "total_signals": total_signals,
            "executed_trades": executed_trades,
            "trading_days": len(trading_dates)
        }
    
    def _generate_backtest_report(self, strategy_name: str, stock_codes: List[str],
                                start_date: datetime, end_date: datetime,
                                config: BacktestConfig, portfolio_manager: PortfolioManager,
                                performance_metrics: Dict[str, float]) -> Dict[str, Any]:
        """生成回测报告"""
        
        # 基础信息
        report = {
            "strategy_name": strategy_name,
            "stock_codes": stock_codes,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "initial_cash": config.initial_cash,
            "final_value": portfolio_manager.get_portfolio_value({}),
            
            # 收益指标
            "total_return": performance_metrics.get("total_return", 0),
            "annualized_return": performance_metrics.get("annualized_return", 0),
            
            # 风险指标
            "volatility": performance_metrics.get("volatility", 0),
            "sharpe_ratio": performance_metrics.get("sharpe_ratio", 0),
            "max_drawdown": performance_metrics.get("max_drawdown", 0),
            
            # 交易统计
            "total_trades": performance_metrics.get("total_trades", 0),
            "win_rate": performance_metrics.get("win_rate", 0),
            "profit_factor": performance_metrics.get("profit_factor", 0),
            "winning_trades": performance_metrics.get("winning_trades", 0),
            "losing_trades": performance_metrics.get("losing_trades", 0),
            
            # 将指标也放在 metrics 字段中，方便优化器使用
            "metrics": {
                "sharpe_ratio": performance_metrics.get("sharpe_ratio", 0),
                "total_return": performance_metrics.get("total_return", 0),
                "annualized_return": performance_metrics.get("annualized_return", 0),
                "max_drawdown": performance_metrics.get("max_drawdown", 0),
                "volatility": performance_metrics.get("volatility", 0),
                "win_rate": performance_metrics.get("win_rate", 0),
                "profit_factor": performance_metrics.get("profit_factor", 0),
                "total_trades": performance_metrics.get("total_trades", 0),
            },
            
            # 配置信息
            "backtest_config": {
                "strategy_name": strategy_name,  # 添加策略名称，方便前端获取
                "start_date": start_date.isoformat(),  # 添加开始日期
                "end_date": end_date.isoformat(),  # 添加结束日期
                "initial_cash": config.initial_cash,  # 添加初始资金
                "commission_rate": config.commission_rate,
                "slippage_rate": config.slippage_rate,
                "max_position_size": config.max_position_size
            },
            
            # 交易记录
            "trade_history": [
                {
                    "trade_id": trade.trade_id,
                    "stock_code": trade.stock_code,
                    "action": trade.action,
                    "quantity": trade.quantity,
                    "price": trade.price,
                    "timestamp": trade.timestamp.isoformat(),
                    "commission": trade.commission,
                    "slippage_cost": getattr(trade, 'slippage_cost', 0.0),
                    "pnl": trade.pnl
                }
                for trade in portfolio_manager.trades
            ],
            
            # 组合历史（包含完整的positions信息）
            "portfolio_history": [
                {
                    "date": snapshot["date"].isoformat(),
                    "portfolio_value": snapshot["portfolio_value"],
                    "portfolio_value_without_cost": snapshot.get("portfolio_value_without_cost", snapshot["portfolio_value"]),
                    "cash": snapshot["cash"],
                    "positions_count": len(snapshot.get("positions", {})),
                    "positions": snapshot.get("positions", {}),  # 包含完整的持仓信息
                    "total_return": (snapshot["portfolio_value"] - config.initial_cash) / config.initial_cash if config.initial_cash > 0 else 0,
                    "total_return_without_cost": (snapshot.get("portfolio_value_without_cost", snapshot["portfolio_value"]) - config.initial_cash) / config.initial_cash if config.initial_cash > 0 else 0
                }
                for snapshot in portfolio_manager.portfolio_history
            ],
            
            # 交易成本统计
            "cost_statistics": {
                "total_commission": portfolio_manager.total_commission,
                "total_slippage": portfolio_manager.total_slippage,
                "total_cost": portfolio_manager.total_commission + portfolio_manager.total_slippage,
                "cost_ratio": (portfolio_manager.total_commission + portfolio_manager.total_slippage) / config.initial_cash if config.initial_cash > 0 else 0
            }
        }
        
        # 添加无成本指标到报告
        metrics_without_cost = portfolio_manager.get_performance_metrics_without_cost()
        report["excess_return_without_cost"] = {
            "mean": metrics_without_cost.get("mean", 0),
            "std": metrics_without_cost.get("std", 0),
            "annualized_return": metrics_without_cost.get("annualized_return", 0),
            "information_ratio": metrics_without_cost.get("information_ratio", 0),
            "max_drawdown": metrics_without_cost.get("max_drawdown", 0)
        }
        
        report["excess_return_with_cost"] = {
            "mean": performance_metrics.get("volatility", 0) / np.sqrt(252) if performance_metrics.get("volatility", 0) > 0 else 0,
            "std": performance_metrics.get("volatility", 0),
            "annualized_return": performance_metrics.get("annualized_return", 0),
            "information_ratio": performance_metrics.get("sharpe_ratio", 0),  # 使用夏普比率作为近似
            "max_drawdown": performance_metrics.get("max_drawdown", 0)
        }
        
        # 计算额外的分析指标
        report.update(self._calculate_additional_metrics(portfolio_manager))
        
        return report
    
    def _calculate_additional_metrics(self, portfolio_manager: PortfolioManager) -> Dict[str, Any]:
        """计算额外的分析指标"""
        additional_metrics = {}
        
        try:
            if not portfolio_manager.portfolio_history:
                return additional_metrics
            
            # 计算月度收益
            portfolio_values = pd.Series([
                snapshot['portfolio_value'] for snapshot in portfolio_manager.portfolio_history
            ], index=[
                snapshot['date'] for snapshot in portfolio_manager.portfolio_history
            ])
            
            monthly_returns = portfolio_values.resample('ME').last().pct_change().dropna()
            
            if len(monthly_returns) > 0:
                additional_metrics.update({
                    "monthly_return_mean": float(monthly_returns.mean()),
                    "monthly_return_std": float(monthly_returns.std()),
                    "best_month": float(monthly_returns.max()),
                    "worst_month": float(monthly_returns.min()),
                    "positive_months": int((monthly_returns > 0).sum()),
                    "negative_months": int((monthly_returns < 0).sum())
                })
            
            # 计算持仓分析
            if portfolio_manager.trades:
                holding_periods = []
                stock_performance = {}
                
                # 分析每只股票的表现
                for stock_code in set(trade.stock_code for trade in portfolio_manager.trades):
                    stock_trades = [t for t in portfolio_manager.trades if t.stock_code == stock_code]
                    stock_pnl = sum(t.pnl for t in stock_trades if t.action == "SELL")
                    stock_performance[stock_code] = stock_pnl
                
                additional_metrics.update({
                    "best_performing_stock": max(stock_performance.items(), key=lambda x: x[1]) if stock_performance else None,
                    "worst_performing_stock": min(stock_performance.items(), key=lambda x: x[1]) if stock_performance else None,
                    "stocks_traded": len(stock_performance)
                })
            
        except Exception as e:
            logger.error(f"计算额外指标失败: {e}")
        
        return additional_metrics
    
    def validate_backtest_parameters(self, strategy_name: str, stock_codes: List[str],
                                   start_date: datetime, end_date: datetime,
                                   strategy_config: Dict[str, Any]) -> bool:
        """验证回测参数"""
        try:
            # 验证策略名称
            available_strategies = StrategyFactory.get_available_strategies()
            if strategy_name.lower() not in available_strategies:
                raise TaskError(
                    message=f"不支持的策略: {strategy_name}，可用策略: {available_strategies}",
                    severity=ErrorSeverity.MEDIUM
                )
            
            # 验证股票代码
            if not stock_codes or len(stock_codes) == 0:
                raise TaskError(
                    message="股票代码列表不能为空",
                    severity=ErrorSeverity.MEDIUM
                )
            
            if len(stock_codes) > 50:
                raise TaskError(
                    message=f"股票数量过多: {len(stock_codes)}，最多支持50只股票",
                    severity=ErrorSeverity.MEDIUM
                )
            
            # 验证日期范围
            if start_date >= end_date:
                raise TaskError(
                    message="开始日期必须早于结束日期",
                    severity=ErrorSeverity.MEDIUM
                )
            
            date_range = (end_date - start_date).days
            if date_range < 30:
                raise TaskError(
                    message=f"回测期间太短: {date_range}天，至少需要30天",
                    severity=ErrorSeverity.MEDIUM
                )
            
            if date_range > 3650:  # 10年
                raise TaskError(
                    message=f"回测期间太长: {date_range}天，最多支持10年",
                    severity=ErrorSeverity.MEDIUM
                )
            
            # 验证策略配置
            if not isinstance(strategy_config, dict):
                raise TaskError(
                    message="策略配置必须是字典格式",
                    severity=ErrorSeverity.MEDIUM
                )
            
            return True
            
        except TaskError:
            raise
        except Exception as e:
            raise TaskError(
                message=f"参数验证失败: {str(e)}",
                severity=ErrorSeverity.MEDIUM,
                original_exception=e
            )
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """获取执行统计信息"""
        return {
            **self.execution_stats,
            "success_rate": (
                self.execution_stats["successful_backtests"] / 
                max(self.execution_stats["total_backtests"], 1)
            ),
            "available_strategies": StrategyFactory.get_available_strategies()
        }
