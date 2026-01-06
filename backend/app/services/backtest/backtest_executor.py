"""
回测执行器 - 完整的回测流程执行和结果分析
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger

from .backtest_engine import (
    BaseStrategy, StrategyFactory, PortfolioManager, BacktestConfig,
    TradingSignal, Trade, Position
)
from .backtest_progress_monitor import backtest_progress_monitor
from app.core.error_handler import TaskError, ErrorSeverity, ErrorContext
from app.models.task_models import BacktestResult


class DataLoader:
    """数据加载器"""
    
    def __init__(self, data_dir: str = "backend/data"):
        self.data_dir = Path(data_dir)
    
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
                           end_date: datetime) -> Dict[str, pd.DataFrame]:
        """加载多只股票数据"""
        stock_data = {}
        failed_stocks = []
        
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
    
    def __init__(self, data_dir: str = "backend/data"):
        self.data_loader = DataLoader(data_dir)
        self.execution_stats = {
            "total_backtests": 0,
            "successful_backtests": 0,
            "failed_backtests": 0
        }
    
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
            
            # 更新总交易日数
            if task_id:
                progress_data = backtest_progress_monitor.get_progress_data(task_id)
                if progress_data:
                    progress_data.total_trading_days = len(trading_dates)
            
            # 执行回测
            if task_id:
                await backtest_progress_monitor.update_stage(
                    task_id, "backtest_execution", status="running"
                )
            
            backtest_results = await self._execute_backtest_loop(
                strategy, portfolio_manager, stock_data, trading_dates, task_id
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
                             task_id: str = None) -> Dict[str, Any]:
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
                
                # 生成交易信号
                all_signals = []
                for stock_code, data in stock_data.items():
                    if current_date in data.index:
                        # 获取到当前日期的历史数据
                        historical_data = data[data.index <= current_date]
                        if len(historical_data) >= 20:  # 确保有足够的历史数据
                            signals = strategy.generate_signals(historical_data, current_date)
                            all_signals.extend(signals)
                
                total_signals += len(all_signals)
                
                # 执行交易信号
                trades_this_day = 0
                for signal in all_signals:
                    if strategy.validate_signal(signal, 
                                              portfolio_manager.get_portfolio_value(current_prices),
                                              portfolio_manager.positions):
                        trade = portfolio_manager.execute_signal(signal, current_prices)
                        if trade:
                            executed_trades += 1
                            trades_this_day += 1
                
                # 记录组合快照
                portfolio_manager.record_portfolio_snapshot(current_date, current_prices)
                
                # 更新进度监控
                if task_id and i % 5 == 0:  # 每5天更新一次进度
                    portfolio_value = portfolio_manager.get_portfolio_value(current_prices)
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
            
            # 配置信息
            "backtest_config": {
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
                    "pnl": trade.pnl
                }
                for trade in portfolio_manager.trades
            ],
            
            # 组合历史
            "portfolio_history": [
                {
                    "date": snapshot["date"].isoformat(),
                    "portfolio_value": snapshot["portfolio_value"],
                    "cash": snapshot["cash"],
                    "positions_count": len(snapshot["positions"])
                }
                for snapshot in portfolio_manager.portfolio_history
            ]
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
