"""
回测执行器 - 完整的回测流程执行和结果分析
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from app.core.error_handler import ErrorContext, ErrorSeverity, TaskError
from app.models.task_models import BacktestResult

from ..core.base_strategy import BaseStrategy
from ..core.portfolio_manager import PortfolioManager
from ..models import BacktestConfig, Position, SignalType, Trade, TradingSignal
from ..strategies.strategy_factory import AdvancedStrategyFactory, StrategyFactory
from .backtest_progress_monitor import backtest_progress_monitor
from .data_loader import DataLoader

# 性能监控（可选导入，避免依赖问题）
try:
    from ..utils.performance_profiler import (
        BacktestPerformanceProfiler,
        PerformanceContext,
    )

    PERFORMANCE_PROFILING_AVAILABLE = True
except ImportError:
    PERFORMANCE_PROFILING_AVAILABLE = False
    BacktestPerformanceProfiler = None
    PerformanceContext = None


class BacktestExecutor:
    """回测执行器"""

    def __init__(
        self,
        data_dir: str = "backend/data",
        enable_parallel: bool = True,
        max_workers: Optional[int] = None,
        enable_performance_profiling: bool = False,
    ):
        """
        初始化回测执行器

        Args:
            data_dir: 数据目录
            enable_parallel: 是否启用并行化（默认True）
            max_workers: 最大工作线程数，默认使用CPU核心数
            enable_performance_profiling: 是否启用性能分析（默认False）
        """
        import os

        if max_workers is None:
            max_workers = min(os.cpu_count() or 4, 8)  # 最多8个线程，避免过多线程导致开销

        self.enable_parallel = enable_parallel
        self.max_workers = max_workers
        self.data_loader = DataLoader(
            data_dir, max_workers=max_workers if enable_parallel else None
        )
        self.execution_stats = {
            "total_backtests": 0,
            "successful_backtests": 0,
            "failed_backtests": 0,
        }

        # 性能分析器（可选）
        self.enable_performance_profiling = (
            enable_performance_profiling and PERFORMANCE_PROFILING_AVAILABLE
        )
        self.performance_profiler: Optional[BacktestPerformanceProfiler] = None

        if enable_parallel:
            logger.info(f"回测执行器已启用并行化，最大工作线程数: {max_workers}")

        if self.enable_performance_profiling:
            logger.info("回测执行器已启用性能分析")

    async def run_backtest(
        self,
        strategy_name: str,
        stock_codes: List[str],
        start_date: datetime,
        end_date: datetime,
        strategy_config: Dict[str, Any],
        backtest_config: Optional[BacktestConfig] = None,
        task_id: str = None,
    ) -> Dict[str, Any]:
        """运行回测"""
        # 初始化性能分析器
        if self.enable_performance_profiling:
            self.performance_profiler = BacktestPerformanceProfiler(
                enable_memory_tracking=True
            )
            self.performance_profiler.start_backtest()
            self.performance_profiler.take_memory_snapshot("backtest_start")

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
                    task_id=task_id, backtest_id=backtest_id
                )
                await backtest_progress_monitor.update_stage(
                    task_id, "initialization", progress=100, status="completed"
                )

            # 创建策略（性能监控）
            if self.enable_performance_profiling:
                self.performance_profiler.start_stage(
                    "strategy_setup",
                    {"strategy_name": strategy_name, "stock_count": len(stock_codes)},
                )

            if task_id:
                await backtest_progress_monitor.update_stage(
                    task_id, "strategy_setup", status="running"
                )

            # 优先使用高级策略工厂
            try:
                strategy = AdvancedStrategyFactory.create_strategy(
                    strategy_name, strategy_config
                )
            except Exception:
                # 如果高级策略工厂没有该策略，回退到基础策略工厂
                strategy = StrategyFactory.create_strategy(
                    strategy_name, strategy_config
                )

            if self.enable_performance_profiling:
                self.performance_profiler.end_stage("strategy_setup")

            if task_id:
                await backtest_progress_monitor.update_stage(
                    task_id, "strategy_setup", progress=100, status="completed"
                )

            # 创建组合管理器
            portfolio_manager = PortfolioManager(backtest_config)

            # 加载数据（性能监控）
            if self.enable_performance_profiling:
                self.performance_profiler.start_stage(
                    "data_loading",
                    {
                        "stock_codes": stock_codes,
                        "start_date": start_date.isoformat(),
                        "end_date": end_date.isoformat(),
                    },
                )

            if task_id:
                await backtest_progress_monitor.update_stage(
                    task_id, "data_loading", status="running"
                )

            logger.info(
                f"开始回测: {strategy_name}, 股票: {stock_codes}, 期间: {start_date} - {end_date}"
            )
            stock_data = self.data_loader.load_multiple_stocks(
                stock_codes, start_date, end_date
            )

            if self.enable_performance_profiling:
                self.performance_profiler.end_stage(
                    "data_loading",
                    {
                        "loaded_stocks": len(stock_data),
                        "total_records": sum(len(df) for df in stock_data.values()),
                    },
                )
                self.performance_profiler.take_memory_snapshot("after_data_loading")

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
                raise TaskError(message=error_msg, severity=ErrorSeverity.MEDIUM)

            # 更新总交易日数（同时写入数据库）
            if task_id:
                progress_data = backtest_progress_monitor.get_progress_data(task_id)
                if progress_data:
                    progress_data.total_trading_days = len(trading_dates)

                # 将总交易日数写入数据库
                try:
                    from app.core.database import SessionLocal
                    from app.models.task_models import TaskStatus
                    from app.repositories.task_repository import TaskRepository

                    session = SessionLocal()
                    try:
                        task_repo = TaskRepository(session)
                        existing_task = task_repo.get_task_by_id(task_id)
                        if existing_task:
                            result_data = existing_task.result or {}
                            progress_data_db = result_data.get("progress_data", {})
                            progress_data_db["total_days"] = len(trading_dates)
                            result_data["progress_data"] = progress_data_db

                            task_repo.update_task_status(
                                task_id=task_id,
                                status=TaskStatus.RUNNING,
                                result=result_data,
                            )
                    finally:
                        session.close()
                except Exception as e:
                    logger.warning(f"更新总交易日数失败: {e}")

            # 执行回测（性能监控）
            if self.enable_performance_profiling:
                self.performance_profiler.start_stage(
                    "backtest_execution",
                    {
                        "total_trading_days": len(trading_dates),
                        "stock_count": len(stock_data),
                    },
                )

            if task_id:
                await backtest_progress_monitor.update_stage(
                    task_id, "backtest_execution", status="running"
                )

            backtest_results = await self._execute_backtest_loop(
                strategy,
                portfolio_manager,
                stock_data,
                trading_dates,
                task_id,
                backtest_id,
            )

            if self.enable_performance_profiling:
                self.performance_profiler.end_stage(
                    "backtest_execution",
                    {
                        "total_signals": backtest_results.get("total_signals", 0),
                        "executed_trades": backtest_results.get("executed_trades", 0),
                        "trading_days": backtest_results.get("trading_days", 0),
                    },
                )
                self.performance_profiler.update_backtest_stats(
                    signals=backtest_results.get("total_signals", 0),
                    trades=backtest_results.get("executed_trades", 0),
                    days=backtest_results.get("trading_days", 0),
                )
                self.performance_profiler.take_memory_snapshot(
                    "after_backtest_execution"
                )

            if task_id:
                await backtest_progress_monitor.update_stage(
                    task_id, "backtest_execution", progress=100, status="completed"
                )

            # 计算绩效指标（性能监控）
            if self.enable_performance_profiling:
                self.performance_profiler.start_stage("metrics_calculation")

            if task_id:
                await backtest_progress_monitor.update_stage(
                    task_id, "metrics_calculation", status="running"
                )

            performance_metrics = portfolio_manager.get_performance_metrics()

            if self.enable_performance_profiling:
                self.performance_profiler.end_stage("metrics_calculation")

            if task_id:
                await backtest_progress_monitor.update_stage(
                    task_id, "metrics_calculation", progress=100, status="completed"
                )

            # 生成回测报告（性能监控）
            if self.enable_performance_profiling:
                self.performance_profiler.start_stage("report_generation")

            if task_id:
                await backtest_progress_monitor.update_stage(
                    task_id, "report_generation", status="running"
                )

            # 记录策略配置信息
            if (
                strategy_config
                and isinstance(strategy_config, dict)
                and len(strategy_config) > 0
            ):
                logger.info(f"生成回测报告，策略配置: {strategy_config}")
            else:
                logger.warning(
                    f"策略配置为空或无效: {strategy_config}, 类型: {type(strategy_config)}"
                )

            backtest_report = self._generate_backtest_report(
                strategy_name,
                stock_codes,
                start_date,
                end_date,
                backtest_config,
                portfolio_manager,
                performance_metrics,
                strategy_config=strategy_config,
            )

            if self.enable_performance_profiling:
                self.performance_profiler.end_stage(
                    "report_generation", {"report_size": len(str(backtest_report))}
                )

            if task_id:
                await backtest_progress_monitor.update_stage(
                    task_id, "report_generation", progress=100, status="completed"
                )
                await backtest_progress_monitor.update_stage(
                    task_id, "data_storage", progress=100, status="completed"
                )

            self.execution_stats["successful_backtests"] += 1
            logger.info(
                f"回测完成: {strategy_name}, 总收益: {performance_metrics.get('total_return', 0):.2%}"
            )

            # 完成监控
            if task_id:
                await backtest_progress_monitor.complete_backtest(
                    task_id,
                    {"total_return": performance_metrics.get("total_return", 0)},
                )

            # 生成性能报告
            if self.enable_performance_profiling:
                self.performance_profiler.end_backtest()
                self.performance_profiler.take_memory_snapshot("backtest_end")

                # 将性能报告添加到回测报告中
                performance_report = self.performance_profiler.generate_report()
                backtest_report["performance_analysis"] = performance_report

                # 打印性能摘要
                self.performance_profiler.print_summary()

                # 保存性能报告到文件（如果提供了task_id）
                if task_id:
                    try:
                        import os

                        performance_dir = Path("backend/data/performance_reports")
                        performance_dir.mkdir(parents=True, exist_ok=True)
                        performance_file = (
                            performance_dir / f"backtest_{task_id}_performance.json"
                        )
                        self.performance_profiler.save_report(str(performance_file))
                        logger.info(f"性能报告已保存到: {performance_file}")
                    except Exception as e:
                        logger.warning(f"保存性能报告失败: {e}")

            return backtest_report

        except Exception as e:
            self.execution_stats["failed_backtests"] += 1
            error_msg = f"回测执行失败: {str(e)}"

            # 即使出错也结束性能分析
            if self.enable_performance_profiling and self.performance_profiler:
                try:
                    self.performance_profiler.end_backtest()
                    logger.warning("回测失败，但性能分析已完成")
                except Exception as perf_error:
                    logger.warning(f"结束性能分析时出错: {perf_error}")

            if task_id:
                await backtest_progress_monitor.set_error(task_id, error_msg)

            raise TaskError(
                message=error_msg, severity=ErrorSeverity.HIGH, original_exception=e
            )

    def _get_trading_calendar(
        self,
        stock_data: Dict[str, pd.DataFrame],
        start_date: datetime,
        end_date: datetime,
    ) -> List[datetime]:
        """获取交易日历"""
        # 合并所有股票的交易日期
        all_dates = set()
        for data in stock_data.values():
            all_dates.update(data.index.tolist())

        # 过滤日期范围并排序
        trading_dates = sorted(
            [date for date in all_dates if start_date <= date <= end_date]
        )

        return trading_dates

    async def _execute_backtest_loop(
        self,
        strategy: BaseStrategy,
        portfolio_manager: PortfolioManager,
        stock_data: Dict[str, pd.DataFrame],
        trading_dates: List[datetime],
        task_id: str = None,
        backtest_id: str = None,
    ) -> Dict[str, Any]:
        """执行回测主循环"""
        total_signals = 0
        executed_trades = 0

        # 性能统计：信号生成时间
        signal_generation_times = []
        trade_execution_times = []

        # 辅助函数：检查任务状态
        def _is_task_running(status) -> bool:
            if status is None:
                return False
            # 支持字符串或Enum
            try:
                return (
                    status == TaskStatus.RUNNING or status == TaskStatus.RUNNING.value
                )
            except Exception:
                return status == TaskStatus.RUNNING.value

        def check_task_status():
            """检查任务是否仍然存在且处于运行状态"""
            if not task_id:
                return True
            try:
                from app.core.database import SessionLocal
                from app.models.task_models import TaskStatus
                from app.repositories.task_repository import TaskRepository

                session = SessionLocal()
                try:
                    task_repo = TaskRepository(session)
                    task = task_repo.get_task_by_id(task_id)
                    if not task:
                        logger.warning(f"任务不存在，停止回测执行: {task_id}")
                        return False
                    if not _is_task_running(task.status):
                        logger.warning(f"任务状态为 {task.status}，停止回测执行: {task_id}")
                        return False
                    return True
                finally:
                    session.close()
            except Exception as e:
                logger.warning(f"检查任务状态失败: {e}，继续执行")
                return True  # 检查失败时继续执行，避免因检查错误而中断

        for i, current_date in enumerate(trading_dates):
            # 在循环开始时检查任务状态（每10个交易日检查一次，避免频繁检查）
            if task_id and i % 10 == 0 and i > 0:
                if not check_task_status():
                    logger.info(f"任务状态检查失败，停止回测执行: {task_id}")
                    raise TaskError(
                        message=f"任务 {task_id} 已被删除或状态已改变，停止回测执行",
                        severity=ErrorSeverity.LOW,
                    )
            try:
                # 获取当前价格
                current_prices = {}
                for stock_code, data in stock_data.items():
                    if current_date in data.index:
                        current_prices[stock_code] = data.loc[current_date, "close"]

                if not current_prices:
                    continue

                # 生成交易信号（支持并行生成多股票信号）
                all_signals = []

                # 性能监控：记录信号生成时间
                signal_start_time = (
                    time.perf_counter() if self.enable_performance_profiling else None
                )

                if self.enable_parallel and len(stock_data) > 3:
                    # 并行生成多股票信号
                    def generate_stock_signals(
                        stock_code: str, data: pd.DataFrame
                    ) -> List[TradingSignal]:
                        """为单只股票生成信号（用于并行执行）"""
                        if current_date in data.index:
                            historical_data = data[data.index <= current_date]
                            if len(historical_data) >= 20:  # 确保有足够的历史数据
                                try:
                                    return strategy.generate_signals(
                                        historical_data, current_date
                                    )
                                except Exception as e:
                                    logger.warning(f"生成信号失败 {stock_code}: {e}")
                                    return []
                        return []

                    # 使用线程池并行生成信号
                    sequential_start = (
                        time.perf_counter()
                        if self.enable_performance_profiling
                        else None
                    )

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

                    # 记录并行化效率（估算顺序执行时间）
                    if self.enable_performance_profiling and sequential_start:
                        parallel_time = time.perf_counter() - sequential_start
                        # 估算顺序执行时间（假设每只股票耗时相同）
                        estimated_sequential_time = (
                            parallel_time * len(stock_data) / self.max_workers
                        )
                        if i == 0:  # 只在第一次记录
                            self.performance_profiler.record_parallel_efficiency(
                                operation_name="signal_generation",
                                sequential_time=estimated_sequential_time,
                                parallel_time=parallel_time,
                                worker_count=self.max_workers,
                            )
                else:
                    # 顺序生成信号（股票数量少或禁用并行）
                    for stock_code, data in stock_data.items():
                        if current_date in data.index:
                            # 获取到当前日期的历史数据
                            historical_data = data[data.index <= current_date]
                            if len(historical_data) >= 20:  # 确保有足够的历史数据
                                try:
                                    signals = strategy.generate_signals(
                                        historical_data, current_date
                                    )
                                    all_signals.extend(signals)
                                except Exception as e:
                                    logger.warning(f"生成信号失败 {stock_code}: {e}")
                                    continue

                # 记录信号生成时间
                if self.enable_performance_profiling and signal_start_time:
                    signal_duration = time.perf_counter() - signal_start_time
                    signal_generation_times.append(signal_duration)
                    self.performance_profiler.record_function_call(
                        "generate_signals", signal_duration
                    )

                total_signals += len(all_signals)

                # 保存信号记录到数据库
                if task_id and all_signals:
                    try:
                        import uuid

                        from app.core.database import get_async_session_context
                        from app.repositories.backtest_detailed_repository import (
                            BacktestDetailedRepository,
                        )

                        # 使用传入的backtest_id或生成一个
                        current_backtest_id = backtest_id or (
                            f"bt_{task_id[:8]}"
                            if task_id
                            else f"bt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        )

                        # 批量保存信号记录
                        signals_data = []
                        for signal in all_signals:
                            signal_data = {
                                "signal_id": f"sig_{uuid.uuid4().hex[:12]}",
                                "stock_code": signal.stock_code,
                                "stock_name": None,  # 可以从股票数据中获取
                                "signal_type": signal.signal_type.name,
                                "timestamp": signal.timestamp,
                                "price": signal.price,
                                "strength": signal.strength,
                                "reason": signal.reason,
                                "metadata": signal.metadata,
                                "executed": False,
                            }
                            signals_data.append(signal_data)

                        # 异步保存信号记录
                        async with get_async_session_context() as session:
                            try:
                                repository = BacktestDetailedRepository(session)
                                await repository.batch_save_signal_records(
                                    task_id=task_id,
                                    backtest_id=current_backtest_id,
                                    signals_data=signals_data,
                                )
                                await session.commit()
                            except Exception as e:
                                await session.rollback()
                                logger.warning(f"保存信号记录失败: {e}")
                    except Exception as e:
                        logger.warning(f"保存信号记录时出错: {e}")

                # 执行交易信号（性能监控）
                trade_start_time = (
                    time.perf_counter() if self.enable_performance_profiling else None
                )
                trades_this_day = 0
                executed_trade_signals = []  # 记录已执行的交易对应的信号
                unexecuted_signals = []  # 记录未执行的信号及原因

                for signal in all_signals:
                    # 验证信号
                    is_valid, validation_reason = strategy.validate_signal(
                        signal,
                        portfolio_manager.get_portfolio_value(current_prices),
                        portfolio_manager.positions,
                    )

                    if not is_valid:
                        # 验证失败，记录未执行原因
                        unexecuted_signals.append(
                            {
                                "stock_code": signal.stock_code,
                                "timestamp": signal.timestamp,
                                "signal_type": signal.signal_type.name,
                                "execution_reason": validation_reason or "信号验证失败",
                            }
                        )
                        continue

                    # 验证通过，尝试执行
                    trade_exec_start = (
                        time.perf_counter()
                        if self.enable_performance_profiling
                        else None
                    )
                    trade, failure_reason = portfolio_manager.execute_signal(
                        signal, current_prices
                    )
                    if self.enable_performance_profiling and trade_exec_start:
                        trade_exec_duration = time.perf_counter() - trade_exec_start
                        trade_execution_times.append(trade_exec_duration)
                        self.performance_profiler.record_function_call(
                            "execute_signal", trade_exec_duration
                        )

                    if trade:
                        executed_trades += 1
                        trades_this_day += 1
                        # 记录已执行的信号，用于后续标记
                        executed_trade_signals.append(
                            {
                                "stock_code": signal.stock_code,
                                "timestamp": signal.timestamp,
                                "signal_type": signal.signal_type.name,
                            }
                        )
                    else:
                        # 执行失败，记录未执行原因（从 execute_signal 直接获取）
                        unexecuted_signals.append(
                            {
                                "stock_code": signal.stock_code,
                                "timestamp": signal.timestamp,
                                "signal_type": signal.signal_type.name,
                                "execution_reason": failure_reason or "执行失败（未知原因）",
                            }
                        )

                # 记录交易执行总时间
                if self.enable_performance_profiling and trade_start_time:
                    trade_duration = time.perf_counter() - trade_start_time
                    self.performance_profiler.record_function_call(
                        "execute_trades_batch", trade_duration
                    )

                # 更新未执行信号的原因
                if task_id and unexecuted_signals:
                    try:
                        from app.core.database import get_async_session_context
                        from app.repositories.backtest_detailed_repository import (
                            BacktestDetailedRepository,
                        )

                        async with get_async_session_context() as session:
                            try:
                                repository = BacktestDetailedRepository(session)
                                for unexecuted_signal in unexecuted_signals:
                                    await repository.update_signal_execution_reason(
                                        task_id=task_id,
                                        stock_code=unexecuted_signal["stock_code"],
                                        timestamp=unexecuted_signal["timestamp"],
                                        signal_type=unexecuted_signal["signal_type"],
                                        execution_reason=unexecuted_signal[
                                            "execution_reason"
                                        ],
                                    )
                                await session.commit()
                            except Exception as e:
                                await session.rollback()
                                logger.warning(f"更新信号未执行原因失败: {e}")
                    except Exception as e:
                        logger.warning(f"更新信号未执行原因时出错: {e}")

                # 标记已执行的信号
                if task_id and executed_trade_signals:
                    try:
                        from app.core.database import get_async_session_context
                        from app.repositories.backtest_detailed_repository import (
                            BacktestDetailedRepository,
                        )

                        async with get_async_session_context() as session:
                            try:
                                repository = BacktestDetailedRepository(session)
                                for executed_signal in executed_trade_signals:
                                    await repository.mark_signal_as_executed(
                                        task_id=task_id,
                                        stock_code=executed_signal["stock_code"],
                                        timestamp=executed_signal["timestamp"],
                                        signal_type=executed_signal["signal_type"],
                                    )
                                await session.commit()
                            except Exception as e:
                                await session.rollback()
                                logger.warning(f"标记信号为已执行失败: {e}")
                    except Exception as e:
                        logger.warning(f"标记信号为已执行时出错: {e}")

                # 记录组合快照
                portfolio_manager.record_portfolio_snapshot(
                    current_date, current_prices
                )

                # 更新进度监控（同时更新数据库）
                if task_id and i % 5 == 0:  # 每5天更新一次进度
                    portfolio_value = portfolio_manager.get_portfolio_value(
                        current_prices
                    )
                    logger.debug(
                        f"准备更新进度: task_id={task_id}, i={i}, total_days={len(trading_dates)}, signals={len(all_signals)}, trades={trades_this_day}, total_signals={total_signals}, total_trades={executed_trades}"
                    )

                    # 计算进度百分比（回测执行阶段占30-90%，即60%的进度范围）
                    execution_progress = (i + 1) / len(trading_dates) * 100
                    overall_progress = 30 + (execution_progress / 100) * 60  # 30%到90%

                    # 更新数据库中的任务进度（包含详细数据）
                    try:
                        from datetime import datetime

                        from app.core.database import SessionLocal
                        from app.models.task_models import TaskStatus
                        from app.repositories.task_repository import TaskRepository

                        session = SessionLocal()
                        try:
                            task_repo = TaskRepository(session)

                            # 读取现有的 result 数据
                            existing_task = task_repo.get_task_by_id(task_id)
                            if not existing_task:
                                logger.warning(f"任务不存在，无法更新进度: {task_id}")
                                # 任务已被删除，停止回测执行
                                raise TaskError(
                                    message=f"任务 {task_id} 已被删除，停止回测执行",
                                    severity=ErrorSeverity.LOW,
                                )
                            # 检查任务状态，如果不是运行中，则停止执行
                            elif not _is_task_running(existing_task.status):
                                logger.warning(
                                    f"任务状态为 {existing_task.status}，停止回测执行: {task_id}"
                                )
                                raise TaskError(
                                    message=f"任务 {task_id} 状态为 {existing_task.status}，停止回测执行",
                                    severity=ErrorSeverity.LOW,
                                )
                            else:
                                result_data = existing_task.result or {}
                                if not isinstance(result_data, dict):
                                    result_data = {}
                                progress_data = result_data.get("progress_data", {})
                                if not isinstance(progress_data, dict):
                                    progress_data = {}

                                # 更新进度数据
                                progress_data.update(
                                    {
                                        "processed_days": i + 1,
                                        "total_days": len(trading_dates),
                                        "current_date": current_date.strftime(
                                            "%Y-%m-%d"
                                        ),
                                        "signals_generated": len(all_signals),
                                        "trades_executed": trades_this_day,
                                        "total_signals": total_signals,
                                        "total_trades": executed_trades,
                                        "portfolio_value": portfolio_value,
                                        "last_updated": datetime.utcnow().isoformat(),
                                    }
                                )

                                result_data["progress_data"] = progress_data

                                # 记录日志以便调试
                                logger.info(
                                    f"更新回测进度数据: task_id={task_id}, processed_days={i+1}, total_days={len(trading_dates)}, signals={total_signals}, trades={executed_trades}, portfolio={portfolio_value:.2f}, progress_data_keys={list(progress_data.keys())}"
                                )

                                task_repo.update_task_status(
                                    task_id=task_id,
                                    status=TaskStatus.RUNNING,
                                    progress=overall_progress,
                                    result=result_data,  # 包含详细进度数据
                                )

                                # 确保 result 字段被标记为已修改并提交
                                session.commit()
                                logger.info(
                                    f"进度数据已提交到数据库: task_id={task_id}, result_data_keys={list(result_data.keys())}, progress_data={progress_data}"
                                )
                        except Exception as inner_error:
                            session.rollback()
                            logger.error(
                                f"更新任务进度到数据库失败（内部错误）: {inner_error}", exc_info=True
                            )
                            raise
                        finally:
                            session.close()
                    except Exception as db_error:
                        logger.error(f"更新任务进度到数据库失败: {db_error}", exc_info=True)

                    # 更新进程内的进度监控（虽然主进程看不到，但保持一致性）
                    await backtest_progress_monitor.update_execution_progress(
                        task_id=task_id,
                        processed_days=i + 1,
                        current_date=current_date.strftime("%Y-%m-%d"),
                        signals_generated=len(all_signals),
                        trades_executed=trades_this_day,
                        portfolio_value=portfolio_value,
                    )

                # 定期输出进度日志
                if i % 50 == 0:
                    progress = (i + 1) / len(trading_dates) * 100
                    portfolio_value = portfolio_manager.get_portfolio_value(
                        current_prices
                    )
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
                current_date=trading_dates[-1].strftime("%Y-%m-%d")
                if trading_dates
                else None,
                signals_generated=0,
                trades_executed=0,
                portfolio_value=final_portfolio_value,
            )

        # 记录性能统计到性能分析器
        if self.enable_performance_profiling and self.performance_profiler:
            if signal_generation_times:
                avg_signal_time = sum(signal_generation_times) / len(
                    signal_generation_times
                )
                self.performance_profiler.end_stage(
                    "backtest_execution",
                    {
                        "avg_signal_generation_time": avg_signal_time,
                        "total_signal_generation_calls": len(signal_generation_times),
                    },
                )
            if trade_execution_times:
                avg_trade_time = sum(trade_execution_times) / len(trade_execution_times)
                self.performance_profiler.end_stage(
                    "backtest_execution",
                    {
                        "avg_trade_execution_time": avg_trade_time,
                        "total_trade_execution_calls": len(trade_execution_times),
                    },
                )

        return {
            "total_signals": total_signals,
            "executed_trades": executed_trades,
            "trading_days": len(trading_dates),
        }

    def _generate_backtest_report(
        self,
        strategy_name: str,
        stock_codes: List[str],
        start_date: datetime,
        end_date: datetime,
        config: BacktestConfig,
        portfolio_manager: PortfolioManager,
        performance_metrics: Dict[str, float],
        strategy_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
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
                "max_position_size": config.max_position_size,
                **(
                    {"strategy_config": strategy_config}
                    if strategy_config
                    and isinstance(strategy_config, dict)
                    and len(strategy_config) > 0
                    else {}
                ),
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
                    "slippage_cost": getattr(trade, "slippage_cost", 0.0),
                    "pnl": trade.pnl,
                }
                for trade in portfolio_manager.trades
            ],
            # 组合历史（包含完整的positions信息）
            "portfolio_history": [
                {
                    "date": snapshot["date"].isoformat(),
                    "portfolio_value": snapshot["portfolio_value"],
                    "portfolio_value_without_cost": snapshot.get(
                        "portfolio_value_without_cost", snapshot["portfolio_value"]
                    ),
                    "cash": snapshot["cash"],
                    "positions_count": len(snapshot.get("positions", {})),
                    "positions": snapshot.get("positions", {}),  # 包含完整的持仓信息
                    "total_return": (snapshot["portfolio_value"] - config.initial_cash)
                    / config.initial_cash
                    if config.initial_cash > 0
                    else 0,
                    "total_return_without_cost": (
                        snapshot.get(
                            "portfolio_value_without_cost", snapshot["portfolio_value"]
                        )
                        - config.initial_cash
                    )
                    / config.initial_cash
                    if config.initial_cash > 0
                    else 0,
                }
                for snapshot in portfolio_manager.portfolio_history
            ],
            # 交易成本统计
            "cost_statistics": {
                "total_commission": portfolio_manager.total_commission,
                "total_slippage": portfolio_manager.total_slippage,
                "total_cost": portfolio_manager.total_commission
                + portfolio_manager.total_slippage,
                "cost_ratio": (
                    portfolio_manager.total_commission
                    + portfolio_manager.total_slippage
                )
                / config.initial_cash
                if config.initial_cash > 0
                else 0,
            },
        }

        # 添加无成本指标到报告
        metrics_without_cost = portfolio_manager.get_performance_metrics_without_cost()
        report["excess_return_without_cost"] = {
            "mean": metrics_without_cost.get("mean", 0),
            "std": metrics_without_cost.get("std", 0),
            "annualized_return": metrics_without_cost.get("annualized_return", 0),
            "information_ratio": metrics_without_cost.get("information_ratio", 0),
            "max_drawdown": metrics_without_cost.get("max_drawdown", 0),
        }

        report["excess_return_with_cost"] = {
            "mean": performance_metrics.get("volatility", 0) / np.sqrt(252)
            if performance_metrics.get("volatility", 0) > 0
            else 0,
            "std": performance_metrics.get("volatility", 0),
            "annualized_return": performance_metrics.get("annualized_return", 0),
            "information_ratio": performance_metrics.get(
                "sharpe_ratio", 0
            ),  # 使用夏普比率作为近似
            "max_drawdown": performance_metrics.get("max_drawdown", 0),
        }

        # 计算额外的分析指标
        report.update(self._calculate_additional_metrics(portfolio_manager))

        return report

    def _calculate_additional_metrics(
        self, portfolio_manager: PortfolioManager
    ) -> Dict[str, Any]:
        """计算额外的分析指标（时间分段表现、个股表现等）"""
        additional_metrics: Dict[str, Any] = {}

        try:
            if not portfolio_manager.portfolio_history:
                return additional_metrics

            # --- 时间分段表现：按月 / 按年收益 ---
            portfolio_values = pd.Series(
                [
                    snapshot["portfolio_value"]
                    for snapshot in portfolio_manager.portfolio_history
                ],
                index=[
                    snapshot["date"] for snapshot in portfolio_manager.portfolio_history
                ],
            ).sort_index()

            # 月度收益（月末权益）
            monthly_values = portfolio_values.resample("M").last()
            monthly_returns = monthly_values.pct_change().dropna()

            if len(monthly_returns) > 0:
                additional_metrics.update(
                    {
                        "monthly_return_mean": float(monthly_returns.mean()),
                        "monthly_return_std": float(monthly_returns.std()),
                        "best_month": float(monthly_returns.max()),
                        "worst_month": float(monthly_returns.min()),
                        "positive_months": int((monthly_returns > 0).sum()),
                        "negative_months": int((monthly_returns < 0).sum()),
                        "monthly_returns_detail": [
                            {
                                "month": period.strftime("%Y-%m"),
                                "return": float(ret),
                            }
                            for period, ret in monthly_returns.items()
                        ],
                    }
                )

            # 年度收益（年末权益）
            yearly_values = portfolio_values.resample("Y").last()
            yearly_returns = yearly_values.pct_change().dropna()

            if len(yearly_returns) > 0:
                additional_metrics["yearly_returns_detail"] = [
                    {
                        "year": period.year,
                        "return": float(ret),
                    }
                    for period, ret in yearly_returns.items()
                ]

            # --- 交易行为与个股表现 ---
            if portfolio_manager.trades:
                stock_performance: Dict[str, Dict[str, Any]] = {}

                for trade in portfolio_manager.trades:
                    stock_stats = stock_performance.setdefault(
                        trade.stock_code,
                        {
                            "stock_code": trade.stock_code,
                            "total_pnl": 0.0,
                            "trade_count": 0,
                        },
                    )
                    stock_stats["trade_count"] += 1
                    # 只有卖出交易才有实现盈亏
                    if trade.action == "SELL":
                        stock_stats["total_pnl"] += float(trade.pnl)

                # 计算每只股票的平均单笔盈亏
                for stats in stock_performance.values():
                    trades = max(stats["trade_count"], 1)
                    stats["avg_pnl_per_trade"] = float(stats["total_pnl"]) / trades

                # 个股表现汇总
                stock_perf_list = list(stock_performance.values())
                additional_metrics.update(
                    {
                        "stock_performance_detail": stock_perf_list,
                        "best_performing_stock": max(
                            stock_perf_list, key=lambda x: x["total_pnl"]
                        )
                        if stock_perf_list
                        else None,
                        "worst_performing_stock": min(
                            stock_perf_list, key=lambda x: x["total_pnl"]
                        )
                        if stock_perf_list
                        else None,
                        "stocks_traded": len(stock_perf_list),
                    }
                )

                # 单笔交易分布的整体特征（便于前端画直方图/统计）
                pnls = [float(t.pnl) for t in portfolio_manager.trades]
                if pnls:
                    pnl_series = pd.Series(pnls)
                    additional_metrics.update(
                        {
                            "trade_pnl_mean": float(pnl_series.mean()),
                            "trade_pnl_median": float(pnl_series.median()),
                            "trade_pnl_std": float(pnl_series.std()),
                        }
                    )

        except Exception as exc:
            logger.error(f"计算额外指标失败: {exc}")

        return additional_metrics

    def validate_backtest_parameters(
        self,
        strategy_name: str,
        stock_codes: List[str],
        start_date: datetime,
        end_date: datetime,
        strategy_config: Dict[str, Any],
    ) -> bool:
        """验证回测参数"""
        try:
            # 验证策略名称
            available_strategies = StrategyFactory.get_available_strategies()
            if strategy_name.lower() not in available_strategies:
                raise TaskError(
                    message=f"不支持的策略: {strategy_name}，可用策略: {available_strategies}",
                    severity=ErrorSeverity.MEDIUM,
                )

            # 验证股票代码
            if not stock_codes or len(stock_codes) == 0:
                raise TaskError(message="股票代码列表不能为空", severity=ErrorSeverity.MEDIUM)

            if len(stock_codes) > 50:
                raise TaskError(
                    message=f"股票数量过多: {len(stock_codes)}，最多支持50只股票",
                    severity=ErrorSeverity.MEDIUM,
                )

            # 验证日期范围
            if start_date >= end_date:
                raise TaskError(message="开始日期必须早于结束日期", severity=ErrorSeverity.MEDIUM)

            date_range = (end_date - start_date).days
            if date_range < 30:
                raise TaskError(
                    message=f"回测期间太短: {date_range}天，至少需要30天",
                    severity=ErrorSeverity.MEDIUM,
                )

            if date_range > 3650:  # 10年
                raise TaskError(
                    message=f"回测期间太长: {date_range}天，最多支持10年",
                    severity=ErrorSeverity.MEDIUM,
                )

            # 验证策略配置
            if not isinstance(strategy_config, dict):
                raise TaskError(message="策略配置必须是字典格式", severity=ErrorSeverity.MEDIUM)

            return True

        except TaskError:
            raise
        except Exception as e:
            raise TaskError(
                message=f"参数验证失败: {str(e)}",
                severity=ErrorSeverity.MEDIUM,
                original_exception=e,
            )

    def get_execution_statistics(self) -> Dict[str, Any]:
        """获取执行统计信息"""
        return {
            **self.execution_stats,
            "success_rate": (
                self.execution_stats["successful_backtests"]
                / max(self.execution_stats["total_backtests"], 1)
            ),
            "available_strategies": StrategyFactory.get_available_strategies(),
        }

    def _get_execution_failure_reason(
        self,
        signal: TradingSignal,
        portfolio_manager: PortfolioManager,
        current_prices: Dict[str, float],
    ) -> str:
        """
        获取执行失败的原因

        Args:
            signal: 交易信号
            portfolio_manager: 组合管理器
            current_prices: 当前价格

        Returns:
            失败原因字符串
        """
        try:
            stock_code = signal.stock_code
            current_price = current_prices.get(stock_code, signal.price)

            if signal.signal_type == SignalType.BUY:
                # 买入失败的可能原因（逻辑与 _execute_buy 保持一致）
                # 计算组合价值（使用与 _execute_buy 相同的逻辑）
                portfolio_value = portfolio_manager.get_portfolio_value(
                    {stock_code: current_price}
                )
                max_position_value = (
                    portfolio_value * portfolio_manager.config.max_position_size
                )

                current_position = portfolio_manager.positions.get(stock_code)
                current_position_value = (
                    current_position.market_value if current_position else 0
                )

                available_cash_for_stock = max_position_value - current_position_value
                available_cash_for_stock = min(
                    available_cash_for_stock, portfolio_manager.cash * 0.95
                )  # 保留5%现金

                if available_cash_for_stock <= 0:
                    if (
                        current_position_value > 0
                        and current_position_value >= max_position_value
                    ):
                        return f"已达到最大持仓限制: 当前持仓 {current_position_value:.2f} >= 最大持仓 {max_position_value:.2f}"
                    else:
                        return f"可用资金不足: 需要保留5%现金，可用资金 {portfolio_manager.cash:.2f}"

                # 计算购买数量（最小交易单位为100股）
                quantity = int(available_cash_for_stock / current_price / 100) * 100
                if quantity <= 0:
                    return f"可买数量不足: 可用资金 {available_cash_for_stock:.2f}，价格 {current_price:.2f}，无法买入100股"

                # 计算实际成本（包含手续费和滑点）
                # 应用滑点（买入时价格上涨）
                execution_price = current_price * (
                    1 + portfolio_manager.config.slippage_rate
                )
                slippage_cost_per_share = (
                    current_price * portfolio_manager.config.slippage_rate
                )

                total_cost = quantity * execution_price
                commission = total_cost * portfolio_manager.config.commission_rate
                slippage_cost = quantity * slippage_cost_per_share
                total_cost_with_all_fees = total_cost + commission

                if total_cost_with_all_fees > portfolio_manager.cash:
                    return f"资金不足: 需要 {total_cost_with_all_fees:.2f}（含手续费 {commission:.2f}），可用 {portfolio_manager.cash:.2f}"

                # 如果所有检查都通过但还是失败了，可能是其他原因
                return f"执行失败: 可能因滑点成本 {slippage_cost:.2f} 或其他限制"

            elif signal.signal_type == SignalType.SELL:
                # 卖出失败的可能原因
                if stock_code not in portfolio_manager.positions:
                    return "无持仓"

                position = portfolio_manager.positions[stock_code]
                if position.quantity <= 0:
                    return "持仓数量为0"

                # 如果所有检查都通过但还是失败了，可能是其他原因
                return "执行失败（未知原因）"

            return "未知信号类型"

        except Exception as e:
            logger.warning(f"获取执行失败原因时出错: {e}")
            return f"执行异常: {str(e)}"
