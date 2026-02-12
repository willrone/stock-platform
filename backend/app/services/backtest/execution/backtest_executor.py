"""
回测执行器 - 完整的回测流程执行和结果分析（重构版）
"""

import asyncio
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from app.core.error_handler import ErrorSeverity, TaskError

from ..core.portfolio_manager import PortfolioManager
from ..core.portfolio_manager_array import PortfolioManagerArray
from ..models import BacktestConfig
from ..strategies.strategy_factory import AdvancedStrategyFactory, StrategyFactory
# from .backtest_progress_monitor import backtest_progress_monitor
from .data_loader import DataLoader

# 导入新模块
from .data_preprocessor import DataPreprocessor
from .backtest_loop_executor import BacktestLoopExecutor
from .report_generator import BacktestReportGenerator
from .performance_tracker import PerformanceTracker
from .validators import validate_backtest_parameters, get_execution_statistics


class BacktestExecutor:
    """回测执行器（重构版）- 协调各模块完成回测"""

    def __init__(
        self,
        data_dir: str = "data",
        enable_parallel: bool = True,
        max_workers: Optional[int] = None,
        enable_performance_profiling: bool = False,
        use_multiprocessing: bool = True,
    ):
        """
        初始化回测执行器

        Args:
            data_dir: 数据目录
            enable_parallel: 是否启用并行化（默认True）
            max_workers: 最大工作线程数，默认使用CPU核心数
            enable_performance_profiling: 是否启用性能分析（默认False）
            use_multiprocessing: 是否使用多进程（突破GIL限制，默认True）
        """
        import os

        if max_workers is None:
            max_workers = min(os.cpu_count() or 4, 8)

        self.enable_parallel = enable_parallel
        self.max_workers = max_workers
        self.use_multiprocessing = use_multiprocessing
        self.use_array_portfolio = True
        
        # 数据加载器
        self.data_loader = DataLoader(
            data_dir, max_workers=max_workers if enable_parallel else None
        )
        
        # 执行统计
        self.execution_stats = {
            "total_backtests": 0,
            "successful_backtests": 0,
            "failed_backtests": 0,
        }

        # 初始化各模块
        self.data_preprocessor = DataPreprocessor(
            enable_parallel=enable_parallel,
            max_workers=max_workers,
            use_multiprocessing=use_multiprocessing
        )
        self.loop_executor = BacktestLoopExecutor()
        self.report_generator = BacktestReportGenerator()
        self.performance_tracker = PerformanceTracker(
            enable_profiling=enable_performance_profiling
        )
        
        # 显式导入进度监控器（避免潜在的循环导入或未定义问题）
        from .backtest_progress_monitor import backtest_progress_monitor
        self.progress_monitor = backtest_progress_monitor

        if enable_parallel:
            mode = "多进程" if use_multiprocessing else "多线程"
            logger.info(f"回测执行器已启用并行化（{mode}），最大工作进程/线程数: {max_workers}")

        if enable_performance_profiling:
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
        """
        运行回测

        Args:
            strategy_name: 策略名称
            stock_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            strategy_config: 策略配置
            backtest_config: 回测配置
            task_id: 任务ID

        Returns:
            回测报告字典
        """
        # 轻量分段计时
        perf_breakdown: Dict[str, float] = {}
        _t_total0 = time.perf_counter()

        # 启动性能追踪
        self.performance_tracker.start_backtest()

        try:
            self.execution_stats["total_backtests"] += 1

            # 生成回测ID
            backtest_id = f"bt_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(str(stock_codes))}"

            # 使用默认配置
            if backtest_config is None:
                backtest_config = BacktestConfig()

            # 开始进度监控
            if task_id:
                await self.progress_monitor.start_backtest_monitoring(
                    task_id=task_id, backtest_id=backtest_id
                )
                await self.progress_monitor.update_stage(
                    task_id, "initialization", progress=100, status="completed"
                )

            # ========== 阶段 1: 创建策略 ==========
            _t0 = time.perf_counter()
            self.performance_tracker.start_stage(
                "strategy_setup",
                {"strategy_name": strategy_name, "stock_count": len(stock_codes)},
            )

            if task_id:
                await self.progress_monitor.update_stage(
                    task_id, "strategy_setup", status="running"
                )

            # 优先使用高级策略工厂
            try:
                strategy = AdvancedStrategyFactory.create_strategy(
                    strategy_name, strategy_config
                )
            except Exception:
                strategy = StrategyFactory.create_strategy(
                    strategy_name, strategy_config
                )

            self.performance_tracker.end_stage("strategy_setup")
            perf_breakdown["strategy_setup_s"] = time.perf_counter() - _t0

            if task_id:
                await self.progress_monitor.update_stage(
                    task_id, "strategy_setup", progress=100, status="completed"
                )

            # ========== 阶段 2: 加载数据 ==========
            _t0 = time.perf_counter()
            self.performance_tracker.start_stage(
                "data_loading",
                {
                    "stock_codes": stock_codes,
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                },
            )

            if task_id:
                await self.progress_monitor.update_stage(
                    task_id, "data_loading", status="running"
                )

            logger.info(
                f"开始回测: {strategy_name}, 股票: {stock_codes}, 期间: {start_date} - {end_date}"
            )
            stock_data = self.data_loader.load_multiple_stocks(
                stock_codes, start_date, end_date
            )

            self.performance_tracker.end_stage(
                "data_loading",
                {
                    "loaded_stocks": len(stock_data),
                    "total_records": sum(len(df) for df in stock_data.values()),
                },
            )
            self.performance_tracker.take_memory_snapshot("after_data_loading")
            perf_breakdown["data_loading_s"] = time.perf_counter() - _t0

            if task_id:
                await self.progress_monitor.update_stage(
                    task_id, "data_loading", progress=100, status="completed"
                )

            # ========== 阶段 3: 创建组合管理器 ==========
            actual_stock_codes = list(stock_data.keys())
            if self.use_array_portfolio:
                portfolio_manager = PortfolioManagerArray(backtest_config, actual_stock_codes)
                logger.info(f"✅ 使用数组化持仓管理器 (stocks={len(actual_stock_codes)})")
            else:
                portfolio_manager = PortfolioManager(backtest_config)
                logger.info(f"使用传统持仓管理器 (stocks={len(actual_stock_codes)})")

            # ========== 阶段 4: 数据预处理 ==========
            _t0 = time.perf_counter()

            # 获取交易日历
            trading_dates = self.data_preprocessor.get_trading_calendar(
                stock_data, start_date, end_date
            )

            # 构建日期索引
            self.data_preprocessor.build_date_index(stock_data)

            # 预计算信号
            self.data_preprocessor.precompute_strategy_signals(strategy, stock_data)

            # 提取预计算信号
            precomputed_signals = self.data_preprocessor.extract_precomputed_signals_to_dict(
                strategy, stock_data
            )

            logger.info(f"🔍 预计算信号字典大小: {len(precomputed_signals)}")

            perf_breakdown["precompute_signals_s"] = time.perf_counter() - _t0

            # 验证交易日���量
            if len(trading_dates) < 20:
                error_msg = f"交易日数量不足: {len(trading_dates)}，至少需要20个交易日"
                if task_id:
                    await self.progress_monitor.set_error(task_id, error_msg)
                raise TaskError(message=error_msg, severity=ErrorSeverity.MEDIUM)

            # 更新总交易日数
            if task_id:
                progress_data = self.progress_monitor.get_progress_data(task_id)
                if progress_data:
                    progress_data.total_trading_days = len(trading_dates)

                # 写入数据库
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

            # ========== 阶段 5: 构建对齐数组 ==========
            _t1 = time.perf_counter()
            aligned_arrays = self.data_preprocessor.build_aligned_arrays(
                strategy, stock_data, trading_dates
            )
            perf_breakdown["align_arrays_s"] = time.perf_counter() - _t1

            # ========== 阶段 6: 执行回测循环 ==========
            self.performance_tracker.start_stage(
                "backtest_execution",
                {
                    "total_trading_days": len(trading_dates),
                    "stock_count": len(stock_data),
                },
            )

            if task_id:
                await self.progress_monitor.update_stage(
                    task_id, "backtest_execution", status="running"
                )

            _t0 = time.perf_counter()
            backtest_results = await self.loop_executor.execute_backtest_loop(
                strategy=strategy,
                portfolio_manager=portfolio_manager,
                stock_data=stock_data,
                trading_dates=trading_dates,
                strategy_config=strategy_config,
                task_id=task_id,
                backtest_id=backtest_id,
                precomputed_signals=precomputed_signals,
                aligned_arrays=aligned_arrays,
            )
            perf_breakdown["main_loop_s"] = time.perf_counter() - _t0

            self.performance_tracker.end_stage(
                "backtest_execution",
                {
                    "total_signals": backtest_results.get("total_signals", 0),
                    "executed_trades": backtest_results.get("executed_trades", 0),
                    "trading_days": backtest_results.get("trading_days", 0),
                },
            )
            self.performance_tracker.take_memory_snapshot("after_backtest_execution")

            if task_id:
                await self.progress_monitor.update_stage(
                    task_id, "backtest_execution", progress=100, status="completed"
                )

            # ========== 阶段 7: 计算绩效指标 ==========
            self.performance_tracker.start_stage("metrics_calculation")

            if task_id:
                await self.progress_monitor.update_stage(
                    task_id, "metrics_calculation", status="running"
                )

            _t0 = time.perf_counter()
            performance_metrics = portfolio_manager.get_performance_metrics()
            perf_breakdown["metrics_s"] = time.perf_counter() - _t0

            self.performance_tracker.end_stage("metrics_calculation")

            if task_id:
                await self.progress_monitor.update_stage(
                    task_id, "metrics_calculation", progress=100, status="completed"
                )

            # ========== 阶段 8: 生成回测报告 ==========
            self.performance_tracker.start_stage("report_generation")

            if task_id:
                await self.progress_monitor.update_stage(
                    task_id, "report_generation", status="running"
                )

            if strategy_config and isinstance(strategy_config, dict) and len(strategy_config) > 0:
                logger.info(f"生成回测报告，策略配置: {strategy_config}")
            else:
                logger.warning(f"策略配置为空或无效: {strategy_config}")

            _t0 = time.perf_counter()
            backtest_report = self.report_generator.generate_backtest_report(
                strategy_name=strategy_name,
                stock_codes=stock_codes,
                start_date=start_date,
                end_date=end_date,
                config=backtest_config,
                portfolio_manager=portfolio_manager,
                performance_metrics=performance_metrics,
                strategy_config=strategy_config,
            )
            perf_breakdown["report_generation_s"] = time.perf_counter() - _t0

            # 添加回测循环统计
            backtest_report["total_signals"] = backtest_results.get("total_signals", 0)
            backtest_report["trading_days"] = backtest_results.get("trading_days", 0)

            # P0: 添加熔断统计到报告
            cb_summary = backtest_results.get("circuit_breaker_summary")
            if cb_summary:
                backtest_report["circuit_breaker"] = cb_summary

            self.performance_tracker.end_stage(
                "report_generation", {"report_size": len(str(backtest_report))}
            )

            if task_id:
                await self.progress_monitor.update_stage(
                    task_id, "report_generation", progress=100, status="completed"
                )
                await self.progress_monitor.update_stage(
                    task_id, "data_storage", progress=100, status="completed"
                )

            self.execution_stats["successful_backtests"] += 1
            logger.info(
                f"回测完成: {strategy_name}, 总收益: {performance_metrics.get('total_return', 0):.2%}"
            )

            # 完成监控
            if task_id:
                await self.progress_monitor.complete_backtest(
                    task_id,
                    {"total_return": performance_metrics.get("total_return", 0)},
                )

            # ========== 阶段 9: 生成性能报告 ==========
            self.performance_tracker.end_backtest()
            self.performance_tracker.take_memory_snapshot("backtest_end")

            # 将性能报告添加到回测报告中
            performance_report = self.performance_tracker.generate_report()
            if performance_report:
                backtest_report["performance_analysis"] = performance_report
                self.performance_tracker.print_summary()

                # 保存性能报告
                if task_id:
                    try:
                        performance_dir = Path("backend/data/performance_reports")
                        performance_dir.mkdir(parents=True, exist_ok=True)
                        performance_file = performance_dir / f"backtest_{task_id}_performance.json"
                        self.performance_tracker.save_report(str(performance_file))
                        logger.info(f"性能报告已保存到: {performance_file}")
                    except Exception as e:
                        logger.warning(f"保存性能报告失败: {e}")

            # 添加分段计时结果
            perf_breakdown["total_wall_s"] = time.perf_counter() - _t_total0
            backtest_report["perf_breakdown"] = perf_breakdown

            return backtest_report

        except Exception as e:
            self.execution_stats["failed_backtests"] += 1
            error_msg = f"回测执行失败: {str(e)}"

            # 结束性能分析
            try:
                self.performance_tracker.end_backtest()
                logger.warning("回测失败，但性能分析已完成")
            except Exception as perf_error:
                logger.warning(f"结束性能分析时出错: {perf_error}")

            if task_id:
                await self.progress_monitor.set_error(task_id, error_msg)

            raise TaskError(
                message=error_msg, severity=ErrorSeverity.HIGH, original_exception=e
            )

    def validate_backtest_parameters(
        self,
        strategy_name: str,
        stock_codes: List[str],
        start_date: datetime,
        end_date: datetime,
        strategy_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """验证回测参数"""
        return validate_backtest_parameters(
            strategy_name, stock_codes, start_date, end_date, strategy_config
        )

    def get_execution_statistics(self) -> Dict[str, Any]:
        """获取执行统计信息"""
        return get_execution_statistics(self.execution_stats)
