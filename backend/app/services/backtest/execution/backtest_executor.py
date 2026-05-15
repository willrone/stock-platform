"""
回测执行器 - 完整的回测流程执行和结果分析
"""

import time
from concurrent.futures import as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped,unused-ignore]
from loguru import logger

from app.core.error_handler import ErrorSeverity, TaskError

from ..core.base_strategy import BaseStrategy
from ..core.portfolio_manager import PortfolioManager
from ..core.portfolio_manager_array import PortfolioManagerArray
from ..models import BacktestConfig, SignalType, TradingSignal
from ..reporting import BacktestReportBuilder, BacktestReportBuildInput
from ..strategies.strategy_factory import AdvancedStrategyFactory, StrategyFactory
from .backtest_progress_monitor import backtest_progress_monitor
from .data_loader import DataLoader
from .trade_modes import TradeModeExecutionContext, get_trade_mode_executor

# 性能监控（可选导入，避免依赖问题）
# isort: off
BacktestPerformanceProfiler: Any = None
PerformanceContext: Any = None
try:
    from ..utils.performance_profiler import (
        BacktestPerformanceProfiler as _BacktestPerformanceProfiler,
        PerformanceContext as _PerformanceContext,
    )

    BacktestPerformanceProfiler = _BacktestPerformanceProfiler
    PerformanceContext = _PerformanceContext
    PERFORMANCE_PROFILING_AVAILABLE = True
except ImportError:
    PERFORMANCE_PROFILING_AVAILABLE = False
# isort: on


def _multiprocess_precompute_worker(
    task: Tuple,
) -> Tuple[bool, str, Optional[Dict], Optional[str]]:
    """
    多进程预计算 worker 函数（模块级，可被 pickle 序列化）。

    Args:
        task: (stock_code, data_dict, strategy_info) 元组

    Returns:
        (success, stock_code, signals_dict, error_message)
    """
    stock_code, data_dict, strategy_info = task

    try:
        # 重建 DataFrame
        df = pd.DataFrame(data_dict["values"], columns=data_dict["columns"])
        df.index = pd.to_datetime(data_dict["index"])
        df.attrs["stock_code"] = data_dict["stock_code"]

        # 重建策略对象
        from ..strategies.strategy_factory import (
            AdvancedStrategyFactory,
            StrategyFactory,
        )

        strategy_name = strategy_info["name"]  # 使用策略名称（如 "MACD"）
        strategy_class_name = strategy_info["class_name"]  # 类名（如 "MACDStrategy"）
        strategy_config = strategy_info["config"]

        # 尝试从工厂创建策略（尝试多种名称格式）
        strategy = None
        names_to_try = [
            strategy_name,  # 原始名称
            strategy_name.lower(),  # 小写
            strategy_class_name,  # 类名
            strategy_class_name.replace("Strategy", ""),  # 去掉 Strategy 后缀
            strategy_class_name.replace("Strategy", "").lower(),  # 去掉后缀并小写
        ]

        for name in names_to_try:
            if strategy is not None:
                break
            try:
                strategy = StrategyFactory.create_strategy(name, strategy_config)
            except Exception:
                try:
                    strategy = AdvancedStrategyFactory.create_strategy(
                        name, strategy_config
                    )
                except Exception:
                    pass

        if strategy is None:
            return (
                False,
                stock_code,
                None,
                f"无法创建策略 {strategy_name} (尝试了: {names_to_try})",
            )

        # 执行向量化预计算
        signals = strategy.precompute_all_signals(df)

        if signals is not None:
            # 将 Series 转换为可序列化格式
            signals_dict = {
                "values": signals.tolist(),
                "index": [str(idx) for idx in signals.index],
            }
            return (True, stock_code, signals_dict, None)
        else:
            return (False, stock_code, None, "precompute_all_signals 返回 None")

    except Exception as e:
        return (False, stock_code, None, str(e))


class BacktestExecutor:
    """回测执行器"""

    def __init__(
        self,
        data_dir: str = "data",
        enable_parallel: bool = True,
        max_workers: Optional[int] = None,
        enable_performance_profiling: bool = False,
        use_multiprocessing: bool = False,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> None:
        """
        初始化回测执行器

        Args:
            data_dir: 数据目录
            enable_parallel: 是否启用并行化（默认True）
            max_workers: 最大工作线程数，默认使用CPU核心数
            enable_performance_profiling: 是否启用性能分析（默认False）
            use_multiprocessing: 是否使用多进程（突破GIL限制，默认False）
                - True: 使用 ProcessPoolExecutor，适合 CPU 密集型策略
                - False: 使用 ThreadPoolExecutor，序列化开销小
            progress_callback: 可选的任务进度回调，参数为 (progress, message)
        """
        import os

        if max_workers is None:
            max_workers = min(
                os.cpu_count() or 4, 8
            )  # 最多8个线程，避免过多线程导致开销

        self.enable_parallel = enable_parallel
        self.max_workers = max_workers
        self.use_multiprocessing = use_multiprocessing
        self.use_array_portfolio = True  # Phase 1: 启用数组化持仓管理
        self.progress_callback = progress_callback
        self.data_loader = DataLoader(
            data_dir, max_workers=max_workers if enable_parallel else None
        )
        self.execution_stats: Dict[str, int] = {
            "total_backtests": 0,
            "successful_backtests": 0,
            "failed_backtests": 0,
        }
        self.report_builder = BacktestReportBuilder()

        # 性能分析器（可选）
        self.enable_performance_profiling = (
            enable_performance_profiling and PERFORMANCE_PROFILING_AVAILABLE
        )
        self.performance_profiler: Any = None

        if enable_parallel:
            mode = "多进程" if use_multiprocessing else "多线程"
            logger.info(
                f"回测执行器已启用并行化（{mode}），最大工作进程/线程数: {max_workers}"
            )

        if self.enable_performance_profiling:
            logger.info("回测执行器已启用性能分析")

    def _profiler(self) -> Any:
        """Return active profiler after runtime availability checks."""
        if self.performance_profiler is None:
            raise RuntimeError("性能分析器尚未初始化")
        return self.performance_profiler

    def _get_required_data_columns(
        self, strategy_name: str, strategy_config: Dict[str, Any]
    ) -> Optional[List[str]]:
        """Return minimal data columns needed for common strategies."""
        normalized = strategy_name.lower()
        if normalized == "moving_average":
            # Keep moving_average semantics based on raw close rolling means.
            # Qlib MA* feature columns are factor-style features, not plain price MAs.
            return ["open", "high", "low", "close", "volume"]
        return None

    async def run_backtest(
        self,
        strategy_name: str,
        stock_codes: List[str],
        start_date: datetime,
        end_date: datetime,
        strategy_config: Dict[str, Any],
        backtest_config: Optional[BacktestConfig] = None,
        task_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """运行回测"""
        # 轻量分段计时（始终可用，不依赖 performance_profiler）
        perf_breakdown: Dict[str, float] = {}
        _t_total0 = time.perf_counter()

        # 初始化性能分析器
        if self.enable_performance_profiling:
            self.performance_profiler = BacktestPerformanceProfiler(
                enable_memory_tracking=True
            )
            self._profiler().start_backtest()
            self._profiler().take_memory_snapshot("backtest_start")

        try:
            self.execution_stats["total_backtests"] += 1

            # 生成回测ID
            backtest_id = f"bt_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(str(stock_codes))}"

            # 使用默认配置
            if backtest_config is None:
                backtest_config = BacktestConfig()

            # 开始进度监控
            if task_id or "":
                await backtest_progress_monitor.start_backtest_monitoring(
                    task_id=(task_id or ""), backtest_id=backtest_id
                )
                await backtest_progress_monitor.update_stage(
                    (task_id or ""), "initialization", progress=100, status="completed"
                )

            # 创建策略（性能监控）
            _t0 = time.perf_counter()
            if self.enable_performance_profiling:
                self._profiler().start_stage(
                    "strategy_setup",
                    {"strategy_name": strategy_name, "stock_count": len(stock_codes)},
                )

            if task_id or "":
                await backtest_progress_monitor.update_stage(
                    task_id or "", "strategy_setup", status="running"
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
                self._profiler().end_stage("strategy_setup")
            perf_breakdown["strategy_setup_s"] = time.perf_counter() - _t0

            if task_id:
                await backtest_progress_monitor.update_stage(
                    task_id, "strategy_setup", progress=100, status="completed"
                )

            # 创建组合管理器
            # Phase 1: 数据加载后再创建（需要 stock_codes）
            portfolio_manager: Any = None

            # 加载数据（性能监控）
            _t0 = time.perf_counter()
            if self.enable_performance_profiling:
                self._profiler().start_stage(
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

            def _data_loading_progress(
                current: int, total: int, message: str = ""
            ) -> None:
                if not self.progress_callback or total <= 0:
                    return
                # Overall task progress: 10% after setup, 30% after data loading.
                data_stage_progress = min(max(current / total, 0.0), 1.0)
                overall_progress = 10.0 + data_stage_progress * 20.0
                self.progress_callback(overall_progress, message)

            required_data_columns = self._get_required_data_columns(
                strategy_name, strategy_config
            )
            if required_data_columns:
                logger.info(
                    f"策略 {strategy_name} 使用列裁剪加载数据: {required_data_columns}"
                )

            stock_data = self.data_loader.load_multiple_stocks(
                stock_codes,
                start_date,
                end_date,
                progress_callback=_data_loading_progress if task_id else None,
                required_columns=required_data_columns,
            )

            if self.enable_performance_profiling:
                self._profiler().end_stage(
                    "data_loading",
                    {
                        "loaded_stocks": len(stock_data),
                        "total_records": sum(len(df) for df in stock_data.values()),
                    },
                )
                self._profiler().take_memory_snapshot("after_data_loading")
            perf_breakdown["data_loading_s"] = time.perf_counter() - _t0

            if task_id:
                await backtest_progress_monitor.update_stage(
                    task_id, "data_loading", progress=100, status="completed"
                )

            # Phase 1: 数据加载后创建组合管理器（使用实际加载的股票列表）
            actual_stock_codes = list(stock_data.keys())
            if self.use_array_portfolio:
                portfolio_manager = PortfolioManagerArray(
                    backtest_config, actual_stock_codes
                )
                logger.info(
                    f"✅ Phase 1: 使用数组化持仓管理器 (stocks={len(actual_stock_codes)})"
                )
            else:
                portfolio_manager = PortfolioManager(backtest_config)
                logger.info(f"使用传统持仓管理器 (stocks={len(actual_stock_codes)})")

            # 获取交易日历
            trading_dates = self._get_trading_calendar(stock_data, start_date, end_date)

            # 预处理（日期索引 + 预计算信号 + 信号提取）
            _t0 = time.perf_counter()

            # ✅ 日期预索引
            _t_sub = time.perf_counter()
            self._build_date_index(stock_data)
            perf_breakdown["precompute_sub_build_date_index_s"] = (
                time.perf_counter() - _t_sub
            )

            # ✅ 策略异步预热（如模型预测序列）
            _t_sub = time.perf_counter()
            await self._prepare_strategy_backtest_data(
                strategy,
                stock_data,
                start_date,
                end_date,
            )
            perf_breakdown["precompute_sub_strategy_prepare_s"] = (
                time.perf_counter() - _t_sub
            )

            # ✅ 信号向量化预计算
            _t_sub = time.perf_counter()
            self._precompute_strategy_signals(strategy, stock_data)
            perf_breakdown["precompute_sub_strategy_signals_s"] = (
                time.perf_counter() - _t_sub
            )

            # ✅ 信号提取优化
            _t_sub = time.perf_counter()
            precomputed_signals = self._extract_precomputed_signals_to_dict(
                strategy, stock_data
            )
            perf_breakdown["precompute_sub_extract_signals_s"] = (
                time.perf_counter() - _t_sub
            )

            # 🔍 调试日志：检查预计算信号
            logger.info(f"🔍 预计算信号字典大小: {len(precomputed_signals)}")
            if precomputed_signals:
                sample_keys = list(precomputed_signals.keys())[:3]
                for k in sample_keys:
                    logger.info(f"  示例 key: {k}, value: {precomputed_signals[k]}")

            perf_breakdown["precompute_signals_s"] = time.perf_counter() - _t0
            # align_arrays_s 统计在 main_loop 前单独记录

            # 注：指标预热（_warm_indicator_cache）如果在主线程顺序执行，可能会把原本并行的指标计算串行化，
            # 因而未默认开启；后续可按需实现并行预热。

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
                        existing_task = task_repo.get_task_by_id((task_id or ""))
                        if existing_task:
                            result_data: Dict[str, Any] = (
                                cast(Any, existing_task.result) or {}
                            )
                            progress_data_db = result_data.get("progress_data", {})
                            progress_data_db["total_days"] = len(trading_dates)
                            result_data["progress_data"] = progress_data_db

                            task_repo.update_task_status(
                                task_id=(task_id or ""),
                                status=TaskStatus.RUNNING,
                                result=result_data,
                            )
                    finally:
                        session.close()
                except Exception as e:
                    logger.warning(f"更新总交易日数失败: {e}")

            # 执行回测（性能监控）
            if self.enable_performance_profiling:
                self._profiler().start_stage(
                    "backtest_execution",
                    {
                        "total_trading_days": len(trading_dates),
                        "stock_count": len(stock_data),
                    },
                )

            if task_id:
                await backtest_progress_monitor.update_stage(
                    (task_id or ""), "backtest_execution", status="running"
                )

            _t0 = time.perf_counter()
            # Phase1 预备：将 close/valid/signal 对齐成 ndarray，减少主循环 DataFrame/dict 访问
            _t1 = time.perf_counter()
            aligned_arrays = self._build_aligned_arrays(
                strategy, stock_data, trading_dates, perf_breakdown=perf_breakdown
            )
            perf_breakdown["align_arrays_s"] = time.perf_counter() - _t1

            backtest_results = await self._execute_backtest_loop(
                strategy,
                portfolio_manager,
                stock_data,
                trading_dates,
                strategy_config=strategy_config,
                task_id=task_id,
                backtest_id=backtest_id,
                precomputed_signals=precomputed_signals,
                aligned_arrays=aligned_arrays,
                perf_breakdown=perf_breakdown,
            )
            perf_breakdown["main_loop_s"] = time.perf_counter() - _t0

            if self.enable_performance_profiling:
                self._profiler().end_stage(
                    "backtest_execution",
                    {
                        "total_signals": backtest_results.get("total_signals", 0),
                        "executed_trades": backtest_results.get("executed_trades", 0),
                        "trading_days": backtest_results.get("trading_days", 0),
                    },
                )
                self._profiler().update_backtest_stats(
                    signals=backtest_results.get("total_signals", 0),
                    trades=backtest_results.get("executed_trades", 0),
                    days=backtest_results.get("trading_days", 0),
                )
                self._profiler().take_memory_snapshot("after_backtest_execution")

            if task_id:
                await backtest_progress_monitor.update_stage(
                    task_id, "backtest_execution", progress=100, status="completed"
                )

            # 计算绩效指标（性能监控）
            if self.enable_performance_profiling:
                self._profiler().start_stage("metrics_calculation")

            if task_id:
                await backtest_progress_monitor.update_stage(
                    task_id, "metrics_calculation", status="running"
                )

            _t0 = time.perf_counter()
            performance_metrics = portfolio_manager.get_performance_metrics()
            perf_breakdown["metrics_s"] = time.perf_counter() - _t0

            if self.enable_performance_profiling:
                self._profiler().end_stage("metrics_calculation")

            if task_id:
                await backtest_progress_monitor.update_stage(
                    task_id, "metrics_calculation", progress=100, status="completed"
                )

            # 生成回测报告（性能监控）
            if self.enable_performance_profiling:
                self._profiler().start_stage("report_generation")

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

            _t0 = time.perf_counter()
            report_input = BacktestReportBuildInput(
                strategy_name=strategy_name,
                stock_codes=stock_codes,
                start_date=start_date,
                end_date=end_date,
                config=backtest_config,
                portfolio_manager=portfolio_manager,
                performance_metrics=performance_metrics,
                strategy_config=strategy_config,
            )
            backtest_report: Dict[str, Any] = self.report_builder.build_report(
                report_input
            )
            perf_breakdown["report_generation_s"] = time.perf_counter() - _t0
            self.report_builder.attach_runtime_diagnostics(
                backtest_report,
                backtest_results,
                perf_breakdown,
            )

            # 汇总信号执行统计，补充可执行口径执行率与 Top 拒绝原因
            if task_id:
                try:
                    from app.core.database import get_async_session_context
                    from app.repositories.backtest_detailed_repository import (
                        BacktestDetailedRepository,
                    )

                    async with get_async_session_context() as async_session:
                        repo = BacktestDetailedRepository(async_session)
                        signal_stats = await repo.get_signal_statistics((task_id or ""))
                    self.report_builder.attach_signal_execution_summary(
                        backtest_report,
                        signal_stats,
                    )
                except Exception as sig_err:
                    logger.warning(f"获取信号执行统计失败: {sig_err}")
                    backtest_report["signal_execution_summary"] = {}

            if self.enable_performance_profiling:
                self._profiler().end_stage(
                    "report_generation", {"report_size": len(str(backtest_report))}
                )

            if task_id:
                await backtest_progress_monitor.update_stage(
                    (task_id or ""),
                    "report_generation",
                    progress=100,
                    status="completed",
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
                self._profiler().end_backtest()
                self._profiler().take_memory_snapshot("backtest_end")

                # 将性能报告添加到回测报告中
                performance_report = self._profiler().generate_report()
                backtest_report["performance_analysis"] = performance_report

                # 打印性能摘要
                self._profiler().print_summary()

                # 保存性能报告到文件（如果提供了task_id）
                if task_id:
                    try:
                        performance_dir = (
                            Path(self.data_loader.data_dir) / "performance_reports"
                        )
                        performance_dir.mkdir(parents=True, exist_ok=True)
                        performance_file = (
                            performance_dir / f"backtest_{task_id}_performance.json"
                        )
                        self._profiler().save_report(str(performance_file))
                        logger.info(f"性能报告已保存到: {performance_file}")
                    except Exception as e:
                        logger.warning(f"保存性能报告失败: {e}")

            # 轻量分段计时结果写入报告（bench脚本唯一入口依赖此字段）
            perf_breakdown["total_wall_s"] = time.perf_counter() - _t_total0
            self.report_builder.attach_runtime_diagnostics(
                backtest_report,
                backtest_results,
                perf_breakdown,
            )

            return backtest_report

        except Exception as e:
            self.execution_stats["failed_backtests"] += 1
            error_msg = f"回测执行失败: {str(e)}"

            # 即使出错也结束性能分析
            if self.enable_performance_profiling and self.performance_profiler:
                try:
                    self._profiler().end_backtest()
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
        all_dates: set[datetime] = set()
        for data in stock_data.values():
            all_dates.update(data.index.tolist())

        # 过滤日期范围并排序
        trading_dates = np.sort(
            np.array([date for date in all_dates if start_date <= date <= end_date])
        ).tolist()

        return cast(List[datetime], trading_dates)

    def _build_date_index(self, stock_data: Dict[str, pd.DataFrame]) -> None:
        """为每只股票建立日期->整数索引，避免回测循环中重复 get_loc。"""
        for data in stock_data.values():
            try:
                if "_date_to_idx" not in data.attrs:
                    data.attrs["_date_to_idx"] = {
                        d: i for i, d in enumerate(data.index)
                    }
            except Exception:
                pass

    def _warm_indicator_cache(
        self,
        strategy: BaseStrategy,
        stock_data: Dict[str, pd.DataFrame],
    ) -> None:
        """回测开始前预计算并缓存所有股票的指标，避免首日/首股现场计算。"""
        try:
            from ..core.strategy_portfolio import StrategyPortfolio

            if isinstance(strategy, StrategyPortfolio):
                for sub in strategy.strategies:
                    self._warm_indicator_cache(sub, stock_data)
                return
        except Exception:
            pass
        for data in stock_data.values():
            try:
                strategy.get_cached_indicators(data)
            except Exception:
                pass

    async def _prepare_strategy_backtest_data(
        self,
        strategy: BaseStrategy,
        stock_data: Dict[str, pd.DataFrame],
        start_date: datetime,
        end_date: datetime,
    ) -> None:
        """递归执行策略的异步预热逻辑（如模型预测序列准备）。"""
        try:
            from ..core.strategy_portfolio import StrategyPortfolio

            if isinstance(strategy, StrategyPortfolio):
                for sub in strategy.strategies:
                    await self._prepare_strategy_backtest_data(
                        sub,
                        stock_data,
                        start_date,
                        end_date,
                    )
                return
        except Exception:
            pass

        prepare_hook = getattr(strategy, "prepare_backtest_data", None)
        if prepare_hook is None:
            return

        await prepare_hook(stock_data, start_date, end_date)

    def _precompute_strategy_signals(
        self,
        strategy: BaseStrategy,
        stock_data: Dict[str, pd.DataFrame],
    ) -> None:
        """[性能优化] 在回测循环开始前，尝试对所有股票进行向量化信号预计算。"""
        try:
            from ..core.strategy_portfolio import StrategyPortfolio

            if isinstance(strategy, StrategyPortfolio):
                logger.info(
                    f"🚀 Portfolio策略检测到，递归预计算 {len(strategy.strategies)} 个子策略"
                )
                for sub in strategy.strategies:
                    self._precompute_strategy_signals(sub, stock_data)
                return
        except Exception as e:
            logger.warning(f"Portfolio策略递归预计算失败: {e}")

        # 统计预计算成功的股票数
        success_count = 0
        total_stocks = len(stock_data)

        # 并行预计算（按股票维度），显著降低整体 wall-time
        # 注：使用 ProcessPoolExecutor 可突破 GIL 限制，但需要序列化数据
        # 这里使用混合策略：CPU 密集型任务用多进程，I/O 密集型用多线程
        getattr(self, "use_multiprocessing", False)

        _stock_times: List[Tuple[str, float, int]] = []  # 收集每只股票的预计算耗时

        def _work_one(
            item: Tuple[str, pd.DataFrame],
        ) -> Tuple[bool, str, Optional[str]]:
            stock_code, data = item
            try:
                import time as _time

                _t = _time.perf_counter()
                all_sigs = strategy.precompute_all_signals(data)
                _elapsed = _time.perf_counter() - _t
                _stock_times.append((stock_code, _elapsed, len(data)))
                if all_sigs is not None:
                    cache = data.attrs.setdefault("_precomputed_signals", {})
                    cache[strategy.name] = all_sigs
                    return True, stock_code, None
                return False, stock_code, None
            except Exception as e:
                return False, stock_code, str(e)

        # [性能优化] 强制串行执行：precompute 是 CPU 密集型（pandas/numpy），
        # ThreadPool 受 GIL 限制反而更慢（实测：串行 28s vs 并行 79s vs Web并行 390s）。
        # 多进程虽能突破 GIL，但 DataFrame 序列化开销远大于计算本身。
        for it in stock_data.items():
            ok, stock_code, err = _work_one(it)
            if ok:
                success_count += 1
            elif err:
                logger.warning(
                    f"策略 {strategy.name} 对股票 {stock_code} 预计算信号失败: {err}"
                )

        # 📊 性能统计：预计算耗时分布
        if _stock_times:
            times_only = [t for _, t, _ in _stock_times]
            total_time = sum(times_only)
            avg_time = total_time / len(times_only)
            max_item = max(_stock_times, key=lambda x: x[1])
            min_item = min(_stock_times, key=lambda x: x[1])
            logger.info(
                f"📊 策略 {strategy.name} 预计算性能统计: "
                f"总计={total_time:.2f}s, 平均={avg_time * 1000:.1f}ms/股, "
                f"最慢={max_item[0]}({max_item[1] * 1000:.1f}ms, {max_item[2]}行), "
                f"最快={min_item[0]}({min_item[1] * 1000:.1f}ms, {min_item[2]}行), "
                f"股票数={len(_stock_times)}"
            )
            # 记录 top5 最慢的股票
            sorted_times = sorted(_stock_times, key=lambda x: x[1], reverse=True)[:5]
            for code, t, rows in sorted_times:
                logger.info(f"  🐢 慢股: {code} = {t * 1000:.1f}ms ({rows}行)")

        if success_count > 0:
            logger.info(
                f"✅ 策略 {strategy.name} 向量化预计算完成: {success_count}/{total_stocks} 只股票"
            )

    def _extract_precomputed_signals_to_dict(
        self,
        strategy: BaseStrategy,
        stock_data: Dict[str, pd.DataFrame],
    ) -> Dict[Tuple[str, datetime], Any]:
        """
        [性能优化] 将预计算的信号从 DataFrame.attrs 提取到扁平字典。

        这样在回测循环中可以直接用 (stock_code, date) 查找信号，
        避免每次都访问 attrs 字典和 id(strategy) 查找。

        Returns:
            Dict[(stock_code, date), signal]: 扁平的信号字典
        """
        signal_dict = {}

        try:
            from ..core.strategy_portfolio import StrategyPortfolio
            from ..models import TradingSignal

            if isinstance(strategy, StrategyPortfolio):
                logger.info(
                    f"🔄 Portfolio策略信号整合开始: {len(strategy.strategies)} 个子策略"
                )

                # 1. 递归提取所有子策略的信号
                all_sub_signals: Dict[Tuple[str, datetime], Any] = {}
                for sub in strategy.strategies:
                    sub_signals = self._extract_precomputed_signals_to_dict(
                        sub, stock_data
                    )
                    all_sub_signals.update(sub_signals)

                logger.info(f"📊 子策略信号总数: {len(all_sub_signals)}")

                # 2. 按日期分组子策略信号
                from collections import defaultdict

                signals_by_date: Dict[datetime, List[TradingSignal]] = defaultdict(list)

                for (stock_code, date), signal_type in all_sub_signals.items():
                    # 构造 TradingSignal 对象

                    if signal_type == SignalType.BUY or signal_type == SignalType.SELL:
                        # 获取价格
                        try:
                            df = stock_data.get(stock_code)
                            if df is not None and date in df.index:
                                price = float(df.loc[date, "close"])
                                signal = TradingSignal(
                                    timestamp=date,
                                    stock_code=stock_code,
                                    signal_type=signal_type,
                                    strength=1.0,
                                    price=price,
                                    reason="precomputed",
                                    metadata={},
                                )
                                signals_by_date[date].append(signal)
                        except Exception as e:
                            logger.warning(f"构造信号失败 {stock_code} @ {date}: {e}")

                # 3. 对每个日期的信号进行整合
                integrated_count = 0
                for _date, signals in signals_by_date.items():
                    if signals:
                        # 调用 Portfolio 的信号整合器
                        integrated = strategy.integrator.integrate(
                            signals, strategy.weights, consistency_threshold=0.6
                        )

                        # 将整合后的信号添加到字典
                        for sig in integrated:
                            signal_dict[(sig.stock_code, sig.timestamp)] = (
                                sig.signal_type
                            )
                            integrated_count += 1

                logger.info(
                    f"✅ Portfolio策略信号整合完成: {integrated_count} 个整合信号"
                )
                return signal_dict

        except Exception as e:
            logger.warning(f"Portfolio策略信号提取失败: {e}")
            import traceback

            logger.warning(traceback.format_exc())

        # 提取单个策略的信号
        # 使用 strategy.name 作为稳定的 key，避免多进程环境下 id() 变化
        strategy_key = strategy.name
        extracted_count = 0

        for stock_code, data in stock_data.items():
            try:
                precomputed = data.attrs.get("_precomputed_signals", {})
                signals = precomputed.get(strategy_key)

                if signals is not None:
                    # signals 可能是 pd.Series 或 dict
                    if isinstance(signals, pd.Series):
                        for date, signal in signals.items():
                            if signal is not None:
                                signal_dict[(stock_code, date)] = signal
                                extracted_count += 1
                    elif isinstance(signals, dict):
                        for date, signal in signals.items():
                            if signal is not None:
                                signal_dict[(stock_code, date)] = signal
                                extracted_count += 1
            except Exception as e:
                logger.warning(f"提取股票 {stock_code} 的信号失败: {e}")

        if extracted_count > 0:
            logger.info(
                f"✅ 策略 {strategy.name} 信号提取完成: {extracted_count} 个信号"
            )

        return signal_dict

    def _build_aligned_arrays(
        self,
        strategy: BaseStrategy,
        stock_data: Dict[str, pd.DataFrame],
        trading_dates: List[datetime],
        perf_breakdown: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """[Phase3] 将数据/信号对齐到 ndarray，减少主循环 DataFrame/字典访问。

        优化点：
        1. 使用 numpy 的 searchsorted 加速日期查找
        2. 批量填充数组，减少循环
        3. 使用 .values 避免 pandas 开销

        Returns:
            {
              'stock_codes': [...],
              'dates': np.ndarray[datetime64],
              'close': float64[N,T] (nan=missing),
              'open':  float64[N,T] (nan=missing),
              'valid': bool[N,T],
              'signal': int8[N,T] (1=BUY, -1=SELL, 0=NONE)
            }
        """
        _t_align_alloc = time.perf_counter()
        stock_codes = list(stock_data.keys())
        T = len(trading_dates)
        N = len(stock_codes)

        dates64 = np.array(trading_dates, dtype="datetime64[ns]")

        # 预分配数组（Phase 3 优化：使用连续内存）
        close: np.ndarray[Any, Any] = np.full(
            (N, T), np.nan, dtype=np.float64, order="C"
        )
        open_: np.ndarray[Any, Any] = np.full(
            (N, T), np.nan, dtype=np.float64, order="C"
        )
        valid: np.ndarray[Any, Any] = np.zeros((N, T), dtype=bool, order="C")
        signal: np.ndarray[Any, Any] = np.zeros((N, T), dtype=np.int8, order="C")
        _align_alloc_s = time.perf_counter() - _t_align_alloc

        # 如果已做向量化预计算��号，尽量直接读取 per-stock Series 并对齐到 trading_dates
        strategy_key = strategy.name  # 使用 strategy.name 作为稳定的 key

        # Phase 4 优化：将 trading_dates 转为 datetime64 数组，用于 searchsorted
        _td_ns = np.array(trading_dates, dtype="datetime64[ns]")

        _t_align_price = time.perf_counter()
        _acc_align_signal = 0.0
        for i, code in enumerate(stock_codes):
            df = stock_data[code]

            # Phase 4 优化：纯 numpy searchsorted 替代 pandas reindex
            # 完全绕过 pandas，避免 attrs 复制和 DataFrame 开销
            try:
                df_idx_ns = df.index.values.astype("datetime64[ns]")
                # searchsorted 找到 trading_dates 在股票日期中的插入位置
                pos = np.searchsorted(df_idx_ns, _td_ns)
                # 限制索引范围
                n_rows = len(df_idx_ns)
                pos_clipped = np.clip(pos, 0, n_rows - 1)
                # 只有精确匹配的日期才有效
                match_mask = df_idx_ns[pos_clipped] == _td_ns

                # 直接从 numpy 数组填充（绕过 pandas）
                close_vals = df["close"].values
                close[i, match_mask] = close_vals[pos_clipped[match_mask]]
                valid[i, :] = match_mask

                if "open" in df.columns:
                    open_vals = df["open"].values
                    open_[i, match_mask] = open_vals[pos_clipped[match_mask]]

            except Exception as e:
                # fallback: pandas reindex（慢速路径）
                logger.warning(f"股票 {code} numpy对齐失败，回退到pandas: {e}")
                try:
                    s_close = df["close"].reindex(trading_dates)
                    close_values = s_close.values
                    close[i, :] = close_values
                    if "open" in df.columns:
                        open_[i, :] = df["open"].reindex(trading_dates).values
                    valid[i, :] = ~np.isnan(close_values)
                except Exception as e2:
                    logger.warning(f"股票 {code} pandas对齐也失败: {e2}")

            _t_align_sig_one = time.perf_counter()
            # 信号对齐：同样用 searchsorted 替代 reindex
            try:
                pre = (
                    df.attrs.get("_precomputed_signals", {})
                    if hasattr(df, "attrs")
                    else {}
                )
                sig_ser = pre.get(strategy_key)
                if isinstance(sig_ser, pd.Series):
                    # Phase 4: numpy searchsorted 对齐信号
                    sig_idx_ns = sig_ser.index.values.astype("datetime64[ns]")
                    sig_pos = np.searchsorted(sig_idx_ns, _td_ns)
                    sig_n = len(sig_idx_ns)
                    sig_pos_clipped = np.clip(sig_pos, 0, max(sig_n - 1, 0))
                    sig_match = sig_idx_ns[sig_pos_clipped] == _td_ns
                    sig_vals = sig_ser.values
                    matched_vals = sig_vals[sig_pos_clipped[sig_match]]
                    buy_mask = matched_vals == SignalType.BUY
                    sell_mask = matched_vals == SignalType.SELL
                    sig_indices = np.where(sig_match)[0]
                    signal[i, sig_indices[buy_mask]] = 1
                    signal[i, sig_indices[sell_mask]] = -1
                elif isinstance(sig_ser, dict):
                    sig_series = pd.Series(sig_ser)
                    s = sig_series.reindex(trading_dates)
                    vals = s.values
                    buy_mask = vals == SignalType.BUY
                    sell_mask = vals == SignalType.SELL
                    signal[i, buy_mask] = 1
                    signal[i, sell_mask] = -1
            except Exception as e:
                logger.warning(f"股票 {code} 信号对齐失败: {e}")
            _acc_align_signal += time.perf_counter() - _t_align_sig_one

        _align_price_s = time.perf_counter() - _t_align_price - _acc_align_signal
        _align_signal_s = _acc_align_signal

        if perf_breakdown is not None:
            perf_breakdown["align_sub_alloc_arrays_s"] = _align_alloc_s
            perf_breakdown["align_sub_price_reindex_s"] = _align_price_s
            perf_breakdown["align_sub_signal_reindex_s"] = _align_signal_s
            logger.info(
                f"📊 align_arrays 细粒度: alloc={_align_alloc_s:.2f}s, "
                f"price_reindex={_align_price_s:.2f}s, signal_reindex={_align_signal_s:.2f}s"
            )

        return {
            "stock_codes": stock_codes,
            "code_to_i": {c: idx for idx, c in enumerate(stock_codes)},
            "dates": dates64,
            "close": close,
            "open": open_,
            "valid": valid,
            "signal": signal,
        }

    @staticmethod
    def _get_position_codes_fast(portfolio_manager: Any) -> List[str]:
        """Return holding codes without forcing full Position object materialization."""
        getter = getattr(portfolio_manager, "get_position_codes", None)
        if callable(getter):
            try:
                codes = getter()
                if isinstance(codes, (str, bytes)):
                    return [str(codes)]
                return list(codes)
            except TypeError:
                # Unit-test mocks may expose get_position_codes as a callable Mock
                # that returns another non-iterable Mock. Fall through to the
                # underlying positions mapping in that case.
                pass
        positions = getattr(portfolio_manager, "positions", {})
        try:
            return list(positions.keys())
        except Exception:
            return []

    @staticmethod
    def _get_position_count_fast(portfolio_manager: Any) -> int:
        """Return holding count without forcing full Position object materialization."""
        getter = getattr(portfolio_manager, "get_position_count", None)
        if callable(getter):
            return int(getter())
        positions = getattr(portfolio_manager, "positions", {})
        try:
            return len(positions)
        except Exception:
            return 0

    @staticmethod
    def _has_position_fast(portfolio_manager: Any, stock_code: str) -> bool:
        """Return whether a position exists without building a full positions dict."""
        getter = getattr(portfolio_manager, "has_position", None)
        if callable(getter):
            return bool(getter(stock_code))
        positions = getattr(portfolio_manager, "positions", {})
        return stock_code in positions

    @staticmethod
    def _get_single_position_dict_fast(
        portfolio_manager: Any, stock_code: str, current_prices: Dict[str, float]
    ) -> Dict[str, Any]:
        """Build a one-position dict for BaseStrategy.validate_signal.

        BaseStrategy only checks the current signal's position on BUY. Passing a
        one-entry mapping preserves that behavior while avoiding full portfolio
        materialization for array-backed managers.
        """
        getter = getattr(portfolio_manager, "get_position", None)
        if callable(getter):
            try:
                position = getter(stock_code, current_prices.get(stock_code))
            except TypeError:
                position = getter(stock_code)
            return {stock_code: position} if position is not None else {}

        positions = getattr(portfolio_manager, "positions", {})
        position = positions.get(stock_code) if hasattr(positions, "get") else None
        return {stock_code: position} if position is not None else {}

    def _validate_signal_fast(
        self,
        *,
        strategy: BaseStrategy,
        signal: TradingSignal,
        portfolio_manager: Any,
        current_prices: Dict[str, float],
    ) -> Tuple[bool, Optional[str]]:
        return strategy.validate_signal(
            signal,
            portfolio_manager.get_portfolio_value(current_prices),
            self._get_single_position_dict_fast(
                portfolio_manager, signal.stock_code, current_prices
            ),
        )

    def _determine_price_lookup_codes(
        self,
        *,
        strategy: BaseStrategy,
        portfolio_manager: PortfolioManager,
        aligned_arrays: Dict[str, Any],
        date_index: int,
    ) -> set[str]:
        """Determine which symbols need prices for the current date.

        Ranking trade modes such as topk_dropout need the full cross-section even
        after the portfolio already holds topk names; otherwise new entrants never
        receive prices and cannot compete into the ranking.
        """
        codes = aligned_arrays.get("stock_codes") or []
        sig_mat = aligned_arrays.get("signal")

        need_codes = set(self._get_position_codes_fast(portfolio_manager))
        if isinstance(sig_mat, np.ndarray):
            sig_idx = np.nonzero(sig_mat[:, date_index])[0]
            for j in sig_idx.tolist():
                need_codes.add(codes[j])

        trade_mode = None
        try:
            trade_mode = strategy.get_trade_mode()
        except Exception:
            trade_mode = None

        if trade_mode == "topk_dropout":
            return set(codes)

        if not need_codes:
            return set(codes)

        return need_codes

    def _precompute_signals_multiprocess(
        self,
        strategy: BaseStrategy,
        stock_data: Dict[str, pd.DataFrame],
    ) -> List[Tuple[bool, str, Optional[str]]]:
        """
        [性能优化] 使用多进程进行信号预计算，突破 GIL 限制。

        注意：多进程需要序列化数据，因此：
        1. 将 DataFrame 转换为可序列化格式
        2. 在子进程中重建策略对象
        3. 计算完成后将结果返回主进程
        """
        from concurrent.futures import ProcessPoolExecutor

        results: List[Tuple[bool, str, Optional[str]]] = []

        # 准备可序列化的任务数据
        tasks: List[Tuple[str, Dict[str, Any], Dict[str, Any]]] = []
        for stock_code, data in stock_data.items():
            try:
                # 序列化策略配置（而非策略对象本身）
                strategy_info = {
                    "name": strategy.name,
                    "class_name": strategy.__class__.__name__,
                    "config": getattr(strategy, "config", {}),
                }
                # 将 DataFrame 转换为字典格式（可序列化）
                data_dict = {
                    "values": data.to_dict("list"),
                    "index": list(data.index),
                    "columns": list(data.columns),
                    "stock_code": data.attrs.get("stock_code", stock_code),
                }
                tasks.append((stock_code, data_dict, strategy_info))
            except Exception as e:
                logger.warning(f"准备股票 {stock_code} 数据失败: {e}")
                results.append((False, stock_code, str(e)))

        if not tasks:
            return results

        # 使用进程池并行计算
        try:
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(_multiprocess_precompute_worker, task): task[0]
                    for task in tasks
                }

                for future in as_completed(futures):
                    stock_code = futures[future]
                    try:
                        ok, code, signals_dict, err = future.result(timeout=60)
                        if ok and signals_dict is not None:
                            # 将结果写回原始 DataFrame 的 attrs
                            original_data = stock_data[code]
                            # 重建 Series
                            signals = pd.Series(
                                signals_dict["values"],
                                index=pd.to_datetime(signals_dict["index"]),
                                dtype=object,
                            )
                            cache = original_data.attrs.setdefault(
                                "_precomputed_signals", {}
                            )
                            cache[strategy.name] = (
                                signals  # 使用 strategy.name 作为稳定的 key
                            )
                            results.append((True, code, None))
                        else:
                            results.append((False, code, err))
                    except Exception as e:
                        results.append((False, stock_code, str(e)))
        except Exception as e:
            logger.error(f"多进程预计算执行失败: {e}")
            # 返回所有任务失败
            for stock_code, _, _ in tasks:
                if not any(r[1] == stock_code for r in results):
                    results.append((False, stock_code, str(e)))

        return results

    async def _execute_backtest_loop(
        self,
        strategy: BaseStrategy,
        portfolio_manager: PortfolioManager,
        stock_data: Dict[str, pd.DataFrame],
        trading_dates: List[datetime],
        strategy_config: Optional[Dict[str, Any]] = None,
        task_id: Optional[str] = None,
        backtest_id: str = "",
        precomputed_signals: Optional[Dict[Tuple[str, datetime], Any]] = None,
        aligned_arrays: Optional[Dict[str, Any]] = None,
        perf_breakdown: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """执行回测主循环"""
        total_signals = 0
        executed_trades = 0
        task_id_str: Optional[str] = task_id

        # 性能统计：信号生成时间
        signal_generation_times = []
        trade_execution_times = []

        # ========== 细粒度性能统计累加器 ==========
        _ml_price_lookup = 0.0
        _ml_signal_extract = 0.0
        _ml_trade_exec = 0.0
        _ml_portfolio_snap = 0.0
        _ml_batch_collect = 0.0
        _ml_batch_flush = 0.0
        _ml_progress_update = 0.0

        # 辅助函数：检查任务状态
        def _is_task_running(status: Any) -> bool:
            if status is None:
                return False
            status_value = getattr(status, "value", status)
            try:
                return str(status_value) == str(TaskStatus.RUNNING.value)
            except Exception:
                return False

        def check_task_status() -> bool:
            """检查任务是否仍然存在且处于运行状态"""
            if not task_id:
                return True
            try:
                from app.core.database import SessionLocal
                from app.repositories.task_repository import TaskRepository

                session = SessionLocal()
                try:
                    task_repo = TaskRepository(session)
                    task = task_repo.get_task_by_id(task_id)
                    if not task:
                        logger.warning(f"任务不存在，停止回测执行: {task_id}")
                        return False
                    if not _is_task_running(task.status):
                        logger.warning(
                            f"任务状态为 {task.status}，停止回测执行: {task_id}"
                        )
                        return False
                    return True
                finally:
                    session.close()
            except Exception as e:
                logger.warning(f"检查任务状态失败: {e}，继续执行")
                return True  # 检查失败时继续执行，避免因检查错误而中断

        # ========== PERF优化：批量收集数据库操作，循环结束后一次性写入 ==========
        # 避免在730天循环内每天都做数据库操作（原来是72秒的主要瓶颈）
        _batch_signals_data: List[dict] = []  # 收集所有信号记录
        _batch_executed_signals: List[dict] = []  # 收集已执行的信号
        _batch_unexecuted_signals: List[dict] = []  # 收集未执行的信号
        _current_backtest_id: Optional[str] = None  # 缓存 backtest_id
        _BATCH_FLUSH_THRESHOLD = 5000  # 流式写入阈值（从1000提升，减少flush次数）

        # 流式写入辅助函数：当积累足够数据时写入数据库
        async def _flush_batch_to_db(
            signals_data: List[dict],
            executed_signals: List[dict],
            unexecuted_signals: List[dict],
            backtest_id: Optional[str],
            clear_after: bool = True,
        ) -> None:
            """流式写入批量数据到数据库

            优化：INSERT 时直接带 executed 和 execution_reason 字段，
            避免后续两个昂贵的 UPDATE 操作（原先用超长 OR 链全表扫描）。
            """
            if not task_id:
                return
            total_count = (
                len(signals_data) + len(executed_signals) + len(unexecuted_signals)
            )
            if total_count == 0:
                return

            logger.info(
                f"🔄 流式写入数据库: 信号={len(signals_data)}, 已执行={len(executed_signals)}, 未执行={len(unexecuted_signals)}"
            )

            try:
                from app.core.database import get_async_session_context
                from app.repositories.backtest_detailed_repository import (
                    BacktestDetailedRepository,
                )

                # ===== 优化核心：INSERT 前合并 executed/reason 状态 =====
                # 构建已执行信号的查找集合 (stock_code, signal_type, date_str)
                executed_set = set()
                for sig in executed_signals:
                    ts = sig["timestamp"]
                    date_str = (
                        ts.strftime("%Y-%m-%d")
                        if hasattr(ts, "strftime")
                        else str(ts)[:10]
                    )
                    executed_set.add((sig["stock_code"], sig["signal_type"], date_str))

                # 构建未执行信号的原因查找字典
                unexecuted_map = {}
                for sig in unexecuted_signals:
                    ts = sig["timestamp"]
                    date_str = (
                        ts.strftime("%Y-%m-%d")
                        if hasattr(ts, "strftime")
                        else str(ts)[:10]
                    )
                    key = (sig["stock_code"], sig["signal_type"], date_str)
                    unexecuted_map[key] = sig.get("execution_reason", "未执行")

                # 在 signals_data 上直接设置 executed 和 execution_reason
                matched_exec = 0
                matched_unexec = 0
                for sig_data in signals_data:
                    ts = sig_data["timestamp"]
                    date_str = (
                        ts.strftime("%Y-%m-%d")
                        if hasattr(ts, "strftime")
                        else str(ts)[:10]
                    )
                    key = (sig_data["stock_code"], sig_data["signal_type"], date_str)

                    if key in executed_set:
                        sig_data["executed"] = True
                        sig_data["execution_reason"] = None
                        matched_exec += 1
                    elif key in unexecuted_map:
                        sig_data["executed"] = False
                        sig_data["execution_reason"] = unexecuted_map[key]
                        matched_unexec += 1
                    # else: 保持默认 executed=False, execution_reason=None

                async with get_async_session_context() as async_session:
                    try:
                        repository = BacktestDetailedRepository(async_session)

                        # 一次 INSERT 搞定，不再需要后续 UPDATE
                        if signals_data:
                            await repository.batch_save_signal_records(
                                task_id=(task_id or ""),
                                backtest_id=str(backtest_id),
                                signals_data=list(signals_data),
                            )

                        await async_session.commit()
                        logger.info(
                            f"✅ 流式写入完成: {len(signals_data)} 条信号记录"
                            f"（executed={matched_exec}, unexecuted={matched_unexec}, "
                            f"default={len(signals_data) - matched_exec - matched_unexec}）"
                        )

                    except Exception as e:
                        await async_session.rollback()
                        logger.warning(f"流式写入数据库失败: {e}")
            except Exception as e:
                logger.warning(f"流式写入数据库时出错: {e}")

        # ========== END PERF优化 ==========

        for i, current_date in enumerate(trading_dates):
            # PERF/BUGFIX: 统一初始化计时变量，避免某些分支/异常路径引用未赋值导致 UnboundLocalError
            slice_time_total = 0.0
            gen_time_total = 0.0
            gen_time_max = 0.0

            # 任务状态检查已合并到进度更新中（每5%检查一次），无需单独检查
            try:
                # 获取当前价格（Phase3：使用向量化优化）
                _t_ml = time.perf_counter()
                current_prices: Dict[str, float] = {}

                if aligned_arrays is not None:
                    # Phase 3 优化：使用向量化价格查找
                    pass

                    codes = aligned_arrays.get("stock_codes")
                    code_to_i = aligned_arrays.get("code_to_i")
                    close_mat = aligned_arrays.get("close")
                    valid_mat = aligned_arrays.get("valid")

                    # 收集需要价格的股票。
                    # 对 ranking trade mode，必须包含全股票池，否则新候选股在持仓建立后
                    # 永远拿不到价格，无法进入 TopK 竞争。
                    need_codes = self._determine_price_lookup_codes(
                        strategy=strategy,
                        portfolio_manager=portfolio_manager,
                        aligned_arrays=aligned_arrays,
                        date_index=i,
                    )

                    if need_codes:
                        # 批量查找价格（向量化）
                        assert (
                            code_to_i is not None
                            and valid_mat is not None
                            and close_mat is not None
                        )
                        for c in need_codes:
                            j = code_to_i.get(c)
                            if j is not None and bool(valid_mat[j, i]):
                                current_prices[c] = float(close_mat[j, i])

                else:
                    # [优化 1] 避免 DataFrame 拷贝：使用 .values 和缓存的索引
                    for stock_code, data in stock_data.items():
                        try:
                            # 优先使用缓存的 date_to_idx 映射
                            date_to_idx = data.attrs.get("_date_to_idx")
                            if date_to_idx is not None and current_date in date_to_idx:
                                idx = date_to_idx[current_date]
                                # 使用 .values 直接访问底层数组
                                current_prices[stock_code] = float(
                                    data["close"].values[idx]
                                )
                            elif current_date in data.index:
                                # Fallback: 使用 iloc（比 loc 快）
                                idx = data.index.get_loc(current_date)
                                current_prices[stock_code] = float(
                                    data["close"].values[idx]
                                )
                        except Exception:
                            pass

                _ml_price_lookup += time.perf_counter() - _t_ml
                if not current_prices:
                    continue

                # 生成交易信号（Phase1：优先用 ndarray signal matrix）
                _t_ml = time.perf_counter()
                all_signals: List[TradingSignal] = []

                if aligned_arrays is not None:
                    sig_mat = cast(
                        Optional[np.ndarray[Any, Any]], aligned_arrays.get("signal")
                    )
                    codes = cast(Optional[List[str]], aligned_arrays.get("stock_codes"))
                    close_mat = cast(
                        Optional[np.ndarray[Any, Any]], aligned_arrays.get("close")
                    )
                    valid_mat = cast(
                        Optional[np.ndarray[Any, Any]], aligned_arrays.get("valid")
                    )
                    if (
                        isinstance(sig_mat, np.ndarray)
                        and isinstance(close_mat, np.ndarray)
                        and isinstance(valid_mat, np.ndarray)
                        and codes is not None
                    ):
                        sig_idx = np.nonzero(sig_mat[:, i])[0]
                        if sig_idx.size > 0:
                            for j in sig_idx.tolist():
                                if not bool(valid_mat[j, i]):
                                    continue
                                st = int(sig_mat[j, i])
                                if st == 1:
                                    stype = SignalType.BUY
                                elif st == -1:
                                    stype = SignalType.SELL
                                else:
                                    continue
                                code = codes[j]
                                price = float(close_mat[j, i])
                                all_signals.append(
                                    TradingSignal(
                                        timestamp=current_date,
                                        stock_code=code,
                                        signal_type=stype,
                                        strength=1.0,
                                        price=price,
                                        reason="[aligned] precomputed",
                                        metadata=None,
                                    )
                                )

                # 若对齐数组未生成信号，再走原有路径（兼容其它策略）
                if not all_signals:
                    # 生成交易信号（支持并行生成多股票信号）
                    all_signals = []

                # 性能监控：记录信号生成时间
                signal_start_time = (
                    time.perf_counter() if self.enable_performance_profiling else None
                )

                # ��分 profiling：把"切片"和"生成信号"拆开计时（变量已在循环开头初始化）

                # 辅助函数：快速查找预计算信号
                def get_precomputed_signal_fast(
                    stock_code: str, date: datetime
                ) -> Optional[List[TradingSignal]]:
                    """
                    [优化 1] 从预计算字典中快速查找信号，避免 DataFrame 拷贝

                    优化点：
                    1. 优先使用 aligned_arrays 的 numpy 数组（O(1) 查找）
                    2. 使用 .values 直接访问底层数组，避免创建 Series 对象
                    3. 缓存 date_to_idx 映射，避免重复 get_loc() 调用
                    """
                    if precomputed_signals:
                        signal = precomputed_signals.get((stock_code, date))
                        if signal is not None:
                            # 将信号类型转换为 TradingSignal 对象
                            from ..models import SignalType, TradingSignal

                            if isinstance(signal, SignalType):
                                # [优化 1] 获取当前价格 - 避免 DataFrame 拷贝
                                current_price = 0.0

                                try:
                                    # 方法 1: 优先使用 aligned_arrays（最快，O(1) 查找）
                                    if aligned_arrays is not None:
                                        code_to_i = aligned_arrays.get("code_to_i")
                                        close_mat = aligned_arrays.get("close")
                                        dates = aligned_arrays.get("dates")

                                        if (
                                            code_to_i is not None
                                            and close_mat is not None
                                            and dates is not None
                                        ):
                                            stock_idx = code_to_i.get(stock_code)
                                            if stock_idx is not None:
                                                # 找到日期索引
                                                date_idx = None
                                                date_np = np.datetime64(date)
                                                # 使用 numpy 的向量化查找
                                                matches = np.where(dates == date_np)[0]
                                                if len(matches) > 0:
                                                    date_idx = int(matches[0])
                                                    # 直接从 numpy 数组读取，无 pandas 开销
                                                    price_val = close_mat[
                                                        stock_idx, date_idx
                                                    ]
                                                    if not np.isnan(price_val):
                                                        current_price = float(price_val)

                                    # 方法 2: 如果 aligned_arrays 不可用，使用优化的 DataFrame 访问
                                    if current_price == 0.0:
                                        data = stock_data.get(stock_code)
                                        if data is not None:
                                            # 使用缓存的 date_to_idx 映射（避免重复 get_loc）
                                            date_to_idx = data.attrs.get("_date_to_idx")
                                            if (
                                                date_to_idx is not None
                                                and date in date_to_idx
                                            ):
                                                idx = date_to_idx[date]
                                                # 使用 .values 直接访问底层数组，避免创建 Series
                                                close_values = data["close"].values
                                                current_price = float(close_values[idx])
                                            elif date in data.index:
                                                # Fallback: 使用 iloc（比 loc 快，但仍会触发一些开销）
                                                idx = data.index.get_loc(date)
                                                current_price = float(
                                                    data["close"].values[idx]
                                                )

                                except Exception:
                                    # 静默失败，使用默认价格 0.0
                                    pass

                                return [
                                    TradingSignal(
                                        signal_type=signal,
                                        stock_code=stock_code,
                                        timestamp=date,
                                        price=current_price,
                                        strength=1.0,
                                        reason="Precomputed signal",
                                    )
                                ]
                            return [signal] if not isinstance(signal, list) else signal
                    return None

                # 只有在 aligned/precomputed 路径未拿到信号时，才走逐股票生成回退路径。
                # 否则会把同一天同一股票的信号重复加入 all_signals，导致 signal_records 计数膨胀。
                if not all_signals:
                    gen_time_max = 0.0
                    # 顺序生成信号（股票数量少或禁用并行）
                    for stock_code, data in stock_data.items():
                        if current_date in data.index:
                            # 获取到当前日期的历史数据
                            t0 = time.perf_counter()
                            # same rationale as parallel path: avoid daily slicing copies
                            idx_map: Optional[Dict[datetime, int]] = None
                            try:
                                idx_map = cast(
                                    Optional[Dict[datetime, int]],
                                    data.attrs.get("_date_to_idx"),
                                )
                            except Exception:
                                idx_map = None
                            current_idx = (
                                int(idx_map[current_date])
                                if idx_map is not None and current_date in idx_map
                                else (
                                    int(data.index.get_loc(current_date))
                                    if current_date in data.index
                                    else -1
                                )
                            )
                            # Provide fast-path hint for strategies (avoid repeated get_loc)
                            try:
                                data.attrs["_current_date"] = current_date
                                data.attrs["_current_idx"] = current_idx
                            except Exception:
                                pass
                            slice_time_total += time.perf_counter() - t0

                            if current_idx >= 20:
                                try:
                                    t1 = time.perf_counter()
                                    # 优先使用预计算信号
                                    signals = get_precomputed_signal_fast(
                                        stock_code, current_date
                                    )

                                    # 调试日志
                                    if current_idx == 20:  # 只在第一次打印
                                        logger.info(
                                            f"🔍 调试: stock={stock_code}, date={current_date}, precomputed_signals={'有' if signals else '无'}"
                                        )

                                    if signals is None:
                                        # Fallback: 调用策略生成
                                        signals = strategy.generate_signals(
                                            data, current_date
                                        )

                                    # 调试日志：记录信号内容
                                    if signals and current_idx == 20:
                                        logger.info(f"🔍 信号内容: {signals}")

                                    _dur = time.perf_counter() - t1
                                    gen_time_total += _dur
                                    if _dur > gen_time_max:
                                        gen_time_max = float(_dur)
                                    all_signals.extend(signals)
                                except Exception as e:
                                    logger.warning(f"生成信号失败 {stock_code}: {e}")
                                    continue

                # 记录信号生成时间
                if (
                    self.enable_performance_profiling
                    and signal_start_time
                    and self.performance_profiler
                ):
                    signal_duration = time.perf_counter() - signal_start_time
                    signal_generation_times.append(signal_duration)

                    # 原有口径：整段信号生成（含切片、计算指标、融合等）
                    self._profiler().record_function_call(
                        "generate_signals", signal_duration
                    )

                    # 新口径：拆开看"切片"与"策略信号生成"的比例
                    # 注意：并行模式下 slice_time_total / gen_time_total 是"各线程耗时求和"(work)，
                    # 不是 wall-clock；用于判断 CPU work 构成，但不能直接当成整体耗时百分比。
                    if slice_time_total > 0:
                        self._profiler().record_function_call(
                            "slice_historical_data_work", float(slice_time_total)
                        )
                    if gen_time_total > 0:
                        self._profiler().record_function_call(
                            "generate_signals_core_work", float(gen_time_total)
                        )

                    # 额外记录 wall-clock 口径（同 generate_signals，但名字更明确，便于报表阅读）
                    self._profiler().record_function_call(
                        "generate_signals_wall", signal_duration
                    )

                    # 并行路径下 critical path 近似：单日最慢股票的 generate_signals wall
                    if gen_time_max > 0:
                        self._profiler().record_function_call(
                            "generate_signals_core_wall_max", float(gen_time_max)
                        )

                        # 线程/调度开销（粗略）：整段 wall - 单日最慢单股 wall
                        overhead = float(signal_duration) - float(gen_time_max)
                        if overhead > 0:
                            self._profiler().record_function_call(
                                "signal_generation_overhead_wall", overhead
                            )

                    # If StrategyPortfolio attached per-strategy timings, record them once per day.
                    try:
                        perf_sig = None
                        for _s in all_signals:
                            md = getattr(_s, "metadata", None) or {}
                            if isinstance(md, dict) and "portfolio_perf" in md:
                                perf_sig = _s
                                break
                        if perf_sig is not None:
                            md = perf_sig.metadata or {}
                            pp = (
                                md.get("portfolio_perf")
                                if isinstance(md, dict)
                                else None
                            )
                            if isinstance(pp, dict):
                                sub = pp.get("sub_strategy_times")
                                if isinstance(sub, dict):
                                    for k, v in sub.items():
                                        self._profiler().record_function_call(
                                            f"portfolio_substrategy__{k}", float(v)
                                        )
                                it = pp.get("integrate_time")
                                if it is not None:
                                    self._profiler().record_function_call(
                                        "portfolio_integrate", float(it)
                                    )
                    except Exception:
                        pass

                _ml_signal_extract += time.perf_counter() - _t_ml
                _t_ml = time.perf_counter()
                # 执行交易信号（性能监控）
                trade_start_time = (
                    time.perf_counter() if self.enable_performance_profiling else None
                )
                trades_this_day = 0
                executed_trade_signals = []  # 记录已执行的交易对应的信号
                unexecuted_signals = []  # 记录未执行的信号及原因

                # ===== trade execution mode =====
                trade_mode = None
                trade_mode_config: Dict[str, Any] = {}
                try:
                    if strategy is not None:
                        trade_mode = strategy.get_trade_mode()
                        trade_mode_config.update(strategy.get_trade_mode_config() or {})
                except Exception:
                    trade_mode = None
                    trade_mode_config = {}

                try:
                    trade_mode_config.update(strategy_config or {})
                    trade_mode = (strategy_config or {}).get("trade_mode", trade_mode)
                except Exception:
                    pass

                mode_executor = get_trade_mode_executor(trade_mode)

                # --- debug aid: log which trade path is used (only when needed) ---
                try:
                    if current_date.strftime("%Y-%m-%d") in (
                        "2023-05-19",
                        "2023-05-22",
                        "2023-05-23",
                    ):
                        logger.info(
                            f"[trade_path] date={current_date.strftime('%Y-%m-%d')} trade_mode={trade_mode} "
                            f"signals={len(all_signals)} strategy_config_keys={list((trade_mode_config or {}).keys())}"
                        )
                except Exception:
                    pass

                if mode_executor is not None:
                    mode_result = mode_executor.execute(
                        TradeModeExecutionContext(
                            current_date=current_date,
                            all_signals=all_signals,
                            current_prices=current_prices,
                            portfolio_manager=portfolio_manager,
                            strategy=strategy,
                            strategy_config=trade_mode_config,
                            stock_universe=list(stock_data.keys()),
                        )
                    )
                    executed_trade_signals = mode_result.executed_trade_signals
                    unexecuted_signals = mode_result.unexecuted_signals
                    trades_this_day = mode_result.trades_this_day
                    if trade_mode_config.get("topk") is not None:
                        int(trade_mode_config["topk"])
                elif trade_mode == "topk_buffer":
                    # Daily TopK selection + buffer zone + max changes/day
                    k = int(trade_mode_config.get("topk", 10))
                    buffer_n = int(trade_mode_config.get("buffer", 20))
                    max_changes = int(trade_mode_config.get("max_changes_per_day", 2))
                    trades_limit = max_changes

                    # Build ranking scores from signals (BUY strength positive, SELL negative)
                    scores: Dict[str, float] = {code: 0.0 for code in stock_data.keys()}
                    for sig in all_signals:
                        s = float(sig.strength or 0.0)
                        if sig.signal_type == SignalType.BUY:
                            scores[sig.stock_code] = max(
                                scores.get(sig.stock_code, 0.0), s
                            )
                        elif sig.signal_type == SignalType.SELL:
                            scores[sig.stock_code] = min(
                                scores.get(sig.stock_code, 0.0), -s
                            )

                    # Rebalance according to TopK+buffer rules
                    (
                        executed_trade_signals,
                        unexecuted_signals,
                        trades_this_day,
                    ) = self._rebalance_topk_buffer(
                        portfolio_manager=portfolio_manager,
                        current_prices=current_prices,
                        current_date=current_date,
                        scores=scores,
                        topk=k,
                        buffer_n=buffer_n,
                        max_changes=trades_limit,
                        strategy=strategy,
                        debug=bool(trade_mode_config.get("debug_topk_buffer", False)),
                    )

                    # Debug: show what was executed on key dates / when trades happen
                    try:
                        if trades_this_day > 0 or current_date.strftime("%Y-%m-%d") in (
                            "2023-05-22",
                        ):
                            logger.info(
                                f"[trade_exec][topk_buffer] date={current_date.strftime('%Y-%m-%d')} trades_this_day={trades_this_day} "
                                f"executed={len(executed_trade_signals)} unexecuted={len(unexecuted_signals)} "
                                f"holdings_after={self._get_position_count_fast(portfolio_manager)}"
                            )
                    except Exception:
                        pass

                else:
                    for signal in all_signals:
                        # 验证信号
                        is_valid, validation_reason = self._validate_signal_fast(
                            strategy=strategy,
                            signal=signal,
                            portfolio_manager=portfolio_manager,
                            current_prices=current_prices,
                        )

                        if not is_valid:
                            # 验证失败，记录未执行原因
                            unexecuted_signals.append(
                                {
                                    "stock_code": signal.stock_code,
                                    "timestamp": signal.timestamp,
                                    "signal_type": signal.signal_type.name,
                                    "execution_reason": validation_reason
                                    or "信号验证失败",
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
                            self._profiler().record_function_call(
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
                                    "execution_reason": failure_reason
                                    or "执行失败（未知原因）",
                                }
                            )

                # 记录交易执行总时间
                if self.enable_performance_profiling and trade_start_time:
                    trade_duration = time.perf_counter() - trade_start_time
                    self._profiler().record_function_call(
                        "execute_trades_batch", trade_duration
                    )

                _ml_trade_exec += time.perf_counter() - _t_ml

                signals_for_recording = all_signals
                try:
                    if mode_executor is not None and getattr(
                        mode_result, "signal_records", None
                    ):
                        signals_for_recording = mode_result.signal_records
                except Exception:
                    pass
                total_signals += len(signals_for_recording)

                # PERF优化：收集信号记录到内存，循环结束后批量写入数据库
                _t_ml = time.perf_counter()
                if task_id and signals_for_recording:
                    try:
                        import uuid

                        if _current_backtest_id is None:
                            _current_backtest_id = backtest_id or (
                                f"bt_{task_id[:8]}"
                                if task_id
                                else f"bt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                            )

                        for signal in signals_for_recording:
                            signal_data = {
                                "signal_id": f"sig_{uuid.uuid4().hex[:12]}",
                                "stock_code": signal.stock_code,
                                "stock_name": None,
                                "signal_type": signal.signal_type.name,
                                "timestamp": signal.timestamp,
                                "price": signal.price,
                                "strength": signal.strength,
                                "reason": signal.reason,
                                "metadata": signal.metadata,
                                "executed": False,
                            }
                            _batch_signals_data.append(signal_data)
                    except Exception as e:
                        logger.warning(f"保存信号记录时出错: {e}")
                _ml_batch_collect += time.perf_counter() - _t_ml

                # PERF优化：收集未执行和已执行信号到内存，循环结束后批量写入
                if task_id and unexecuted_signals:
                    _batch_unexecuted_signals.extend(unexecuted_signals)

                if task_id and executed_trade_signals:
                    _batch_executed_signals.extend(executed_trade_signals)

                _t_ml = time.perf_counter()
                # PERF优化A：流式增量写入 - 每积累1000条记录就写入一次数据库
                if (
                    task_id
                    and (
                        len(_batch_signals_data)
                        + len(_batch_executed_signals)
                        + len(_batch_unexecuted_signals)
                    )
                    >= _BATCH_FLUSH_THRESHOLD
                ):
                    await _flush_batch_to_db(
                        signals_data=_batch_signals_data,
                        executed_signals=_batch_executed_signals,
                        unexecuted_signals=_batch_unexecuted_signals,
                        backtest_id=_current_backtest_id,
                    )
                    # 写入后清空列表
                    _batch_signals_data.clear()
                    _batch_executed_signals.clear()
                    _batch_unexecuted_signals.clear()
                _ml_batch_flush += time.perf_counter() - _t_ml

                _t_ml = time.perf_counter()
                # 记录组合快照
                portfolio_manager.record_portfolio_snapshot(
                    current_date, current_prices
                )
                _ml_portfolio_snap += time.perf_counter() - _t_ml

                # --- Sanity check (debug): topk_buffer must never exceed topk holdings ---
                # 这条只做告警，不改变交易行为，用于定位"持仓数为何会>topk"。
                try:
                    tm = None
                    k_limit = None
                    try:
                        tm = (strategy_config or {}).get("trade_mode")
                        k_limit = int((strategy_config or {}).get("topk", 10))
                    except Exception:
                        tm = None
                        k_limit = None

                    if tm == "topk_buffer" and k_limit is not None:
                        current_holdings = self._get_position_codes_fast(
                            portfolio_manager
                        )
                        if len(current_holdings) > int(k_limit):
                            logger.error(
                                f"[topk_buffer][sanity] positions_count={len(current_holdings)} > topk={k_limit} "
                                f"date={current_date.strftime('%Y-%m-%d')} holdings={sorted(current_holdings)}"
                            )
                except Exception as e:
                    logger.warning(f"[topk_buffer][sanity] check failed: {e}")

                # 更新进度监控（同时更新数据库）
                _t_ml = time.perf_counter()
                # 性能优化：进度更新从每5天改为每5%，减少DB写入次数（~14次 vs ~146次）
                # 同时合并任务状态检查，避免额外的DB读取
                _progress_pct = (i + 1) / len(trading_dates) * 100
                _should_update_progress = (
                    task_id
                    and (int(_progress_pct) % 5 == 0)
                    and (i == 0 or int(((i) / len(trading_dates)) * 100) % 5 != 0)
                )
                if _should_update_progress:
                    portfolio_value = portfolio_manager.get_portfolio_value(
                        current_prices
                    )
                    overall_progress = 30 + (_progress_pct / 100) * 60  # 30%到90%

                    try:
                        from app.core.database import SessionLocal
                        from app.models.task_models import TaskStatus
                        from app.repositories.task_repository import TaskRepository

                        session = SessionLocal()
                        try:
                            task_repo = TaskRepository(session)
                            existing_task = task_repo.get_task_by_id((task_id or ""))

                            # 合并任务状态检查（原来每50天单独检查）
                            if not existing_task:
                                raise TaskError(
                                    message=f"任务 {task_id} 已被删除，停止回测执行",
                                    severity=ErrorSeverity.LOW,
                                )
                            if not _is_task_running(existing_task.status):
                                raise TaskError(
                                    message=f"任务 {task_id} 状态为 {existing_task.status}，停止回测执行",
                                    severity=ErrorSeverity.LOW,
                                )

                            result_data: Dict[str, Any] = (
                                cast(Any, existing_task.result) or {}
                            )
                            progress_data = result_data.get("progress_data", {})
                            if not isinstance(progress_data, dict):
                                progress_data = {}

                            progress_data.update(
                                {
                                    "processed_days": i + 1,
                                    "total_days": len(trading_dates),
                                    "current_date": current_date.strftime("%Y-%m-%d"),
                                    "total_signals": total_signals,
                                    "total_trades": executed_trades,
                                    "portfolio_value": portfolio_value,
                                    "last_updated": datetime.utcnow().isoformat(),
                                }
                            )
                            result_data["progress_data"] = progress_data

                            task_repo.update_task_status(
                                task_id=(task_id or ""),
                                status=TaskStatus.RUNNING,
                                progress=overall_progress,
                                result=result_data,
                            )
                            session.commit()
                            logger.info(
                                f"进度更新: {_progress_pct:.0f}%, days={i + 1}/{len(trading_dates)}, "
                                f"signals={total_signals}, trades={executed_trades}"
                            )
                        except Exception as inner_error:
                            session.rollback()
                            if isinstance(inner_error, TaskError):
                                raise
                            logger.warning(f"更新进度失败: {inner_error}")
                        finally:
                            session.close()
                    except TaskError:
                        raise
                    except Exception as db_error:
                        logger.warning(f"进度更新DB错误: {db_error}")

                    await backtest_progress_monitor.update_execution_progress(
                        task_id=(task_id or ""),
                        processed_days=i + 1,
                        current_date=current_date.strftime("%Y-%m-%d"),
                        signals_generated=len(all_signals),
                        trades_executed=trades_this_day,
                        portfolio_value=portfolio_value,
                    )

                _ml_progress_update += time.perf_counter() - _t_ml
                # 定期输出进度日志
                if i % 50 == 0:
                    progress = (i + 1) / len(trading_dates) * 100
                    portfolio_value = portfolio_manager.get_portfolio_value(
                        current_prices
                    )
                    logger.debug(
                        f"回测进度: {progress:.1f}%, 组合价值: {portfolio_value:.2f}"
                    )

            except Exception as e:
                error_msg = f"回测循环错误，日期: {current_date}, 错误: {e}"
                logger.error(error_msg)

                # 添加警告到进度监控
                if task_id:
                    await backtest_progress_monitor.add_warning(task_id, error_msg)

                continue

        # ========== 写入 main_loop 细粒度计时 ==========
        if perf_breakdown is not None:
            perf_breakdown["mainloop_sub_price_lookup_s"] = _ml_price_lookup
            perf_breakdown["mainloop_sub_signal_extract_s"] = _ml_signal_extract
            perf_breakdown["mainloop_sub_trade_exec_s"] = _ml_trade_exec
            perf_breakdown["mainloop_sub_portfolio_snap_s"] = _ml_portfolio_snap
            perf_breakdown["mainloop_sub_batch_collect_s"] = _ml_batch_collect
            perf_breakdown["mainloop_sub_batch_flush_s"] = _ml_batch_flush
            perf_breakdown["mainloop_sub_progress_update_s"] = _ml_progress_update
            _ml_total = (
                _ml_price_lookup
                + _ml_signal_extract
                + _ml_trade_exec
                + _ml_portfolio_snap
                + _ml_batch_collect
                + _ml_batch_flush
                + _ml_progress_update
            )
            perf_breakdown["mainloop_sub_accounted_s"] = _ml_total
            logger.info(
                f"📊 main_loop 细粒度: price={_ml_price_lookup:.1f}s, "
                f"signal={_ml_signal_extract:.1f}s, trade={_ml_trade_exec:.1f}s, "
                f"snap={_ml_portfolio_snap:.1f}s, batch_collect={_ml_batch_collect:.1f}s, "
                f"flush={_ml_batch_flush:.1f}s, progress={_ml_progress_update:.1f}s, "
                f"accounted={_ml_total:.1f}s"
            )

        # ========== PERF优化：循环结束后写入剩余数据 ==========
        # 写入流式写入未处理完的剩余数据
        if (
            task_id
            and (
                len(_batch_signals_data)
                + len(_batch_executed_signals)
                + len(_batch_unexecuted_signals)
            )
            > 0
        ):
            logger.info(
                f"🔄 写入剩余数据: 信号={len(_batch_signals_data)}, 已执行={len(_batch_executed_signals)}, 未执行={len(_batch_unexecuted_signals)}"
            )
            await _flush_batch_to_db(
                signals_data=_batch_signals_data,
                executed_signals=_batch_executed_signals,
                unexecuted_signals=_batch_unexecuted_signals,
                backtest_id=_current_backtest_id,
            )
        # ========== END PERF优化 ==========

        # 最终进度更新
        if task_id_str:
            final_portfolio_value = portfolio_manager.get_portfolio_value({})
            await backtest_progress_monitor.update_execution_progress(
                task_id=task_id_str,
                processed_days=len(trading_dates),
                current_date=(
                    trading_dates[-1].strftime("%Y-%m-%d") if trading_dates else ""
                ),
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
                self._profiler().end_stage(
                    "backtest_execution",
                    {
                        "avg_signal_generation_time": avg_signal_time,
                        "total_signal_generation_calls": len(signal_generation_times),
                    },
                )
            if trade_execution_times:
                avg_trade_time = sum(trade_execution_times) / len(trade_execution_times)
                self._profiler().end_stage(
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

    def _rebalance_topk_buffer(
        self,
        portfolio_manager: PortfolioManager,
        current_prices: Dict[str, float],
        current_date: datetime,
        scores: Dict[str, float],
        topk: int = 10,
        buffer_n: int = 20,
        max_changes: int = 2,
        strategy: Optional[BaseStrategy] = None,
        debug: bool = False,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
        """每日 TopK 选股 + buffer 换仓 + 每天最多换 max_changes 只。

        规则（实盘对齐版）：
        - 目标持仓数量=topk
        - 若持仓仍在 Top(topk+buffer_n) 内，则尽量保留（减少换手）
        - 每天最多做 max_changes 个 "卖出+买入" 的替换

        Returns:
            executed_trade_signals, unexecuted_signals, trades_this_day
        """
        executed_trade_signals: List[Dict[str, Any]] = []
        unexecuted_signals: List[Dict[str, Any]] = []
        trades_this_day = 0

        if topk <= 0:
            return executed_trade_signals, unexecuted_signals, trades_this_day

        # rank by score desc, tie-break by stock_code for determinism
        ranked = sorted(scores.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
        topk_list = [c for c, _ in ranked[:topk]]
        buffer_list = [c for c, _ in ranked[: max(topk, topk + buffer_n)]]
        buffer_set = set(buffer_list)

        holdings = self._get_position_codes_fast(portfolio_manager)
        set(holdings)

        # Keep holdings inside buffer zone
        kept = [c for c in holdings if c in buffer_set]

        # If kept > topk, trim lowest-ranked among kept
        rank_index = {c: i for i, (c, _) in enumerate(ranked)}
        if len(kept) > topk:
            kept_sorted = sorted(kept, key=lambda c: rank_index.get(c, 10**9))
            kept = kept_sorted[:topk]

        kept_set = set(kept)

        # Sell candidates: holdings outside buffer OR trimmed
        to_sell = [c for c in holdings if c not in kept_set]

        # Buy candidates: topk names not already kept
        to_buy = [c for c in topk_list if c not in kept_set]

        # Decide actions under max_changes
        # - If current holdings < topk: allow buys even without sells (build initial positions)
        # - Otherwise: do replacement pairs (sell+buy) up to max_changes
        current_n = len(holdings)
        if current_n < topk:
            # how many new names to buy today
            buy_quota = min(max_changes, topk - current_n, len(to_buy))
            to_sell = []
            to_buy = to_buy[:buy_quota]
        else:
            # replacement pairs
            n_pairs = min(max_changes, len(to_sell), len(to_buy))
            to_sell = to_sell[:n_pairs]
            to_buy = to_buy[:n_pairs]

        if debug:
            try:
                nonzero = sum(
                    1
                    for _, s in scores.items()
                    if isinstance(s, (int, float)) and s != 0
                )
                logger.info(
                    f"[topk_buffer] {current_date.date()} holdings={len(holdings)} nonzero_scores={nonzero} "
                    f"topk={topk} buffer={buffer_n} max_changes={max_changes} "
                    f"to_sell={len(to_sell)} to_buy={len(to_buy)}"
                )
                logger.info(
                    f"[topk_buffer] topk_list(head)={topk_list[:min(5, len(topk_list))]} "
                    f"holdings(head)={holdings[:min(5, len(holdings))]}"
                )
            except Exception:
                pass

        # Execute sells first
        successful_sells = 0
        for code in to_sell:
            sig = TradingSignal(
                timestamp=current_date,
                stock_code=code,
                signal_type=SignalType.SELL,
                strength=1.0,
                price=float(current_prices.get(code, 0.0) or 0.0),
                reason="topk_buffer rebalance sell (out of buffer/topk)",
                metadata={"trade_mode": "topk_buffer"},
            )
            if strategy is not None:
                is_valid, validation_reason = self._validate_signal_fast(
                    strategy=strategy,
                    signal=sig,
                    portfolio_manager=portfolio_manager,
                    current_prices=current_prices,
                )
                if not is_valid:
                    unexecuted_signals.append(
                        {
                            "stock_code": code,
                            "timestamp": current_date,
                            "signal_type": sig.signal_type.name,
                            "execution_reason": validation_reason or "信号验证失败",
                        }
                    )
                    continue

            trade, failure_reason = portfolio_manager.execute_signal(
                sig, current_prices
            )
            if trade:
                successful_sells += 1
                trades_this_day += 1
                executed_trade_signals.append(
                    {
                        "stock_code": code,
                        "timestamp": current_date,
                        "signal_type": sig.signal_type.name,
                    }
                )
            else:
                unexecuted_signals.append(
                    {
                        "stock_code": code,
                        "timestamp": current_date,
                        "signal_type": sig.signal_type.name,
                        "execution_reason": failure_reason or "执行失败（未知原因）",
                    }
                )

        # Execute buys
        # Guardrails:
        # 1) replacement 模式下：只允许用「成功卖出」换入，避免卖失败仍买导致持仓膨胀
        # 2) 任何情况下都不允许持仓数超过 topk
        current_positions_n = self._get_position_count_fast(portfolio_manager)
        remaining_capacity = max(0, topk - current_positions_n)

        if current_n >= topk:
            # replacement mode: buys must be backed by successful sells
            buy_quota = min(len(to_buy), successful_sells, remaining_capacity)
        else:
            # build mode: still respect capacity
            buy_quota = min(len(to_buy), remaining_capacity)

        to_buy = to_buy[:buy_quota]

        for code in to_buy:
            # Hard cap: never allow positions to exceed topk (even if earlier logic misbehaves)
            if self._get_position_count_fast(portfolio_manager) >= topk:
                unexecuted_signals.append(
                    {
                        "stock_code": code,
                        "timestamp": current_date,
                        "signal_type": SignalType.BUY.name,
                        "execution_reason": f"超过topk持仓上限(topk={topk})，跳过买入",
                    }
                )
                break

            sig = TradingSignal(
                timestamp=current_date,
                stock_code=code,
                signal_type=SignalType.BUY,
                strength=1.0,
                price=float(current_prices.get(code, 0.0) or 0.0),
                reason=f"topk_buffer rebalance buy (enter top{topk})",
                metadata={"trade_mode": "topk_buffer"},
            )
            if strategy is not None:
                is_valid, validation_reason = self._validate_signal_fast(
                    strategy=strategy,
                    signal=sig,
                    portfolio_manager=portfolio_manager,
                    current_prices=current_prices,
                )
                if not is_valid:
                    unexecuted_signals.append(
                        {
                            "stock_code": code,
                            "timestamp": current_date,
                            "signal_type": sig.signal_type.name,
                            "execution_reason": validation_reason or "信号验证失败",
                        }
                    )
                    continue

            trade, failure_reason = portfolio_manager.execute_signal(
                sig, current_prices
            )
            if trade:
                trades_this_day += 1
                executed_trade_signals.append(
                    {
                        "stock_code": code,
                        "timestamp": current_date,
                        "signal_type": sig.signal_type.name,
                    }
                )
            else:
                unexecuted_signals.append(
                    {
                        "stock_code": code,
                        "timestamp": current_date,
                        "signal_type": sig.signal_type.name,
                        "execution_reason": failure_reason or "执行失败（未知原因）",
                    }
                )

        return executed_trade_signals, unexecuted_signals, trades_this_day

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
            # pandas>=3.0: 'M' deprecated, use month-end 'ME'
            monthly_values = portfolio_values.resample("ME").last()
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

                # 辅助函数：统一访问 trade 属性（支持 Trade 对象和字典）
                def get_trade_attr(trade: Any, attr: str) -> Any:
                    if isinstance(trade, dict):
                        return trade.get(attr)
                    return getattr(trade, attr, None)

                for trade in portfolio_manager.trades:
                    stock_code = get_trade_attr(trade, "stock_code")
                    action = get_trade_attr(trade, "action")
                    pnl = get_trade_attr(trade, "pnl") or 0.0

                    stock_stats = stock_performance.setdefault(
                        stock_code,
                        {
                            "stock_code": stock_code,
                            "total_pnl": 0.0,
                            "trade_count": 0,
                        },
                    )
                    stock_stats["trade_count"] += 1
                    # 只有卖出交易才有实现盈亏
                    if action == "SELL":
                        stock_stats["total_pnl"] += float(pnl)

                # 计算每只股票的平均单笔盈亏
                for stats in stock_performance.values():
                    trades = max(stats["trade_count"], 1)
                    stats["avg_pnl_per_trade"] = float(stats["total_pnl"]) / trades

                # 个股表现汇总
                stock_perf_list = list(stock_performance.values())
                additional_metrics.update(
                    {
                        "stock_performance_detail": stock_perf_list,
                        "best_performing_stock": (
                            max(stock_perf_list, key=lambda x: x["total_pnl"])
                            if stock_perf_list
                            else None
                        ),
                        "worst_performing_stock": (
                            min(stock_perf_list, key=lambda x: x["total_pnl"])
                            if stock_perf_list
                            else None
                        ),
                        "stocks_traded": len(stock_perf_list),
                    }
                )

                # 单笔交易分布的整体特征（便于前端画直方图/统计）
                pnls = [
                    float(get_trade_attr(t, "pnl") or 0.0)
                    for t in portfolio_manager.trades
                ]
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
                raise TaskError(
                    message="股票代码列表不能为空", severity=ErrorSeverity.MEDIUM
                )

            if len(stock_codes) > 1000:
                raise TaskError(
                    message=f"股票数量过多: {len(stock_codes)}，最多支持1000只股票",
                    severity=ErrorSeverity.MEDIUM,
                )

            # 验证日期范围
            if start_date >= end_date:
                raise TaskError(
                    message="开始日期必须早于结束日期", severity=ErrorSeverity.MEDIUM
                )

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
                raise TaskError(
                    message="策略配置必须是字典格式", severity=ErrorSeverity.MEDIUM
                )

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

                cash_reserve_ratio = float(
                    getattr(portfolio_manager.config, "cash_reserve_ratio", 0.05) or 0.0
                )
                cash_reserve_ratio = min(max(cash_reserve_ratio, 0.0), 0.99)
                reserve_cash = portfolio_manager.cash * (1 - cash_reserve_ratio)
                reserve_pct = "{cash_reserve_ratio:.0%}"
                board_lot_size = max(
                    1,
                    int(
                        getattr(portfolio_manager.config, "board_lot_size", 100) or 100
                    ),
                )

                available_cash_for_stock = max_position_value - current_position_value
                available_cash_for_stock = min(available_cash_for_stock, reserve_cash)

                if available_cash_for_stock <= 0:
                    if (
                        current_position_value > 0
                        and current_position_value >= max_position_value
                    ):
                        return f"已达到最大持仓限制: 当前持仓 {current_position_value:.2f} >= 最大持仓 {max_position_value:.2f}"
                    else:
                        return f"可用资金不足: 需要保留{reserve_pct}现金，可用资金 {portfolio_manager.cash:.2f}"

                # 计算购买数量（按配置的最小交易单位取整）
                quantity = (
                    int(available_cash_for_stock / current_price / board_lot_size)
                    * board_lot_size
                )
                if quantity <= 0:
                    return f"可买数量不足: 可用资金 {available_cash_for_stock:.2f}，价格 {current_price:.2f}，无法买入{board_lot_size}股"

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
