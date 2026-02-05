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
from ..core.portfolio_manager_array import PortfolioManagerArray
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


def _multiprocess_precompute_worker(task: Tuple) -> Tuple[bool, str, Optional[Dict], Optional[str]]:
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
        df = pd.DataFrame(data_dict['values'], columns=data_dict['columns'])
        df.index = pd.to_datetime(data_dict['index'])
        df.attrs['stock_code'] = data_dict['stock_code']

        # 重建策略对象
        from ..strategies.strategy_factory import StrategyFactory, AdvancedStrategyFactory

        strategy_name = strategy_info['name']  # 使用策略名称（如 "MACD"）
        strategy_class_name = strategy_info['class_name']  # 类名（如 "MACDStrategy"）
        strategy_config = strategy_info['config']

        # 尝试从工厂创建策略（尝试多种名称格式）
        strategy = None
        names_to_try = [
            strategy_name,  # 原始名称
            strategy_name.lower(),  # 小写
            strategy_class_name,  # 类名
            strategy_class_name.replace('Strategy', ''),  # 去掉 Strategy 后缀
            strategy_class_name.replace('Strategy', '').lower(),  # 去掉后缀并小写
        ]

        for name in names_to_try:
            if strategy is not None:
                break
            try:
                strategy = StrategyFactory.create_strategy(name, strategy_config)
            except Exception:
                try:
                    strategy = AdvancedStrategyFactory.create_strategy(name, strategy_config)
                except Exception:
                    pass

        if strategy is None:
            return (False, stock_code, None, f"无法创建策略 {strategy_name} (尝试了: {names_to_try})")

        # 执行向量化预计算
        signals = strategy.precompute_all_signals(df)

        if signals is not None:
            # 将 Series 转换为可序列化格式
            signals_dict = {
                'values': signals.tolist(),
                'index': [str(idx) for idx in signals.index],
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
        data_dir: str = "backend/data",
        enable_parallel: bool = True,
        max_workers: Optional[int] = None,
        enable_performance_profiling: bool = False,
        use_multiprocessing: bool = False,
    ):
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
        """
        import os

        if max_workers is None:
            max_workers = min(os.cpu_count() or 4, 8)  # 最多8个线程，避免过多线程导致开销

        self.enable_parallel = enable_parallel
        self.max_workers = max_workers
        self.use_multiprocessing = use_multiprocessing
        self.use_array_portfolio = True  # Phase 1: 启用数组化持仓管理
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
            mode = "多进程" if use_multiprocessing else "多线程"
            logger.info(f"回测执行器已启用并行化（{mode}），最大工作进程/线程数: {max_workers}")

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
        # 轻量分段计时（始终可用，不依赖 performance_profiler）
        perf_breakdown: Dict[str, float] = {}
        _t_total0 = time.perf_counter()

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
            _t0 = time.perf_counter()
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
            perf_breakdown["strategy_setup_s"] = time.perf_counter() - _t0

            if task_id:
                await backtest_progress_monitor.update_stage(
                    task_id, "strategy_setup", progress=100, status="completed"
                )

            # 创建组合管理器
            # Phase 1: 数据加载后再创建（需要 stock_codes）
            portfolio_manager = None

            # 加载数据（性能监控）
            _t0 = time.perf_counter()
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
            perf_breakdown["data_loading_s"] = time.perf_counter() - _t0

            if task_id:
                await backtest_progress_monitor.update_stage(
                    task_id, "data_loading", progress=100, status="completed"
                )

            # Phase 1: 数据加载后创建组合管理器（使用实际加载的股票列表）
            actual_stock_codes = list(stock_data.keys())
            if self.use_array_portfolio:
                portfolio_manager = PortfolioManagerArray(backtest_config, actual_stock_codes)
                logger.info(f"✅ Phase 1: 使用数组化持仓管理器 (stocks={len(actual_stock_codes)})")
            else:
                portfolio_manager = PortfolioManager(backtest_config)
                logger.info(f"使用传统持仓管理器 (stocks={len(actual_stock_codes)})")

            # 获取交易日历
            trading_dates = self._get_trading_calendar(stock_data, start_date, end_date)

            # 预处理（日期索引 + 预计算信号 + 信号提取）
            _t0 = time.perf_counter()

            # ✅ 日期预索引：为每只股票建立 date->idx 映射，回测循环里用 O(1) 查找替代 get_loc
            # 经验上这是纯收益（相比指标预热，不会把计算串行化）。
            self._build_date_index(stock_data)

            # ✅ 信号向量化预计算：在进入每日循环前，先尝试一次性算出全量买卖点
            self._precompute_strategy_signals(strategy, stock_data)
            
            # ✅ 信号提取优化：将预计算信号提取到扁平字典，避免回测循环中重复查找 attrs
            precomputed_signals = self._extract_precomputed_signals_to_dict(strategy, stock_data)
            
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

            _t0 = time.perf_counter()
            # Phase1 预备：将 close/valid/signal 对齐成 ndarray，减少主循环 DataFrame/dict 访问
            _t1 = time.perf_counter()
            aligned_arrays = self._build_aligned_arrays(strategy, stock_data, trading_dates)
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
            )
            perf_breakdown["main_loop_s"] = time.perf_counter() - _t0

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

            _t0 = time.perf_counter()
            performance_metrics = portfolio_manager.get_performance_metrics()
            perf_breakdown["metrics_s"] = time.perf_counter() - _t0

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

            _t0 = time.perf_counter()
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
            perf_breakdown["report_generation_s"] = time.perf_counter() - _t0
            # 将回测循环统计（信号数、交易日等）写入报告，便于排查"无信号记录"等问题
            backtest_report["total_signals"] = backtest_results.get("total_signals", 0)
            backtest_report["trading_days"] = backtest_results.get("trading_days", 0)

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

            # 轻量分段计时结果写入报告（bench脚本唯一入口依赖此字段）
            perf_breakdown["total_wall_s"] = time.perf_counter() - _t_total0
            backtest_report["perf_breakdown"] = perf_breakdown

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

    def _precompute_strategy_signals(
        self,
        strategy: BaseStrategy,
        stock_data: Dict[str, pd.DataFrame],
    ) -> None:
        """[性能优化] 在回测循环开始前，尝试对所有股票进行向量化信号预计算。"""
        try:
            from ..core.strategy_portfolio import StrategyPortfolio

            if isinstance(strategy, StrategyPortfolio):
                logger.info(f"🚀 Portfolio策略检测到，递归预计算 {len(strategy.strategies)} 个子策略")
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
        use_multiprocessing = getattr(self, 'use_multiprocessing', False)

        def _work_one(item):
            stock_code, data = item
            try:
                all_sigs = strategy.precompute_all_signals(data)
                if all_sigs is not None:
                    cache = data.attrs.setdefault("_precomputed_signals", {})
                    # 使用 strategy.name 作为稳定的 key，避免多进程环境下 id() 变化
                    cache[strategy.name] = all_sigs
                    return True, stock_code, None
                return False, stock_code, None
            except Exception as e:
                return False, stock_code, str(e)

        if self.enable_parallel and total_stocks >= 4:
            if use_multiprocessing:
                # 多进程模式：突破 GIL 限制，适合 CPU 密集型策略计算
                # 注意：需要将数据序列化传递，开销较大但可真正并行
                try:
                    from concurrent.futures import ProcessPoolExecutor as PoolExecutor
                    # 多进程需要使用模块级函数，这里使用包装器
                    results = self._precompute_signals_multiprocess(
                        strategy, stock_data
                    )
                    for ok, stock_code, err in results:
                        if ok:
                            success_count += 1
                        elif err:
                            logger.warning(
                                f"策略 {strategy.name} 对股票 {stock_code} 预计算信号失败: {err}"
                            )
                except Exception as e:
                    logger.warning(f"多进程预计算失败，回退到多线程: {e}")
                    use_multiprocessing = False

            if not use_multiprocessing:
                # 多线程模式：受 GIL 限制，但序列化开销小
                with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
                    futures = [ex.submit(_work_one, it) for it in stock_data.items()]
                    for fu in as_completed(futures):
                        ok, stock_code, err = fu.result()
                        if ok:
                            success_count += 1
                        elif err:
                            logger.warning(
                                f"策略 {strategy.name} 对股票 {stock_code} 预计算信号失败: {err}"
                            )
        else:
            for it in stock_data.items():
                ok, stock_code, err = _work_one(it)
                if ok:
                    success_count += 1
                elif err:
                    logger.warning(
                        f"策略 {strategy.name} 对股票 {stock_code} 预计算信号失败: {err}"
                    )

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
                logger.info(f"🔄 Portfolio策略信号整合开始: {len(strategy.strategies)} 个子策略")
                
                # 1. 递归提取所有子策略的信号
                all_sub_signals: Dict[Tuple[str, datetime], Any] = {}
                for sub in strategy.strategies:
                    sub_signals = self._extract_precomputed_signals_to_dict(sub, stock_data)
                    all_sub_signals.update(sub_signals)
                
                logger.info(f"📊 子策略信号总数: {len(all_sub_signals)}")
                
                # 2. 按日期分组子策略信号
                from collections import defaultdict
                signals_by_date: Dict[datetime, List[TradingSignal]] = defaultdict(list)
                
                for (stock_code, date), signal_type in all_sub_signals.items():
                    # 构造 TradingSignal 对象
                    from ..models import SignalType
                    if signal_type == SignalType.BUY or signal_type == SignalType.SELL:
                        # 获取价格
                        try:
                            df = stock_data.get(stock_code)
                            if df is not None and date in df.index:
                                price = float(df.loc[date, 'close'])
                                signal = TradingSignal(
                                    timestamp=date,
                                    stock_code=stock_code,
                                    signal_type=signal_type,
                                    strength=1.0,
                                    price=price,
                                    reason="precomputed",
                                    metadata={}
                                )
                                signals_by_date[date].append(signal)
                        except Exception as e:
                            logger.warning(f"构造信号失败 {stock_code} @ {date}: {e}")
                
                # 3. 对每个日期的信号进行整合
                integrated_count = 0
                for date, signals in signals_by_date.items():
                    if signals:
                        # 调用 Portfolio 的信号整合器
                        integrated = strategy.integrator.integrate(
                            signals, 
                            strategy.weights,
                            consistency_threshold=0.6
                        )
                        
                        # 将整合后的信号添加到字典
                        for sig in integrated:
                            signal_dict[(sig.stock_code, sig.timestamp)] = sig.signal_type
                            integrated_count += 1
                
                logger.info(f"✅ Portfolio策略信号整合完成: {integrated_count} 个整合信号")
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
        stock_codes = list(stock_data.keys())
        T = len(trading_dates)
        N = len(stock_codes)

        dates64 = np.array(trading_dates, dtype='datetime64[ns]')

        # 预分配数组（Phase 3 优化：使用连续内存）
        close = np.full((N, T), np.nan, dtype=np.float64, order='C')
        open_ = np.full((N, T), np.nan, dtype=np.float64, order='C')
        valid = np.zeros((N, T), dtype=bool, order='C')
        signal = np.zeros((N, T), dtype=np.int8, order='C')

        # 如果已做向量化预计算��号，尽量直接读取 per-stock Series 并对齐到 trading_dates
        strategy_key = strategy.name  # 使用 strategy.name 作为稳定的 key

        for i, code in enumerate(stock_codes):
            df = stock_data[code]

            # Phase 3 优化：使用 reindex 批量对齐（比逐个查找快）
            try:
                # 价格对齐（使用 reindex 一次性完成）
                s_close = df['close'].reindex(trading_dates)
                close_values = s_close.values  # 直接获取 numpy 数组
                close[i, :] = close_values
                
                if 'open' in df.columns:
                    s_open = df['open'].reindex(trading_dates)
                    open_[i, :] = s_open.values
                
                # 使用向量化操作判断有效性
                valid[i, :] = ~np.isnan(close_values)
                
            except Exception as e:
                # fallback: per-date fill (slow path, should be rare)
                logger.warning(f"股票 {code} 数组对齐失败，使用慢速路径: {e}")
                idx_map = df.attrs.get('_date_to_idx') if hasattr(df, 'attrs') else None
                for t, d in enumerate(trading_dates):
                    try:
                        if idx_map and d in idx_map:
                            k = int(idx_map[d])
                            close[i, t] = float(df['close'].iloc[k])
                            if 'open' in df.columns:
                                open_[i, t] = float(df['open'].iloc[k])
                            valid[i, t] = True
                        elif d in df.index:
                            k = df.index.get_loc(d)
                            close[i, t] = float(df['close'].values[k])
                            if 'open' in df.columns:
                                open_[i, t] = float(df['open'].values[k])
                            valid[i, t] = True
                    except Exception:
                        pass

            # 信号对齐（Phase 3 优化：使用 reindex 批量对齐）
            try:
                pre = df.attrs.get('_precomputed_signals', {}) if hasattr(df, 'attrs') else {}
                sig_ser = pre.get(strategy_key)
                if isinstance(sig_ser, pd.Series):
                    # 使用 reindex 批量对齐
                    s = sig_ser.reindex(trading_dates)
                    vals = s.values  # 直接获取 numpy 数组
                    # 向量化映射 SignalType to int8
                    for t, v in enumerate(vals):
                        if v == SignalType.BUY:
                            signal[i, t] = 1
                        elif v == SignalType.SELL:
                            signal[i, t] = -1
                elif isinstance(sig_ser, dict):
                    # dict 路径：逐个填充
                    for t, d in enumerate(trading_dates):
                        v = sig_ser.get(d)
                        if v == SignalType.BUY:
                            signal[i, t] = 1
                        elif v == SignalType.SELL:
                            signal[i, t] = -1
            except Exception as e:
                logger.warning(f"股票 {code} 信号对齐失败: {e}")

        return {
            'stock_codes': stock_codes,
            'code_to_i': {c: idx for idx, c in enumerate(stock_codes)},
            'dates': dates64,
            'close': close,
            'open': open_,
            'valid': valid,
            'signal': signal,
        }


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
        import pickle

        results = []

        # 准备可序列化的任务数据
        tasks = []
        for stock_code, data in stock_data.items():
            try:
                # 序列化策略配置（而非策略对象本身）
                strategy_info = {
                    'name': strategy.name,
                    'class_name': strategy.__class__.__name__,
                    'config': getattr(strategy, 'config', {}),
                }
                # 将 DataFrame 转换为字典格式（可序列化）
                data_dict = {
                    'values': data.to_dict('list'),
                    'index': list(data.index),
                    'columns': list(data.columns),
                    'stock_code': data.attrs.get('stock_code', stock_code),
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
                    executor.submit(
                        _multiprocess_precompute_worker, task
                    ): task[0] for task in tasks
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
                                signals_dict['values'],
                                index=pd.to_datetime(signals_dict['index']),
                                dtype=object
                            )
                            cache = original_data.attrs.setdefault("_precomputed_signals", {})
                            cache[strategy.name] = signals  # 使用 strategy.name 作为稳定的 key
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
        task_id: str = None,
        backtest_id: str = None,
        precomputed_signals: Optional[Dict[Tuple[str, datetime], Any]] = None,
        aligned_arrays: Optional[Dict[str, Any]] = None,
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

        # ========== PERF优化：批量收集数据库操作，循环结束后一次性写入 ==========
        # 避免在730天循环内每天都做数据库操作（原来是72秒的主要瓶颈）
        _batch_signals_data: List[dict] = []  # 收集所有信号记录
        _batch_executed_signals: List[dict] = []  # 收集已执行的信号
        _batch_unexecuted_signals: List[dict] = []  # 收集未执行的信号
        _current_backtest_id: str | None = None  # 缓存 backtest_id
        # ========== END PERF优化 ==========

        for i, current_date in enumerate(trading_dates):
            # PERF/BUGFIX: 统一初始化计时变量，避免某些分支/异常路径引用未赋值导致 UnboundLocalError
            slice_time_total = 0.0
            gen_time_total = 0.0
            gen_time_max = 0.0

            # 在循环开始时检查任务状态（每50个交易日检查一次，避免频繁检查）
            if task_id and i % 50 == 0 and i > 0:
                if not check_task_status():
                    logger.info(f"任务状态检查失败，停止回测执行: {task_id}")
                    raise TaskError(
                        message=f"任务 {task_id} 已被删除或状态已改变，停止回测执行",
                        severity=ErrorSeverity.LOW,
                    )
            try:
                # 获取当前价格（Phase3：使用向量化优化）
                current_prices: Dict[str, float] = {}

                if aligned_arrays is not None:
                    # Phase 3 优化：使用向量化价格查找
                    from .vectorized_loop import vectorized_price_lookup, get_portfolio_stocks
                    
                    codes = aligned_arrays.get("stock_codes")
                    code_to_i = aligned_arrays.get("code_to_i")
                    close_mat = aligned_arrays.get("close")
                    valid_mat = aligned_arrays.get("valid")
                    sig_mat = aligned_arrays.get("signal")

                    # 收集需要价格的股票（持仓 + 有信号的股票）
                    need_codes = set(get_portfolio_stocks(portfolio_manager))
                    
                    if isinstance(sig_mat, np.ndarray):
                        sig_idx = np.nonzero(sig_mat[:, i])[0]
                        for j in sig_idx.tolist():
                            need_codes.add(codes[j])

                    if need_codes:
                        # 批量查找价格（向量化）
                        for c in need_codes:
                            j = code_to_i.get(c) if isinstance(code_to_i, dict) else None
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
                                current_prices[stock_code] = float(data['close'].values[idx])
                            elif current_date in data.index:
                                # Fallback: 使用 iloc（比 loc 快）
                                idx = data.index.get_loc(current_date)
                                current_prices[stock_code] = float(data['close'].values[idx])
                        except Exception:
                            pass

                if not current_prices:
                    continue

                # 生成交易信号（Phase1：优先用 ndarray signal matrix）
                all_signals: List[TradingSignal] = []

                if aligned_arrays is not None:
                    sig_mat = aligned_arrays.get("signal")
                    codes = aligned_arrays.get("stock_codes")
                    close_mat = aligned_arrays.get("close")
                    valid_mat = aligned_arrays.get("valid")
                    if isinstance(sig_mat, np.ndarray):
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
                def get_precomputed_signal_fast(stock_code: str, date: datetime):
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
                            from ..models import TradingSignal, SignalType
                            if isinstance(signal, SignalType):
                                # [优化 1] 获取当前价格 - 避免 DataFrame 拷贝
                                current_price = 0.0
                                
                                try:
                                    # 方法 1: 优先使用 aligned_arrays（最快，O(1) 查找）
                                    if aligned_arrays is not None:
                                        code_to_i = aligned_arrays.get("code_to_i")
                                        close_mat = aligned_arrays.get("close")
                                        dates = aligned_arrays.get("dates")
                                        
                                        if code_to_i is not None and close_mat is not None and dates is not None:
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
                                                    price_val = close_mat[stock_idx, date_idx]
                                                    if not np.isnan(price_val):
                                                        current_price = float(price_val)
                                    
                                    # 方法 2: 如果 aligned_arrays 不可用，使用优化的 DataFrame 访问
                                    if current_price == 0.0:
                                        data = stock_data.get(stock_code)
                                        if data is not None:
                                            # 使用缓存的 date_to_idx 映射（避免重复 get_loc）
                                            date_to_idx = data.attrs.get("_date_to_idx")
                                            if date_to_idx is not None and date in date_to_idx:
                                                idx = date_to_idx[date]
                                                # 使用 .values 直接访问底层数组，避免创建 Series
                                                close_values = data['close'].values
                                                current_price = float(close_values[idx])
                                            elif date in data.index:
                                                # Fallback: 使用 iloc（比 loc 快，但仍会触发一些开销）
                                                idx = data.index.get_loc(date)
                                                current_price = float(data['close'].values[idx])
                                
                                except Exception as e:
                                    # 静默失败，使用默认价格 0.0
                                    pass
                                
                                return [TradingSignal(
                                    signal_type=signal,
                                    stock_code=stock_code,
                                    timestamp=date,
                                    price=current_price,
                                    strength=1.0,
                                    reason=f"Precomputed signal"
                                )]
                            return [signal] if not isinstance(signal, list) else signal
                    return None

                # PERF OPTIMIZATION: 禁用per-day并行，因为信号已经预计算，串行更快
                if False and self.enable_parallel and len(stock_data) > 3:
                    # 并行生成多股票信号
                    # PERF: avoid per-day ThreadPoolExecutor creation and avoid per-stock futures.
                    # We batch stocks into coarse tasks to reduce scheduling overhead.

                    # PERF: switch from "per-day submit many tasks" to "persistent workers".
                    # This dramatically reduces thread scheduling overhead when stock_count is large.
                    import threading

                    # Initialize worker context once (first trading day)
                    if not hasattr(self, "_signal_worker_ctx") or self._signal_worker_ctx is None:
                        items = list(stock_data.items())

                        # Greedy balance chunks by estimated per-stock compute cost.
                        # Cost proxy: number of trading days the stock participates (after warmup) with
                        # a small penalty for missing days.
                        scored = []
                        total_days = len(trading_dates) if trading_dates else 0

                        for code, df in items:
                            try:
                                # count how many trading_dates exist in this df
                                # (O(T) per stock; ok for init and much better load balance than len(df))
                                avail = df.index
                                avail_days = 0
                                for _d in trading_dates:
                                    if _d in avail:
                                        avail_days += 1
                                missing_ratio = (
                                    1.0 - (avail_days / total_days)
                                    if total_days > 0
                                    else 0.0
                                )
                                # warmup skip (executor only calls strategy when idx>=20)
                                effective_days = max(0, avail_days - 20)
                                cost = float(effective_days) * (1.0 + 0.10 * missing_ratio)
                                scored.append((cost, code, df))
                            except Exception:
                                scored.append((0.0, code, df))

                        scored.sort(reverse=True)

                        worker_n = max(1, int(self.max_workers or 1))
                        buckets = [([], 0.0) for _ in range(worker_n)]  # ([(code,df)], total_cost)
                        for cost, code, df in scored:
                            # pick bucket with smallest total_cost
                            bi = min(range(worker_n), key=lambda x: buckets[x][1])
                            buckets[bi][0].append((code, df))
                            buckets[bi] = (buckets[bi][0], buckets[bi][1] + float(cost))

                        chunks: List[List[Tuple[str, pd.DataFrame]]] = [b[0] for b in buckets]

                        shared = {"date": None, "error": None}
                        results: List[Tuple[List[TradingSignal], float, float, float]] = [
                            ([], 0.0, 0.0, 0.0) for _ in range(worker_n)
                        ]

                        barrier_start = threading.Barrier(worker_n + 1)
                        barrier_end = threading.Barrier(worker_n + 1)

                        def _worker(idx: int):
                            nonlocal chunks, shared, results
                            while True:
                                try:
                                    barrier_start.wait()
                                except Exception:
                                    return

                                cd = shared.get("date")
                                if cd is None:
                                    # shutdown signal
                                    try:
                                        barrier_end.wait()
                                    except Exception:
                                        pass
                                    return

                                batch_signals: List[TradingSignal] = []
                                slice_sum = 0.0
                                gen_sum = 0.0
                                gen_max = 0.0

                                try:
                                    for stock_code, data in chunks[idx]:
                                        if cd not in data.index:
                                            continue

                                        t0 = time.perf_counter()
                                        idx_map = None
                                        try:
                                            idx_map = data.attrs.get("_date_to_idx")
                                        except Exception:
                                            idx_map = None
                                        current_idx = (
                                            int(idx_map.get(cd))
                                            if isinstance(idx_map, dict) and cd in idx_map
                                            else int(data.index.get_loc(cd))
                                        )
                                        try:
                                            data.attrs["_current_date"] = cd
                                            data.attrs["_current_idx"] = current_idx
                                        except Exception:
                                            pass
                                        slice_dur = time.perf_counter() - t0
                                        slice_sum += float(slice_dur)

                                        if current_idx < 20:
                                            continue

                                        t1 = time.perf_counter()
                                        # 优先使用预计算信号
                                        sigs = get_precomputed_signal_fast(stock_code, cd)
                                        if sigs is None:
                                            # Fallback: 调用策略生成
                                            sigs = strategy.generate_signals(data, cd)
                                        gen_dur = time.perf_counter() - t1
                                        gen_sum += float(gen_dur)
                                        if gen_dur > gen_max:
                                            gen_max = float(gen_dur)

                                        if sigs:
                                            try:
                                                md = getattr(sigs[0], "metadata", None)
                                                if md is None:
                                                    sigs[0].metadata = {}
                                                    md = sigs[0].metadata
                                                if isinstance(md, dict):
                                                    md["_perf"] = {
                                                        "gen_wall": float(gen_dur),
                                                        "slice_wall": float(slice_dur),
                                                    }
                                            except Exception:
                                                pass

                                        batch_signals.extend(sigs)

                                    results[idx] = (batch_signals, slice_sum, gen_sum, gen_max)
                                except Exception as e:
                                    shared["error"] = e
                                    results[idx] = ([], slice_sum, gen_sum, gen_max)

                                try:
                                    barrier_end.wait()
                                except Exception:
                                    return

                        threads = []
                        for wi in range(worker_n):
                            t = threading.Thread(target=_worker, args=(wi,), daemon=True)
                            t.start()
                            threads.append(t)

                        self._signal_worker_ctx = {
                            "worker_n": worker_n,
                            "shared": shared,
                            "results": results,
                            "barrier_start": barrier_start,
                            "barrier_end": barrier_end,
                            "threads": threads,
                        }

                    ctx = self._signal_worker_ctx

                    sequential_start = (
                        time.perf_counter() if self.enable_performance_profiling else None
                    )

                    gen_time_max = 0.0

                    # Broadcast date to workers and collect
                    ctx["shared"]["date"] = current_date
                    ctx["shared"]["error"] = None

                    try:
                        ctx["barrier_start"].wait()
                        ctx["barrier_end"].wait()
                    except Exception as e:
                        logger.error(f"并行生成信号同步失败: {e}")

                    err = ctx["shared"].get("error")
                    if err is not None:
                        raise err

                    for (signals, slice_sum, gen_sum, gen_max) in ctx["results"]:
                        all_signals.extend(signals)
                        slice_time_total += float(slice_sum)
                        gen_time_total += float(gen_sum)
                        if gen_max and gen_max > gen_time_max:
                            gen_time_max = float(gen_max)

                    # 记录并行化效率（估算顺序执行时间）
                    if self.enable_performance_profiling and sequential_start:
                        parallel_time = time.perf_counter() - sequential_start
                        estimated_sequential_time = parallel_time * len(stock_data) / max(1, self.max_workers)
                        if i == 0:
                            self.performance_profiler.record_parallel_efficiency(
                                operation_name="signal_generation",
                                sequential_time=estimated_sequential_time,
                                parallel_time=parallel_time,
                                worker_count=self.max_workers,
                            )
                else:
                    gen_time_max = 0.0
                    # 顺序生成信号（股票数量少或禁用并行）
                    for stock_code, data in stock_data.items():
                        if current_date in data.index:
                            # 获取到当前日期的历史数据
                            t0 = time.perf_counter()
                            # same rationale as parallel path: avoid daily slicing copies
                            idx_map = None
                            try:
                                idx_map = data.attrs.get("_date_to_idx")
                            except Exception:
                                idx_map = None
                            current_idx = (
                                int(idx_map.get(current_date))
                                if isinstance(idx_map, dict) and current_date in idx_map
                                else int(data.index.get_loc(current_date))
                                if current_date in data.index
                                else -1
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
                                    signals = get_precomputed_signal_fast(stock_code, current_date)
                                    
                                    # 调试日志
                                    if current_idx == 20:  # 只在第一次打印
                                        logger.info(f"🔍 调试: stock={stock_code}, date={current_date}, precomputed_signals={'有' if signals else '无'}")
                                    
                                    if signals is None:
                                        # Fallback: 调用策略生成
                                        signals = strategy.generate_signals(data, current_date)
                                    
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
                if self.enable_performance_profiling and signal_start_time and self.performance_profiler:
                    signal_duration = time.perf_counter() - signal_start_time
                    signal_generation_times.append(signal_duration)

                    # 原有口径：整段信号生成（含切片、计算指标、融合等）
                    self.performance_profiler.record_function_call(
                        "generate_signals", signal_duration
                    )

                    # 新口径：拆开看"切片"与"策略信号生成"的比例
                    # 注意：并行模式下 slice_time_total / gen_time_total 是"各线程耗时求和"(work)，
                    # 不是 wall-clock；用于判断 CPU work 构成，但不能直接当成整体耗时百分比。
                    if slice_time_total > 0:
                        self.performance_profiler.record_function_call(
                            "slice_historical_data_work", float(slice_time_total)
                        )
                    if gen_time_total > 0:
                        self.performance_profiler.record_function_call(
                            "generate_signals_core_work", float(gen_time_total)
                        )

                    # 额外记录 wall-clock 口径（同 generate_signals，但名字更明确，便于报表阅读）
                    self.performance_profiler.record_function_call(
                        "generate_signals_wall", signal_duration
                    )

                    # 并行路径下 critical path 近似：单日最慢股票的 generate_signals wall
                    if gen_time_max > 0:
                        self.performance_profiler.record_function_call(
                            "generate_signals_core_wall_max", float(gen_time_max)
                        )

                        # 线程/调度开销（粗略）：整段 wall - 单日最慢单股 wall
                        overhead = float(signal_duration) - float(gen_time_max)
                        if overhead > 0:
                            self.performance_profiler.record_function_call(
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
                            pp = md.get("portfolio_perf") if isinstance(md, dict) else None
                            if isinstance(pp, dict):
                                sub = pp.get("sub_strategy_times")
                                if isinstance(sub, dict):
                                    for k, v in sub.items():
                                        self.performance_profiler.record_function_call(
                                            f"portfolio_substrategy__{k}", float(v)
                                        )
                                it = pp.get("integrate_time")
                                if it is not None:
                                    self.performance_profiler.record_function_call(
                                        "portfolio_integrate", float(it)
                                    )
                    except Exception:
                        pass

                total_signals += len(all_signals)

                # PERF优化：收集信号记录到内存，循环结束后批量写入数据库
                if task_id and all_signals:
                    try:
                        import uuid

                        # 使用传入的backtest_id或生成一个（只生成一次）
                        if _current_backtest_id is None:
                            _current_backtest_id = backtest_id or (
                                f"bt_{task_id[:8]}"
                                if task_id
                                else f"bt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                            )

                        # 收集信号记录到内存列表（不再每天写数据库）
                        for signal in all_signals:
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

                # 执行交易信号（性能监控）
                trade_start_time = (
                    time.perf_counter() if self.enable_performance_profiling else None
                )
                trades_this_day = 0
                executed_trade_signals = []  # 记录已执行的交易对应的信号
                unexecuted_signals = []  # 记录未执行的信号及原因

                # ===== trade execution mode =====
                trade_mode = None
                topk_limit: int | None = None  # for post-trade sanity checks
                try:
                    trade_mode = (strategy_config or {}).get("trade_mode")
                except Exception:
                    trade_mode = None

                # --- debug aid: log which trade path is used (only when needed) ---
                try:
                    if current_date.strftime("%Y-%m-%d") in ("2023-05-19", "2023-05-22", "2023-05-23"):
                        logger.info(
                            f"[trade_path] date={current_date.strftime('%Y-%m-%d')} trade_mode={trade_mode} "
                            f"signals={len(all_signals)} strategy_config_keys={list((strategy_config or {}).keys())}"
                        )
                except Exception:
                    pass

                if trade_mode == "topk_buffer":
                    # Daily TopK selection + buffer zone + max changes/day
                    k = int((strategy_config or {}).get("topk", 10))
                    topk_limit = k
                    buffer_n = int((strategy_config or {}).get("buffer", 20))
                    max_changes = int((strategy_config or {}).get("max_changes_per_day", 2))
                    trades_limit = max_changes

                    # Build ranking scores from signals (BUY strength positive, SELL negative)
                    scores: Dict[str, float] = {code: 0.0 for code in stock_data.keys()}
                    for sig in all_signals:
                        s = float(sig.strength or 0.0)
                        if sig.signal_type == SignalType.BUY:
                            scores[sig.stock_code] = max(scores.get(sig.stock_code, 0.0), s)
                        elif sig.signal_type == SignalType.SELL:
                            scores[sig.stock_code] = min(scores.get(sig.stock_code, 0.0), -s)

                    # Rebalance according to TopK+buffer rules
                    executed_trade_signals, unexecuted_signals, trades_this_day = self._rebalance_topk_buffer(
                        portfolio_manager=portfolio_manager,
                        current_prices=current_prices,
                        current_date=current_date,
                        scores=scores,
                        topk=k,
                        buffer_n=buffer_n,
                        max_changes=trades_limit,
                        strategy=strategy,
                        debug=bool((strategy_config or {}).get("debug_topk_buffer", False)),
                    )

                    # Debug: show what was executed on key dates / when trades happen
                    try:
                        if trades_this_day > 0 or current_date.strftime("%Y-%m-%d") in ("2023-05-22",):
                            logger.info(
                                f"[trade_exec][topk_buffer] date={current_date.strftime('%Y-%m-%d')} trades_this_day={trades_this_day} "
                                f"executed={len(executed_trade_signals)} unexecuted={len(unexecuted_signals)} holdings_after={len(portfolio_manager.positions)}"
                            )
                    except Exception:
                        pass

                else:
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

                # PERF优化：收集未执行和已执行信号到内存，循环结束后批量写入
                if task_id and unexecuted_signals:
                    _batch_unexecuted_signals.extend(unexecuted_signals)

                if task_id and executed_trade_signals:
                    _batch_executed_signals.extend(executed_trade_signals)

                # 记录组合快照
                portfolio_manager.record_portfolio_snapshot(
                    current_date, current_prices
                )

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
                        current_holdings = list(portfolio_manager.positions.keys())
                        if len(current_holdings) > int(k_limit):
                            logger.error(
                                f"[topk_buffer][sanity] positions_count={len(current_holdings)} > topk={k_limit} "
                                f"date={current_date.strftime('%Y-%m-%d')} holdings={sorted(current_holdings)}"
                            )
                except Exception as e:
                    logger.warning(f"[topk_buffer][sanity] check failed: {e}")

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

        # ========== PERF优化：循环结束后批量写入数据库 ==========
        # 将循环内收集的所有数据一次性写入，避免730次数据库操作
        if task_id:
            logger.info(f"🔄 开始批量写入数据库: 信号={len(_batch_signals_data)}, 已执行={len(_batch_executed_signals)}, 未执行={len(_batch_unexecuted_signals)}")
            
            try:
                from app.core.database import get_async_session_context
                from app.repositories.backtest_detailed_repository import (
                    BacktestDetailedRepository,
                )

                async with get_async_session_context() as session:
                    try:
                        repository = BacktestDetailedRepository(session)
                        
                        # 1. 批量保存所有信号记录
                        if _batch_signals_data:
                            await repository.batch_save_signal_records(
                                task_id=task_id,
                                backtest_id=_current_backtest_id,
                                signals_data=_batch_signals_data,
                            )
                            logger.info(f"✅ 批量保存信号记录完成: {len(_batch_signals_data)} 条")
                        
                        # 2. 批量更新未执行信号的原因
                        if _batch_unexecuted_signals:
                            for unexecuted_signal in _batch_unexecuted_signals:
                                await repository.update_signal_execution_reason(
                                    task_id=task_id,
                                    stock_code=unexecuted_signal["stock_code"],
                                    timestamp=unexecuted_signal["timestamp"],
                                    signal_type=unexecuted_signal["signal_type"],
                                    execution_reason=unexecuted_signal["execution_reason"],
                                )
                            logger.info(f"✅ 批量更新未执行原因完成: {len(_batch_unexecuted_signals)} 条")
                        
                        # 3. 批量标记已执行的信号
                        if _batch_executed_signals:
                            for executed_signal in _batch_executed_signals:
                                await repository.mark_signal_as_executed(
                                    task_id=task_id,
                                    stock_code=executed_signal["stock_code"],
                                    timestamp=executed_signal["timestamp"],
                                    signal_type=executed_signal["signal_type"],
                                )
                            logger.info(f"✅ 批量标记已执行完成: {len(_batch_executed_signals)} 条")
                        
                        await session.commit()
                        logger.info("✅ 所有数据库操作批量提交成功")
                        
                    except Exception as e:
                        await session.rollback()
                        logger.warning(f"批量写入数据库失败: {e}")
            except Exception as e:
                logger.warning(f"批量写入数据库时出错: {e}")
        # ========== END PERF优化 ==========

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
            # NOTE: Do NOT call get_portfolio_value({}) here - passing an empty price map
            # will value all positions at 0 and return cash-only, which makes final_value
            # inconsistent with total_return/portfolio_history.
            # Use the last recorded portfolio value (already computed with prices) when available.
            "final_value": (
                portfolio_manager.portfolio_history[-1]["portfolio_value"]
                if getattr(portfolio_manager, "portfolio_history", None)
                else portfolio_manager.get_portfolio_value({})
            ),
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
                    "trade_id": trade.trade_id if hasattr(trade, 'trade_id') else trade['trade_id'],
                    "stock_code": trade.stock_code if hasattr(trade, 'stock_code') else trade['stock_code'],
                    "action": trade.action if hasattr(trade, 'action') else trade['action'],
                    "quantity": trade.quantity if hasattr(trade, 'quantity') else trade['quantity'],
                    "price": trade.price if hasattr(trade, 'price') else trade['price'],
                    "timestamp": (trade.timestamp if hasattr(trade, 'timestamp') else trade['timestamp']).isoformat(),
                    "commission": trade.commission if hasattr(trade, 'commission') else trade['commission'],
                    "slippage_cost": getattr(trade, "slippage_cost", 0.0) if hasattr(trade, 'slippage_cost') else trade.get('slippage_cost', 0.0),
                    "pnl": trade.pnl if hasattr(trade, 'pnl') else trade['pnl'],
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

        holdings = list(portfolio_manager.positions.keys())
        holdings_set = set(holdings)

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
                nonzero = sum(1 for _, s in scores.items() if isinstance(s, (int, float)) and s != 0)
                logger.info(
                    f"[topk_buffer] {current_date.date()} holdings={len(holdings)} nonzero_scores={nonzero} "
                    f"topk={topk} buffer={buffer_n} max_changes={max_changes} "
                    f"to_sell={len(to_sell)} to_buy={len(to_buy)}"
                )
                logger.info(
                    f"[topk_buffer] topk_list(head)={topk_list[:min(5,len(topk_list))]} "
                    f"holdings(head)={holdings[:min(5,len(holdings))]}"
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
                reason=f"topk_buffer rebalance sell (out of buffer/topk)",
                metadata={"trade_mode": "topk_buffer"},
            )
            if strategy is not None:
                is_valid, validation_reason = strategy.validate_signal(
                    sig,
                    portfolio_manager.get_portfolio_value(current_prices),
                    portfolio_manager.positions,
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

            trade, failure_reason = portfolio_manager.execute_signal(sig, current_prices)
            if trade:
                successful_sells += 1
                trades_this_day += 1
                executed_trade_signals.append(
                    {"stock_code": code, "timestamp": current_date, "signal_type": sig.signal_type.name}
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
        current_positions_n = len(portfolio_manager.positions)
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
            if len(portfolio_manager.positions) >= topk:
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
                is_valid, validation_reason = strategy.validate_signal(
                    sig,
                    portfolio_manager.get_portfolio_value(current_prices),
                    portfolio_manager.positions,
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

            trade, failure_reason = portfolio_manager.execute_signal(sig, current_prices)
            if trade:
                trades_this_day += 1
                executed_trade_signals.append(
                    {"stock_code": code, "timestamp": current_date, "signal_type": sig.signal_type.name}
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
                def get_trade_attr(trade, attr: str):
                    if isinstance(trade, dict):
                        return trade.get(attr)
                    return getattr(trade, attr, None)

                for trade in portfolio_manager.trades:
                    stock_code = get_trade_attr(trade, 'stock_code')
                    action = get_trade_attr(trade, 'action')
                    pnl = get_trade_attr(trade, 'pnl') or 0.0

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
                pnls = [float(get_trade_attr(t, 'pnl') or 0.0) for t in portfolio_manager.trades]
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

            if len(stock_codes) > 1000:
                raise TaskError(
                    message=f"股票数量过多: {len(stock_codes)}，最多支持1000只股票",
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
