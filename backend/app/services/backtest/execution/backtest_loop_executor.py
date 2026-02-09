"""
回测循环执行模块
负责核心回测循环的执行
"""

from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import time
from datetime import datetime
from loguru import logger

from ..core.base_strategy import BaseStrategy
from ..core.portfolio_manager import PortfolioManager
from ..models import SignalType, TradingSignal
from app.core.error_handler import ErrorSeverity, TaskError
# 延迟导入以避免循环依赖
# from .backtest_progress_monitor import backtest_progress_monitor


class BacktestLoopExecutor:
    """回测循环执行器"""

    def __init__(self):
        """初始化回测循环执行器"""
        self.enable_performance_profiling = False
        self.performance_profiler = None

    async def execute_backtest_loop(
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
        # 延迟导入以避免循环依赖
        from .backtest_progress_monitor import backtest_progress_monitor
        
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
        _BATCH_FLUSH_THRESHOLD = 1000  # 流式写入阈值

        # 流式写入辅助函数：当积累足够数据时写入数据库
        async def _flush_batch_to_db(
            signals_data: List[dict],
            executed_signals: List[dict],
            unexecuted_signals: List[dict],
            backtest_id: str | None,
            clear_after: bool = True,
        ) -> None:
            """流式写入批量数据到数据库"""
            if not task_id:
                return
            total_count = len(signals_data) + len(executed_signals) + len(unexecuted_signals)
            if total_count == 0:
                return

            logger.info(f"🔄 流式写入数据库: 信号={len(signals_data)}, 已执行={len(executed_signals)}, 未执行={len(unexecuted_signals)}")

            try:
                from app.core.database import get_async_session_context
                from app.repositories.backtest_detailed_repository import (
                    BacktestDetailedRepository,
                )

                async with get_async_session_context() as session:
                    try:
                        repository = BacktestDetailedRepository(session)

                        # 1. 批量保存所有信号记录
                        if signals_data:
                            await repository.batch_save_signal_records(
                                task_id=task_id,
                                backtest_id=backtest_id,
                                signals_data=list(signals_data),  # 复制列表避免清空后问题
                            )

                        # 2. 批量更新未执行信号的原因
                        if unexecuted_signals:
                            signal_reasons = [
                                (
                                    sig["stock_code"],
                                    sig["timestamp"],
                                    sig["signal_type"],
                                    sig["execution_reason"]
                                )
                                for sig in unexecuted_signals
                            ]
                            await repository.batch_update_signal_execution_reasons(
                                task_id=task_id,
                                signal_reasons=signal_reasons
                            )

                        # 3. 批量标记已执行的信号
                        if executed_signals:
                            signal_keys = [
                                (
                                    sig["stock_code"],
                                    sig["timestamp"],
                                    sig["signal_type"]
                                )
                                for sig in executed_signals
                            ]
                            await repository.batch_mark_signals_as_executed(
                                task_id=task_id,
                                signal_keys=signal_keys
                            )

                        await session.commit()
                        logger.info(f"✅ 流式写入完成: {total_count} 条记录")

                    except Exception as e:
                        await session.rollback()
                        logger.warning(f"流式写入数据库失败: {e}")
            except Exception as e:
                logger.warning(f"流式写入数据库时出错: {e}")
        # ========== END PERF优化 ==========

        for i, current_date in enumerate(trading_dates):
            # PERF/BUGFIX: 统一初始化计时变量，避免某些分支/异常路径引用未赋值导致 UnboundLocalError
            slice_time_total = 0.0
            gen_time_total = 0.0
            gen_time_max = 0.0

            # 在循环开始时检查任务状态（每10个交易日检查一次，避免频繁检查）
            if task_id and i % 10 == 0 and i > 0:
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

                    # 优化 #5：缓存 portfolio stocks set，避免重复调用
                    portfolio_stocks = set(get_portfolio_stocks(portfolio_manager))
                    need_codes = portfolio_stocks.copy()
                    
                    if isinstance(sig_mat, np.ndarray):
                        # 优化 #5：使用向量化操作获取有信号的股票
                        sig_idx = np.nonzero(sig_mat[:, i])[0]
                        if len(sig_idx) > 0:
                            need_codes.update(codes[j] for j in sig_idx)

                    # BUGFIX: 如果没有预计算信号且持仓为空，需要为所有股票获取价格
                    # 否则无法生成信号（因为 generate_signals 需要当前价格）
                    if not need_codes:
                        need_codes = set(codes)

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

                # [P2 优化] 批量设置当前价格到数组，后续的 get_portfolio_value 等方法
                # 可以直接使用向量化计算，避免重复的字典查找
                if hasattr(portfolio_manager, 'set_current_prices'):
                    portfolio_manager.set_current_prices(current_prices)

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
                                                # [P1 优化] 使用 O(1) 字典查找替代 O(n) 的 np.where
                                                date_to_i = aligned_arrays.get("date_to_i")
                                                date_idx = date_to_i.get(date) if date_to_i else None
                                                
                                                if date_idx is not None:
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
                    # P1优化：每日缓存 portfolio_value 和 positions，避免循环内重复计算
                    daily_portfolio_value = portfolio_manager.get_portfolio_value(current_prices)
                    daily_positions = portfolio_manager.positions

                    for signal in all_signals:
                        # 验证信号
                        is_valid, validation_reason = strategy.validate_signal(
                            signal,
                            daily_portfolio_value,
                            daily_positions,
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

                # PERF优化A：流式增量写入 - 每积累1000条记录就写入一次数据库
                if task_id and (len(_batch_signals_data) + len(_batch_executed_signals) + len(_batch_unexecuted_signals)) >= _BATCH_FLUSH_THRESHOLD:
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
                # 性能优化: 降低数据库更新频率，从每5天改为每100天，减少I/O开销
                if task_id and i % 100 == 0:
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
                        # 注意：datetime 已在文件顶部导入，不要在此重复导入
                        # 否则会导致 "cannot access local variable 'datetime'" 错误
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

        # ========== PERF优化：循环结束后写入剩余数据 ==========
        # 写入流式写入未处理完的剩余数据
        if task_id and (len(_batch_signals_data) + len(_batch_executed_signals) + len(_batch_unexecuted_signals)) > 0:
            logger.info(f"🔄 写入剩余数据: 信号={len(_batch_signals_data)}, 已执行={len(_batch_executed_signals)}, 未执行={len(_batch_unexecuted_signals)}")
            await _flush_batch_to_db(
                signals_data=_batch_signals_data,
                executed_signals=_batch_executed_signals,
                unexecuted_signals=_batch_unexecuted_signals,
                backtest_id=_current_backtest_id,
            )
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


