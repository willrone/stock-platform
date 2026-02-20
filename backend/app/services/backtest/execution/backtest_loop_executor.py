"""
回测循环执行模块
负责核心回测循环的执行
"""

import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from app.core.error_handler import ErrorSeverity, TaskError

from ..core.base_strategy import BaseStrategy
from ..core.portfolio_manager import PortfolioManager
from ..core.risk_manager import PositionPriceInfo, RiskManager
from ..models import SignalType, TradingSignal

# 延迟导入以避免循环依赖
# from .backtest_progress_monitor import backtest_progress_monitor


def _check_and_execute_stop_loss_take_profit(
    risk_manager: RiskManager,
    portfolio_manager: PortfolioManager,
    current_prices: Dict[str, float],
    current_date: datetime,
) -> int:
    """
    检查并执行止损止盈信号（优先级高于策略信号）

    Returns:
        执行的交易数量
    """
    # 构建持仓价格信息
    positions = portfolio_manager.positions
    if not positions:
        return 0

    positions_info: Dict[str, PositionPriceInfo] = {}
    for code, pos in positions.items():
        price = current_prices.get(code)
        if price is not None and price > 0:
            positions_info[code] = PositionPriceInfo(
                stock_code=code,
                quantity=pos.quantity,
                avg_cost=pos.avg_cost,
                current_price=price,
                timestamp=current_date,
            )

    sl_tp_signals = risk_manager.check_stop_loss_take_profit(positions_info)
    if not sl_tp_signals:
        return 0

    trades_count = 0
    for signal in sl_tp_signals:
        trade, _ = portfolio_manager.execute_signal(signal, current_prices)
        if trade:
            trades_count += 1

    return trades_count


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
        signal_writer=None,
    ) -> Dict[str, Any]:
        """执行回测主循环"""
        # 延迟导入以避免循环依赖
        from .backtest_progress_monitor import backtest_progress_monitor

        total_signals = 0
        executed_trades = 0

        # P0: 初始化风险管理器（止损止盈 + 最大回撤熔断）
        risk_manager = RiskManager(portfolio_manager.config)

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

        # ========== PERF优化：批量收集数据库操作，分批写入 ==========
        # 避免在730天循环内每天都做数据库操作（原来是72秒的主要瓶颈）
        # 内存优化：每积累 _SIGNAL_FLUSH_THRESHOLD 条信号就写入一次DB并释放内存
        _batch_signals_data: List[dict] = []  # 收集信号记录（含 executed/execution_reason）
        _current_backtest_id: str | None = None  # 缓存 backtest_id
        _SIGNAL_FLUSH_THRESHOLD = 3000  # 每 3000 条信号刷一次DB（内存优化）
        _total_flushed_signals = 0  # 已刷入DB的信号总数
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
                    from .vectorized_loop import get_portfolio_stocks

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
                            j = (
                                code_to_i.get(c)
                                if isinstance(code_to_i, dict)
                                else None
                            )
                            if j is not None and bool(valid_mat[j, i]):
                                current_prices[c] = float(close_mat[j, i])

                    # BUGFIX: 对于持仓股票，如果当天 valid_mat 为 False（停牌等），
                    # 使用最近一个有效交易日的收盘价，避免持仓市值被计为0导致
                    # 组合价值剧烈跳变，从而严重放大波动率（80-130% → 正常应<30%）
                    for c in portfolio_stocks:
                        if c not in current_prices:
                            j = (
                                code_to_i.get(c)
                                if isinstance(code_to_i, dict)
                                else None
                            )
                            if j is not None:
                                # 向前搜索最近的有效价格
                                for k in range(i - 1, -1, -1):
                                    if bool(valid_mat[j, k]):
                                        current_prices[c] = float(close_mat[j, k])
                                        break

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

                    # BUGFIX: 对于持仓股票，如果当天没有数据（停牌等），
                    # 使用最近一个有效交易日的收盘价，避免持仓市值被计为0
                    if hasattr(portfolio_manager, 'positions'):
                        for stock_code in list(portfolio_manager.positions.keys()):
                            if stock_code not in current_prices:
                                data = stock_data.get(stock_code)
                                if data is not None and len(data) > 0:
                                    # 找到 current_date 之前最近的有效价格
                                    valid_dates = data.index[data.index <= current_date]
                                    if len(valid_dates) > 0:
                                        last_valid_idx = len(valid_dates) - 1
                                        current_prices[stock_code] = float(
                                            data["close"].values[last_valid_idx]
                                        )

                if not current_prices:
                    continue

                # [P2 优化] 批量设置当前价格到数组，后续的 get_portfolio_value 等方法
                # 可以直接使用向量化计算，避免重复的字典查找
                if hasattr(portfolio_manager, "set_current_prices"):
                    portfolio_manager.set_current_prices(current_prices)

                # ===== P0: 止损止盈检查（优先级高于策略信号） =====
                sl_tp_signals = _check_and_execute_stop_loss_take_profit(
                    risk_manager,
                    portfolio_manager,
                    current_prices,
                    current_date,
                )
                executed_trades += sl_tp_signals

                # ===== P0: 最大回撤熔断更新 =====
                portfolio_value_for_cb = portfolio_manager.get_portfolio_value(
                    current_prices
                )
                risk_manager.update_circuit_breaker(
                    portfolio_value_for_cb,
                    current_date,
                )

                # 生成交易信号（Phase1：优先用 ndarray signal matrix）
                all_signals: List[TradingSignal] = []

                if aligned_arrays is not None:
                    sig_mat = aligned_arrays.get("signal")
                    codes = aligned_arrays.get("stock_codes")
                    close_mat = aligned_arrays.get("close")
                    valid_mat = aligned_arrays.get("valid")
                    strength_mat = aligned_arrays.get("strength")
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
                                sig_strength = (
                                    float(strength_mat[j, i])
                                    if strength_mat is not None
                                    and strength_mat[j, i] > 0
                                    else 1.0
                                )
                                all_signals.append(
                                    TradingSignal(
                                        timestamp=current_date,
                                        stock_code=code,
                                        signal_type=stype,
                                        strength=sig_strength,
                                        price=price,
                                        reason="[aligned] precomputed",
                                        metadata=None,
                                    )
                                )

                # 若对齐数组已生成信号，跳过逐股票回退路径
                _skip_per_stock_fallback = len(all_signals) > 0

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
                                sig_strength = 1.0

                                try:
                                    # 方法 1: 优先使用 aligned_arrays（最快，O(1) 查找）
                                    if aligned_arrays is not None:
                                        code_to_i = aligned_arrays.get("code_to_i")
                                        close_mat = aligned_arrays.get("close")
                                        dates = aligned_arrays.get("dates")
                                        _strength_mat = aligned_arrays.get("strength")

                                        if (
                                            code_to_i is not None
                                            and close_mat is not None
                                            and dates is not None
                                        ):
                                            stock_idx = code_to_i.get(stock_code)
                                            if stock_idx is not None:
                                                # [P1 优化] 使用 O(1) 字典查找替代 O(n) 的 np.where
                                                date_to_i = aligned_arrays.get(
                                                    "date_to_i"
                                                )
                                                date_idx = (
                                                    date_to_i.get(date)
                                                    if date_to_i
                                                    else None
                                                )

                                                if date_idx is not None:
                                                    # 直接从 numpy 数组读取，无 pandas 开销
                                                    price_val = close_mat[
                                                        stock_idx, date_idx
                                                    ]
                                                    if not np.isnan(price_val):
                                                        current_price = float(price_val)
                                                    # 读取信号强度
                                                    if (
                                                        _strength_mat is not None
                                                        and _strength_mat[
                                                            stock_idx, date_idx
                                                        ]
                                                        > 0
                                                    ):
                                                        sig_strength = float(
                                                            _strength_mat[
                                                                stock_idx, date_idx
                                                            ]
                                                        )

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
                                        strength=sig_strength,
                                        reason="Precomputed signal",
                                    )
                                ]
                            return [signal] if not isinstance(signal, list) else signal
                    return None

                # PERF OPTIMIZATION: 禁用per-day并行，因为信号已经预计算，串行更快
                if _skip_per_stock_fallback:
                    pass  # aligned_arrays 已生成信号，跳过逐股票回退
                elif False and self.enable_parallel and len(stock_data) > 3:
                    # 并行生成多股票信号
                    # PERF: avoid per-day ThreadPoolExecutor creation and avoid per-stock futures.
                    # We batch stocks into coarse tasks to reduce scheduling overhead.

                    # PERF: switch from "per-day submit many tasks" to "persistent workers".
                    # This dramatically reduces thread scheduling overhead when stock_count is large.
                    import threading

                    # Initialize worker context once (first trading day)
                    if (
                        not hasattr(self, "_signal_worker_ctx")
                        or self._signal_worker_ctx is None
                    ):
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
                                cost = float(effective_days) * (
                                    1.0 + 0.10 * missing_ratio
                                )
                                scored.append((cost, code, df))
                            except Exception:
                                scored.append((0.0, code, df))

                        scored.sort(reverse=True)

                        worker_n = max(1, int(self.max_workers or 1))
                        buckets = [
                            ([], 0.0) for _ in range(worker_n)
                        ]  # ([(code,df)], total_cost)
                        for cost, code, df in scored:
                            # pick bucket with smallest total_cost
                            bi = min(range(worker_n), key=lambda x: buckets[x][1])
                            buckets[bi][0].append((code, df))
                            buckets[bi] = (buckets[bi][0], buckets[bi][1] + float(cost))

                        chunks: List[List[Tuple[str, pd.DataFrame]]] = [
                            b[0] for b in buckets
                        ]

                        shared = {"date": None, "error": None}
                        results: List[
                            Tuple[List[TradingSignal], float, float, float]
                        ] = [([], 0.0, 0.0, 0.0) for _ in range(worker_n)]

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
                                            if isinstance(idx_map, dict)
                                            and cd in idx_map
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
                                        sigs = get_precomputed_signal_fast(
                                            stock_code, cd
                                        )
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

                                    results[idx] = (
                                        batch_signals,
                                        slice_sum,
                                        gen_sum,
                                        gen_max,
                                    )
                                except Exception as e:
                                    shared["error"] = e
                                    results[idx] = ([], slice_sum, gen_sum, gen_max)

                                try:
                                    barrier_end.wait()
                                except Exception:
                                    return

                        threads = []
                        for wi in range(worker_n):
                            t = threading.Thread(
                                target=_worker, args=(wi,), daemon=True
                            )
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
                        time.perf_counter()
                        if self.enable_performance_profiling
                        else None
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

                    for signals, slice_sum, gen_sum, gen_max in ctx["results"]:
                        all_signals.extend(signals)
                        slice_time_total += float(slice_sum)
                        gen_time_total += float(gen_sum)
                        if gen_max and gen_max > gen_time_max:
                            gen_time_max = float(gen_max)

                    # 记录并行化效率（估算顺序执行时间）
                    if self.enable_performance_profiling and sequential_start:
                        parallel_time = time.perf_counter() - sequential_start
                        estimated_sequential_time = (
                            parallel_time * len(stock_data) / max(1, self.max_workers)
                        )
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
                                    signals = get_precomputed_signal_fast(
                                        stock_code, current_date
                                    )

                                    # 调试日志
                                    if current_idx == 20:  # 只在第一次打印
                                        logger.info(
                                            f"🔍 调试: stock={stock_code}, date={current_date}, precomputed_signals={'有' if signals is not None else '无'}"
                                        )

                                    if signals is None:
                                        # Fallback: 调用策略生成
                                        signals = strategy.generate_signals(
                                            data, current_date
                                        )

                                    # 调试日志：记录信号内容
                                    if signals is not None and current_idx == 20:
                                        logger.info(f"🔍 信号内容: {signals}")

                                    _dur = time.perf_counter() - t1
                                    gen_time_total += _dur
                                    if _dur > gen_time_max:
                                        gen_time_max = float(_dur)
                                    all_signals.extend(signals)
                                except Exception as e:
                                    import traceback as _tb

                                    logger.warning(
                                        f"生成信号失败 {stock_code}: {e}\n{_tb.format_exc()}"
                                    )
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
                            pp = (
                                md.get("portfolio_perf")
                                if isinstance(md, dict)
                                else None
                            )
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
                            _current_backtest_id = backtest_id or str(uuid.uuid4())

                        # 构建信号 lookup key → index 映射，用于后续标记 executed/execution_reason
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
                                "execution_reason": None,
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

                # ===== P0: 熔断过滤（阻止 BUY 信号，保留 SELL） =====
                all_signals = risk_manager.filter_signals_by_circuit_breaker(
                    all_signals
                )

                # ===== trade execution mode =====
                trade_mode = None
                _topk_limit: int | None = None  # for post-trade sanity checks
                try:
                    trade_mode = (strategy_config or {}).get("trade_mode")
                except Exception:
                    trade_mode = None

                # --- debug aid: log which trade path is used (only when needed) ---
                try:
                    if current_date.strftime("%Y-%m-%d") in (
                        "2023-05-19",
                        "2023-05-22",
                        "2023-05-23",
                    ):
                        logger.info(
                            f"[trade_path] date={current_date.strftime('%Y-%m-%d')} trade_mode={trade_mode} "
                            f"signals={len(all_signals)} strategy_config_keys={list((strategy_config or {}).keys())}"
                        )
                except Exception:
                    pass

                if trade_mode == "topk_buffer":
                    # Daily TopK selection + buffer zone + max changes/day
                    k = int((strategy_config or {}).get("topk", 10))
                    _topk_limit = k
                    buffer_n = int((strategy_config or {}).get("buffer", 20))
                    max_changes = int(
                        (strategy_config or {}).get("max_changes_per_day", 2)
                    )
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
                    min_buy_score = float(
                        (strategy_config or {}).get("min_buy_score", 0.0)
                    )
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
                        debug=bool(
                            (strategy_config or {}).get("debug_topk_buffer", False)
                        ),
                        min_buy_score=min_buy_score,
                    )

                    # Debug: show what was executed on key dates / when trades happen
                    try:
                        if trades_this_day > 0 or current_date.strftime("%Y-%m-%d") in (
                            "2023-05-22",
                        ):
                            logger.info(
                                f"[trade_exec][topk_buffer] date={current_date.strftime('%Y-%m-%d')} trades_this_day={trades_this_day} "
                                f"executed={len(executed_trade_signals)} unexecuted={len(unexecuted_signals)} holdings_after={len(portfolio_manager.positions)}"
                            )
                    except Exception:
                        pass

                else:
                    # P1优化：每日缓存 portfolio_value 和 positions，避免循环内重复计算
                    daily_portfolio_value = portfolio_manager.get_portfolio_value(
                        current_prices
                    )
                    daily_positions = portfolio_manager.positions

                    for signal in all_signals:
                        # 验证信号
                        is_valid, validation_reason = strategy.validate_signal(
                            signal,
                            daily_portfolio_value,
                            daily_positions,
                            entry_dates=getattr(portfolio_manager, "entry_dates", None),
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

                # PERF优化：直接在 _batch_signals_data 中更新 executed/execution_reason
                # 避免后续 UPDATE 操作，所有状态在内存中一次性确定
                if task_id and (executed_trade_signals or unexecuted_signals):
                    # 构建当天信号的快速查找索引（从 _batch_signals_data 尾部回溯）
                    # 当天新增的信号数量 = len(all_signals)（刚刚 append 的）
                    _today_start_idx = len(_batch_signals_data) - len(all_signals)
                    _today_lookup: Dict[tuple, int] = {}
                    for _si in range(_today_start_idx, len(_batch_signals_data)):
                        _sd = _batch_signals_data[_si]
                        _key = (_sd["stock_code"], _sd["signal_type"])
                        _today_lookup[_key] = _si

                    # 标记已执行的信号
                    for _exec_sig in executed_trade_signals:
                        _key = (_exec_sig["stock_code"], _exec_sig["signal_type"])
                        _idx = _today_lookup.get(_key)
                        if _idx is not None:
                            _batch_signals_data[_idx]["executed"] = True
                            _batch_signals_data[_idx]["execution_reason"] = None

                    # 标记未执行的信号及原因
                    for _unexec_sig in unexecuted_signals:
                        _key = (_unexec_sig["stock_code"], _unexec_sig["signal_type"])
                        _idx = _today_lookup.get(_key)
                        if _idx is not None:
                            _batch_signals_data[_idx]["executed"] = False
                            _batch_signals_data[_idx][
                                "execution_reason"
                            ] = _unexec_sig.get("execution_reason", "未知原因")

                # 内存优化：当信号积累超过阈值时，中间刷入DB并释放内存
                if task_id and len(_batch_signals_data) >= _SIGNAL_FLUSH_THRESHOLD:
                    if signal_writer is not None:
                        # 使用 StreamSignalWriter
                        signal_writer.buffer_many(_batch_signals_data)
                        signal_writer.flush()
                        _total_flushed_signals = signal_writer.total_written
                    else:
                        # 向后兼容：使用旧路径
                        _flushed = self._flush_signals_to_db(
                            _batch_signals_data, task_id, _current_backtest_id
                        )
                        _total_flushed_signals += _flushed
                    _batch_signals_data.clear()

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

                # 更新进度监控（使用内存缓存，只在关键节点写入数据库）
                # 性能优化: 每个交易日更新内存缓存，缓存判断是否需要写 DB（每 10% 进度）
                if task_id:
                    # 计算进度百分比（回测执行阶段占30-90%，即60%的进度范围）
                    execution_progress = (i + 1) / len(trading_dates) * 100
                    overall_progress = 30 + (execution_progress / 100) * 60  # 30%到90%

                    # 构建进度数据（内存中始终保持最新）
                    portfolio_value = portfolio_manager.get_portfolio_value(
                        current_prices
                    )
                    progress_update_data = {
                        "processed_days": i + 1,
                        "total_days": len(trading_dates),
                        "current_date": current_date.strftime("%Y-%m-%d"),
                        "signals_generated": len(all_signals),
                        "trades_executed": trades_this_day,
                        "total_signals": total_signals,
                        "total_trades": executed_trades,
                        "portfolio_value": portfolio_value,
                        "last_updated": datetime.utcnow().isoformat(),
                    }

                    # 更新内存缓存，由缓存判断是否需要写 DB
                    from app.utils.task_progress_cache import task_progress_cache

                    should_flush = task_progress_cache.update_progress(
                        task_id=task_id,
                        progress=overall_progress,
                        result_data={"progress_data": progress_update_data},
                    )

                    if should_flush:
                        logger.debug(
                            f"准备写入进度到DB: task_id={task_id}, i={i}, "
                            f"total_days={len(trading_dates)}, progress={overall_progress:.1f}%"
                        )
                        try:
                            from app.core.database import SessionLocal
                            from app.models.task_models import TaskStatus
                            from app.repositories.task_repository import TaskRepository

                            session = SessionLocal()
                            try:
                                task_repo = TaskRepository(session)
                                existing_task = task_repo.get_task_by_id(task_id)
                                if not existing_task:
                                    logger.warning(f"任务不存在，无法更新进度: {task_id}")
                                    raise TaskError(
                                        message=f"任务 {task_id} 已被删除，停止回测执行",
                                        severity=ErrorSeverity.LOW,
                                    )
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
                                    result_data["progress_data"] = progress_update_data

                                    task_repo.update_task_status(
                                        task_id=task_id,
                                        status=TaskStatus.RUNNING,
                                        progress=overall_progress,
                                        result=result_data,
                                    )
                                    session.commit()
                                    task_progress_cache.mark_flushed(task_id)
                                    logger.info(
                                        f"进度已写入DB: task_id={task_id}, "
                                        f"progress={overall_progress:.1f}%, "
                                        f"days={i + 1}/{len(trading_dates)}"
                                    )
                            except Exception as inner_error:
                                session.rollback()
                                logger.error(
                                    f"更新任务进度到数据库失败: {inner_error}",
                                    exc_info=True,
                                )
                                raise
                            finally:
                                session.close()
                        except TaskError:
                            raise
                        except Exception as db_error:
                            logger.error(f"更新任务进度到数据库失败: {db_error}", exc_info=True)

                    # 进度监控已通过同步DB写入完成，跳过async调用避免子进程死锁

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

                # 警告已通过logger记录，跳过async调用避免子进程死锁
                if task_id:
                    logger.warning(f"回测警告 task={task_id}: {error_msg}")

                continue

        # ========== PERF优化：循环结束后写入剩余信号数据 ==========
        if task_id and _batch_signals_data:
            if signal_writer is not None:
                signal_writer.buffer_many(_batch_signals_data)
                signal_writer.finalize()
                _total_flushed_signals = signal_writer.total_written
            else:
                _flushed = self._flush_signals_to_db(
                    _batch_signals_data, task_id, _current_backtest_id
                )
                _total_flushed_signals += _flushed
            _batch_signals_data.clear()
        elif task_id and signal_writer is not None:
            # 缓冲区为空但 signal_writer 可能还有未 finalize 的数据
            signal_writer.finalize()
            _total_flushed_signals = signal_writer.total_written

        if task_id and _total_flushed_signals > 0:
            logger.info(f"✅ 信号写入完成: 共 {_total_flushed_signals} 条记录")
        # ========== END PERF优化 ==========

        # 最终进度更新 + 清理内存缓存
        if task_id:
            from app.utils.task_progress_cache import task_progress_cache

            task_progress_cache.remove(task_id)

            final_portfolio_value = portfolio_manager.get_portfolio_value({})
            # 跳过async progress monitor调用，避免子进程死锁
            logger.info(f"回测循环完成 task={task_id}, days={len(trading_dates)}, portfolio={final_portfolio_value:.2f}")

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
            "circuit_breaker_summary": risk_manager.get_circuit_breaker_summary(),
        }

    @staticmethod
    def _flush_signals_to_db(
        batch_signals_data: List[dict],
        task_id: str,
        backtest_id: str,
    ) -> int:
        """将信号数据批量写入DB并返回写入条数。

        内存优化：调用方在 flush 后应 clear() 列表释放内存。
        使用 psycopg2 直接连接 PostgreSQL 批量插入。
        """
        if not batch_signals_data:
            return 0

        import json as _json

        import psycopg2
        import psycopg2.extras

        from app.core.config import settings as _settings

        count = len(batch_signals_data)

        try:
            # 预处理数据
            _insert_rows = []
            for _sd in batch_signals_data:
                _ts = _sd["timestamp"]
                _ts_str = _ts.isoformat() if hasattr(_ts, "isoformat") else str(_ts)

                _meta = _sd.get("metadata")
                _meta_str = None
                if _meta is not None:
                    try:
                        _meta_str = _json.dumps(_meta, ensure_ascii=False, default=str)
                    except Exception:
                        pass

                _insert_rows.append((
                    backtest_id,
                    _sd["signal_id"],
                    _sd["stock_code"],
                    _sd.get("stock_name"),
                    _sd["signal_type"],
                    _ts_str,
                    float(_sd["price"]),
                    float(_sd.get("strength", 0.0)),
                    _sd.get("reason"),
                    _meta_str,
                    True if _sd.get("executed") else False,
                    _sd.get("execution_reason"),
                ))

            # 从 DATABASE_URL 构建 psycopg2 连接字符串
            _dsn = _settings.database_url_sync

            _raw_insert_sql = """
                INSERT INTO signal_records
                    (backtest_id, signal_id, stock_code, stock_name,
                     signal_type, timestamp, price, strength, reason,
                     signal_metadata, executed, execution_reason)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """

            _WRITE_BATCH_SIZE = 5000
            _max_retries = 3

            for _attempt in range(_max_retries + 1):
                try:
                    _conn = psycopg2.connect(_dsn)
                    try:
                        _cur = _conn.cursor()
                        for _bi in range(0, len(_insert_rows), _WRITE_BATCH_SIZE):
                            psycopg2.extras.execute_batch(
                                _cur,
                                _raw_insert_sql,
                                _insert_rows[_bi : _bi + _WRITE_BATCH_SIZE],
                            )
                        _conn.commit()
                        _cur.close()
                    finally:
                        _conn.close()
                    logger.debug(f"信号批量写入: {count} 条")
                    return count
                except Exception as e:
                    err_msg = str(e).lower()
                    if ("deadlock" in err_msg or "could not serialize" in err_msg) and _attempt < _max_retries:
                        time.sleep(0.5 * (2 ** _attempt))
                    else:
                        logger.error(f"信号写入DB失败: {e}")
                        return 0
        except Exception as e:
            logger.error(f"信号写入预处理失败: {e}")
            return 0

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
        min_buy_score: float = 0.0,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
        """每日 TopK 选股 + buffer 换仓 + 每天最多换 max_changes 只。

        规则（实盘对齐版）：
        - 目标持仓数量=topk
        - 若持仓仍在 Top(topk+buffer_n) 内，则尽量保留（减少换手）
        - 每天最多做 max_changes 个 "卖出+买入" 的替换
        - min_buy_score: 最低买入分数阈值，低于此分数的股票不会被买入（但已持有的可保留在buffer内）

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
        # 过滤：只有分数 > min_buy_score 的股票才能进入 topk 候选
        qualified = [(c, s) for c, s in ranked if s > min_buy_score]
        effective_topk = min(topk, len(qualified))
        topk_list = [c for c, _ in qualified[:effective_topk]]
        # buffer 仍然基于全排名（已持有的低分股票可以在 buffer 内保留，避免频繁卖出）
        buffer_list = [c for c, _ in ranked[: max(topk, topk + buffer_n)]]
        buffer_set = set(buffer_list)

        holdings = list(portfolio_manager.positions.keys())

        # Keep holdings inside buffer zone, but force-sell if score is actively negative
        score_map = dict(ranked)
        kept = [c for c in holdings if c in buffer_set and score_map.get(c, 0.0) >= -min_buy_score]

        # If kept > topk, trim lowest-ranked among kept
        rank_index = {c: i for i, (c, _) in enumerate(ranked)}
        if len(kept) > topk:
            kept_sorted = sorted(kept, key=lambda c: rank_index.get(c, 10**9))
            kept = kept_sorted[:topk]

        kept_set = set(kept)

        # Sell candidates: holdings outside buffer OR trimmed OR actively bearish
        to_sell = [c for c in holdings if c not in kept_set]

        # Buy candidates: topk names not already kept
        to_buy = [c for c in topk_list if c not in kept_set]

        # 独立限制卖出和买入（修复：买卖耦合导致初始建仓失败）
        n_sell = min(max_changes, len(to_sell))
        n_buy = min(max_changes, len(to_buy))

        # 初始建仓：holdings 为空时，允许直接买入 topk 只股票
        if not holdings:
            n_buy = min(topk, len(to_buy))

        to_sell = to_sell[:n_sell]
        to_buy = to_buy[:n_buy]

        # Execute sells first
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
                is_valid, validation_reason = strategy.validate_signal(
                    sig,
                    portfolio_manager.get_portfolio_value(current_prices),
                    portfolio_manager.positions,
                    entry_dates=getattr(portfolio_manager, "entry_dates", None),
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

        # Execute buys
        for code in to_buy:
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
                    entry_dates=getattr(portfolio_manager, "entry_dates", None),
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
