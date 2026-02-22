"""
P2 多进程回测 Worker 模块

每个 worker 进程独立执行一组股票的回测：
- 独立加载数据（ThreadPool 并行）
- 独立预计算信号
- 独立执行回测循环
- 通过 multiprocessing.Queue 发送进度到主进程

关键约束：
- 纯同步执行，不使用 asyncio
- 子进程中独立创建 DB 连接（不依赖主进程连接池）
- 子进程中重新配置 loguru
"""

import os
import time
from datetime import datetime
from multiprocessing import Queue
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from ..models import BacktestConfig, SignalType, TradingSignal


# ── 进度消息类型 ──

PROGRESS_UPDATE = "progress"
PROGRESS_DONE = "done"
PROGRESS_ERROR = "error"


def _setup_worker_logging(worker_id: int) -> None:
    """子进程中重新配置 loguru"""
    logger.remove()
    log_dir = Path("/home/willrone/Projects/willrone/data/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(
        log_dir / f"mp_worker_{worker_id}.log",
        format=(
            "{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | "
            "PID:{process} | W{extra[worker_id]} | {message}"
        ),
        level="INFO",
        rotation="50 MB",
        retention="7 days",
        enqueue=False,
    )
    logger.configure(extra={"worker_id": worker_id})


def _send_progress(
    queue: Optional[Queue],
    worker_id: int,
    processed_days: int,
    total_days: int,
    current_date: str,
    portfolio_value: float,
    total_signals: int,
    total_trades: int,
) -> None:
    """向主进程发送进度更新"""
    if queue is None:
        return
    try:
        queue.put_nowait({
            "type": PROGRESS_UPDATE,
            "worker_id": worker_id,
            "processed_days": processed_days,
            "total_days": total_days,
            "current_date": current_date,
            "portfolio_value": portfolio_value,
            "total_signals": total_signals,
            "total_trades": total_trades,
            "timestamp": time.time(),
        })
    except Exception:
        pass  # 非关键路径，不阻塞回测


def worker_backtest(args: tuple) -> Dict[str, Any]:
    """
    Worker 进程入口（供 Pool.map 调用）

    Args 通过 tuple 传入（Pool.map 限制）:
        worker_id, stock_codes, data_dir, start_date_str, end_date_str,
        strategy_name, strategy_config, backtest_config_dict,
        task_id, progress_queue

    Returns:
        回测结果字典
    """
    (
        worker_id,
        stock_codes,
        data_dir,
        start_date_str,
        end_date_str,
        strategy_name,
        strategy_config,
        backtest_config_dict,
        task_id,
        progress_queue,
    ) = args

    _setup_worker_logging(worker_id)
    pid = os.getpid()
    logger.info(
        f"Worker {worker_id} 启动: PID={pid}, "
        f"stocks={len(stock_codes)}"
    )

    t_total = time.perf_counter()

    try:
        start_date = datetime.fromisoformat(start_date_str)
        end_date = datetime.fromisoformat(end_date_str)

        # ── 1. 加载数据（ThreadPool 并行） ──
        t0 = time.perf_counter()
        from .data_loader import DataLoader

        loader = DataLoader(
            data_dir=data_dir,
            max_workers=min(4, len(stock_codes)),
        )
        stock_data = loader.load_multiple_stocks_sync(
            stock_codes, start_date, end_date,
        )
        t_load = time.perf_counter() - t0
        logger.info(
            f"Worker {worker_id}: 数据加载完成, "
            f"{len(stock_data)}/{len(stock_codes)} 只, "
            f"耗时 {t_load:.2f}s"
        )

        if not stock_data:
            return _error_result(
                worker_id, stock_codes, "所有股票数据加载失败"
            )

        actual_codes = list(stock_data.keys())

        # ── 2. 创建策略 ──
        from ..strategies.strategy_factory import (
            AdvancedStrategyFactory,
            StrategyFactory,
        )

        try:
            strategy = AdvancedStrategyFactory.create_strategy(
                strategy_name, strategy_config
            )
        except Exception:
            strategy = StrategyFactory.create_strategy(
                strategy_name, strategy_config
            )

        # ── 3. 创建回测配置和组合管理器 ──
        config = BacktestConfig(**backtest_config_dict)
        from ..core.portfolio_manager_array import PortfolioManagerArray

        portfolio = PortfolioManagerArray(config, actual_codes)

        # ── 4. 获取交易日历 ──
        from .data_preprocessor import DataPreprocessor

        preprocessor = DataPreprocessor(
            enable_parallel=False, max_workers=1
        )
        trading_dates = preprocessor.get_trading_calendar(
            stock_data, start_date, end_date
        )

        if len(trading_dates) < 20:
            return _error_result(
                worker_id, stock_codes,
                f"交易日不足: {len(trading_dates)}",
            )

        # ── 5. 预计算信号 ──
        t0 = time.perf_counter()
        preprocessor.build_date_index(stock_data)
        preprocessor.precompute_strategy_signals(strategy, stock_data)
        precomputed_signals = (
            preprocessor.extract_precomputed_signals_to_dict(
                strategy, stock_data
            )
        )
        t_precompute = time.perf_counter() - t0

        # ── 6. 构建对齐数组 ──
        t0 = time.perf_counter()
        aligned_arrays = preprocessor.build_aligned_arrays(
            strategy, stock_data, trading_dates
        )
        t_align = time.perf_counter() - t0

        # 释放 attrs 缓存
        import gc
        for df in stock_data.values():
            try:
                if hasattr(df, "attrs") and "_precomputed_signals" in df.attrs:
                    del df.attrs["_precomputed_signals"]
            except Exception:
                pass
        gc.collect()

        # ── 7. 执行回测循环（纯同步） ──
        t0 = time.perf_counter()
        loop_result = _run_backtest_loop(
            worker_id=worker_id,
            strategy=strategy,
            portfolio=portfolio,
            stock_data=stock_data,
            trading_dates=trading_dates,
            strategy_config=strategy_config,
            precomputed_signals=precomputed_signals,
            aligned_arrays=aligned_arrays,
            task_id=task_id,
            progress_queue=progress_queue,
        )
        t_loop = time.perf_counter() - t0

        # 释放大对象
        del aligned_arrays
        precomputed_signals.clear()
        gc.collect()

        # ── 8. 计算绩效 ──
        metrics = portfolio.get_performance_metrics()

        t_total_elapsed = time.perf_counter() - t_total
        logger.info(
            f"Worker {worker_id} 完成: "
            f"load={t_load:.1f}s precompute={t_precompute:.1f}s "
            f"align={t_align:.1f}s loop={t_loop:.1f}s "
            f"total={t_total_elapsed:.1f}s "
            f"signals={loop_result['total_signals']} "
            f"trades={loop_result['executed_trades']}"
        )

        # 通知主进程完成
        if progress_queue is not None:
            try:
                progress_queue.put_nowait({
                    "type": PROGRESS_DONE,
                    "worker_id": worker_id,
                })
            except Exception:
                pass

        # 序列化 trade_history（与 report_generator 格式一致）
        trade_history = []
        for trade in portfolio.trades:
            trade_history.append({
                "trade_id": trade.trade_id if hasattr(trade, "trade_id") else trade.get("trade_id", ""),
                "stock_code": trade.stock_code if hasattr(trade, "stock_code") else trade.get("stock_code", ""),
                "action": trade.action if hasattr(trade, "action") else trade.get("action", ""),
                "quantity": trade.quantity if hasattr(trade, "quantity") else trade.get("quantity", 0),
                "price": trade.price if hasattr(trade, "price") else trade.get("price", 0),
                "timestamp": (trade.timestamp if hasattr(trade, "timestamp") else trade.get("timestamp", "")).isoformat() if hasattr(trade.timestamp if hasattr(trade, "timestamp") else "", "isoformat") else str(trade.timestamp if hasattr(trade, "timestamp") else trade.get("timestamp", "")),
                "commission": trade.commission if hasattr(trade, "commission") else trade.get("commission", 0),
                "slippage_cost": getattr(trade, "slippage_cost", 0.0) if hasattr(trade, "slippage_cost") else trade.get("slippage_cost", 0.0),
                "pnl": trade.pnl if hasattr(trade, "pnl") else trade.get("pnl", 0),
            })

        # 序列化 portfolio_history（与 report_generator 格式一致）
        serialized_history = []
        for snapshot in portfolio.portfolio_history:
            serialized_history.append({
                "date": snapshot["date"].isoformat() if hasattr(snapshot["date"], "isoformat") else str(snapshot["date"]),
                "portfolio_value": snapshot["portfolio_value"],
                "portfolio_value_without_cost": snapshot.get("portfolio_value_without_cost", snapshot["portfolio_value"]),
                "cash": snapshot["cash"],
                "positions_count": len(snapshot.get("positions", {})),
                "positions": snapshot.get("positions", {}),
                "total_trades": snapshot.get("total_trades", 0),
                "total_commission": snapshot.get("total_commission", 0),
                "total_slippage": snapshot.get("total_slippage", 0),
            })

        return {
            "worker_id": worker_id,
            "stock_codes": actual_codes,
            "total_signals": loop_result["total_signals"],
            "executed_trades": loop_result["executed_trades"],
            "trading_days": len(trading_dates),
            "performance_metrics": metrics,
            "equity_curve": portfolio.equity_curve,
            "trade_history": trade_history,
            "portfolio_history": serialized_history,
            "final_cash": portfolio.cash,
            "initial_cash": config.initial_cash,
            "total_commission": portfolio.total_commission,
            "total_slippage": portfolio.total_slippage,
            "total_capital_injection": getattr(portfolio, "total_capital_injection", 0.0),
            "circuit_breaker_summary": loop_result.get(
                "circuit_breaker_summary"
            ),
            "timing": {
                "data_loading_s": t_load,
                "precompute_s": t_precompute,
                "align_s": t_align,
                "loop_s": t_loop,
                "total_s": t_total_elapsed,
            },
        }

    except Exception as e:
        logger.error(
            f"Worker {worker_id} 失败: {e}", exc_info=True
        )
        if progress_queue is not None:
            try:
                progress_queue.put_nowait({
                    "type": PROGRESS_ERROR,
                    "worker_id": worker_id,
                    "error": str(e),
                })
            except Exception:
                pass
        return _error_result(worker_id, stock_codes, str(e))


def _error_result(
    worker_id: int,
    stock_codes: List[str],
    error: str,
) -> Dict[str, Any]:
    """构建错误结果"""
    return {
        "worker_id": worker_id,
        "stock_codes": stock_codes,
        "error": error,
    }


def _run_backtest_loop(
    worker_id: int,
    strategy,
    portfolio,
    stock_data: Dict[str, pd.DataFrame],
    trading_dates: list,
    strategy_config: Dict[str, Any],
    precomputed_signals: Dict,
    aligned_arrays: Dict[str, Any],
    task_id: Optional[str],
    progress_queue: Optional[Queue],
) -> Dict[str, Any]:
    """
    纯同步回测循环（worker 内部）

    简化版的 BacktestLoopExecutor.execute_backtest_loop，
    去掉 async/await，去掉内存缓存依赖。
    """
    from ..core.risk_manager import PositionPriceInfo, RiskManager
    from app.core.error_handler import ErrorSeverity, TaskError

    total_signals = 0
    executed_trades = 0
    risk_manager = RiskManager(portfolio.config)

    # 对齐数组快速路径
    use_aligned = aligned_arrays is not None
    if use_aligned:
        a_close = aligned_arrays.get("close")
        a_open = aligned_arrays.get("open")
        a_high = aligned_arrays.get("high")
        a_low = aligned_arrays.get("low")
        a_volume = aligned_arrays.get("volume")
        a_signals = aligned_arrays.get("signals")
        a_codes = aligned_arrays.get("stock_codes", [])
        a_dates = aligned_arrays.get("dates", [])
        # 构建日期→行索引映射
        date_to_row = {}
        for idx, d in enumerate(a_dates):
            date_to_row[d] = idx

    # topk_buffer 模式参数
    trade_mode = strategy_config.get("trade_mode", "default") if strategy_config else "default"
    is_topk_buffer = trade_mode == "topk_buffer"
    k_limit = strategy_config.get("topk", 10) if strategy_config else 10
    buffer_size = strategy_config.get("buffer", 20) if strategy_config else 20
    max_changes = strategy_config.get("max_changes_per_day", 3) if strategy_config else 3
    min_buy_score = strategy_config.get("min_buy_score", 0.0) if strategy_config else 0.0

    circuit_breaker_triggered = False
    circuit_breaker_date = None

    progress_interval = max(len(trading_dates) // 20, 10)  # 每 5% 或至少每 10 天

    for i, current_date in enumerate(trading_dates):
        # ── 获取当前价格 ──
        current_prices: Dict[str, float] = {}
        if use_aligned:
            row = date_to_row.get(current_date)
            if row is not None and a_close is not None:
                for col_idx, code in enumerate(a_codes):
                    val = a_close[row, col_idx]
                    if not np.isnan(val) and val > 0:
                        current_prices[code] = float(val)
        else:
            for code, data in stock_data.items():
                if current_date in data.index:
                    try:
                        idx = data.index.get_loc(current_date)
                        current_prices[code] = float(
                            data["close"].iloc[idx]
                        )
                    except Exception:
                        pass

        if not current_prices:
            continue

        # ── 熔断检查 ──
        if not circuit_breaker_triggered:
            cb_result = risk_manager.check_circuit_breaker(
                portfolio.get_portfolio_value(current_prices),
                current_date,
            )
            if cb_result and cb_result.get("triggered"):
                circuit_breaker_triggered = True
                circuit_breaker_date = current_date
                logger.warning(
                    f"Worker {worker_id}: 熔断触发 "
                    f"date={current_date}"
                )

        if circuit_breaker_triggered:
            portfolio.record_portfolio_snapshot(
                current_date, current_prices
            )
            continue

        # ── 止损止盈 ──
        sl_tp_trades = _check_stop_loss_take_profit(
            risk_manager, portfolio, current_prices, current_date
        )
        executed_trades += sl_tp_trades

        # ── 生成信号 ──
        all_signals: List[TradingSignal] = []

        if use_aligned and a_signals is not None:
            row = date_to_row.get(current_date)
            if row is not None:
                for col_idx, code in enumerate(a_codes):
                    sig_val = a_signals[row, col_idx]
                    if np.isnan(sig_val):
                        continue
                    sig_int = int(sig_val)
                    if sig_int == 0:
                        continue
                    try:
                        sig_type = SignalType(sig_int)
                    except (ValueError, KeyError):
                        continue
                    price = current_prices.get(code, 0.0)
                    if price <= 0:
                        continue
                    all_signals.append(TradingSignal(
                        timestamp=current_date,
                        stock_code=code,
                        signal_type=sig_type,
                        strength=0.8,
                        price=price,
                        reason="precomputed",
                        metadata={},
                    ))
        else:
            # fallback: precomputed_signals dict
            for code in stock_data:
                key = (code, current_date)
                sig_data = precomputed_signals.get(key)
                if sig_data is None:
                    continue
                if isinstance(sig_data, dict):
                    sig_type = sig_data.get("signal_type")
                    strength = sig_data.get("strength", 0.8)
                elif isinstance(sig_data, SignalType):
                    sig_type = sig_data
                    strength = 0.8
                else:
                    continue
                if sig_type is None or sig_type == SignalType.HOLD:
                    continue
                price = current_prices.get(code, 0.0)
                if price <= 0:
                    continue
                all_signals.append(TradingSignal(
                    timestamp=current_date,
                    stock_code=code,
                    signal_type=sig_type,
                    strength=strength,
                    price=price,
                    reason="precomputed",
                    metadata={},
                ))

        total_signals += len(all_signals)
        trades_this_day = 0

        # ── 执行交易 ──
        if is_topk_buffer and all_signals:
            trades_this_day = _execute_topk_buffer(
                strategy, portfolio, all_signals,
                current_prices, current_date,
                k_limit, buffer_size, max_changes, min_buy_score,
            )
        else:
            for signal in all_signals:
                is_valid, _ = strategy.validate_signal(
                    signal,
                    portfolio.get_portfolio_value(current_prices),
                    portfolio.positions,
                )
                if is_valid:
                    trade, _ = portfolio.execute_signal(
                        signal, current_prices
                    )
                    if trade:
                        trades_this_day += 1

        executed_trades += trades_this_day

        # ── 记录快照 ──
        portfolio.record_portfolio_snapshot(
            current_date, current_prices
        )

        # ── 发送进��� ──
        if i % progress_interval == 0 or i == len(trading_dates) - 1:
            pv = portfolio.get_portfolio_value(current_prices)
            _send_progress(
                progress_queue, worker_id,
                i + 1, len(trading_dates),
                current_date.strftime("%Y-%m-%d"),
                pv, total_signals, executed_trades,
            )

    result = {
        "total_signals": total_signals,
        "executed_trades": executed_trades,
    }
    if circuit_breaker_triggered:
        result["circuit_breaker_summary"] = {
            "triggered": True,
            "trigger_date": (
                circuit_breaker_date.strftime("%Y-%m-%d")
                if circuit_breaker_date else None
            ),
        }
    return result


def _check_stop_loss_take_profit(
    risk_manager, portfolio, current_prices, current_date,
) -> int:
    """检查并执行止损止盈"""
    from ..core.risk_manager import PositionPriceInfo

    positions = portfolio.positions
    if not positions:
        return 0

    positions_info = {}
    for code, pos in positions.items():
        price = current_prices.get(code)
        if price and price > 0:
            positions_info[code] = PositionPriceInfo(
                stock_code=code,
                quantity=pos.quantity,
                avg_cost=pos.avg_cost,
                current_price=price,
            )

    if not positions_info:
        return 0

    sl_tp_signals = risk_manager.check_stop_loss_take_profit(
        positions_info
    )
    trades_count = 0
    for signal in sl_tp_signals:
        trade, _ = portfolio.execute_signal(signal, current_prices)
        if trade:
            trades_count += 1
    return trades_count


def _execute_topk_buffer(
    strategy, portfolio, signals, current_prices,
    current_date, k_limit, buffer_size, max_changes,
    min_buy_score,
) -> int:
    """topk_buffer 交易模式"""
    # 按信号强度排序
    buy_signals = sorted(
        [s for s in signals if s.signal_type == SignalType.BUY],
        key=lambda s: s.strength,
        reverse=True,
    )
    sell_signals = [
        s for s in signals if s.signal_type == SignalType.SELL
    ]

    trades_count = 0
    changes = 0

    # 先执行卖出
    for signal in sell_signals:
        if changes >= max_changes:
            break
        is_valid, _ = strategy.validate_signal(
            signal,
            portfolio.get_portfolio_value(current_prices),
            portfolio.positions,
        )
        if is_valid:
            trade, _ = portfolio.execute_signal(
                signal, current_prices
            )
            if trade:
                trades_count += 1
                changes += 1

    # 再执行买入（受 topk 限制）
    current_holdings = set(
        code for code, pos in portfolio.positions.items()
        if pos.quantity > 0
    )
    for signal in buy_signals:
        if changes >= max_changes:
            break
        if len(current_holdings) >= k_limit:
            break
        if signal.strength < min_buy_score:
            continue
        is_valid, _ = strategy.validate_signal(
            signal,
            portfolio.get_portfolio_value(current_prices),
            portfolio.positions,
        )
        if is_valid:
            trade, _ = portfolio.execute_signal(
                signal, current_prices
            )
            if trade:
                trades_count += 1
                changes += 1
                current_holdings.add(signal.stock_code)

    return trades_count
