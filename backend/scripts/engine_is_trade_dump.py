#!/usr/bin/env python3
"""引擎 vs 模拟器：分窗口（IS/OOS）逐笔对账。

先确认一个前提：**引擎分段跑**（IS、OOS 各起一次）与**全窗跑一次再切片**
不是同一件事 —— 分段跑时每段都从 100 万现金、空仓、因子冷启动开始。
本脚本用真实引擎跑 IS 段并 dump 逐笔成交，与模拟器的 IS 段逐笔比对，
定位差异到底来自「因子冷启动」还是「撮合语义」。

用法:
  env -u PYTHONPATH -u PYTHONHOME backend/.venv-py313/bin/python3 \
    backend/scripts/engine_is_trade_dump.py \
      --legs 42f mf --normalize winsor --topk 30 --n-drop 5 --out /tmp/ens_is_trades.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(BACKEND)
sys.path.insert(0, BACKEND)
sys.path.insert(0, os.path.join(BACKEND, "scripts"))

import ensemble_lab as L  # noqa: E402

IS_START, IS_END = datetime(2023, 5, 8), datetime(2025, 5, 8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--legs", nargs="*", default=["42f", "mf"])
    ap.add_argument("--normalize", default="winsor")
    ap.add_argument("--topk", type=int, default=30)
    ap.add_argument("--n-drop", type=int, default=5)
    ap.add_argument("--out", default="/tmp/ens_is_trades.json")
    args = ap.parse_args()

    from app.services.backtest import BacktestExecutor
    from app.services.backtest.models import BacktestConfig
    from app.services.backtest.strategies.ensemble_strategy import EnsembleStrategy
    from app.services.backtest.strategies.strategy_factory import StrategyFactory
    import app.services.backtest.core.portfolio_manager_array as pma

    StrategyFactory.register_strategy("ensemble", EnsembleStrategy)

    trades: List[Dict[str, Any]] = []
    orig = pma.PortfolioManagerArray.execute_signal

    def patched(self, signal, current_prices, *a, **kw):
        out = orig(self, signal, current_prices, *a, **kw)
        trade = out[0] if isinstance(out, tuple) else out
        if trade is not None:
            trades.append({
                "date": str(signal.timestamp)[:10],
                "code": signal.stock_code,
                "action": signal.signal_type.name,
                "price": float(getattr(trade, "price", 0) or 0),
                "qty": float(getattr(trade, "quantity", 0) or 0),
            })
        return out

    pma.PortfolioManagerArray.execute_signal = patched
    codes = L.universe_from_instruments()
    ex = BacktestExecutor(data_dir=ROOT + "/data", enable_parallel=True,
                          max_workers=8, enable_performance_profiling=False,
                          use_multiprocessing=False)
    t0 = time.perf_counter()
    res = asyncio.run(ex.run_backtest(
        strategy_name="ensemble", stock_codes=codes,
        start_date=IS_START, end_date=IS_END,
        strategy_config={"topk": args.topk, "n_drop": args.n_drop,
                         "rebalance_days": "weekly", "rebalance_weekday": 4,
                         "lag": 1, "legs": args.legs,
                         "normalize": args.normalize, "weight_mode": "fixed"},
        backtest_config=BacktestConfig(
            enable_tradability=True, initial_cash=1_000_000.0,
            commission_rate=0.0003, slippage_rate=0.0001,
            max_position_size=(1 - 0.05) / args.topk, cash_reserve_ratio=0.05,
            board_lot_size=100, stop_loss_pct=0.0, take_profit_pct=0.0,
            rebalance_frequency="daily", open_cost=0.0, close_cost=0.0,
            min_cost=0.0),
    ))
    pma.PortfolioManagerArray.execute_signal = orig
    out = {"metrics": {k: res.get(k) for k in
                       ("sharpe_ratio", "total_return", "max_drawdown",
                        "total_trades")},
           "trades": trades}
    json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=1, default=str)
    print(f"wrote {args.out}: {len(trades)} trades, SR={res.get('sharpe_ratio')}, "
          f"ret={res.get('total_return')}, n={res.get('total_trades')} "
          f"({time.perf_counter()-t0:.0f}s)")


if __name__ == "__main__":
    main()
