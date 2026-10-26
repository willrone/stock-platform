#!/usr/bin/env python3
"""校验：我的 IS/OOS 切分口径 vs 引擎自己报的 sharpe/return/mdd。

跑一次完整 3 年真实盘回测，然后：
1. 用 equity_curve 全段算 sharpe → 应等于引擎的 sharpe_ratio
2. 用 equity_curve 全段算 return → 应等于引擎的 total_return
3. 打印 IS / OOS 切分结果
"""
from __future__ import annotations

import asyncio
import json
import os
import resource
import sys
import time
from datetime import datetime

BACKEND = "/Users/rone/Projects/stock-platform/backend"
ROOT = "/Users/rone/Projects/stock-platform"
sys.path.insert(0, BACKEND)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.join(BACKEND, "scripts"))
from search_gbm_params import (  # noqa: E402
    _metrics_from_equity, _split_metrics, _yearly_from_equity,
    universe_from_instruments, QLIB_OFFICIAL,
)
from app.services.backtest import BacktestExecutor  # noqa: E402
from app.services.backtest.models import BacktestConfig  # noqa: E402
from app.services.backtest.strategies.strategy_factory import (  # noqa: E402
    StrategyFactory,
)
from app.services.backtest.strategies.gbm_lab_strategy import (  # noqa: E402
    GbmLabStrategy,
)

StrategyFactory.register_strategy("gbm_lab", GbmLabStrategy)

CFG = {
    "topk": 60, "n_drop": 5, "rebalance_days": "weekly",
    "rebalance_weekday": 4, "horizon": 1, "label_anchor": "trade",
    "target": "cs_z", "model": "lgbm", "n_estimators": 100, "max_depth": 3,
    "learning_rate": 0.05, "retrain": "monthly", "train_window": "expand",
    "warmup_months": 2, "feature_set": "f42", "use_presence": False,
    "target_vol": 0.20, "blend": 1.0, "purge": True, "n_jobs": 2, "lag": 1,
}


def main():
    codes = universe_from_instruments(QLIB_OFFICIAL)
    print(f"[validate] 股票池 {len(codes)} 只", flush=True)
    executor = BacktestExecutor(
        data_dir=ROOT + "/data", enable_parallel=True, max_workers=8,
        enable_performance_profiling=False, use_multiprocessing=False,
    )
    t0 = time.perf_counter()
    res = asyncio.run(executor.run_backtest(
        strategy_name="gbm_lab", stock_codes=codes,
        start_date=datetime(2023, 5, 8), end_date=datetime(2026, 5, 8),
        strategy_config=CFG,
        backtest_config=BacktestConfig(
            enable_tradability=True, initial_cash=1_000_000.0,
            commission_rate=0.0003, slippage_rate=0.0001,
            max_position_size=0.015833, cash_reserve_ratio=0.05,
            board_lot_size=100, stop_loss_pct=0.0, take_profit_pct=0.0,
            rebalance_frequency="daily", open_cost=0.0, close_cost=0.0,
            min_cost=0.0,
        ),
    ))
    wall = time.perf_counter() - t0
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
    print(f"\n[validate] wall={wall:.1f}s  peak_rss={rss:.2f} GB", flush=True)

    curve = res.get("equity_curve") or []
    trades = res.get("trade_history") or []
    mine = _metrics_from_equity(curve)
    print(f"\n[validate] equity_curve 点数={len(curve)}")
    print(f"  引擎 sharpe_ratio = {res.get('sharpe_ratio'):.10f}")
    print(f"  我的 sharpe      = {mine['sharpe']:.10f}   "
          f"差={abs(mine['sharpe'] - res.get('sharpe_ratio')):.2e}")
    print(f"  引擎 total_return= {res.get('total_return'):.10f}")
    print(f"  我的 return      = {mine['return']:.10f}   "
          f"差={abs(mine['return'] - res.get('total_return')):.2e}")
    print(f"  引擎 max_drawdown= {res.get('max_drawdown'):.10f}")
    print(f"  我的 mdd         = {mine['mdd']:.10f}   "
          f"差={abs(mine['mdd'] - res.get('max_drawdown')):.2e}")
    print(f"  引擎 total_trades= {res.get('total_trades')}  我的={mine['days']}")

    seg = _split_metrics(curve, trades)
    print(f"\n[validate] IS/OOS 切分:")
    for k in ("is", "oos", "full"):
        m = seg[k]
        print(f"  {k:5s} sharpe={m['sharpe']:.4f} ret={m['return']:.3%} "
              f"mdd={m['mdd']:.3%} trades={m['trades']} win={m['win_rate']:.2%} "
              f"days={m['days']}")
    print(f"\n[validate] 分年收益: {_yearly_from_equity(curve)}")
    print(f"[validate] 引擎 yearly: "
          f"{[x for x in (res.get('yearly_returns_detail') or [])]}")


if __name__ == "__main__":
    main()
