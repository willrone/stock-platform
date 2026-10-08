#!/usr/bin/env python
"""B 对照: portfolio 组合策略 (multi_factor 0.5 + low_volatility 0.3 + value_factor 0.2) 全市场。"""
from __future__ import annotations

import asyncio
import glob
import json
import os
import sys
import time
from datetime import datetime

BACKEND_ROOT = "/Users/rone/Projects/stock-platform/backend"
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

from app.core.config import settings  # noqa: E402
from app.services.backtest import BacktestExecutor  # noqa: E402
from app.services.backtest.models import BacktestConfig  # noqa: E402

START = datetime(2023, 5, 12)
END = datetime(2026, 5, 8)
OUT = "/Users/rone/.hermes/cache/scratch/bt_portfolio_resp.json"


def universe():
    fs = glob.glob(os.path.join(str(settings.DATA_ROOT_PATH), "qlib_data", "features", "day", "*.parquet"))
    return sorted(os.path.basename(f)[: -len(".parquet")].replace("_", ".") for f in fs)


async def main():
    cfg = {
        "strategies": [
            {"name": "multi_factor", "weight": 0.5, "config": {}},
            {"name": "low_volatility", "weight": 0.3, "config": {}},
            {"name": "value_factor", "weight": 0.2, "config": {}},
        ],
        "integration_method": "weighted_voting",
    }
    executor = BacktestExecutor(
        data_dir=str(settings.DATA_ROOT_PATH),
        enable_parallel=True,
        max_workers=8,
        enable_performance_profiling=False,
        use_multiprocessing=False,
    )
    t0 = time.perf_counter()
    res = await executor.run_backtest(
        strategy_name="portfolio",
        stock_codes=universe(),
        start_date=START,
        end_date=END,
        strategy_config=cfg,
        backtest_config=BacktestConfig(
            initial_cash=100000.0,
            commission_rate=0.0003,
            slippage_rate=0.0001,
            max_position_size=0.2,
            cash_reserve_ratio=0.05,
            board_lot_size=100,
            stop_loss_pct=0.05,
            take_profit_pct=0.15,
            rebalance_frequency="daily",
            open_cost=0.0,
            close_cost=0.0,
            min_cost=0.0,
        ),
    )
    wall = time.perf_counter() - t0

    def _conv(o):
        import numpy as np

        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, datetime):
            return o.isoformat()
        raise TypeError(str(type(o)))

    res["__wall_seconds__"] = wall
    with open(OUT, "w") as f:
        json.dump(res, f, default=_conv, ensure_ascii=False)

    row = {k: res.get(k) for k in [
        "total_return", "annualized_return", "volatility", "sharpe_ratio",
        "max_drawdown", "total_trades", "win_rate", "profit_factor", "stocks_traded"]}
    row["wall_s"] = round(wall, 1)
    ph = res.get("portfolio_history") or []
    if ph:
        row["end_positions"] = ph[-1].get("positions_count")
        row["end_cash"] = round(ph[-1].get("cash", 0), 0)
    print("[portfolio] " + json.dumps(row, ensure_ascii=False, default=float), flush=True)
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
