#!/usr/bin/env python
"""对照实验：V1 去掉紧止损（multi_factor）；V2 momentum_factor 默认参数。"""
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


def universe():
    fs = glob.glob(os.path.join(str(settings.DATA_ROOT_PATH), "qlib_data", "features", "day", "*.parquet"))
    return sorted(os.path.basename(f)[: -len(".parquet")].replace("_", ".") for f in fs)


VARIANTS = [
    ("v1_nostop", "multi_factor", {}, dict(stop_loss_pct=1.0, take_profit_pct=10.0)),
    ("v2_momentum", "momentum_factor", {}, dict()),
]


async def run_one(tag, strat, strat_cfg, cfg_over):
    executor = BacktestExecutor(
        data_dir=str(settings.DATA_ROOT_PATH),
        enable_parallel=True,
        max_workers=8,
        enable_performance_profiling=False,
        use_multiprocessing=False,
    )
    base = dict(
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
    )
    base.update(cfg_over)
    t0 = time.perf_counter()
    res = await executor.run_backtest(
        strategy_name=strat,
        stock_codes=universe(),
        start_date=START,
        end_date=END,
        strategy_config=strat_cfg,
        backtest_config=BacktestConfig(**base),
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
    out = f"/Users/rone/.hermes/cache/scratch/bt_{tag}_resp.json"
    with open(out, "w") as f:
        json.dump(res, f, default=_conv, ensure_ascii=False)

    keys = ["total_return", "annualized_return", "volatility", "sharpe_ratio",
            "max_drawdown", "total_trades", "win_rate", "profit_factor", "stocks_traded"]
    row = {k: res.get(k) for k in keys}
    row["wall_s"] = round(wall, 1)
    ph = res.get("portfolio_history") or []
    if ph:
        row["end_positions"] = ph[-1].get("positions_count")
        row["end_cash"] = round(ph[-1].get("cash", 0), 0)
    print(f"[{tag}] {json.dumps(row, ensure_ascii=False)}", flush=True)
    return row


async def main():
    for tag, strat, scfg, cfg in VARIANTS:
        print(f"=== start {tag} ({strat}) ===", flush=True)
        await run_one(tag, strat, scfg, cfg)
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
