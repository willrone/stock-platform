#!/usr/bin/env python
"""Profile backtest with a larger universe.

Usage:
  cd backend
  ./venv/bin/python scripts/profile_backtest_universe.py --universe-size 50 --start 2020-01-01 --end 2020-12-31

Notes:
- Uses topk_buffer trade_mode.
- Samples codes from qlib feature dir if available, else parquet dir.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import random
import sys
from datetime import datetime

BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

from app.core.config import settings
from app.services.backtest import BacktestExecutor


def _list_codes() -> list[str]:
    data_root = str(settings.DATA_ROOT_PATH)
    qlib_day_dir = os.path.join(data_root, "qlib_data", "features", "day")
    codes: list[str] = []
    if os.path.isdir(qlib_day_dir):
        for name in os.listdir(qlib_day_dir):
            if name.endswith(".parquet"):
                codes.append(name.replace(".parquet", "").replace("_", "."))
    else:
        stock_parquet_dir = os.path.join(data_root, "parquet", "stock_data")
        if os.path.isdir(stock_parquet_dir):
            for name in os.listdir(stock_parquet_dir):
                if name.endswith(".parquet"):
                    codes.append(name.replace(".parquet", "").replace("_", "."))
    return sorted(set([c for c in codes if c]))


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe-size", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--start", type=str, default="2020-01-01")
    ap.add_argument("--end", type=str, default="2020-12-31")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--buffer", type=int, default=20)
    ap.add_argument("--max-changes-per-day", type=int, default=2)
    args = ap.parse_args()

    all_codes = _list_codes()
    if len(all_codes) < args.universe_size:
        raise SystemExit(f"Not enough codes: have={len(all_codes)} need={args.universe_size}")

    rng = random.Random(args.seed)
    stock_codes = rng.sample(all_codes, args.universe_size)

    executor = BacktestExecutor(
        data_dir=str(settings.DATA_ROOT_PATH),
        enable_parallel=True,
        enable_performance_profiling=True,
    )

    res = await executor.run_backtest(
        strategy_name="portfolio",
        stock_codes=stock_codes,
        start_date=datetime.fromisoformat(args.start),
        end_date=datetime.fromisoformat(args.end),
        strategy_config={
            "integration_method": "weighted_voting",
            "trade_mode": "topk_buffer",
            "topk": int(args.topk),
            "buffer": int(args.buffer),
            "max_changes_per_day": int(args.max_changes_per_day),
            "strategies": [
                {"name": "rsi", "weight": 0.5, "config": {"rsi_period": 14, "oversold_threshold": 30, "overbought_threshold": 70}},
                {"name": "multi_factor", "weight": 0.5, "config": {}},
            ],
        },
    )

    profiler = getattr(executor, "performance_profiler", None)
    if profiler:
        top_funcs = sorted(
            profiler.function_calls.values(), key=lambda x: x.total_time, reverse=True
        )[:20]
        print("\n=== TOP FUNCTION CALLS (by total_time) ===")
        for fs in top_funcs:
            print(
                f"{fs.name}: calls={fs.call_count} total={fs.total_time:.3f}s avg={fs.avg_time:.6f}s max={fs.max_time:.6f}s"
            )

    print("\n=== METRICS ===")
    print(res.get("metrics", {}))


if __name__ == "__main__":
    asyncio.run(main())
