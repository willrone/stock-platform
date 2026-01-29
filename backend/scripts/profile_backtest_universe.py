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


def _is_data_valid(df, start: datetime, end: datetime, *, min_rows: int = 30, min_coverage: float = 0.7) -> bool:
    try:
        n = int(len(df))
    except Exception:
        return False
    if n < int(min_rows):
        return False

    # rough expected trading days (~5/7 of calendar days)
    expected = max(1, int(((end - start).days + 1) * 5 / 7))
    coverage = n / expected
    return coverage >= float(min_coverage)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe-size", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--start", type=str, default="2020-01-01")
    ap.add_argument("--end", type=str, default="2020-12-31")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--buffer", type=int, default=20)
    ap.add_argument("--max-changes-per-day", type=int, default=2)

    # universe validity filter (defaults match DataLoader heuristic)
    ap.add_argument("--min-rows", type=int, default=30)
    ap.add_argument("--min-coverage", type=float, default=0.7)
    ap.add_argument("--oversample", type=int, default=5, help="sample N*universe first, then filter")

    args = ap.parse_args()

    all_codes = _list_codes()

    start_dt = datetime.fromisoformat(args.start)
    end_dt = datetime.fromisoformat(args.end)

    rng = random.Random(args.seed)
    pool_size = min(len(all_codes), int(args.universe_size) * int(args.oversample))
    if pool_size < args.universe_size:
        raise SystemExit(f"Not enough codes: have={len(all_codes)} need={args.universe_size}")

    candidate_codes = rng.sample(all_codes, pool_size)

    executor = BacktestExecutor(
        data_dir=str(settings.DATA_ROOT_PATH),
        enable_parallel=True,
        enable_performance_profiling=True,
    )

    # Load data once (DataLoader already does its own filtering, but we pre-filter universe here
    # to avoid lots of missing/short-history tickers skewing profiling).
    stock_data = executor.data_loader.load_multiple_stocks(candidate_codes, start_dt, end_dt)

    valid_codes: list[str] = []
    for code, df in stock_data.items():
        if _is_data_valid(df, start_dt, end_dt, min_rows=args.min_rows, min_coverage=args.min_coverage):
            valid_codes.append(code)

    valid_codes = sorted(set(valid_codes))
    if len(valid_codes) < args.universe_size:
        raise SystemExit(
            f"Not enough valid codes after filter: valid={len(valid_codes)} need={args.universe_size} "
            f"(min_rows={args.min_rows} min_coverage={args.min_coverage} pool={pool_size})"
        )

    stock_codes = rng.sample(valid_codes, args.universe_size)
    print(
        f"[universe] candidates={pool_size} loaded={len(stock_data)} valid={len(valid_codes)} selected={len(stock_codes)}"
    )

    res = await executor.run_backtest(
        strategy_name="portfolio",
        stock_codes=stock_codes,
        start_date=start_dt,
        end_date=end_dt,
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
