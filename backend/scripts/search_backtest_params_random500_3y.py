#!/usr/bin/env python3
"""Random-search backtest parameters on a random universe.

Goal:
- Sample ~500 stocks (filtered for data coverage)
- Use most recent ~3 years of data available locally
- Try multiple parameter combinations for the portfolio strategy (topk_buffer)
- Report the best-performing parameter set.

This script is intentionally standalone (no changes to core backtest code).

Usage:
  cd backend
  ./venv/bin/python ../scripts/search_backtest_params_random500_3y.py --trials 20

Notes:
- If <500 valid tickers, it uses as many as available.
- Objective uses a simple composite score (Sharpe + 0.5*annualized_return - 0.5*|max_drawdown|).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

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


def _is_data_valid(df, start: datetime, end: datetime, *, min_rows: int = 200, min_coverage: float = 0.7) -> bool:
    try:
        n = int(len(df))
    except Exception:
        return False
    if n < int(min_rows):
        return False

    expected = max(1, int(((end - start).days + 1) * 5 / 7))
    coverage = n / expected
    return coverage >= float(min_coverage)


def _latest_date_from_samples(executor: BacktestExecutor, codes: list[str], *, sample_n: int = 30) -> datetime:
    # Try a small sample to find the max available trading date.
    sample = codes[:]
    random.shuffle(sample)
    sample = sample[: min(len(sample), sample_n)]
    end_candidates: list[datetime] = []
    for c in sample:
        try:
            df = executor.data_loader.load_single_stock(c, None, None)
            if df is None or len(df) == 0:
                continue
            mx = getattr(df.index, "max", lambda: None)()
            if mx is None:
                continue
            # index may be Timestamp
            try:
                mx = mx.to_pydatetime()
            except Exception:
                pass
            if isinstance(mx, datetime):
                end_candidates.append(mx)
        except Exception:
            continue

    if end_candidates:
        return max(end_candidates)

    # fallback: today
    return datetime.now()


@dataclass
class TrialResult:
    trial: int
    params: dict[str, Any]
    metrics: dict[str, Any]
    score: float
    wall_s: float


def _score(metrics: dict[str, Any]) -> float:
    # Composite score: encourage Sharpe and return, penalize drawdown.
    sharpe = float(metrics.get("sharpe_ratio", 0.0) or 0.0)
    ann = float(metrics.get("annualized_return", 0.0) or 0.0)
    mdd = float(metrics.get("max_drawdown", 0.0) or 0.0)
    return sharpe + 0.5 * ann - 0.5 * abs(mdd)


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--universe", type=int, default=500)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--max-workers", type=int, default=8)

    # Universe validity filter
    ap.add_argument("--min-rows", type=int, default=200)
    ap.add_argument("--min-coverage", type=float, default=0.7)
    ap.add_argument("--oversample", type=int, default=3)

    # Date range override (optional)
    ap.add_argument("--end", type=str, default="")
    ap.add_argument("--years", type=float, default=3.0)

    # Search space knobs
    ap.add_argument("--topk-min", type=int, default=10)
    ap.add_argument("--topk-max", type=int, default=50)
    ap.add_argument("--buffer-min", type=int, default=0)
    ap.add_argument("--buffer-max", type=int, default=50)
    ap.add_argument("--max-changes-min", type=int, default=1)
    ap.add_argument("--max-changes-max", type=int, default=10)

    args = ap.parse_args()

    rng = random.Random(args.seed)

    # Base executor for data loading
    base_exec = BacktestExecutor(
        data_dir=str(settings.DATA_ROOT_PATH),
        enable_parallel=True,
        enable_performance_profiling=False,
        max_workers=int(args.max_workers),
    )

    all_codes = _list_codes()
    if not all_codes:
        raise SystemExit("No stock codes found under DATA_ROOT_PATH")

    if args.end:
        end_dt = datetime.fromisoformat(args.end)
    else:
        end_dt = _latest_date_from_samples(base_exec, all_codes)

    start_dt = end_dt - timedelta(days=int(365.25 * float(args.years)))

    # Oversample then filter for valid data
    want = int(args.universe)
    pool = all_codes[:]
    rng.shuffle(pool)
    pool = pool[: min(len(pool), want * int(args.oversample))]

    stock_data = base_exec.data_loader.load_multiple_stocks(pool, start_dt, end_dt)
    valid_codes: list[str] = []
    for code, df in stock_data.items():
        if _is_data_valid(df, start_dt, end_dt, min_rows=args.min_rows, min_coverage=args.min_coverage):
            valid_codes.append(code)

    rng.shuffle(valid_codes)
    selected = valid_codes[: min(len(valid_codes), want)]

    print(f"[universe] total_codes={len(all_codes)} pool={len(pool)} loaded={len(stock_data)} valid={len(valid_codes)} selected={len(selected)}")
    print(f"[dates] start={start_dt.date()} end={end_dt.date()} (~{args.years}y)")

    if len(selected) < 50:
        raise SystemExit(f"Too few valid tickers to run meaningful search: {len(selected)}")

    results: list[TrialResult] = []

    for t in range(1, int(args.trials) + 1):
        topk = rng.randint(int(args.topk_min), int(args.topk_max))
        buffer_n = rng.randint(int(args.buffer_min), int(args.buffer_max))
        max_changes = rng.randint(int(args.max_changes_min), int(args.max_changes_max))

        # Ensure buffer not smaller than topk in an odd way (allow any, but keep reasonable)
        if buffer_n < 0:
            buffer_n = 0

        executor = BacktestExecutor(
            data_dir=str(settings.DATA_ROOT_PATH),
            enable_parallel=True,
            enable_performance_profiling=False,
            max_workers=int(args.max_workers),
        )

        import time

        t0 = time.perf_counter()
        res = await executor.run_backtest(
            strategy_name="portfolio",
            stock_codes=selected,
            start_date=start_dt,
            end_date=end_dt,
            strategy_config={
                "integration_method": "weighted_voting",
                "trade_mode": "topk_buffer",
                "topk": int(topk),
                "buffer": int(buffer_n),
                "max_changes_per_day": int(max_changes),
                "strategies": [
                    {
                        "name": "rsi",
                        "weight": 0.5,
                        "config": {
                            "rsi_period": 14,
                            "oversold_threshold": 30,
                            "overbought_threshold": 70,
                        },
                    },
                    {"name": "multi_factor", "weight": 0.5, "config": {}},
                ],
            },
        )
        wall = time.perf_counter() - t0

        metrics = (res or {}).get("metrics", {}) or {}
        sc = _score(metrics)
        tr = TrialResult(
            trial=t,
            params={"topk": topk, "buffer": buffer_n, "max_changes_per_day": max_changes},
            metrics=metrics,
            score=sc,
            wall_s=float(wall),
        )
        results.append(tr)

        print(
            f"[trial {t:02d}/{args.trials}] score={sc:.4f} wall={wall:.1f}s "
            f"topk={topk} buffer={buffer_n} maxchg={max_changes} "
            f"sharpe={float(metrics.get('sharpe_ratio', 0.0) or 0.0):.3f} "
            f"ann={float(metrics.get('annualized_return', 0.0) or 0.0):.3f} "
            f"mdd={float(metrics.get('max_drawdown', 0.0) or 0.0):.3f} "
            f"trades={int(metrics.get('total_trades', 0) or 0)}"
        )

    results.sort(key=lambda x: x.score, reverse=True)

    print("\n=== TOP 5 PARAM SETS (by composite score) ===")
    for r in results[:5]:
        m = r.metrics
        print(
            f"trial={r.trial:02d} score={r.score:.4f} wall={r.wall_s:.1f}s params={r.params} "
            f"metrics={{sharpe={m.get('sharpe_ratio')}, ann={m.get('annualized_return')}, mdd={m.get('max_drawdown')}, total_return={m.get('total_return')}, trades={m.get('total_trades')}}}"
        )

    best = results[0]
    print("\n=== BEST ===")
    print(best.params)
    print(best.metrics)

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
