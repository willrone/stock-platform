#!/usr/bin/env python
"""Small profiling run for BacktestExecutor.

Goal: capture BacktestPerformanceProfiler stage timings + function call stats.

Usage:
  cd backend
  ./venv/bin/python scripts/profile_backtest_small.py

This script is intentionally small and safe: universe is tiny, date range short.
"""

from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime

# Ensure `import app...` works when running as a script
BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

from app.core.config import settings
from app.services.backtest import BacktestExecutor


async def main():
    executor = BacktestExecutor(
        data_dir=str(settings.DATA_ROOT_PATH),
        enable_parallel=True,
        enable_performance_profiling=True,
    )

    # Minimal run (adjust later)
    stock_codes = [
        "000001.SZ",
        "000002.SZ",
        "000333.SZ",
        "600000.SH",
        "600519.SH",
    ]

    res = await executor.run_backtest(
        strategy_name="portfolio",
        stock_codes=stock_codes,
        start_date=datetime.fromisoformat("2020-01-01"),
        end_date=datetime.fromisoformat("2020-06-30"),
        strategy_config={
            "integration_method": "weighted_voting",
            "trade_mode": "topk_buffer",
            "topk": 2,
            "buffer": 3,
            "max_changes_per_day": 1,
            "strategies": [
                {"name": "rsi", "weight": 0.5, "config": {"rsi_period": 14, "oversold_threshold": 30, "overbought_threshold": 70}},
                {"name": "stochastic", "weight": 0.5, "config": {"k_period": 14, "d_period": 3, "oversold": 20, "overbought": 80}},
            ],
        },
    )

    # BacktestExecutor already logs stage timings via profiler; print a compact summary too.
    profiler = getattr(executor, "performance_profiler", None)
    if profiler:
        print("\n=== PROFILER SUMMARY (stages) ===")
        for name, st in profiler.stages.items():
            if st.end_time is None:
                continue
            print(f"{name}: {st.duration:.3f}s mem_after={st.memory_after:.1f}MB cpu_avg={st.cpu_avg:.1f}%")

        top_funcs = sorted(
            profiler.function_calls.values(), key=lambda x: x.total_time, reverse=True
        )[:15]
        print("\n=== TOP FUNCTION CALLS (by total_time) ===")
        for fs in top_funcs:
            print(
                f"{fs.name}: calls={fs.call_count} total={fs.total_time:.3f}s avg={fs.avg_time:.6f}s max={fs.max_time:.6f}s"
            )

    print("\n=== METRICS ===")
    print(res.get("metrics", {}))


if __name__ == "__main__":
    asyncio.run(main())
