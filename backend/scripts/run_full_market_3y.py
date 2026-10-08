#!/usr/bin/env python
"""全量股票 × 最近3年 回测（multi_factor）。

- Universe: data/qlib_data/features/day/*.parquet 全部（≈5513 只，沪深北全市场）
- 窗口: 2023-05-12 → 2026-05-08（本地数据的最近3年，数据止于 2026-05-08）
- 直接驱动 BacktestExecutor（与 backend/scripts/bench_backtest_500_3y.py 同模式），
  绕过 HTTP 路由里硬编码的 1000 只上限；参数与 POST /api/v1/backtest 对齐。
"""

from __future__ import annotations

import asyncio
import glob
import json
import os
import sys
import time
from datetime import datetime

BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "stock-platform", "backend"))
if not os.path.isdir(BACKEND_ROOT):
    BACKEND_ROOT = "/Users/rone/Projects/stock-platform/backend"
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

from app.core.config import settings  # noqa: E402
from app.services.backtest import BacktestExecutor  # noqa: E402
from app.services.backtest.models import BacktestConfig  # noqa: E402

OUT_JSON = "/Users/rone/.hermes/cache/scratch/bt_full_executor_resp.json"
START = datetime(2023, 5, 12)
END = datetime(2026, 5, 8)


def list_universe() -> list[str]:
    files = glob.glob(os.path.join(str(settings.DATA_ROOT_PATH), "qlib_data", "features", "day", "*.parquet"))
    codes = sorted(os.path.basename(f)[: -len(".parquet")].replace("_", ".") for f in files)
    return codes


async def main() -> None:
    codes = list_universe()
    print(f"universe={len(codes)} window={START.date()}..{END.date()} strategy=multi_factor", flush=True)

    executor = BacktestExecutor(
        data_dir=str(settings.DATA_ROOT_PATH),
        enable_parallel=True,
        max_workers=8,
        enable_performance_profiling=False,
        use_multiprocessing=False,
    )

    t0 = time.perf_counter()
    res = await executor.run_backtest(
        strategy_name="multi_factor",
        stock_codes=codes,
        start_date=START,
        end_date=END,
        strategy_config={},
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

    # 收敛 numpy 类型，便于 json 落盘
    def _conv(o):
        import numpy as np

        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
        if isinstance(o, (datetime,)):
            return o.isoformat()
        raise TypeError(str(type(o)))

    res["__wall_seconds__"] = wall
    with open(OUT_JSON, "w") as f:
        json.dump(res, f, default=_conv, ensure_ascii=False)

    perf = res.get("perf_breakdown", {}) or {}
    print(f"wall={wall:.1f}s perf={json.dumps(perf, default=_conv)}", flush=True)
    for key in ("portfolio", "risk_metrics", "trading_stats", "strategy_name", "period"):
        if key in res:
            print(f"{key}: {json.dumps(res[key], default=_conv, ensure_ascii=False)[:500]}", flush=True)
    print(f"saved -> {OUT_JSON}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
