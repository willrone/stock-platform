#!/usr/bin/env python
"""C 方案: factor_ranking (横截面复合因子 + TopK/Dropout) 全市场回测。

用法:
  python run_bt_factor_ranking.py            # 全市场 5513
  python run_bt_factor_ranking.py 300        # 冒烟: 前300只
"""
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
TOPK = 50
N_DROP = 5


def universe(limit: int | None = None):
    fs = glob.glob(
        os.path.join(str(settings.DATA_ROOT_PATH), "qlib_data", "features", "day", "*.parquet")
    )
    codes = sorted(os.path.basename(f)[: -len(".parquet")].replace("_", ".") for f in fs)
    return codes[:limit] if limit else codes


async def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    tag = f"smoke{limit}" if limit else "full"
    out = f"/Users/rone/.hermes/cache/scratch/bt_factor_ranking_{tag}_resp.json"
    codes = universe(limit)
    print(f"[{tag}] universe={len(codes)} topk={TOPK} n_drop={N_DROP}", flush=True)

    executor = BacktestExecutor(
        data_dir=str(settings.DATA_ROOT_PATH),
        enable_parallel=True,
        max_workers=8,
        enable_performance_profiling=False,
        use_multiprocessing=False,
    )
    t0 = time.perf_counter()
    res = await executor.run_backtest(
        strategy_name="factor_ranking",
        stock_codes=codes,
        start_date=START,
        end_date=END,
        strategy_config={"topk": TOPK, "n_drop": N_DROP, "hold_thresh": 0},
        backtest_config=BacktestConfig(
            initial_cash=1000000.0,   # 100万: topk=50 等权后单仓约2万, 规避100股手数买不起高价股
            commission_rate=0.0003,
            slippage_rate=0.0001,
            max_position_size=0.019,  # 50 × 1.9% ≈ 95% 仓位, 留5%现金缓冲
            cash_reserve_ratio=0.05,
            board_lot_size=100,
            stop_loss_pct=1.0,        # TopK 范式由排名驱动换仓, 关掉逐股止盈止损
            take_profit_pct=10.0,
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
    with open(out, "w") as f:
        json.dump(res, f, default=_conv, ensure_ascii=False)

    row = {
        k: res.get(k)
        for k in [
            "total_return", "annualized_return", "volatility", "sharpe_ratio",
            "max_drawdown", "total_trades", "win_rate", "profit_factor", "stocks_traded",
            "total_signals",
        ]
    }
    row["wall_s"] = round(wall, 1)
    ph = res.get("portfolio_history") or []
    if ph:
        pcs = [p.get("positions_count", 0) for p in ph]
        row["end_positions"] = ph[-1].get("positions_count")
        row["min_positions"] = min(pcs)
        row["max_positions"] = max(pcs)
        row["end_cash"] = round(ph[-1].get("cash", 0), 0)
    cs = res.get("cost_statistics") or {}
    row["total_cost"] = round(float(cs.get("total_cost", 0) or 0), 1)
    print(f"[{tag}] " + json.dumps(row, ensure_ascii=False, default=float), flush=True)
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
