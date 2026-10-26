#!/usr/bin/env python3
"""参考基线：用**同一套 IS/OOS 切分口径**跑因子线基线，供模型线对比。

跑的策略：
- honest_composite 25f（当前真实基线 0.880）
- honest_composite 42f（use_panel=True）
- gbm_voltgt（正式模型线基线，注意它**忽略 lag** → 实际是 lag0）

口径：lag=1 + 可成交性开 + 5238 只 PIT 池 + 3 年窗口，
指标从 equity_curve 切 IS(→2025-05-08) / OOS，与 search_gbm_params.py 一致。
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from datetime import datetime

BACKEND = "/Users/rone/Projects/stock-platform/backend"
ROOT = "/Users/rone/Projects/stock-platform"
sys.path.insert(0, BACKEND)
sys.path.insert(0, os.path.join(BACKEND, "scripts"))

from app.services.backtest import BacktestExecutor  # noqa: E402
from app.services.backtest.models import BacktestConfig  # noqa: E402

from search_gbm_params import (  # noqa: E402
    _metrics_from_equity, _split_metrics, _yearly_from_equity,
    universe_from_instruments, QLIB_OFFICIAL, IS_END, MPS,
)

BASE = {
    "honest_composite_25f": {
        "cfg": {"topk": 60, "n_drop": 5, "rebalance_days": "weekly",
                "rebalance_weekday": 4},
        "mps": MPS,
    },
    "honest_composite_42f": {
        "cfg": {"topk": 60, "n_drop": 5, "rebalance_days": "weekly",
                "rebalance_weekday": 4, "use_panel": True},
        "mps": MPS,
    },
    "gbm_voltgt": {
        "cfg": {"topk": 60, "n_drop": 5, "rebalance_days": "weekly",
                "rebalance_weekday": 4, "n_estimators": 100, "max_depth": 3,
                "learning_rate": 0.05, "warmup_months": 2,
                "target_vol": 0.20, "vt_window": 20},
        "mps": MPS,
    },
}


def run(name: str, spec: dict, codes, start, end):
    executor = BacktestExecutor(
        data_dir=ROOT + "/data", enable_parallel=True, max_workers=8,
        enable_performance_profiling=False, use_multiprocessing=False,
    )
    t0 = time.perf_counter()
    res = asyncio.run(executor.run_backtest(
        strategy_name=name, stock_codes=codes,
        start_date=start, end_date=end,
        strategy_config={**spec["cfg"], "lag": 1},
        backtest_config=BacktestConfig(
            enable_tradability=True, initial_cash=1_000_000.0,
            commission_rate=0.0003, slippage_rate=0.0001,
            max_position_size=spec["mps"], cash_reserve_ratio=0.05,
            board_lot_size=100, stop_loss_pct=0.0, take_profit_pct=0.0,
            rebalance_frequency="daily", open_cost=0.0, close_cost=0.0,
            min_cost=0.0,
        ),
    ))
    wall = time.perf_counter() - t0
    curve = res.get("equity_curve") or []
    trades = res.get("trade_history") or []
    seg = _split_metrics(curve, trades)
    out = {
        "strategy": name,
        "config": spec["cfg"],
        "wall_s": round(wall, 1),
        "engine": {k: res.get(k) for k in (
            "sharpe_ratio", "total_return", "max_drawdown", "total_trades",
            "win_rate", "final_value")},
        "is": seg["is"], "oos": seg["oos"], "full": seg["full"],
        "yearly": _yearly_from_equity(curve),
    }
    print(f"  {name}: 引擎 sharpe={res.get('sharpe_ratio'):.4f} "
          f"| IS={seg['is']['sharpe']:.4f} OOS={seg['oos']['sharpe']:.4f} "
          f"full={seg['full']['sharpe']:.4f} "
          f"ret={seg['full']['return']:.2%} mdd={seg['full']['mdd']:.2%} "
          f"trades={seg['full']['n_trades']} ({wall:.0f}s)", flush=True)
    return out


def main():
    codes = universe_from_instruments(QLIB_OFFICIAL)
    print(f"[ref] 股票池 {len(codes)} 只", flush=True)
    start = datetime(2023, 5, 8)
    end = datetime(2026, 5, 8)
    results = {}
    for name, spec in BASE.items():
        try:
            results[name] = run(name, spec, codes, start, end)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            results[name] = {"strategy": name, "error": str(e)}
    out_path = os.path.join(ROOT, "data", "backtest", "gbm_ref_baselines.json")
    with open(out_path, "w") as fh:
        json.dump({
            "methodology": {
                "lag": 1, "enable_tradability": True,
                "window": "2023-05-08→2026-05-08",
                "is_window": "2023-05-08→2025-05-08",
                "oos_window": "2025-05-08→2026-05-08",
                "note": "gbm_voltgt 无 self.lag → 它报的 'lag1' 实际是 lag0（不可实现）",
            },
            "results": results,
        }, fh, ensure_ascii=False, indent=1, default=str)
    print(f"\n[ref] 已写入 {out_path}")


if __name__ == "__main__":
    main()
