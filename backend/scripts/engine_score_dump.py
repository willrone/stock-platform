#!/usr/bin/env python3
"""抓引擎某一天的完整排名分（TopkDropoutTradeModeExecutor 的 scores 字典）。

用于对账：把引擎算出来的截面排名分与我抽的腿分数逐股比。

用法:
  env -u PYTHONPATH -u PYTHONHOME backend/.venv-py313/bin/python3 \
    backend/scripts/engine_score_dump.py --strategy honest_composite \
      --date 2023-06-05 --out /tmp/scores_42f.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict

BACKEND = "/Users/rone/Projects/stock-platform/backend"
ROOT = "/Users/rone/Projects/stock-platform"
sys.path.insert(0, BACKEND)
sys.path.insert(0, os.path.join(BACKEND, "scripts"))

import ensemble_lab as L  # noqa: E402

END = datetime(2026, 5, 8)
START = datetime(2023, 5, 8)
CFG = {
    "rsi_71421": {"topk": 30, "n_drop": 3, "rebalance_days": "weekly",
                  "rebalance_weekday": 4},
    "honest_composite": {"topk": 60, "n_drop": 5, "rebalance_days": "weekly",
                         "rebalance_weekday": 4},
}
MPS = {"rsi_71421": 0.031667, "honest_composite": 0.015833}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="honest_composite")
    ap.add_argument("--date", default="2023-06-05")
    ap.add_argument("--out", default="/tmp/scores_engine.json")
    args = ap.parse_args()
    st = args.strategy
    target = pd_ts(args.date)

    from app.services.backtest import BacktestExecutor
    from app.services.backtest.models import BacktestConfig
    import app.services.backtest.execution.trade_modes as tm

    dump: Dict[str, Any] = {}
    orig = tm._extract_scores

    def patched(context):
        s = orig(context)
        d = str(context.current_date)[:10]
        if d == args.date and "scores" not in dump:
            dump["scores"] = {k: float(v) for k, v in s.items()}
            dump["n"] = len(s)
        return s

    tm._extract_scores = patched

    codes = L.universe_from_instruments()
    executor = BacktestExecutor(data_dir=ROOT + "/data", enable_parallel=True,
                                max_workers=8, enable_performance_profiling=False,
                                use_multiprocessing=False)
    res = asyncio.run(
        executor.run_backtest(
            strategy_name=st, stock_codes=codes,
            start_date=START, end_date=END,
            strategy_config=dict(CFG[st], lag=1),
            backtest_config=BacktestConfig(
                enable_tradability=True, initial_cash=1_000_000.0,
                commission_rate=0.0003, slippage_rate=0.0001,
                max_position_size=MPS[st], cash_reserve_ratio=0.05,
                board_lot_size=100, stop_loss_pct=0.0, take_profit_pct=0.0,
                rebalance_frequency="daily", open_cost=0.0, close_cost=0.0,
                min_cost=0.0,
            ),
        )
    )
    tm._extract_scores = orig
    dump["metrics"] = {k: res.get(k) for k in
                       ("total_return", "sharpe_ratio", "max_drawdown",
                        "total_trades")}
    with open(args.out, "w") as fh:
        json.dump(dump, fh, ensure_ascii=False, indent=1, default=str)
    print(f"wrote {args.out}: n_scores={dump.get('n')} SR={res.get('sharpe_ratio')}")


def pd_ts(s: str):
    import pandas as pd
    return pd.Timestamp(s)


if __name__ == "__main__":
    main()
