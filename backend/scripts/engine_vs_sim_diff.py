#!/usr/bin/env python3
"""引擎 vs 模拟器 逐日对账：定位保真度差异。

跑真实引擎（rsi_71421 / factor_ranking），记录每天的
  (日期, 组合市值, 现金, 持仓数, 持仓代码集合)
再跑 ensemble_lab 模拟器同配置，逐日比对第一个分歧点。

用法:
  env -u PYTHONPATH -u PYTHONHOME backend/.venv-py313/bin/python3 \
    backend/scripts/engine_vs_sim_diff.py --strategy rsi_71421
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import pandas as pd

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(BACKEND)
sys.path.insert(0, BACKEND)
sys.path.insert(0, os.path.join(BACKEND, "scripts"))

import ensemble_lab as L  # noqa: E402
import ensemble_legs as LEGS  # noqa: E402

END = datetime(2026, 5, 8)
START = datetime(2023, 5, 8)

CFG = {
    "rsi_71421": {"topk": 30, "n_drop": 3, "rebalance_days": "weekly",
                  "rebalance_weekday": 4},
    "factor_ranking": {"topk": 50, "n_drop": 5, "rebalance_days": "weekly",
                       "rebalance_weekday": 4},
    "honest_composite": {"topk": 60, "n_drop": 5, "rebalance_days": "weekly",
                         "rebalance_weekday": 4},
}
MPS = {"rsi_71421": 0.031667, "factor_ranking": 0.019, "honest_composite": 0.015833}


def run_engine(strategy: str, codes: List[str]) -> Dict[str, Any]:
    """跑真实引擎并逐日抓组合状态。"""
    from app.services.backtest import BacktestExecutor
    from app.services.backtest.models import BacktestConfig
    import app.services.backtest.core.portfolio_manager_array as pma

    daily: Dict[str, Any] = {}
    orig_pv = pma.PortfolioManagerArray.get_portfolio_value

    # 包一层：在每次 get_portfolio_value 时记快照（每天多次调用，用日期去重）
    cur_date: Dict[str, Any] = {"d": None}
    orig_exec = pma.PortfolioManagerArray.execute_signal

    def patched_exec(self, signal, current_prices, *a, **kw):
        out = orig_exec(self, signal, current_prices, *a, **kw)
        return out

    executor = BacktestExecutor(data_dir=ROOT + "/data", enable_parallel=True,
                                max_workers=8, enable_performance_profiling=False,
                                use_multiprocessing=False)

    # 用 monkeypatch 记录：包 execute_backtest_loop 不方便 → 直接改 executor 的
    # _execute_backtest_loop？改为在 run_backtest 之后从 portfolio_manager 取
    # equity_curve（引擎已经记录）。
    res = asyncio.run(
        executor.run_backtest(
            strategy_name=strategy,
            stock_codes=codes,
            start_date=START,
            end_date=END,
            strategy_config=dict(CFG[strategy], lag=1),
            backtest_config=BacktestConfig(
                enable_tradability=True,
                initial_cash=1_000_000.0,
                commission_rate=0.0003,
                slippage_rate=0.0001,
                max_position_size=MPS[strategy],
                cash_reserve_ratio=0.05,
                board_lot_size=100,
                stop_loss_pct=0.0,
                take_profit_pct=0.0,
                rebalance_frequency="daily",
                open_cost=0.0,
                close_cost=0.0,
                min_cost=0.0,
            ),
        )
    )
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="rsi_71421")
    args = ap.parse_args()
    st = args.strategy
    codes = L.universe_from_instruments()
    print(f"[engine] {st} ...", flush=True)
    res = run_engine(st, codes)
    out = {k: res.get(k) for k in ("total_return", "sharpe_ratio", "max_drawdown",
                                   "total_trades", "yearly")}
    print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    path = os.path.join("/tmp", f"engine_{st}.json")
    with open(path, "w") as fh:
        json.dump(res, fh, ensure_ascii=False, default=str)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
