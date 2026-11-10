#!/usr/bin/env python3
"""引擎 vs 模拟器：逐笔成交对账（定位保真度差异的最后一步）。

跑真实引擎（monkeypatch 记录每笔成交），再跑 ensemble_lab 模拟器同配置，
输出前 N 笔差异。

用法:
  env -u PYTHONPATH -u PYTHONHOME backend/.venv-py313/bin/python3 \
    backend/scripts/engine_trade_dump.py --strategy honest_composite --out /tmp/trades_42f.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(BACKEND)
sys.path.insert(0, BACKEND)
sys.path.insert(0, os.path.join(BACKEND, "scripts"))

import ensemble_lab as L  # noqa: E402

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="honest_composite")
    ap.add_argument("--out", default="/tmp/trades_engine.json")
    args = ap.parse_args()
    st = args.strategy

    from app.services.backtest import BacktestExecutor
    from app.services.backtest.models import BacktestConfig
    import app.services.backtest.core.portfolio_manager_array as pma

    trades: List[Dict[str, Any]] = []
    orig = pma.PortfolioManagerArray.execute_signal

    def patched(self, signal, current_prices, *a, **kw):
        out = orig(self, signal, current_prices, *a, **kw)
        trade = out[0] if isinstance(out, tuple) else out
        if trade is not None:
            trades.append({
                "date": str(signal.timestamp)[:10],
                "code": signal.stock_code,
                "action": signal.signal_type.name,
                "price": float(getattr(trade, "price", 0) or 0),
                "qty": float(getattr(trade, "quantity", 0) or 0),
                "reason": signal.reason,
            })
        return out

    pma.PortfolioManagerArray.execute_signal = patched

    codes = L.universe_from_instruments()
    executor = BacktestExecutor(data_dir=ROOT + "/data", enable_parallel=True,
                                max_workers=8, enable_performance_profiling=False,
                                use_multiprocessing=False)
    res = asyncio.run(
        executor.run_backtest(
            strategy_name=st,
            stock_codes=codes,
            start_date=START,
            end_date=END,
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
    pma.PortfolioManagerArray.execute_signal = orig
    out = {
        "strategy": st,
        "metrics": {k: res.get(k) for k in
                    ("total_return", "sharpe_ratio", "max_drawdown",
                     "total_trades", "yearly", "monthly")},
        "trades": trades,
    }
    with open(args.out, "w") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1, default=str)
    print(f"wrote {args.out}: {len(trades)} trades, "
          f"SR={res.get('sharpe_ratio')}")


if __name__ == "__main__":
    main()
