#!/usr/bin/env python3
"""可成交性检查的冒烟测试（2026-10-26）

目的：在跑完整 A/B 之前，先用**短窗口**确认：
  1. 回测能跑通不报错（改动没把引擎改坏）
  2. 可成交性矩阵真的被构建并生效（日志里有矩阵统计）
  3. 开关开/关能产生**不同的**成交笔数（说明检查真的在起作用）

跑法：
  env -u PYTHONPATH -u PYTHONHOME backend/.venv-py313/bin/python3 \
    backend/scripts/smoke_tradability.py
"""
from __future__ import annotations

import asyncio
import os
import re
import sys
from datetime import datetime

BACKEND = "/Users/rone/Projects/stock-platform/backend"
ROOT = "/Users/rone/Projects/stock-platform"
sys.path.insert(0, BACKEND)

from app.services.backtest import BacktestExecutor  # noqa: E402
from app.services.backtest.models import BacktestConfig  # noqa: E402

START = datetime(2025, 1, 1)
END = datetime(2025, 6, 30)


def universe(limit: int = 800) -> list:
    path = os.path.join(ROOT, "data", "qlib_official_data", "instruments", "all.txt")
    codes = []
    with open(path) as fh:
        for line in fh:
            parts = line.strip().split("\t")
            if parts and "_" in parts[0]:
                c, _, m = parts[0].partition("_")
                if m.upper() in ("SH", "SZ", "BJ") and c.isdigit():
                    codes.append(f"{c}.{m.upper()}")
    return sorted(codes)[:limit]


def run(codes, enable: bool) -> dict:
    ex = BacktestExecutor(
        data_dir=os.path.join(ROOT, "data"),
        enable_parallel=True, max_workers=4,
        enable_performance_profiling=False, use_multiprocessing=False,
    )
    res = asyncio.run(
        ex.run_backtest(
            strategy_name="honest_composite",
            stock_codes=codes,
            start_date=START, end_date=END,
            strategy_config={"topk": 60, "n_drop": 5,
                             "rebalance_days": "weekly", "rebalance_weekday": 4},
            backtest_config=BacktestConfig(
                initial_cash=1_000_000.0,
                commission_rate=0.0003, slippage_rate=0.0001,
                max_position_size=0.015833,
                cash_reserve_ratio=0.05, board_lot_size=100,
                stop_loss_pct=0.0, take_profit_pct=0.0,
                rebalance_frequency="daily",
                open_cost=0.0, close_cost=0.0, min_cost=0.0,
                enable_tradability=enable,
            ),
        )
    )
    return {
        "sharpe": res.get("sharpe_ratio"),
        "ret": res.get("total_return"),
        "mdd": res.get("max_drawdown"),
        "trades": res.get("total_trades"),
    }


def main() -> int:
    codes = universe(800)
    print(f"[smoke] 股票池 {len(codes)} 只  窗口 {START.date()}→{END.date()}", flush=True)

    out = {}
    for enable in (False, True):
        tag = "开可成交性" if enable else "关可成交性"
        print(f"\n[smoke] === {tag} ===", flush=True)
        try:
            r = run(codes, enable)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"[smoke] ❌ {tag} 失败: {type(e).__name__}: {e}")
            return 1
        out[enable] = r
        print(f"[smoke] {tag}: sharpe={r['sharpe']} ret={r['ret']} "
              f"mdd={r['mdd']} trades={r['trades']}", flush=True)

    t_off = out[False]["trades"]
    t_on = out[True]["trades"]
    print(f"\n[smoke] 成交笔数：关={t_off} 开={t_on} 差={t_off - t_on}")
    if t_off == t_on:
        print("[smoke] ❌ 开关无效（笔数相同）—— 可成交性检查可能没生效")
        return 1
    print("[smoke] ✅ 开关有效，可成交性检查在起作用")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
