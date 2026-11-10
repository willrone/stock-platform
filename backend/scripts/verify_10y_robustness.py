#!/usr/bin/env python3
"""10 年稳健性检验（真实盘口径）。

## 为什么现在能做
早先「10 年窗口只有 268 只有数据」的结论是基于 `data/qlib_data/features/day`
——那里 5241/5513 只是 2021-05-11 起的两段拼接增量块。
现在有两份新数据：
  * `data/qlib_data/features_full/`（5513 只，**各自上市首日**起，交易日格单段）
  * `data/parquet/stock_data/`（未复权 OHLCV，**各自上市首日**起）
两者合起来 → 10 年窗口也有 ~90%+ 股票覆盖。

## 口径
窗口 2016-05-08 → 2026-05-08 / PIT 股票池 / lag=1 / enable_tradability=True /
TopK-Dropout 等权 / 佣金 0.03% + 滑点 0.01% / 整手 100 股 / 现金预留 5%

## 用法
  env -u PYTHONPATH -u PYTHONHOME backend/.venv-py313/bin/python3 \
    backend/scripts/verify_10y_robustness.py --years 10
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import pandas as pd

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(BACKEND)
sys.path.insert(0, BACKEND)
sys.path.insert(0, os.path.join(BACKEND, "scripts"))

import ensemble_lab as L  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=10)
    ap.add_argument("--out", default="ensemble_10y_robustness.json")
    args = ap.parse_args()

    t0 = time.time()
    end = datetime(2026, 5, 8)
    start = datetime(end.year - args.years, end.month, end.day)
    L.START = start
    L.END = end
    print(f"[10y] window {start.date()} → {end.date()}", flush=True)

    codes = L.universe_from_instruments()
    mats = L.load_matrices(codes, start=start, end=end)
    print(f"[10y] loaded in {time.time()-t0:.0f}s  "
          f"n_dates={len(mats['master'])} n_stocks={len(mats['codes'])}", flush=True)

    trad = L.build_tradability_mats(mats)
    print(f"[10y] tradability {time.time()-t0:.0f}s", flush=True)

    # 面板因子：features_full（交易日格，全历史）
    s42_full, z_full = L.build_42f_score(
        mats, panel_aligned=True, panel_dir=L.PANEL_FULL)
    # 面板因子：features/day（两段拼接，策略原口径 ≈ 只在 2021 年后有值）
    s42_day, z_day = L.build_42f_score(
        mats, panel_aligned=True,
        panel_dir=os.path.join(L.QLIB_LIVE, "features", "day"))
    s_rsi = L.rsi_71421_score(mats)
    s_fr = L.factor_ranking_score(mats)
    print(f"[10y] scores {time.time()-t0:.0f}s", flush=True)

    out: Dict[str, Any] = {
        "window": f"{start.date()}→{end.date()}",
        "n_codes_requested": len(codes),
        "n_codes_loaded": len(mats["codes"]),
        "n_trading_days": len(mats["master"]),
        "caliber": {"lag": 1, "enable_tradability": True, "commission": 0.0003,
                    "slippage": 0.0001, "cash_reserve": 0.05,
                    "board_lot": 100, "rebalance": "weekly(周五)"},
        "runs": [],
    }

    def run(name: str, sc, cfg: L.SimConfig, note: str = "") -> None:
        res = L.simulate(sc, mats, trad, cfg)
        e = {"name": name, "note": note,
             "topk": cfg.topk, "n_drop": cfg.n_drop, "buffer": cfg.buffer,
             "sharpe_ratio": round(float(res["sharpe_ratio"]), 4),
             "total_return": round(float(res["total_return"]), 4),
             "max_drawdown": round(float(res["max_drawdown"]), 4),
             "total_trades": int(res["total_trades"]),
             "volatility": round(float(res.get("volatility", 0.0)), 4),
             "yearly": res.get("yearly", {})}
        out["runs"].append(e)
        print(f"  {name:34s} SR={e['sharpe_ratio']:6.3f} ret={e['total_return']:+7.2%} "
              f"mdd={e['max_drawdown']:7.2%} n={e['total_trades']:5d} "
              f"{e['yearly']}", flush=True)

    print("\n=== 10 年稳健性 ===", flush=True)
    for topk in (20, 30, 60):
        cfg = L.SimConfig(topk=topk, n_drop=max(2, topk // 12), lag=1,
                          rebalance_days="weekly",
                          max_position_size=(1 - 0.05) / topk)
        run(f"42f_full_k{topk}", s42_full, cfg, "features_full 面板")
        run(f"42f_day_k{topk}", s42_day, cfg, "features/day 面板（两段拼接）")
    cfg60 = L.SimConfig(topk=60, n_drop=5, lag=1, rebalance_days="weekly",
                        max_position_size=0.015833)
    run("rsi_71421_k30", s_rsi,
        L.SimConfig(topk=30, n_drop=3, lag=1, rebalance_days="weekly",
                    max_position_size=0.031667))
    run("factor_ranking_k50", s_fr,
        L.SimConfig(topk=50, n_drop=5, lag=1, rebalance_days="weekly",
                    max_position_size=0.019))
    # 合成腿
    blend = L.cs_zscore(s42_full, 30) + L.cs_zscore(s_rsi, 30)
    run("42f_full+rsi_k20", blend,
        L.SimConfig(topk=20, n_drop=2, lag=1, rebalance_days="weekly",
                    max_position_size=(1 - 0.05) / 20))

    path = os.path.join(L.OUT_DIR, args.out)
    with open(path, "w") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1, default=str)
    print(f"\n写入 {path}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
