#!/usr/bin/env python3
"""组合构建层权重重测（修正 HRP / maxdiv / invvol 三个实现 bug 之后）。

## 修的三个 bug（都是我自己第一轮写的）
1. `np.corrcoef(rg)` 少了 `rowvar=False` → 变量在**行**上，
   得到 120×120 的矩阵而不是 8×8 的股票间相关矩阵 → HRP 聚类全错。
2. HRP 递归二分：论文（López de Prado 2016, Algorithm 2 / Code snippet 2）
   是对**准对角化后的有序列表保序对半切分**，不是对层次树切分。
   层次树版本已重写为论文版本。
3. 权重回填映射：`w_qd[i]` 是**准对角位置 i** 的权重，对应原始下标
   `order[i]`，之前误当成 `good[i]` → 下标越界。
4. `_trailing_returns` 少取一行 → lookback=20 时恒返回 None →
   逆波动率在早期静默退化成等权（F_invvol_20 与基线完全相同就是征兆）。

## 结论预期
组合构建层（逆波动率 / HRP / 最大分散化 / 分数比例）在**等权 TopK 之上**
是否还有增量？本脚本用修正后的实现重新回答。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

BACKEND = "/Users/rone/Projects/stock-platform/backend"
ROOT = "/Users/rone/Projects/stock-platform"
sys.path.insert(0, BACKEND)
sys.path.insert(0, os.path.join(BACKEND, "scripts"))

import ensemble_lab as L  # noqa: E402
import methods_lab as M  # noqa: E402

OUT_DIR = os.path.join(ROOT, "data", "backtest")
TRIALS: List[Dict[str, Any]] = []


def record(name, group, res, cfg, note=""):
    e = {
        "name": name, "group": group,
        "sharpe_ratio": round(float(res["sharpe_ratio"]), 4),
        "total_return": round(float(res["total_return"]), 4),
        "max_drawdown": round(float(res["max_drawdown"]), 4),
        "total_trades": int(res["total_trades"]),
        "volatility": round(float(res.get("volatility", 0.0)), 4),
        "yearly": res.get("yearly", {}),
        "config": {"topk": cfg.topk, "n_drop": cfg.n_drop, "lag": cfg.lag,
                   "rebalance_days": cfg.rebalance_days},
        "note": note,
    }
    TRIALS.append(e)
    print(f"  {name[:40]:42s} SR={e['sharpe_ratio']:6.3f} "
          f"ret={e['total_return']:+8.2%} mdd={e['max_drawdown']:7.2%} "
          f"n={e['total_trades']:5d}  {note[:40]}", flush=True)
    return e


class WeightSelector(L.Selector):
    """按外部权重函数做目标权重再平衡。"""

    def __init__(self, cfg: L.SimConfig, close: np.ndarray, lookback: int,
                 kind: str):
        self.cfg = cfg
        self.close = close
        self.lookback = lookback
        self.kind = kind
        self.rebalance_to_targets = True
        self.n_fallback = 0
        self.n_calls = 0

    def select(self, order, t, holdings, n_hold):
        keep = order[: self.cfg.topk]
        self.n_calls += 1
        if self.kind == "invvol":
            w = M.inverse_vol_weights(self.close, t, keep, self.lookback)
        elif self.kind == "hrp":
            w = M.hrp_weights(self.close, t, keep, self.lookback)
        elif self.kind == "maxdiv":
            w = M.maxdiv_weights(self.close, t, keep, self.lookback)
        else:
            w = None
        if w is None or len(w) != len(keep) or not np.isfinite(w).all():
            self.n_fallback += 1
            return keep, {}, order
        w = np.clip(w, 0.0, None)
        if w.sum() <= 0:
            self.n_fallback += 1
            return keep, {}, order
        w = w / w.sum() * (1 - self.cfg.cash_reserve)
        return keep, {int(c): float(w[i]) for i, c in enumerate(keep)}, order


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="methods_weights_fixed.json")
    ap.add_argument("--topk", type=int, default=60)
    ap.add_argument("--ndrop", type=int, default=5)
    args = ap.parse_args()

    t0 = time.time()
    print("=" * 84)
    print("组合构建层权重重测（修正实现后）· 真实盘口径 lag=1")
    print("=" * 84)

    cache = M.load_cached_mats()
    mats, trad = cache["mats"], cache["trad"]
    CFG = L.SimConfig(topk=args.topk, n_drop=args.ndrop, lag=1,
                      rebalance_days="weekly",
                      max_position_size=(1 - 0.05) / args.topk)
    s42, _ = L.build_42f_score(
        mats, panel_aligned=True,
        panel_dir=os.path.join(L.QLIB_LIVE, "features", "day"))
    is_rows, oos_rows = L.split_index(mats["master"], pd.Timestamp("2025-05-08"))
    sub_is = L.slice_mats(mats, is_rows)
    sub_oos = L.slice_mats(mats, oos_rows)
    tr_is = {k: v[is_rows] for k, v in trad.items()}
    tr_oos = {k: v[oos_rows] for k, v in trad.items()}

    res_all: Dict[str, Any] = {
        "meta": {
            "window": f"{mats['master'][0].date()}→{mats['master'][-1].date()}",
            "n_codes": len(mats["codes"]),
            "n_trading_days": len(mats["master"]),
            "oos_split": "2025-05-08",
            "caliber": {"lag": 1, "enable_tradability": True,
                        "topk": args.topk, "n_drop": args.ndrop,
                        "commission": 0.0003, "slippage": 0.0001,
                        "board_lot": 100, "cash_reserve": 0.05,
                        "rebalance": "weekly(周五)"},
            "engine_baseline_42f": {
                "sharpe_ratio": 0.7991795250472001,
                "total_return": 0.34779248262737783,
                "max_drawdown": -0.16460289642093642,
                "total_trades": 1080,
            },
            "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        },
        "trials": [],
    }

    print("\n  -- 基线（等权 TopK）--")
    r = L.simulate(s42, mats, trad, CFG)
    record("W_base_equal", "W", r, CFG, "等权 TopK（基线）")
    r_is = L.simulate(s42[is_rows], sub_is, tr_is, CFG)
    r_oos = L.simulate(s42[oos_rows], sub_oos, tr_oos, CFG)
    print(f"    └─ IS SR={r_is['sharpe_ratio']:.3f}  OOS SR={r_oos['sharpe_ratio']:.3f}")

    print("\n  -- 逆波动率 / HRP / 最大分散化（修正后）--")
    for kind in ("invvol", "hrp", "maxdiv"):
        for lb in (20, 60, 120, 250):
            sel = WeightSelector(CFG, mats["close"], lb, kind)
            res = L.simulate(s42, mats, trad, CFG, selector=sel)
            fb = sel.n_fallback
            record(f"W_{kind}_{lb}", "W", res, CFG,
                   f"{kind} lookback={lb}（回退 {fb}/{sel.n_calls} 次）")
            # OOS
            sel2 = WeightSelector(CFG, mats["close"], lb, kind)
            ro = L.simulate(s42[oos_rows], sub_oos, tr_oos, CFG, selector=sel2)
            print(f"    └─ OOS SR={ro['sharpe_ratio']:.3f} "
                  f"ret={ro['total_return']:+.2%} "
                  f"mdd={ro['max_drawdown']:.2%} n={ro['total_trades']}")

    print("\n  -- 分数比例加权（w ∝ score^p）--")
    for power in (0.5, 1.0, 2.0):
        sel = L.ScoreWeightSelector(CFG, power=power)
        res = L.simulate(s42, mats, trad, CFG, selector=sel)
        record(f"W_scorew_p{power}", "W", res, CFG, f"w ∝ score^{power}")

    res_all["trials"] = TRIALS
    res_all["n_trials"] = len(TRIALS)
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, args.out)
    with open(path, "w") as fh:
        json.dump(res_all, fh, ensure_ascii=False, indent=1, default=str)
    print(f"\n写入 {path}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
