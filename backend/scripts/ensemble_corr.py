#!/usr/bin/env python3
"""第一步：各策略分数矩阵的横截面 Spearman 相关系数矩阵（真实盘口径）。

窗口 2023-05-08 → 2026-05-08，5238 只 PIT 股票池，同一 master 日历。
逐日算横截面 Spearman（RankIC），再对全窗口取平均。

用法:
  env -u PYTHONPATH -u PYTHONHOME backend/.venv-py313/bin/python3 \
    backend/scripts/ensemble_corr.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(BACKEND)
sys.path.insert(0, BACKEND)
sys.path.insert(0, os.path.join(BACKEND, "scripts"))

import ensemble_lab as L  # noqa: E402
import ensemble_legs as LEGS  # noqa: E402

LEGS_ALL = ["42f", "rsi", "fr", "gbm", "gbm_raw", "mf"]


def main() -> None:
    mats = LEGS.cache_mats()
    legs = {}
    for n in LEGS_ALL:
        p = os.path.join(LEGS.LEG_DIR, f"{n}.npz")
        if os.path.exists(p):
            legs[n] = np.load(p, allow_pickle=False)["score"]
    print(f"[legs] {list(legs)}", flush=True)

    corr = L.score_correlation_matrix(legs)
    names = corr["names"]
    print("\n=== 平均横截面 Spearman 相关矩阵（全窗口 727 日）===")
    print("  " + " " * 10 + " ".join(f"{n:>9s}" for n in names))
    for i, n in enumerate(names):
        print(f"  {n:10s} " + " ".join(
            f"{corr['mean_rank_ic_matrix'][i][j]:9.3f}"
            for j in range(len(names))))

    fwd1 = np.full_like(mats["close"], np.nan)
    fwd1[:-1] = mats["close"][1:] / mats["close"][:-1] - 1.0
    print("\n=== 各腿对 fwd1 收益的 RankIC ===")
    stats = {}
    for n, sc in legs.items():
        ic = L.cs_rank_ic(sc, fwd1)
        mu, sd = float(np.nanmean(ic)), float(np.nanstd(ic))
        stats[n] = {"mean_ic": round(mu, 5), "std_ic": round(sd, 5),
                    "icir": round(mu / sd, 4) if sd > 0 else 0.0,
                    "n_days": int(np.isfinite(ic).sum())}
        print(f"  {n:10s} meanIC={mu:+.4f} std={sd:.4f} ICIR={mu/sd:+.3f} "
              f"n={np.isfinite(ic).sum()}")

    # 分年相关（看相关结构是否稳定）
    master = mats["master"]
    print("\n=== 分年平均横截面相关（42f vs 其它）===")
    for y in sorted(set(master.year)):
        rows = np.where(master.year == y)[0]
        sub = {n: sc[rows] for n, sc in legs.items()}
        c = L.score_correlation_matrix(sub)
        idx = c["names"].index("42f")
        vals = {c["names"][j]: c["mean_rank_ic_matrix"][idx][j]
                for j in range(len(c["names"])) if j != idx}
        print(f"  {y}: " + "  ".join(f"{k}={v:+.3f}" for k, v in vals.items()))

    out = {"names": names,
           "mean_rank_ic_matrix": corr["mean_rank_ic_matrix"],
           "leg_ic_stats": stats}
    path = os.path.join(L.OUT_DIR, "ensemble_correlation.json")
    with open(path, "w") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1, default=str)
    print(f"\n写入 {path}")


if __name__ == "__main__":
    main()
