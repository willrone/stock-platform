#!/usr/bin/env python3
"""IS/OOS 正确口径评估：每段用**该段窗口重抽的腿**（与引擎分段跑同口径）。

## 为什么必须这样
引擎跑 IS 段（2023-05→2025-05）时，``prepare_backtest_data`` 只看得到该段的
数据 → 60 日滚动因子在段首冷启动、面板因子读到的行数不同、截面股票数不同
（5238 → 5203）→ 腿的分数矩阵与全窗口抽的不一样。
用全窗口腿去模拟分段跑，IS 会偏乐观（实测 1.147 vs 引擎 0.666）。

本脚本：
  1. 用 ensemble_legs_window.py 按 IS / OOS 两段分别重抽腿
  2. 用 ensemble_sim 在各自窗口内模拟
  3. IS 选参 → OOS 验证（OOS 只跑 IS 选出的那一组）

用法:
  env -u PYTHONPATH -u PYTHONHOME backend/.venv-py313/bin/python3 \
    backend/scripts/ensemble_oos_window.py --legs 42f mf rsi fr gbm
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

BACKEND = "/Users/rone/Projects/stock-platform/backend"
ROOT = "/Users/rone/Projects/stock-platform"
sys.path.insert(0, BACKEND)
sys.path.insert(0, os.path.join(BACKEND, "scripts"))

import ensemble_lab as L  # noqa: E402
import ensemble_legs as LEGS  # noqa: E402
import ensemble_sim as S  # noqa: E402

W = "/Users/rone/.hermes/cache/scratch/stock_ens/cache/legs_window"
IS_START, IS_END = "2023-05-08", "2025-05-08"
OOS_START, OOS_END = "2025-05-08", "2026-05-08"


def ensure_window_legs(tag: str, start: str, end: str,
                       legs: Sequence[str]) -> None:
    """按窗口重抽腿（已存在则跳过）。"""
    todo = [x for x in legs
            if not os.path.exists(os.path.join(W, f"{tag}_{x}.npz"))]
    if not todo:
        print(f"[legs] {tag} 全部命中缓存", flush=True)
        return
    cmd = [sys.executable, os.path.join(BACKEND, "scripts",
                                       "ensemble_legs_window.py"),
           "--start", start, "--end", end, "--tag", tag, "--legs", *todo]
    print(f"[legs] 抽 {tag} 窗口的腿: {todo}", flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def load_window(tag: str, legs: Sequence[str], mats_all: Dict[str, Any],
                trad_all: Dict[str, np.ndarray]
                ) -> Tuple[Dict[str, Any], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """读某窗口的腿 + 重建该窗口的 mats / trad。"""
    z = np.load(os.path.join(W, f"{tag}_{legs[0]}.npz"), allow_pickle=False)
    master = pd.DatetimeIndex(z["master"])
    wcodes = [str(c) for c in z["codes"]]
    leg_scores = {x: np.load(os.path.join(W, f"{tag}_{x}.npz"),
                             allow_pickle=False)["score"] for x in legs}

    pos = {d: i for i, d in enumerate(mats_all["master"])}
    rows = np.array([pos[d] for d in master])
    cpos = {c: i for i, c in enumerate(mats_all["codes"])}
    idx = np.array([cpos.get(c, -1) for c in wcodes])
    valid = idx >= 0

    sub: Dict[str, Any] = {"master": master, "codes": wcodes, "stock_data": {}}
    for k in ("close", "high", "low", "volume"):
        out = np.full((len(rows), len(wcodes)), np.nan)
        out[:, valid] = mats_all[k][np.ix_(rows, idx[valid])]
        sub[k] = out
    subtrad: Dict[str, np.ndarray] = {}
    for k, v in trad_all.items():
        fill = np.nan if v.dtype.kind == "f" else True
        out = np.full((len(rows), len(wcodes)), fill)
        out[:, valid] = v[np.ix_(rows, idx[valid])]
        subtrad[k] = out
    return sub, subtrad, leg_scores


def winsor(m: np.ndarray, clip: float = 3.0) -> np.ndarray:
    z = L.cs_zscore(m, 30)
    return np.where(np.isfinite(z), np.clip(z, -clip, clip), np.nan)


def cs_rank_norm(m: np.ndarray) -> np.ndarray:
    df = pd.DataFrame(m)
    cnt = df.notna().sum(axis=1)
    r = df.rank(axis=1, na_option="keep")
    n = df.notna().sum(axis=1).replace(0, np.nan)
    out = ((r - 0.5) / n.to_numpy()[:, None]).to_numpy()
    out[(cnt < 30).to_numpy()] = np.nan
    return out


PREPROC = {"raw": lambda m: m, "z": lambda m: L.cs_zscore(m, 30),
           "rank": cs_rank_norm, "winsor3": winsor}


def blend_count(fs: List[np.ndarray], fn) -> np.ndarray:
    tot = None
    cnt = None
    for f in fs:
        m = fn(f)
        g = np.isfinite(m)
        c = np.where(g, m, 0.0)
        n_ = g.astype(float)
        tot = c if tot is None else tot + c
        cnt = n_ if cnt is None else cnt + n_
    return np.where(cnt > 0, tot / np.where(cnt > 0, cnt, 1.0), np.nan)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--legs", nargs="*",
                    default=["42f", "rsi", "fr", "gbm", "gbm_raw", "mf"])
    ap.add_argument("--out", default="ensemble_oos_window.json")
    ap.add_argument("--topks", nargs="*", type=int, default=[20, 30, 40, 60, 80])
    ap.add_argument("--ndrops", nargs="*", type=int, default=[2, 3, 5, 10])
    args = ap.parse_args()

    mats_all = LEGS.cache_mats()
    mats_all["stock_data"] = {}
    trad_all = LEGS.cache_trad(mats_all, {})
    ensure_window_legs("is", IS_START, IS_END, args.legs)
    ensure_window_legs("oos", OOS_START, OOS_END, args.legs)

    sub_is, trad_is, legs_is = load_window("is", args.legs, mats_all, trad_all)
    sub_oos, trad_oos, legs_oos = load_window("oos", args.legs, mats_all, trad_all)
    print(f"[setup] IS {len(sub_is['master'])} 日 × {len(sub_is['codes'])} 股 | "
          f"OOS {len(sub_oos['master'])} 日 × {len(sub_oos['codes'])} 股",
          flush=True)

    # ── 候选组合（腿子集 × 预处理）──
    subsets: List[Tuple[str, List[str]]] = []
    names = [x for x in args.legs]
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            subsets.append((f"{names[i]}+{names[j]}", [names[i], names[j]]))
    subsets.append(("42f+rsi+fr+gbm",
                    [x for x in ("42f", "rsi", "fr", "gbm") if x in names]))
    subsets.append(("42f+rsi+fr+gbm+mf",
                    [x for x in ("42f", "rsi", "fr", "gbm", "mf") if x in names]))

    trials = 0
    rows: List[Dict[str, Any]] = []
    print("\n=== IS 网格（窗口口径）===", flush=True)
    for sname, sl in subsets:
        for prep in ("raw", "z", "rank", "winsor3"):
            sc_is = blend_count([legs_is[x] for x in sl], PREPROC[prep])
            sc_oos = blend_count([legs_oos[x] for x in sl], PREPROC[prep])
            for topk in args.topks:
                for n_drop in args.ndrops:
                    if n_drop >= topk:
                        continue
                    cfg = S.SimConfig(
                        topk=topk, n_drop=n_drop, lag=1,
                        rebalance_days="weekly",
                        max_position_size=(1 - 0.05) / topk)
                    pr_is = S.prepare_score(sc_is, sub_is, cfg)
                    pr_oos = S.prepare_score(sc_oos, sub_oos, cfg)
                    ris = S.simulate(pr_is, sub_is, trad_is, cfg)
                    roos = S.simulate(pr_oos, sub_oos, trad_oos, cfg)
                    trials += 1
                    rows.append({
                        "combo": sname, "legs": sl, "prep": prep,
                        "topk": topk, "n_drop": n_drop,
                        "IS_sharpe": round(float(ris["sharpe_ratio"]), 4),
                        "IS_return": round(float(ris["total_return"]), 4),
                        "IS_mdd": round(float(ris["max_drawdown"]), 4),
                        "IS_trades": int(ris["total_trades"]),
                        "OOS_sharpe": round(float(roos["sharpe_ratio"]), 4),
                        "OOS_return": round(float(roos["total_return"]), 4),
                        "OOS_mdd": round(float(roos["max_drawdown"]), 4),
                        "OOS_trades": int(roos["total_trades"]),
                        "OOS_yearly": roos.get("yearly", {}),
                    })
    print(f"  IS 网格 {trials} 个组合", flush=True)

    rows.sort(key=lambda r: -r["IS_sharpe"])
    print("\n  IS Top12（含 OOS 验证）:")
    for r in rows[:12]:
        print(f"    {r['combo']:22s} {r['prep']:9s} k={r['topk']:3d} d={r['n_drop']:2d} "
              f"IS={r['IS_sharpe']:6.3f} (ret={r['IS_return']:+7.2%} "
              f"mdd={r['IS_mdd']:7.2%} n={r['IS_trades']}) → "
              f"OOS={r['OOS_sharpe']:6.3f} (ret={r['OOS_return']:+7.2%} "
              f"mdd={r['OOS_mdd']:7.2%} n={r['OOS_trades']})")

    best = rows[0]
    print(f"\n  ▶ IS 最优 = {best['combo']} {best['prep']} k={best['topk']} "
          f"d={best['n_drop']}")
    print(f"    IS  SR={best['IS_sharpe']:.3f} ret={best['IS_return']:+.2%} "
          f"mdd={best['IS_mdd']:.2%}")
    print(f"    OOS SR={best['OOS_sharpe']:.3f} ret={best['OOS_return']:+.2%} "
          f"mdd={best['OOS_mdd']:.2%} {best['OOS_yearly']}")

    # OOS 分布（诚实报告）
    oos_vals = np.array([r["OOS_sharpe"] for r in rows])
    is_vals = np.array([r["IS_sharpe"] for r in rows])
    print(f"\n  全体 {len(rows)} 个组合的 SR 分布:")
    print(f"    IS : mean={is_vals.mean():.3f} sd={is_vals.std():.3f} "
          f"max={is_vals.max():.3f}")
    print(f"    OOS: mean={oos_vals.mean():.3f} sd={oos_vals.std():.3f} "
          f"max={oos_vals.max():.3f}")
    print(f"    IS→OOS 相关系数 = {np.corrcoef(is_vals, oos_vals)[0,1]:.3f}")

    out = {"n_trials": trials, "is_selected": best, "all": rows,
           "summary": {"IS_mean": float(is_vals.mean()),
                       "IS_sd": float(is_vals.std()),
                       "IS_max": float(is_vals.max()),
                       "OOS_mean": float(oos_vals.mean()),
                       "OOS_sd": float(oos_vals.std()),
                       "OOS_max": float(oos_vals.max()),
                       "IS_OOS_corr": float(np.corrcoef(is_vals, oos_vals)[0, 1])}}
    path = os.path.join(L.OUT_DIR, args.out)
    json.dump(out, open(path, "w"), ensure_ascii=False, indent=1, default=str)
    print(f"\n写入 {path}")


if __name__ == "__main__":
    main()
