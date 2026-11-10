#!/usr/bin/env python3
"""因子横截面「分层收益剖面」诊断。

回答一个问题：**42f 复合分的 alpha 到底集中在截面的哪一段？**

如果 top-20 的 fwd 收益远高于 top-60，说明「少而精」是对的
（这与引擎实测 topk=20 → 夏普 1.026 一致）；
如果 alpha 均匀分布在整个截面，那 topk 变小只是放大噪声。

用法:
  env -u PYTHONPATH -u PYTHONHOME backend/.venv-py313/bin/python3 \
    backend/scripts/diagnose_alpha_profile.py
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List

import numpy as np
import pandas as pd

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(BACKEND)
sys.path.insert(0, BACKEND)
sys.path.insert(0, os.path.join(BACKEND, "scripts"))

import ensemble_lab as L  # noqa: E402


def bucket_profile(
    score: np.ndarray,
    fwd: np.ndarray,
    n_buckets: int = 10,
    min_names: int = 30,
) -> Dict[str, Any]:
    """按分数从高到低分 n_buckets 桶，算每桶的等权 fwd 收益。

    「等权」= 桶内每只股票等权持有 1 天的平均收益。
    """
    n_d = score.shape[0]
    rows: List[Dict[str, Any]] = []
    for t in range(n_d):
        s = score[t]
        f = fwd[t]
        m = np.isfinite(s) & np.isfinite(f)
        if int(m.sum()) < min_names * n_buckets:
            continue
        idx = np.where(m)[0]
        order = idx[np.argsort(-s[idx], kind="mergesort")]
        n = len(order)
        edges = np.linspace(0, n, n_buckets + 1).astype(int)
        for b in range(n_buckets):
            sel = order[edges[b]:edges[b + 1]]
            if len(sel) < min_names:
                continue
            rows.append({"date": t, "bucket": b, "n": len(sel),
                         "mean_fwd": float(np.mean(f[sel])),
                         "mean_score": float(np.mean(s[sel]))})
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    agg = df.groupby("bucket").agg(
        mean_fwd=("mean_fwd", "mean"),
        median_fwd=("mean_fwd", "median"),
        n_days=("date", "count"),
        avg_n=("n", "mean"),
    ).reset_index()
    agg["rank"] = agg["bucket"]
    agg["bucket"] = agg["bucket"].map(
        lambda b: f"Q{int(b)+1}" if b < n_buckets // 2 else
        (f"Q{int(b)+1}(worst)" if b >= n_buckets - 1 else f"Q{int(b)+1}"))
    # 年化（252 交易日）
    agg["ann_mean_fwd_pct"] = agg["mean_fwd"] * 252 * 100
    return {
        "buckets": agg.to_dict("records"),
        "top_minus_bottom_ann_pct": float(
            (agg.iloc[0]["mean_fwd"] - agg.iloc[-1]["mean_fwd"]) * 252 * 100),
    }


def topn_profile(score: np.ndarray, fwd: np.ndarray,
                 tops=(10, 20, 30, 60, 100, 200, 400)) -> Dict[str, Any]:
    """指定 topN 的等权 fwd 收益（累计到年化）。"""
    n_d = score.shape[0]
    out: Dict[str, Any] = {}
    for n in tops:
        acc = []
        for t in range(n_d):
            s = score[t]
            f = fwd[t]
            m = np.isfinite(s) & np.isfinite(f)
            if int(m.sum()) < n:
                continue
            idx = np.where(m)[0]
            order = idx[np.argsort(-s[idx], kind="mergesort")][:n]
            acc.append(float(np.mean(f[order])))
        if not acc:
            continue
        a = np.array(acc)
        out[f"top{n}"] = {
            "daily_mean": round(float(a.mean()), 6),
            "ann_pct": round(float(a.mean() * 252 * 100), 3),
            "ann_vol_pct": round(float(a.std() * np.sqrt(252) * 100), 3),
            "sharpe_1d": round(float(a.mean() / a.std() * np.sqrt(252)), 3),
            "n_days": len(a),
        }
    return out


def main() -> None:
    import time
    t0 = time.time()
    codes = L.universe_from_instruments()
    mats = L.load_matrices(codes)
    print(f"[load] {time.time()-t0:.0f}s", flush=True)

    scores = {}
    scores["42f_engine"], _ = L.build_42f_score(mats, panel_aligned=False)
    scores["rsi_71421"] = L.rsi_71421_score(mats)
    scores["factor_ranking"] = L.factor_ranking_score(mats)

    out: Dict[str, Any] = {"window": "2023-05-08→2026-05-08", "legs": {}}
    for hz in (1, 5):
        fwd = np.full_like(mats["close"], np.nan)
        if hz < mats["close"].shape[0]:
            fwd[:-hz] = mats["close"][hz:] / mats["close"][:-hz] - 1.0
        for name, sc in scores.items():
            bp = bucket_profile(sc, fwd)
            tn = topn_profile(sc, fwd)
            out["legs"].setdefault(name, {})[f"fwd{hz}"] = {
                "buckets": bp.get("buckets", []),
                "top_minus_bottom_ann_pct": bp.get("top_minus_bottom_ann_pct"),
                "topN": tn,
            }
            print(f"\n=== {name} fwd{hz} ===")
            print(f"  top-bottom 年化差: {bp.get('top_minus_bottom_ann_pct'):+.2f}%")
            for r in bp.get("buckets", []):
                print(f"    {r['bucket']:14s} n={r['avg_n']:7.0f} "
                      f"年化均值={r['ann_mean_fwd_pct']:+7.2f}%  "
                      f"日均={r['mean_fwd']*100:+.4f}%")
            for k, v in tn.items():
                print(f"    {k:8s} 年化={v['ann_pct']:+7.2f}% "
                      f"年化波动={v['ann_vol_pct']:5.2f}% "
                      f"1日夏普={v['sharpe_1d']:+.3f}")

    path = os.path.join(L.OUT_DIR, "alpha_profile.json")
    with open(path, "w") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1, default=str)
    print(f"\n写入 {path}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
