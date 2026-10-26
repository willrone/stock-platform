#!/usr/bin/env python3
"""安慰剂检验：合成增益是真的信号，还是选股扰动的噪声？

## 做法
把第二条腿换成**安慰剂腿**（保留 NaN 模式 / 分布特征，但破坏信号）：
  A. 同 NaN 模式 + 标准正态随机值
  B. 同 NaN 模式 + mf 截面内随机置换（保留边际分布，破坏截面信息）
对每个安慰剂种子跑完整回测，得到「空分布」。

如果真实腿的夏普落在安慰剂分布内 → 增益是噪声，不能信。

用法:
  env -u PYTHONPATH -u PYTHONHOME backend/.venv-py313/bin/python3 \
    backend/scripts/ensemble_placebo.py --n 30
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

import numpy as np
import pandas as pd

BACKEND = "/Users/rone/Projects/stock-platform/backend"
ROOT = "/Users/rone/Projects/stock-platform"
sys.path.insert(0, BACKEND)
sys.path.insert(0, os.path.join(BACKEND, "scripts"))

import ensemble_lab as L  # noqa: E402
import ensemble_legs as LEGS  # noqa: E402
import ensemble_sim as S  # noqa: E402

OOS_SPLIT = pd.Timestamp("2025-05-08")


def winsor(m: np.ndarray, clip: float = 3.0) -> np.ndarray:
    z = L.cs_zscore(m, 30)
    return np.where(np.isfinite(z), np.clip(z, -clip, clip), np.nan)


def blend(fs: List[np.ndarray], fn) -> np.ndarray:
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
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--base", default="42f")
    ap.add_argument("--second", default="mf")
    ap.add_argument("--clip", type=float, default=3.0)
    ap.add_argument("--topk", type=int, default=30)
    ap.add_argument("--n-drop", type=int, default=5)
    ap.add_argument("--out", default="ensemble_placebo.json")
    args = ap.parse_args()

    mats = LEGS.cache_mats()
    mats["stock_data"] = {}
    trad = LEGS.cache_trad(mats, {})
    base = np.load(os.path.join(LEGS.LEG_DIR, f"{args.base}.npz"),
                   allow_pickle=False)["score"]
    second = np.load(os.path.join(LEGS.LEG_DIR, f"{args.second}.npz"),
                     allow_pickle=False)["score"]
    is_rows, oos_rows = L.split_index(mats["master"], OOS_SPLIT)
    cfg = S.SimConfig(topk=args.topk, n_drop=args.n_drop, lag=1,
                      rebalance_days="weekly",
                      max_position_size=(1 - 0.05) / args.topk)
    fn = lambda m: winsor(m, args.clip)  # noqa: E731

    def evaluate(mat: np.ndarray) -> Dict[str, float]:
        pr = S.prepare_score(blend([base, mat], fn), mats, cfg)
        a = S.simulate(pr, mats, trad, cfg)
        i = S.simulate(pr, mats, trad, cfg, rows=is_rows)
        o = S.simulate(pr, mats, trad, cfg, rows=oos_rows)
        return {"full": float(a["sharpe_ratio"]),
                "is": float(i["sharpe_ratio"]),
                "oos": float(o["sharpe_ratio"]),
                "ret": float(a["total_return"]),
                "mdd": float(a["max_drawdown"])}

    real = evaluate(second)
    print(f"真实腿 {args.base}+{args.second}: 全窗 SR={real['full']:.3f} "
          f"IS={real['is']:.3f} OOS={real['oos']:.3f}", flush=True)

    fin2 = np.isfinite(winsor(second, args.clip))
    results: List[Dict[str, Any]] = []
    for seed in range(args.n):
        rng = np.random.default_rng(seed)
        # A: 同 NaN 模式 + 正态随机
        fake = np.where(fin2, rng.standard_normal(fin2.shape), np.nan)
        ra = evaluate(fake)
        # B: 截面内置换
        perm = np.full_like(second, np.nan)
        for t in range(second.shape[0]):
            ok = np.where(np.isfinite(second[t]))[0]
            if len(ok) > 1:
                perm[t, ok] = rng.permutation(ok)
        rb = evaluate(perm)
        results.append({"seed": seed, "A_random": ra, "B_permuted": rb})
        print(f"  seed={seed:3d} A: full={ra['full']:6.3f} oos={ra['oos']:6.3f} "
              f"| B: full={rb['full']:6.3f} oos={rb['oos']:6.3f}", flush=True)

    for tag in ("A_random", "B_permuted"):
        for field in ("full", "is", "oos"):
            vals = np.array([r[tag][field] for r in results])
            p = float((vals >= real[field]).mean())
            print(f"{tag:12s} {field:5s}: null mean={vals.mean():.3f} "
                  f"sd={vals.std():.3f} min={vals.min():.3f} max={vals.max():.3f} "
                  f"| real={real[field]:.3f}  p(placebo>=real)={p:.3f}")

    out = {"base": args.base, "second": args.second, "clip": args.clip,
           "topk": args.topk, "n_drop": args.n_drop,
           "real": real, "placebo": results}
    path = os.path.join(L.OUT_DIR, args.out)
    with open(path, "w") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1, default=str)
    print(f"写入 {path}")


if __name__ == "__main__":
    main()
