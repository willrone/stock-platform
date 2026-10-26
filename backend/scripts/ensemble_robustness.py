#!/usr/bin/env python3
"""合成增益的稳健性总检：多候选 × 安慰剂 + 固定 OOS 评估。

## 回答三个问题
1. **增益来自合成还是来自"截断/扰动"？**
   对每个候选配置，跑 N 个安慰剂（破坏第二条腿的截面信号，保留 NaN 模式
   与边际分布），得到空分布 → 真实值在空分布中的分位。
2. **IS 上有效的配置，OOS 上还剩多少？**
   每个候选固定 (腿, 预处理, topk, n_drop)，IS/OOS 一起报。
3. **合成是否真的优于最好的单腿？**
   同 topk/n_drop 下对比单腿。

用法:
  env -u PYTHONPATH -u PYTHONHOME backend/.venv-py313/bin/python3 \
    backend/scripts/ensemble_robustness.py --n 20
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple

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


def cs_rank_norm(m: np.ndarray) -> np.ndarray:
    df = pd.DataFrame(m)
    cnt = df.notna().sum(axis=1)
    r = df.rank(axis=1, na_option="keep")
    n = df.notna().sum(axis=1).replace(0, np.nan)
    out = ((r - 0.5) / n.to_numpy()[:, None]).to_numpy()
    out[(cnt < 30).to_numpy()] = np.nan
    return out


PREPROC = {
    "raw": lambda m: m,
    "z": lambda m: L.cs_zscore(m, 30),
    "rank": cs_rank_norm,
    "winsor3": winsor,
}

# 候选：(名字, 腿列表, 预处理, topk, n_drop)
CANDIDATES: List[Tuple[str, List[str], str, int, int]] = [
    ("42f+mf|winsor3|k30|d5", ["42f", "mf"], "winsor3", 30, 5),
    ("42f+mf|winsor3|k40|d5", ["42f", "mf"], "winsor3", 40, 5),
    ("42f+rsi|winsor3|k20|d3", ["42f", "rsi"], "winsor3", 20, 3),
    ("42f+gbm|corrpen|k20|d2", ["42f", "gbm"], "z", 20, 2),
    ("42f+rsi|icir250|k20|d3", ["42f", "rsi"], "z", 20, 3),
    ("42f+fr|ortho|k60|d10", ["42f", "fr"], "z", 60, 10),
    ("42f+mf|z|k30|d5", ["42f", "mf"], "z", 30, 5),
]


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
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default="ensemble_robustness.json")
    args = ap.parse_args()

    mats = LEGS.cache_mats()
    mats["stock_data"] = {}
    trad = LEGS.cache_trad(mats, {})
    is_rows, oos_rows = L.split_index(mats["master"], OOS_SPLIT)

    leg_cache: Dict[str, np.ndarray] = {}
    for n in {x for _, ls, _, _, _ in CANDIDATES for x in ls} | {
            "42f", "rsi", "fr", "gbm", "mf"}:
        p = os.path.join(LEGS.LEG_DIR, f"{n}.npz")
        if os.path.exists(p):
            leg_cache[n] = np.load(p, allow_pickle=False)["score"]

    out_rows: List[Dict[str, Any]] = []
    for name, ls, prep, topk, n_drop in CANDIDATES:
        if any(x not in leg_cache for x in ls):
            print(f"!! 跳过 {name}（缺腿）", flush=True)
            continue
        fn = PREPROC[prep]
        cfg = S.SimConfig(topk=topk, n_drop=n_drop, lag=1,
                          rebalance_days="weekly",
                          max_position_size=(1 - 0.05) / topk)

        def ev(sc: np.ndarray) -> Dict[str, float]:
            pr = S.prepare_score(sc, mats, cfg)
            a = S.simulate(pr, mats, trad, cfg)
            i = S.simulate(pr, mats, trad, cfg, rows=is_rows)
            o = S.simulate(pr, mats, trad, cfg, rows=oos_rows)
            return {"full": float(a["sharpe_ratio"]),
                    "is": float(i["sharpe_ratio"]),
                    "oos": float(o["sharpe_ratio"]),
                    "ret": float(a["total_return"]),
                    "mdd": float(a["max_drawdown"]),
                    "n": int(a["total_trades"])}

        real = ev(blend_count([leg_cache[x] for x in ls], fn))
        # 单腿对照（同样的 topk/n_drop）
        singles = {}
        for x in ls:
            singles[x] = ev(leg_cache[x])
        # 安慰剂：破坏第二条腿的截面信号
        base = leg_cache[ls[0]]
        second = leg_cache[ls[1]]
        fin2 = np.isfinite(fn(second))
        placebos = []
        for seed in range(args.n):
            rng = np.random.default_rng(seed)
            perm = np.full_like(second, np.nan)
            for t in range(second.shape[0]):
                ok = np.where(np.isfinite(second[t]))[0]
                if len(ok) > 1:
                    perm[t, ok] = rng.permutation(ok)
            placebos.append(ev(blend_count([base, perm], fn)))
        row = {
            "name": name, "legs": ls, "prep": prep, "topk": topk,
            "n_drop": n_drop,
            "real": real, "singles": singles,
            "placebo_full": [p["full"] for p in placebos],
            "placebo_is": [p["is"] for p in placebos],
            "placebo_oos": [p["oos"] for p in placebos],
        }
        for f in ("full", "is", "oos"):
            v = np.array(row[f"placebo_{f}"])
            row[f"p_{f}"] = round(float((v >= real[f]).mean()), 3)
            row[f"null_{f}_mean"] = round(float(v.mean()), 3)
            row[f"null_{f}_sd"] = round(float(v.std()), 3)
            row[f"null_{f}_max"] = round(float(v.max()), 3)
        out_rows.append(row)
        best_single = max(singles.items(), key=lambda kv: kv[1]["full"])
        print(f"\n{name}  k={topk} d={n_drop}")
        print(f"  合成: 全窗={real['full']:6.3f} IS={real['is']:6.3f} "
              f"OOS={real['oos']:6.3f} ret={real['ret']:+7.2%} "
              f"mdd={real['mdd']:7.2%} n={real['n']}")
        for x, r in singles.items():
            print(f"  单腿 {x:6s}: 全窗={r['full']:6.3f} IS={r['is']:6.3f} "
                  f"OOS={r['oos']:6.3f}")
        print(f"  安慰剂空分布(全窗): mean={row['null_full_mean']:.3f} "
              f"sd={row['null_full_sd']:.3f} max={row['null_full_max']:.3f} "
              f"| p={row['p_full']}")
        print(f"  安慰剂空分布(IS)  : mean={row['null_is_mean']:.3f} "
              f"sd={row['null_is_sd']:.3f} max={row['null_is_max']:.3f} "
              f"| p={row['p_is']}")
        print(f"  安慰剂空分布(OOS) : mean={row['null_oos_mean']:.3f} "
              f"sd={row['null_oos_sd']:.3f} max={row['null_oos_max']:.3f} "
              f"| p={row['p_oos']}", flush=True)

    path = os.path.join(L.OUT_DIR, args.out)
    with open(path, "w") as fh:
        json.dump({"candidates": out_rows}, fh, ensure_ascii=False, indent=1,
                  default=str)
    print(f"\n写入 {path}")


if __name__ == "__main__":
    main()
