#!/usr/bin/env python3
"""Deflated Sharpe + PBO 统计终审（2026-10-27）

## 为什么必须做
本轮 + 子代理一共做了 **80+ 次变体尝试**。3 年（727 交易日）窗口上反复试参数，
样本内夏普必然虚高 —— 哪怕每次都诚实、每次都报 OOS，「从 82 个里挑最好的那个」
这件事本身就会抬高夏普。

Bailey & López de Prado (2014) 的 DSR 把「试了 N 次」折算进显著性检验。
本脚本沿用**项目既有约定**（`backend/scripts/risk_parity_dsr_final.py` 的
「正态极值近似」版，纯 numpy/math，无 scipy 依赖）：

    sr_d   = μ/σ                        日度夏普
    var_sr = (1 - γ₃·sr_d + (γ₄-1)/4·sr_d²) / T
    sd_sr  = √var_sr
    SR0    = sd_sr · √(2 ln N)          N 次尝试下「运气最优策略」的期望夏普
    DSR    = Φ((sr_d - SR0) / sd_sr)    >0.95 = 显著

注意：SR0 随 T 增大而降低（sd_sr ∝ 1/√T），所以 **3 年窗口的门槛远高于 10 年**。
本项目历史终审（N=40，12 候选）全部未过线（风险平价 1.24 的 DSR 仅 0.28，
SR0 门槛 1.56）—— 与本次结果相互印证。

## PBO
同一批候选的日收益矩阵做 200 次随机 50/50 对半分，前半选最优 → 记录后半排名。
PBO > 0.5 = 前半选出的「最优」在后半系统性跑输 = 选择过拟合证据。

## 用法
  env -u PYTHONPATH -u PYTHONHOME backend/.venv-py313/bin/python3 \
    backend/scripts/verify_deflated_sharpe.py --trials 82
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "data", "backtest", "deflated_sharpe.json")
NAV_CACHE = os.path.join(ROOT, "data", "backtest", "_blend_nav_25f.json")
SQRT252 = math.sqrt(252.0)


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _skew_kurt(x):
    import numpy as np
    x = np.asarray(x, dtype=float)
    mu = x.mean()
    m2 = float(((x - mu) ** 2).mean())
    m3 = float(((x - mu) ** 3).mean())
    m4 = float(((x - mu) ** 4).mean())
    skew = m3 / m2 ** 1.5 if m2 > 0 else 0.0
    kurt = m4 / m2 ** 2 if m2 > 0 else 3.0
    return skew, kurt


def deflated_sharpe(daily, n_trials: int) -> dict:
    """项目约定版 DSR。daily 含 r[0]=0（净值起点对齐）。"""
    import numpy as np
    r = np.asarray(daily, dtype=float)[1:]
    T = len(r)
    if T < 60:
        return {"error": "too short"}
    mu, sd = r.mean(), r.std(ddof=1)
    sr_d = mu / sd if sd > 0 else 0.0
    sr_ann = sr_d * SQRT252
    skew, kurt = _skew_kurt(r)
    var_sr = max(1e-12, (1 - skew * sr_d + (kurt - 1) / 4 * sr_d ** 2) / T)
    sd_sr = math.sqrt(var_sr)
    sr0 = sd_sr * math.sqrt(2.0 * math.log(max(2, n_trials)))
    dsr = norm_cdf((sr_d - sr0) / sd_sr)
    eq = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(eq)
    mdd = float(((eq - peak) / peak).min())
    return {
        "sharpe_annualized": round(sr_ann, 4),
        "n_obs": T,
        "skew": round(skew, 4),
        "kurt": round(kurt, 4),
        "sr0_threshold_annualized": round(sr0 * SQRT252, 4),
        "deflated_sharpe": round(dsr, 5),
        "pass_95": bool(dsr > 0.95),
        "n_trials": n_trials,
        "max_drawdown": round(mdd, 6),
        "total_return": round(float(eq[-1] - 1.0), 6),
    }


def pbo_cscv(rets: dict, n_splits: int = 200, seed: int = 7) -> dict:
    """PBO: 200 次随机 50/50 对半分, 前半选优→后半排名均值/V。"""
    import numpy as np
    names = list(rets)
    if len(names) < 2:
        return {"pbo": None, "note": "候选不足"}
    R = np.stack([np.asarray(rets[n], dtype=float)[1:] for n in names], axis=1)
    T, V = R.shape
    acc = 0.0
    rng = np.random.default_rng(seed)
    for _ in range(n_splits):
        perm = rng.permutation(T)
        h1, h2 = perm[: T // 2], perm[T // 2:]
        s1 = R[h1].mean(axis=0) / R[h1].std(axis=0, ddof=1)
        s2 = R[h2].mean(axis=0) / R[h2].std(axis=0, ddof=1)
        best = int(np.argmax(s1))
        rank = float((s2 < s2[best]).sum() + 1) / V
        acc += rank
    return {"pbo": round(acc / n_splits, 4), "n_variants": V,
            "n_splits": n_splits, "variants": names}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=82,
                    help="总变体尝试次数（决定 SR0 门槛）")
    args = ap.parse_args()

    import numpy as np
    import pandas as pd

    if not os.path.exists(NAV_CACHE):
        print(f"[dsr] 缺少 {NAV_CACHE}")
        return 1
    with open(NAV_CACHE) as fh:
        d = json.load(fh)
    navs = {}
    for k, v in d["navs"].items():
        s = pd.Series(v)
        s.index = pd.to_datetime(s.index)
        navs[k] = s.sort_index()
    idx = None
    for s in navs.values():
        idx = s.index if idx is None else idx.union(s.index)
    idx = idx.sort_values()

    rets = {}
    for k, s in navs.items():
        rets[k] = (s.reindex(idx).ffill().pct_change(fill_method=None)
                   .fillna(0.0).to_numpy())

    print(f"[dsr] 尝试次数 N={args.trials}  候选 {len(rets)} 条  "
          f"样本 {len(idx)} 交易日", flush=True)
    print(f"{'策略':<22}{'夏普':>8}{'SR0门槛':>10}{'DSR':>9}{'过95%':>8}")
    print("-" * 60)
    results = {}
    for k, r in rets.items():
        ds = deflated_sharpe(r, args.trials)
        if "error" in ds:
            continue
        results[k] = ds
        print(f"{k:<22}{ds['sharpe_annualized']:>8.3f}"
              f"{ds['sr0_threshold_annualized']:>10.3f}"
              f"{ds['deflated_sharpe']:>9.4f}"
              f"{'  PASS' if ds['pass_95'] else '  FAIL':>8}", flush=True)

    print("\n[dsr] PBO（候选间选择过拟合）", flush=True)
    pbo = pbo_cscv(rets)
    print(f"  PBO = {pbo.get('pbo')}  候选 {pbo.get('n_variants')}  "
          f"（>0.5 = 前半选出的最优在后半系统性跑输）", flush=True)

    print("\n[dsr] 反向测算：给定年化夏普 X，要多少交易日才能在 N 次尝试下 95% 显著",
          flush=True)
    base = results.get("honest_composite") or next(iter(results.values()))
    T_needed = {}
    for target in (0.88, 1.0, 1.2, 1.5, 2.0):
        need = None
        for n_needed in (243, 486, 727, 1000, 1500, 2000, 2500, 3000, 4000, 5000):
            sr_d = target / SQRT252
            var_sr = (1 - base["skew"] * sr_d
                      + (base["kurt"] - 1) / 4 * sr_d ** 2) / n_needed
            sd_sr = math.sqrt(max(var_sr, 1e-12))
            sr0 = sd_sr * math.sqrt(2.0 * math.log(args.trials))
            if norm_cdf((sr_d - sr0) / sd_sr) > 0.95:
                need = n_needed
                break
        T_needed[str(target)] = need
        if need:
            print(f"  夏普 {target:.2f} → 需 ≥{need} 个交易日"
                  f"（{need/243:.1f} 年）", flush=True)
        else:
            print(f"  夏普 {target:.2f} → 即使 5000 个交易日也不显著", flush=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump({
            "methodology": {
                "n_trials": args.trials,
                "formula": "项目约定版 DSR（正态极值近似，Bailey & LdP 2014）",
                "window": f"{idx[0].date()}→{idx[-1].date()}",
                "n_obs": len(idx),
            },
            "results": results,
            "pbo": pbo,
            "days_needed_for_95pct": T_needed,
        }, fh, ensure_ascii=False, indent=1)
    print(f"\n[dsr] 已写入 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
