#!/usr/bin/env python3
"""10 年风险平价 DSR 终审（纯 numpy+math, 无引擎依赖）。

读 risk_parity_10y_experiment.json 持久化的逐日 returns, 对每个候选算:
  DSR = Bailey-Lopez de Prado deflated Sharpe (N=40 独立试验, Gumbel SR0, 偏峰度校正)
  PBO = CSCV 组合交叉验证 (lookback 变体 20/40/60 + 等权)
口径与 3 年 DSR (risk_parity_dsr_final) 完全一致, 仅窗口 3→10 年(T 722→2423)。
目的: 3 年 12/12 fail 的最终长尺度裁决——10 年 R1 夏普掉到 0.573, DSR 是否更彻底 fail。
产物: data/strategy_baseline/risk_parity_10y_dsr_final.json
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
from datetime import datetime

import numpy as np

DATA_ROOT = "/Users/rone/projects/stock-platform/data"
IN_JSON = os.path.join(DATA_ROOT, "strategy_baseline", "risk_parity_10y_experiment.json")
OUT_JSON = os.path.join(DATA_ROOT, "strategy_baseline", "risk_parity_10y_dsr_final.json")
N_TRIALS = 40  # 五轮研究独立试验数(保守上界, 与 3 年 DSR 同口径)
SQRT252 = math.sqrt(252.0)


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _skew_kurt(x: np.ndarray) -> tuple:
    x = np.asarray(x, dtype=float)
    mu = x.mean()
    m2 = float(((x - mu) ** 2).mean())
    m3 = float(((x - mu) ** 3).mean())
    m4 = float(((x - mu) ** 4).mean())
    skew = m3 / m2 ** 1.5 if m2 > 0 else 0.0
    kurt = m4 / m2 ** 2 if m2 > 0 else 3.0
    return skew, kurt


def deflated_sharpe(daily: np.ndarray, n_trials: int = N_TRIALS) -> dict:
    """与 3 年 DSR 逐位一致的 Bailey-Lopez de Prado 公式。"""
    r = np.asarray(daily, dtype=float)[1:]  # 去掉首日 0
    T = len(r)
    mu, sd = r.mean(), r.std(ddof=1)
    sr_d = mu / sd if sd > 0 else 0.0
    sr_ann = sr_d * SQRT252
    skew, kurt = _skew_kurt(r)
    var_sr = max(1e-12, (1 - skew * sr_d + (kurt - 1) / 4 * sr_d ** 2) / T)
    sd_sr = math.sqrt(var_sr)
    sr0 = sd_sr * math.sqrt(2.0 * math.log(max(2, n_trials)))
    dsr = norm_cdf((sr_d - sr0) / sd_sr)
    return {
        "sharpe_annualized": round(sr_ann, 4),
        "n_obs": T,
        "skew": round(skew, 4),
        "kurt": round(kurt, 4),
        "sr0_threshold_ann": round(sr0 * SQRT252, 4),
        "deflated_sharpe": round(dsr, 5),
        "pass_95": bool(dsr > 0.95),
        "n_trials": n_trials,
    }


def pbo_cscv(rets: dict, n_splits: int = 200, seed: int = 7) -> dict:
    """与 3 年 DSR 逐位一致。"""
    names = list(rets)
    R = np.stack([np.asarray(rets[n], dtype=float)[1:] for n in names], axis=1)
    T, V = R.shape
    acc = 0.0
    rng = np.random.default_rng(seed)
    for i in range(n_splits):
        idx = rng.permutation(T)
        A, B = R[idx[: T // 2]], R[idx[T // 2:]]
        sa, sb = A.std(0, ddof=1), B.std(0, ddof=1)
        srA = A.mean(0) / np.where(sa > 0, sa, 1.0) * SQRT252
        srB = B.mean(0) / np.where(sb > 0, sb, 1.0) * SQRT252
        best = int(np.argmax(srA))
        acc += (1 + int((srB > srB[best]).sum())) / V
    pbo = acc / n_splits
    return {"pbo": round(float(pbo), 4), "n_variants": V, "n_splits": n_splits, "variants": names}


def main() -> None:
    with open(IN_JSON) as f:
        d = json.load(f)
    res = d["results"]
    print(f"加载 {IN_JSON}")
    print(f"窗口: {d['meta'].get('window')}  生成: {d['meta'].get('generated_at')}")

    out = {
        "meta": {
            "generated_at": datetime.now().isoformat(),
            "source": "risk_parity_10y_experiment.json",
            "n_trials_for_dsr": N_TRIALS,
            "window": d["meta"].get("window"),
            "note": "10年(2016-05-12→2026-05-08)风险平价DSR终审; 4329 point-in-time宇宙; stock_data前复权; 时段化跳变清洗; 公式与3年DSR一致; 夏普=μ/σ·√252",
        },
        "dsr": {},
        "pbo": {},
    }

    def save():
        with open(OUT_JSON, "w") as f:
            json.dump(out, f, ensure_ascii=False, indent=1, default=str)

    # 有 returns 的候选
    cand = {k: np.asarray(v["returns"], dtype=float) for k, v in res.items() if "returns" in v}
    print(f"\n候选(带returns): {list(cand)}\n")

    # 组合变体(风险平价 lookback) 用于 PBO: R1 60d / R1 40d(若有) / R1 20d / R0
    pbo_rets = {}
    for k in cand:
        if k.startswith("R"):
            pbo_rets[k] = cand[k]
    out["pbo"] = pbo_cscv(pbo_rets)

    for k, r in cand.items():
        ds = deflated_sharpe(r)
        out["dsr"][k] = ds
        s = res[k]["stats"]
        print(f"[DSR] {k}")
        print(f"   夏普(脚本stats)={s['sharpe_ratio']}  年化={s['annualized_return']*100:+.1f}%  "
              f"总={s['total_return']*100:+.0f}%  MDD={s['max_drawdown']*100:.0f}%")
        print(f"   → DSR={ds['deflated_sharpe']:<10} SR0门槛(年化)={ds['sr0_threshold_ann']}  "
              f"偏度={ds['skew']} 峰度={ds['kurt']}  N={ds['n_obs']}  过95%={'✓' if ds['pass_95'] else '✗'}")
    save()

    print(f"\n[PBO] {out['pbo']['pbo']}  across {out['pbo']['n_variants']} 变体 {out['pbo']['variants']}")
    save()

    # 与 3 年 DSR 对照
    d3 = os.path.join(DATA_ROOT, "strategy_baseline", "risk_parity_dsr_final.json")
    if os.path.exists(d3):
        with open(d3) as f:
            j3 = json.load(f)
        ds3 = j3.get("dsr", {})
        print("\n=== 3年 vs 10年 DSR 对照 (同公式, 不同窗口) ===")
        print(f"{'候选':<22} {'3年夏普':>8} {'3年DSR':>8} {'10年夏普':>8} {'10年DSR':>8}")
        # 3年候选名 → 10年候选名 映射
        m3 = {"R1_lb60": "R1_风险平价月度60d", "R1_lb20": "R2_风险平价月度20d",
              "R0_等权": "R0_等权13", "C_stochastic单腿": "C_stochastic单腿",
              "C_factor_ranking单腿": "C_factor_ranking单腿", "C_bollinger单腿": "C_bollinger单腿"}
        for k3, k10 in m3.items():
            if k3 in ds3 and k10 in out["dsr"]:
                a3, a10 = ds3[k3], out["dsr"][k10]
                print(f"{k10:<22} {a3['sharpe_annualized']:>8.3f} {a3['deflated_sharpe']:>8.3f} "
                      f"{a10['sharpe_annualized']:>8.3f} {a10['deflated_sharpe']:>8.3f}")
    print(f"\n[saved] {OUT_JSON}")


if __name__ == "__main__":
    main()
