#!/usr/bin/env python3
"""方向 D: 质量加权 + regime 门 (10 年, 纯组合层, 零回测)。

背景: 10 年终裁证伪无条件逆波动配权 (R1 0.573 < R0 0.638) — 机制是"押低波",
而 factor_ranking 2016-23 夏普≈0 → 55% 权重押在最弱腿上拖垮组合。
本刀把配权从"押低波"改为"押质量(滚动夏普)", 并加 regime 门 (熊市空仓)。

变体 (同管线 returns, 与 risk_parity_10y_experiment 逐位同源; 质量窗止于 t-2 无未来函数):
  Q20/Q40/Q60      滚动夏普质量加权 (Sharpe>0 的腿按夏普加权, 负夏普腿权重 0 再归一)
  Q60F             同上但负夏普腿保留负权重(对照, 展示"押质量"不加门的风险)
  G60              Q60 + R0 均线门: MA60>MA120(质量窗, ≤t-2) → 质量权重, 否则 100% 现金
  G60L             Q60 + 慢门: MA40>MA120 (降低熊市滞后成本)
裁决三关: (1) 10 年夏普 > R1 0.573 且 > R0 0.638; (2) 邻格同向;
  (3) 分段: 2016-23(熊牛轮转) 不跑输现金 + 2023-26(3年研究窗) 保住 edge。
另算 DSR(公式同 3年/10年终审) + PBO(变体间)。
产物: data/strategy_baseline/quality_weight_10y.json
"""
from __future__ import annotations

import json
import math
import os
from datetime import datetime

import numpy as np

DATA_ROOT = "/Users/rone/projects/stock-platform/data"
IN_JSON = os.path.join(DATA_ROOT, "strategy_baseline", "risk_parity_10y_experiment.json")
OUT_JSON = os.path.join(DATA_ROOT, "strategy_baseline", "quality_weight_10y.json")
LEGS = ["stochastic", "factor_ranking", "bollinger"]
SQRT252 = math.sqrt(252.0)
N_TRIALS = 40


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _skew_kurt(x: np.ndarray) -> tuple:
    x = np.asarray(x, dtype=float)
    mu = x.mean()
    m2 = float(((x - mu) ** 2).mean())
    m3 = float(((x - mu) ** 3).mean())
    m4 = float(((x - mu) ** 4).mean())
    return (m3 / m2 ** 1.5 if m2 > 0 else 0.0, m4 / m2 ** 2 if m2 > 0 else 3.0)


def deflated_sharpe(daily: np.ndarray, n_trials: int = N_TRIALS) -> dict:
    r = np.asarray(daily, dtype=float)[1:]
    T = len(r)
    mu, sd = r.mean(), r.std(ddof=1)
    sr_d = mu / sd if sd > 0 else 0.0
    skew, kurt = _skew_kurt(r)
    var_sr = max(1e-12, (1 - skew * sr_d + (kurt - 1) / 4 * sr_d ** 2) / T)
    sd_sr = math.sqrt(var_sr)
    sr0 = sd_sr * math.sqrt(2.0 * math.log(max(2, n_trials)))
    dsr = norm_cdf((sr_d - sr0) / sd_sr)
    return {
        "sharpe_annualized": round(sr_d * SQRT252, 4), "n_obs": T,
        "skew": round(skew, 4), "kurt": round(kurt, 4),
        "sr0_threshold_ann": round(sr0 * SQRT252, 4),
        "deflated_sharpe": round(dsr, 5), "pass_95": bool(dsr > 0.95),
    }


def pbo_cscv(rets: dict, n_splits: int = 200, seed: int = 7) -> dict:
    names = list(rets)
    R = np.stack([np.asarray(rets[n], dtype=float)[1:] for n in names], axis=1)
    T, V = R.shape
    acc = 0.0
    rng = np.random.default_rng(seed)
    for _ in range(n_splits):
        idx = rng.permutation(T)
        A, B = R[idx[: T // 2]], R[idx[T // 2:]]
        sa, sb = A.std(0, ddof=1), B.std(0, ddof=1)
        srA = A.mean(0) / np.where(sa > 0, sa, 1.0) * SQRT252
        srB = B.mean(0) / np.where(sb > 0, sb, 1.0) * SQRT252
        best = int(np.argmax(srA))
        acc += (1 + int((srB > srB[best]).sum())) / V
    return {"pbo": round(acc / n_splits, 4), "n_variants": V, "variants": names}


def stats_from_returns(r: np.ndarray, dates: list, init=1_000_000.0) -> dict:
    eq = init * np.cumprod(1.0 + r)
    vals = np.concatenate([[init], eq])
    rets = np.diff(vals) / np.where(vals[:-1] > 0, vals[:-1], 1.0)
    years = max(1.0, len(rets) / 252.0)
    total = vals[-1] / vals[0] - 1.0
    ann = (1 + total) ** (1 / years) - 1 if total > -1 else -1.0
    vol = float(rets.std(ddof=1) * SQRT252)
    sharpe = float(rets.mean() / rets.std(ddof=1) * SQRT252) if rets.std() > 0 else 0.0
    peak = np.maximum.accumulate(vals)
    mdd = float(((vals - peak) / peak).min())
    return {
        "total_return": round(total, 6), "annualized_return": round(ann, 6),
        "volatility": round(vol, 6), "sharpe_ratio": round(sharpe, 4),
        "max_drawdown": round(mdd, 6),
    }


def yearly_from_returns(r: np.ndarray, dates: list) -> dict:
    by: dict = {}
    cum = 1.0
    prev_end = 1.0
    last_year = None
    for d, ri in zip(dates, r):
        y = int(str(d)[:4])
        if last_year is not None and y != last_year:
            by[last_year] = cum / prev_end - 1.0
            prev_end = cum
        cum *= 1.0 + ri
        last_year = y
    if last_year is not None:
        by[last_year] = cum / prev_end - 1.0
    return {str(y): round(v, 4) for y, v in by.items()}


def subperiod(r: np.ndarray, dates: list, lo: str, hi: str) -> dict:
    dt = np.array(dates, dtype="datetime64[D]")
    m = (dt >= np.datetime64(lo)) & (dt < np.datetime64(hi))
    rs = r[m]
    eq = np.cumprod(1.0 + rs)
    tot = eq[-1] / eq[0] - 1.0
    yrs = max(len(rs) / 252.0, 1e-9)
    mu, sd = rs[1:].mean(), rs[1:].std(ddof=1)
    return {
        "days": int(m.sum()), "total": round(tot, 4),
        "ann": round((1 + tot) ** (1 / yrs) - 1 if tot > -1 else -1.0, 4),
        "sharpe": round(mu / sd * SQRT252 if sd > 0 else 0.0, 4),
    }


def main() -> None:
    with open(IN_JSON) as f:
        d = json.load(f)
    res = d["results"]
    leg_r = {s: np.asarray(res[f"C_{s}单腿"]["returns"], dtype=float) for s in LEGS}
    dates = res["R0_等权13"]["dates"]
    r0 = np.asarray(res["R0_等权13"]["returns"], dtype=float)
    r1 = np.asarray(res["R1_风险平价月度60d"]["returns"], dtype=float)
    n = len(dates)
    assert all(len(x) == n for x in leg_r.values()) and len(r0) == n

    # R0 组合日收益 = 等权 (门信号源, 与管线同口径)
    r0_from_legs = np.mean([leg_r[s] for s in LEGS], axis=0)
    assert np.allclose(r0_from_legs, r0, atol=1e-12), "R0 与单腿等权不一致"

    def quality_w(t: int, lb: int, gate: int | None, floor: bool = True) -> np.ndarray:
        """t 日(重平衡日)权重: 质量窗 ≤t-2, 门同窗口径。返回三腿权重(现金=1-sum)。"""
        win = [leg_r[s][max(0, t - lb): t - 1] for s in LEGS]  # 止于 t-2, 无未来函数
        if gate is not None:
            gw = r0_from_legs[max(0, t - gate): t - 1]  # ≤t-2, 无未来函数
            # 切片 [t-gate, t-2] 含 gate-1 个元素; 要求短 MA 的 20 个样本即可, 长 MA 用全部可用
            if len(gw) >= 20:
                short_ma = np.mean(gw[-20:])
                long_ma = np.mean(gw)
                if short_ma <= long_ma:
                    return np.zeros(3)  # 熊市 → 全现金
        w = np.zeros(3)
        for i, s in enumerate(LEGS):
            if len(win[i]) < 15:  # 窗含 lb-1 个样本(lb=20→19); 15 以下才退等权
                w[i] = 1.0 / 3
                continue
            mu, sd = win[i].mean(), win[i].std(ddof=1)
            q = mu / sd * SQRT252 if sd > 0 else 0.0
            w[i] = max(q, 0.0) if floor else q
        ssum = w.sum()
        if ssum <= 1e-6:
            w = np.ones(3) / 3  # 全腿负/零夏普 → 退回等权(保守)
        else:
            w = w / ssum
        return w

    variants = {
        "Q20": dict(lb=20, gate=None), "Q40": dict(lb=40, gate=None), "Q60": dict(lb=60, gate=None),
        "Q60F": dict(lb=60, gate=None, floor=False),
        "G60": dict(lb=60, gate=60), "G60L": dict(lb=60, gate=120),
    }

    out = {
        "meta": {
            "generated_at": datetime.now().isoformat(),
            "source": os.path.basename(IN_JSON),
            "window": d["meta"].get("window"),
            "note": "方向D: 质量加权(滚动夏普, 窗≤t-2, 月度重平衡) + regime门(R0 MA60/MA120 熊市现金); 纯组合层零回测; 与 10 年风险管线同 returns 源",
        },
        "results": {},
        "dsr": {},
        "pbo": {},
    }

    def save():
        with open(OUT_JSON, "w") as f:
            json.dump(out, f, ensure_ascii=False, indent=1, default=str)

    port_rets = {}
    for tag, v in variants.items():
        w = np.zeros((n, 3))
        last = np.ones(3) / 3
        for t in range(n):
            if t > 0 and t % 21 == 0:
                last = quality_w(t, v["lb"], v["gate"], floor=v.get("floor", True))
            w[t] = last
        r = np.zeros(n)
        for i, s in enumerate(LEGS):
            r += leg_r[s] * w[:, i]
        port_rets[tag] = r
        out["results"][tag] = {
            "stats": stats_from_returns(r, dates),
            "yearly": yearly_from_returns(r, dates),
            "sub_2016_23": subperiod(r, dates, "2016-05-12", "2023-05-12"),
            "sub_2023_26": subperiod(r, dates, "2023-05-12", "2026-05-09"),
            "avg_weights": {s: round(float(w[int(n * 0.1):, i].mean()), 3) for i, s in enumerate(LEGS)},
            "returns": [float(x) for x in r],
        }
        s = out["results"][tag]["stats"]
        print(f"[{tag}] sharpe={s['sharpe_ratio']} ann={s['annualized_return']*100:+.1f}% "
              f"total={s['total_return']*100:+.0f}% mdd={s['max_drawdown']*100:.0f}% "
              f"avg_w={out['results'][tag]['avg_weights']}")
    save()

    # 基准返回 (供对照)
    port_rets["R0_等权"] = r0
    port_rets["R1_风险平价60d"] = r1
    for base, r in (("R0_等权", r0), ("R1_风险平价60d", r1)):
        out["dsr"][base] = deflated_sharpe(r)

    # DSR + 三关
    print("\n=== 三关裁决 ===")
    base_sh = {k: res[k]["stats"]["sharpe_ratio"] for k in ("R0_等权13", "R1_风险平价月度60d")}
    for tag in variants:
        ds = deflated_sharpe(port_rets[tag])
        out["dsr"][tag] = ds
        s = out["results"][tag]["stats"]
        yr = out["results"][tag]["yearly"]
        bull = {y: v for y, v in yr.items() if int(y) >= 2023}
        bear = {y: v for y, v in yr.items() if int(y) <= 2022}
        bull_min = min(bull.values())
        bear_tot = float(np.prod([1 + v for v in bear.values()])) - 1
        g1 = s["sharpe_ratio"] > base_sh["R0_等权13"] > 0  # 跑赢等权(更严格)
        print(f"[{tag}] sharpe={s['sharpe_ratio']:.3f} DSR={ds['deflated_sharpe']} "
              f"bear2016-22总={bear_tot*100:+.1f}% 熊段最弱年={min(bear.values())*100:+.1f}% "
              f"牛段最弱年(2023+)={bull_min*100:+.1f}%  关1(>R0 {'0.638' if tag else ''})={'PASS' if g1 else 'fail'}")
    # 关2: 邻格同向 (Q20/Q40/Q60 方向一致, 以 R0 为基准)
    shs = {t: out["results"][t]["stats"]["sharpe_ratio"] for t in ("Q20", "Q40", "Q60")}
    same_dir = len({s > base_sh["R0_等权13"] for s in shs.values()}) == 1
    print(f"关2(邻格 Q20/Q40/Q60 同向): {shs} → {'PASS' if same_dir else 'FAIL'}")
    out["pbo"] = pbo_cscv({k: port_rets[k] for k in variants})
    print(f"[PBO] {out['pbo']['pbo']} across {out['pbo']['variants']}")
    save()
    print(f"[saved] {OUT_JSON}")


if __name__ == "__main__":
    main()
