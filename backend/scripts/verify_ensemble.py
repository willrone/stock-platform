#!/usr/bin/env python3
"""多策略信号合成 —— 真实盘口径批量实验 + OOS 验证。

## 口径（与 verify_10y_pit.py --lag 1 逐项一致）
  窗口 2023-05-08 → 2026-05-08 / 727 交易日 / 5238 只 PIT 股票池
  lag=1 / enable_tradability=True / 佣金 0.03% + 滑点 0.01% / 整手 100 /
  现金预留 5% / 周频调仓（周五）

## 腿的分数从**真实策略类**抽（ensemble_legs.py）
  42f  honest_composite 42 因子等权复合      引擎基线 SR 0.7992
  rsi  rsi_71421  RSI(7/14/21) 均值回归      引擎基线 SR 0.7716
  fr   factor_ranking 动量/低波动/低振幅
  gbm  gbm_voltgt  LightGBM 元模型 + 波动率目标仓位
  gbm_raw  gbm 但关掉波动率目标截断（保留原始预测分）
  mf   multi_factor value/momentum/low-vol

## 合成方法
  C1 等权（4 种预处理：raw / 截面 z / 截面 rank / winsorized z）
  C2 滚动 RankICIR 加权（walk-forward，无前视，权重做时序平滑）
  C3 相关性惩罚权重 w ∝ IR/(1+ρ)  与  逆协方差最小方差
  C4 残差正交化（腿对其它腿做截面回归取残差 → 构造低相关 alpha）
  C5 分层合成（每腿各取前 N 只，腿间轮流分配名额）

## 防过拟合
  IS = 2023-05→2025-05（选参）  OOS = 2025-05→2026-05（验证）
  报告总尝试次数；OOS 只用 IS 选出的那一组配置。

## 用法
  env -u PYTHONPATH -u PYTHONHOME backend/.venv-py313/bin/python3 \
    backend/scripts/verify_ensemble.py --out ensemble_results.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(BACKEND)
sys.path.insert(0, BACKEND)
sys.path.insert(0, os.path.join(BACKEND, "scripts"))

import ensemble_lab as L  # noqa: E402
import ensemble_legs as LEGS  # noqa: E402
import ensemble_sim as S  # noqa: E402

OOS_SPLIT = pd.Timestamp("2025-05-08")

# 引擎实测基线（2026-10-27 用 verify_10y_pit.py --lag 1 重跑，仓库根目录）
ENGINE_LEGS = {
    # 2026-10-27 用 verify_10y_pit.py --years 3 --lag 1 从仓库根**重跑**实测
    # （不是沿用旧表 —— 旧表里 rsi 1.0147 / mf 0.950 与当前代码不符）
    "42f": {"sharpe_ratio": 0.7991795250472001,
            "total_return": 0.34779248262737783,
            "max_drawdown": -0.16460289642093642, "total_trades": 1080},
    "rsi": {"sharpe_ratio": 0.771559886815966,
            "total_return": 0.35295059711506194,
            "max_drawdown": -0.19887429856304242, "total_trades": 726},
    "fr": {"sharpe_ratio": 0.6201707703609705,
           "total_return": 0.251172632980105,
           "max_drawdown": -0.16225955358231306, "total_trades": 816},
    "gbm": {"sharpe_ratio": 0.6047078622436564,
            "total_return": 0.3693672298421755,
            "max_drawdown": -0.33367037659499155, "total_trades": 410},
    "mf": {"sharpe_ratio": -0.4523391124517872,
           "total_return": -0.3600687175588425,
           "max_drawdown": -0.6024189532939825, "total_trades": 2357},
}

TRIALS: List[Dict[str, Any]] = []


def record(name: str, group: str, res: Dict[str, Any], cfg: Any,
           note: str = "", **extra: Any) -> Dict[str, Any]:
    e: Dict[str, Any] = {
        "name": name,
        "group": group,
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
    e.update(extra)
    TRIALS.append(e)
    return e


def print_row(e: Dict[str, Any]) -> None:
    print(f"  {e['name']:46s} SR={e['sharpe_ratio']:6.3f} "
          f"ret={e['total_return']:+7.2%} mdd={e['max_drawdown']:7.2%} "
          f"n={e['total_trades']:5d}", flush=True)


# ══════════════════════════════════════════════════════════════════
# 分数预处理
# ══════════════════════════════════════════════════════════════════
def cs_rank(mat: np.ndarray, min_cross: int = 30) -> np.ndarray:
    df = pd.DataFrame(mat)
    cnt = df.notna().sum(axis=1)
    r = df.rank(axis=1, na_option="keep")
    n = df.notna().sum(axis=1).replace(0, np.nan)
    out = ((r - 0.5) / n.to_numpy()[:, None]).to_numpy()
    out[(cnt < min_cross).to_numpy()] = np.nan
    return out


def cs_winsor_z(mat: np.ndarray, min_cross: int = 30, clip: float = 3.0) -> np.ndarray:
    z = L.cs_zscore(mat, min_cross)
    return np.where(np.isfinite(z), np.clip(z, -clip, clip), np.nan)


PREPROC: Dict[str, Any] = {
    "raw": lambda m: m,
    "z": lambda m: L.cs_zscore(m, 30),
    "rank": cs_rank,
    "winsor3": cs_winsor_z,
}


# ══════════════════════════════════════════════════════════════════
# 合成器
# ══════════════════════════════════════════════════════════════════
def blend(legs: Dict[str, np.ndarray], weights: Dict[str, float],
          preproc: str = "z") -> np.ndarray:
    fn = PREPROC[preproc]
    total = None
    wsum = None
    for name, w in weights.items():
        if w == 0 or name not in legs:
            continue
        m = fn(legs[name])
        good = np.isfinite(m)
        contrib = np.where(good, m * w, 0.0)
        cnt = np.where(good, abs(w), 0.0)
        total = contrib if total is None else total + contrib
        wsum = cnt if wsum is None else wsum + cnt
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(wsum > 0, total / np.where(wsum > 0, wsum, 1.0), np.nan)


def fwd_returns(close: np.ndarray, horizon: int) -> np.ndarray:
    out = np.full_like(close, np.nan)
    if horizon < close.shape[0]:
        out[:-horizon] = close[horizon:] / close[:-horizon] - 1.0
    return out


def rolling_icir(legs: Dict[str, np.ndarray], fwd: np.ndarray,
                 window: int = 120, min_periods: int = 40,
                 smooth: int = 20) -> Dict[str, np.ndarray]:
    """每条腿的滚动 RankICIR（walk-forward，权重做时序平滑）。"""
    out: Dict[str, np.ndarray] = {}
    for name, sc in legs.items():
        ics = pd.Series(L.cs_rank_ic(sc, fwd))
        m = ics.rolling(window, min_periods=min_periods).mean()
        s = ics.rolling(window, min_periods=min_periods).std()
        icir = (m / s.replace(0.0, np.nan)).shift(1)
        if smooth and smooth > 1:
            icir = icir.rolling(smooth, min_periods=max(3, smooth // 3)).mean()
        out[name] = icir.fillna(0.0).to_numpy()
    return out


def icir_blend(legs: Dict[str, np.ndarray], icir: Dict[str, np.ndarray],
               names: Sequence[str], preproc: str = "z",
               power: float = 1.0) -> np.ndarray:
    fn = PREPROC[preproc]
    total = None
    wsum = None
    for n in names:
        z = fn(legs[n])
        w = np.abs(icir[n])[:, None] ** power
        good = np.isfinite(z) & np.isfinite(w)
        contrib = np.where(good, z * w, 0.0)
        cnt = np.where(good, w, 0.0)
        total = contrib if total is None else total + contrib
        wsum = cnt if wsum is None else wsum + cnt
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(wsum > 0, total / np.where(wsum > 0, wsum, 1.0), np.nan)


def corr_penalty_weights(corr: np.ndarray, ir: np.ndarray) -> np.ndarray:
    """w ∝ IR / (1 + ρ̄)：IR 高的腿权重大，高相关的腿权重被惩罚。"""
    n = len(ir)
    rho = np.array([float(np.mean(corr[i, [j for j in range(n) if j != i]]))
                    if n > 1 else 0.0 for i in range(n)])
    w = np.clip(np.asarray(ir, dtype=float), 0.0, None) / (
        1.0 + np.clip(rho, 0.0, 0.99))
    if w.sum() <= 0:
        return np.ones(n) / n
    return w / w.sum()


def inv_cov_weights(corr: np.ndarray, ir: np.ndarray) -> np.ndarray:
    """最小方差近似：w ∝ Σ⁻¹μ。"""
    n = len(ir)
    Cm = np.clip(np.array(corr, dtype=float).copy(), -0.99, 0.99)
    np.fill_diagonal(Cm, 1.0)
    mu = np.asarray(ir, dtype=float)
    try:
        w = np.linalg.solve(Cm + 0.15 * np.eye(n), mu)
    except np.linalg.LinAlgError:
        return np.ones(n) / n
    if w.sum() < 0:
        w = -w
    w = np.clip(w, 0.0, None)
    if w.sum() <= 0:
        return np.ones(n) / n
    return w / w.sum()


def cs_resid_against(y: np.ndarray, Xs: Sequence[np.ndarray],
                      min_cross: int = 30) -> np.ndarray:
    """逐日截面回归残差：y 对 Xs 的暴露做 OLS，取残差。"""
    out = np.full(y.shape, np.nan)
    X = np.stack(list(Xs), axis=2)
    for t in range(y.shape[0]):
        yy, xx = y[t], X[t]
        m = np.isfinite(yy) & np.isfinite(xx).all(axis=1)
        if int(m.sum()) < min_cross:
            continue
        A = np.vstack([xx[m].T, np.ones(int(m.sum()))]).T
        beta, *_ = np.linalg.lstsq(A, yy[m], rcond=None)
        out[t, m] = yy[m] - A @ beta
    return out


# ══════════════════════════════════════════════════════════════════
# 分层合成选择器
# ══════════════════════════════════════════════════════════════════
class LayeredBlendSelector(L.Selector):
    """分层合成：每条腿各取自己的前 N 只，腿间轮流分配名额。

    动机：等权平均分数在腿间高相关时 ≈ 单腿。分层 = 强制组合里有多样化的
    来源，而不是把高相关的邻居堆叠在一起。
    """

    def __init__(self, cfg: Any, leg_scores: Dict[str, np.ndarray],
                 leg_names: Sequence[str], per_leg: int) -> None:
        self.cfg = cfg
        self.leg_scores = leg_scores
        self.leg_names = list(leg_names)
        self.per_leg = per_leg
        self.rebalance_to_targets = True

    def select(self, order, t, holdings, n_hold):
        per: List[List[int]] = []
        for n in self.leg_names:
            sc = self.leg_scores[n][t]
            ok = np.where(np.isfinite(sc))[0]
            if len(ok) == 0:
                per.append([])
                continue
            srt = ok[np.argsort(sc[ok])[::-1]][: self.per_leg]
            per.append([int(c) for c in srt])
        keep: List[int] = []
        seen = set()
        max_depth = max((len(x) for x in per), default=0)
        for depth in range(max_depth):
            for lst in per:
                if depth < len(lst):
                    c = lst[depth]
                    if c not in seen:
                        seen.add(c)
                        keep.append(c)
                    if len(keep) >= self.cfg.topk:
                        break
            if len(keep) >= self.cfg.topk:
                break
        arr = np.array(keep, dtype=int)
        if len(arr) == 0:
            arr = order[: self.cfg.topk]
        w = {int(c): (1 - self.cfg.cash_reserve) / len(arr) for c in arr}
        return arr, w, arr


# ══════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="ensemble_results.json")
    ap.add_argument("--legs", nargs="*",
                    default=["42f", "rsi", "fr", "gbm", "gbm_raw", "mf"])
    args = ap.parse_args()

    t0 = time.time()
    print("=" * 78)
    print("多策略信号合成 · 真实盘口径 (lag=1, tradability=True, 3年周频)")
    print("=" * 78)

    mats = LEGS.cache_mats()
    mats["stock_data"] = {}
    trad = LEGS.cache_trad(mats, {})
    legs: Dict[str, np.ndarray] = {}
    for name in args.legs:
        p = os.path.join(LEGS.LEG_DIR, f"{name}.npz")
        if not os.path.exists(p):
            print(f"!! 缺腿 {name}")
            continue
        legs[name] = np.load(p, allow_pickle=False)["score"]
    print(f"[setup] legs: {list(legs)}  ({time.time()-t0:.0f}s)", flush=True)

    results: Dict[str, Any] = {
        "meta": {
            "window": "2023-05-08→2026-05-08",
            "n_codes_loaded": len(mats["codes"]),
            "n_trading_days": len(mats["master"]),
            "oos_split": str(OOS_SPLIT.date()),
            "caliber": {"lag": 1, "enable_tradability": True,
                        "commission": 0.0003, "slippage": 0.0001,
                        "board_lot": 100, "cash_reserve": 0.05,
                        "rebalance": "weekly(周五)"},
            "engine_legs": ENGINE_LEGS,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "groups": {},
    }
    is_rows, oos_rows = L.split_index(mats["master"], OOS_SPLIT)
    print(f"[setup] IS={len(is_rows)} 天, OOS={len(oos_rows)} 天", flush=True)

    # ═══════ A. 腿基线复现 ═══════
    print("\n" + "=" * 78)
    print("A. 各腿基线：模拟器 vs 真实引擎（保真度验收）")
    print("=" * 78)
    LEG_CFG = {
        "42f": S.SimConfig(topk=60, n_drop=5, lag=1, rebalance_days="weekly",
                           max_position_size=0.015833),
        "rsi": S.SimConfig(topk=30, n_drop=3, lag=1, rebalance_days="weekly",
                           max_position_size=0.031667),
        "fr": S.SimConfig(topk=50, n_drop=5, lag=1, rebalance_days="weekly",
                          max_position_size=0.019),
        "gbm": S.SimConfig(topk=60, n_drop=5, lag=1, rebalance_days="weekly",
                           max_position_size=0.015833),
        "gbm_raw": S.SimConfig(topk=60, n_drop=5, lag=1, rebalance_days="weekly",
                                max_position_size=0.015833),
        "mf": S.SimConfig(topk=30, n_drop=3, lag=1, rebalance_days="daily",
                          max_position_size=0.031667),
    }
    gA = []
    for name, sc in legs.items():
        cfg = LEG_CFG[name]
        res = S.simulate(S.prepare_score(sc, mats, cfg), mats, trad, cfg)
        e = record(f"A_{name}", "A", res, cfg, note="单腿基线")
        e["engine_sharpe"] = ENGINE_LEGS.get(name, {}).get("sharpe_ratio")
        e["engine_trades"] = ENGINE_LEGS.get(name, {}).get("total_trades")
        if e["engine_sharpe"] is not None:
            e["engine_diff"] = round(e["sharpe_ratio"] - e["engine_sharpe"], 4)
        gA.append(e)
        print_row(e)
        print(f"    └─ 引擎 SR={e['engine_sharpe']} n={e['engine_trades']} "
              f"diff={e.get('engine_diff')}")
    results["groups"]["A_leg_baselines"] = gA

    # ═══════ B. 横截面相关矩阵 ═══════
    print("\n" + "=" * 78)
    print("B. 各腿分数矩阵的横截面 Spearman 相关矩阵（逐日截面 → 平均）")
    print("=" * 78)
    corr = L.score_correlation_matrix(legs)
    names = corr["names"]
    print("  " + " " * 10 + " ".join(f"{n:>9s}" for n in names))
    for i, n in enumerate(names):
        print(f"  {n:10s} " + " ".join(
            f"{corr['mean_rank_ic_matrix'][i][j]:9.3f}"
            for j in range(len(names))))
    fwd1 = fwd_returns(mats["close"], 1)
    print("\n  各腿对 fwd1 收益的 RankIC:")
    leg_ic: Dict[str, float] = {}
    leg_icir: Dict[str, float] = {}
    for n, sc in legs.items():
        ic = L.cs_rank_ic(sc, fwd1)
        mu, sd = float(np.nanmean(ic)), float(np.nanstd(ic))
        leg_ic[n] = round(mu, 5)
        leg_icir[n] = round(mu / sd, 4) if sd > 0 else 0.0
        print(f"    {n:10s} meanIC={mu:+.4f} std={sd:.4f} ICIR={leg_icir[n]:+.3f}")
    results["groups"]["B_correlation"] = {
        "names": names,
        "mean_rank_ic_matrix": corr["mean_rank_ic_matrix"],
        "leg_ic_vs_fwd1": leg_ic,
        "leg_icir_vs_fwd1": leg_icir,
    }
    corr_arr = np.array(corr["mean_rank_ic_matrix"], dtype=float)

    # ═══════ C. 合成方法 ═══════
    print("\n" + "=" * 78)
    print("C. 合成方法对比（统一 topk=60 n_drop=5 周频 lag=1）")
    print("=" * 78)
    gC: List[Dict[str, Any]] = []
    base_cfg = S.SimConfig(topk=60, n_drop=5, lag=1, rebalance_days="weekly",
                           max_position_size=0.015833)

    subsets = [
        ("42f+rsi", ["42f", "rsi"]),
        ("42f+fr", ["42f", "fr"]),
        ("42f+gbm", ["42f", "gbm"]),
        ("42f+gbmraw", ["42f", "gbm_raw"]),
        ("42f+mf", ["42f", "mf"]),
        ("rsi+fr", ["rsi", "fr"]),
        ("rsi+gbm", ["rsi", "gbm"]),
        ("42f+rsi+fr", ["42f", "rsi", "fr"]),
        ("42f+rsi+gbm", ["42f", "rsi", "gbm"]),
        ("42f+rsi+gbmraw", ["42f", "rsi", "gbm_raw"]),
        ("42f+rsi+fr+gbm", ["42f", "rsi", "fr", "gbm"]),
        ("all6", ["42f", "rsi", "fr", "gbm", "gbm_raw", "mf"]),
    ]
    subsets = [(n, [x for x in s if x in legs]) for n, s in subsets]
    subsets = [(n, s) for n, s in subsets if len(s) >= 2]

    print("\n  -- C1 等权合成（4 种预处理）--")
    eq_scores: Dict[str, np.ndarray] = {}
    for sname, sl in subsets:
        for prep in ("raw", "z", "rank", "winsor3"):
            sc = blend(legs, {x: 1.0 for x in sl}, prep)
            key = f"{sname}|{prep}"
            eq_scores[key] = sc
            res = S.simulate(S.prepare_score(sc, mats, base_cfg), mats, trad, base_cfg)
            e = record(f"C1_eq_{sname}_{prep}", "C", res, base_cfg,
                       note=f"等权 {prep}")
            e["_score_key"] = key
            gC.append(e)
            print_row(e)

    print("\n  -- C2 滚动 RankICIR 加权（walk-forward）--")
    icir_scores: Dict[str, np.ndarray] = {}
    for win in (60, 120, 250):
        icir = rolling_icir(legs, fwd1, window=win, min_periods=40, smooth=20)
        for sname, sl in subsets:
            sc = icir_blend(legs, icir, sl, "z", power=1.0)
            key = f"{sname}|icir{win}"
            icir_scores[key] = sc
            res = S.simulate(S.prepare_score(sc, mats, base_cfg), mats, trad, base_cfg)
            e = record(f"C2_icir_w{win}_{sname}", "C", res, base_cfg,
                       note=f"滚动{win}日RankICIR加权")
            e["_score_key"] = key
            gC.append(e)
            print_row(e)

    print("\n  -- C3 相关性惩罚权重 w∝IR/(1+ρ) 与 逆协方差最小方差 --")
    corrpen_scores: Dict[str, np.ndarray] = {}
    all_names = list(legs)
    for sname, sl in subsets:
        idx = [all_names.index(x) for x in sl]
        sub_corr = corr_arr[np.ix_(idx, idx)]
        sub_ir = np.array([leg_icir[x] for x in sl])
        for tag, w in (("corrpen", corr_penalty_weights(sub_corr, sub_ir)),
                       ("invcov", inv_cov_weights(sub_corr, sub_ir))):
            sc = blend(legs, dict(zip(sl, w)), "z")
            corrpen_scores[f"{sname}|{tag}"] = sc
            res = S.simulate(S.prepare_score(sc, mats, base_cfg), mats, trad, base_cfg)
            e = record(f"C3_{tag}_{sname}", "C", res, base_cfg,
                       note=f"{tag} " +
                            str({k: round(float(v), 3) for k, v in zip(sl, w)}))
            e["_score_key"] = f"{sname}|{tag}"
            gC.append(e)
            print_row(e)

    print("\n  -- C4 残差正交化（构造低相关 alpha）--")
    ortho_scores: Dict[str, np.ndarray] = {}
    for sname, sl in subsets:
        base = sl[0]
        zbase = L.cs_zscore(legs[base], 30)
        resid: Dict[str, np.ndarray] = {}
        for other in sl[1:]:
            r = cs_resid_against(L.cs_zscore(legs[other], 30), [zbase])
            resid[other] = r
            c = float(np.nanmean(L.cs_rank_ic(r, zbase)))
            print(f"    {other} 对 {base} 残差后 RankIC 相关 = {c:+.3f}")
        sc = blend({base: legs[base], **resid},
                   {base: 1.0, **{k: 1.0 for k in resid}}, "z")
        ortho_scores[f"{sname}|ortho"] = sc
        res = S.simulate(S.prepare_score(sc, mats, base_cfg), mats, trad, base_cfg)
        e = record(f"C4_ortho_{sname}", "C", res, base_cfg, note="残差正交后等权")
        e["_score_key"] = f"{sname}|ortho"
        gC.append(e)
        print_row(e)
    results["groups"]["C_ensemble"] = gC

    # ═══════ D. (topk, n_drop) 网格 ═══════
    print("\n" + "=" * 78)
    print("D. 合成后 TopK 参数扫描")
    print("=" * 78)
    gD: List[Dict[str, Any]] = []
    all_scores = {**eq_scores, **icir_scores, **corrpen_scores, **ortho_scores}
    # 候选 = C 组全窗 SR 最高的 10 个合成分数（复用 C 组已算好的结果，不重复模拟）
    cands = [e["_score_key"] for e in sorted(gC, key=lambda x: -x["sharpe_ratio"])[:10]
             if e.get("_score_key")]
    print(f"  候选合成分数: {cands}", flush=True)

    for key in cands:
        sc = all_scores[key]
        best = None
        for topk in (20, 30, 40, 60, 80):
            for n_drop in (2, 3, 5, 10):
                if n_drop >= topk:
                    continue
                cfg = S.SimConfig(topk=topk, n_drop=n_drop, lag=1,
                                  rebalance_days="weekly",
                                  max_position_size=(1 - 0.05) / topk)
                res = S.simulate(S.prepare_score(sc, mats, cfg), mats, trad, cfg)
                e = record(f"D_{key}_k{topk}_d{n_drop}", "D", res, cfg)
                gD.append(e)
                if best is None or e["sharpe_ratio"] > best["sharpe_ratio"]:
                    best = e
        print(f"  {key:26s} best {best['name'].split('_k')[-1]:10s} "
              f"SR={best['sharpe_ratio']:.3f} ret={best['total_return']:+.2%} "
              f"mdd={best['max_drawdown']:.2%} n={best['total_trades']}")
    results["groups"]["D_topk_grid"] = gD

    # ═══════ E. OOS：IS 选参 → OOS 验证 ═══════
    print("\n" + "=" * 78)
    print("E. OOS 切分（IS=2023-05→2025-05 选参, OOS=2025-05→2026-05 验证）")
    print("=" * 78)
    gE: List[Dict[str, Any]] = []
    # 先在 IS 上粗筛（每个合成分数用 base_cfg 跑一次 IS），只保留
    # IS SR 最高的 40 个合成分数进入完整网格 → 网格规模 40×5×4 = 800
    coarse: List[Tuple[float, str]] = []
    for key, sc in all_scores.items():
        pr = S.prepare_score(sc, mats, base_cfg)
        isr0 = S.simulate(pr, mats, trad, base_cfg, rows=is_rows)
        coarse.append((float(isr0["sharpe_ratio"]), key))
    coarse.sort(key=lambda x: -x[0])
    keep_keys = [k for _, k in coarse[:40]]
    print(f"  IS 粗筛 {len(all_scores)} → 保留 {len(keep_keys)} 个合成分数", flush=True)

    grid: List[Tuple[str, np.ndarray, Any]] = []
    for key, sc in all_scores.items():
        if key not in keep_keys:
            continue
        for topk in (20, 30, 40, 60, 80):
            for n_drop in (2, 3, 5, 10):
                if n_drop >= topk:
                    continue
                cfg = S.SimConfig(topk=topk, n_drop=n_drop, lag=1,
                                  rebalance_days="weekly",
                                  max_position_size=(1 - 0.05) / topk)
                grid.append((key, sc, cfg))
    print(f"  IS 网格 {len(grid)} 个组合（{len(all_scores)} 合成分数 × "
          f"5 topk × 4 n_drop）", flush=True)
    # 说明：IS 网格是**防过拟合统计的一部分**，全部计入尝试次数。
    is_res_list = []
    for key, sc, cfg in grid:
        isr = S.simulate(S.prepare_score(sc, mats, cfg), mats, trad, cfg, rows=is_rows)
        is_res_list.append((key, sc, cfg, float(isr["sharpe_ratio"]),
                            float(isr["total_return"]), isr))
    is_res_list.sort(key=lambda r: -r[3])
    print("  IS Top10:")
    for key, sc, cfg, sr, rt, isr in is_res_list[:10]:
        print(f"    {key:26s} k={cfg.topk:3d} d={cfg.n_drop:2d} "
              f"IS SR={sr:.3f} ret={rt:+.2%}")

    best_key, best_sc, best_cfg, best_is_sr, best_is_rt, _ = is_res_list[0]
    oos = S.simulate(S.prepare_score(best_sc, mats, best_cfg), mats, trad, best_cfg, rows=oos_rows)
    full = S.simulate(S.prepare_score(best_sc, mats, best_cfg), mats, trad, best_cfg)
    e = record(f"E_ISbest_{best_key}_k{best_cfg.topk}_d{best_cfg.n_drop}",
               "E", full, best_cfg, note="IS 选出的配置")
    e["IS_sharpe"] = round(best_is_sr, 4)
    e["IS_return"] = round(best_is_rt, 4)
    e["OOS_sharpe"] = round(float(oos["sharpe_ratio"]), 4)
    e["OOS_return"] = round(float(oos["total_return"]), 4)
    e["OOS_mdd"] = round(float(oos["max_drawdown"]), 4)
    e["OOS_trades"] = int(oos["total_trades"])
    e["OOS_yearly"] = oos.get("yearly", {})
    gE.append(e)
    print(f"\n  ▶ IS 最优 = {best_key} k={best_cfg.topk} d={best_cfg.n_drop}")
    print(f"    IS  SR={best_is_sr:.3f} ret={best_is_rt:+.2%}")
    print(f"    OOS SR={e['OOS_sharpe']:.3f} ret={e['OOS_return']:+.2%} "
          f"mdd={e['OOS_mdd']:.2%} n={e['OOS_trades']} {e['OOS_yearly']}")
    print(f"    全窗 SR={e['sharpe_ratio']:.3f} ret={e['total_return']:+.2%} "
          f"mdd={e['max_drawdown']:.2%}")

    print("\n  -- IS Top15 的 OOS 表现（OOS 只作验证，不用于选参）--")
    oos_table = []
    for key, sc, cfg, sr, rt, isr in is_res_list[:15]:
        o = S.simulate(S.prepare_score(sc, mats, cfg), mats, trad, cfg, rows=oos_rows)
        t = {"key": key, "topk": cfg.topk, "n_drop": cfg.n_drop,
             "IS_sharpe": round(sr, 4),
             "OOS_sharpe": round(float(o["sharpe_ratio"]), 4),
             "OOS_return": round(float(o["total_return"]), 4),
             "OOS_mdd": round(float(o["max_drawdown"]), 4),
             "OOS_trades": int(o["total_trades"]),
             "OOS_yearly": o.get("yearly", {})}
        oos_table.append(t)
        print(f"    {key:26s} k={cfg.topk:3d} d={cfg.n_drop:2d} "
              f"IS={sr:6.3f} → OOS={t['OOS_sharpe']:6.3f} "
              f"ret={t['OOS_return']:+7.2%} mdd={t['OOS_mdd']:7.2%} "
              f"n={t['OOS_trades']}")
    results["groups"]["E_oos"] = gE
    results["groups"]["E_oos_table"] = oos_table

    # ═══════ F. 分层合成 ═══════
    print("\n" + "=" * 78)
    print("F. 分层合成（每腿各取前 N 只，腿间轮流分配名额）")
    print("=" * 78)
    gF: List[Dict[str, Any]] = []
    for sname, sl in subsets:
        for per_leg in (3, 5, 8):
            for topk in (30, 60):
                cfg = S.SimConfig(topk=topk, n_drop=max(2, topk // 12), lag=1,
                                  rebalance_days="weekly",
                                  max_position_size=(1 - 0.05) / topk)
                sel = LayeredBlendSelector(cfg, legs, sl, per_leg=per_leg)
                sc = blend(legs, {x: 1.0 for x in sl}, "z")
                res = S.simulate(S.prepare_score(sc, mats, cfg), mats, trad, cfg, selector=sel)
                e = record(f"F_layer_{sname}_p{per_leg}_k{topk}", "F", res, cfg,
                           note=f"分层 每腿{per_leg}只")
                gF.append(e)
                print_row(e)
    results["groups"]["F_layered"] = gF

    # ═══════ 汇总 ═══════
    results["trials"] = TRIALS
    results["n_trials"] = len(TRIALS)
    ranked = sorted(TRIALS, key=lambda e: -e["sharpe_ratio"])
    results["top15"] = ranked[:15]
    print("\n" + "=" * 78)
    print(f"共 {len(TRIALS)} 个变体。Top15（全窗夏普）：")
    for e in ranked[:15]:
        extra = ""
        if "OOS_sharpe" in e:
            extra = f"  [IS {e['IS_sharpe']:.3f} / OOS {e['OOS_sharpe']:.3f}]"
        print(f"  {e['name']:48s} SR={e['sharpe_ratio']:6.3f} "
              f"ret={e['total_return']:+7.2%} mdd={e['max_drawdown']:7.2%} "
              f"n={e['total_trades']:5d}{extra}")
    print("=" * 78)

    path = os.path.join(L.OUT_DIR, args.out)
    with open(path, "w") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=1, default=str)
    print(f"写入 {path}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
