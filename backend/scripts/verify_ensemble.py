#!/usr/bin/env python3
"""多策略信号合成 —— 真实盘口径批量实验 + OOS 验证。

## 口径（与 verify_10y_pit.py --lag 1 逐项一致）
  窗口 2023-05-08 → 2026-05-08 / 727 交易日
  股票池 data/qlib_official_data/instruments/all.txt（5238 只有效）
  lag=1 / enable_tradability=True / 佣金 0.03% + 滑点 0.01% / 整手 100 /
  现金预留 5% / 周频调仓（周五）

## 腿的分数从**真实策略类**抽（ensemble_legs.py）
  42f   honest_composite 42 因子等权复合
  rsi   rsi_71421  RSI(7/14/21) 均值回归
  fr    factor_ranking 动量/低波动/低振幅
  gbm   gbm_voltgt  LightGBM 元模型 + 波动率目标仓位
  gbm_raw gbm 但关掉波动率目标截断（保留原始预测分）
  mf    multi_factor value/momentum/low-vol

## 合成方法（C 组）
  1. 等权平均分数（三种预处理：原始分 / 截面 z / 截面 rank）
  2. 滚动 RankICIR 加权（walk-forward，无前视，权重做时序平滑）
  3. 相关性惩罚权重（w ∝ IR/(1+ρ)）与 逆协方差最小方差
  4. 分层合成（按腿分桶，桶内选前 N，再在桶间分配名额）
  5. 残差正交化（把每条腿对其它腿做截面回归取残差 → 构造低相关 alpha）

## 防过拟合
  IS = 2023-05→2025-05（选参）  OOS = 2025-05→2026-05（验证）
  报告总尝试次数；OOS 只用 IS 选出的那一个配置。

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

BACKEND = "/Users/rone/Projects/stock-platform/backend"
ROOT = "/Users/rone/Projects/stock-platform"
sys.path.insert(0, BACKEND)
sys.path.insert(0, os.path.join(BACKEND, "scripts"))

import ensemble_lab as L  # noqa: E402
import ensemble_legs as LEGS  # noqa: E402

OOS_SPLIT = pd.Timestamp("2025-05-08")
CACHE = LEGS.CACHE
LEG_DIR = LEGS.LEG_DIR

# 各腿在 verify_10y_pit.py 里的真实配置（topk / n_drop 各不相同）
#
# 重要：factor_ranking 与 gbm_voltgt **完全没有实现 lag**（源码里没有 self.lag），
# 引擎给它们的 lag=1 配置被静默忽略 → 它们公布的基线（fr 0.727 / gbm 0.496）
# 其实是 **lag=0**（当日收盘信号→当日收盘成交，实盘做不到）。
# 这里单腿基线复现时用 lag=0 对齐引擎（parity），合成实验里所有腿统一 lag=1
# （真实盘口径）。
LEG_SIM_CFG: Dict[str, L.SimConfig] = {
    "42f": L.SimConfig(topk=60, n_drop=5, lag=1, rebalance_days="weekly",
                       max_position_size=0.015833),
    "rsi": L.SimConfig(topk=30, n_drop=3, lag=1, rebalance_days="weekly",
                       max_position_size=0.031667),
    "fr": L.SimConfig(topk=50, n_drop=5, lag=0, rebalance_days="weekly",
                      max_position_size=0.019),
    "gbm": L.SimConfig(topk=60, n_drop=5, lag=0, rebalance_days="weekly",
                       max_position_size=0.015833),
    "gbm_raw": L.SimConfig(topk=60, n_drop=5, lag=0, rebalance_days="weekly",
                            max_position_size=0.015833),
    "mf": L.SimConfig(topk=30, n_drop=3, lag=1, rebalance_days="daily",
                      max_position_size=0.031667),
}

# 引擎公布的基线口径（lag 语义各异，见上）
ENGINE_LEG_LAG = {"42f": 1, "rsi": 1, "fr": 0, "gbm": 0, "gbm_raw": 0, "mf": 1}

ENGINE_LEGS = {
    "42f": {"sharpe_ratio": 0.7991795250472001, "total_return": 0.34779248262737783,
            "max_drawdown": -0.16460289642093642, "total_trades": 1080},
    "rsi": {"sharpe_ratio": 1.0147284879698606, "total_return": 0.5190253848549126,
            "max_drawdown": -0.16665780106518946, "total_trades": 822},
    "fr": {"sharpe_ratio": 0.7269197037154874, "total_return": 0.3065268411849565,
           "max_drawdown": -0.1355705126634873, "total_trades": 1170},
    "gbm": {"sharpe_ratio": 0.49570467646690825, "total_return": 0.28418470353380754,
            "max_drawdown": -0.2976551851189215, "total_trades": 610},
    "mf": {"sharpe_ratio": 0.9500139504937557, "total_return": 1.0347074684645836,
           "max_drawdown": -0.4653110633502326, "total_trades": 3520},
}

TRIALS: List[Dict[str, Any]] = []


def record(name: str, group: str, res: Dict[str, Any], cfg: L.SimConfig,
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
    print(f"  {e['name']:44s} SR={e['sharpe_ratio']:6.3f} "
          f"ret={e['total_return']:+7.2%} mdd={e['max_drawdown']:7.2%} "
          f"n={e['total_trades']:5d}", flush=True)


# ══════════════════════════════════════════════════════════════════
# 分数预处理
# ══════════════════════════════════════════════════════════════════
def cs_z(mat: np.ndarray, min_cross: int = 30) -> np.ndarray:
    return L.cs_zscore(mat, min_cross)


def cs_rank(mat: np.ndarray, min_cross: int = 30) -> np.ndarray:
    """逐日截面 rank → [-0.5, 0.5] 均匀分布。"""
    df = pd.DataFrame(mat)
    cnt = df.notna().sum(axis=1)
    r = df.rank(axis=1, na_option="keep")
    n = df.notna().sum(axis=1).replace(0, np.nan)
    out = ((r - 0.5) / n.to_numpy()[:, None]).to_numpy()
    out[(cnt < min_cross).to_numpy()] = np.nan
    return out


def cs_winsor_z(mat: np.ndarray, min_cross: int = 30, clip: float = 3.0) -> np.ndarray:
    z = cs_z(mat, min_cross)
    return np.where(np.isfinite(z), np.clip(z, -clip, clip), np.nan)


PREPROC: Dict[str, Any] = {
    "raw": lambda m: m,
    "z": cs_z,
    "rank": cs_rank,
    "winsor3": cs_winsor_z,
}


# ══════════════════════════════════════════════════════════════════
# 合成器
# ══════════════════════════════════════════════════════════════════
def blend(legs: Dict[str, np.ndarray], weights: Dict[str, float],
          preproc: str = "z") -> np.ndarray:
    """按权重线性合成（先各自按 preproc 标准化，再加权求和，nan-aware）。"""
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
    """每条腿的滚动 RankICIR（walk-forward）。

    - IC 用横截面 RankIC（对 fwd 收益）
    - 第 t 日的权重只用 [t-window, t-1] 的 IC（.shift(1)）
    - smooth: 对 ICIR 序列再做一次滚动平均（时序动量平滑），进一步降噪
    """
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
    """按 |ICIR| 加权合成（逐日权重）。"""
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


def corr_penalty_weights(corr: np.ndarray, ir: np.ndarray,
                         names: Sequence[str]) -> np.ndarray:
    """w ∝ IR / (1 + ρ̄)：IR 高的腿权重大，与其它腿高相关的腿权重被惩罚。"""
    n = len(names)
    rho = np.zeros(n)
    for i in range(n):
        others = [j for j in range(n) if j != i]
        rho[i] = float(np.mean(corr[i, others])) if others else 0.0
    w = np.asarray(ir, dtype=float) / (1.0 + np.clip(rho, 0.0, 0.99))
    w = np.clip(w, 0.0, None)
    if w.sum() <= 0:
        return np.ones(n) / n
    return w / w.sum()


def inv_cov_weights(corr: np.ndarray, ir: np.ndarray,
                    names: Sequence[str]) -> np.ndarray:
    """最小方差近似：w ∝ Σ⁻¹μ（Σ = 相关矩阵，μ = IR 向量）。"""
    n = len(names)
    Cm = np.array(corr, dtype=float).copy()
    np.fill_diagonal(Cm, 1.0)
    Cm = np.clip(Cm, -0.99, 0.99)
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
    """分层合成：把股票按「在哪些腿里排前」分桶，每桶给固定名额。

    动机：等权平均分数 = 42f（相关性高时）。分层 = 「从每条腿自己的强票里
    各取 N 只，再在腿之间分配名额」→ 强制组合里有多样化的来源，而不是
    42f 的高相关邻居堆叠。
    """

    def __init__(self, cfg: L.SimConfig, leg_scores: Dict[str, np.ndarray],
                 leg_names: Sequence[str], per_leg: int) -> None:
        self.cfg = cfg
        self.leg_scores = leg_scores
        self.leg_names = list(leg_names)
        self.per_leg = per_leg
        self.rebalance_to_targets = True

    def select(self, order, t, holdings, n_hold):
        # 每个腿自己的 top per_leg 集合
        per: List[List[int]] = []
        for n in self.leg_names:
            sc = self.leg_scores[n][t]
            ok = np.where(np.isfinite(sc))[0]
            if len(ok) == 0:
                per.append([])
                continue
            srt = ok[np.argsort(sc[ok])[::-1]][: self.per_leg]
            per.append([int(c) for c in srt])
        # 轮转分配名额：每轮每腿取一只（按腿顺序），直到 topk 满
        keep: List[int] = []
        seen = set()
        depth = 0
        while len(keep) < self.cfg.topk and depth < self.per_leg:
            progressed = False
            for lst in per:
                if depth < len(lst):
                    c = lst[depth]
                    if c not in seen:
                        seen.add(c)
                        keep.append(c)
                        progressed = True
                    if len(keep) >= self.cfg.topk:
                        break
            if not progressed and depth > max(len(x) for x in per):
                break
            depth += 1
        arr = np.array(keep, dtype=int)
        if len(arr) == 0:
            arr = order[: self.cfg.topk]
        # 等权目标权重
        w = {int(c): (1 - self.cfg.cash_reserve) / len(arr) for c in arr}
        return arr, w, arr


# ══════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="ensemble_results.json")
    ap.add_argument("--legs", nargs="*",
                    default=["42f", "rsi", "fr", "gbm", "mf"])
    ap.add_argument("--skip-grid", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    print("=" * 78)
    print("多策略信号合成 · 真实盘口径 (lag=1, tradability=True, 3年周频)")
    print("=" * 78)

    mats = LEGS.cache_mats()
    mats["stock_data"] = {}          # simulate 只用 close/codes/master
    trad = LEGS.cache_trad(mats, {})
    legs: Dict[str, np.ndarray] = {}
    for name in args.legs:
        p = os.path.join(LEG_DIR, f"{name}.npz")
        if not os.path.exists(p):
            print(f"!! 缺腿 {name}（先跑 ensemble_legs.py --legs {name}）")
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

    # ═══════ A. 腿基线复现（模拟器 vs 引擎）═══════
    print("\n" + "=" * 78)
    print("A. 各腿基线：模拟器 vs 真实引擎（保真度验收）")
    print("=" * 78)
    gA = []
    for name, sc in legs.items():
        cfg = LEG_SIM_CFG[name]
        res = L.simulate(sc, mats, trad, cfg)
        e = record(f"A_{name}", "A", res, cfg, note="单腿基线")
        e["engine_sharpe"] = ENGINE_LEGS.get(name, {}).get("sharpe_ratio")
        if e["engine_sharpe"] is not None:
            e["engine_diff"] = round(e["sharpe_ratio"] - e["engine_sharpe"], 4)
        gA.append(e)
        print_row(e)
        print(f"    └─ 引擎 SR={e['engine_sharpe']}  diff={e.get('engine_diff')}")
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
    print("C. 合成方法对比")
    print("=" * 78)
    gC: List[Dict[str, Any]] = []
    base_cfg = L.SimConfig(topk=60, n_drop=5, lag=1, rebalance_days="weekly",
                           max_position_size=0.015833)

    # C1 等权：所有腿子集 × 4 种预处理
    subsets = [
        ("42f+rsi", ["42f", "rsi"]),
        ("42f+fr", ["42f", "fr"]),
        ("42f+gbm", ["42f", "gbm"]),
        ("42f+mf", ["42f", "mf"]),
        ("rsi+fr", ["rsi", "fr"]),
        ("rsi+gbm", ["rsi", "gbm"]),
        ("42f+rsi+fr", ["42f", "rsi", "fr"]),
        ("42f+rsi+gbm", ["42f", "rsi", "gbm"]),
        ("42f+rsi+fr+gbm", ["42f", "rsi", "fr", "gbm"]),
        ("42f+rsi+fr+gbm+mf", ["42f", "rsi", "fr", "gbm", "mf"]),
    ]
    subsets = [(n, [x for x in s if x in legs]) for n, s in subsets]
    subsets = [(n, s) for n, s in subsets if len(s) >= 2]

    print("\n  -- C1 等权合成（4 种预处理 × 10 个子集）--")
    eq_scores: Dict[str, np.ndarray] = {}
    for sname, sl in subsets:
        for prep in ("raw", "z", "rank", "winsor3"):
            w = {x: 1.0 for x in sl}
            sc = blend(legs, w, prep)
            key = f"{sname}|{prep}"
            eq_scores[key] = sc
            res = L.simulate(sc, mats, trad, base_cfg)
            e = record(f"C1_eq_{sname}_{prep}", "C", res, base_cfg,
                       note=f"等权 {prep}")
            gC.append(e)
            print_row(e)

    # C2 滚动 ICIR 加权
    print("\n  -- C2 滚动 RankICIR 加权（walk-forward，无前视）--")
    for win in (60, 120, 250):
        icir = rolling_icir(legs, fwd1, window=win, min_periods=40, smooth=20)
        for sname, sl in subsets:
            if len(sl) < 2:
                continue
            sc = icir_blend(legs, icir, sl, "z", power=1.0)
            res = L.simulate(sc, mats, trad, base_cfg)
            e = record(f"C2_icir_w{win}_{sname}", "C", res, base_cfg,
                       note=f"滚动{win}日RankICIR加权")
            gC.append(e)
            print_row(e)

    # C3 相关性惩罚 / 最小方差
    print("\n  -- C3 相关性惩罚权重（w ∝ IR/(1+ρ)）与 逆协方差最小方差 --")
    sl_all = [n for n in names]
    ir_vec = np.array([leg_icir[n] for n in sl_all], dtype=float)
    for sname, sl in subsets:
        if len(sl) < 2:
            continue
        idx = [sl_all.index(x) for x in sl]
        sub_corr = corr_arr[np.ix_(idx, idx)]
        sub_ir = ir_vec[idx]
        w_corr = corr_penalty_weights(sub_corr, sub_ir, sl)
        w_inv = inv_cov_weights(sub_corr, sub_ir, sl)
        for tag, w in (("corrpen", w_corr), ("invcov", w_inv)):
            sc = blend(legs, dict(zip(sl, w)), "z")
            res = L.simulate(sc, mats, trad, base_cfg)
            e = record(f"C3_{tag}_{sname}", "C", res, base_cfg,
                       note=f"{tag} 权重 " +
                            str({k: round(float(v), 3) for k, v in zip(sl, w)}))
            gC.append(e)
            print_row(e)

    # C4 残差正交化（构造低相关 alpha）
    print("\n  -- C4 残差正交化（腿对其它腿截面回归取残差）--")
    orth_scores: Dict[str, np.ndarray] = {}
    for sname, sl in subsets:
        if len(sl) < 2:
            continue
        base = sl[0]
        zbase = cs_z(legs[base])
        resid: Dict[str, np.ndarray] = {}
        for other in sl[1:]:
            r = cs_resid_against(cs_z(legs[other]), [zbase])
            resid[other] = r
            c = float(np.nanmean(L.cs_rank_ic(r, zbase)))
            print(f"    {other} 对 {base} 残差后 RankIC 相关 = {c:+.3f}")
        sc = blend({base: legs[base], **resid}, {base: 1.0, **{
            k: 1.0 for k in resid}}, "z")
        orth_scores[sname] = sc
        res = L.simulate(sc, mats, trad, base_cfg)
        e = record(f"C4_ortho_{sname}", "C", res, base_cfg, note="残差正交后等权")
        gC.append(e)
        print_row(e)
    results["groups"]["C_ensemble"] = gC

    # ═══════ D. (topk, n_drop) 网格 ═══════
    print("\n" + "=" * 78)
    print("D. 合成后 TopK 参数扫描")
    print("=" * 78)
    gD: List[Dict[str, Any]] = []
    # 候选：各合成方法里全窗 SR 最高的几个 + 单腿最佳
    cand_keys: List[str] = list(eq_scores.keys())
    top_cand = sorted(gC, key=lambda e: -e["sharpe_ratio"])[:8]
    for e in top_cand:
        m = e["name"]
        for sname, sl in subsets:
            for prep in ("raw", "z", "rank", "winsor3"):
                if m == f"C1_eq_{sname}_{prep}":
                    cand_keys.append(f"{sname}|{prep}")
    seen_c = set()
    cand_keys = [k for k in cand_keys if not (k in seen_c or seen_c.add(k))]
    for key in cand_keys:
        if key not in eq_scores:
            continue
        sc = eq_scores[key]
        row = []
        for topk in (20, 30, 40, 60, 80):
            for n_drop in (2, 3, 5, 10):
                if n_drop >= topk:
                    continue
                cfg = L.SimConfig(topk=topk, n_drop=n_drop, lag=1,
                                  rebalance_days="weekly",
                                  max_position_size=(1 - 0.05) / topk)
                res = L.simulate(sc, mats, trad, cfg)
                e = record(f"D_{key}_k{topk}_d{n_drop}", "D", res, cfg)
                gD.append(e)
                row.append((topk, n_drop, e["sharpe_ratio"]))
        best = max(row, key=lambda r: r[2])
        print(f"  {key:24s} best k={best[0]} d={best[1]} SR={best[2]:.3f} "
              f"| " + " ".join(f"k{t}d{d}={sr:.2f}"
                               for t, d, sr in row if t == best[0]))
    results["groups"]["D_topk_grid"] = gD

    # ═══════ E. OOS：IS 选参 → OOS 验证 ═══════
    print("\n" + "=" * 78)
    print("E. OOS 切分（IS=2023-05→2025-05 选参, OOS=2025-05→2026-05 验证）")
    print("=" * 78)
    gE: List[Dict[str, Any]] = []
    # 在 IS 上对每个 (合成方法, topk, n_drop) 组合打分
    is_scores: List[Tuple[str, np.ndarray, L.SimConfig, str]] = []
    for key, sc in eq_scores.items():
        for topk in (20, 30, 40, 60, 80):
            for n_drop in (2, 3, 5, 10):
                if n_drop >= topk:
                    continue
                cfg = L.SimConfig(topk=topk, n_drop=n_drop, lag=1,
                                  rebalance_days="weekly",
                                  max_position_size=(1 - 0.05) / topk)
                is_scores.append((f"{key}", sc, cfg, "eq"))
    for e in gC:
        if e["name"].startswith(("C2_", "C3_", "C4_")):
            continue
    print(f"  IS 网格 {len(is_scores)} 个组合", flush=True)
    is_rows_list = []
    for key, sc, cfg, _ in is_scores:
        isr = L.simulate_subset(sc, mats, trad, cfg, is_rows)
        is_rows_list.append((key, sc, cfg, float(isr["sharpe_ratio"]),
                             float(isr["total_return"]), isr))
    is_rows_list.sort(key=lambda r: -r[3])
    print("  IS Top10:")
    for key, sc, cfg, sr, rt, isr in is_rows_list[:10]:
        print(f"    {key:24s} k={cfg.topk:3d} d={cfg.n_drop:2d} "
              f"IS SR={sr:.3f} ret={rt:+.2%} {isr.get('yearly')}")

    # IS 最优 → OOS
    best_key, best_sc, best_cfg, best_is_sr, best_is_rt, _ = is_rows_list[0]
    oos = L.simulate_subset(best_sc, mats, trad, best_cfg, oos_rows)
    full = L.simulate(best_sc, mats, trad, best_cfg)
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

    # 全部候选都跑 OOS（看分布，诚实报告）
    print("\n  -- IS Top15 的 OOS 表现（OOS 只作验证，不用于选参）--")
    oos_table = []
    for key, sc, cfg, sr, rt, isr in is_rows_list[:15]:
        o = L.simulate_subset(sc, mats, trad, cfg, oos_rows)
        oos_table.append({
            "key": key, "topk": cfg.topk, "n_drop": cfg.n_drop,
            "IS_sharpe": round(sr, 4), "OOS_sharpe": round(float(o["sharpe_ratio"]), 4),
            "OOS_return": round(float(o["total_return"]), 4),
            "OOS_mdd": round(float(o["max_drawdown"]), 4),
            "OOS_trades": int(o["total_trades"]),
            "OOS_yearly": o.get("yearly", {}),
        })
        t = oos_table[-1]
        print(f"    {key:24s} k={cfg.topk:3d} d={cfg.n_drop:2d} "
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
        if len(sl) < 2:
            continue
        for per_leg in (3, 5, 8):
            for topk in (30, 60):
                cfg = L.SimConfig(topk=topk, n_drop=max(2, topk // 12), lag=1,
                                  rebalance_days="weekly",
                                  max_position_size=(1 - 0.05) / topk)
                sel = LayeredBlendSelector(cfg, legs, sl, per_leg=per_leg)
                sc = blend(legs, {x: 1.0 for x in sl}, "z")
                res = L.simulate(sc, mats, trad, cfg, selector=sel)
                e = record(f"F_layer_{sname}_p{per_leg}_k{topk}", "F", res, cfg,
                           note=f"分层 每腿{per_leg}只")
                gF.append(e)
                print_row(e)
    results["groups"]["F_layered"] = gF

    # ═══════ 汇总 ═══════
    results["trials"] = TRIALS
    results["n_trials"] = len(TRIALS)
    ranked = sorted(TRIALS, key=lambda e: -e["sharpe_ratio"])
    results["top10"] = ranked[:10]
    print("\n" + "=" * 78)
    print(f"共 {len(TRIALS)} 个变体。Top10（全窗夏普，含 IS/OOS 标注）：")
    for e in ranked[:10]:
        extra = ""
        if "OOS_sharpe" in e:
            extra = f"  [IS {e['IS_sharpe']:.3f} / OOS {e['OOS_sharpe']:.3f}]"
        print(f"  {e['name']:46s} SR={e['sharpe_ratio']:6.3f} "
              f"ret={e['total_return']:+7.2%} mdd={e['max_drawdown']:7.2%} "
              f"n={e['total_trades']:5d}{extra}")
    print("=" * 78)

    path = os.path.join(L.OUT_DIR, args.out)
    with open(path, "w") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=1, default=str)
    print(f"写入 {path}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
