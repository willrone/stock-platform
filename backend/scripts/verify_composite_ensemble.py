#!/usr/bin/env python3
"""多策略信号合成 + 因子层 + 组合构建层 —— 真实盘口径批量实验。

## 口径
完全沿用 verify_10y_pit.py --lag 1：
  窗口 2023-05-08 → 2026-05-08 / PIT 股票池 / lag=1 /
  enable_tradability=True / TopK-Dropout topk=60 n_drop=5 等权 /
  佣金 0.03% + 滑点 0.01% / 整手 100 股 / 现金预留 5%

## 实验分组
  A. 基线复现（引擎 vs 模拟器）
  B. 分数矩阵横截面相关矩阵（各策略之间的 RankIC）
  C. 合成方法（等权 / ICIR 加权 / 相关性惩罚 / 分层）
  D. 因子层改进（面板因子对齐 / ICIR 加权 / 去冗余 / 残差正交）
  E. 组合构建层（topk / n_drop / 逆波动率加权 / 分层选股）
  F. OOS 切分（2023-05→2025-05 选参，2025-05→2026-05 验证）

## 用法
  env -u PYTHONPATH -u PYTHONHOME backend/.venv-py313/bin/python3 \
    backend/scripts/verify_composite_ensemble.py --groups A B C D E F
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(BACKEND)
sys.path.insert(0, BACKEND)
sys.path.insert(0, os.path.join(BACKEND, "scripts"))

import ensemble_lab as L  # noqa: E402

OOS_SPLIT = pd.Timestamp("2025-05-08")
BASE_CFG = L.SimConfig(topk=60, n_drop=5, lag=1, rebalance_days="weekly")

# 引擎实测基线（verify_10y_pit.py --years 3 --lag 1 --only honest_composite）
ENGINE_BASELINE = {
    "sharpe_ratio": 0.8795668740640868,
    "total_return": 0.35505797455182114,
    "max_drawdown": -0.1337121157482497,
    "total_trades": 1190,
    "yearly": {"2024": 0.160944, "2025": 0.090579, "2026": 0.046826},
}
# 引擎实测的其它策略（同一口径，verify_10y_pit.py --lag 1）
ENGINE_LEGS = {
    "rsi_71421": {"sharpe_ratio": 1.0147284879698606, "total_return": 0.5190253848549126,
                  "max_drawdown": -0.16665780106518946, "total_trades": 822},
    "factor_ranking": {"sharpe_ratio": 0.7269197037154874, "total_return": 0.3065268411849565,
                       "max_drawdown": -0.1355705126634873, "total_trades": 1170},
    "gbm_voltgt": {"sharpe_ratio": 0.49570467646690825, "total_return": 0.28418470353380754,
                   "max_drawdown": -0.2976551851189215, "total_trades": 610},
    "multi_factor": {"sharpe_ratio": 0.9500139504937557, "total_return": 1.0347074684645836,
                     "max_drawdown": -0.4653110633502326, "total_trades": 3520},
}

TRIALS: List[Dict[str, Any]] = []   # 记录所有试过的变体（用于防过拟合统计）

# 各策略线在 verify_10y_pit.py 里的真实配置（**topk 各不相同**）
LEG_CFG: Dict[str, L.SimConfig] = {
    "42f": L.SimConfig(topk=60, n_drop=5, lag=1, rebalance_days="weekly",
                       max_position_size=0.015833),
    "rsi": L.SimConfig(topk=30, n_drop=3, lag=1, rebalance_days="weekly",
                       max_position_size=0.031667),
    "fr": L.SimConfig(topk=50, n_drop=5, lag=1, rebalance_days="weekly",
                      max_position_size=0.019),
}


def record(name: str, group: str, res: Dict[str, Any],
           cfg: L.SimConfig, note: str = "") -> Dict[str, Any]:
    entry = {
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
    TRIALS.append(entry)
    return entry


def print_row(e: Dict[str, Any]) -> None:
    print(f"  {e['name']:38s} SR={e['sharpe_ratio']:6.3f} "
          f"ret={e['total_return']:+7.2%} mdd={e['max_drawdown']:7.2%} "
          f"n={e['total_trades']:5d}  {e.get('yearly')}", flush=True)


# ══════════════════════════════════════════════════════════════════
# 因子层工具
# ══════════════════════════════════════════════════════════════════
def fwd_returns(close: np.ndarray, horizon: int) -> np.ndarray:
    """未来 horizon 日收益（label）。末尾 horizon 行为 NaN。"""
    out = np.full_like(close, np.nan)
    if horizon < close.shape[0]:
        out[:-horizon] = close[horizon:] / close[:-horizon] - 1.0
    return out


def rolling_icir(
    z_factors: Dict[str, np.ndarray],
    fwd: np.ndarray,
    window: int = 60,
    min_periods: int = 20,
) -> Dict[str, pd.Series]:
    """滚动 RankICIR（带符号），逐因子计算。

    无前视：第 t 日的权重只用 [t-window, t-1] 的历史 IC。
    """
    out: Dict[str, pd.Series] = {}
    for name, z in z_factors.items():
        ics = pd.Series(L.cs_rank_ic(z, fwd))
        # shift(1) → 第 t 日只用 t-1 及之前的数据
        mean = ics.rolling(window, min_periods=min_periods).mean()
        std = ics.rolling(window, min_periods=min_periods).std()
        out[name] = (mean / std.replace(0.0, np.nan)).shift(1)
    return out


def icir_weighted_score(
    z_factors: Dict[str, np.ndarray],
    fwd: np.ndarray,
    window: int = 60,
    min_periods: int = 20,
    clip: float = 1.0,
) -> np.ndarray:
    """按滚动 RankICIR 加权的因子复合（walk-forward，无前视）。"""
    icir = rolling_icir(z_factors, fwd, window=window, min_periods=min_periods)
    n_d, n_s = next(iter(z_factors.values())).shape
    total = np.zeros((n_d, n_s))
    wsum = np.zeros((n_d, n_s))
    for name, ser in icir.items():
        z = z_factors[name]
        w = ser.clip(-clip, clip).fillna(0.0).to_numpy()[:, None]   # (n_dates,1)
        good = np.isfinite(z) & np.isfinite(w)
        total = total + np.where(good, z * w, 0.0)
        wsum = wsum + np.where(good, np.abs(w), 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(wsum > 0, total / np.where(wsum > 0, wsum, 1.0), np.nan)


def zstack(score: np.ndarray) -> np.ndarray:
    """分数矩阵 → 逐日截面 z（用于合成前统一量纲）。"""
    return L.cs_zscore(score, 30)


def blend(scores: Dict[str, np.ndarray],
          weights: Dict[str, float]) -> np.ndarray:
    """按权重线性合成（先各自截面 z，再加权求和）。"""
    total = None
    wsum = None
    for name, w in weights.items():
        if w == 0:
            continue
        z = zstack(scores[name])
        good = np.isfinite(z)
        contrib = np.where(good, z * w, 0.0)
        cnt = np.where(good, abs(w), 0.0)
        total = contrib if total is None else total + contrib
        wsum = cnt if wsum is None else wsum + cnt
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(wsum > 0, total / np.where(wsum > 0, wsum, 1.0), np.nan)


# ══════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="*", default=["A", "B", "C", "D", "E", "F"],
                    help="要跑的实验组")
    ap.add_argument("--out", default="ensemble_results.json")
    ap.add_argument("--engine-check", action="store_true",
                    help="额外用真实引擎跑最佳变体做交叉验证")
    args = ap.parse_args()
    groups = set(args.groups)

    t0 = time.time()
    print("=" * 78)
    print("多策略合成研究实验室 · 真实盘口径 (lag=1, tradability=True)")
    print("=" * 78)

    codes = L.universe_from_instruments()
    mats = L.load_matrices(codes)
    print(f"[setup] loaded in {time.time()-t0:.0f}s", flush=True)

    trad = L.build_tradability_mats(mats)
    print(f"[setup] tradability in {time.time()-t0:.0f}s", flush=True)

    # ---- 基础分数矩阵 ----
    # 42f_engine: 引擎在 CWD=仓库根 时的真实行为
    #   panel_dir="../data/qlib_data/features/day" 解析失败 → 17 面板因子全 NaN
    #   → 实际参与复合的只有 25 个公式因子（≠ 42）
    s42_engine, z_eng = L.build_42f_score(mats, panel_aligned=False)
    # 42f_day_aligned: 用 features/day 的面板，但按日期正确对齐
    #   （features/day 是两段拼接，策略原读法只拿到 ~51% 的交易日）
    s42_day, z_day = L.build_42f_score(
        mats, panel_aligned=True, panel_dir=os.path.join(L.QLIB_LIVE, "features", "day"))
    # 42f_full: 用 features_full（交易日格单段文件，面板因子 100% 对齐）
    s42_full, z_full = L.build_42f_score(
        mats, panel_aligned=True, panel_dir=L.PANEL_FULL)
    print(f"[setup] scores in {time.time()-t0:.0f}s", flush=True)

    s_rsi = L.rsi_71421_score(mats)
    s_fr = L.factor_ranking_score(mats)
    print(f"[setup] leg scores in {time.time()-t0:.0f}s", flush=True)

    all_scores = {
        "42f_engine": s42_engine,
        "42f_day_aligned": s42_day,
        "42f_full": s42_full,
        "rsi_71421": s_rsi,
        "factor_ranking": s_fr,
    }

    results: Dict[str, Any] = {
        "meta": {
            "window": "2023-05-08→2026-05-08",
            "n_codes_requested": len(codes),
            "n_codes_loaded": len(mats["codes"]),
            "n_trading_days": len(mats["master"]),
            "oos_split": str(OOS_SPLIT.date()),
            "caliber": {
                "lag": 1, "enable_tradability": True, "topk": 60, "n_drop": 5,
                "max_position_size": 0.015833, "commission": 0.0003,
                "slippage": 0.0001, "board_lot": 100, "cash_reserve": 0.05,
                "rebalance": "weekly(周五)",
            },
            "engine_baseline": ENGINE_BASELINE,
            "engine_legs": ENGINE_LEGS,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        },
        "groups": {},
    }

    is_rows, oos_rows = L.split_index(mats["master"], OOS_SPLIT)
    print(f"[setup] IS={len(is_rows)} days, OOS={len(oos_rows)} days", flush=True)

    # ══════════ A. 基线复现 ══════════
    if "A" in groups:
        print("\n" + "=" * 78)
        print("A. 基线复现（模拟器 vs 真实引擎，各策略用**自己的** topk/n_drop）")
        print("=" * 78)
        gA = []
        legs = (
            ("42f_engine", s42_engine, LEG_CFG["42f"], 0.8795668740640868),
            ("42f_day_aligned", s42_day, LEG_CFG["42f"], 0.7991795250472001),
            ("42f_full", s42_full, LEG_CFG["42f"], None),
            ("rsi_71421", s_rsi, LEG_CFG["rsi"], 1.0147284879698606),
            ("factor_ranking", s_fr, LEG_CFG["fr"], 0.7269197037154874),
        )
        for name, sc, cfg, eng_sr in legs:
            res = L.simulate(sc, mats, trad, cfg)
            e = record(f"A_{name}", "A", res, cfg)
            gA.append(e)
            print_row(e)
            if eng_sr is not None:
                e["engine_sharpe"] = round(eng_sr, 4)
                e["engine_diff"] = round(e["sharpe_ratio"] - eng_sr, 4)
                print(f"    └─ 引擎 SR={eng_sr:.4f}  差={e['sharpe_ratio']-eng_sr:+.4f}")
        results["groups"]["A_baseline"] = gA

        # A2: topk 敏感性（每条腿自己的分数，扫 topk）—— 组合构建层最敏感的旋钮
        print("\n  -- topk 敏感性（周频, n_drop=topk/12）--")
        for name, sc in (("42f_engine", s42_engine), ("42f_day_aligned", s42_day),
                         ("rsi_71421", s_rsi), ("factor_ranking", s_fr)):
            row = []
            for topk in (20, 30, 40, 50, 60, 80, 100, 150):
                cfg = L.SimConfig(topk=topk, n_drop=max(2, topk // 12), lag=1,
                                  rebalance_days="weekly",
                                  max_position_size=(1 - 0.05) / topk)
                res = L.simulate(sc, mats, trad, cfg)
                e = record(f"A2_topk{topk}_{name}", "A", res, cfg)
                row.append((topk, e["sharpe_ratio"], e["total_return"],
                            e["max_drawdown"], e["total_trades"]))
            best = max(row, key=lambda r: r[1])
            print(f"    {name:18s} " + "  ".join(
                f"k{t}={sr:.3f}" for t, sr, _, _, _ in row) +
                f"   | best k={best[0]} SR={best[1]:.3f} "
                f"ret={best[2]:+.2%} mdd={best[3]:.2%} n={best[4]}")
            results["groups"].setdefault("A2_topk_sensitivity", []).append(
                {"leg": name, "scan": [
                    {"topk": t, "sharpe": sr, "ret": r, "mdd": m, "n": n}
                    for t, sr, r, m, n in row]})

    # ══════════ B. 分数矩阵相关矩阵 ══════════
    if "B" in groups:
        print("\n" + "=" * 78)
        print("B. 各策略分数矩阵的横截面 RankIC 相关矩阵（全窗口）")
        print("=" * 78)
        corr = L.score_correlation_matrix(all_scores)
        names = corr["names"]
        print("  " + " " * 22 + " ".join(f"{n[:10]:>10s}" for n in names))
        for i, n in enumerate(names):
            row = " ".join(f"{corr['mean_rank_ic_matrix'][i][j]:10.3f}"
                           for j in range(len(names)))
            print(f"  {n:22s} {row}")
        print("\n  自身 RankIC 均值/标准差（对 fwd1 收益）:")
        fwd1 = fwd_returns(mats["close"], 1)
        for n, sc in all_scores.items():
            ic = L.cs_rank_ic(sc, fwd1)
            print(f"    {n:22s} mean={np.nanmean(ic):+.4f} std={np.nanstd(ic):.4f} "
                  f"ICIR={np.nanmean(ic)/np.nanstd(ic):+.3f}")
        results["groups"]["B_correlation"] = corr

    # ══════════ C. 合成方法 ══════════
    if "C" in groups:
        print("\n" + "=" * 78)
        print("C. 多策略信号合成方法")
        print("=" * 78)
        gC = []

        # C1 等权平均
        combos = {
            "42f+rsi": {"42f_engine": 1.0, "rsi_71421": 1.0},
            "42f+fr": {"42f_engine": 1.0, "factor_ranking": 1.0},
            "42f+rsi+fr": {"42f_engine": 1.0, "rsi_71421": 1.0,
                           "factor_ranking": 1.0},
            "full+rsi": {"42f_full": 1.0, "rsi_71421": 1.0},
            "full+rsi+fr": {"42f_full": 1.0, "rsi_71421": 1.0,
                            "factor_ranking": 1.0},
            "day+rsi+fr": {"42f_day_aligned": 1.0, "rsi_71421": 1.0,
                           "factor_ranking": 1.0},
            "rsi+fr": {"rsi_71421": 1.0, "factor_ranking": 1.0},
        }
        for cname, w in combos.items():
            sc = blend(all_scores, w)
            res = L.simulate(sc, mats, trad, BASE_CFG)
            e = record(f"C_eq_{cname}", "C", res, BASE_CFG, note="等权截面 z 合成")
            gC.append(e)
            print_row(e)

        # C1b 合成后用**最小腿的 topk**（rsi 30 / fr 50）重跑 —— topk 敏感性很大
        print("\n  -- 合成后改用不同 topk --")
        for cname, w in (("42f+rsi", {"42f_engine": 1.0, "rsi_71421": 1.0}),
                         ("42f+rsi+fr", {"42f_engine": 1.0, "rsi_71421": 1.0,
                                         "factor_ranking": 1.0})):
            sc = blend(all_scores, w)
            for topk in (30, 50, 60):
                cfg = L.SimConfig(topk=topk, n_drop=max(2, topk // 12), lag=1,
                                  rebalance_days="weekly",
                                  max_position_size=(1 - 0.05) / topk)
                res = L.simulate(sc, mats, trad, cfg)
                e = record(f"C_eq_{cname}_k{topk}", "C", res, cfg)
                gC.append(e)
                print_row(e)

        # C2 按腿的自身 ICIR 加权（用 fwd1 的历史 ICIR，walk-forward）
        print("\n  -- 按各腿自身 RankICIR 加权（walk-forward, 无前视）--")
        fwd1 = fwd_returns(mats["close"], 1)
        leg_icir = {}
        for n, sc in all_scores.items():
            ics = pd.Series(L.cs_rank_ic(sc, fwd1))
            roll = (ics.rolling(120, min_periods=40).mean()
                    / ics.rolling(120, min_periods=40).std()).shift(1)
            leg_icir[n] = roll.fillna(0.0).to_numpy()
        # 逐日按 ICIR 符号取绝对值权重合成
        def icir_blend(names_: Sequence[str]) -> np.ndarray:
            total = np.zeros_like(s42_engine)
            wsum = np.zeros_like(s42_engine)
            for n in names_:
                z = zstack(all_scores[n])
                w = np.abs(leg_icir[n])[:, None]      # (n_dates, 1) → 广播
                good = np.isfinite(z) & np.isfinite(w)
                total = total + np.where(good, z * w, 0.0)
                wsum = wsum + np.where(good, w, 0.0)
            with np.errstate(invalid="ignore", divide="ignore"):
                return np.where(wsum > 0, total / np.where(wsum > 0, wsum, 1.0), np.nan)

        for cname, legs in (("42f+rsi", ["42f_engine", "rsi_71421"]),
                            ("42f+rsi+fr", ["42f_engine", "rsi_71421",
                                             "factor_ranking"]),
                            ("full+rsi+fr", ["42f_full", "rsi_71421",
                                             "factor_ranking"]),
                            ("all", list(all_scores))):
            sc = icir_blend(legs)
            res = L.simulate(sc, mats, trad, BASE_CFG)
            e = record(f"C_icirw_{cname}", "C", res, BASE_CFG,
                       note="按腿自身 walk-forward RankICIR 加权")
            gC.append(e)
            print_row(e)

        # C3 相关性惩罚（最小化组合腿间相关性 → 近似最大分散化权重）
        print("\n  -- 相关性惩罚权重（w ∝ 1/相关度矩阵特征值，负相关腿加成）--")
        corr_mat = np.array(
            L.score_correlation_matrix(all_scores)["mean_rank_ic_matrix"])
        # 用 leg 间相关矩阵做简单对角加载逆
        leg_names = list(all_scores)
        Cm = corr_mat.copy()
        np.fill_diagonal(Cm, 1.0)
        Cm = np.clip(Cm, -0.99, 0.99)
        try:
            inv = np.linalg.inv(Cm + 0.1 * np.eye(len(Cm)))
            w = inv.sum(axis=1)
            if w.sum() < 0:
                w = -w
            w = w / np.abs(w).sum()
        except np.linalg.LinAlgError:
            w = np.ones(len(Cm)) / len(Cm)
        print("    权重:", {n: round(float(x), 4) for n, x in zip(leg_names, w)})
        sc = blend(all_scores, dict(zip(leg_names, w)))
        res = L.simulate(sc, mats, trad, BASE_CFG)
        e = record("C_corrpenalty_all4", "C", res, BASE_CFG,
                   note="腿间相关矩阵求逆的分散化权重")
        gC.append(e)
        print_row(e)

        # C4 信号正交化：把一条腿对另一条腿做截面回归取残差 → 构造低相关 alpha
        # 动机：42f 与 rsi 的分数 RankIC 相关 0.79，等权平均几乎等于 42f。
        # 把 rsi 对 42f 回归取残差，得到「42f 解释不了的那部分 rsi」，
        # 再与 42f 等权合成 —— 这是真正的增量信息，不是重复计数。
        print("\n  -- 信号正交化（残差化构造低相关 alpha）--")
        z42 = L.cs_zscore(s42_engine, 30)
        zrsi = L.cs_zscore(s_rsi, 30)
        zfr = L.cs_zscore(s_fr, 30)
        rsi_resid = cs_resid_against(zrsi, [z42])
        fr_resid = cs_resid_against(zfr, [z42])
        print(f"    rsi 对 42f 残差后 RankIC 相关: "
              f"{np.nanmean(L.cs_rank_ic(rsi_resid, z42)):.3f}")
        print(f"    fr  对 42f 残差后 RankIC 相关: "
              f"{np.nanmean(L.cs_rank_ic(fr_resid, z42)):.3f}")
        ortho_combos = {
            "42f+resid(rsi)": {"42f_engine": 1.0, "__resid_rsi": 1.0},
            "42f+resid(fr)": {"42f_engine": 1.0, "__resid_fr": 1.0},
            "42f+resid(rsi)+resid(fr)": {"42f_engine": 1.0, "__resid_rsi": 1.0,
                                        "__resid_fr": 1.0},
        }
        extra_scores = {"__resid_rsi": rsi_resid, "__resid_fr": fr_resid}
        for cname, w in ortho_combos.items():
            sc = blend({**all_scores, **extra_scores}, w)
            for topk in (30, 60):
                cfg = L.SimConfig(topk=topk, n_drop=max(2, topk // 12), lag=1,
                                  rebalance_days="weekly",
                                  max_position_size=(1 - 0.05) / topk)
                res = L.simulate(sc, mats, trad, cfg)
                e = record(f"C_ortho_{cname}_k{topk}", "C", res, cfg,
                           note="残差正交后合成")
                gC.append(e)
                print_row(e)

        results["groups"]["C_ensemble"] = gC

    # ══════════ D. 因子层 ══════════
    if "D" in groups:
        print("\n" + "=" * 78)
        print("D. 因子层改进")
        print("=" * 78)
        gD = []

        # D1 面板因子正确对齐（引擎的 42f 实际是 25 因子）
        for nm, sc in (("day", s42_day), ("full", s42_full)):
            res = L.simulate(sc, mats, trad, BASE_CFG)
            e = record(f"D_panel_aligned_42f_{nm}", "D", res, BASE_CFG,
                       note="17 个 Alpha158 面板因子按交易日正确对齐")
            gD.append(e)
            print_row(e)

        # D2 ICIR 加权复合（42 因子，walk-forward）
        print("\n  -- ICIR 加权复合（滚动 60d RankIC vs fwd1，walk-forward）--")
        fwd1 = fwd_returns(mats["close"], 1)
        for zname, all_z in (("eng25", z_eng), ("full42", z_full)):
            sc_icir = icir_weighted_score(all_z, fwd1, window=60, min_periods=20)
            res = L.simulate(sc_icir, mats, trad, LEG_CFG["42f"])
            e = record(f"D_icir_weighted_{zname}", "D", res, LEG_CFG["42f"],
                       note=f"{zname} 按滚动 RankICIR 加权 (fwd1)")
            gD.append(e)
            print_row(e)

        # D3 去冗余（按因子间截面相关聚类，保留每簇代表）
        print("\n  -- 去冗余：因子 z 矩阵两两相关 > 阈值则只留一个 --")
        for zname, all_z in (("eng25", z_eng), ("full42", z_full)):
            for thr in (0.7, 0.9):
                names = list(all_z)
                # 用平均 RankIC(对 fwd1) 的绝对值排序，优先保留信息量大的
                ic_abs = {}
                for n in names:
                    ic = L.cs_rank_ic(all_z[n], fwd1)
                    ic_abs[n] = float(np.nanmean(np.abs(ic)))
                order = sorted(names, key=lambda n: -ic_abs[n])
                keep: List[str] = []
                flat = {n: all_z[n].ravel() for n in names}
                for n in order:
                    v = flat[n]
                    ok = np.isfinite(v)
                    if not ok.any():
                        continue
                    redundant = False
                    for k in keep:
                        w = flat[k]
                        m = ok & np.isfinite(w)
                        if m.sum() < 100:
                            continue
                        a, b = v[m], w[m]
                        if np.std(a) > 0 and np.std(b) > 0:
                            c = abs(float(np.corrcoef(a, b)[0, 1]))
                            if c > thr:
                                redundant = True
                                break
                    if not redundant:
                        keep.append(n)
                print(f"    {zname} thr={thr}: 保留 {len(keep)}/{len(names)} 因子")
                tot = np.zeros_like(s42_engine)
                ws = np.zeros_like(s42_engine)
                for n in keep:
                    z = all_z[n]
                    good = np.isfinite(z)
                    tot[good] += np.nan_to_num(z)[good]
                    ws[good] += 1.0
                with np.errstate(invalid="ignore", divide="ignore"):
                    sc_dedup = np.where(ws > 0, tot / np.where(ws > 0, ws, 1.0),
                                        np.nan)
                res = L.simulate(sc_dedup, mats, trad, BASE_CFG)
                e = record(f"D_dedup_{zname}_thr{thr}", "D", res, BASE_CFG,
                           note=f"{zname} 去冗余 {len(keep)}/{len(names)} 因子")
                gD.append(e)
                print_row(e)

        # D4 残差正交：把每个因子 z 对「其它因子均值」做截面回归残差
        print("\n  -- 残差正交：因子 z 对其余因子均值的截面回归残差 --")
        for zname, all_z in (("eng25", z_eng), ("full42", z_full)):
            names = list(all_z)
            resid = {}
            for n in names:
                others = [m for m in names if m != n]
                X = np.vstack([all_z[m].ravel() for m in others])
                y = all_z[n].ravel()
                M = np.isfinite(X).all(axis=0) & np.isfinite(y)
                r = np.full(y.shape, np.nan)
                if M.sum() > 100:
                    A = np.vstack([X[:, M], np.ones(M.sum())]).T
                    beta, *_ = np.linalg.lstsq(A, y[M], rcond=None)
                    r[M] = y[M] - A @ beta
                resid[n] = r.reshape(all_z[n].shape)
            tot = np.zeros_like(s42_engine)
            ws = np.zeros_like(s42_engine)
            for n in names:
                z = resid[n]
                good = np.isfinite(z)
                tot[good] += np.nan_to_num(z)[good]
                ws[good] += 1.0
            with np.errstate(invalid="ignore", divide="ignore"):
                sc_orth = np.where(ws > 0, tot / np.where(ws > 0, ws, 1.0), np.nan)
            res = L.simulate(sc_orth, mats, trad, BASE_CFG)
            e = record(f"D_orthogonalized_{zname}", "D", res, BASE_CFG,
                       note=f"{zname} 截面残差正交后等权")
            gD.append(e)
            print_row(e)

        # D2b 持有期匹配的 IC 加权
        # 关键洞察：周频调仓 → 分数要预测的是**未来 ~5 个交易日**的收益，
        # 而 42f 的符号表是按 fwd1（1 日）的 IC 定的。用 fwd1 定符号、用 fwd5
        # 定权重才与持有期匹配。
        print("\n  -- 持有期匹配：按 fwd5 ICIR 加权（周频调仓对应 ~5 日持有）--")
        for hz in (1, 3, 5, 10):
            fwd_h = fwd_returns(mats["close"], hz)
            for zname, all_z in (("eng25", z_eng), ("full42", z_full)):
                sc_h = icir_weighted_score(all_z, fwd_h, window=60,
                                          min_periods=20)
                res = L.simulate(sc_h, mats, trad, LEG_CFG["42f"])
                e = record(f"D_icir_fwd{hz}_{zname}", "D", res, LEG_CFG["42f"],
                           note=f"{zname} 按 fwd{hz} RankICIR 加权")
                gD.append(e)
                print_row(e)

        # D6 稳健化标准化：rank / winsorized-z 替代裸 z-score
        # 动机：42f 等权复合用裸 z-score，肥尾因子（VSTD5/VMA5 等）会
        # 少数极端票主导复合分。用 rank 或截断 z 可以消掉这个脆弱点。
        print("\n  -- 稳健化标准化：rank / winsorized-z --")
        for zname, all_z in (("eng25", z_eng), ("full42", z_full)):
            for norm in ("rank", "winsor3"):
                tot = np.zeros_like(s42_engine)
                ws = np.zeros_like(s42_engine)
                for name, z in all_z.items():
                    # z_by_factor 已经带过符号（build_42f_score 里乘过 sign），
                    # 这里只换标准化方式，不再重复乘符号。
                    zz = (cs_rank(z) if norm == "rank"
                          else cs_winsor_z(z, clip=3.0))
                    good = np.isfinite(zz)
                    tot = tot + np.where(good, np.nan_to_num(zz), 0.0)
                    ws = ws + np.where(good, 1.0, 0.0)
                with np.errstate(invalid="ignore", divide="ignore"):
                    sc_n = np.where(ws > 0, tot / np.where(ws > 0, ws, 1.0),
                                    np.nan)
                res = L.simulate(sc_n, mats, trad, LEG_CFG["42f"])
                e = record(f"D_norm_{norm}_{zname}", "D", res, LEG_CFG["42f"],
                           note=f"{zname} 用 {norm} 标准化后等权复合")
                gD.append(e)
                print_row(e)

        # D5 分数平滑（EMA）：周频调仓下对日频分数做时间平滑，抑制换手噪声
        print("\n  -- 分数 EMA 平滑 --")
        for alpha in (0.3, 0.5, 0.7):
            for src, sc in (("eng25", s42_engine), ("rsi", s_rsi)):
                ema = _ema_smooth(sc, alpha)
                cfg = LEG_CFG["42f"] if src == "eng25" else LEG_CFG["rsi"]
                res = L.simulate(ema, mats, trad, cfg)
                e = record(f"D_ema{alpha}_{src}", "D", res, cfg,
                           note=f"{src} 分数 EMA 平滑 alpha={alpha}")
                gD.append(e)
                print_row(e)

        results["groups"]["D_factors"] = gD

    # ══════════ E. 组合构建层 ══════════
    if "E" in groups:
        print("\n" + "=" * 78)
        print("E. 组合构建层")
        print("=" * 78)
        gE = []
        base_score = s42_engine   # 底座 = 引擎当前行为（0.880 对应的那条线）
        for topk, n_drop in ((30, 3), (40, 4), (60, 5), (80, 8), (100, 10),
                             (120, 12), (150, 15)):
            cfg = L.SimConfig(topk=topk, n_drop=n_drop, lag=1,
                              rebalance_days="weekly",
                              max_position_size=(1 - 0.05) / topk)
            res = L.simulate(base_score, mats, trad, cfg)
            e = record(f"E_topk{topk}_drop{n_drop}", "E", res, cfg)
            gE.append(e)
            print_row(e)

        print("\n  -- 调仓频率（42f, topk=60）--")
        for freq in ("daily", "weekly", "monthly"):
            cfg = L.SimConfig(topk=60, n_drop=5, lag=1, rebalance_days=freq)
            res = L.simulate(base_score, mats, trad, cfg)
            e = record(f"E_freq_{freq}", "E", res, cfg)
            gE.append(e)
            print_row(e)

        print("\n  -- 逆波动率加权（相对波动率加权会改变夏普）--")
        inv_vol = _inv_vol(mats, window=60)
        for name, sc, cfg in (("42f", s42_engine, LEG_CFG["42f"]),
                              ("rsi", s_rsi, LEG_CFG["rsi"]),
                              ("42f+rsi", blend(all_scores,
                                                {"42f_engine": 1.0,
                                                 "rsi_71421": 1.0}),
                               LEG_CFG["42f"])):
            sel = L.TopKInverseVolSelector(cfg, inv_vol)
            res = L.simulate(sc, mats, trad, cfg, selector=sel)
            e = record(f"E_invvol_{name}", "E", res, cfg,
                       note="买入金额 ∝ 1/60日波动率")
            gE.append(e)
            print_row(e)

        print("\n  -- 分层选股：按流动性/波动率/行业分 5 层，每层取 top12 --")
        for gname, layer_ids in _layer_ids(mats).items():
            sel = L.LayeredSelector(LEG_CFG["42f"], layer_ids, per_layer=12)
            res = L.simulate(s42_engine, mats, trad, LEG_CFG["42f"], selector=sel)
            e = record(f"E_layered_{gname}", "E", res, LEG_CFG["42f"],
                       note=f"按{gname}分 5 层 × 每层 12 只")
            gE.append(e)
            print_row(e)

        results["groups"]["E_portfolio"] = gE

    # ══════════ F. OOS ══════════
    if "F" in groups:
        print("\n" + "=" * 78)
        print("F. OOS 切分（IS=2023-05→2025-05, OOS=2025-05→2026-05）")
        print("=" * 78)
        gF = []
        variants = {
            "42f_engine": (s42_engine, LEG_CFG["42f"]),
            "42f_day_aligned": (s42_day, LEG_CFG["42f"]),
            "42f_full": (s42_full, LEG_CFG["42f"]),
            "rsi_71421": (s_rsi, LEG_CFG["rsi"]),
            "factor_ranking": (s_fr, LEG_CFG["fr"]),
            "eq_42f+rsi": (blend(all_scores, {"42f_engine": 1.0,
                                             "rsi_71421": 1.0}), LEG_CFG["42f"]),
            "eq_42f+rsi+fr": (blend(all_scores, {"42f_engine": 1.0,
                                                 "rsi_71421": 1.0,
                                                 "factor_ranking": 1.0}),
                              LEG_CFG["42f"]),
            "eq_full+rsi+fr": (blend(all_scores, {"42f_full": 1.0,
                                                  "rsi_71421": 1.0,
                                                  "factor_ranking": 1.0}),
                               LEG_CFG["42f"]),
        }
        for name, (sc, cfg) in variants.items():
            is_res = L.simulate_subset(sc, mats, trad, cfg, is_rows)
            oos_res = L.simulate_subset(sc, mats, trad, cfg, oos_rows)
            e = {
                "name": f"F_{name}",
                "group": "F",
                "IS": {k: (round(float(v), 4) if isinstance(v, (int, float)) else v)
                       for k, v in is_res.items()},
                "OOS": {k: (round(float(v), 4) if isinstance(v, (int, float)) else v)
                        for k, v in oos_res.items()},
            }
            gF.append(e)
            print(f"  {name:22s} IS SR={e['IS']['sharpe_ratio']:6.3f} "
                  f"ret={e['IS']['total_return']:+7.2%}  |  "
                  f"OOS SR={e['OOS']['sharpe_ratio']:6.3f} "
                  f"ret={e['OOS']['total_return']:+7.2%} "
                  f"mdd={e['OOS']['max_drawdown']:7.2%} n={e['OOS']['total_trades']}")
        results["groups"]["F_oos"] = gF

    # ══════════ G. 参数网格 + OOS 切分（防过拟合核心）══════════
    if "G" in groups:
        print("\n" + "=" * 78)
        print("G. (topk, n_drop) 网格 + IS/OOS 切分")
        print("   IS = 2023-05→2025-05 (选参)  OOS = 2025-05→2026-05 (验证)")
        print("=" * 78)
        gG = []
        grid_rows = []
        for sc_name, sc, base in (
                ("42f_engine", s42_engine, LEG_CFG["42f"]),
                ("42f+rsi+fr", blend(all_scores, {"42f_engine": 1.0,
                                                 "rsi_71421": 1.0,
                                                 "factor_ranking": 1.0}),
                 LEG_CFG["42f"]),
        ):
            for topk in (15, 20, 25, 30, 40):
                for n_drop in (1, 2, 3, 5):
                    cfg = L.SimConfig(topk=topk, n_drop=n_drop, lag=1,
                                      rebalance_days="weekly",
                                      max_position_size=(1 - 0.05) / topk)
                    full = L.simulate(sc, mats, trad, cfg)
                    isr = L.simulate_subset(sc, mats, trad, cfg, is_rows)
                    oosr = L.simulate_subset(sc, mats, trad, cfg, oos_rows)
                    e = record(f"G_{sc_name}_k{topk}_d{n_drop}", "G", full, cfg)
                    e["IS_sharpe"] = round(float(isr["sharpe_ratio"]), 4)
                    e["IS_return"] = round(float(isr["total_return"]), 4)
                    e["OOS_sharpe"] = round(float(oosr["sharpe_ratio"]), 4)
                    e["OOS_return"] = round(float(oosr["total_return"]), 4)
                    e["OOS_mdd"] = round(float(oosr["max_drawdown"]), 4)
                    e["OOS_yearly"] = oosr.get("yearly", {})
                    grid_rows.append((sc_name, topk, n_drop, e))
                    gG.append(e)
            print(f"  -- {sc_name} --")
            for s2, t2, d2, e2 in grid_rows:
                if s2 != sc_name:
                    continue
                print(f"    k={t2:3d} d={d2} | 全窗 SR={e2['sharpe_ratio']:6.3f} "
                      f"IS SR={e2['IS_sharpe']:6.3f} | OOS SR={e2['OOS_sharpe']:6.3f} "
                      f"OOS ret={e2['OOS_return']:+7.2%} "
                      f"OOS mdd={e2['OOS_mdd']:7.2%} "
                      f"OOS年={e2['OOS_yearly']}")
            # IS 选参 → OOS 验证（只允许用 IS 选出的那一组）
            best_is = max((r for r in grid_rows if r[0] == sc_name),
                          key=lambda r: r[3]["IS_sharpe"])
            print(f"    ▶ IS 最优 = k{best_is[1]} d{best_is[2]} "
                  f"(IS SR={best_is[3]['IS_sharpe']:.3f}) → "
                  f"OOS SR={best_is[3]['OOS_sharpe']:.3f} "
                  f"OOS ret={best_is[3]['OOS_return']:+.2%}")
            results["groups"].setdefault("G_oos_selected", []).append({
                "score": sc_name,
                "is_selected": {"topk": best_is[1], "n_drop": best_is[2]},
                "is_sharpe": best_is[3]["IS_sharpe"],
                "oos_sharpe": best_is[3]["OOS_sharpe"],
                "oos_return": best_is[3]["OOS_return"],
                "oos_mdd": best_is[3]["OOS_mdd"],
                "oos_yearly": best_is[3]["OOS_yearly"],
            })
        results["groups"]["G_grid"] = gG

    # ══════════ H. 因子子集 + 缓冲区（qlib topk_buffer）══════════
    if "H" in groups:
        print("\n" + "=" * 78)
        print("H. 因子子集选择 + qlib 缓冲区（topk_buffer）")
        print("=" * 78)
        gH = []
        fwd1 = fwd_returns(mats["close"], 1)

        # H1 单因子：每个因子单独打分（sign 已在 z 里），扫 topk
        print("  -- 单因子（eng25 的 25 个公式因子）单独打分 --")
        single_rows = []
        for name, z in z_eng.items():
            sc = z.copy()
            cfg = L.SimConfig(topk=20, n_drop=2, lag=1, rebalance_days="weekly",
                              max_position_size=(1 - 0.05) / 20)
            res = L.simulate(sc, mats, trad, cfg)
            e = record(f"H1_single_{name}_k20", "H", res, cfg)
            single_rows.append((name, e["sharpe_ratio"], e["total_return"],
                                e["max_drawdown"]))
        single_rows.sort(key=lambda r: -r[1])
        for n, sr, rt, md in single_rows[:10]:
            print(f"    {n:10s} SR={sr:6.3f} ret={rt:+7.2%} mdd={md:7.2%}")
        results["groups"].setdefault("H1_single_factors", []).append(
            [{"factor": n, "sharpe": sr, "ret": rt, "mdd": md}
             for n, sr, rt, md in single_rows])

        # H2 walk-forward 因子子集：每 60 天用历史 RankIC 选 top-N 因子
        print("\n  -- walk-forward 因子子集（每 60 天按历史 RankIC 选 topN 因子）--")
        for topn in (3, 5, 8, 12, 25):
            sc = _walkforward_subset_score(z_eng, fwd1, topn=topn,
                                           rebalance_every=60, window=120)
            for topk in (20, 30):
                cfg = L.SimConfig(topk=topk, n_drop=max(2, topk // 12), lag=1,
                                  rebalance_days="weekly",
                                  max_position_size=(1 - 0.05) / topk)
                res = L.simulate(sc, mats, trad, cfg)
                e = record(f"H2_subset{topn}_k{topk}", "H", res, cfg,
                           note=f"walk-forward 选 top{topn} 因子")
                gH.append(e)
                print_row(e)

        # H3 qlib 缓冲区：持仓掉出 topk+buffer 才卖（换手更慢）
        print("\n  -- qlib topk_buffer 缓冲区 --")
        for topk in (20, 30, 60):
            for buf in (0, 5, 10, 20):
                cfg = L.SimConfig(topk=topk, n_drop=max(2, topk // 12), lag=1,
                                  rebalance_days="weekly", buffer=buf,
                                  max_position_size=(1 - 0.05) / topk)
                res = L.simulate(s42_engine, mats, trad, cfg)
                e = record(f"H3_buffer{buf}_k{topk}", "H", res, cfg,
                           note=f"缓冲区 {buf}（换手更慢）")
                gH.append(e)
                print_row(e)

        results["groups"]["H_subset_buffer"] = gH

    # ══════════ I. 新 alpha 源 + 分数比例加权 + 最佳组合 ══════════
    if "I" in groups:
        print("\n" + "=" * 78)
        print("I. 低相关新 alpha 源（长窗口反转 / 换手反转）+ 分数比例加权")
        print("=" * 78)
        gI = []

        # I1 新 alpha 源：长窗口反转（90/120/180/250 日）
        # 动机：42f 的窗口全是 5~60 日，alpha 集中在短周期。
        # 90~250 日的「距长期低点」是**另一个频段**的 alpha，理论上低相关。
        print("  -- 新 alpha 源 A：长窗口反转（距 N 日低点）--")
        long_rev = {}
        for n in (90, 120, 180, 250):
            s = pd.DataFrame(mats["close"])
            long_rev[f"LONGREV{n}"] = (s.rolling(n).min() / s - 1.0).to_numpy()
        z_long = {k: L.cs_zscore(v, 30) for k, v in long_rev.items()}
        for n in (90, 120, 180, 250):
            for k in (20, 60):
                cfg = L.SimConfig(topk=k, n_drop=max(2, k // 12), lag=1,
                                  rebalance_days="weekly",
                                  max_position_size=(1 - 0.05) / k)
                res = L.simulate(z_long[f"LONGREV{n}"], mats, trad, cfg)
                e = record(f"I1_longrev{n}_k{k}", "I", res, cfg)
                gI.append(e)
                print_row(e)

        # I2 新 alpha 源 B：换手率反转（高换手 → 未来低收益）
        # 动机：42f 只用 close/volume 的**水平**（CORR/CORD/MIN…），
        # 没用「换手率水平」本身。这是完全不同的数据维度。
        print("\n  -- 新 alpha 源 B：换手率反转（log 成交额 20 日均值，取负）--")
        amt = mats["close"] * mats["volume"]
        with np.errstate(divide="ignore", invalid="ignore"):
            to = np.log(np.where(amt > 0, amt, np.nan))
        to_mean = pd.DataFrame(to).rolling(20).mean().to_numpy()
        z_to = L.cs_zscore(-to_mean, 30)
        for k in (20, 60):
            cfg = L.SimConfig(topk=k, n_drop=max(2, k // 12), lag=1,
                              rebalance_days="weekly",
                              max_position_size=(1 - 0.05) / k)
            res = L.simulate(z_to, mats, trad, cfg)
            e = record(f"I2_turnover_rev_k{k}", "I", res, cfg)
            gI.append(e)
            print_row(e)

        # I3 新源与 42f 的相关性
        z42 = L.cs_zscore(s42_engine, 30)
        print("\n  -- 新源与 42f 的横截面 RankIC 相关 --")
        for nm, zz in (("LONGREV120", z_long["LONGREV120"]),
                       ("LONGREV250", z_long["LONGREV250"]),
                       ("turnover_rev", z_to)):
            r = float(np.nanmean(L.cs_rank_ic(zz, z42)))
            print(f"    {nm:16s} corr={r:+.3f}")
            results["groups"].setdefault("I_new_alpha_corr", {})[nm] = round(r, 4)

        # I4 三源合成（新源 + 42f + rsi）
        print("\n  -- 三源合成（42f + 长反转 + 换手反转）--")
        for nm, extra in (("42f+longrev120",
                           {"42f_engine": 1.0, "__lr": z_long["LONGREV120"]}),
                          ("42f+to", {"42f_engine": 1.0, "__to": z_to}),
                          ("42f+longrev120+to",
                           {"42f_engine": 1.0, "__lr": z_long["LONGREV120"],
                            "__to": z_to})):
            pool = dict(all_scores)
            pool["__lr"] = z_long["LONGREV120"]
            pool["__to"] = z_to
            wts = {k: v for k, v in extra.items() if not k.startswith("__")}
            wts.update({"__lr": extra.get("__lr") is not None and 1.0 or 0.0,
                        "__to": extra.get("__to") is not None and 1.0 or 0.0})
            sc = blend(pool, wts)
            for k in (20, 60):
                cfg = L.SimConfig(topk=k, n_drop=max(2, k // 12), lag=1,
                                  rebalance_days="weekly",
                                  max_position_size=(1 - 0.05) / k)
                res = L.simulate(sc, mats, trad, cfg)
                e = record(f"I4_{nm}_k{k}", "I", res, cfg)
                gI.append(e)
                print_row(e)

        # I5 分数比例加权（weight ∝ score^power）
        print("\n  -- 分数比例加权（weight ∝ max(score,0)^power）--")
        for k in (20, 60):
            cfg = L.SimConfig(topk=k, n_drop=max(2, k // 12), lag=1,
                              rebalance_days="weekly",
                              max_position_size=(1 - 0.05) / k)
            for power in (0.5, 1.0, 2.0):
                sel = L.ScoreWeightSelector(cfg, power=power)
                res = L.simulate(s42_engine, mats, trad, cfg, selector=sel)
                e = record(f"I5_scorew_p{power}_k{k}", "I", res, cfg,
                           note=f"分数比例加权 power={power}")
                gI.append(e)
                print_row(e)

        results["groups"]["I_new_alpha"] = gI

    # ══════════ J. 单因子发现 + OOS 验证（多重检验防过拟合）══════════
    if "J" in groups:
        print("\n" + "=" * 78)
        print("J. 单因子 + OOS 验证（扫 25 个因子 → 必须用 OOS 复核）")
        print("=" * 78)
        gJ = []
        fwd1 = fwd_returns(mats["close"], 1)
        # 先算每个因子的 IS/OOS 单因子夏普
        print("  -- 25 个单因子 × (IS / OOS) @ k=20, d=2 --")
        single_oos = []
        for name, z in z_eng.items():
            cfg = L.SimConfig(topk=20, n_drop=2, lag=1, rebalance_days="weekly",
                              max_position_size=(1 - 0.05) / 20)
            full = L.simulate(z, mats, trad, cfg)
            isr = L.simulate_subset(z, mats, trad, cfg, is_rows)
            oosr = L.simulate_subset(z, mats, trad, cfg, oos_rows)
            e = record(f"J_single_{name}", "J", full, cfg)
            e["IS_sharpe"] = round(float(isr["sharpe_ratio"]), 4)
            e["OOS_sharpe"] = round(float(oosr["sharpe_ratio"]), 4)
            e["OOS_return"] = round(float(oosr["total_return"]), 4)
            e["OOS_mdd"] = round(float(oosr["max_drawdown"]), 4)
            e["OOS_yearly"] = oosr.get("yearly", {})
            single_oos.append((name, e["sharpe_ratio"], e["IS_sharpe"],
                               e["OOS_sharpe"], e["OOS_return"]))
            gJ.append(e)
        single_oos.sort(key=lambda r: -r[1])
        print(f"    {'因子':10s} {'全窗':>7s} {'IS':>7s} {'OOS':>7s} {'OOS收益':>9s}")
        for n, sf, isf, osf, orf in single_oos:
            print(f"    {n:10s} {sf:7.3f} {isf:7.3f} {osf:7.3f} {orf:+8.2%}")
        # 选 IS 最优 → 看 OOS
        best_is = max(single_oos, key=lambda r: r[2])
        best_full = max(single_oos, key=lambda r: r[1])
        print(f"    ▶ IS 最优单因子 = {best_is[0]} (IS {best_is[2]:.3f}) → "
              f"OOS {best_is[3]:.3f}")
        print(f"    ▶ 全窗最优单因子 = {best_full[0]} (全窗 {best_full[1]:.3f}) → "
              f"OOS {best_full[3]:.3f}")
        results["groups"].setdefault("J_single_oos", []).append({
            "is_selected": best_is[0], "is_sharpe": best_is[2],
            "is_selected_oos": best_is[3],
            "full_selected": best_full[0], "full_sharpe": best_full[1],
            "full_selected_oos": best_full[3],
            "n_factors_tried": len(single_oos),
        })

        # J2: top-3 单因子按 IS 合成（walk-forward 选因子太慢，先用静态 top3）
        top3 = [r[0] for r in sorted(single_oos, key=lambda r: -r[2])[:3]]
        print(f"\n  -- IS top3 单因子等权合成: {top3} --")
        for k in (20, 30, 60):
            cfg = L.SimConfig(topk=k, n_drop=max(2, k // 12), lag=1,
                              rebalance_days="weekly",
                              max_position_size=(1 - 0.05) / k)
            zmix = np.zeros_like(s42_engine)
            cnt = np.zeros_like(s42_engine)
            for n in top3:
                good = np.isfinite(z_eng[n])
                zmix = zmix + np.where(good, np.nan_to_num(z_eng[n]), 0.0)
                cnt = cnt + np.where(good, 1.0, 0.0)
            sc = np.where(cnt > 0, zmix / np.where(cnt > 0, cnt, 1.0), np.nan)
            full = L.simulate(sc, mats, trad, cfg)
            isr = L.simulate_subset(sc, mats, trad, cfg, is_rows)
            oosr = L.simulate_subset(sc, mats, trad, cfg, oos_rows)
            e = record(f"J2_top3mix_k{k}", "J", full, cfg,
                       note=f"IS top3 单因子合成 {top3}")
            e["IS_sharpe"] = round(float(isr["sharpe_ratio"]), 4)
            e["OOS_sharpe"] = round(float(oosr["sharpe_ratio"]), 4)
            e["OOS_return"] = round(float(oosr["total_return"]), 4)
            e["OOS_mdd"] = round(float(oosr["max_drawdown"]), 4)
            gJ.append(e)
            print(f"    k={k:3d} 全窗 SR={e['sharpe_ratio']:6.3f} "
                  f"IS={e['IS_sharpe']:6.3f} OOS={e['OOS_sharpe']:6.3f} "
                  f"OOS ret={e['OOS_return']:+7.2%} mdd={e['OOS_mdd']:7.2%}")

        # J3: 42f 与单因子混合（单因子占小权重）
        print("\n  -- 42f + 单因子（不同权重）--")
        for fname in ("CORD5", "CORR20", "STD20"):
            for w in (0.25, 0.5, 1.0):
                cfg = L.SimConfig(topk=20, n_drop=2, lag=1,
                                  rebalance_days="weekly",
                                  max_position_size=(1 - 0.05) / 20)
                zmix = (L.cs_zscore(s42_engine, 30)
                        + w * L.cs_zscore(z_eng[fname], 30))
                full = L.simulate(zmix, mats, trad, cfg)
                isr = L.simulate_subset(zmix, mats, trad, cfg, is_rows)
                oosr = L.simulate_subset(zmix, mats, trad, cfg, oos_rows)
                e = record(f"J3_42f+{fname}_w{w}", "J", full, cfg,
                           note=f"42f + {w}×{fname}")
                e["IS_sharpe"] = round(float(isr["sharpe_ratio"]), 4)
                e["OOS_sharpe"] = round(float(oosr["sharpe_ratio"]), 4)
                e["OOS_return"] = round(float(oosr["total_return"]), 4)
                e["OOS_mdd"] = round(float(oosr["max_drawdown"]), 4)
                gJ.append(e)
                print(f"    +{fname:8s} w={w:<5} 全窗 SR={e['sharpe_ratio']:6.3f} "
                      f"IS={e['IS_sharpe']:6.3f} OOS={e['OOS_sharpe']:6.3f} "
                      f"OOS ret={e['OOS_return']:+7.2%} mdd={e['OOS_mdd']:7.2%}")

        results["groups"]["J_single_oos"] = gJ

    # ---- 汇总 ----
    results["trials"] = TRIALS
    results["n_trials"] = len(TRIALS)
    ranked = sorted(TRIALS, key=lambda e: -e["sharpe_ratio"])
    results["top5"] = ranked[:5]
    print("\n" + "=" * 78)
    print(f"共 {len(TRIALS)} 个变体。Top5：")
    for e in ranked[:5]:
        print_row(e)
    print("=" * 78)

    path = os.path.join(L.OUT_DIR, args.out)
    with open(path, "w") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=1, default=str)
    print(f"写入 {path}  ({time.time()-t0:.0f}s)")


# ══════════════════════════════════════════════════════════════════
# 需要额外实现的模拟器变体 + GBM 腿
# ══════════════════════════════════════════════════════════════════
def _inv_vol(mats: Dict[str, Any], window: int = 60) -> np.ndarray:
    """每只股票的逆波动率（1/60日收益标准差），用于相对波动率加权。"""
    close = mats["close"]
    with np.errstate(divide="ignore", invalid="ignore"):
        ret = close[1:] / close[:-1] - 1.0
        ret = np.vstack([np.full((1, close.shape[1]), np.nan), ret])
        vol = pd.DataFrame(ret).rolling(window).std().to_numpy()
    inv = 1.0 / np.where(vol > 1e-9, vol, np.nan)
    # 归一到截面均值 1
    with np.errstate(invalid="ignore"):
        m = np.nanmean(inv, axis=1, keepdims=True)
        inv = np.where(np.isfinite(inv), inv / np.where(m > 0, m, 1.0), np.nan)
    return inv


def _walkforward_subset_score(
    z_factors: Dict[str, np.ndarray],
    fwd: np.ndarray,
    topn: int,
    rebalance_every: int = 60,
    window: int = 120,
) -> np.ndarray:
    """walk-forward 因子子集：每 rebalance_every 天用**过去** window 天的
    RankIC 选出 topn 个因子，等权复合；其它日期沿用上一次的子集。

    无前视：第 t 天用的 IC 只来自 [t-window, t-1]。
    """
    names = list(z_factors)
    n_d = z_factors[names[0]].shape[0]
    ics: Dict[str, np.ndarray] = {
        n: L.cs_rank_ic(z_factors[n], fwd) for n in names
    }
    score = np.full((n_d, z_factors[names[0]].shape[1]), np.nan)
    chosen = names[:topn]
    for t in range(n_d):
        if t % rebalance_every == 0 and t >= window:
            seg = {n: ics[n][t - window:t] for n in names}
            means = {n: (float(np.nanmean(v)) if np.isfinite(v).any() else -9.9)
                     for n, v in seg.items()}
            chosen = sorted(means, key=lambda n: -means[n])[:topn]
        tot = np.zeros(score.shape[1])
        cnt = np.zeros(score.shape[1])
        for n in chosen:
            z = z_factors[n]
            good = np.isfinite(z[t])
            tot = tot + np.where(good, np.nan_to_num(z[t]), 0.0)
            cnt = cnt + np.where(good, 1.0, 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            score[t] = np.where(cnt > 0, tot / np.where(cnt > 0, cnt, 1.0), np.nan)
    return score


def _ema_smooth(score: np.ndarray, alpha: float) -> np.ndarray:
    """对分数做时间维 EMA 平滑（逐列，忽略 NaN）。

    alpha=1 → 原分；alpha 越小 → 越平滑。用于抑制周频调仓下的换手噪声。
    """
    out = np.full_like(score, np.nan)
    for j in range(score.shape[1]):
        prev = np.nan
        for t in range(score.shape[0]):
            v = score[t, j]
            if not np.isfinite(v):
                out[t, j] = prev
                continue
            prev = v if not np.isfinite(prev) else alpha * v + (1 - alpha) * prev
            out[t, j] = prev
    return out


def cs_rank(mat: np.ndarray, min_cross: int = 30) -> np.ndarray:
    """逐日截面 rank → 均匀分布 [-0.5, 0.5]（对异常值稳健）。"""
    df = pd.DataFrame(mat)
    cnt = df.notna().sum(axis=1)
    r = df.rank(axis=1, na_option="keep")
    n = df.notna().sum(axis=1).replace(0, np.nan)
    out = ((r - 0.5) / n.to_numpy()[:, None]).to_numpy()
    bad = (cnt < min_cross).to_numpy()
    out[bad] = np.nan
    return out


def cs_winsor_z(mat: np.ndarray, min_cross: int = 30, clip: float = 3.0) -> np.ndarray:
    """逐日截面 z-score 后把两端截断到 ±clip（防肥尾因子污染等权复合）。"""
    z = L.cs_zscore(mat, min_cross)
    return np.where(np.isfinite(z), np.clip(z, -clip, clip), np.nan)


def cs_resid_against(y: np.ndarray, Xs: Sequence[np.ndarray],
                      min_cross: int = 30) -> np.ndarray:
    """逐日截面回归残差：y 对 Xs 里的每个暴露做 OLS，取残差。

    y : (n_dates, n_stocks) 目标
    Xs: 若干个 (n_dates, n_stocks) 待剔除的暴露
    """
    out = np.full(y.shape, np.nan)
    X = np.stack([x for x in Xs], axis=2)      # (n_dates, n_stocks, n_k)
    for t in range(y.shape[0]):
        yy = y[t]
        xx = X[t]
        m = np.isfinite(yy) & np.isfinite(xx).all(axis=1)
        if int(m.sum()) < min_cross:
            continue
        A = np.vstack([xx[m].T, np.ones(int(m.sum()))]).T
        beta, *_ = np.linalg.lstsq(A, yy[m], rcond=None)
        out[t, m] = yy[m] - A @ beta
    return out


def _layer_ids(mats: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """分层选股用的分组变量：按流动性/波动率/行业分层。

    返回 {名称: 每只股票的 layer id 数组}。
    """
    out: Dict[str, np.ndarray] = {}
    close = mats["close"]
    vol = mats["volume"]

    # 1) 成交额分层（用窗口内累计成交额中位数近似市值/流动性）
    amt = close * vol
    with np.errstate(invalid="ignore"):
        liq = np.nanmedian(amt, axis=0)
    out["liquidity"] = _quantile_bins(liq, 5)

    # 2) 波动率分层
    with np.errstate(divide="ignore", invalid="ignore"):
        ret = close[1:] / close[:-1] - 1.0
        ret = np.vstack([np.full((1, close.shape[1]), np.nan), ret])
        v = np.nanmedian(pd.DataFrame(ret).rolling(60).std().to_numpy(), axis=0)
    out["volatility"] = _quantile_bins(v, 5)

    # 3) 行业分层（申万一级，取 top5 行业 + 其他）
    try:
        from app.services.trading.execution_rules import (
            industry_ids, load_industry_map,
        )
        imap = load_industry_map()
        ids = industry_ids(list(mats["codes"]), imap)
        uniq = np.unique(ids[ids >= 0])
        top5 = set(uniq[np.argsort([-(ids == u).sum() for u in uniq])[:5]].tolist())
        lay = np.full(len(mats["codes"]), len(top5), dtype=int)
        for i, v_ in enumerate(ids):
            if v_ in top5:
                lay[i] = list(top5).index(v_)
        out["industry"] = lay
    except Exception:
        pass
    return out


def _quantile_bins(vals: np.ndarray, n_bins: int) -> np.ndarray:
    """按分位数分箱（NaN → 中间箱）。"""
    v = np.asarray(vals, dtype=np.float64)
    finite = np.isfinite(v)
    if finite.sum() == 0:
        return np.zeros(len(v), dtype=int)
    qs = np.nanquantile(v[finite], np.linspace(0, 1, n_bins + 1)[1:-1])
    b = np.digitize(v, qs)
    b[~finite] = n_bins // 2
    return b.astype(int)


def _gbm_score(mats: Dict[str, Any]) -> np.ndarray:
    """gbm_voltgt 的日频分数（复用策略类，月度滚动重训）。"""
    from app.services.backtest.strategies.gbm_voltgt_strategy import (
        GbmVoltgtStrategy,
    )
    import asyncio

    strategy = GbmVoltgtStrategy({"topk": 60, "n_drop": 5, "lag": 0,
                                  "rebalance_days": "daily"})
    # gbm 的 prepare 会读 panel（相对路径）→ 用绝对路径
    strategy.panel_dir = os.path.join(L.QLIB_LIVE, "features", "day")
    asyncio.run(strategy.prepare_backtest_data(mats["stock_data"],
                                               L.START, L.END))
    attr = "_gbm_voltgt_scores"
    out = np.full_like(mats["close"], np.nan)
    pos_of = {d: i for i, d in enumerate(mats["master"])}
    for j, code in enumerate(mats["codes"]):
        sm = mats["stock_data"][code].attrs.get(attr) or {}
        for d, v in sm.items():
            i = pos_of.get(pd.Timestamp(d))
            if i is not None:
                out[i, j] = v
    return out


if __name__ == "__main__":
    main()
