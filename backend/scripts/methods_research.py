#!/usr/bin/env python3
"""横截面选股「组合构建层 / 信号处理层」方法调研与实测。

## 目的
从成熟开源项目与学术方法里，找出**具体可复现**、能实打实提升 A 股横截面
选股夏普的做法，在本平台用**真实盘口径**跑出数字。

## 口径（与 verify_10y_pit.py --lag 1 逐项一致）
- 窗口 2023-05-08 → 2026-05-08，727 个交易日
- 股票池 data/qlib_official_data/instruments/all.txt（PIT 上市/退市日）
- lag=1（收盘出信号，下一交易日成交）
- enable_tradability=True（涨停买不进 / 跌停卖不出 / 停牌 / 流动性上限）
- TopK-Dropout（topk=60, n_drop=5, 等权, max_position_size=(1-0.05)/60）
- 成本：commission 0.03% + slippage 0.01%（单边）、整手 100 股、现金预留 5%

## 基线说明（重要）
引擎实测 honest_composite（42 因子，panel 路径已修）→ **夏普 0.7992**。
本脚本的 `42f_engine` 变体（复刻引擎 CWD=根 的 25 因子降级行为）→ **0.8796**。
两者都作为基线报出，所有改进都在**同一个底座**上比较。

## 实验组
  A. 基线复现 + 保真度验收（模拟器 vs 真实引擎）
  B. 信号平滑（EWMA 半衰期 / 移动平均）—— Barra 因子暴露平滑
  C. IC 加权复合（ICIR / Newey-West 调整 + 收缩）—— Grinold-Kahn
  D. 换手惩罚 / 成本内生化（Amihud 非流动性惩罚）
  E. 分组配额选股（Barra APT 组约束）
  F. 组合构建层权重（逆波动率 / HRP / 最大分散化 / 分数比例）
  G. 多周期信号合成
  H. alpha decay 分析（因子半衰期 → 持有期）
  I. 组合拳（各层叠加）
  J. OOS 切分（IS 2023-05→2025-05 选参 / OOS 2025-05→2026-05 验证）

## 用法
  env -u PYTHONPATH -u PYTHONHOME backend/.venv-py313/bin/python3 \
    backend/scripts/methods_research.py --groups A B C D E F G H I J
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
import methods_lab as M  # noqa: E402

OOS_SPLIT = pd.Timestamp("2025-05-08")
OUT_DIR = os.path.join(ROOT, "data", "backtest")

# 引擎实测（verify_10y_pit.py --years 3 --lag 1 --only honest_composite，
# 2026-10-26 面板路径修正后 = 真正的 42 因子）
ENGINE_BASELINE_42F = {
    "sharpe_ratio": 0.7991795250472001,
    "total_return": 0.34779248262737783,
    "max_drawdown": -0.16460289642093642,
    "total_trades": 1080,
    "yearly": {"2024": 0.148041, "2025": 0.092864, "2026": 0.045197},
}
# 引擎实测的 25 因子降级行为（CWD=根 → panel 解析失败）
ENGINE_BASELINE_25F = {
    "sharpe_ratio": 0.8795668740640868,
    "total_return": 0.35505797455182114,
    "max_drawdown": -0.1337121157482497,
    "total_trades": 1190,
    "yearly": {"2024": 0.160944, "2025": 0.090579, "2026": 0.046826},
}

TRIALS: List[Dict[str, Any]] = []


def record(
    name: str,
    group: str,
    res: Dict[str, Any],
    cfg: L.SimConfig,
    note: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    e = {
        "name": name,
        "group": group,
        "sharpe_ratio": round(float(res["sharpe_ratio"]), 4),
        "total_return": round(float(res["total_return"]), 4),
        "max_drawdown": round(float(res["max_drawdown"]), 4),
        "total_trades": int(res["total_trades"]),
        "volatility": round(float(res.get("volatility", 0.0)), 4),
        "yearly": res.get("yearly", {}),
        "config": {
            "topk": cfg.topk, "n_drop": cfg.n_drop, "lag": cfg.lag,
            "rebalance_days": cfg.rebalance_days,
        },
        "note": note,
    }
    if extra:
        e.update(extra)
    TRIALS.append(e)
    return e


def print_row(e: Dict[str, Any]) -> None:
    print(f"  {e['name'][:44]:46s} SR={e['sharpe_ratio']:6.3f} "
          f"ret={e['total_return']:+8.2%} mdd={e['max_drawdown']:7.2%} "
          f"n={e['total_trades']:5d}  {e.get('note','')[:44]}", flush=True)


def oos_eval(
    score: np.ndarray,
    mats: Dict[str, Any],
    trad: Dict[str, np.ndarray],
    cfg: L.SimConfig,
    selector: Optional[L.Selector] = None,
    is_rows: Optional[np.ndarray] = None,
    oos_rows: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """返回 (IS 结果, OOS 结果)。"""
    if is_rows is None or oos_rows is None:
        is_rows, oos_rows = L.split_index(mats["master"], OOS_SPLIT)
    sub_mats_is = L.slice_mats(mats, is_rows)
    sub_mats_oos = L.slice_mats(mats, oos_rows)
    sub_trad_is = {k: v[is_rows] for k, v in trad.items()}
    sub_trad_oos = {k: v[oos_rows] for k, v in trad.items()}
    r_is = L.simulate(score[is_rows], sub_mats_is, sub_trad_is, cfg,
                      selector=selector)
    r_oos = L.simulate(score[oos_rows], sub_mats_oos, sub_trad_oos, cfg,
                       selector=selector)
    return r_is, r_oos


def r4(d: Dict[str, Any]) -> Dict[str, Any]:
    return {
        k: (round(float(v), 4) if isinstance(v, (int, float)) else v)
        for k, v in d.items()
    }


# ══════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="*", default=["A", "B", "C", "D", "E",
                                                    "F", "G", "H", "I", "J"])
    ap.add_argument("--out", default="methods_research.json")
    ap.add_argument("--topk", type=int, default=60)
    ap.add_argument("--ndrop", type=int, default=5)
    args = ap.parse_args()
    groups = set(args.groups)

    t0 = time.time()
    print("=" * 84)
    print("横截面选股 · 组合构建层 / 信号处理层 方法实测（真实盘口径 lag=1）")
    print("=" * 84)

    cache = M.load_cached_mats()
    mats = cache["mats"]
    trad = cache["trad"]
    print(f"[setup] {len(mats['codes'])} stocks × {len(mats['master'])} days "
          f"({time.time()-t0:.0f}s)", flush=True)

    # ── 分数底座 ──────────────────────────────────────────────────
    # 42f_engine: 复刻引擎「CWD=根 → panel 解析失败 → 25 因子降级」行为
    s25, z_eng = L.build_42f_score(mats, panel_aligned=False)
    # 42f_day: 17 个 Alpha158 面板因子按交易日正确对齐（= 今天引擎的真实行为）
    s42_day, z_day = L.build_42f_score(
        mats, panel_aligned=True,
        panel_dir=os.path.join(L.QLIB_LIVE, "features", "day"))
    print(f"[setup] scores ready ({time.time()-t0:.0f}s)", flush=True)

    # 底座选择：今天的引擎真实跑的是 42f_day（0.7992）。
    # 但 25f（0.8796）在同口径下更高 → 两条线都作为研究底座。
    BASE = s42_day
    BASE_NAME = "42f_day"
    CFG = L.SimConfig(topk=args.topk, n_drop=args.ndrop, lag=1,
                      rebalance_days="weekly",
                      max_position_size=(1 - 0.05) / args.topk)
    is_rows, oos_rows = L.split_index(mats["master"], OOS_SPLIT)
    print(f"[setup] IS={len(is_rows)} days, OOS={len(oos_rows)} days", flush=True)

    results: Dict[str, Any] = {
        "meta": {
            "window": f"{mats['master'][0].date()}→{mats['master'][-1].date()}",
            "n_codes": len(mats["codes"]),
            "n_trading_days": len(mats["master"]),
            "oos_split": str(OOS_SPLIT.date()),
            "caliber": {
                "lag": 1, "enable_tradability": True,
                "topk": args.topk, "n_drop": args.ndrop,
                "max_position_size": CFG.max_position_size,
                "commission": 0.0003, "slippage": 0.0001,
                "board_lot": 100, "cash_reserve": 0.05,
                "rebalance": "weekly(周五)",
            },
            "engine_baseline_42f": ENGINE_BASELINE_42F,
            "engine_baseline_25f": ENGINE_BASELINE_25F,
            "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        },
        "groups": {},
    }

    # ══════════ A. 基线复现 + 保真度验收 ══════════
    if "A" in groups:
        print("\n" + "=" * 84)
        print("A. 基线复现（模拟器 vs 真实引擎）")
        print("=" * 84)
        gA = []
        for nm, sc, ref in (("25f_engine", s25, ENGINE_BASELINE_25F),
                            ("42f_day", s42_day, ENGINE_BASELINE_42F)):
            res = L.simulate(sc, mats, trad, CFG)
            e = record(f"A_{nm}", "A", res, CFG, note="基线")
            e["engine_sharpe"] = ref["sharpe_ratio"]
            e["engine_diff"] = round(e["sharpe_ratio"] - ref["sharpe_ratio"], 4)
            e["engine_trades"] = ref["total_trades"]
            gA.append(e)
            print_row(e)
            print(f"    └─ 引擎 SR={ref['sharpe_ratio']:.4f}  差={e['engine_diff']:+.4f}")
        results["groups"]["A_baseline"] = gA

    # ══════════ B. 信号平滑 ══════════
    if "B" in groups:
        print("\n" + "=" * 84)
        print("B. 信号平滑（Barra 因子暴露 EWMA / 移动平均）")
        print("=" * 84)
        gB = []
        for hl in (0, 2, 5, 10, 20, 40, 60):
            sc = BASE if hl == 0 else M.ewma_smooth(BASE, halflife=hl)
            res = L.simulate(sc, mats, trad, CFG)
            e = record(f"B_ewma_hl{hl}", "B", res, CFG,
                       note=f"EWMA 半衰期 {hl} 日" if hl else "不平滑(基线)")
            gB.append(e)
            print_row(e)
        for w in (3, 5, 10, 20):
            sc = M.moving_average_smooth(BASE, w)
            res = L.simulate(sc, mats, trad, CFG)
            e = record(f"B_ma{w}", "B", res, CFG, note=f"{w} 日移动平均")
            gB.append(e)
            print_row(e)
        results["groups"]["B_smoothing"] = gB

    # ══════════ C. IC 加权复合 ══════════
    if "C" in groups:
        print("\n" + "=" * 84)
        print("C. IC 加权复合（ICIR / Newey-West + 收缩，walk-forward）")
        print("=" * 84)
        gC = []
        # ensemble_lab 没有 fwd_returns → 自己实现
        def fwd_returns(close: np.ndarray, horizon: int) -> np.ndarray:
            out = np.full_like(close, np.nan)
            if horizon < close.shape[0]:
                out[:-horizon] = close[horizon:] / close[:-horizon] - 1.0
            return out

        fwd1 = fwd_returns(mats["close"], 1)
        fwd5 = fwd_returns(mats["close"], 5)
        for hz, fwd in (("fwd1", fwd1), ("fwd5", fwd5)):
            for use_nw in (False, True):
                for shrink in (False, True):
                    for win in (60, 120):
                        sc, diag = M.icir_weighted_score(
                            z_day, fwd, window=win, min_periods=max(20, win // 3),
                            use_nw=use_nw, shrink=shrink)
                        res = L.simulate(sc, mats, trad, CFG)
                        e = record(
                            f"C_icir_{hz}_nw{int(use_nw)}_sh{int(shrink)}_w{win}",
                            "C", res, CFG,
                            note=f"{hz} ICIR, NW={use_nw}, 收缩={shrink}, win={win}")
                        gC.append(e)
                        print_row(e)
        results["groups"]["C_ic_weighting"] = gC

    # ══════════ D. 换手惩罚 / 成本内生化 ══════════
    if "D" in groups:
        print("\n" + "=" * 84)
        print("D. 成本内生化（Amihud 非流动性惩罚，扣在选股层）")
        print("=" * 84)
        gD = []
        illiq = M.amihud_illiq(mats["close"], mats["volume"], window=20)
        for lam in (0.0, 0.05, 0.1, 0.2, 0.4, 0.8):
            sc = M.cost_penalized_score(BASE, illiq, lam)
            res = L.simulate(sc, mats, trad, CFG)
            e = record(f"D_cost_lam{lam}", "D", res, CFG,
                       note=f"score - {lam}×z(Amihud)")
            gD.append(e)
            print_row(e)
        results["groups"]["D_cost_penalty"] = gD

    # ══════════ E. 分组配额选股 ══════════
    if "E" in groups:
        print("\n" + "=" * 84)
        print("E. 分组配额选股（Barra APT 式组约束）")
        print("=" * 84)
        gE = []
        # E1 行业配额（静态）
        try:
            from app.services.trading.execution_rules import (
                industry_ids, load_industry_map,
            )
            imap = load_industry_map()
            ids = industry_ids(list(mats["codes"]), imap)
            n_ind = int(ids.max()) + 1
            for cap in (2, 3, 4, 6):
                sc = M.group_cap_mask(BASE, ids, cap=cap, min_cross=30)
                res = L.simulate(sc, mats, trad, CFG)
                e = record(f"E_industry_cap{cap}", "E", res, CFG,
                           note=f"申万一级每组最多 {cap} 只")
                gE.append(e)
                print_row(e)
            results["groups"].setdefault("E_group", {})["n_industries"] = n_ind
        except Exception as exc:
            print(f"  [E] 行业配额跳过: {exc}", flush=True)

        # E2 流动性分层配额（逐日动态层）
        amount = mats["close"] * mats["volume"]
        for n_layers in (3, 5):
            layer = M.dynamic_layer_ids(amount, n_layers=n_layers)
            # 逐日配额：需要按日构造分数 → 用「层内排名归一化 + 组配额」
            # 简化：直接把每层内的分数改成「层内秩」，再全局按配额截断。
            base_z = L.cs_zscore(BASE, 30)
            for cap in (int(args.topk / n_layers), int(args.topk / n_layers) + 4):
                sc = np.full_like(BASE, np.nan)
                for t in range(BASE.shape[0]):
                    row = base_z[t]
                    lay = layer[t]
                    valid = np.where(np.isfinite(row))[0]
                    if len(valid) < 30:
                        continue
                    order = valid[np.argsort(-row[valid], kind="mergesort")]
                    counts: Dict[int, int] = {}
                    keep = []
                    for j in order:
                        g = int(lay[j])
                        if counts.get(g, 0) >= cap:
                            continue
                        counts[g] = counts.get(g, 0) + 1
                        keep.append(j)
                        if len(keep) >= args.topk:
                            break
                    if keep:
                        sc[t, keep] = row[keep]
                res = L.simulate(sc, mats, trad, CFG)
                e = record(f"E_liqlayer{n_layers}_cap{cap}", "E", res, CFG,
                           note=f"{n_layers} 流动性层 × 每层 {cap}")
                gE.append(e)
                print_row(e)
        results["groups"]["E_group_cap"] = gE

    # ══════════ F. 组合构建层权重 ══════════
    if "F" in groups:
        print("\n" + "=" * 84)
        print("F. 组合构建层权重（逆波动率 / HRP / 最大分散化 / 分数比例）")
        print("=" * 84)
        gF = []

        # F1 分数比例加权（score ∝ 分数^p）
        for power in (0.5, 1.0, 2.0):
            sel = L.ScoreWeightSelector(CFG, power=power)
            res = L.simulate(BASE, mats, trad, CFG, selector=sel)
            e = record(f"F_scorew_p{power}", "F", res, CFG,
                       note=f"权重 ∝ score^{power}")
            gF.append(e)
            print_row(e)

        # F2 逆波动率
        for lookback in (20, 60, 120):
            sel = _InvVolSelector(CFG, mats["close"], lookback)
            res = L.simulate(BASE, mats, trad, CFG, selector=sel)
            e = record(f"F_invvol_{lookback}", "F", res, CFG,
                       note=f"w ∝ 1/σ({lookback}d)")
            gF.append(e)
            print_row(e)

        # F3 HRP（Lopez de Prado 2016）
        for lookback in (60, 120):
            sel = _HRPSelector(CFG, mats["close"], lookback)
            res = L.simulate(BASE, mats, trad, CFG, selector=sel)
            e = record(f"F_hrp_{lookback}", "F", res, CFG,
                       note=f"HRP 权重 lookback={lookback}")
            gF.append(e)
            print_row(e)

        # F4 最大分散化（Maillard et al. 2010）
        for lookback in (60, 120):
            sel = _MaxDivSelector(CFG, mats["close"], lookback)
            res = L.simulate(BASE, mats, trad, CFG, selector=sel)
            e = record(f"F_maxdiv_{lookback}", "F", res, CFG,
                       note=f"w ∝ Σ⁻¹σ lookback={lookback}")
            gF.append(e)
            print_row(e)
        results["groups"]["F_portfolio_weights"] = gF

    # ══════════ G. 多周期信号合成 ══════════
    if "G" in groups:
        print("\n" + "=" * 84)
        print("G. 多周期信号合成（快/慢信号混合）")
        print("=" * 84)
        gG = []
        # 快 = 原始日频；慢 = EWMA 平滑 / 移动平均
        fast = BASE
        for slow_hl in (10, 20, 40):
            slow = M.ewma_smooth(BASE, halflife=slow_hl)
            for w_slow in (0.25, 0.5, 0.75):
                sc = M.blend_scores({"fast": fast, "slow": slow},
                                    {"fast": 1 - w_slow, "slow": w_slow})
                res = L.simulate(sc, mats, trad, CFG)
                e = record(f"G_blend_ewma{slow_hl}_w{w_slow}", "G", res, CFG,
                           note=f"日频 + {w_slow}×EWMA({slow_hl})")
                gG.append(e)
                print_row(e)
        results["groups"]["G_multihorizon"] = gG

    # ══════════ H. alpha decay 分析 ══════════
    if "H" in groups:
        print("\n" + "=" * 84)
        print("H. alpha decay 分析（复合分对未来 h 日收益的 IC 衰减）")
        print("=" * 84)

        def fwd_returns(close: np.ndarray, horizon: int) -> np.ndarray:
            out = np.full_like(close, np.nan)
            if horizon < close.shape[0]:
                out[:-horizon] = close[horizon:] / close[:-horizon] - 1.0
            return out

        decay_rows = []
        for nm, sc in (("42f_day", s42_day), ("25f", s25)):
            for hz in (1, 2, 3, 5, 10, 20, 40, 60):
                fwd = fwd_returns(mats["close"], hz)
                ic = L.cs_rank_ic(sc, fwd)
                # 只用 IS 段算（避免用 OOS 选持有期）
                ic_is = ic[is_rows]
                mean_ic = float(np.nanmean(ic_is))
                se, t = _mean_t(ic_is)
                decay_rows.append({
                    "score": nm, "horizon": hz, "ic_mean": round(mean_ic, 5),
                    "ic_std": round(float(np.nanstd(ic_is)), 5),
                    "icir": round(mean_ic / float(np.nanstd(ic_is)), 4)
                            if np.nanstd(ic_is) > 0 else None,
                    "t_naive": round(t, 3),
                })
                print(f"  {nm:10s} h={hz:3d}  IC={mean_ic:+.5f}  "
                      f"ICIR={mean_ic/max(float(np.nanstd(ic_is)),1e-12):+.3f}  "
                      f"t={t:+.2f}")
        results["groups"]["H_alpha_decay"] = decay_rows

    # ══════════ I. 组合拳 ══════════
    if "I" in groups:
        print("\n" + "=" * 84)
        print("I. 组合拳（平滑 + IC 加权 + 成本惩罚 + 权重层）")
        print("=" * 84)
        gI = []
        illiq = M.amihud_illiq(mats["close"], mats["volume"], window=20)

        def fwd_returns(close: np.ndarray, horizon: int) -> np.ndarray:
            out = np.full_like(close, np.nan)
            if horizon < close.shape[0]:
                out[:-horizon] = close[horizon:] / close[:-horizon] - 1.0
            return out

        fwd5 = fwd_returns(mats["close"], 5)
        sc_ic, _ = M.icir_weighted_score(z_day, fwd5, window=120,
                                         min_periods=40, use_nw=True,
                                         shrink=True)
        cands: Dict[str, np.ndarray] = {
            "base": BASE,
            "ewma10": M.ewma_smooth(BASE, halflife=10),
            "ewma20": M.ewma_smooth(BASE, halflife=20),
            "ic_nw": sc_ic,
            "ic_nw_ewma20": M.ewma_smooth(sc_ic, halflife=20),
        }
        for cname, sc in cands.items():
            for lam in (0.0, 0.1, 0.2):
                s2 = M.cost_penalized_score(sc, illiq, lam) if lam else sc
                res = L.simulate(s2, mats, trad, CFG)
                e = record(f"I_{cname}_cost{lam}", "I", res, CFG,
                           note=f"{cname} + 成本惩罚 {lam}")
                gI.append(e)
                print_row(e)
        results["groups"]["I_combos"] = gI

    # ══════════ J. OOS ══════════
    if "J" in groups:
        print("\n" + "=" * 84)
        print("J. OOS 切分（IS 2023-05→2025-05 选参 / OOS 2025-05→2026-05 验证）")
        print("=" * 84)

        def fwd_returns(close: np.ndarray, horizon: int) -> np.ndarray:
            out = np.full_like(close, np.nan)
            if horizon < close.shape[0]:
                out[:-horizon] = close[horizon:] / close[:-horizon] - 1.0
            return out

        illiq = M.amihud_illiq(mats["close"], mats["volume"], window=20)
        fwd5 = fwd_returns(mats["close"], 5)
        sc_ic, _ = M.icir_weighted_score(z_day, fwd5, window=120,
                                         min_periods=40, use_nw=True,
                                         shrink=True)
        variants: Dict[str, Tuple[np.ndarray, Optional[L.Selector]]] = {
            "base_42f_day": (BASE, None),
            "base_25f": (s25, None),
            "ewma10": (M.ewma_smooth(BASE, halflife=10), None),
            "ewma20": (M.ewma_smooth(BASE, halflife=20), None),
            "ewma40": (M.ewma_smooth(BASE, halflife=40), None),
            "ic_nw": (sc_ic, None),
            "ic_nw_ewma20": (M.ewma_smooth(sc_ic, halflife=20), None),
            "base_cost0.1": (M.cost_penalized_score(BASE, illiq, 0.1), None),
            "base_cost0.2": (M.cost_penalized_score(BASE, illiq, 0.2), None),
            "ewma20_cost0.1": (
                M.cost_penalized_score(M.ewma_smooth(BASE, halflife=20),
                                       illiq, 0.1), None),
        }
        gJ = []
        for vname, (sc, sel) in variants.items():
            r_is, r_oos = oos_eval(sc, mats, trad, CFG, selector=sel,
                                   is_rows=is_rows, oos_rows=oos_rows)
            e = {
                "name": f"J_{vname}", "group": "J",
                "IS": r4(r_is), "OOS": r4(r_oos),
            }
            gJ.append(e)
            print(f"  {vname:18s} IS SR={r_is['sharpe_ratio']:6.3f} "
                  f"ret={r_is['total_return']:+7.2%}  |  "
                  f"OOS SR={r_oos['sharpe_ratio']:6.3f} "
                  f"ret={r_oos['total_return']:+7.2%} "
                  f"mdd={r_oos['max_drawdown']:7.2%} n={r_oos['total_trades']} "
                  f"win~{_winrate(r_oos)}")
            TRIALS.append({
                "name": f"J_{vname}", "group": "J",
                "sharpe_ratio": round(float(r_oos["sharpe_ratio"]), 4),
                "total_return": round(float(r_oos["total_return"]), 4),
                "max_drawdown": round(float(r_oos["max_drawdown"]), 4),
                "total_trades": int(r_oos["total_trades"]),
                "IS_sharpe": round(float(r_is["sharpe_ratio"]), 4),
                "OOS_sharpe": round(float(r_oos["sharpe_ratio"]), 4),
                "OOS_return": round(float(r_oos["total_return"]), 4),
                "OOS_mdd": round(float(r_oos["max_drawdown"]), 4),
                "note": f"IS={r_is['sharpe_ratio']:.3f} OOS={r_oos['sharpe_ratio']:.3f}",
            })
        results["groups"]["J_oos"] = gJ

    # ── 汇总 ──────────────────────────────────────────────────────
    results["trials"] = TRIALS
    results["n_trials"] = len(TRIALS)
    ranked = sorted(TRIALS, key=lambda e: -e.get("sharpe_ratio", 0))
    results["top10"] = ranked[:10]
    print("\n" + "=" * 84)
    print(f"共 {len(TRIALS)} 次试验。全窗夏普 Top10：")
    for e in ranked[:10]:
        print(f"  {e['name'][:44]:46s} SR={e['sharpe_ratio']:6.3f} "
              f"ret={e['total_return']:+8.2%} mdd={e['max_drawdown']:7.2%} "
              f"n={e['total_trades']:5d}")
    print("=" * 84)

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, args.out)
    with open(path, "w") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=1, default=str)
    print(f"写入 {path}  ({time.time()-t0:.0f}s)")


# ══════════════════════════════════════════════════════════════════
# 自定义 selector（权重类）
# ══════════════════════════════════════════════════════════════════
class _WeightSelectorBase(L.Selector):
    """按外部函数算权重 → 目标权重。"""

    def __init__(self, cfg: L.SimConfig, close: np.ndarray, lookback: int):
        self.cfg = cfg
        self.close = close
        self.lookback = lookback
        self.rebalance_to_targets = True

    def _weights(self, t: int, keep: np.ndarray) -> Optional[np.ndarray]:
        raise NotImplementedError

    def select(self, order, t, holdings, n_hold):
        keep = order[: self.cfg.topk]
        w = self._weights(t, keep)
        if w is None:
            return keep, {}, order
        total = 1.0 - self.cfg.cash_reserve
        w = np.clip(np.nan_to_num(w, nan=0.0), 0.0, None)
        if w.sum() <= 0:
            w = np.ones(len(keep))
        w = w / w.sum() * total
        return keep, {int(c): float(w[i]) for i, c in enumerate(keep)}, order


class _InvVolSelector(_WeightSelectorBase):
    def _weights(self, t, keep):
        w = M.inverse_vol_weights(self.close, t, keep, self.lookback)
        return w


class _HRPSelector(_WeightSelectorBase):
    def _weights(self, t, keep):
        w = M.hrp_weights(self.close, t, keep, self.lookback)
        if w is None:
            return None
        # HRP 可能给 0 权重 → 至少保留一点点，否则持仓会被清掉
        if w.sum() <= 0:
            return None
        return w


class _MaxDivSelector(_WeightSelectorBase):
    def _weights(self, t, keep):
        return M.maxdiv_weights(self.close, t, keep, self.lookback)


def _mean_t(x: np.ndarray) -> Tuple[float, float]:
    v = x[np.isfinite(x)]
    if len(v) < 5:
        return float("nan"), float("nan")
    se = float(np.std(v, ddof=1) / np.sqrt(len(v)))
    return se, float(v.mean() / se) if se > 0 else float("nan")


def _winrate(res: Dict[str, Any]) -> str:
    return "-"


if __name__ == "__main__":
    main()
