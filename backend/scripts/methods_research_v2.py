#!/usr/bin/env python3
"""第二轮：修正 + 归因 + OOS 验证。

## 第一轮发现的问题（自己抓出来的）
1. **IC 加权有前视**：IC[d] 用 fwd[d] = close[d+h]/close[d]，
   要到 d+h 才已知。原实现只 shift(1) → h≥2 时泄漏 h-1 天未来。
   修正：权重序列用 `ics.shift(h)`，使 t 日权重只用 d ≤ t-h 的 IC。
2. **HRP/maxdiv 权重长度错位**：内部过滤掉无效列后返回的权重与
   `keep` 索引对不上 → IndexError。修正：映射回原始 idx。
3. **逆波动率 lookback 越界**：`_trailing_returns` 少取一行 → 早期
   恒返回 None → 静默退化成等权（F_invvol_20 与基线完全相同就是征兆）。
4. **成本惩罚 λ 尺度**：λ×z(Amihud) 相对信号 std(0.405) 是 30% 的扰动，
   掉 0.19 夏普不是「惩罚无效」的证据，而是「信号本身有小盘暴露」的证据。
   → 需要做**暴露归因**（本轮 A2/A3）。

## 本轮实验组
  A2. 暴露归因：信号/组合对规模、流动性、波动率、行业的暴露
  A3. 规模中性化：把规模/流动性暴露剔掉后，alpha 还在不在
  B2. alpha decay（IC 半衰期）—— 决定持有期与平滑是否合理
  C2. IC 加权（修正前视）
  E2. 组合构建层（修正 HRP/maxdiv）
  F2. 换手惩罚 + 换手率因子（OOS 验证第一轮的 SR 1.49 是否过拟合）
  G2. 多周期合成
  J2. 全部候选 IS/OOS 切分
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

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
TRIALS: List[Dict[str, Any]] = []


def fwd_returns(close: np.ndarray, horizon: int) -> np.ndarray:
    out = np.full_like(close, np.nan)
    if horizon < close.shape[0]:
        out[:-horizon] = close[horizon:] / close[:-horizon] - 1.0
    return out


def record(name, group, res, cfg, note="", extra=None):
    e = {
        "name": name, "group": group,
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
    if extra:
        e.update(extra)
    TRIALS.append(e)
    print(f"  {name[:44]:46s} SR={e['sharpe_ratio']:6.3f} "
          f"ret={e['total_return']:+8.2%} mdd={e['max_drawdown']:7.2%} "
          f"n={e['total_trades']:5d}  {note[:42]}", flush=True)
    return e


def r4(d):
    return {k: (round(float(v), 4) if isinstance(v, (int, float)) else v)
            for k, v in d.items()}


# ══════════════════════════════════════════════════════════════════
# 修正后的 IC 加权（无前视）
# ══════════════════════════════════════════════════════════════════
def icir_weighted_no_lookahead(
    z_factors: Dict[str, np.ndarray],
    fwd: np.ndarray,
    horizon: int,
    window: int = 120,
    min_periods: int = 40,
    use_nw: bool = True,
    shrink: bool = True,
    clip: float = 1.0,
) -> Tuple[np.ndarray, Dict[str, Dict[str, np.ndarray]]]:
    """按滚动 ICIR 加权（**无前视**修正版）。

    关键：IC[d] 的标签 fwd[d] = close[d+h]/close[d] 要到 d+h 才已知。
    所以 t 日可用的 IC 只到 d = t-h → 权重序列必须用 `ics.shift(h)`。
    原实现（只 shift(1)）在 h≥2 时泄漏 h-1 天未来 —— 已修正。
    """
    names = list(z_factors)
    n_d, n_s = z_factors[names[0]].shape
    wmat = np.zeros((n_d, len(names)))
    diag: Dict[str, Dict[str, np.ndarray]] = {}
    for j, name in enumerate(names):
        ics_raw = pd.Series(L.cs_rank_ic(z_factors[name], fwd))
        # 标签成熟日对齐：ics_lagged[t] = ics_raw[t-h]（t 日已实现）
        ics = ics_raw.shift(horizon)
        roll_mean = ics.rolling(window, min_periods=min_periods).mean()
        roll_std = ics.rolling(window, min_periods=min_periods).std()
        icir = (roll_mean / roll_std.replace(0.0, np.nan))
        if use_nw:
            w = M._blockwise_nw_tstat(ics, window=window,
                                      min_periods=min_periods, block=20)
        else:
            w = icir
        w = w.fillna(0.0)
        if shrink:
            t2 = w * w
            w = w * t2 / (t2 + 1.0)
        w = w.clip(-clip, clip)
        wmat[:, j] = w.to_numpy()
        diag[name] = {"ic_mean": roll_mean.to_numpy(),
                      "icir": icir.to_numpy(), "weight": w.to_numpy()}
    total = np.zeros((n_d, n_s))
    wsum = np.zeros((n_d, n_s))
    for j, name in enumerate(names):
        z = z_factors[name]
        w = wmat[:, j][:, None]
        good = np.isfinite(z) & np.isfinite(w)
        total += np.where(good, z * w, 0.0)
        wsum += np.where(good, np.abs(w), 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        score = np.where(wsum > 0, total / np.where(wsum > 0, wsum, 1.0), np.nan)
    return score, diag


# ══════════════════════════════════════════════════════════════════
# 暴露中性化（Barra 式回归中性化）
# ══════════════════════════════════════════════════════════════════
def neutralize_score(
    score: np.ndarray,
    exposures: Dict[str, np.ndarray],
) -> np.ndarray:
    """逐日截面回归：score 对各暴露回归，取残差。

    Barra APT 风格的中性化：把组合对风格/规模/流动性的暴露剔掉。
    实现：每日做 OLS（y = score, X = [1, exposures...]），取残差。
    """
    n_d, n_s = score.shape
    names = list(exposures)
    out = np.full_like(score, np.nan)
    for t in range(n_d):
        y = score[t]
        X = np.column_stack([exposures[k][t] for k in names])
        m = np.isfinite(y) & np.isfinite(X).all(axis=1)
        if int(m.sum()) < 50:
            continue
        A = np.column_stack([X[m], np.ones(int(m.sum()))])
        beta, *_ = np.linalg.lstsq(A, y[m], rcond=None)
        r = y[m] - A @ beta
        out[t, m] = r
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="*",
                    default=["A2", "A3", "B2", "C2", "E2", "F2", "G2", "J2"])
    ap.add_argument("--out", default="methods_research_v2.json")
    ap.add_argument("--topk", type=int, default=60)
    ap.add_argument("--ndrop", type=int, default=5)
    args = ap.parse_args()
    groups = set(args.groups)

    t0 = time.time()
    print("=" * 84)
    print("第二轮 · 修正 + 归因 + OOS 验证（真实盘口径 lag=1）")
    print("=" * 84)

    cache = M.load_cached_mats()
    mats, trad = cache["mats"], cache["trad"]
    CFG = L.SimConfig(topk=args.topk, n_drop=args.ndrop, lag=1,
                      rebalance_days="weekly",
                      max_position_size=(1 - 0.05) / args.topk)
    is_rows, oos_rows = L.split_index(mats["master"], OOS_SPLIT)

    s25, z_eng = L.build_42f_score(mats, panel_aligned=False)
    s42, z_day = L.build_42f_score(
        mats, panel_aligned=True,
        panel_dir=os.path.join(L.QLIB_LIVE, "features", "day"))
    BASE = s42
    print(f"[setup] ready ({time.time()-t0:.0f}s)", flush=True)

    amount = mats["close"] * mats["volume"]
    with np.errstate(divide="ignore", invalid="ignore"):
        log_amt = np.log(np.where(amount > 0, amount, np.nan))
    size_proxy = pd.DataFrame(log_amt).rolling(20, min_periods=5).mean().to_numpy()
    illiq = M.amihud_illiq(mats["close"], mats["volume"], 20)
    with np.errstate(divide="ignore", invalid="ignore"):
        ret = mats["close"][1:] / mats["close"][:-1] - 1.0
        ret = np.vstack([np.full((1, mats["close"].shape[1]), np.nan), ret])
    vol_proxy = pd.DataFrame(ret).rolling(60, min_periods=20).std().to_numpy()

    results: Dict[str, Any] = {
        "meta": {
            "window": f"{mats['master'][0].date()}→{mats['master'][-1].date()}",
            "n_codes": len(mats["codes"]),
            "n_trading_days": len(mats["master"]),
            "oos_split": str(OOS_SPLIT.date()),
            "caliber": {"lag": 1, "enable_tradability": True,
                        "topk": args.topk, "n_drop": args.ndrop,
                        "commission": 0.0003, "slippage": 0.0001,
                        "board_lot": 100, "cash_reserve": 0.05,
                        "rebalance": "weekly(周五)"},
            "engine_baseline_42f": {
                "sharpe_ratio": 0.7991795250472001,
                "total_return": 0.34779248262737783,
                "max_drawdown": -0.16460289642093642,
                "total_trades": 1080,
            },
            "engine_baseline_25f": {
                "sharpe_ratio": 0.8795668740640868,
                "total_return": 0.35505797455182114,
                "max_drawdown": -0.1337121157482497,
                "total_trades": 1190,
            },
            "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        },
        "groups": {},
    }

    # ══════════ A2. 暴露归因 ══════════
    if "A2" in groups:
        print("\n" + "=" * 84)
        print("A2. 暴露归因：信号与组合对 规模/流动性/波动率 的暴露")
        print("=" * 84)
        rows = []
        exposures = {"log_amount": size_proxy, "illiq": illiq,
                     "vol60": vol_proxy}
        for nm, sc in (("42f_day", s42), ("25f", s25)):
            # 信号与暴露的截面相关（IC 风格）
            for ename, ex in exposures.items():
                ic = L.cs_rank_ic(sc, ex)
                rows.append({"score": nm, "exposure": ename,
                             "mean_rank_corr": round(float(np.nanmean(ic)), 4),
                             "std": round(float(np.nanstd(ic)), 4)})
                print(f"  {nm:10s} vs {ename:12s} 平均截面秩相关 "
                      f"{float(np.nanmean(ic)):+.4f} (std {float(np.nanstd(ic)):.4f})")
            # 组合暴露：选出的 top60 的暴露均值 vs 全市场均值
            top_exp = {}
            for ename, ex in exposures.items():
                top_vals, mkt_vals = [], []
                for t in range(0, len(mats["master"]), 5):   # 每 5 天采样一次
                    row = sc[t]
                    valid = np.where(np.isfinite(row))[0]
                    if len(valid) < 60:
                        continue
                    order = valid[np.argsort(-row[valid], kind="mergesort")][:60]
                    ev = ex[t]
                    top_vals.append(np.nanmean(ev[order]))
                    mkt_vals.append(np.nanmean(ev[valid]))
                top_exp[ename] = {
                    "portfolio_mean": round(float(np.nanmean(top_vals)), 5),
                    "market_mean": round(float(np.nanmean(mkt_vals)), 5),
                    "diff": round(float(np.nanmean(top_vals)
                                        - np.nanmean(mkt_vals)), 5),
                }
                print(f"    └─ 组合 {ename}: 组合均值 "
                      f"{top_exp[ename]['portfolio_mean']:.4f} vs "
                      f"全市场 {top_exp[ename]['market_mean']:.4f} "
                      f"(差 {top_exp[ename]['diff']:+.4f})")
            rows.append({"score": nm, "portfolio_exposure": top_exp})
        results["groups"]["A2_exposure"] = rows

    # ══════════ A3. 规模/流动性中性化 ══════════
    if "A3" in groups:
        print("\n" + "=" * 84)
        print("A3. 中性化：剔除规模/流动性/波动暴露后 alpha 还在不在")
        print("=" * 84)
        gA3 = []
        for nm, sc in (("42f_day", s42), ("25f", s25)):
            res = L.simulate(sc, mats, trad, CFG)
            e = record(f"A3_{nm}_raw", "A3", res, CFG, note="未中性化")
            gA3.append(e)
            for label, ex in (
                ("size", {"log_amount": size_proxy}),
                ("size+illiq", {"log_amount": size_proxy, "illiq": illiq}),
                ("size+illiq+vol", {"log_amount": size_proxy, "illiq": illiq,
                                     "vol60": vol_proxy}),
            ):
                sn = neutralize_score(sc, ex)
                res = L.simulate(sn, mats, trad, CFG)
                e = record(f"A3_{nm}_neut_{label}", "A3", res, CFG,
                           note=f"中性化 {label}")
                gA3.append(e)
        results["groups"]["A3_neutralize"] = gA3

    # ══════════ B2. alpha decay ══════════
    if "B2" in groups:
        print("\n" + "=" * 84)
        print("B2. alpha decay：IC 随持有期衰减 → 半衰期 → 持有期选择")
        print("=" * 84)
        rows = []
        for nm, sc in (("42f_day", s42), ("25f", s25)):
            prev = None
            for hz in (1, 2, 3, 5, 8, 10, 15, 20, 30, 40):
                fwd = fwd_returns(mats["close"], hz)
                ic = L.cs_rank_ic(sc, fwd)[is_rows]
                m = float(np.nanmean(ic))
                s = float(np.nanstd(ic))
                hl = None
                if prev and m > 0:
                    # 拟合 IC(h) = IC0 * exp(-lambda h) → 半衰期 ln2/lambda
                    lam = -np.log(max(m / prev, 1e-9)) if prev > 0 else np.nan
                    hl = float(np.log(2) / lam) if lam and lam > 0 else None
                rows.append({"score": nm, "horizon": hz, "ic_mean": round(m, 5),
                             "ic_std": round(s, 5),
                             "icir": round(m / s, 4) if s > 0 else None,
                             "implied_halflife_days": hl})
                print(f"  {nm:10s} h={hz:3d}  IC(IS)={m:+.5f}  "
                      f"ICIR={m/s if s > 0 else 0:+.3f}  "
                      f"隐含半衰期={hl if hl else float('nan'):.1f} 日")
                prev = m
        results["groups"]["B2_alpha_decay"] = rows

    # ══════════ C2. IC 加权（修正前视） ══════════
    if "C2" in groups:
        print("\n" + "=" * 84)
        print("C2. IC 加权（修正前视后）")
        print("=" * 84)
        gC2 = []
        for hz in (1, 5):
            fwd = fwd_returns(mats["close"], hz)
            for use_nw in (False, True):
                for shrink in (False, True):
                    for win in (60, 120, 250):
                        sc, _ = icir_weighted_no_lookahead(
                            z_day, fwd, horizon=hz, window=win,
                            min_periods=max(20, win // 3),
                            use_nw=use_nw, shrink=shrink)
                        res = L.simulate(sc, mats, trad, CFG)
                        e = record(
                            f"C2_icir_h{hz}_nw{int(use_nw)}_sh{int(shrink)}_w{win}",
                            "C2", res, CFG,
                            note=f"h={hz} NW={use_nw} 收缩={shrink} win={win}")
                        gC2.append(e)
        results["groups"]["C2_ic_weighting"] = gC2

    # ══════════ E2. 组合构建层（修正） ══════════
    if "E2" in groups:
        print("\n" + "=" * 84)
        print("E2. 组合构建层权重（修正 HRP / maxdiv / invvol）")
        print("=" * 84)
        gE2 = []

        class WSel(L.Selector):
            def __init__(self, cfg, close, lookback, kind):
                self.cfg, self.close, self.lookback, self.kind = cfg, close, lookback, kind
                self.rebalance_to_targets = True

            def select(self, order, t, holdings, n_hold):
                keep = order[: self.cfg.topk]
                if self.kind == "invvol":
                    w = M.inverse_vol_weights(self.close, t, keep, self.lookback)
                elif self.kind == "hrp":
                    w = M.hrp_weights(self.close, t, keep, self.lookback)
                else:
                    w = M.maxdiv_weights(self.close, t, keep, self.lookback)
                if w is None or len(w) != len(keep):
                    return keep, {}, order
                w = np.clip(np.nan_to_num(w, nan=0.0), 0.0, None)
                if w.sum() <= 0:
                    return keep, {}, order
                w = w / w.sum() * (1 - self.cfg.cash_reserve)
                return keep, {int(c): float(w[i]) for i, c in enumerate(keep)}, order

        for kind in ("invvol", "hrp", "maxdiv"):
            for lb in (20, 60, 120, 250):
                sel = WSel(CFG, mats["close"], lb, kind)
                res = L.simulate(BASE, mats, trad, CFG, selector=sel)
                e = record(f"E2_{kind}_{lb}", "E2", res, CFG,
                           note=f"{kind} lookback={lb}")
                gE2.append(e)
        results["groups"]["E2_weights"] = gE2

    # ══════════ F2. 换手惩罚 + 换手率因子 ══════════
    if "F2" in groups:
        print("\n" + "=" * 84)
        print("F2. 换手惩罚 / 换手率因子（OOS 验证第一轮 SR 1.49）")
        print("=" * 84)
        gF2 = []
        # F2a 换手率因子（第一轮 I2_turnover_rev：负 log 成交额 20 日均值）
        to_mean = pd.DataFrame(log_amt).rolling(20, min_periods=5).mean().to_numpy()
        z_to = L.cs_zscore(-to_mean, 30)
        for k in (20, 60):
            cfg2 = L.SimConfig(topk=k, n_drop=max(2, k // 12), lag=1,
                               rebalance_days="weekly",
                               max_position_size=(1 - 0.05) / k)
            res = L.simulate(z_to, mats, trad, cfg2)
            e = record(f"F2a_turnoverrev_k{k}", "F2", res, cfg2,
                       note="低换手率因子单独")
            gF2.append(e)
        # F2b 42f + 换手率因子
        for w in (0.25, 0.5, 1.0, 2.0):
            sc = M.blend_scores({"base": BASE, "to": z_to},
                                 {"base": 1.0, "to": w})
            for k in (20, 60):
                cfg2 = L.SimConfig(topk=k, n_drop=max(2, k // 12), lag=1,
                                   rebalance_days="weekly",
                                   max_position_size=(1 - 0.05) / k)
                res = L.simulate(sc, mats, trad, cfg2)
                e = record(f"F2b_base+to_w{w}_k{k}", "F2", res, cfg2,
                           note=f"42f + {w}×换手率反转")
                gF2.append(e)
        # F2c 成本惩罚（对非流动性的温和惩罚，尺度对齐）
        for lam in (0.01, 0.02, 0.05):
            sc = BASE - lam * L.cs_zscore(illiq, 30) * np.nanstd(BASE)
            res = L.simulate(sc, mats, trad, CFG)
            e = record(f"F2c_cost_lam{lam}", "F2", res, CFG,
                       note=f"score - {lam}×std×z(Amihud)")
            gF2.append(e)
        results["groups"]["F2_turnover"] = gF2

    # ══════════ G2. 多周期合成 ══════════
    if "G2" in groups:
        print("\n" + "=" * 84)
        print("G2. 多周期信号合成（快/慢）")
        print("=" * 84)
        gG2 = []
        for slow_hl in (5, 10, 20, 40):
            slow = M.ewma_smooth(BASE, halflife=slow_hl)
            for w in (0.25, 0.5, 0.75):
                sc = M.blend_scores({"fast": BASE, "slow": slow},
                                    {"fast": 1 - w, "slow": w})
                res = L.simulate(sc, mats, trad, CFG)
                e = record(f"G2_blend_ewma{slow_hl}_w{w}", "G2", res, CFG,
                           note=f"日频 + {w}×EWMA({slow_hl})")
                gG2.append(e)
        # G2b 不同持有期的 IC 加权版本（持有期匹配）
        fwd5 = fwd_returns(mats["close"], 5)
        sc_ic5, _ = icir_weighted_no_lookahead(z_day, fwd5, horizon=5,
                                                window=120, min_periods=40,
                                                use_nw=True, shrink=True)
        for w in (0.5, 1.0):
            sc = M.blend_scores({"base": BASE, "ic5": sc_ic5},
                                {"base": 1 - w, "ic5": w})
            res = L.simulate(sc, mats, trad, CFG)
            e = record(f"G2b_base+ic5_w{w}", "G2", res, CFG,
                       note=f"42f + {w}×IC加权(fwd5)")
            gG2.append(e)
        results["groups"]["G2_multihorizon"] = gG2

    # ══════════ J2. OOS 切分 ══════════
    if "J2" in groups:
        print("\n" + "=" * 84)
        print("J2. OOS 切分（IS 2023-05→2025-05 选参 / OOS 2025-05→2026-05 验证）")
        print("=" * 84)
        fwd5 = fwd_returns(mats["close"], 5)
        sc_ic5, _ = icir_weighted_no_lookahead(z_day, fwd5, horizon=5,
                                                window=120, min_periods=40,
                                                use_nw=True, shrink=True)
        to_mean = pd.DataFrame(log_amt).rolling(20, min_periods=5).mean().to_numpy()
        z_to = L.cs_zscore(-to_mean, 30)
        sn_size = neutralize_score(BASE, {"log_amount": size_proxy})
        sn_sizeliq = neutralize_score(BASE, {"log_amount": size_proxy,
                                             "illiq": illiq})
        variants: Dict[str, np.ndarray] = {
            "base_42f": BASE,
            "base_25f": s25,
            "ewma5": M.ewma_smooth(BASE, halflife=5),
            "ewma20": M.ewma_smooth(BASE, halflife=20),
            "ic5_nw": sc_ic5,
            "ic5_ewma5": M.ewma_smooth(sc_ic5, halflife=5),
            "base+to0.5": M.blend_scores({"base": BASE, "to": z_to},
                                          {"base": 1.0, "to": 0.5}),
            "base+to1.0": M.blend_scores({"base": BASE, "to": z_to},
                                          {"base": 1.0, "to": 1.0}),
            "neut_size": sn_size,
            "neut_sizeliq": sn_sizeliq,
            "base+to0.5+ic5": M.blend_scores(
                {"base": BASE, "to": z_to, "ic5": sc_ic5},
                {"base": 1.0, "to": 0.5, "ic5": 0.5}),
        }
        gJ2 = []
        for vname, sc in variants.items():
            sub_is = L.slice_mats(mats, is_rows)
            sub_oos = L.slice_mats(mats, oos_rows)
            tr_is = {k: v[is_rows] for k, v in trad.items()}
            tr_oos = {k: v[oos_rows] for k, v in trad.items()}
            r_is = L.simulate(sc[is_rows], sub_is, tr_is, CFG)
            r_oos = L.simulate(sc[oos_rows], sub_oos, tr_oos, CFG)
            e = {"name": f"J2_{vname}", "group": "J2",
                 "IS": r4(r_is), "OOS": r4(r_oos)}
            gJ2.append(e)
            print(f"  {vname:18s} IS SR={r_is['sharpe_ratio']:6.3f} "
                  f"ret={r_is['total_return']:+7.2%} mdd={r_is['max_drawdown']:7.2%} "
                  f"| OOS SR={r_oos['sharpe_ratio']:6.3f} "
                  f"ret={r_oos['total_return']:+7.2%} "
                  f"mdd={r_oos['max_drawdown']:7.2%} n={r_oos['total_trades']}")
            TRIALS.append({
                "name": f"J2_{vname}", "group": "J2",
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
        results["groups"]["J2_oos"] = gJ2

    results["trials"] = TRIALS
    results["n_trials"] = len(TRIALS)
    ranked = sorted(TRIALS, key=lambda e: -e.get("sharpe_ratio", 0))
    results["top10"] = ranked[:10]
    print("\n" + "=" * 84)
    print(f"共 {len(TRIALS)} 次试验。Top10：")
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


if __name__ == "__main__":
    main()
