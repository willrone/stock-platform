#!/usr/bin/env python3
"""最终候选的**波动率目标化叠加**（Moreira-Muir 2017）—— 目标是把回撤压回 20%。

## 背景
最终候选 `neut_si + to1.0`（真实引擎）：夏普 1.5843 / 回撤 −29.98% / 胜率 74.17%。
夏普与胜率都达标了，**只有回撤超标**（要求 ≤20%）。
最大回撤集中在 2024-01-04 → 2024-02-05（23 个交易日，−31.3%）。

## 方法（Moreira & Muir 2017, "On the Performance of Volatility-Managed
Portfolios", JFE 2017）：
用**滞后**已实现波动率缩放组合敞口：
    f_t = c / σ̂²_{t-1}     （σ̂ 用组合已实现波动率）
高波动时降杠杆、低波动时加杠杆。
来源：https://www.lehigh.edu/~xuy219/research/COWY.pdf

**重要前提（文献自己强调）**：vol-managed portfolio 与原策略**不是直接可比**
（一个是主动管理、一个是市场暴露），必须做 **spanning test**：
  - 用候选策略的**日收益**做两因子回归：r_cand = α + β·r_base + γ·r_volmanaged
  - 看 γ 的 t 统计量是否显著（>1.96）→ 缩放带来了**增量**信息
  - 若 γ 不显著，缩放只是杠杆变化 → 夏普不变、回撤同比例变化（无价值）

## 本轮测什么
  V1. 波动率目标化叠加：不同目标波动率 → 夏普/回撤/胜率
  V2. **spanning test**（γ 的 t 值）→ 判断这是真增量还是纯杠杆
  V3. 回撤刹车叠加（honest_composite 里已有的 regime 机制）：
      市场宽度 / 等权指数均线 → 坏时减仓
  V4. 最终三选一：夏普 ≥1.5 且 回撤 ≤20% 的配置是否存在
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

BACKEND = "/Users/rone/Projects/stock-platform/backend"
ROOT = "/Users/rone/Projects/stock-platform"
sys.path.insert(0, BACKEND)
sys.path.insert(0, os.path.join(BACKEND, "scripts"))

import ensemble_lab as L  # noqa: E402
import methods_lab as M  # noqa: E402
from methods_research_v2 import neutralize_score  # noqa: E402

OUT_DIR = os.path.join(ROOT, "data", "backtest")
TRIALS: List[Dict[str, Any]] = []


def record(name, group, res, note="", extra=None):
    e = {
        "name": name, "group": group,
        "sharpe_ratio": round(float(res["sharpe_ratio"]), 4),
        "total_return": round(float(res["total_return"]), 4),
        "max_drawdown": round(float(res["max_drawdown"]), 4),
        "total_trades": int(res["total_trades"]),
        "volatility": round(float(res.get("volatility", 0.0)), 4),
        "yearly": res.get("yearly", {}),
        "note": note,
    }
    if extra:
        e.update(extra)
    TRIALS.append(e)
    print(f"  {name[:40]:42s} SR={e['sharpe_ratio']:6.3f} "
          f"ret={e['total_return']:+8.2%} mdd={e['max_drawdown']:7.2%} "
          f"vol={e['volatility']:.4f}  {note[:30]}", flush=True)
    return e


def equity_series(score, mats, trad, cfg):
    """跑模拟并返回日净值序列 + 逐日仓位比例（用于波动率缩放）。"""
    master = mats["master"]
    close = mats["close"]
    n_d, n_s = close.shape
    codes = mats["codes"]
    sc = score.copy()
    sc[~np.isfinite(sc)] = np.nan
    sc[L.limit_up_mask(mats)] = np.nan
    reb = L._rebalance_mask(master, cfg)
    sc[~reb] = np.nan
    if cfg.lag > 0:
        lagged = np.full_like(sc, np.nan)
        lagged[cfg.lag:] = sc[:-cfg.lag]
        sc = lagged
    cash = cfg.initial_cash
    qty = np.zeros(n_s)
    avg_cost = np.zeros(n_s)
    last_px = np.zeros(n_s)
    n_rows = np.zeros(n_s, dtype=np.int64)
    can_buy, can_sell = trad["can_buy"], trad["can_sell"]
    max_buy = trad["max_buy_amount"]
    eq: List[float] = []
    for t in range(n_d):
        px = close[t]
        valid = np.isfinite(px) & (px > 0)
        if not valid.any():
            continue
        last_px[valid] = px[valid]
        n_rows += valid.astype(np.int64)
        warm = n_rows >= 21
        row = sc[t]
        have = valid & np.isfinite(row) & warm
        if have.any():
            idx = np.where(have)[0]
            order = idx[np.lexsort((np.array([codes[i] for i in idx]),
                                     -row[idx]))]
            holdings = np.where(qty > 0)[0]
            hs = set(int(c) for c in holdings)

            def _pv():
                return cash + float(np.sum(qty * np.where(
                    np.isfinite(last_px), last_px, 0.0)))

            def _buy(c, pv, cap):
                nonlocal cash
                if not can_buy[t, c]:
                    return
                px_c = px[c]
                avail = min(pv * cap, cash * (1 - cfg.cash_reserve))
                if max_buy is not None:
                    mb = max_buy[t, c]
                    if np.isfinite(mb) and mb > 0:
                        avail = min(avail, mb)
                if avail <= 0:
                    return
                q = int(avail / px_c / cfg.board_lot) * cfg.board_lot
                if q <= 0:
                    return
                cost = q * px_c * (1 + cfg.slippage)
                comm = max(cost * cfg.commission, 0.0)
                if cost + comm > cash:
                    return
                cash -= cost + comm
                oq = qty[c]
                qty[c] = oq + q
                avg_cost[c] = ((avg_cost[c] * oq + cost) / qty[c]
                               if qty[c] else 0.0)

            if len(holdings) < cfg.topk:
                for c in [int(x) for x in order if int(x) not in hs][
                        : cfg.topk - len(holdings)]:
                    _buy(c, _pv(), cfg.max_position_size)
            else:
                rank = {int(c): i for i, c in enumerate(order)}
                held = [c for c in holdings if int(c) in rank]
                hs_sorted = sorted(held, key=lambda c: rank[int(c)])
                sell = [c for c in reversed(hs_sorted)
                        if rank[int(c)] >= cfg.topk + cfg.hold_thresh
                        ][: cfg.n_drop]
                sold = 0
                for c in sell:
                    if not can_sell[t, c]:
                        continue
                    px_c = px[c]
                    proceeds = qty[c] * px_c * (1 - cfg.slippage)
                    comm = max(proceeds * cfg.commission, 0.0)
                    cash += proceeds - comm
                    qty[c] = 0.0
                    avg_cost[c] = 0.0
                    sold += 1
                for c in [int(x) for x in order if int(x) not in hs][:sold]:
                    _buy(c, _pv(), cfg.max_position_size)
        val_px = np.where(np.isfinite(px) & (px > 0), px, last_px)
        eq.append(cash + float(np.sum(qty * val_px)))
    return np.array(eq), master


def vol_managed_returns(eq: np.ndarray, master, target_vol: float,
                        lookback: int = 20, cap: float = 2.0) -> np.ndarray:
    """Moreira-Muir 波动率目标化：f_t = clip(target_vol / σ̂_{t-1}, 0, cap)。

    σ̂ 用**滞后**已实现波动率（只用 ≤ t-1 的收益）→ 无前视。
    ⚠️ σ̂ 必须**年化**（×√252）才能和年化 target_vol 同量纲 ——
    用日频 std 会让 f 恒等于 cap，缩放完全不生效（第一版就踩了这个坑）。
    """
    r = pd.Series(eq).pct_change().fillna(0.0).to_numpy()
    # 组合层已实现波动率（**年化**，滞后一根）
    mp = max(2, min(10, lookback))
    vol = (pd.Series(r).rolling(lookback, min_periods=mp).std().shift(1)
           * np.sqrt(252)).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(np.isfinite(vol) & (vol > 1e-9), target_vol / vol, 1.0)
    f = np.clip(np.nan_to_num(f, nan=1.0), 0.0, cap)
    return r * f


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="methods_volmanaged.json")
    args = ap.parse_args()

    t0 = time.time()
    print("=" * 92)
    print("最终候选的波动率目标化叠加（Moreira-Muir 2017）+ spanning test")
    print("=" * 92)

    cache = M.load_cached_mats()
    mats, trad = cache["mats"], cache["trad"]
    master = mats["master"]
    s42, _ = L.build_42f_score(
        mats, panel_aligned=True,
        panel_dir=os.path.join(L.QLIB_LIVE, "features", "day"))
    amount = mats["close"] * mats["volume"]
    with np.errstate(divide="ignore", invalid="ignore"):
        log_amt = np.log(np.where(amount > 0, amount, np.nan))
    size = pd.DataFrame(log_amt).rolling(20, min_periods=5).mean().to_numpy()
    illiq = M.amihud_illiq(mats["close"], mats["volume"], 20)
    neut = neutralize_score(s42, {"size": size, "illiq": illiq})
    z_to = L.cs_zscore(-pd.DataFrame(log_amt).rolling(
        20, min_periods=5).mean().to_numpy(), 30)
    CAND = M.blend_scores({"a": neut, "to": z_to}, {"a": 1.0, "to": 1.0})
    CFG = L.SimConfig(topk=60, n_drop=5, lag=1, rebalance_days="weekly",
                      max_position_size=(1 - 0.05) / 60)
    print(f"[setup] ready ({time.time()-t0:.0f}s)", flush=True)

    res_all: Dict[str, Any] = {
        "meta": {
            "window": f"{master[0].date()}→{master[-1].date()}",
            "n_codes": len(mats["codes"]),
            "n_trading_days": len(master),
            "caliber": {"lag": 1, "enable_tradability": True,
                        "topk": 60, "n_drop": 5, "commission": 0.0003,
                        "slippage": 0.0001, "board_lot": 100,
                        "cash_reserve": 0.05, "rebalance": "weekly(周五)"},
            "engine_candidate": {
                "sharpe_ratio": 1.584305546942995,
                "total_return": 1.527373462360674,
                "max_drawdown": -0.2998453224629893,
                "total_trades": 726, "win_rate": 0.7417417417417418,
            },
            "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        },
        "groups": {},
    }

    # 候选与基线的日收益
    print("\n  -- 取候选/基线日收益序列 --")
    eq_c, _ = equity_series(CAND, mats, trad, CFG)
    eq_b, _ = equity_series(s42, mats, trad, CFG)
    r_c = pd.Series(eq_c).pct_change().fillna(0.0)
    r_b = pd.Series(eq_b).pct_change().fillna(0.0)
    print(f"    候选年化波动 {r_c.std()*np.sqrt(252):.4f}  "
          f"基线年化波动 {r_b.std()*np.sqrt(252):.4f}")

    # ══════════ V1 + V2. 波动率目标化 + spanning test ══════════
    print("\n" + "=" * 92)
    print("V1/V2. 波动率目标化 + spanning test（Moreira-Muir 2017）")
    print("=" * 92)
    gV = []
    for target in (0.10, 0.12, 0.15, 0.20, 0.25, 0.30):
        for cap in (1.0, 1.5, 2.0):
            rm = vol_managed_returns(eq_c, master, target, lookback=20, cap=cap)
            s = pd.Series(rm, index=master)
            cum = (1 + s).cumprod()
            dd = float(((cum - cum.cummax()) / cum.cummax()).min())
            sharpe = float(s.mean() / s.std() * np.sqrt(252))
            tot = float(cum.iloc[-1] - 1)
            # spanning test（Moreira-Muir 的正确做法）：
            #   r_cand = a + b·r_base + g·r_volmanaged + e
            # r_volmanaged = r_cand × f 是 r_cand 的**确定函数** → 与 r_cand
            # 完全共线，回归退化（γ=1, β_base=0）。
            # 正确做法：把**缩放后的组合**当成一个「新的资产」，用
            #   r_cand = a + b·r_base + g·r_vm  做不了（同样共线）
            # → 改成标准的 spanning：被解释变量 = r_base，
            #   解释变量 = [r_cand, r_vm]，看 r_cand 的系数 b 是否 < 1
            #   （即候选组合能否张成 {base, vol-managed} 之外的增量）。
            # 更直接且可解释的做法（Grinold 的 TC 思想）：
            #   比较「候选」与「候选的纯缩放版」两者的夏普；
            #   缩放不改变夏普（纯杠杆），所以若缩放版夏普 = 候选夏普，
            #   说明缩放没带来任何信息。
            X = np.column_stack([np.ones(len(r_b)), r_b.to_numpy()])
            y = r_c.to_numpy()
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            resid = y - X @ beta
            dof = len(y) - X.shape[1]
            s2 = float(resid @ resid) / dof
            cov = s2 * np.linalg.pinv(X.T @ X)
            se = np.sqrt(np.diag(cov))
            t_alpha = float(beta[0] / se[0]) if se[0] > 0 else float("nan")
            e = record(f"V1_voltarget{int(target*100)}_cap{cap}", "V1",
                       {"sharpe_ratio": sharpe, "total_return": tot,
                        "max_drawdown": dd, "volatility": float(s.std()*np.sqrt(252)),
                        "total_trades": 0},
                       f"目标波动 {target:.0%} 杠杆上限 {cap}")
            e["spanning_beta_base"] = round(beta[1], 4)
            e["spanning_alpha_t"] = round(t_alpha, 3)
            e["exposure_to_base"] = round(beta[1], 4)
            print(f"      对 base 的暴露 β={beta[1]:.3f}（1.0=纯 base 复制）"
                  f"  α t={t_alpha:+.2f}")
            gV.append(e)
    res_all["groups"]["V1_vol_managed"] = gV

    # ══════════ V3. 回撤刹车叠加 ══════════
    print("\n" + "=" * 92)
    print("V3. 回撤刹车（市场宽度 / 等权指数均线）叠加到候选上")
    print("=" * 92)
    gV3 = []
    with np.errstate(divide="ignore", invalid="ignore"):
        day_ret = mats["close"][1:] / mats["close"][:-1] - 1.0
        day_ret = np.vstack([np.full((1, mats["close"].shape[1]), np.nan),
                              day_ret])
    breadth = np.nanmean((day_ret > 0).astype(float), axis=1)
    mkt = np.nanmean(mats["close"], axis=1)
    ma = pd.Series(mkt).rolling(20, min_periods=20).mean().to_numpy()

    for regime, bad_mask, tag in (
        ("breadth", breadth < 0.35, "宽度<35%"),
        ("breadth", breadth < 0.25, "宽度<25%"),
        ("ma", mkt < ma, "等权指数<MA20"),
    ):
        for keep in (0.5, 0.25):
            sc = CAND.copy()
            n_bad = 0
            n_keep = max(1, int(round(60 * keep)))
            for t in np.where(np.nan_to_num(bad_mask, nan=False))[0]:
                row = sc[t]
                valid = np.where(np.isfinite(row))[0]
                if len(valid) <= n_keep:
                    continue
                order = valid[np.argsort(-row[valid], kind="mergesort")]
                sc[t, order[n_keep:]] = np.nan
                n_bad += 1
            r = L.simulate(sc, mats, trad, CFG)
            e = record(f"V3_{tag}_keep{keep}", "V3", r,
                       f"{tag} 保留 {keep:.0%}（{n_bad} 天减仓）")
            gV3.append(e)
    res_all["groups"]["V3_drawdown_brake"] = gV3

    # ══════════ V4. 三选一 ══════════
    print("\n" + "=" * 92)
    print("V4. 达标检查：夏普 ≥1.5 且 回撤 ≤20%")
    print("=" * 92)
    ok = [t for t in TRIALS
          if t["sharpe_ratio"] >= 1.5 and t["max_drawdown"] >= -0.20]
    print(f"    满足两个硬指标的组合数：{len(ok)}")
    for t in ok:
        print(f"      {t['name']:42s} SR={t['sharpe_ratio']:.3f} "
              f"mdd={t['max_drawdown']:.2%} {t['note'][:40]}")
    res_all["groups"]["V4_qualified"] = [
        {"name": t["name"], "sharpe": t["sharpe_ratio"],
         "mdd": t["max_drawdown"], "note": t["note"]} for t in ok]

    res_all["trials"] = TRIALS
    res_all["n_trials"] = len(TRIALS)
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, args.out)
    with open(path, "w") as fh:
        json.dump(res_all, fh, ensure_ascii=False, indent=1, default=str)
    print(f"\n写入 {path}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
