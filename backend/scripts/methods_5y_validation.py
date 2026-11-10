#!/usr/bin/env python3
"""最终候选的 **5 年窗口复核**（2021-05-11 → 2026-05-08）。

## 为什么必须做
3 年窗口 + ≈330 次试验 + 分段里前 3 段为负 →
**过拟合风险太高**。`data/qlib_data/calendars/day.txt` 覆盖
2021-05-11 → 2026-05-11（1305 个交易日），够做一次独立复核。

## 复核什么
  Y1. 5 年全窗：基线 vs 中性化 vs 最终候选（模拟器）
  Y2. 5 年**逐年级**表现 → 看超额收益是不是只集中在 2025
      （真实引擎 3 年结果：2024 +12.37% / **2025 +67.50%** / 2026 +6.97%）
  Y3. 5 年**分段一致性**（10 段）→ 与 3 年的 8 段对照
  Y4. 严格 IS/OOS（5 年切 2021-05→2024-05 选参 / 2024-05→2026-05 验证）
  Y5. 换手因子的**规模暴露**在 5 年里的稳定性
      （这是它最大的风险点，必须看全周期）
  Y6. 波动率目标化在 5 年里的表现（回撤能否压到 20% 内）

## 口径
模拟器口径与 `ensemble_lab` 一致：lag=1、涨停掩码、可成交性、
commission 0.03% + slippage 0.01%（单边）、整手 100 股、现金预留 5%、
周频（周五）调仓、TopK=60 / n_drop=5。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(BACKEND)
sys.path.insert(0, BACKEND)
sys.path.insert(0, os.path.join(BACKEND, "scripts"))

import ensemble_lab as L  # noqa: E402
import methods_lab as M  # noqa: E402
from methods_research_v2 import neutralize_score  # noqa: E402
from methods_volmanaged import equity_series, vol_managed_returns  # noqa: E402

OUT_DIR = os.path.join(ROOT, "data", "backtest")
START_5Y = datetime(2021, 5, 11)
END_5Y = datetime(2026, 5, 8)
OOS_SPLIT_5Y = pd.Timestamp("2024-05-08")


def metrics_from_returns(r: np.ndarray, master) -> Dict[str, Any]:
    s = pd.Series(r, index=master)
    cum = (1 + s).cumprod()
    dd = float(((cum - cum.cummax()) / cum.cummax()).min())
    sd = float(s.std())
    return {
        "sharpe_ratio": float(s.mean() / sd * np.sqrt(252)) if sd > 0 else 0.0,
        "total_return": float(cum.iloc[-1] - 1.0),
        "max_drawdown": dd,
        "volatility": sd * np.sqrt(252),
    }


def yearly_from_returns(r: np.ndarray, master) -> Dict[str, float]:
    s = pd.Series(r, index=master)
    out: Dict[str, float] = {}
    for y, g in s.groupby(s.index.year):
        out[str(y)] = float((1 + g).prod() - 1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="methods_5y_validation.json")
    args = ap.parse_args()

    t0 = time.time()
    print("=" * 92)
    print(f"5 年窗口复核 {START_5Y.date()} → {END_5Y.date()}")
    print("=" * 92)

    # ── 加载 5 年矩阵 ──
    cache_dir = os.path.join(ROOT, "data", "backtest", "cache")
    os.makedirs(cache_dir, exist_ok=True)
    tag = f"mats_{START_5Y.date()}_{END_5Y.date()}"
    path = os.path.join(cache_dir, tag + ".npz")
    codes = L.universe_from_instruments()
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        mats = {
            "codes": list(z["codes"]),
            "master": pd.DatetimeIndex(z["master"]),
            "close": z["close"], "high": z["high"], "low": z["low"],
            "volume": z["volume"], "stock_data": None,
        }
        trad = {"can_buy": z["can_buy"], "can_sell": z["can_sell"],
                "max_buy_amount": z["max_buy_amount"]}
        print(f"[mats] cache hit {path}", flush=True)
    else:
        print(f"[mats] loading 5y universe {len(codes)}", flush=True)
        mats = L.load_matrices(codes, start=START_5Y, end=END_5Y)
        print("[mats] building tradability ...", flush=True)
        trad = L.build_tradability_mats(mats)
        np.savez_compressed(
            path, codes=np.array(mats["codes"], dtype=object),
            master=mats["master"].to_numpy(),
            close=mats["close"], high=mats["high"], low=mats["low"],
            volume=mats["volume"],
            can_buy=trad["can_buy"], can_sell=trad["can_sell"],
            max_buy_amount=trad["max_buy_amount"])
        mats = dict(mats)
        mats["stock_data"] = None
        print(f"[mats] cached -> {path}", flush=True)

    master = mats["master"]
    print(f"[setup] {master[0].date()} → {master[-1].date()} "
          f"({len(master)} 交易日, {len(mats['codes'])} 只)", flush=True)

    # ── 因子 ──
    s42, _ = L.build_42f_score(
        mats, panel_aligned=True,
        panel_dir=os.path.join(L.QLIB_LIVE, "features", "day"))
    amount = mats["close"] * mats["volume"]
    with np.errstate(divide="ignore", invalid="ignore"):
        log_amt = np.log(np.where(amount > 0, amount, np.nan))
    amt20 = pd.DataFrame(log_amt).rolling(20, min_periods=5).mean().to_numpy()
    size = amt20
    illiq = M.amihud_illiq(mats["close"], mats["volume"], 20)
    neut = neutralize_score(s42, {"size": size, "illiq": illiq})
    z_to = L.cs_zscore(-amt20, 30)
    CAND = M.blend_scores({"a": neut, "to": z_to}, {"a": 1.0, "to": 1.0})
    print(f"[setup] factors ready ({time.time()-t0:.0f}s)", flush=True)

    CFG = L.SimConfig(topk=60, n_drop=5, lag=1, rebalance_days="weekly",
                      max_position_size=(1 - 0.05) / 60)
    res_all: Dict[str, Any] = {
        "meta": {
            "window": f"{master[0].date()}→{master[-1].date()}",
            "n_codes": len(mats["codes"]),
            "n_trading_days": len(master),
            "oos_split": str(OOS_SPLIT_5Y.date()),
            "caliber": {"lag": 1, "enable_tradability": True, "topk": 60,
                        "n_drop": 5, "commission": 0.0003, "slippage": 0.0001,
                        "board_lot": 100, "cash_reserve": 0.05,
                        "rebalance": "weekly(周五)"},
            "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        },
        "groups": {},
    }

    # ── Y1/Y2. 全窗 + 逐年 ──
    print("\n" + "=" * 92)
    print("Y1/Y2. 5 年全窗 + 逐年")
    print("=" * 92)
    variants = {
        "base_42f": s42,
        "neut_si": neut,
        "cand_neut_to1.0": CAND,
    }
    gY = []
    rets: Dict[str, np.ndarray] = {}
    for nm, sc in variants.items():
        r = L.simulate(sc, mats, trad, CFG)
        eq, _ = equity_series(sc, mats, trad, CFG)
        rr = pd.Series(eq).pct_change().fillna(0.0).to_numpy()
        rets[nm] = rr
        yr = yearly_from_returns(rr, master)
        e = {"name": nm, "sharpe_ratio": round(r["sharpe_ratio"], 4),
             "total_return": round(r["total_return"], 4),
             "max_drawdown": round(r["max_drawdown"], 4),
             "total_trades": int(r["total_trades"]),
             "yearly": {k: round(v, 4) for k, v in yr.items()}}
        gY.append(e)
        print(f"  {nm:20s} SR={e['sharpe_ratio']:6.3f} ret={e['total_return']:+8.2%} "
              f"mdd={e['max_drawdown']:7.2%} n={e['total_trades']:5d}")
        print(f"      逐年: {e['yearly']}")
    res_all["groups"]["Y1_Y2_full_window"] = gY

    # ── Y3. 分段一致性（10 段）──
    print("\n" + "=" * 92)
    print("Y3. 分段一致性（5 年切 10 段）")
    print("=" * 92)
    n_seg = 10
    seg_len = len(master) // n_seg
    segs = []
    for i in range(n_seg):
        lo = i * seg_len
        hi = (i + 1) * seg_len if i < n_seg - 1 else len(master)
        idx = np.arange(lo, hi)
        sub = L.slice_mats(mats, idx)
        tr = {k: v[idx] for k, v in trad.items()}
        r0 = L.simulate(s42[idx], sub, tr, CFG)
        r1 = L.simulate(CAND[idx], sub, tr, CFG)
        d = r1["sharpe_ratio"] - r0["sharpe_ratio"]
        print(f"    段{i+1:2d} {master[lo].date()}→{master[hi-1].date()}  "
              f"base={r0['sharpe_ratio']:6.3f} | cand={r1['sharpe_ratio']:6.3f} "
              f"| Δ={d:+.3f}")
        segs.append({"segment": i + 1, "start": str(master[lo].date()),
                     "end": str(master[hi - 1].date()),
                     "base_sharpe": round(r0["sharpe_ratio"], 4),
                     "cand_sharpe": round(r1["sharpe_ratio"], 4),
                     "base_mdd": round(r0["max_drawdown"], 4),
                     "cand_mdd": round(r1["max_drawdown"], 4),
                     "delta": round(d, 4)})
    pos = float(np.mean([s["delta"] > 0 for s in segs]))
    print(f"    ▶ Δ>0 比例 = {pos:.0%}  （3 年 8 段时是 62.5%）")
    res_all["groups"]["Y3_segments"] = {"segments": segs,
                                        "positive_frac": round(pos, 3)}

    # ── Y4. 严格 IS/OOS（5 年）──
    print("\n" + "=" * 92)
    print("Y4. 严格 IS/OOS（2021-05→2024-05 选参 / 2024-05→2026-05 验证）")
    print("=" * 92)
    is_m = np.asarray(master < OOS_SPLIT_5Y)
    oos_m = np.asarray(master >= OOS_SPLIT_5Y)
    gY4 = []
    for nm, rr in rets.items():
        m_is = metrics_from_returns(rr[is_m], master[is_m])
        m_oos = metrics_from_returns(rr[oos_m], master[oos_m])
        e = {"name": nm,
             "IS": {k: round(v, 4) for k, v in m_is.items()},
             "OOS": {k: round(v, 4) for k, v in m_oos.items()}}
        gY4.append(e)
        print(f"  {nm:20s} IS SR={m_is['sharpe_ratio']:6.3f} "
              f"ret={m_is['total_return']:+8.2%} mdd={m_is['max_drawdown']:7.2%} | "
              f"OOS SR={m_oos['sharpe_ratio']:6.3f} ret={m_oos['total_return']:+8.2%} "
              f"mdd={m_oos['max_drawdown']:7.2%}")
    res_all["groups"]["Y4_oos"] = gY4

    # ── Y5. 规模暴露稳定性 ──
    print("\n" + "=" * 92)
    print("Y5. 规模暴露：候选组合的规模暴露 vs 市场（5 年）")
    print("=" * 92)
    with np.errstate(divide="ignore", invalid="ignore"):
        day_ret = mats["close"][1:] / mats["close"][:-1] - 1.0
        day_ret = np.vstack([np.full((1, mats["close"].shape[1]), np.nan),
                              day_ret])
    mkt_ret = np.nanmean(day_ret, axis=1)
    mkt_size = np.nanmean(np.where(np.isfinite(size), size, np.nan), axis=1)
    df = pd.DataFrame({"mkt_size": mkt_size})
    for nm, rr in rets.items():
        s = pd.Series(rr, index=master)
        df[nm] = s.values
    # 组合规模暴露：用候选组合的持仓估算（用等权近似：候选选中的股票平均规模）
    print(f"    市场平均 log 成交额: 均值 {np.nanmean(mkt_size):.2f} "
          f"标准差 {np.nanstd(mkt_size):.2f}")
    gY5 = []
    for nm, sc in variants.items():
        # 每 21 天取一次截面，估计组合的规模暴露（用 Top60 的平均 size z）
        exps = []
        for t in range(0, len(master), 21):
            row = sc[t]
            ok = np.isfinite(row)
            if ok.sum() < 60:
                continue
            idx = np.where(ok)[0]
            top = idx[np.argsort(-row[idx], kind="mergesort")[:60]]
            v = size[t, top]
            if np.isfinite(v).sum() < 30:
                continue
            exps.append(float(np.nanmean(v)))
        if exps:
            m = float(np.mean(exps))
            print(f"    {nm:20s} 组合规模暴露(log成交额) = {m:.3f}")
            gY5.append({"name": nm, "size_exposure": round(m, 4)})
    res_all["groups"]["Y5_size_exposure"] = {
        "market_mean_log_amount": round(float(np.nanmean(mkt_size)), 4),
        "variants": gY5}

    # ── Y6. 波动率目标化在 5 年 ──
    print("\n" + "=" * 92)
    print("Y6. 波动率目标化在 5 年窗口")
    print("=" * 92)
    gY6 = []
    eq_c, _ = equity_series(CAND, mats, trad, CFG)
    for target in (0.08, 0.10, 0.12, 0.15, 0.20):
        for lb in (20, 40, 60):
            rm = vol_managed_returns(eq_c, master, target, lookback=lb, cap=1.0)
            m = metrics_from_returns(rm, master)
            m_is = metrics_from_returns(rm[is_m], master[is_m])
            m_oos = metrics_from_returns(rm[oos_m], master[oos_m])
            e = {"name": f"Y6_t{int(target*100)}_lb{lb}",
                 "sharpe_ratio": round(m["sharpe_ratio"], 4),
                 "total_return": round(m["total_return"], 4),
                 "max_drawdown": round(m["max_drawdown"], 4),
                 "IS_sharpe": round(m_is["sharpe_ratio"], 4),
                 "OOS_sharpe": round(m_oos["sharpe_ratio"], 4),
                 "OOS_mdd": round(m_oos["max_drawdown"], 4)}
            gY6.append(e)
            ok = "✅" if (e["sharpe_ratio"] >= 1.5 and e["max_drawdown"] >= -0.20) else "  "
            print(f"    target={target:.0%} lb={lb:3d}  SR={e['sharpe_ratio']:6.3f} "
                  f"ret={e['total_return']:+8.2%} mdd={e['max_drawdown']:7.2%} | "
                  f"OOS SR={e['OOS_sharpe']:6.3f} mdd={e['OOS_mdd']:7.2%}  {ok}")
    res_all["groups"]["Y6_vol_managed_5y"] = gY6
    n_ok = sum(1 for e in gY6
               if e["sharpe_ratio"] >= 1.5 and e["max_drawdown"] >= -0.20)
    print(f"    ▶ 5 年窗口同时满足「夏普≥1.5 且回撤≤20%」的配置数：{n_ok}/{len(gY6)}")

    res_all["n_trials"] = len(gY) + len(segs) + len(gY4) + len(gY5) + len(gY6)
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, args.out)
    with open(out, "w") as fh:
        json.dump(res_all, fh, ensure_ascii=False, indent=1, default=str)
    print(f"\n写入 {out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
