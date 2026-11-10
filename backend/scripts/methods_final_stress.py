#!/usr/bin/env python3
"""最终候选「中性化 → 再混换手因子」的**过拟合压力测试**。

## 候选
`neut_si + to1.0`（先中性化 size+illiq，再混 w=1.0 的低成交额因子）
真实引擎 3 年全窗：夏普 **1.5843** / 收益 +152.7% / 回撤 −29.98% /
笔数 726 / **胜率 74.17%**。

这个数字**同时越过**了用户目标里的「夏普 ≥1.5」和「胜率 ≥75%」（差一点），
但**回撤 −29.98% 超了 20% 的上限**。所以必须搞清楚：
1. 这个 1.58 是真的吗？（分段 + IS/OOS + 随机对照）
2. 回撤为什么爆？（是不是集中在某一段）
3. 有没有办法在保住夏普/胜率的前提下把回撤压回 20% 以内？

## 压力测试清单
  U1. 分段一致性（8 段）→ 收益是否集中在某一段
  U2. 严格 IS/OOS（IS 选参 → OOS 验证）+ 增量同向性
  U3. 随机对照：把「低成交额因子」换成**随机因子**做同样混合
     → 如果随机因子也有效，说明不是这个因子的功劳
  U4. 回撤来源定位：最大回撤区间在什么时候、贡献来自哪些持仓
  U5. 回撤控制杠杆（都在组合构建层）：
     - 中性化暴露集合（加 beta / 加 vol60）
     - n_drop（换手 → 回撤）
     - topk
     - 调仓频率
     - 与「回撤刹车」regime 过滤叠加（honest_composite 里已有的机制）
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
from methods_research_v2 import neutralize_score  # noqa: E402
from methods_winrate_scan import simulate_winrate  # noqa: E402

OOS_SPLIT = pd.Timestamp("2025-05-08")
OUT_DIR = os.path.join(ROOT, "data", "backtest")
TRIALS: List[Dict[str, Any]] = []
RNG = np.random.default_rng(20261026)


def record(name, group, res, note="", extra=None):
    e = {
        "name": name, "group": group,
        "sharpe_ratio": round(float(res["sharpe_ratio"]), 4),
        "total_return": round(float(res["total_return"]), 4),
        "max_drawdown": round(float(res["max_drawdown"]), 4),
        "total_trades": int(res["total_trades"]),
        "volatility": round(float(res.get("volatility", 0.0)), 4),
        "yearly": res.get("yearly", {}),
        "win_rate": (round(float(res["win_rate"]), 4)
                     if "win_rate" in res else None),
        "note": note,
    }
    if extra:
        e.update(extra)
    TRIALS.append(e)
    print(f"  {name[:40]:42s} SR={e['sharpe_ratio']:6.3f} "
          f"ret={e['total_return']:+8.2%} mdd={e['max_drawdown']:7.2%} "
          f"win={e['win_rate'] if e['win_rate'] is None else format(e['win_rate'],'.2%')} "
          f"n={e['total_trades']:5d}  {note[:26]}", flush=True)
    return e


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="methods_final_stress.json")
    args = ap.parse_args()

    t0 = time.time()
    print("=" * 92)
    print("最终候选「中性化 + 低成交额因子」过拟合压力测试")
    print("=" * 92)

    cache = M.load_cached_mats()
    mats, trad = cache["mats"], cache["trad"]
    master = mats["master"]
    s42, z_day = L.build_42f_score(
        mats, panel_aligned=True,
        panel_dir=os.path.join(L.QLIB_LIVE, "features", "day"))
    amount = mats["close"] * mats["volume"]
    with np.errstate(divide="ignore", invalid="ignore"):
        log_amt = np.log(np.where(amount > 0, amount, np.nan))
    size = pd.DataFrame(log_amt).rolling(20, min_periods=5).mean().to_numpy()
    to_mean = pd.DataFrame(log_amt).rolling(20, min_periods=5).mean().to_numpy()
    illiq = M.amihud_illiq(mats["close"], mats["volume"], 20)
    with np.errstate(divide="ignore", invalid="ignore"):
        ret = mats["close"][1:] / mats["close"][:-1] - 1.0
        ret = np.vstack([np.full((1, mats["close"].shape[1]), np.nan), ret])
    vol60 = pd.DataFrame(ret).rolling(60, min_periods=20).std().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        cov = pd.DataFrame(ret).rolling(120, min_periods=40).cov(
            pd.Series(np.nanmean(ret, axis=1))).to_numpy()
        var = pd.Series(np.nanmean(ret, axis=1)).rolling(
            120, min_periods=40).var().to_numpy()
        beta = cov / var[:, None]

    neut = neutralize_score(s42, {"size": size, "illiq": illiq})
    z_to = L.cs_zscore(-to_mean, 30)
    CAND = M.blend_scores({"a": neut, "to": z_to}, {"a": 1.0, "to": 1.0})
    BASE = s42
    print(f"[setup] ready ({time.time()-t0:.0f}s)", flush=True)

    res_all: Dict[str, Any] = {
        "meta": {
            "window": f"{master[0].date()}→{master[-1].date()}",
            "n_codes": len(mats["codes"]),
            "n_trading_days": len(master),
            "oos_split": str(OOS_SPLIT.date()),
            "caliber": {"lag": 1, "enable_tradability": True,
                        "topk": 60, "n_drop": 5,
                        "commission": 0.0003, "slippage": 0.0001,
                        "board_lot": 100, "cash_reserve": 0.05,
                        "rebalance": "weekly(周五)"},
            "engine_candidate": {
                "sharpe_ratio": 1.584305546942995,
                "total_return": 1.527373462360674,
                "max_drawdown": -0.2998453224629893,
                "total_trades": 726,
                "win_rate": 0.7417417417417418,
            },
            "engine_baseline_42f": {
                "sharpe_ratio": 0.7991795250472001,
                "total_return": 0.34779248262737783,
                "max_drawdown": -0.16460289642093642,
                "total_trades": 1080,
                "win_rate": 0.6705882352941176,
            },
            "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        },
        "groups": {},
    }

    CFG = L.SimConfig(topk=60, n_drop=5, lag=1, rebalance_days="weekly",
                      max_position_size=(1 - 0.05) / 60)
    is_rows, oos_rows = L.split_index(master, OOS_SPLIT)
    sub_is = L.slice_mats(mats, is_rows)
    sub_oos = L.slice_mats(mats, oos_rows)
    tr_is = {k: v[is_rows] for k, v in trad.items()}
    tr_oos = {k: v[oos_rows] for k, v in trad.items()}

    # ══════════ U1. 分段一致性 ══════════
    print("\n" + "=" * 92)
    print("U1. 分段一致性（3 年切 8 段）")
    print("=" * 92)
    n_seg = 8
    seg_len = len(master) // n_seg
    rows = []
    for i in range(n_seg):
        lo = i * seg_len
        hi = (i + 1) * seg_len if i < n_seg - 1 else len(master)
        idx = np.arange(lo, hi)
        sub = L.slice_mats(mats, idx)
        tr = {k: v[idx] for k, v in trad.items()}
        r0 = L.simulate(BASE[idx], sub, tr, CFG)
        r1 = L.simulate(CAND[idx], sub, tr, CFG)
        d = r1["sharpe_ratio"] - r0["sharpe_ratio"]
        print(f"    段{i+1} {master[lo].date()}→{master[hi-1].date()}  "
              f"基线 SR={r0['sharpe_ratio']:6.3f} mdd={r0['max_drawdown']:7.2%} | "
              f"候选 SR={r1['sharpe_ratio']:6.3f} mdd={r1['max_drawdown']:7.2%} "
              f"| Δ={d:+.3f}")
        rows.append({"segment": i + 1, "start": str(master[lo].date()),
                     "end": str(master[hi - 1].date()),
                     "base_sharpe": round(r0["sharpe_ratio"], 4),
                     "cand_sharpe": round(r1["sharpe_ratio"], 4),
                     "base_mdd": round(r0["max_drawdown"], 4),
                     "cand_mdd": round(r1["max_drawdown"], 4),
                     "delta": round(d, 4)})
    pos = float(np.mean([r["delta"] > 0 for r in rows]))
    print(f"    ▶ Δ>0 的比例 = {pos:.0%}")
    res_all["groups"]["U1_segments"] = {"segments": rows,
                                         "positive_frac": round(pos, 3)}

    # ══════════ U2. 严格 IS/OOS ══════════
    print("\n" + "=" * 92)
    print("U2. 严格 IS/OOS（增量同向性）")
    print("=" * 92)
    r_is0 = L.simulate(BASE[is_rows], sub_is, tr_is, CFG)
    r_is1 = L.simulate(CAND[is_rows], sub_is, tr_is, CFG)
    r_oos0 = L.simulate(BASE[oos_rows], sub_oos, tr_oos, CFG)
    r_oos1 = L.simulate(CAND[oos_rows], sub_oos, tr_oos, CFG)
    print(f"    基线   IS SR={r_is0['sharpe_ratio']:6.3f} ret={r_is0['total_return']:+7.2%} "
          f"mdd={r_is0['max_drawdown']:7.2%} | "
          f"OOS SR={r_oos0['sharpe_ratio']:6.3f} ret={r_oos0['total_return']:+7.2%} "
          f"mdd={r_oos0['max_drawdown']:7.2%}")
    print(f"    候选   IS SR={r_is1['sharpe_ratio']:6.3f} ret={r_is1['total_return']:+7.2%} "
          f"mdd={r_is1['max_drawdown']:7.2%} | "
          f"OOS SR={r_oos1['sharpe_ratio']:6.3f} ret={r_oos1['total_return']:+7.2%} "
          f"mdd={r_oos1['max_drawdown']:7.2%}")
    print(f"    → ΔIS = {r_is1['sharpe_ratio']-r_is0['sharpe_ratio']:+.3f}  "
          f"ΔOOS = {r_oos1['sharpe_ratio']-r_oos0['sharpe_ratio']:+.3f}")
    res_all["groups"]["U2_oos"] = {
        "base": {"IS": r4(r_is0), "OOS": r4(r_oos0)},
        "cand": {"IS": r4(r_is1), "OOS": r4(r_oos1)},
        "delta_IS": round(r_is1["sharpe_ratio"] - r_is0["sharpe_ratio"], 4),
        "delta_OOS": round(r_oos1["sharpe_ratio"] - r_oos0["sharpe_ratio"], 4),
    }
    for nm, r in (("base", r_is0), ("cand", r_is1), ("base_oos", r_oos0),
                  ("cand_oos", r_oos1)):
        record(f"U2_{nm}", "U2", r, nm)

    # ══════════ U3. 随机对照（换成随机因子）══════════
    print("\n" + "=" * 92)
    print("U3. 随机对照：把「低成交额因子」换成随机因子做同样混合")
    print("=" * 92)
    gU3 = []
    deltas = []
    for trial in range(5):
        rand = np.full_like(z_to, np.nan)
        for t in range(z_to.shape[0]):
            v = z_to[t]
            ok = np.isfinite(v)
            if ok.sum() < 50:
                continue
            rand[t, ok] = RNG.permutation(v[ok])
        sc = M.blend_scores({"a": neut, "to": rand}, {"a": 1.0, "to": 1.0})
        r1 = L.simulate(sc, mats, trad, CFG)
        d = r1["sharpe_ratio"] - r_is1["sharpe_ratio"]
        deltas.append(d)
        e = record(f"U3_randfactor_t{trial}", "U3", r1,
                   f"随机因子混合 (真因子 SR={r_is1['sharpe_ratio']:.3f})")
        e["delta_vs_real"] = round(d, 4)
        print(f"    └─ Δ vs 真因子 = {d:+.4f}")
        gU3.append(e)
    print(f"    ▶ 随机因子 Δ均值 = {np.mean(deltas):+.4f} "
          f"（应显著 < 0 才说明是真因子的功劳）")
    res_all["groups"]["U3_random_factor"] = {
        "deltas": [round(float(d), 4) for d in deltas],
        "mean": round(float(np.mean(deltas)), 4)}

    # ══════════ U4. 回撤来源定位 ══════════
    print("\n" + "=" * 92)
    print("U4. 回撤来源定位（候选组合的最大回撤区间）")
    print("=" * 92)

    def equity_curve(sc):
        """跑一遍并返回日净值序列（用 simulate 的内部逻辑简化版）。"""
        sub = L.slice_mats(mats, np.arange(len(master)))
        return L.simulate(sc, mats, trad, CFG)

    # 用 simulate 的结果反推不了日序列 → 直接重跑一个简化版取净值
    eq = _equity_series(CAND, mats, trad, CFG)
    if eq is not None:
        s = pd.Series(eq, index=master)
        cum = s / s.iloc[0]
        dd = (cum - cum.cummax()) / cum.cummax()
        trough = dd.idxmin()
        peak = s.loc[:trough].idxmax()
        print(f"    最大回撤区间: {peak.date()} → {trough.date()}"
              f"  回撤 {dd.min():.2%}")
        print(f"    峰值净值 {s.loc[peak]/s.iloc[0]:.3f} → "
              f"谷底 {s.loc[trough]/s.iloc[0]:.3f}")
        seg = s.loc[peak:trough]
        print(f"    区间长度 {len(seg)} 个交易日，区间收益 "
              f"{seg.iloc[-1]/seg.iloc[0]-1:+.2%}")
        res_all["groups"]["U4_drawdown"] = {
            "peak": str(peak.date()), "trough": str(trough.date()),
            "mdd": round(float(dd.min()), 4),
            "trading_days": int(len(seg)),
        }

    # ══════════ U5. 回撤控制杠杆 ══════════
    print("\n" + "=" * 92)
    print("U5. 回撤控制杠杆（目标：回撤 ≤ 20% 且保住夏普）")
    print("=" * 92)
    gU5 = []
    # U5a 中性化暴露集合
    for nm, ex in (
        ("size+illiq", {"size": size, "illiq": illiq}),
        ("size+illiq+vol60", {"size": size, "illiq": illiq, "vol60": vol60}),
        ("size+illiq+beta", {"size": size, "illiq": illiq, "beta": beta}),
        ("size+illiq+vol60+beta", {"size": size, "illiq": illiq,
                                    "vol60": vol60, "beta": beta}),
    ):
        sn = neutralize_score(s42, ex)
        sc = M.blend_scores({"a": sn, "to": z_to}, {"a": 1.0, "to": 1.0})
        r = L.simulate(sc, mats, trad, CFG)
        e = record(f"U5a_neut_{nm}", "U5", r, f"中性化 {nm}")
        gU5.append(e)

    # U5b n_drop / topk
    for nd in (1, 2, 3, 5, 8):
        cfg = L.SimConfig(topk=60, n_drop=nd, lag=1, rebalance_days="weekly",
                          max_position_size=(1 - 0.05) / 60)
        r = L.simulate(CAND, mats, trad, cfg)
        e = record(f"U5b_ndrop{nd}", "U5", r, f"n_drop={nd}")
        gU5.append(e)
    for tk in (20, 30, 40, 60, 80):
        cfg = L.SimConfig(topk=tk, n_drop=max(1, tk // 12), lag=1,
                          rebalance_days="weekly",
                          max_position_size=(1 - 0.05) / tk)
        r = L.simulate(CAND, mats, trad, cfg)
        e = record(f"U5c_topk{tk}", "U5", r, f"topk={tk}")
        gU5.append(e)

    # U5d 调仓频率
    for freq in ("daily", "weekly", "monthly"):
        cfg = L.SimConfig(topk=60, n_drop=5, lag=1, rebalance_days=freq,
                          max_position_size=(1 - 0.05) / 60)
        r = L.simulate(CAND, mats, trad, cfg)
        e = record(f"U5d_{freq}", "U5", r, f"{freq}")
        gU5.append(e)
    res_all["groups"]["U5_drawdown_levers"] = gU5

    res_all["trials"] = TRIALS
    res_all["n_trials"] = len(TRIALS)
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, args.out)
    with open(path, "w") as fh:
        json.dump(res_all, fh, ensure_ascii=False, indent=1, default=str)
    print(f"\n写入 {path}  ({time.time()-t0:.0f}s)")


def r4(d):
    return {k: (round(float(v), 4) if isinstance(v, (int, float)) else v)
            for k, v in d.items()}


def _equity_series(score, mats, trad, cfg):
    """跑一遍模拟并返回日净值序列（复制 ensemble_lab.simulate 的核心逻辑）。"""
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

            def _buy(c, pv):
                nonlocal cash
                if not can_buy[t, c]:
                    return
                px_c = px[c]
                avail = min(pv * cfg.max_position_size,
                            cash * (1 - cfg.cash_reserve))
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
                    _buy(c, _pv())
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
                    _buy(c, _pv())
        val_px = np.where(np.isfinite(px) & (px > 0), px, last_px)
        eq.append(cash + float(np.sum(qty * val_px)))
    return eq


if __name__ == "__main__":
    main()
