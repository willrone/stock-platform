#!/usr/bin/env python3
"""第四轮：把「换手率反转」因子的真实身份查清楚 + 中性化后的最终验证。

## 为什么必须查
第二轮发现 `turnover_rev`（负的 20 日 log 成交额均值）单独跑出
**夏普 1.465 / 三年 +278.9%**。这个数字大到需要怀疑：
A 股 2023-2026 是小盘/微盘股大牛市（"炒小"），而 log 成交额几乎是
市值/流动性的代理 → **这个因子可能只是一个小盘 beta**，不是 alpha。

判定方法（Barra APT 归因逻辑）：
  1. 与规模/流动性暴露的截面秩相关
  2. 组合的实际规模暴露 vs 全市场
  3. **把规模暴露中性化掉之后还剩多少收益** ← 决定性检验
  4. 与沪深300 / 中证1000 等宽基的对照（用等权市场近似）

## 分组
  P. 换手率因子的身份解剖（相关性 / 暴露 / 中性化后残存）
  Q. 中性化后的最终候选（IS 选参 → OOS 验证 + 分年）
  R. 组合拳候选（中性化 + 换手因子 + topk）
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
from methods_research_v3 import build_exposures  # noqa: E402

OOS_SPLIT = pd.Timestamp("2025-05-08")
OUT_DIR = os.path.join(ROOT, "data", "backtest")
TRIALS: List[Dict[str, Any]] = []


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
    print(f"  {name[:46]:48s} SR={e['sharpe_ratio']:6.3f} "
          f"ret={e['total_return']:+8.2%} mdd={e['max_drawdown']:7.2%} "
          f"n={e['total_trades']:5d}  {note[:38]}", flush=True)
    return e


def r4(d):
    return {k: (round(float(v), 4) if isinstance(v, (int, float)) else v)
            for k, v in d.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="*", default=["P", "Q", "R"])
    ap.add_argument("--out", default="methods_research_v4.json")
    ap.add_argument("--topk", type=int, default=60)
    ap.add_argument("--ndrop", type=int, default=5)
    args = ap.parse_args()
    groups = set(args.groups)

    t0 = time.time()
    print("=" * 88)
    print("第四轮 · 因子身份解剖 + 最终候选（真实盘口径 lag=1）")
    print("=" * 88)

    cache = M.load_cached_mats()
    mats, trad = cache["mats"], cache["trad"]
    CFG = L.SimConfig(topk=args.topk, n_drop=args.ndrop, lag=1,
                      rebalance_days="weekly",
                      max_position_size=(1 - 0.05) / args.topk)
    is_rows, oos_rows = L.split_index(mats["master"], OOS_SPLIT)

    s42, z_day = L.build_42f_score(
        mats, panel_aligned=True,
        panel_dir=os.path.join(L.QLIB_LIVE, "features", "day"))
    expo = build_exposures(mats)
    amount = mats["close"] * mats["volume"]
    with np.errstate(divide="ignore", invalid="ignore"):
        log_amt = np.log(np.where(amount > 0, amount, np.nan))
    to_mean = pd.DataFrame(log_amt).rolling(20, min_periods=5).mean().to_numpy()
    z_to = L.cs_zscore(-to_mean, 30)          # 低成交额 = 高分
    print(f"[setup] ready ({time.time()-t0:.0f}s)", flush=True)

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

    # ══════════ P. 换手率因子身份解剖 ══════════
    if "P" in groups:
        print("\n" + "=" * 88)
        print("P. 换手率反转因子的身份解剖（是不是只是小盘 beta？）")
        print("=" * 88)
        rows = []
        # P1 与各暴露的截面秩相关
        print("  -- P1: 与风格暴露的平均截面秩相关 --")
        for ename, ex in expo.items():
            ic = L.cs_rank_ic(z_to, ex)
            m, s = float(np.nanmean(ic)), float(np.nanstd(ic))
            rows.append({"exposure": ename, "mean_rank_corr": round(m, 4),
                         "std": round(s, 4)})
            print(f"    turnover_rev vs {ename:10s}  corr={m:+.4f} (std {s:.4f})")
        # P2 组合实际暴露
        print("\n  -- P2: 组合实际暴露 vs 全市场（每 5 天采样）--")
        for nm, sc in (("turnover_rev", z_to), ("42f_day", s42)):
            out = {}
            for ename, ex in expo.items():
                tops, mkts = [], []
                for t in range(0, len(mats["master"]), 5):
                    row = sc[t]
                    valid = np.where(np.isfinite(row))[0]
                    if len(valid) < 60:
                        continue
                    order = valid[np.argsort(-row[valid], kind="mergesort")][:60]
                    ev = ex[t]
                    tops.append(np.nanmean(ev[order]))
                    mkts.append(np.nanmean(ev[valid]))
                out[ename] = {
                    "portfolio": round(float(np.nanmean(tops)), 5),
                    "market": round(float(np.nanmean(mkts)), 5),
                    "diff": round(float(np.nanmean(tops) - np.nanmean(mkts)), 5),
                }
                print(f"    {nm:14s} {ename:8s} 组合 {out[ename]['portfolio']:.4f} "
                      f"vs 全市场 {out[ename]['market']:.4f} "
                      f"(差 {out[ename]['diff']:+.4f})")
            rows.append({"factor": nm, "portfolio_exposure": out})
        # P3 决定性检验：中性化规模后还剩多少
        print("\n  -- P3: 决定性检验 —— 中性化规模暴露后 alpha 还剩多少 --")
        gP = []
        for ename in ("size", "size+illiq", "size+illiq+vol60"):
            ex = {"size": expo["size"]}
            if "illiq" in ename:
                ex["illiq"] = expo["illiq"]
            if "vol60" in ename:
                ex["vol60"] = expo["vol60"]
            sn = neutralize_score(z_to, ex)
            for k in (20, 60):
                cfg = L.SimConfig(topk=k, n_drop=max(2, k // 12), lag=1,
                                  rebalance_days="weekly",
                                  max_position_size=(1 - 0.05) / k)
                res = L.simulate(sn, mats, trad, cfg)
                e = record(f"P3_turnover_neut[{ename}]_k{k}", "P", res, cfg,
                           note=f"换手因子中性化 {ename}")
                gP.append(e)
            # 中性化后与规模的残余相关
            resid_ic = L.cs_rank_ic(sn, expo["size"])
            print(f"    → 中性化 {ename} 后，与 size 的残余秩相关 "
                  f"{float(np.nanmean(resid_ic)):+.4f}")
        results["groups"]["P_identity"] = {"correlations": rows,
                                           "P3": gP}

    # ══════════ Q. 中性化后的最终候选 ══════════
    if "Q" in groups:
        print("\n" + "=" * 88)
        print("Q. 中性化后的最终候选（IS 选参 → OOS 验证）")
        print("=" * 88)
        sub_is = L.slice_mats(mats, is_rows)
        sub_oos = L.slice_mats(mats, oos_rows)
        tr_is = {k: v[is_rows] for k, v in trad.items()}
        tr_oos = {k: v[oos_rows] for k, v in trad.items()}

        def oos_run(sc, cfg):
            r_is = L.simulate(sc[is_rows], sub_is, tr_is, cfg)
            r_oos = L.simulate(sc[oos_rows], sub_oos, tr_oos, cfg)
            return r_is, r_oos

        neut_si = neutralize_score(s42, {"size": expo["size"],
                                         "illiq": expo["illiq"]})
        cands: Dict[str, np.ndarray] = {
            "base_42f": s42,
            "neut_si": neut_si,
            "turnover_rev": z_to,
            "neut_si+to0.5": M.blend_scores({"a": neut_si, "to": z_to},
                                             {"a": 1.0, "to": 0.5}),
            "neut_si+to1.0": M.blend_scores({"a": neut_si, "to": z_to},
                                             {"a": 1.0, "to": 1.0}),
            "base+to1.0": M.blend_scores({"a": s42, "to": z_to},
                                         {"a": 1.0, "to": 1.0}),
        }
        gQ = []
        for nm, sc in cands.items():
            r_is, r_oos = oos_run(sc, CFG)
            print(f"    {nm:16s} IS SR={r_is['sharpe_ratio']:6.3f} "
                  f"ret={r_is['total_return']:+7.2%} mdd={r_is['max_drawdown']:7.2%}"
                  f" | OOS SR={r_oos['sharpe_ratio']:6.3f} "
                  f"ret={r_oos['total_return']:+7.2%} "
                  f"mdd={r_oos['max_drawdown']:7.2%} n={r_oos['total_trades']}")
            print(f"    {'':16s}   分年 IS {r_is.get('yearly')} | "
                  f"OOS {r_oos.get('yearly')}")
            gQ.append({"name": f"Q_{nm}", "IS": r4(r_is), "OOS": r4(r_oos)})
            TRIALS.append({
                "name": f"Q_{nm}", "group": "Q",
                "sharpe_ratio": round(float(r_oos["sharpe_ratio"]), 4),
                "total_return": round(float(r_oos["total_return"]), 4),
                "max_drawdown": round(float(r_oos["max_drawdown"]), 4),
                "total_trades": int(r_oos["total_trades"]),
                "IS_sharpe": round(float(r_is["sharpe_ratio"]), 4),
                "OOS_sharpe": round(float(r_oos["sharpe_ratio"]), 4),
                "OOS_return": round(float(r_oos["total_return"]), 4),
                "OOS_mdd": round(float(r_oos["max_drawdown"]), 4),
                "OOS_yearly": r_oos.get("yearly", {}),
                "note": f"IS={r_is['sharpe_ratio']:.3f} "
                        f"OOS={r_oos['sharpe_ratio']:.3f}",
            })
        results["groups"]["Q_final"] = gQ

    # ══════════ R. 组合拳候选 ══════════
    if "R" in groups:
        print("\n" + "=" * 88)
        print("R. 组合拳（中性化 + 换手因子 × topk），IS 选参 → OOS")
        print("=" * 88)
        sub_is = L.slice_mats(mats, is_rows)
        sub_oos = L.slice_mats(mats, oos_rows)
        tr_is = {k: v[is_rows] for k, v in trad.items()}
        tr_oos = {k: v[oos_rows] for k, v in trad.items()}
        neut_si = neutralize_score(s42, {"size": expo["size"],
                                         "illiq": expo["illiq"]})
        blends = {
            "neut_si": M.blend_scores({"a": neut_si}, {"a": 1.0}),
            "neut_si+to0.5": M.blend_scores({"a": neut_si, "to": z_to},
                                             {"a": 1.0, "to": 0.5}),
            "neut_si+to1.0": M.blend_scores({"a": neut_si, "to": z_to},
                                             {"a": 1.0, "to": 1.0}),
        }
        gR = []
        grid = []
        for bname, sc in blends.items():
            for topk in (20, 30, 40, 60):
                cfg = L.SimConfig(topk=topk, n_drop=max(2, topk // 12), lag=1,
                                  rebalance_days="weekly",
                                  max_position_size=(1 - 0.05) / topk)
                r_is = L.simulate(sc[is_rows], sub_is, tr_is, cfg)
                r_oos = L.simulate(sc[oos_rows], sub_oos, tr_oos, cfg)
                grid.append((bname, topk, cfg, r_is, r_oos))
                print(f"    {bname:14s} k={topk:3d} "
                      f"IS SR={r_is['sharpe_ratio']:6.3f} "
                      f"ret={r_is['total_return']:+7.2%} | "
                      f"OOS SR={r_oos['sharpe_ratio']:6.3f} "
                      f"ret={r_oos['total_return']:+7.2%} "
                      f"mdd={r_oos['max_drawdown']:7.2%}")
                gR.append({"name": f"R_{bname}_k{topk}", "IS": r4(r_is),
                           "OOS": r4(r_oos)})
        best = max(grid, key=lambda t: t[3]["sharpe_ratio"])
        print(f"    ▶ IS 最优 = {best[0]} topk={best[1]} "
              f"(IS SR={best[3]['sharpe_ratio']:.3f}) → "
              f"OOS SR={best[4]['sharpe_ratio']:.3f} "
              f"ret={best[4]['total_return']:+.2%} "
              f"mdd={best[4]['max_drawdown']:.2%} "
              f"分年 {best[4].get('yearly')}")
        results["groups"].setdefault("R_oos_selected", []).append({
            "selected_on_IS": f"{best[0]} topk={best[1]}",
            "IS_sharpe": round(float(best[3]["sharpe_ratio"]), 4),
            "IS_return": round(float(best[3]["total_return"]), 4),
            "OOS_sharpe": round(float(best[4]["sharpe_ratio"]), 4),
            "OOS_return": round(float(best[4]["total_return"]), 4),
            "OOS_mdd": round(float(best[4]["max_drawdown"]), 4),
            "OOS_trades": int(best[4]["total_trades"]),
            "OOS_yearly": best[4].get("yearly", {}),
        })
        results["groups"]["R_grid"] = gR

    results["trials"] = TRIALS
    results["n_trials"] = len(TRIALS)
    ranked = sorted(TRIALS, key=lambda e: -e.get("sharpe_ratio", 0))
    results["top10"] = ranked[:10]
    print("\n" + "=" * 88)
    print(f"共 {len(TRIALS)} 次试验。Top10：")
    for e in ranked[:10]:
        print(f"  {e['name'][:46]:48s} SR={e['sharpe_ratio']:6.3f} "
              f"ret={e['total_return']:+8.2%} mdd={e['max_drawdown']:7.2%} "
              f"n={e['total_trades']:5d}")
    print("=" * 88)
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, args.out)
    with open(path, "w") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=1, default=str)
    print(f"写入 {path}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
