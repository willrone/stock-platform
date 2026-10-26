#!/usr/bin/env python3
"""第三轮：暴露中性化（Barra APT 风格）+ 持有期 + 最优组合，附严格 OOS。

## 前两轮的结论（浓缩）
- **信号平滑（EWMA/MA）：❌ 证伪**。0.806 → 0.71~0.74（半衰期 2~20 日
  全部更差，40/60 日崩塌到 0.34/0.21）。原因在 B2 的 alpha decay：
  **IC 不随持有期衰减，反而单调上升**（h=1: 0.070 → h=40: 0.106）。
  这个信号的 alpha 是「慢变量」，平滑只会引入滞后、把最新信息抹掉。
  → 顺带证伪了「平滑降噪提夏普」的通用建议在**本信号**上成立。
- **IC 加权复合：❌ 证伪**（修正前视后仍不及基线）。且修前视后结果
  明显变化 → 上一轮报的 IC 加权数字含 h 天前视，不可信。
- **组合构建层权重（逆波动率/HRP/最大分散化/分数比例）：❌ 证伪或无效**。
  等权 TopK 本身已是「隐式风险分散」，二次优化只加换手不加 alpha。
- **成本惩罚（Amihud）：❌ 证伪**。但归因发现它掉分不是因为「惩罚无效」，
  而是因为**信号本身带很强的低波动/小盘暴露**（A2：信号 vs 60 日波动率
  平均截面秩相关 −0.55；组合波动 0.018 vs 全市场 0.029）。
- **暴露中性化：✅ 唯一有效**（A3：42f_day 中性化 size+illiq
  → 夏普 0.806 → **1.031**，回撤 −16.6% → −12.4%，笔数 1176 → 720）。

## 本轮要回答
1. 中性化效果**是不是过拟合**？（严格 IS 选参 → OOS 验证）
2. 到底该中性化哪些暴露？（size / illiq / vol / 组合）
3. 中性化 + 别的层叠加后能到哪？
4. 中性化后持有期/调仓频率怎么选？（B2 显示 IC 随 h 上升 → 应拉长持有）

## 用法
  env -u PYTHONPATH -u PYTHONHOME backend/.venv-py313/bin/python3 \
    backend/scripts/methods_research_v3.py --groups K L M N
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
import methods_lab as M  # noqa: E402
from methods_research_v2 import neutralize_score  # noqa: E402

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
          f"n={e['total_trades']:5d}  {note[:40]}", flush=True)
    return e


def r4(d):
    return {k: (round(float(v), 4) if isinstance(v, (int, float)) else v)
            for k, v in d.items()}


def build_exposures(mats):
    """构造风格暴露矩阵：规模代理、流动性(Amihud)、波动率、beta。"""
    close, vol = mats["close"], mats["volume"]
    amount = close * vol
    with np.errstate(divide="ignore", invalid="ignore"):
        log_amt = np.log(np.where(amount > 0, amount, np.nan))
    size = pd.DataFrame(log_amt).rolling(20, min_periods=5).mean().to_numpy()
    illiq = M.amihud_illiq(close, vol, 20)
    with np.errstate(divide="ignore", invalid="ignore"):
        ret = close[1:] / close[:-1] - 1.0
        ret = np.vstack([np.full((1, close.shape[1]), np.nan), ret])
    vol60 = pd.DataFrame(ret).rolling(60, min_periods=20).std().to_numpy()
    # beta：相对等权市场
    mkt = np.nanmean(ret, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        cov = pd.DataFrame(ret).rolling(120, min_periods=40).cov(
            pd.Series(mkt)).to_numpy()
        var = pd.Series(mkt).rolling(120, min_periods=40).var().to_numpy()
        beta = cov / var[:, None]
    return {"size": size, "illiq": illiq, "vol60": vol60, "beta": beta}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="*", default=["K", "L", "M", "N"])
    ap.add_argument("--out", default="methods_research_v3.json")
    ap.add_argument("--topk", type=int, default=60)
    ap.add_argument("--ndrop", type=int, default=5)
    args = ap.parse_args()
    groups = set(args.groups)

    t0 = time.time()
    print("=" * 88)
    print("第三轮 · 暴露中性化 + 持有期（真实盘口径 lag=1）")
    print("=" * 88)

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
    expo = build_exposures(mats)
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

    # ══════════ K. 中性化暴露组合扫描 ══════════
    NEUT_SETS: Dict[str, Dict[str, np.ndarray]] = {
        "none": {},
        "size": {"size": expo["size"]},
        "illiq": {"illiq": expo["illiq"]},
        "vol60": {"vol60": expo["vol60"]},
        "size+illiq": {"size": expo["size"], "illiq": expo["illiq"]},
        "size+vol60": {"size": expo["size"], "vol60": expo["vol60"]},
        "illiq+vol60": {"illiq": expo["illiq"], "vol60": expo["vol60"]},
        "size+illiq+vol60": {"size": expo["size"], "illiq": expo["illiq"],
                              "vol60": expo["vol60"]},
        "size+illiq+vol60+beta": {"size": expo["size"], "illiq": expo["illiq"],
                                   "vol60": expo["vol60"], "beta": expo["beta"]},
    }
    neut_scores: Dict[str, np.ndarray] = {}
    if "K" in groups or "L" in groups or "M" in groups or "N" in groups:
        for nm, ex in NEUT_SETS.items():
            neut_scores[nm] = BASE if not ex else neutralize_score(BASE, ex)
        # 25f 底座也做一份（对照）
        neut_scores_25f = {
            nm: (s25 if not ex else neutralize_score(s25, ex))
            for nm, ex in NEUT_SETS.items()
        }
    else:
        neut_scores_25f = {}

    if "K" in groups:
        print("\n" + "=" * 88)
        print("K. 中性化暴露组合扫描（全窗）")
        print("=" * 88)
        gK = []
        for nm, sc in neut_scores.items():
            res = L.simulate(sc, mats, trad, CFG)
            e = record(f"K_neut_{nm}", "K", res, CFG, note=f"中性化 {nm}")
            gK.append(e)
        for nm, sc in neut_scores_25f.items():
            if nm == "none":
                continue
            res = L.simulate(sc, mats, trad, CFG)
            e = record(f"K25f_neut_{nm}", "K", res, CFG, note=f"25f 中性化 {nm}")
            gK.append(e)
        results["groups"]["K_neutralize_scan"] = gK

    # ══════════ L. 中性化 × topk × 持有期 ══════════
    if "L" in groups:
        print("\n" + "=" * 88)
        print("L. 中性化 × topk（组合构建层）")
        print("=" * 88)
        gL = []
        for nm in ("none", "size+illiq", "size+illiq+vol60"):
            for topk in (20, 30, 40, 60, 80):
                for ndrop in (max(2, topk // 12),):
                    cfg = L.SimConfig(topk=topk, n_drop=ndrop, lag=1,
                                      rebalance_days="weekly",
                                      max_position_size=(1 - 0.05) / topk)
                    res = L.simulate(neut_scores[nm], mats, trad, cfg)
                    e = record(f"L_{nm}_k{topk}_d{ndrop}", "L", res, cfg,
                               note=f"{nm} topk={topk}")
                    gL.append(e)
        results["groups"]["L_neut_topk"] = gL

    # ══════════ M. 中性化后的调仓频率 ══════════
    if "M" in groups:
        print("\n" + "=" * 88)
        print("M. 中性化后的调仓频率（IC 随 h 上升 → 应拉长持有）")
        print("=" * 88)
        gM = []
        for nm in ("none", "size+illiq"):
            for freq in ("daily", "weekly", "monthly"):
                cfg = L.SimConfig(topk=args.topk, n_drop=args.ndrop, lag=1,
                                  rebalance_days=freq,
                                  max_position_size=(1 - 0.05) / args.topk)
                res = L.simulate(neut_scores[nm], mats, trad, cfg)
                e = record(f"M_{nm}_{freq}", "M", res, cfg,
                           note=f"{nm} {freq}")
                gM.append(e)
        # qlib topk_buffer（换手更慢）—— 只对周频有意义
        print("\n  -- qlib topk_buffer（持仓掉出 topk+buffer 才卖）--")
        for nm in ("none", "size+illiq"):
            for buf in (0, 10, 20, 40):
                cfg = L.SimConfig(topk=args.topk, n_drop=args.ndrop, lag=1,
                                  rebalance_days="weekly", buffer=buf,
                                  max_position_size=(1 - 0.05) / args.topk)
                res = L.simulate(neut_scores[nm], mats, trad, cfg)
                e = record(f"M_{nm}_buf{buf}", "M", res, cfg,
                           note=f"{nm} buffer={buf}")
                gM.append(e)
        results["groups"]["M_freq"] = gM

    # ══════════ N. 严格 OOS（IS 选参 → OOS 验证）══════════
    if "N" in groups:
        print("\n" + "=" * 88)
        print("N. 严格 OOS：IS(2023-05→2025-05) 选参 → OOS(2025-05→2026-05) 验证")
        print("=" * 88)
        sub_is = L.slice_mats(mats, is_rows)
        sub_oos = L.slice_mats(mats, oos_rows)
        tr_is = {k: v[is_rows] for k, v in trad.items()}
        tr_oos = {k: v[oos_rows] for k, v in trad.items()}

        def oos_run(sc, cfg):
            r_is = L.simulate(sc[is_rows], sub_is, tr_is, cfg)
            r_oos = L.simulate(sc[oos_rows], sub_oos, tr_oos, cfg)
            return r_is, r_oos

        gN = []
        # N1 所有中性化组合的 IS/OOS
        print("  -- 中性化组合 IS/OOS --")
        table = []
        for nm, sc in neut_scores.items():
            r_is, r_oos = oos_run(sc, CFG)
            table.append((nm, r_is, r_oos))
            print(f"    {nm:22s} IS SR={r_is['sharpe_ratio']:6.3f} "
                  f"ret={r_is['total_return']:+7.2%} | "
                  f"OOS SR={r_oos['sharpe_ratio']:6.3f} "
                  f"ret={r_oos['total_return']:+7.2%} "
                  f"mdd={r_oos['max_drawdown']:7.2%} "
                  f"n={r_oos['total_trades']}")
            gN.append({"name": f"N_neut_{nm}", "group": "N",
                       "IS": r4(r_is), "OOS": r4(r_oos)})
            TRIALS.append({
                "name": f"N_neut_{nm}", "group": "N",
                "sharpe_ratio": round(float(r_oos["sharpe_ratio"]), 4),
                "total_return": round(float(r_oos["total_return"]), 4),
                "max_drawdown": round(float(r_oos["max_drawdown"]), 4),
                "total_trades": int(r_oos["total_trades"]),
                "IS_sharpe": round(float(r_is["sharpe_ratio"]), 4),
                "OOS_sharpe": round(float(r_oos["sharpe_ratio"]), 4),
                "OOS_return": round(float(r_oos["total_return"]), 4),
                "OOS_mdd": round(float(r_oos["max_drawdown"]), 4),
                "note": f"IS={r_is['sharpe_ratio']:.3f} "
                        f"OOS={r_oos['sharpe_ratio']:.3f}",
            })
        # IS 最优 → OOS 验证（只允许用 IS 选出的那一组）
        best_is = max(table, key=lambda t: t[1]["sharpe_ratio"])
        print(f"    ▶ IS 最优中性化 = {best_is[0]} "
              f"(IS SR={best_is[1]['sharpe_ratio']:.3f}) → "
              f"OOS SR={best_is[2]['sharpe_ratio']:.3f} "
              f"ret={best_is[2]['total_return']:+.2%}")
        results["groups"].setdefault("N_oos_selected", []).append({
            "selected_on_IS": best_is[0],
            "IS_sharpe": round(float(best_is[1]["sharpe_ratio"]), 4),
            "OOS_sharpe": round(float(best_is[2]["sharpe_ratio"]), 4),
            "OOS_return": round(float(best_is[2]["total_return"]), 4),
            "OOS_mdd": round(float(best_is[2]["max_drawdown"]), 4),
            "OOS_yearly": best_is[2].get("yearly", {}),
        })

        # N2 IS 选 topk → OOS 验证
        print("\n  -- IS 选 topk（中性化 size+illiq）→ OOS --")
        grid = []
        for topk in (20, 30, 40, 60, 80):
            cfg = L.SimConfig(topk=topk, n_drop=max(2, topk // 12), lag=1,
                              rebalance_days="weekly",
                              max_position_size=(1 - 0.05) / topk)
            r_is, r_oos = oos_run(neut_scores["size+illiq"], cfg)
            grid.append((topk, cfg, r_is, r_oos))
            print(f"    k={topk:3d} IS SR={r_is['sharpe_ratio']:6.3f} | "
                  f"OOS SR={r_oos['sharpe_ratio']:6.3f} "
                  f"ret={r_oos['total_return']:+7.2%} "
                  f"mdd={r_oos['max_drawdown']:7.2%}")
            gN.append({"name": f"N_k{topk}", "group": "N",
                       "IS": r4(r_is), "OOS": r4(r_oos)})
        bk = max(grid, key=lambda t: t[2]["sharpe_ratio"])
        print(f"    ▶ IS 最优 topk = {bk[0]} (IS SR={bk[2]['sharpe_ratio']:.3f}) → "
              f"OOS SR={bk[3]['sharpe_ratio']:.3f} ret={bk[3]['total_return']:+.2%}")
        results["groups"].setdefault("N_oos_selected", []).append({
            "selected_on_IS": f"topk={bk[0]} (neut=size+illiq)",
            "IS_sharpe": round(float(bk[2]["sharpe_ratio"]), 4),
            "OOS_sharpe": round(float(bk[3]["sharpe_ratio"]), 4),
            "OOS_return": round(float(bk[3]["total_return"]), 4),
            "OOS_mdd": round(float(bk[3]["max_drawdown"]), 4),
            "OOS_yearly": bk[3].get("yearly", {}),
        })

        # N3 中性化 + 其它层的叠加（IS/OOS）
        print("\n  -- 中性化 + 其它层叠加（IS/OOS）--")
        extra: Dict[str, np.ndarray] = {
            "neut": neut_scores["size+illiq"],
            "neut_ewma5": M.ewma_smooth(neut_scores["size+illiq"], halflife=5),
            "neut_ewma20": M.ewma_smooth(neut_scores["size+illiq"], halflife=20),
        }
        # 换手率反转因子
        amount = mats["close"] * mats["volume"]
        with np.errstate(divide="ignore", invalid="ignore"):
            log_amt = np.log(np.where(amount > 0, amount, np.nan))
        to_mean = pd.DataFrame(log_amt).rolling(20, min_periods=5).mean().to_numpy()
        z_to = L.cs_zscore(-to_mean, 30)
        extra["neut+to0.5"] = M.blend_scores(
            {"a": neut_scores["size+illiq"], "to": z_to},
            {"a": 1.0, "to": 0.5})
        extra["neut+to1.0"] = M.blend_scores(
            {"a": neut_scores["size+illiq"], "to": z_to},
            {"a": 1.0, "to": 1.0})
        for nm, sc in extra.items():
            r_is, r_oos = oos_run(sc, CFG)
            print(f"    {nm:16s} IS SR={r_is['sharpe_ratio']:6.3f} | "
                  f"OOS SR={r_oos['sharpe_ratio']:6.3f} "
                  f"ret={r_oos['total_return']:+7.2%} "
                  f"mdd={r_oos['max_drawdown']:7.2%} "
                  f"n={r_oos['total_trades']}")
            gN.append({"name": f"N_{nm}", "group": "N",
                       "IS": r4(r_is), "OOS": r4(r_oos)})
            TRIALS.append({
                "name": f"N_{nm}", "group": "N",
                "sharpe_ratio": round(float(r_oos["sharpe_ratio"]), 4),
                "total_return": round(float(r_oos["total_return"]), 4),
                "max_drawdown": round(float(r_oos["max_drawdown"]), 4),
                "total_trades": int(r_oos["total_trades"]),
                "IS_sharpe": round(float(r_is["sharpe_ratio"]), 4),
                "OOS_sharpe": round(float(r_oos["sharpe_ratio"]), 4),
                "OOS_return": round(float(r_oos["total_return"]), 4),
                "OOS_mdd": round(float(r_oos["max_drawdown"]), 4),
                "note": f"IS={r_is['sharpe_ratio']:.3f} "
                        f"OOS={r_oos['sharpe_ratio']:.3f}",
            })
        results["groups"]["N_oos"] = gN

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
