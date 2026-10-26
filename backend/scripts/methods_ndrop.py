#!/usr/bin/env python3
"""n_drop=1 组合（中性化 + n_drop=1）的严格 OOS 验证。

## 动机（来自 methods_winrate_scan.py）
组合构建层里，**n_drop 是同时改善夏普和胜率的唯一杠杆**：

| 变体 | n_drop | 夏普 | 胜率 | 盈亏比 | 平仓笔数 |
|---|---|---|---|---|---|
| base | 5 | 0.806 | 62.72% | 2.048 | 558 |
| base | **1** | 0.910 | **67.77%** | 3.729 | 121 |
| neut | 5 | 1.031 | 59.09% | 2.277 | 330 |
| neut | **1** | **1.251** | **67.37%** | **4.380** | 95 |

机制：n_drop=1 → 每周只换掉**最差的那 1 只**（而不是最差的 5 只）→
换手降一个数量级 → 交易成本几乎消失 + 避免「把还在小赚的票割掉」→ 胜率上升。

## 本轮要回答
1. n_drop=1 的收益是**真实机制**还是**样本内巧合**？（严格 IS/OOS）
2. n_drop=1 在**未中性化**的底座上也有效吗？（是否与中性化正交）
3. topk × n_drop 的联合网格（IS 选参 → OOS 验证）
4. 分年收益
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
from methods_winrate_scan import simulate_winrate  # noqa: E402

OOS_SPLIT = pd.Timestamp("2025-05-08")
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
        "win_rate": round(float(res.get("win_rate", 0.0)), 4),
        "payoff_ratio": (round(float(res["payoff_ratio"]), 4)
                         if np.isfinite(res.get("payoff_ratio", np.nan)) else None),
        "note": note,
    }
    if extra:
        e.update(extra)
    TRIALS.append(e)
    print(f"  {name[:36]:38s} SR={e['sharpe_ratio']:6.3f} "
          f"ret={e['total_return']:+8.2%} mdd={e['max_drawdown']:7.2%} "
          f"win={e['win_rate']:6.2%} n={e['total_trades']:5d}  {note[:34]}",
          flush=True)
    return e


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="methods_ndrop.json")
    args = ap.parse_args()

    t0 = time.time()
    print("=" * 92)
    print("n_drop 杠杆的严格 OOS 验证（真实盘口径 lag=1）")
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

    is_rows, oos_rows = L.split_index(master, OOS_SPLIT)
    sub_is = L.slice_mats(mats, is_rows)
    sub_oos = L.slice_mats(mats, oos_rows)
    tr_is = {k: v[is_rows] for k, v in trad.items()}
    tr_oos = {k: v[oos_rows] for k, v in trad.items()}

    res_all: Dict[str, Any] = {
        "meta": {
            "window": f"{master[0].date()}→{master[-1].date()}",
            "n_codes": len(mats["codes"]),
            "n_trading_days": len(master),
            "oos_split": str(OOS_SPLIT.date()),
            "caliber": {"lag": 1, "enable_tradability": True,
                        "commission": 0.0003, "slippage": 0.0001,
                        "board_lot": 100, "cash_reserve": 0.05,
                        "rebalance": "weekly(周五)"},
            "engine_baseline_42f": {
                "sharpe_ratio": 0.7991795250472001,
                "win_rate": 0.6705882352941176,
                "max_drawdown": -0.16460289642093642,
                "total_trades": 1080,
            },
            "engine_neut_size_illiq": {
                "sharpe_ratio": 1.0524378432259367,
                "win_rate": 0.5764331210191083,
                "max_drawdown": -0.10622457102291379,
                "total_trades": 687,
            },
            "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        },
        "groups": {},
    }

    # ══════════ 网格 IS/OOS ══════════
    print("\n" + "=" * 92)
    print("X1. topk × n_drop 网格（底座 = 中性化），IS/OOS 逐格")
    print("=" * 92)
    grid: List[Tuple[str, int, int, Dict, Dict]] = []
    gX = []
    for tag, sc in (("base", s42), ("neut", neut)):
        for topk in (20, 30, 40, 60, 80):
            for n_drop in (1, 2, 3, 5):
                cfg = L.SimConfig(topk=topk, n_drop=n_drop, lag=1,
                                  rebalance_days="weekly",
                                  max_position_size=(1 - 0.05) / topk)
                r_is = simulate_winrate(sc[is_rows], sub_is, tr_is, cfg)
                r_oos = simulate_winrate(sc[oos_rows], sub_oos, tr_oos, cfg)
                grid.append((tag, topk, n_drop, r_is, r_oos))
                e = {"name": f"X1_{tag}_k{topk}_d{n_drop}", "group": "X1",
                     "IS": {k: (round(float(v), 4) if isinstance(v, (int, float))
                                and np.isfinite(v) else v)
                            for k, v in r_is.items()},
                     "OOS": {k: (round(float(v), 4) if isinstance(v, (int, float))
                                and np.isfinite(v) else v)
                             for k, v in r_oos.items()}}
                gX.append(e)
                print(f"    {tag:5s} k={topk:3d} d={n_drop}  "
                      f"IS SR={r_is['sharpe_ratio']:6.3f} "
                      f"win={r_is['win_rate']:6.2%} | "
                      f"OOS SR={r_oos['sharpe_ratio']:6.3f} "
                      f"ret={r_oos['total_return']:+7.2%} "
                      f"mdd={r_oos['max_drawdown']:7.2%} "
                      f"win={r_oos['win_rate']:6.2%} n={r_oos['total_trades']}")
                TRIALS.append({
                    "name": f"X1_{tag}_k{topk}_d{n_drop}", "group": "X1",
                    "sharpe_ratio": round(float(r_oos["sharpe_ratio"]), 4),
                    "total_return": round(float(r_oos["total_return"]), 4),
                    "max_drawdown": round(float(r_oos["max_drawdown"]), 4),
                    "total_trades": int(r_oos["total_trades"]),
                    "IS_sharpe": round(float(r_is["sharpe_ratio"]), 4),
                    "OOS_sharpe": round(float(r_oos["sharpe_ratio"]), 4),
                    "OOS_return": round(float(r_oos["total_return"]), 4),
                    "OOS_mdd": round(float(r_oos["max_drawdown"]), 4),
                    "OOS_win_rate": round(float(r_oos["win_rate"]), 4),
                    "note": f"IS={r_is['sharpe_ratio']:.3f} "
                            f"OOS={r_oos['sharpe_ratio']:.3f}",
                })
    res_all["groups"]["X1_grid"] = gX

    # IS 选参 → OOS 验证
    for tag in ("base", "neut"):
        sub = [g for g in grid if g[0] == tag]
        best = max(sub, key=lambda t: t[3]["sharpe_ratio"])
        print(f"\n    ▶ IS 最优（{tag}）= topk={best[1]} n_drop={best[2]} "
              f"(IS SR={best[3]['sharpe_ratio']:.3f} "
              f"win={best[3]['win_rate']:.2%}) → "
              f"OOS SR={best[4]['sharpe_ratio']:.3f} "
              f"ret={best[4]['total_return']:+.2%} "
              f"mdd={best[4]['max_drawdown']:.2%} "
              f"win={best[4]['win_rate']:.2%}")
        res_all["groups"].setdefault("X1_oos_selected", []).append({
            "base": tag,
            "selected": f"topk={best[1]} n_drop={best[2]}",
            "IS_sharpe": round(float(best[3]["sharpe_ratio"]), 4),
            "IS_win_rate": round(float(best[3]["win_rate"]), 4),
            "OOS_sharpe": round(float(best[4]["sharpe_ratio"]), 4),
            "OOS_return": round(float(best[4]["total_return"]), 4),
            "OOS_mdd": round(float(best[4]["max_drawdown"]), 4),
            "OOS_win_rate": round(float(best[4]["win_rate"]), 4),
            "OOS_trades": int(best[4]["total_trades"]),
            "OOS_yearly": best[4].get("yearly", {}),
        })

    # ══════════ X2. n_drop 单变量：IS/OOS 一致性 ══════════
    print("\n" + "=" * 92)
    print("X2. n_drop 单变量（topk=60）的 IS/OOS 一致性")
    print("=" * 92)
    gX2 = []
    for tag, sc in (("base", s42), ("neut", neut)):
        for n_drop in (1, 2, 3, 5, 8):
            cfg = L.SimConfig(topk=60, n_drop=n_drop, lag=1,
                              rebalance_days="weekly",
                              max_position_size=(1 - 0.05) / 60)
            r_is = simulate_winrate(sc[is_rows], sub_is, tr_is, cfg)
            r_oos = simulate_winrate(sc[oos_rows], sub_oos, tr_oos, cfg)
            e = record(f"X2_{tag}_d{n_drop}", "X2", r_oos,
                       f"{tag} n_drop={n_drop} (OOS)",
                       extra={"IS_sharpe": round(r_is["sharpe_ratio"], 4),
                              "IS_win_rate": round(r_is["win_rate"], 4),
                              "OOS_sharpe": round(r_oos["sharpe_ratio"], 4),
                              "OOS_win_rate": round(r_oos["win_rate"], 4),
                              "OOS_yearly": r_oos.get("yearly", {})})
            print(f"      IS SR={r_is['sharpe_ratio']:6.3f} "
                  f"win={r_is['win_rate']:6.2%}  |  "
                  f"OOS SR={r_oos['sharpe_ratio']:6.3f} "
                  f"win={r_oos['win_rate']:6.2%}  "
                  f"分年 {r_oos.get('yearly')}")
            gX2.append(e)
    res_all["groups"]["X2_ndrop"] = gX2

    res_all["trials"] = TRIALS
    res_all["n_trials"] = len(TRIALS)
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, args.out)
    with open(path, "w") as fh:
        json.dump(res_all, fh, ensure_ascii=False, indent=1, default=str)
    print(f"\n写入 {path}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
