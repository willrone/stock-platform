#!/usr/bin/env python3
"""波动率目标化叠加的**过拟合压力测试**（最后一轮）。

## 现状
模拟器上，`候选(neut+to1.0) + 波动率目标化` 有 **10 个配置同时满足**
「夏普 ≥1.5 且回撤 ≤20%」，最好的是 target_vol=10% / cap=1.0：
夏普 2.092 / 回撤 −13.33%。

## 但必须先问三个问题
Q1. **是不是纯杠杆？** 缩放不改变夏普（Sharpe 是尺度不变的）。
    若「缩放版夏普」≈「未缩放候选夏普」，说明缩放没带来信息，
    只是把同一条收益曲线压扁了 → 回撤降了，夏普没变 → 那也**没有价值**
    （除非目标是降回撤，那就有效）。
Q2. **IS/OOS 同向吗？** 缩放参数（target_vol、lookback）也是可过拟合的旋钮。
Q3. **引擎能执行吗？** 缩放是**组合层**操作（按组合波动率调仓仓比例），
    引擎的 TopK-Dropout 执行器不支持 per-stock 目标权重，
    也不支持组合级杠杆调整 → **要落地必须改引擎**。

## 本轮
  W1. Sharpe 尺度不变性验证（缩放不改变夏普）→ 量化「缩放的信息量」
  W2. 严格 IS/OOS（缩放参数在 IS 上选 → OOS 验证）
  W3. lookback 敏感性
  W4. 「减仓」vs「加仓」的分解：只做 f≤1（纯去杠杆）时的效果
      → 如果只去杠杆就能达标，说明不需要加杠杆能力（更易落地）
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
from methods_volmanaged import equity_series, vol_managed_returns  # noqa: E402

OOS_SPLIT = pd.Timestamp("2025-05-08")
OUT_DIR = os.path.join(ROOT, "data", "backtest")
TRIALS: List[Dict[str, Any]] = []


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


def record(name, group, m, note="", extra=None):
    e = {"name": name, "group": group,
         "sharpe_ratio": round(m["sharpe_ratio"], 4),
         "total_return": round(m["total_return"], 4),
         "max_drawdown": round(m["max_drawdown"], 4),
         "volatility": round(m["volatility"], 4),
         "note": note}
    if extra:
        e.update(extra)
    TRIALS.append(e)
    print(f"  {name[:40]:42s} SR={e['sharpe_ratio']:6.3f} "
          f"ret={e['total_return']:+8.2%} mdd={e['max_drawdown']:7.2%} "
          f"vol={e['volatility']:.4f}  {note[:30]}", flush=True)
    return e


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="methods_vol_stress.json")
    args = ap.parse_args()

    t0 = time.time()
    print("=" * 92)
    print("波动率目标化的过拟合压力测试")
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
            "oos_split": str(OOS_SPLIT.date()),
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

    eq_c, _ = equity_series(CAND, mats, trad, CFG)
    r_c = pd.Series(eq_c).pct_change().fillna(0.0).to_numpy()

    # ══════════ W1. Sharpe 尺度不变性 ══════════
    print("\n" + "=" * 92)
    print("W1. 尺度不变性：缩放是否改变夏普？（Sharpe 对尺度不变）")
    print("=" * 92)
    base_m = metrics_from_returns(r_c, master)
    record("W1_cand_unscaled", "W1", base_m, "候选未缩放")
    for k in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0):
        m = metrics_from_returns(r_c * k, master)
        record(f"W1_scaled_x{k}", "W1", m, f"固定缩放 {k}（纯杠杆）")
    print("    → 夏普应完全不变；回撤按 k 线性缩放 → **缩放只是杠杆**")

    # ══════════ W2. 严格 IS/OOS ══════════
    print("\n" + "=" * 92)
    print("W2. 严格 IS/OOS：缩放参数在 IS 选 → OOS 验证")
    print("=" * 92)
    is_mask = np.asarray(master < OOS_SPLIT)
    oos_mask = np.asarray(master >= OOS_SPLIT)
    gW2 = []
    grid = []
    for lookback in (10, 20, 40, 60):
        for target in (0.08, 0.10, 0.12, 0.15, 0.20):
            rm = vol_managed_returns(eq_c, master, target,
                                     lookback=lookback, cap=2.0)
            m_is = metrics_from_returns(rm[is_mask], master[is_mask])
            m_oos = metrics_from_returns(rm[oos_mask], master[oos_mask])
            grid.append((lookback, target, m_is, m_oos))
            e = {"name": f"W2_lb{lookback}_t{int(target*100)}", "group": "W2",
                 "IS": {k: round(v, 4) for k, v in m_is.items()},
                 "OOS": {k: round(v, 4) for k, v in m_oos.items()}}
            gW2.append(e)
            print(f"    lb={lookback:3d} target={target:.0%}  "
                  f"IS SR={m_is['sharpe_ratio']:6.3f} mdd={m_is['max_drawdown']:7.2%} | "
                  f"OOS SR={m_oos['sharpe_ratio']:6.3f} "
                  f"mdd={m_oos['max_drawdown']:7.2%}")
    best = max(grid, key=lambda t: t[2]["sharpe_ratio"])
    print(f"    ▶ IS 最优 = lb={best[0]} target={best[1]:.0%} "
          f"(IS SR={best[2]['sharpe_ratio']:.3f}) → "
          f"OOS SR={best[3]['sharpe_ratio']:.3f} "
          f"mdd={best[3]['max_drawdown']:.2%}")
    res_all["groups"]["W2_oos"] = {
        "grid": gW2,
        "is_selected": {"lookback": best[0], "target": best[1],
                        "IS_sharpe": round(best[2]["sharpe_ratio"], 4),
                        "OOS_sharpe": round(best[3]["sharpe_ratio"], 4),
                        "OOS_mdd": round(best[3]["max_drawdown"], 4)},
    }

    # ══════════ W3. lookback 敏感性 ══════════
    print("\n" + "=" * 92)
    print("W3. lookback 敏感性（固定 target=10%）")
    print("=" * 92)
    gW3 = []
    for lb in (5, 10, 15, 20, 30, 40, 60, 90, 120):
        rm = vol_managed_returns(eq_c, master, 0.10, lookback=lb, cap=2.0)
        m = metrics_from_returns(rm, master)
        e = record(f"W3_lb{lb}", "W3", m, f"lookback={lb}, target=10%")
        gW3.append(e)
    res_all["groups"]["W3_lookback"] = gW3

    # ══════════ W4. 只去杠杆（f ≤ 1）══════════
    print("\n" + "=" * 92)
    print("W4. 只做去杠杆（cap=1.0，f ≤ 1）—— 落地最容易的形态")
    print("=" * 92)
    gW4 = []
    for target in (0.06, 0.08, 0.10, 0.12, 0.15):
        for lb in (10, 20, 40):
            rm = vol_managed_returns(eq_c, master, target, lookback=lb, cap=1.0)
            m_is = metrics_from_returns(rm[is_mask], master[is_mask])
            m_oos = metrics_from_returns(rm[oos_mask], master[oos_mask])
            e = record(f"W4_t{int(target*100)}_lb{lb}", "W4", m_is,
                       f"IS: target={target:.0%} lb={lb}")
            e["IS_sharpe"] = round(m_is["sharpe_ratio"], 4)
            e["OOS_sharpe"] = round(m_oos["sharpe_ratio"], 4)
            e["OOS_mdd"] = round(m_oos["max_drawdown"], 4)
            e["OOS_return"] = round(m_oos["total_return"], 4)
            print(f"      OOS SR={m_oos['sharpe_ratio']:6.3f} "
                  f"mdd={m_oos['max_drawdown']:7.2%} "
                  f"ret={m_oos['total_return']:+7.2%}")
            gW4.append(e)
    res_all["groups"]["W4_deleverage_only"] = gW4

    res_all["trials"] = TRIALS
    res_all["n_trials"] = len(TRIALS)
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, args.out)
    with open(path, "w") as fh:
        json.dump(res_all, fh, ensure_ascii=False, indent=1, default=str)
    print(f"\n写入 {path}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
