#!/usr/bin/env python3
"""中性化为什么有效？——机制解剖。

## 待检验的三个机制假设
**M1 成交截断假设（最可能）**
引擎有可成交性检查：单笔买入额 ≤ 当日成交额 × 参与率 0.5%。
原始 42 因子信号带小盘/低流动性倾向（信号 vs size 秩相关 −0.26），
→ 目标仓位买不进去 → **实际组合与目标组合不一致**。
中性化 size+illiq 后目标更可成交 → 实际组合终于等于目标组合。
若成立：基线的「实际规模暴露」应该比「目标规模暴露」更接近市场。

**M2 剔除坏暴露假设**
42 因子里 MIN/MAX/STD/CORR/CORD 都是反转 + 低波类，
这些因子在低流动性小票上取值极端 → 复合分被极端值主导。
中性化相当于把这些票拉回可比。

**M3 纯粹的风格 tilt 假设**
中性化就是去规模 beta，收益提升全部来自 beta 调整。

## 检验
  T1. 成交截断：基线 vs 中性化 的**实际**成交额 / **目标**成交额（填充率）
  T2. 目标持仓 vs 实际持仓的规模暴露对比
  T3. 目标组合（不考虑可成交性）与可成交组合的收益差
  T4. 去掉流动性上限（tradability off）后中性化还有效吗？
      → 若失效，说明是 M1；若仍有效，说明是 M2/M3
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

OUT_DIR = os.path.join(ROOT, "data", "backtest")
TRIALS: List[Dict[str, Any]] = []


def record(name, group, res, note=""):
    e = {
        "name": name, "group": group,
        "sharpe_ratio": round(float(res["sharpe_ratio"]), 4),
        "total_return": round(float(res["total_return"]), 4),
        "max_drawdown": round(float(res["max_drawdown"]), 4),
        "total_trades": int(res["total_trades"]),
        "volatility": round(float(res.get("volatility", 0.0)), 4),
        "note": note,
    }
    TRIALS.append(e)
    print(f"  {name[:40]:42s} SR={e['sharpe_ratio']:6.3f} "
          f"ret={e['total_return']:+8.2%} mdd={e['max_drawdown']:7.2%} "
          f"n={e['total_trades']:5d}  {note[:36]}", flush=True)
    return e


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="methods_mechanism.json")
    ap.add_argument("--topk", type=int, default=60)
    ap.add_argument("--ndrop", type=int, default=5)
    args = ap.parse_args()

    t0 = time.time()
    print("=" * 88)
    print("机制解剖 · 中性化为什么有效")
    print("=" * 88)

    cache = M.load_cached_mats()
    mats, trad = cache["mats"], cache["trad"]
    CFG = L.SimConfig(topk=args.topk, n_drop=args.ndrop, lag=1,
                      rebalance_days="weekly",
                      max_position_size=(1 - 0.05) / args.topk)
    s42, _ = L.build_42f_score(
        mats, panel_aligned=True,
        panel_dir=os.path.join(L.QLIB_LIVE, "features", "day"))
    amount = mats["close"] * mats["volume"]
    with np.errstate(divide="ignore", invalid="ignore"):
        log_amt = np.log(np.where(amount > 0, amount, np.nan))
    size = pd.DataFrame(log_amt).rolling(20, min_periods=5).mean().to_numpy()
    illiq = M.amihud_illiq(mats["close"], mats["volume"], 20)
    neut = neutralize_score(s42, {"size": size, "illiq": illiq})
    master = mats["master"]
    print(f"[setup] ready ({time.time()-t0:.0f}s)", flush=True)

    res_all: Dict[str, Any] = {
        "meta": {
            "window": f"{master[0].date()}→{master[-1].date()}",
            "n_codes": len(mats["codes"]),
            "n_trading_days": len(master),
            "caliber": {"lag": 1, "enable_tradability": True,
                        "topk": args.topk, "n_drop": args.ndrop,
                        "commission": 0.0003, "slippage": 0.0001,
                        "board_lot": 100, "cash_reserve": 0.05,
                        "rebalance": "weekly(周五)"},
            "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        },
        "groups": {},
    }

    # ══════════ T1 + T2. 成交截断 & 实际 vs 目标暴露 ══════════
    print("\n" + "=" * 88)
    print("T1/T2. 成交截断：目标持仓的可成交性 & 实际 vs 目标规模暴露")
    print("=" * 88)
    can_buy = trad["can_buy"]
    max_buy = trad["max_buy_amount"]
    reb = (master.dayofweek == 4)          # 周频调仓日
    out_rows = []
    for nm, sc in (("base", s42), ("neut", neut)):
        # 目标持仓（每周五，取 top60）
        n_blocked, n_target, cap_limited = 0, 0, 0
        tgt_size, actual_size = [], []
        for t in np.where(reb)[0]:
            row = sc[t]
            valid = np.where(np.isfinite(row))[0]
            if len(valid) < 60:
                continue
            order = valid[np.argsort(-row[valid], kind="mergesort")][:60]
            n_target += len(order)
            blocked = ~can_buy[t, order]
            n_blocked += int(blocked.sum())
            # 流动性上限：单只最多 max_buy；目标单只市值 = 组合/60
            tgt_val = 1_000_000.0 * (0.95 / 60)
            capped = (max_buy[t, order] > 0) & (max_buy[t, order] < tgt_val)
            cap_limited += int(capped.sum())
            tgt_size.append(np.nanmean(size[t, order]))
            # 实际能买到的（can_buy 且未被上限截断）
            act = order[~blocked]
            if len(act):
                actual_size.append(np.nanmean(size[t, act]))
        out_rows.append({
            "variant": nm,
            "target_positions": n_target,
            "blocked_by_tradability_pct": round(100.0 * n_blocked / max(n_target, 1), 2),
            "capped_by_liquidity_pct": round(100.0 * cap_limited / max(n_target, 1), 2),
            "target_mean_size_exposure": round(float(np.nanmean(tgt_size)), 4),
            "tradable_mean_size_exposure": round(float(np.nanmean(actual_size)), 4),
            "market_mean_size_exposure": round(
                float(np.nanmean([np.nanmean(size[t]) for t in np.where(reb)[0]])), 4),
        })
        r = out_rows[-1]
        print(f"    {nm:6s} 目标持仓 {n_target}  "
              f"不可买 {r['blocked_by_tradability_pct']:.1f}%  "
              f"流动性截断 {r['capped_by_liquidity_pct']:.1f}%  "
              f"目标规模暴露 {r['target_mean_size_exposure']:.4f} vs "
              f"可成交部分 {r['tradable_mean_size_exposure']:.4f} vs "
              f"全市场 {r['market_mean_size_exposure']:.4f}")
    res_all["groups"]["T1_fill_and_exposure"] = out_rows

    # ══════════ T3/T4. 关闭可成交性后中性化还有效吗？══════════
    print("\n" + "=" * 88)
    print("T3/T4. 关闭可成交性检查（--no-tradability 口径）后重测")
    print("=" * 88)
    trad_off = {
        "can_buy": np.ones_like(can_buy),
        "can_sell": np.ones_like(can_buy),
        "max_buy_amount": np.full_like(max_buy, np.inf),
    }
    gT = []
    for tag, tr in (("tradability_ON", trad), ("tradability_OFF", trad_off)):
        for nm, sc in (("base", s42), ("neut", neut)):
            r = L.simulate(sc, mats, tr, CFG)
            e = record(f"T4_{tag}_{nm}", "T4", r, f"{tag} / {nm}")
            gT.append(e)
    base_on = next(e for e in gT if e["name"] == "T4_tradability_ON_base")
    neut_on = next(e for e in gT if e["name"] == "T4_tradability_ON_neut")
    base_off = next(e for e in gT if e["name"] == "T4_tradability_OFF_base")
    neut_off = next(e for e in gT if e["name"] == "T4_tradability_OFF_neut")
    d_on = neut_on["sharpe_ratio"] - base_on["sharpe_ratio"]
    d_off = neut_off["sharpe_ratio"] - base_off["sharpe_ratio"]
    print(f"    → 可成交性开：Δ夏普 = {d_on:+.4f}")
    print(f"    → 可成交性关：Δ夏普 = {d_off:+.4f}")
    print(f"    → {'M1 成立（是成交截断机制）' if abs(d_off) < abs(d_on) * 0.5 else 'M1 不成立（是风格/信号质量机制）'}")
    res_all["groups"].setdefault("T4", []).append({
        "delta_with_tradability": round(d_on, 4),
        "delta_without_tradability": round(d_off, 4),
        "verdict": ("M1_fill_constraint" if abs(d_off) < abs(d_on) * 0.5
                     else "M2_M3_style_or_signal_quality"),
    })

    # ══════════ T5. 目标组合（不受可成交性影响）与实际组合的收益差 ══════════
    print("\n" + "=" * 88)
    print("T5. 收益来自「目标组合」还是「实际成交组合」？")
    print("=" * 88)
    # 用模拟器自身的持仓快照不方便 → 用「关闭可成交性」的结果近似目标组合
    print("    （T4 的 tradability_OFF 一行即「目标组合」的理论表现）")
    print(f"    基线: 可成交 {base_on['sharpe_ratio']:.3f} / "
          f"不可成交(目标) {base_off['sharpe_ratio']:.3f}")
    print(f"    中性: 可成交 {neut_on['sharpe_ratio']:.3f} / "
          f"不可成交(目标) {neut_off['sharpe_ratio']:.3f}")
    res_all["groups"]["T5_target_vs_realized"] = {
        "base_trad_on": base_on["sharpe_ratio"],
        "base_trad_off": base_off["sharpe_ratio"],
        "neut_trad_on": neut_on["sharpe_ratio"],
        "neut_trad_off": neut_off["sharpe_ratio"],
    }

    res_all["trials"] = TRIALS
    res_all["n_trials"] = len(TRIALS)
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, args.out)
    with open(path, "w") as fh:
        json.dump(res_all, fh, ensure_ascii=False, indent=1, default=str)
    print(f"\n写入 {path}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
