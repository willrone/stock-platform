#!/usr/bin/env python3
"""回归测试：L3.simulate 的**顺序不变性**（持仓顺序不同 ⇒ 净值必须逐位相同）。

背景：P6 第一次跑 main（held 用 P3 的 by_class_raw 顺序）得 w=0.50 → 1.4092，
改成 P5 固定集顺序后得 1.4147。两者持仓**集合完全相同**，只有类内顺序不同。
若 simulate 顺序敏感 → 定稿的持仓顺序必须与 P3 复现门对齐，否则数值不可复现。

跑法：cd backend && env -u PYTHONPATH .venv-py313/bin/python3 scripts/p6_order_diag.py
"""
from __future__ import annotations

import json
import os
import sys

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT = os.path.abspath(os.path.join(BACKEND_DIR, ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)
sys.path.insert(0, os.path.join(BACKEND_DIR, "scripts"))

import multi_asset_l3_momentum as L3  # noqa: E402
import multi_asset_p2 as P2  # noqa: E402
import multi_asset_p3_bond as P3  # noqa: E402
from multi_asset_leg_experiment import (  # noqa: E402
    END, START, load_etf_universe, merge_by_date, sharpe,
)

CLASSES = ["gold", "bond", "commodity"]

HELD_P3_MAIN = {            # P3 main baseline_held 的顺序
    "gold": ["518880", "159934", "159937"],
    "bond": ["511260", "511010", "511020"],
    "commodity": ["159985", "159981", "159980"],
}
HELD_P5 = {                  # P5 文档里写的顺序
    "gold": ["518880", "159934", "159937"],
    "bond": ["511260", "511010", "511020"],
    "commodity": ["159980", "159985", "159981"],
}


def main() -> None:
    start, end = START, END
    s_str, e_str = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

    # A 股腿（读缓存）
    nav_cache = os.path.join(ROOT, "data", "backtest", "_a_share_nav_main.json")
    with open(nav_cache, encoding="utf-8") as fh:
        cached = json.load(fh)
    a_nav = [(d, float(v)) for d, v in cached["a_share_nav"]]

    pool = P3.build_pool(start)
    universe = load_etf_universe()
    by_class_raw = L3.candidates_by_class(universe, start, end)
    base_codes = sorted({c for lst in by_class_raw.values() for c in lst})
    all_codes = sorted(set(base_codes) | {r["code"] for r in pool})
    prices = L3.load_prices(all_codes)
    calendar = L3.build_calendar(prices, base_codes, s_str, e_str)
    print(f"[diag] 日历 {len(calendar)} 天 {calendar[0]} → {calendar[-1]}", flush=True)

    out = {}
    for label, held in (("P3顺序", HELD_P3_MAIN), ("P5顺序", HELD_P5)):
        tg, tmeta = P3.build_targets(calendar, held, P3.fixed_weights_for(CLASSES))
        c_of = {c: cls for cls, lst in held.items() for c in lst}
        sleeve_nav, srec = L3.simulate(
            calendar, prices, tg, c_of, {c: 1.0 / len(CLASSES) for c in CLASSES})
        comb = merge_by_date([a_nav, sleeve_nav], [0.5, 0.5])
        # 目标权重表：同一调仓日、同一代码，权重是否完全一致
        first_target = list(tg.values())[0]
        out[label] = {
            "held": held,
            "sleeve_sharpe": round(sharpe(sleeve_nav), 6),
            "sleeve_last": round(sleeve_nav[-1][1], 6),
            "combined_sharpe_w050": round(sharpe(comb), 6),
            "turnover": round(srec["turnover_ratio"], 6),
            "commission": round(srec.get("commission", 0.0), 4),
            "first_target_snapshot": {k: round(v, 6) for k, v in sorted(first_target.items())},
            "n_rebalances": tmeta["n_rebalances"],
        }
        print(f"[diag] {label}: 袖套夏普 {out[label]['sleeve_sharpe']:.6f}  "
              f"w050 组合夏普 {out[label]['combined_sharpe_w050']:.6f}  "
              f"换手 {out[label]['turnover']:.4f}x  "
              f"佣金 {out[label]['commission']:.2f}", flush=True)
        print(f"[diag]   首个调仓日目标权重: {out[label]['first_target_snapshot']}", flush=True)

    same_target = (out["P3顺序"]["first_target_snapshot"]
                   == out["P5顺序"]["first_target_snapshot"])
    same_sharpe = (out["P3顺序"]["sleeve_sharpe"] == out["P5顺序"]["sleeve_sharpe"])
    print(f"\n[diag] 目标权重表相同: {same_target}", flush=True)
    print(f"[diag] 袖套夏普相同: {same_sharpe}  "
          f"(差 {abs(out['P3顺序']['sleeve_sharpe'] - out['P5顺序']['sleeve_sharpe']):.8f})",
          flush=True)
    print(f"[diag] w=0.50 组合夏普: P3顺序 {out['P3顺序']['combined_sharpe_w050']:.6f}  "
          f"P5顺序 {out['P5顺序']['combined_sharpe_w050']:.6f}  "
          f"(差 {abs(out['P3顺序']['combined_sharpe_w050'] - out['P5顺序']['combined_sharpe_w050']):.6f})",
          flush=True)

    # ── 回归断言：目标权重表相同 ⇒ 净值必须逐位相同（顺序无关性）──
    if not same_target:
        raise SystemExit("[diag] ❌ 目标权重表不同，无法检验顺序不变性（配置有误）")
    if not same_sharpe:
        raise SystemExit("[diag] ❌ 回归失败：持仓顺序不同导致净值不同"
                         "（L3.simulate 的现金约束必须 pro-rata 缩放）")
    if out["P3顺序"]["combined_sharpe_w050"] != out["P5顺序"]["combined_sharpe_w050"]:
        raise SystemExit("[diag] ❌ 回归失败：组合夏普仍随持仓顺序变化")

    res = {"diagnosis": "class_order_sensitivity",
           "regression_passed": True,
           "results": out,
           "same_target_weights": same_target,
           "same_sleeve_sharpe": same_sharpe,
           "combined_sharpe_w050": out["P3顺序"]["combined_sharpe_w050"]}
    p = os.path.join(ROOT, "data", "backtest", "p6_order_diag.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2)
    print(f"[diag] ✓ 已写入 {p}", flush=True)


if __name__ == "__main__":
    main()
