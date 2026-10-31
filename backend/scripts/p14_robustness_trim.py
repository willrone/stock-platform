#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P14 稳健性检验：oos 窗口 past_s 全败，是否只是因为前几期 A 腿历史缺失退化成等权？

做法：把 past_s 与定稿（base_equal）的**日收益序列**都算出来，
在三个截断口径上比较夏普：
  全窗口 / 剔除前 6 期 / 剔除前 12 期
若 past_s 在所有截断口径上都低于 base_equal → 结论不依赖前期 fallback。
"""
from __future__ import annotations

import os
import sys

import numpy as np

BACKEND_DIR = "/Users/rone/Projects/stock-platform/backend"
sys.path.insert(0, os.path.join(BACKEND_DIR, "scripts"))
os.chdir(BACKEND_DIR)

import multi_asset_p14_oracle_bound as P14  # noqa: E402
import multi_asset_p13_asset_weight as P13  # noqa: E402
from multi_asset_leg_experiment import maxdd, sharpe  # noqa: E402


def nav_to_daily(nav):
    d = dict(nav)
    ds = [x for x, _ in nav]
    r = []
    for k in range(1, len(ds)):
        r.append(d[ds[k]] / d[ds[k - 1]] - 1.0)
    return np.array(r), ds[1:]


def build_nav(w, sched, a_sched):
    calendar = w["calendar"]
    idx = P13.build_price_index(w["prices"])

    def weight_fn(signal_date, _i):
        cw = sched.get(signal_date)
        if not cw:
            return None
        s = sum(cw.values())
        return {c: v / s for c, v in cw.items() if v > 0}

    targets, wmeta = P13.build_targets_code(calendar, weight_fn, P14.REB_EVERY)
    s_nav, rec = P13.simulate_portfolio(calendar, idx, targets)
    exec_of = {row["date"]: row["exec"] for row in wmeta["weight_log"]}
    if all(abs(a_sched.get(k, P14.A_LEG_W) - P14.A_LEG_W) < 1e-9 for k in a_sched):
        comb = P14.merge_by_date([w["a_nav"], s_nav], [P14.A_LEG_W, 1.0 - P14.A_LEG_W])
    else:
        comb = P14.merge_variable(w["a_nav"], s_nav, a_sched, exec_of)
    return comb, s_nav


def main():
    for window, cur, rnd in (("oos", P14.CURRENT_OOS, P14.RANDOM_BEST_OOS),
                              ("main", P14.CURRENT_MAIN, P14.RANDOM_BEST_MAIN)):
        w = P14.load_window(window)
        codes = w["codes"]

        eq = {r["date"]: {c: 1.0 / len(codes) for c in codes}
              for r in P14._rebal_dates(w["calendar"])}
        base_comb, base_sleeve = build_nav(w, eq, {k: P14.A_LEG_W for k in eq})

        print(f"\n===== {window} =====")
        print(f"  定稿 组合夏普 {sharpe(base_comb):.4f} | 袖套夏普 {sharpe(base_sleeve):.4f}")
        for mode in ("past_s", "past_j"):
            sc = P14.oracle_schedule(w, mode)
            comb, sleeve = build_nav(w, sc["sched"], sc["a_sched"])
            rb, _ = nav_to_daily(base_comb)
            rm, _ = nav_to_daily(comb)
            n = min(len(rb), len(rm))
            rb, rm = rb[:n], rm[:n]
            print(f"  {mode}: 全窗口 组合 {sharpe(comb):.4f} 袖套 {sharpe(sleeve):.4f}")
            for trim_lbl, cut in (("剔除前 6 期", 6 * P14.REB_EVERY),
                                  ("剔除前 12 期", 12 * P14.REB_EVERY)):
                if cut >= n - 120:
                    print(f"    {trim_lbl}: 样本不足({n-cut} 天)，跳过")
                    continue
                sb = sharpe({"nav": base_comb}) if False else None
                # 子窗口夏普：直接用日收益重算
                from multi_asset_leg_experiment import sharpe as _sh

                def _sharr(x):
                    mu = x.mean(); sd = x.std(ddof=1)
                    return mu / sd * np.sqrt(252) if sd > 0 else 0.0
                print(f"    {trim_lbl} ({n-cut} 天): "
                      f"past { _sharr(rm[cut:]):.4f} vs 定稿 { _sharr(rb[cut:]):.4f} "
                      f"→ {'❌ 仍败' if _sharr(rm[cut:]) < _sharr(rb[cut:]) else '✅ 翻盘'}")
        print(f"  参考：定稿 {cur} | 随机族最优 {rnd}")


if __name__ == "__main__":
    main()
