#!/usr/bin/env python3
"""P2 机制归因：为什么逆波动率配权改善了袖套、却没改善组合。

## 要回答的问题

P2b 出现一个表面矛盾：
- **长历史（袖套单独，7.7 年）**：逆波动率配权大幅改善袖套（Δ夏普 +0.21~+1.00，
  12 个独立子窗口 8~11/12 更好）
- **组合口径（A 股腿 50% + 袖套 50%，两个短窗口）**：没有任何配置在两窗口都优于等权

假说：**逆波动率配权把权重压向低波资产（bond），袖套波动降一半、夏普升，
但袖套与 A 股腿的日收益相关性 ρ 同时升高 → 组合层面的分散化收益被抵消。**

本脚本把 ρ 拆开：
1. 每个资产类**单独**与 A 股腿的 ρ、波动、夏普
2. 逆波动率配权实际给出的**类权重时间序列**（平均/区间）
3. 用「收益 / 波动 / ρ」三要素解释每个配置的组合夏普

## 用法
    cd backend && env -u PYTHONPATH .venv-py313/bin/python3 \
        scripts/multi_asset_p2_attrib.py
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Tuple

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT = os.path.abspath(os.path.join(BACKEND_DIR, ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)
sys.path.insert(0, os.path.join(BACKEND_DIR, "scripts"))

import multi_asset_l3_momentum as L3  # noqa: E402
import multi_asset_p2 as P2  # noqa: E402
from multi_asset_leg_experiment import (  # noqa: E402
    END,
    OOS_END,
    OOS_START,
    START,
    load_etf_universe,
    sharpe,
    total_ret,
)

CLASSES = ["gold", "bond", "commodity"]


def main() -> None:
    out: Dict[str, Any] = {}
    universe = load_etf_universe()

    for tag, (start, end) in {
        "main": (START, END),
        "oos": (OOS_START, OOS_END),
    }.items():
        s_str, e_str = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
        cache = os.path.join(ROOT, "data", "backtest", f"_a_share_nav_{tag}.json")
        with open(cache, encoding="utf-8") as fh:
            a_nav = [(d, float(v)) for d, v in json.load(fh)["a_share_nav"]]

        by_class = L3.candidates_by_class(universe, start, end)
        held = {c: lst[:P2.K_IN_CLASS] for c, lst in by_class.items()}
        all_codes = sorted({c for lst in by_class.values() for c in lst})
        prices = L3.load_prices(all_codes)
        calendar = L3.build_calendar(prices, all_codes, s_str, e_str)

        print(f"\n===== {tag}  {calendar[0]} → {calendar[-1]} ({len(calendar)} 天) =====",
              flush=True)
        rec: Dict[str, Any] = {"classes": {}}

        # 每个类单独：买入持有袖套 → 波动 / 夏普 / 与 A 股腿的 ρ
        class_navs: Dict[str, List[Tuple[str, float]]] = {}
        for c in CLASSES:
            codes = held.get(c) or []
            if not codes:
                continue
            tg, _ = P2.build_weighted_targets(
                calendar, {c: codes}, P2.fixed_weights({c: codes}))
            nav, _ = L3.simulate(calendar, prices, tg, {x: c for x in codes}, {c: 1.0})
            class_navs[c] = nav
            vol = P2.nav_vol(nav)
            rho = P2.nav_corr(nav, a_nav)
            rec["classes"][c] = {
                "codes": codes,
                "sleeve_sharpe": round(sharpe(nav), 4),
                "sleeve_vol": round(vol, 4),
                "sleeve_return": round(total_ret(nav), 4),
                "corr_with_a_share": rho,
            }
            print(f"  {c:10s} {codes} 夏普 {sharpe(nav):.4f}  波动 {vol:.2%}  "
                  f"收益 {total_ret(nav):+.2%}  ρ(类,A) {rho}", flush=True)

        # 逆波动率配权实际给出的类权重
        wseries: Dict[str, List[float]] = {c: [] for c in CLASSES}
        fn = P2.rvol_weights(prices, held, 120, True)
        fn_unc = P2.rvol_weights(prices, held, 120, False)
        for i in range(1, len(calendar)):
            if (i - 1) % P2.REB_EVERY != 0:
                continue
            w = fn(calendar[i - 1])
            for c in CLASSES:
                wseries[c].append(w.get(c, 0.0))
        wseries_unc = {c: [] for c in CLASSES}
        for i in range(1, len(calendar)):
            if (i - 1) % P2.REB_EVERY != 0:
                continue
            w = fn_unc(calendar[i - 1])
            for c in CLASSES:
                wseries_unc[c].append(w.get(c, 0.0))

        def summarize(ws):
            return {c: {"mean": round(sum(v) / len(v), 4) if v else None,
                        "min": round(min(v), 4) if v else None,
                        "max": round(max(v), 4) if v else None}
                    for c, v in ws.items() if v}

        rec["rvol120_cap_weights"] = summarize(wseries)
        rec["rvol120_unc_weights"] = summarize(wseries_unc)
        print("  逆波动率配权(cap) 类权重均值: "
              f"{json.dumps(rec['rvol120_cap_weights'], ensure_ascii=False)}", flush=True)
        print("  逆波动率配权(无cap) 类权重均值: "
              f"{json.dumps(rec['rvol120_unc_weights'], ensure_ascii=False)}", flush=True)

        # 相关性分解：把袖套的 ρ 拆成「各类的 ρ 加权」
        out[tag] = rec

    path = os.path.join(ROOT, "data", "backtest", "multi_asset_p2_attrib.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(f"\n✓ 已写入 {path}", flush=True)


if __name__ == "__main__":
    main()
