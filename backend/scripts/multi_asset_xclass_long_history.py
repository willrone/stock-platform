#!/usr/bin/env python3
"""P1 长历史压力测试 —— 在 7.7 年独立数据上判定跨类动量是否有真信号。

## 为什么需要这一步

P1 两个窗口的判定里，只有 `xclass_mom_L250_top2` 通过了全部三条判据
（两窗口都优于随机对照均值 + 两窗口都优于基线 + 换手没失控）。

**但它是典型的参数挖掘产物**：
- 在推导窗口（main）它是同族**最差**的（1.3958，而 L60 是 1.5037）
- 只有在样本外窗口它才最好（1.1457，而 L60 是 0.8101）

也就是说，**要选中它就必须用样本外数据选参数** —— 这本身就作废了整场实验。
按推导窗口选，正确答案是 L60 系（main 最高），而 L60 系在样本外全面崩溃。

所以本脚本的作用不是"再挑一个赢家"，而是**对唯一那个疑似赢家做独立审判**：
如果 L250_top2 真的有信号，它应该在**完全独立的 7.7 年数据**上依然成立；
如果它只是噪声，在长历史上必然露出马脚（滚动窗口胜率回到 50% 附近）。

## 长历史口径

- 窗口：2019-01 → 2026-10（与 `multi_asset_long_history.py` 同一区间）
- 候选池：**窗口前 120 个交易日**成交额排序 → 冻结整个窗口（避免前视）
- 对照：类间固定等权 + 每月再平衡（= 定稿配置的袖套口径）
- 度量：滚动 504 日 / 252 日夏普胜率 + 12 个不重叠独立子窗口
- 注意：长历史是**袖套单独**（没有 A 股腿），所以只能比袖套夏普

## 用法
    cd backend && env -u PYTHONPATH .venv-py313/bin/python3 \
        scripts/multi_asset_xclass_long_history.py
"""
from __future__ import annotations

import json
import os
import statistics as st
import sys
from typing import Any, Dict, List, Tuple

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT = os.path.abspath(os.path.join(BACKEND_DIR, ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)
sys.path.insert(0, os.path.join(BACKEND_DIR, "scripts"))

import multi_asset_l3_momentum as L  # noqa: E402
import multi_asset_long_history as LH  # noqa: E402
import multi_asset_xclass as X  # noqa: E402
from multi_asset_leg_experiment import (  # noqa: E402
    load_etf_universe,
    maxdd,
    sharpe,
    total_ret,
)

LONG_START = "2019-01-01"
LONG_END = "2026-10-30"

CLASSES = ["gold", "bond", "commodity"]
K_IN_CLASS = 3
REB_EVERY = 21
SEEDS = (20261030, 7, 99, 1234, 555, 8888)

#: 待审配置：main 窗口最优系 + 唯一"通过全部判据"的那个
CANDIDATES = [
    ("baseline_fixed_equal", None),      # 基线：类间固定等权 + 每月再平衡
    ("mom_L60_top2", dict(lookback=60, top_m=2)),
    ("mom_L60_weighted", dict(lookback=60, top_m=None, weighted=True)),
    ("mom_L250_top2", dict(lookback=250, top_m=2)),   # ← 唯一"通过判据"的那个
    ("mom_L250_weighted", dict(lookback=250, top_m=None, weighted=True)),
]


def main() -> None:
    universe = load_etf_universe()
    picks = LH.pick_point_in_time(universe, CLASSES, LONG_START, LONG_END)
    print(f"[xlh] 候选（窗口前 {LH.LIQ_LOOKBACK_DAYS} 交易日流动性冻结）: "
          f"{json.dumps(picks, ensure_ascii=False)}", flush=True)

    # 类内固定 top-3（与 P1 主实验一致）
    held = {c: lst[:K_IN_CLASS] for c, lst in picks.items()}
    print(f"[xlh] 类内固定持仓: {json.dumps(held, ensure_ascii=False)}", flush=True)

    all_codes = sorted({c for lst in picks.values() for c in lst})
    prices = L.load_prices(all_codes)
    cal = L.build_calendar(prices, all_codes, LONG_START, LONG_END)
    print(f"[xlh] 交易日历 {len(cal)} 天  {cal[0]} → {cal[-1]}", flush=True)
    if len(cal) < 1000:
        raise SystemExit("[xlh] ❌ 历史长度不足，退出")

    class_of = {c: cls for cls, lst in picks.items() for c in lst}
    cw = {c: 1.0 / len(CLASSES) for c in CLASSES}

    store: Dict[str, Any] = {
        "window": {"start": cal[0], "end": cal[-1], "n_days": len(cal)},
        "candidates": picks,
        "held_per_class": held,
    }

    # ── 基线：类间固定等权 + 每月再平衡 ──
    tg_base, _ = L.build_rotation_targets(cal, prices, held, cw, 60, 99, REB_EVERY, "all")
    base_nav, base_cost = L.simulate(cal, prices, tg_base, class_of, cw)
    base_sharpe = sharpe(base_nav)
    print(f"\n[xlh] 基线（类间固定等权 + 每月再平衡）: 夏普 {base_sharpe:.4f}  "
          f"收益 {total_ret(base_nav):+.2%}  回撤 {maxdd(base_nav):.2%}  "
          f"换手 {base_cost['turnover_ratio']:.1f}x", flush=True)

    results: Dict[str, Any] = {}
    navs: Dict[str, List[Tuple[str, float]]] = {}
    results["baseline_fixed_equal"] = {
        "sleeve_sharpe": round(base_sharpe, 4),
        "sleeve_return": round(total_ret(base_nav), 4),
        "sleeve_maxdd": round(maxdd(base_nav), 4),
        "cost": base_cost,
    }

    def add_cfg(label: str, targets: Dict[str, Dict[str, float]], meta: Dict[str, Any]) -> None:
        nav, cost = L.simulate(cal, prices, targets, class_of, cw)
        if not nav:
            print(f"  {label}: ❌ 无净值", flush=True)
            return
        s = sharpe(nav)
        navs[label] = nav
        r24 = LH.rolling_sharpe_win(base_nav, nav, 504)
        r12 = LH.rolling_sharpe_win(base_nav, nav, 252)
        w24 = LH.rolling_window_win(base_nav, nav, 504)
        results[label] = {
            "sleeve_sharpe": round(s, 4),
            "sleeve_return": round(total_ret(nav), 4),
            "sleeve_maxdd": round(maxdd(nav), 4),
            "delta_sharpe_vs_base": round(s - base_sharpe, 4),
            "rolling_sharpe_504d": r24,
            "rolling_sharpe_252d": r12,
            "rolling_return_504d": w24,
            "cost": cost,
            **meta,
        }
        print(f"  {label:26s} 夏普 {s:.4f} (Δ基线 {s-base_sharpe:+.4f})  "
              f"收益 {total_ret(nav):+.2%}  回撤 {maxdd(nav):.2%}  "
              f"| 滚动504日夏普胜率 {r24['sharpe_win_rate']}  "
              f"Δ {r24['mean_sharpe_diff']:+.4f}  "
              f"滚动收益胜率 {w24['win_rate']}  "
              f"换手 {cost['turnover_ratio']:.1f}x", flush=True)

    # ── 待审配置 ──
    print("\n[xlh] === 待审：跨类动量 ===", flush=True)
    for label, cfg in CANDIDATES:
        if cfg is None:
            continue
        tg, meta = X.build_xclass_targets(
            cal, prices, held, cfg["lookback"], cfg.get("top_m"),
            cfg.get("weighted", False), REB_EVERY, "mom")
        add_cfg(label, tg, {**meta, "rule": "xclass_momentum"})

    # ── 随机对照 ──
    print("\n[xlh] === 随机对照（6 种子，同频率同持仓类数）===", flush=True)
    rand_sharpes: Dict[str, List[float]] = {"top2": [], "weighted": []}
    for grp, (top_m, weighted) in {"top2": (2, False), "weighted": (None, True)}.items():
        for seed in SEEDS:
            tg, meta = X.build_xclass_targets(
                cal, prices, held, 60, top_m, weighted, REB_EVERY, "rand", seed=seed)
            nav, cost = L.simulate(cal, prices, tg, class_of, cw)
            if not nav:
                continue
            s = sharpe(nav)
            navs[f"rand_{grp}_s{seed}"] = nav
            rand_sharpes[grp].append(s)
            r24 = LH.rolling_sharpe_win(base_nav, nav, 504)
            results[f"rand_{grp}_s{seed}"] = {
                "sleeve_sharpe": round(s, 4),
                "sleeve_return": round(total_ret(nav), 4),
                "sleeve_maxdd": round(maxdd(nav), 4),
                "delta_sharpe_vs_base": round(s - base_sharpe, 4),
                "rolling_sharpe_504d": r24,
                "cost": cost,
                **meta,
            }
            print(f"  rand_{grp}_s{seed:<9d} 夏普 {s:.4f} (Δ基线 {s-base_sharpe:+.4f})  "
                  f"滚动504日夏普胜率 {r24['sharpe_win_rate']}  换手 {cost['turnover_ratio']:.1f}x",
                  flush=True)

    for grp, vals in rand_sharpes.items():
        if vals:
            print(f"  → rand_{grp}: 均值 {st.mean(vals):.4f}  "
                  f"[{min(vals):.4f}, {max(vals):.4f}]  极差 {max(vals)-min(vals):.4f}")

    # ── 12 个不重叠独立子窗口 ──
    print("\n[xlh] === 12 个不重叠独立子窗口（对基线的夏普差）===", flush=True)
    n_sub = 12
    # ⚠️ 最后一个边界会越界（int(n_sub * len / n_sub) == len(cal)）
    bounds = [cal[min(int(i * len(cal) / n_sub), len(cal) - 1)] for i in range(n_sub + 1)]
    navs["baseline_fixed_equal"] = base_nav
    sub_rows: Dict[str, List[Dict[str, Any]]] = {}
    for label, nav in navs.items():
        if label == "baseline_fixed_equal":
            continue
        rows = []
        for i in range(n_sub):
            d0, d1 = bounds[i], bounds[i + 1]
            seg_b = [(d, v) for d, v in nav if d0 <= d < d1]
            seg_a = [(d, v) for d, v in base_nav if d0 <= d < d1]
            if len(seg_b) < 60 or len(seg_a) < 60:
                continue
            rows.append({
                "i": i,
                "start": seg_b[0][0], "end": seg_b[-1][0],
                "sharpe_base": round(sharpe(seg_a), 4),
                "sharpe_cfg": round(sharpe(seg_b), 4),
                "sharpe_delta": round(sharpe(seg_b) - sharpe(seg_a), 4),
            })
        sub_rows[label] = rows
        if rows:
            better = sum(1 for r in rows if r["sharpe_delta"] > 0)
            avg = sum(r["sharpe_delta"] for r in rows) / len(rows)
            print(f"  {label:26s} 夏普更好 {better}/{len(rows)}  平均 Δ夏普 {avg:+.4f}")

    store["results"] = results
    store["sub_windows"] = sub_rows
    store["random_summary"] = {
        g: {"n": len(v), "mean": round(st.mean(v), 4),
            "min": round(min(v), 4), "max": round(max(v), 4)}
        for g, v in rand_sharpes.items() if v
    }
    out = os.path.join(ROOT, "data", "backtest", "multi_asset_xclass_long_history.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(store, fh, ensure_ascii=False, indent=2)
    print(f"\n[xlh] ✓ 已写入 {out}", flush=True)


if __name__ == "__main__":
    main()
