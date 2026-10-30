#!/usr/bin/env python3
"""样本外验证 —— 主窗口推荐配置在 2021-05~2023-05 的独立检验。

## 为什么必须做

`MULTI_ASSET_LEG_2026-10-30.md` 的主窗口结论（2023-05-08~2026-05-08）：
最优配置「A股腿 50% + 黄金/债券/商品/红利 各 12.5%」夏普 1.2007 (+38.6%)。

**但这条规则是从主窗口推导出来的。** 数字漂亮不等于规则成立 —— 必须拿
一个**未参与推导**的窗口独立验证。

## 窗口选择

| 窗口 | 区间 | 角色 |
|---|---|---|
| main | 2023-05-08 ~ 2026-05-08 | 推导窗口（规则来自这里） |
| **oos** | **2021-05-11 ~ 2023-05-08** | **样本外**（规则未见过这里） |

OOS 的起点 2021-05-11 是硬约束：25 因子策略的面板数据从这天起，
在此之前 17 个因子为 NaN，策略退化为 23 公式版 → 那不是同一个策略。

OOS 的这 2 年恰好是 **A 股弱势期**（2021H2 暴跌 + 2022 全年熊市），
是检验「分散化在弱势市场是否真的有效」的最严苛场景。

## 验证什么

在**两个窗口上跑完全相同的一组配置**，看主窗口的最优配置在 OOS 上是否
仍然成立、方向是否一致。

## 用法

    cd backend && env -u PYTHONPATH .venv-py313/bin/python3 scripts/multi_asset_oos_validation.py
"""
from __future__ import annotations

import json
import math
import os
import statistics
import sys
from typing import Any, Dict, List, Tuple

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT = os.path.abspath(os.path.join(BACKEND_DIR, ".."))

MAIN_NAVS = os.path.join(ROOT, "data", "backtest", "multi_asset_navs.json")
OOS_NAVS = os.path.join(ROOT, "data", "backtest", "multi_asset_navs_oos.json")
OUT_PATH = os.path.join(ROOT, "data", "backtest", "multi_asset_oos_validation.json")

Nav = List[Tuple[str, float]]


def sharpe(nav: Nav) -> float:
    if len(nav) < 3:
        return 0.0
    rets = []
    for i in range(1, len(nav)):
        p0, p1 = nav[i - 1][1], nav[i][1]
        if p0 and p0 > 0:
            rets.append(p1 / p0 - 1.0)
    if len(rets) < 3:
        return 0.0
    sd = statistics.pstdev(rets)
    return statistics.fmean(rets) / sd * math.sqrt(244.0) if sd > 0 else 0.0


def maxdd(nav: Nav) -> float:
    peak, mdd = None, 0.0
    for _, v in nav:
        if peak is None or v > peak:
            peak = v
        if peak and peak > 0:
            mdd = min(mdd, v / peak - 1.0)
    return mdd


def total_ret(nav: Nav) -> float:
    if not nav or nav[0][1] <= 0:
        return 0.0
    return nav[-1][1] / nav[0][1] - 1.0


def daily_rets(nav: Nav) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for i in range(1, len(nav)):
        p0, p1 = nav[i - 1][1], nav[i][1]
        if p0 and p0 > 0:
            out[nav[i][0]] = p1 / p0 - 1.0
    return out


def ann_vol(nav: Nav) -> float:
    vals = list(daily_rets(nav).values())
    if len(vals) < 30:
        return 0.0
    return statistics.pstdev(vals) * math.sqrt(244.0)


def corr(a: Dict[str, float], b: Dict[str, float]) -> float:
    keys = sorted(set(a) & set(b))
    if len(keys) < 30:
        return 0.0
    xa = [a[k] for k in keys]
    xb = [b[k] for k in keys]
    ma, mb = statistics.fmean(xa), statistics.fmean(xb)
    num = sum((x - ma) * (y - mb) for x, y in zip(xa, xb))
    da = math.sqrt(sum((x - ma) ** 2 for x in xa))
    db = math.sqrt(sum((y - mb) ** 2 for y in xb))
    return num / (da * db) if da > 0 and db > 0 else 0.0


def merge_by_date(navs: List[Nav], weights: List[float]) -> Nav:
    if not navs or len(navs) != len(weights):
        return []
    norm = []
    for nav in navs:
        if not nav or nav[0][1] <= 0:
            return []
        b = nav[0][1]
        norm.append({d: v / b for d, v in nav})
    all_dates = sorted(set().union(*[set(c) for c in norm]))
    return [(d, sum(w * c.get(d, 1.0) for c, w in zip(norm, weights))) for d in all_dates]


def load(path: str) -> Tuple[Nav, Dict[str, Nav]]:
    with open(path, encoding="utf-8") as fh:
        c = json.load(fh)
    return (
        [(d, v) for d, v in c["a_share"]],
        {k: [(d, v) for d, v in n] for k, n in c["classes"].items()},
    )


def main() -> None:
    for p in (MAIN_NAVS, OOS_NAVS):
        if not os.path.exists(p):
            print(f"❌ 缺 {p}")
            print("   先跑 multi_asset_leg_experiment.py（main）与 --window oos")
            return

    results: Dict[str, Any] = {}

    for tag, path in (("main(推导)", MAIN_NAVS), ("oos(样本外)", OOS_NAVS)):
        a_nav, class_navs = load(path)
        base_s = sharpe(a_nav)
        print("=" * 80)
        print(f"{tag}  {a_nav[0][0]} → {a_nav[-1][0]}")
        print("=" * 80)
        print(f"  A 股基线: 夏普 {base_s:.4f}  收益 {total_ret(a_nav):+.2%}  "
              f"回撤 {maxdd(a_nav):.2%}  波动 {ann_vol(a_nav):.4f}")
        print()

        res: Dict[str, Any] = {
            "window": f"{a_nav[0][0]}→{a_nav[-1][0]}",
            "baseline": {
                "sharpe": round(base_s, 6),
                "total_return": round(total_ret(a_nav), 6),
                "max_drawdown": round(maxdd(a_nav), 6),
                "annual_vol": round(ann_vol(a_nav), 6),
            },
            "configs": {},
        }

        # 统一的配置清单：主窗口推荐配置 + 若干对照
        all_cls = sorted(class_navs)
        configs: Dict[str, Dict[str, Any]] = {}

        # ★ 主窗口推荐配置：A股50% + 黄金/债券/商品/红利 各12.5%（剔除 cross_border）
        if "cross_border" in class_navs:
            keep = [c for c in all_cls if c != "cross_border"]
            configs["★推荐_去cross_border_4类"] = {
                "classes": keep,
                "scheme": "capital_equal",
            }
        # 对照 1：全部类别（含 cross_border）
        configs["对照_全部类别"] = {"classes": all_cls, "scheme": "capital_equal"}
        # 对照 2：黄金 + 债券
        if "gold" in class_navs and "bond" in class_navs:
            configs["对照_gold+bond"] = {"classes": ["gold", "bond"], "scheme": "capital_equal"}
        # 对照 3：仅黄金
        if "gold" in class_navs:
            configs["对照_仅gold"] = {"classes": ["gold"], "scheme": "capital_equal"}
        # 对照 4：剔除 gold
        if "gold" in class_navs:
            keep_ng = [c for c in all_cls if c != "gold"]
            configs["对照_剔除gold"] = {"classes": keep_ng, "scheme": "capital_equal"}
        # 对照 5：不含 gold 也不含 cross_border
        keep2 = [c for c in all_cls if c not in ("gold", "cross_border")]
        if len(keep2) >= 2:
            configs["对照_去gold且去cross_border"] = {"classes": keep2, "scheme": "capital_equal"}

        for name, cfg in configs.items():
            cls = cfg["classes"]
            legs = [a_nav] + [class_navs[c] for c in cls]
            wts = [0.5] + [0.5 / len(cls)] * len(cls)
            combo = merge_by_date(legs, wts)
            s = sharpe(combo)
            ok = s > base_s
            print(f"  {name:26s} 夏普 {s:7.4f} ({(s/base_s-1)*100:+7.2f}%)  "
                  f"收益 {total_ret(combo):+7.2%}  回撤 {maxdd(combo):7.2%}  "
                  f"波动 {ann_vol(combo):.4f}  {'✅' if ok else '❌'}")
            res["configs"][name] = {
                "classes": cls,
                "weight_a_share": 0.5,
                "sharpe": round(s, 6),
                "vs_base_pct": round((s / base_s - 1) * 100, 3),
                "total_return": round(total_ret(combo), 6),
                "max_drawdown": round(maxdd(combo), 6),
                "annual_vol": round(ann_vol(combo), 6),
                "beats": ok,
            }

        # 各资产类别与 A 股腿的相关性（两窗口对比）
        print()
        print("  各类别与 A 股腿的日收益相关：")
        res["correlations"] = {}
        for c in all_cls:
            cr = corr(daily_rets(a_nav), daily_rets(class_navs[c]))
            cs = sharpe(class_navs[c])
            print(f"    {c:16s} 相关 {cr:+.4f}   腿夏普 {cs:7.4f}   "
                  f"腿收益 {total_ret(class_navs[c]):+8.2%}  腿回撤 {maxdd(class_navs[c]):7.2%}")
            res["correlations"][c] = {
                "corr_with_a_share": round(cr, 4),
                "leg_sharpe": round(cs, 6),
                "leg_return": round(total_ret(class_navs[c]), 6),
                "leg_max_drawdown": round(maxdd(class_navs[c]), 6),
                "leg_annual_vol": round(ann_vol(class_navs[c]), 6),
            }
        print()
        results[tag] = res

    # ── 两窗口对比总结 ──
    print("=" * 80)
    print("两窗口对比：主窗口推导出的配置，样本外是否仍然成立？")
    print("=" * 80)
    summary: Dict[str, Any] = {}
    for name in results["main(推导)"]["configs"]:
        m = results["main(推导)"]["configs"][name]
        o = results["oos(样本外)"]["configs"].get(name)
        if not o:
            continue
        consistent = (m["beats"] == o["beats"])
        print(f"  {name:26s} main {m['vs_base_pct']:+7.2f}% ({'✅' if m['beats'] else '❌'})   "
              f"oos {o['vs_base_pct']:+7.2f}% ({'✅' if o['beats'] else '❌'})   "
              f"{'✔ 方向一致' if consistent else '✘ 方向不一致'}")
        summary[name] = {
            "main_vs_base_pct": m["vs_base_pct"],
            "main_beats": m["beats"],
            "oos_vs_base_pct": o["vs_base_pct"],
            "oos_beats": o["beats"],
            "consistent": consistent,
        }
    results["cross_window_summary"] = summary

    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=1)
    print(f"\n[oos] ✓ 已写入 {OUT_PATH}")


if __name__ == "__main__":
    main()
