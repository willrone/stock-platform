#!/usr/bin/env python3
"""多资产腿的**稳健性检验** —— 用缓存净值做子窗口/留一/替换分析。

## 为什么需要这个

`multi_asset_leg_experiment.py` 的主结果里：

- C6_best_gold（A股 + 黄金）夏普 +74%
- 但 **C7_no_gold（剔除黄金）夏普 −8.4%**，比 A 股基线还差

→ 所有"多资产突破"的收益几乎全部来自**黄金一根腿**，而黄金 +126% 里
2025 年独占 +56.8%。这正是 `POOL_PARTITION_2026-10-30.md` 标记过的
「2025H1 共同异常窗口」。

**在把它当成策略之前，必须回答：这是真实分散化，还是窗口运气？**

## 四组检验

| 检验 | 问的问题 | 判定标准 |
|---|---|---|
| **T1 子窗口** | 提升在每一段都成立吗？ | 3 个子窗口里至少 2 个 beat 基线 |
| **T2 留一类别** | 去掉每个类别后还成立吗？ | 去掉 gold 仍 beat 基线才算稳健 |
| **T3 替换资产** | 用同类别的其他 ETF 替代还成立吗？ | 替代后仍 beat 基线 |
| **T4 黄金回撤敏感** | 黄金腿的收益有多少来自不可持续的单边行情？ | 看 2023/2024 子窗口表现 |

## 用法

    cd backend && env -u PYTHONPATH .venv-py313/bin/python3 scripts/multi_asset_robustness.py
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

NAVS_PATH = os.path.join(ROOT, "data", "backtest", "multi_asset_navs.json")
RESULT_PATH = os.path.join(ROOT, "data", "backtest", "multi_asset_leg.json")
ETF_PARQUET = os.path.join(ROOT, "data", "etf", "parquet")

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


def slice_nav(nav: Nav, start: str, end: str) -> Nav:
    """截取 [start, end] 区间，并在起点处重新归一（用起点前一日的收盘做基准）。"""
    out = [(d, v) for d, v in nav if start <= d <= end]
    return out


def slice_with_base(nav: Nav, start: str, end: str) -> Nav:
    """截取区间并带上区间起点前的最后一个点（保证起点收益被计入）。"""
    before = [(d, v) for d, v in nav if d < start]
    inside = [(d, v) for d, v in nav if start <= d <= end]
    if before and inside:
        return [before[-1]] + inside
    return inside


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


# --------------------------------------------------------------------------
# ETF 腿（复现 multi_asset_leg_experiment 的口径）
# --------------------------------------------------------------------------
def etf_static_leg(codes: List[str], start: str, end: str, capital: float = 1_000_000.0) -> Nav:
    import pandas as pd

    frames: Dict[str, pd.DataFrame] = {}
    for code in codes:
        path = os.path.join(ETF_PARQUET, f"{code}.parquet")
        if not os.path.exists(path):
            continue
        df = pd.read_parquet(path).sort_values("date")
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        df = df[(df["date"] >= start) & (df["date"] <= end)]
        if len(df) > 60:
            frames[code] = df.reset_index(drop=True)
    if not frames:
        return []

    common = set(frames[next(iter(frames))]["date"])
    for df in frames.values():
        common &= set(df["date"])
    common_dates = sorted(d for d in common if d >= start)
    if len(common_dates) < 30:
        return []
    buy_date = common_dates[1]

    fillable = []
    for code, df in frames.items():
        row = df[df["date"] == buy_date]
        if row.empty:
            continue
        o = float(row.iloc[0]["open"])
        limit = 0.20 if code.startswith(("30", "68")) else 0.10
        prev = df[df["date"] < buy_date]
        if not prev.empty:
            pc = float(prev.iloc[-1]["close"])
            if pc > 0 and (o - pc) / pc >= limit - 0.001:
                continue
        fillable.append((code, o * (1 + 0.0001)))
    if not fillable:
        return []

    weight = capital / len(fillable)
    fills = [(c, weight / px, px) for c, px in fillable]
    comm = sum(sh * px for _, sh, px in fills) * 0.0003

    nav = []
    for d in common_dates:
        val = 0.0
        for code, sh, px in fills:
            df = frames[code]
            row = df[df["date"] == d]
            if row.empty:
                prev = df[df["date"] < d]
                p = float(prev.iloc[-1]["close"]) if not prev.empty else px
            else:
                p = float(row.iloc[0]["close"])
            val += sh * p
        nav.append((d, val))
    nav.insert(0, (common_dates[0], capital - comm))
    return nav


def main() -> None:
    if not os.path.exists(NAVS_PATH):
        print(f"❌ 缺净值缓存 {NAVS_PATH}，先跑 multi_asset_leg_experiment.py")
        return
    with open(NAVS_PATH, encoding="utf-8") as fh:
        cache = json.load(fh)
    a_nav: Nav = [(d, v) for d, v in cache["a_share"]]
    class_navs: Dict[str, Nav] = {c: [(d, v) for d, v in n] for c, n in cache["classes"].items()}

    with open(RESULT_PATH, encoding="utf-8") as fh:
        prior = json.load(fh)["results"]
    base_sharpe = sharpe(a_nav)

    print(f"[rb] A 股基线: 夏普 {base_sharpe:.4f}  收益 {total_ret(a_nav):+.2%}  "
          f"回撤 {maxdd(a_nav):.2%}  波动 {ann_vol(a_nav):.4f}")
    print()

    out: Dict[str, Any] = {"baseline_sharpe": base_sharpe}

    # ── T1 子窗口 ──
    # 用「起点前一日」作为基准点，保证区间首日收益被计入
    SUB_WINDOWS = [
        ("2023-05-08", "2024-05-08", "W1_2023-2024"),
        ("2024-05-08", "2025-05-08", "W2_2024-2025"),
        ("2025-05-08", "2026-05-08", "W3_2025-2026"),
    ]
    print("=" * 78)
    print("T1 子窗口稳健性 —— 提升是否在每一段都成立？")
    print("=" * 78)
    t1: Dict[str, Any] = {}

    # 参与对比的组合：A股+全类别、A股+gold、A股+gold+bond
    combo_defs = {
        "A_全类别": (a_nav, [class_navs[c] for c in sorted(class_navs)],
                     [1 / len(class_navs)] * len(class_navs)),
    }
    if "gold" in class_navs:
        combo_defs["A_gold"] = (a_nav, [class_navs["gold"]], [1.0])
    if "gold" in class_navs and "bond" in class_navs:
        combo_defs["A_gold_bond"] = (
            a_nav, [class_navs["gold"], class_navs["bond"]], [0.5, 0.5]
        )

    for name, (a_leg, others, ow) in combo_defs.items():
        print(f"\n--- {name} ---")
        print(f"  {'窗口':16s} {'A股基线':>10s} {'组合':>10s} {'差值':>9s} {'A股回撤':>10s} {'组合回撤':>10s}")
        wins = 0
        for s, e, label in SUB_WINDOWS:
            a_sub = slice_with_base(a_leg, s, e)
            o_sub = [slice_with_base(n, s, e) for n in others]
            combo = merge_by_date([a_sub] + o_sub, [0.5] + [w * 0.5 for w in ow])
            sb, sc = sharpe(a_sub), sharpe(combo)
            ok = sc > sb
            wins += 1 if ok else 0
            print(f"  {label:16s} {sb:10.4f} {sc:10.4f} {sc-sb:+9.4f} "
                  f"{maxdd(a_sub):10.2%} {maxdd(combo):10.2%}  {'✅' if ok else '❌'}")
            t1.setdefault(name, {})[label] = {
                "baseline_sharpe": round(sb, 6),
                "combo_sharpe": round(sc, 6),
                "delta": round(sc - sb, 6),
                "baseline_mdd": round(maxdd(a_sub), 6),
                "combo_mdd": round(maxdd(combo), 6),
                "combo_beats": ok,
            }
        print(f"  → {wins}/{len(SUB_WINDOWS)} 个子窗口 beat 基线")
        t1[name]["windows_win"] = wins
        t1[name]["total_windows"] = len(SUB_WINDOWS)
        t1[name]["robust"] = wins >= 2
    out["T1_subwindow"] = t1

    # ── T2 留一类别 ──
    print()
    print("=" * 78)
    print("T2 留一类别 —— 去掉每个类别后，组合是否仍 beat A 股基线？")
    print("=" * 78)
    t2: Dict[str, Any] = {}
    all_cls = sorted(class_navs)
    for drop in [None] + all_cls:
        keep = [c for c in all_cls if c != drop]
        if not keep:
            continue
        o_sub = [class_navs[c] for c in keep]
        combo = merge_by_date([a_nav] + o_sub, [0.5] + [0.5 / len(keep)] * len(keep))
        s = sharpe(combo)
        label = "全部类别" if drop is None else f"去掉 {drop}"
        ok = s > base_sharpe
        print(f"  {label:22s} 组合夏普 {s:7.4f} ({(s/base_sharpe-1)*100:+7.2f}%)  "
              f"收益 {total_ret(combo):+.2%}  回撤 {maxdd(combo):.2%}  {'✅' if ok else '❌'}")
        t2[label] = {
            "classes": keep,
            "sharpe": round(s, 6),
            "vs_base_pct": round((s / base_sharpe - 1) * 100, 3),
            "total_return": round(total_ret(combo), 6),
            "max_drawdown": round(maxdd(combo), 6),
            "beats": ok,
        }
    t2["robust_without_gold"] = t2.get("去掉 gold", {}).get("beats", False)
    out["T2_leave_one_out"] = t2

    # ── T3 替换资产（同类别换一只 ETF）──
    print()
    print("=" * 78)
    print("T3 替换资产 —— 黄金腿换成同类其他 ETF，结果还成立吗？")
    print("=" * 78)
    t3: Dict[str, Any] = {}
    gold_alts = ["518800", "518850", "159980", "518600", "159937"]
    with open(os.path.join(ROOT, "data", "etf", "etf_universe.json"), encoding="utf-8") as fh:
        uni = json.load(fh)["etfs"]
    for code in gold_alts:
        if code not in uni:
            continue
        leg = etf_static_leg([code], "2023-05-08", "2026-05-08")
        if not leg:
            print(f"  {code} {uni[code]['name'][:20]:22s} ❌ 无数据")
            continue
        combo = merge_by_date([a_nav, leg], [0.5, 0.5])
        s = sharpe(combo)
        ok = s > base_sharpe
        print(f"  {code} {uni[code]['name'][:20]:22s} 夏普 {s:7.4f} ({(s/base_sharpe-1)*100:+7.2f}%)  "
              f"腿收益 {total_ret(leg):+7.2%}  腿波动 {ann_vol(leg):.4f}  "
              f"相关 {corr(daily_rets(a_nav), daily_rets(leg)):+.4f}  {'✅' if ok else '❌'}")
        t3[code] = {
            "name": uni[code]["name"],
            "leg_sharpe": round(sharpe(leg), 6),
            "leg_return": round(total_ret(leg), 6),
            "leg_vol": round(ann_vol(leg), 6),
            "corr_with_a_share": round(corr(daily_rets(a_nav), daily_rets(leg)), 4),
            "combo_sharpe": round(s, 6),
            "beats": ok,
        }
    # 原 gold 腿
    if "gold" in class_navs:
        g = class_navs["gold"]
        combo = merge_by_date([a_nav, g], [0.5, 0.5])
        t3["_primary_gold_leg"] = {
            "name": "黄金类 top3 等权 (518880/159934/159937)",
            "leg_sharpe": round(sharpe(g), 6),
            "leg_return": round(total_ret(g), 6),
            "leg_vol": round(ann_vol(g), 6),
            "corr_with_a_share": round(corr(daily_rets(a_nav), daily_rets(g)), 4),
            "combo_sharpe": round(sharpe(combo), 6),
            "beats": sharpe(combo) > base_sharpe,
        }
    beats_n = sum(1 for v in t3.values() if v.get("beats"))
    print(f"  → {beats_n}/{len(t3)} 个黄金替代标的使组合 beat 基线")
    t3["_summary_beats"] = beats_n
    t3["_summary_total"] = len(t3)
    out["T3_gold_substitution"] = t3

    # ── T4 黄金腿的可持续性 ──
    print()
    print("=" * 78)
    print("T4 黄金腿的收益来源分解 —— 有多少来自不可持续的单边行情？")
    print("=" * 78)
    t4: Dict[str, Any] = {}
    if "gold" in class_navs:
        g = class_navs["gold"]
        a_sub_all = slice_with_base(a_nav, "2023-05-08", "2026-05-08")
        combo_all = merge_by_date([a_nav, g], [0.5, 0.5])
        for s, e, label in SUB_WINDOWS:
            g_sub = slice_with_base(g, s, e)
            a_sub = slice_with_base(a_nav, s, e)
            combo_sub = merge_by_date([a_sub, g_sub], [0.5, 0.5])
            print(f"  {label:16s} 黄金腿 收益 {total_ret(g_sub):+7.2%} 夏普 {sharpe(g_sub):6.3f}  "
                  f"波动 {ann_vol(g_sub):.4f}  |  组合 {sharpe(combo_sub):6.4f} vs 基线 {sharpe(a_sub):6.4f}")
            t4[label] = {
                "gold_return": round(total_ret(g_sub), 6),
                "gold_sharpe": round(sharpe(g_sub), 6),
                "gold_vol": round(ann_vol(g_sub), 6),
                "combo_sharpe": round(sharpe(combo_sub), 6),
                "baseline_sharpe": round(sharpe(a_sub), 6),
            }
    out["T4_gold_decomposition"] = t4

    # ── T5 滚动窗口胜率（决定性检验）──
    # 3 个年度窗口太少。滚动窗口给出每个时点的独立判定，
    # 是判断「组合真的更稳」还是「只是碰巧赢了一整段」的标准做法。
    print()
    print("=" * 78)
    print("T5 滚动窗口胜率 —— 组合在多少个滚动窗口里 beat A 股基线？")
    print("=" * 78)
    t5: Dict[str, Any] = {}

    def rolling_winrate(
        a_leg: Nav,
        others: List[Nav],
        ow: List[float],
        months: int,
    ) -> Dict[str, Any]:
        """滚动 months 个月窗口，统计组合夏普 > 基线夏普的比例。"""
        # 用日期索引构造滚动切片
        a_dates = [d for d, _ in a_leg]
        n = len(a_dates)
        # 交易日/月 ≈ 21
        step = months * 21
        wins = 0
        total = 0
        deltas: List[float] = []
        for i in range(0, n - step, 10):  # 步长 10 天，避免窗口高度重叠
            s, e = a_dates[i], a_dates[i + step]
            a_sub = slice_with_base(a_leg, s, e)
            o_sub = [slice_with_base(nv, s, e) for nv in others]
            combo = merge_by_date([a_sub] + o_sub, [0.5] + [w * 0.5 for w in ow])
            sb, sc = sharpe(a_sub), sharpe(combo)
            total += 1
            deltas.append(sc - sb)
            if sc > sb:
                wins += 1
        return {
            "window_months": months,
            "total_windows": total,
            "wins": wins,
            "win_rate": round(wins / total * 100, 2) if total else 0.0,
            "median_delta_sharpe": round(statistics.fmean(deltas), 4) if deltas else 0.0,
            "mean_delta_sharpe": round(statistics.fmean(deltas), 4) if deltas else 0.0,
        }

    for name, (a_leg, others, ow) in combo_defs.items():
        t5[name] = {}
        for months in (3, 6, 12):
            r = rolling_winrate(a_leg, others, ow, months)
            t5[name][f"{months}m"] = r
            print(f"  {name:14s} 滚动{months:2d}月: {r['wins']}/{r['total_windows']} "
                  f"= {r['win_rate']:5.1f}%  平均Δ夏普 {r['mean_delta_sharpe']:+.4f}")
        print()

    # 额外的组合：去掉 cross_border（留一类别里它是最大的拖累）
    if "cross_border" in class_navs:
        keep = [c for c in sorted(class_navs) if c != "cross_border"]
        combo_defs2 = {
            "A_去cross_border": (
                a_nav, [class_navs[c] for c in keep], [1 / len(keep)] * len(keep)
            ),
        }
        for name, (a_leg, others, ow) in combo_defs2.items():
            t5[name] = {}
            for months in (3, 6, 12):
                r = rolling_winrate(a_leg, others, ow, months)
                t5[name][f"{months}m"] = r
                print(f"  {name:14s} 滚动{months:2d}月: {r['wins']}/{r['total_windows']} "
                      f"= {r['win_rate']:5.1f}%  平均Δ夏普 {r['mean_delta_sharpe']:+.4f}")
            # 同时看该组合的子窗口
            print(f"  {name:14s} 子窗口:")
            for s, e, label in SUB_WINDOWS:
                a_sub = slice_with_base(a_leg, s, e)
                o_sub = [slice_with_base(nv, s, e) for nv in others]
                combo = merge_by_date([a_sub] + o_sub, [0.5] + [w * 0.5 for w in ow])
                sb, sc = sharpe(a_sub), sharpe(combo)
                ok = sc > sb
                print(f"      {label:16s} 基线 {sb:7.4f}  组合 {sc:7.4f}  Δ{sc-sb:+7.4f}  "
                      f"回撤 {maxdd(a_sub):.2%}→{maxdd(combo):.2%}  {'✅' if ok else '❌'}")
                t5.setdefault(name, {}).setdefault("subwindows", {})[label] = {
                    "baseline_sharpe": round(sb, 6),
                    "combo_sharpe": round(sc, 6),
                    "beats": ok,
                }
        # 全窗口指标
        keep = [c for c in sorted(class_navs) if c != "cross_border"]
        combo = merge_by_date([a_nav] + [class_navs[c] for c in keep], [0.5] + [0.5 / len(keep)] * len(keep))
        t5["A_去cross_border"]["full_window"] = {
            "sharpe": round(sharpe(combo), 6),
            "vs_base_pct": round((sharpe(combo) / base_sharpe - 1) * 100, 3),
            "total_return": round(total_ret(combo), 6),
            "max_drawdown": round(maxdd(combo), 6),
        }
        print(f"\n  A_去cross_border 全窗口: 夏普 {sharpe(combo):.4f} "
              f"({(sharpe(combo)/base_sharpe-1)*100:+.2f}%)  收益 {total_ret(combo):+.2%}  "
              f"回撤 {maxdd(combo):.2%}")
    out["T5_rolling_window"] = t5

    # ── T6 再平衡口径（重要！合成不能是「买入持有不管」）──
    # 静态加权 = 期初权重永不再平衡。A 股腿 +35%、黄金腿 +126% →
    # 期末实际权重已严重漂移，「50/50」只是期初值。真实部署必须定期再平衡，
    # 且再平衡有交易成本。本检验对比：买入持有 vs 月度/季度度再平衡。
    print()
    print("=" * 78)
    print("T6 再平衡口径 —— 静态买入持有 vs 定期再平衡（后者才是可部署的）")
    print("=" * 78)
    t6: Dict[str, Any] = {}

    def leg_returns(nav: Nav) -> Dict[str, float]:
        return daily_rets(nav)

    def rebalanced_portfolio(
        legs: List[Nav],
        weights: List[float],
        rebalance_every_days: int,  # 0 = 不再平衡
        cost_rate: float = 0.0004,  # ETF 佣金+滑点
    ) -> Nav:
        """按日推进 + 定期再平衡的组合净值。

        legs[i] 是第 i 条腿的净值序列；weights 是目标权重。
        """
        rets = [leg_returns(nv) for nv in legs]
        dates = sorted(set().union(*[set(r) for r in rets]))
        if not dates:
            return []
        # 起始：各腿按目标权重投入
        leg_vals = [w for w in weights]  # 归一化起点=1
        nav: Nav = []
        total_in = sum(weights)
        leg_vals = [w / total_in for w in weights]
        nav.append((dates[0], 1.0))
        for i, d in enumerate(dates[1:], start=1):
            for li in range(len(legs)):
                r = rets[li].get(d, 0.0)
                leg_vals[li] *= (1.0 + r)
            total = sum(leg_vals)
            if rebalance_every_days and i % rebalance_every_days == 0:
                # 再平衡：按目标权重重置，扣除换手成本
                before = list(leg_vals)
                new_vals = [w / total_in * total for w in weights]
                turnover = sum(abs(nv - bv) for nv, bv in zip(new_vals, before))
                cost = turnover * cost_rate
                leg_vals = [v * (1.0 - cost / total) if total > 0 else v
                            for v in new_vals]
                total = sum(leg_vals)
            nav.append((d, total))
        return nav

    t6_configs = {}
    if "cross_border" in class_navs:
        keep = [c for c in sorted(class_navs) if c != "cross_border"]
        t6_configs["A_去cross_border"] = (
            [a_nav] + [class_navs[c] for c in keep],
            [0.5] + [0.5 / len(keep)] * len(keep),
        )
    t6_configs["A_全类别"] = (
        [a_nav] + [class_navs[c] for c in sorted(class_navs)],
        [0.5] + [0.5 / len(class_navs)] * len(class_navs),
    )

    for name, (legs, wts) in t6_configs.items():
        t6[name] = {}
        print(f"\n--- {name} ---")
        for label, every in [("买入持有", 0), ("月度再平衡(21d)", 21),
                             ("季度再平衡(63d)", 63), ("半年再平衡(126d)", 126)]:
            nav = rebalanced_portfolio(legs, wts, every)
            s = sharpe(nav)
            ok = s > base_sharpe
            print(f"  {label:18s} 夏普 {s:7.4f} ({(s/base_sharpe-1)*100:+7.2f}%)  "
                  f"收益 {total_ret(nav):+7.2%}  回撤 {maxdd(nav):7.2%}  "
                  f"波动 {ann_vol(nav):.4f}  {'✅' if ok else '❌'}")
            t6[name][label] = {
                "sharpe": round(s, 6),
                "vs_base_pct": round((s / base_sharpe - 1) * 100, 3),
                "total_return": round(total_ret(nav), 6),
                "max_drawdown": round(maxdd(nav), 6),
                "annual_vol": round(ann_vol(nav), 6),
                "beats": ok,
            }
        # 再平衡口径下的子窗口
        best_label = max(t6[name], key=lambda k: t6[name][k]["sharpe"])
        print(f"  最优再平衡口径 = {best_label}")
        nav = rebalanced_portfolio(legs, wts, {"买入持有": 0, "月度再平衡(21d)": 21,
                                               "季度再平衡(63d)": 63,
                                               "半年再平衡(126d)": 126}[best_label])
        t6[name]["subwindows_best"] = {}
        for s0, e0, label in SUB_WINDOWS:
            a_sub = slice_with_base(a_nav, s0, e0)
            c_sub = slice_with_base(nav, s0, e0)
            sb, sc = sharpe(a_sub), sharpe(c_sub)
            print(f"    {label:16s} 基线 {sb:7.4f}  组合 {sc:7.4f}  Δ{sc-sb:+7.4f}  "
                  f"回撤 {maxdd(a_sub):.2%}→{maxdd(c_sub):.2%}  {'✅' if sc > sb else '❌'}")
            t6[name]["subwindows_best"][label] = {
                "baseline_sharpe": round(sb, 6),
                "combo_sharpe": round(sc, 6),
                "beats": sc > sb,
            }
    out["T6_rebalancing"] = t6

    # ── 落盘 ──
    out_path = os.path.join(ROOT, "data", "backtest", "multi_asset_robustness.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print(f"\n[rb] ✓ 已写入 {out_path}")


if __name__ == "__main__":
    main()
