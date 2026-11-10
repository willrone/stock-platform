#!/usr/bin/env python3
"""定稿配置的完整稳健性诊断 —— 把 T1/T2/T3/T5 全部跑在**最终推荐配置**上。

## 为什么要有这个脚本

`multi_asset_robustness.py` 的 T1~T6 跑的是**旧的 5 类别配置**
（含 cross_border / dividend），而推荐配置早已定稿为
「A股腿 50% + gold/bond/commodity 各 16.67%（每类 top-3 流动性 ETF 等权）
+ ETF 级每月再平衡」。

也就是说：**定稿配置从来没有被做过完整的稳健性诊断**。
只有 `multi_asset_l3_momentum.py` 里的单一对比（vs A 股基线）。
这正是本项目最容易被"单一数字好看"骗到的地方。

## 本脚本做什么

在**同一个候选池、同一套撮合引擎**（复用 `multi_asset_l3_momentum`）下，
对定稿配置做 7 组诊断：

| 编号 | 诊断 | 回答什么问题 |
|---|---|---|
| F1 | 定稿配置全景 | 到底多少？ |
| F2 | **留一类别** | 去掉 gold / bond / commodity 后还 beat 基线吗？谁是命门？ |
| F3 | **替换 gold ETF** | 结果是 gold 这个**资产**的功劳，还是碰巧选中那只 ETF？ |
| F4 | **非重叠子窗口** | 每一段都成立吗？ |
| F5 | **滚动窗口胜率** | 在多少个滚动窗口里 beat 基线？ |
| F6 | **gold 依赖度** | 组合的收益里有多少是金价 beta？ |
| F7 | **再平衡贡献** | 定稿配置里，再平衡贡献了多少？ |

## 口径纪律

- ETF 成交：信号日后**次日 open**，佣金 0.03% + 滑点 0.01%（ETF 免印花税）
- 候选池：**窗口前**流动性排序冻结，全程不换（零前视）
- A 股腿：`run_a_share_leg()` 读 MA 模块**全局** START/END → 必须显式同步
  （曾因此让 OOS 静默用 main 的 A 股腿，得出"OOS 全军覆没"的假结论）
- 护栏：A 股腿净值起点必须落在本窗口内，否则退出

用法：
    cd backend && env -u PYTHONPATH BASELINE_LOG_LEVEL=ERROR .venv-py313/bin/python3 \
        scripts/multi_asset_final_robustness.py --window main
    cd backend && env -u PYTHONPATH BASELINE_LOG_LEVEL=ERROR .venv-py313/bin/python3 \
        scripts/multi_asset_final_robustness.py --window oos
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT = os.path.abspath(os.path.join(BACKEND_DIR, ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)
sys.path.insert(0, os.path.join(BACKEND_DIR, "scripts"))

import multi_asset_l3_momentum as L3  # noqa: E402
import multi_asset_leg_experiment as MA  # noqa: E402
from multi_asset_leg_experiment import (  # noqa: E402
    corr,
    daily_rets,
    load_etf_universe,
    maxdd,
    merge_by_date,
    run_a_share_leg,
    sharpe,
    total_ret,
    verify_baseline,
    yearly,
)
from verify_methods_engine import QLIB_OFFICIAL, universe_from_instruments  # noqa: E402

#: 定稿配置：三类，各占 ETF 袖套的 1/3
FINAL_CLASSES = ["gold", "bond", "commodity"]
#: 每类持仓数（按流动性 top-N）
TOP_N = 3
#: 再平衡频率（交易日）
REBAL_EVERY = 21
#: A 股腿权重（其余 50% 给 ETF 袖套）
A_WEIGHT = 0.5


def ann_vol(nav: List[Tuple[str, float]]) -> float:
    r = list(daily_rets(nav).values())
    if len(r) < 2:
        return 0.0
    m = sum(r) / len(r)
    var = sum((x - m) ** 2 for x in r) / (len(r) - 1)
    return (var ** 0.5) * (252 ** 0.5)


def slice_with_base(nav: List[Tuple[str, float]], start: str, end: str
                    ) -> List[Tuple[str, float]]:
    """取子区间，并把**区间前一日**的净值点带上，保证首日收益被计入。"""
    prev_pt = None
    out: List[Tuple[str, float]] = []
    for d, v in nav:
        if d < start:
            prev_pt = (d, v)
            continue
        if d > end:
            break
        out.append((d, v))
    if prev_pt is not None and out:
        out.insert(0, prev_pt)
    return out


def build_sleeve(
    calendar: List[str],
    prices: Dict[str, Dict[str, Dict[str, float]]],
    pool: Dict[str, List[str]],
    class_weights: Dict[str, float],
    rebalance: bool,
    snapshot: bool = False,
) -> Tuple[List[Tuple[str, float]], Dict[str, Any], List[Tuple[str, Dict[str, float]]]]:
    """给定每类候选与权重 → ETF 袖套净值。rebalance=False 时是买入持有。

    `class_weights` 是**袖套内部**的目标权重（可以不合计到 1，剩余部分留现金）。
    """
    class_of = {c: cls for cls, lst in pool.items() for c in lst}
    if rebalance:
        targets, meta = L3.build_rotation_targets(
            calendar, prices, pool, class_weights, 60, 99, REBAL_EVERY, "all")
    else:
        targets = L3.build_static_targets(calendar, prices, pool, class_weights)
        meta = {"n_exec": 1, "rule": "buyhold"}
    snap: List[Tuple[str, Dict[str, float]]] = []
    nav, rec = L3.simulate(
        calendar, prices, targets, class_of, class_weights,
        snapshot_out=snap if snapshot else None)
    return nav, {**rec, **meta}, snap


def class_weight_trace(
    snap: List[Tuple[str, Dict[str, float]]],
    class_of: Dict[str, str],
) -> Dict[str, Any]:
    """各类别在袖套里的**真实**权重轨迹（用收盘持仓市值算，不是比值法）。

    比值法（gold_leg_nav / sleeve_nav）给出的是「相对表现」，
    起点恒为 1.0 而不是 1/3 —— 会把「gold 涨得多」误报成「gold 权重高」。
    """
    if not snap:
        return {}
    totals: List[float] = []
    per_class: Dict[str, List[float]] = {}
    for _d, pos in snap:
        tot = sum(pos.values())
        totals.append(tot)
        agg: Dict[str, float] = {}
        for code, v in pos.items():
            c = class_of.get(code, "?")
            agg[c] = agg.get(c, 0.0) + v
        for c in agg:
            per_class.setdefault(c, []).append(agg[c] / tot if tot > 0 else 0.0)
    out: Dict[str, Any] = {
        "n_snapshots": len(snap),
        "class_weight_end": {c: round(v[-1], 4) for c, v in per_class.items()},
        "class_weight_start": {c: round(v[0], 4) for c, v in per_class.items()},
        "class_weight_peak": {c: round(max(v), 4) for c, v in per_class.items()},
        "class_weight_mean": {
            c: round(sum(v) / len(v), 4) for c, v in per_class.items()},
    }
    if "gold" in per_class:
        gw = per_class["gold"]
        out["gold_weight_start"] = round(gw[0], 4)
        out["gold_weight_end"] = round(gw[-1], 4)
        out["gold_weight_peak"] = round(max(gw), 4)
        out["gold_weight_mean"] = round(sum(gw) / len(gw), 4)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", default="main", choices=["main", "oos"])
    ap.add_argument("--quick", action="store_true", help="复用 L3 缓存的 A 股腿")
    args = ap.parse_args()

    if args.window == "oos":
        start, end = MA.OOS_START, MA.OOS_END
        out_path = os.path.join(ROOT, "data", "backtest", "multi_asset_final_robustness_oos.json")
        nav_cache = os.path.join(ROOT, "data", "backtest", "multi_asset_l3_navs_oos.json")
    else:
        start, end = MA.START, MA.END
        out_path = os.path.join(ROOT, "data", "backtest", "multi_asset_final_robustness.json")
        nav_cache = os.path.join(ROOT, "data", "backtest", "multi_asset_l3_navs.json")

    print(f"[fr] 窗口 {start.date()} → {end.date()}", flush=True)

    # ⚠️ run_a_share_leg() 读 MA 模块全局 START/END，必须显式同步
    MA.START, MA.END = start, end

    # ── A 股腿 ──
    a_nav: List[Tuple[str, float]] = []
    a_rec: Dict[str, Any] = {}
    if args.quick and os.path.exists(nav_cache):
        with open(nav_cache, encoding="utf-8") as fh:
            cached = json.load(fh)
        a_nav = [(d, float(v)) for d, v in cached["a_share_nav"]]
        a_rec = cached.get("a_share_leg", {})
        print("[fr] 读缓存 A 股腿", flush=True)
    else:
        codes = universe_from_instruments(QLIB_OFFICIAL)
        print(f"[fr] 跑 A 股腿（{len(codes)} 只）...", flush=True)
        t0 = time.perf_counter()
        a_rec, a_nav = run_a_share_leg(codes)
        a_rec["wall_s"] = round(time.perf_counter() - t0, 1)
        ver = verify_baseline(a_rec)
        if args.window == "main":
            print(f"[fr] 基线复现门: {json.dumps(ver, ensure_ascii=False)}", flush=True)
            if ver.get("checked") and not ver.get("ok"):
                raise SystemExit(1)

    a_first = a_nav[0][0] if a_nav else ""
    a_last = a_nav[-1][0] if a_nav else ""
    if not (start.strftime("%Y-%m-%d") <= a_first <= end.strftime("%Y-%m-%d")):
        raise SystemExit(
            f"[fr] ❌ A 股腿起点 {a_first} 不在窗口 {start.date()}~{end.date()} 内 → "
            f"模块全局 START/END 没同步，退出"
        )
    if a_last < end.strftime("%Y-%m-%d"):
        print(f"[fr] ⚠️ A 股腿终点 {a_last} 早于窗口终点", flush=True)

    base_sharpe = sharpe(a_nav)
    store: Dict[str, Any] = {
        "window_mode": args.window,
        "window": [start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")],
        "baseline": {
            "sharpe": round(base_sharpe, 6),
            "total_return": round(total_ret(a_nav), 6),
            "max_drawdown": round(maxdd(a_nav), 6),
            "annual_vol": round(ann_vol(a_nav), 6),
            "engine_sharpe": a_rec.get("engine_sharpe"),
            "engine_max_drawdown": a_rec.get("engine_max_drawdown"),
            "total_trades": a_rec.get("total_trades"),
        },
        "final_config": {
            "a_share_weight": A_WEIGHT,
            "etf_sleeve_weight": 1 - A_WEIGHT,
            "classes": FINAL_CLASSES,
            "class_weight_each": round((1 - A_WEIGHT) / len(FINAL_CLASSES), 6),
            "top_n_per_class": TOP_N,
            "rebalance_every_trading_days": REBAL_EVERY,
        },
    }
    print(f"[fr] A 股基线: 夏普 {base_sharpe:.4f} 收益 {total_ret(a_nav):+.2%} "
          f"回撤 {maxdd(a_nav):.2%} 波动 {ann_vol(a_nav):.4f}", flush=True)

    # ── 候选池 ──
    universe = load_etf_universe()
    by_class = L3.candidates_by_class(universe, start, end)
    pool = {c: lst[:TOP_N] for c, lst in by_class.items() if c in FINAL_CLASSES}
    print(f"[fr] 候选池 top{TOP_N}: {json.dumps(pool, ensure_ascii=False)}", flush=True)
    store["candidates"] = by_class
    store["final_pool"] = pool

    all_codes = sorted({c for lst in by_class.values() for c in lst})
    prices = L3.load_prices(all_codes)
    s_str, e_str = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
    calendar = L3.build_calendar(prices, all_codes, s_str, e_str)
    print(f"[fr] 交易日历 {len(calendar)} 天 {calendar[0]} → {calendar[-1]}", flush=True)
    store["calendar"] = {"n": len(calendar), "start": calendar[0], "end": calendar[-1]}

    cw = {c: 1.0 / len(FINAL_CLASSES) for c in FINAL_CLASSES}

    # gold 单腿（供依赖度度量）
    gold_pool = {"gold": pool.get("gold", [])}
    gold_nav, _, _ = build_sleeve(calendar, prices, gold_pool, {"gold": 1.0}, False)

    def combo(sleeve_nav: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        return merge_by_date([a_nav, sleeve_nav], [A_WEIGHT, 1 - A_WEIGHT])

    # ══════════════════════════════════════════════════════════════
    # F1 定稿配置全景
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 78)
    print("F1 定稿配置全景 —— A股50% + gold/bond/commodity 各16.67%（top3）+ 每月再平衡")
    print("=" * 78)
    sleeve_nav, sleeve_rec, sleeve_snap = build_sleeve(
        calendar, prices, pool, cw, True, snapshot=True)
    comb = combo(sleeve_nav)
    class_of_all = {c: cls for cls, lst in pool.items() for c in lst}
    f1 = {
        "sleeve_sharpe": round(sharpe(sleeve_nav), 4),
        "sleeve_return": round(total_ret(sleeve_nav), 4),
        "sleeve_maxdd": round(maxdd(sleeve_nav), 4),
        "sleeve_vol": round(ann_vol(sleeve_nav), 4),
        "combined_sharpe": round(sharpe(comb), 4),
        "vs_baseline_pct": round((sharpe(comb) / base_sharpe - 1) * 100, 2),
        "combined_return": round(total_ret(comb), 4),
        "combined_maxdd": round(maxdd(comb), 4),
        "combined_vol": round(ann_vol(comb), 4),
        "corr_with_a_share": round(corr(daily_rets(comb), daily_rets(a_nav)), 4),
        "corr_with_gold": round(corr(daily_rets(comb), daily_rets(gold_nav)), 4),
        "yearly": yearly(comb),
        "cost": sleeve_rec,
    }
    f1.update(class_weight_trace(sleeve_snap, class_of_all))
    store["F1_final"] = f1
    print(f"  袖套    夏普 {f1['sleeve_sharpe']:.4f}  收益 {f1['sleeve_return']:+.2%}  "
          f"回撤 {f1['sleeve_maxdd']:.2%}  波动 {f1['sleeve_vol']:.4f}")
    print(f"  组合    夏普 {f1['combined_sharpe']:.4f} ({f1['vs_baseline_pct']:+.2f}%)  "
          f"收益 {f1['combined_return']:+.2%}  回撤 {f1['combined_maxdd']:.2%}  "
          f"波动 {f1['combined_vol']:.4f}")
    print(f"  相关    与A股 {f1['corr_with_a_share']}  与gold {f1['corr_with_gold']}")
    print(f"  袖套内权重轨迹（真实持仓市值口径）:")
    for c in FINAL_CLASSES:
        st = f1["class_weight_start"].get(c)
        en = f1["class_weight_end"].get(c)
        pk = f1["class_weight_peak"].get(c)
        mn = f1.get("class_weight_mean", {}).get(c)
        print(f"    {c:10s} 起 {st:.2%} → 末 {en:.2%}（峰 {pk:.2%}，均值 {mn:.2%}）")
    print(f"  成本    换手 {f1['cost'].get('turnover_ratio')}  佣金 {f1['cost'].get('commission_total')}  "
          f"调仓 {f1['cost'].get('n_rebalances')} 次")
    print(f"  分年度  {json.dumps(f1['yearly'], ensure_ascii=False)}")

    # ══════════════════════════════════════════════════════════════
    # F2 留一类别
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 78)
    print("F2 留一类别 —— 两种口径：重新归一化 vs 权重中性")
    print("=" * 78)
    print("  [归一化] 剩余类别瓜分整个 ETF 袖套（2 类各 50%）")
    print("            → 但这会**抬高 gold 的绝对权重**（16.67%→25%），混杂了权重效应")
    print("  [中性]   剩余类别维持原绝对权重（各 1/3 袖套），空出的 1/3 留现金")
    print("            → 干净地测「这一类的边际贡献」")
    f2: Dict[str, Any] = {}
    for drop in FINAL_CLASSES:
        keep = [c for c in FINAL_CLASSES if c != drop]
        if not keep:
            continue
        p2 = {c: pool[c] for c in keep if pool.get(c)}
        # (a) 重新归一化
        cw2 = {c: 1.0 / len(p2) for c in p2}
        sl2, _, _ = build_sleeve(calendar, prices, p2, cw2, True)
        cb2 = combo(sl2)
        s2 = sharpe(cb2)
        # (b) 权重中性：维持原权重 1/3，空出的部分留现金
        cw2n = {c: 1.0 / len(FINAL_CLASSES) for c in p2}
        sl2n, _, _ = build_sleeve(calendar, prices, p2, cw2n, True)
        cb2n = combo(sl2n)
        s2n = sharpe(cb2n)
        rec2 = {
            "renorm": {
                "combined_sharpe": round(s2, 4),
                "vs_baseline_pct": round((s2 / base_sharpe - 1) * 100, 2),
                "combined_return": round(total_ret(cb2), 4),
                "combined_maxdd": round(maxdd(cb2), 4),
                "corr_with_gold": round(corr(daily_rets(cb2), daily_rets(gold_nav)), 4),
                "note": "剩余类别瓜分整个袖套 → gold 绝对权重被抬高",
            },
            "neutral": {
                "combined_sharpe": round(s2n, 4),
                "vs_baseline_pct": round((s2n / base_sharpe - 1) * 100, 2),
                "combined_return": round(total_ret(cb2n), 4),
                "combined_maxdd": round(maxdd(cb2n), 4),
                "corr_with_gold": round(corr(daily_rets(cb2n), daily_rets(gold_nav)), 4),
                "note": "维持原绝对权重，空出的 1/3 袖套留现金",
            },
            "marginal_sharpe_contribution": round(sharpe(comb) - s2n, 4),
        }
        f2[f"drop_{drop}"] = rec2
        print(f"  去掉 {drop:10s}")
        print(f"    [归一化] 夏普 {s2:.4f} ({rec2['renorm']['vs_baseline_pct']:+6.2f}%)  "
              f"收益 {rec2['renorm']['combined_return']:+.2%}  "
              f"回撤 {rec2['renorm']['combined_maxdd']:.2%}  "
              f"与gold相关 {rec2['renorm']['corr_with_gold']}")
        print(f"    [中性]   夏普 {s2n:.4f} ({rec2['neutral']['vs_baseline_pct']:+6.2f}%)  "
              f"收益 {rec2['neutral']['combined_return']:+.2%}  "
              f"回撤 {rec2['neutral']['combined_maxdd']:.2%}  "
              f"与gold相关 {rec2['neutral']['corr_with_gold']}")
        print(f"    → 该类对夏普的边际贡献 {rec2['marginal_sharpe_contribution']:+.4f}  "
              f"{'（正贡献）' if rec2['marginal_sharpe_contribution'] > 0 else '（负贡献！）'}")
    store["F2_leave_one_out"] = f2

    # ══════════════════════════════════════════════════════════════
    # F3 替换 gold ETF（排除"碰巧选中赢家"）
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 78)
    print("F3 替换 gold ETF —— 结论是 gold 这个资产的功劳，还是碰巧选中那只？")
    print("=" * 78)
    gold_full = by_class.get("gold", [])
    f3: Dict[str, Any] = {}
    for rank, code in enumerate(gold_full[:6], start=1):
        p3 = dict(pool)
        p3["gold"] = [code]
        sl3, _, _ = build_sleeve(calendar, prices, p3, cw, True)
        cb3 = combo(sl3)
        s3 = sharpe(cb3)
        rec3 = {
            "code": code,
            "liquidity_rank": rank,
            "combined_sharpe": round(s3, 4),
            "vs_baseline_pct": round((s3 / base_sharpe - 1) * 100, 2),
            "combined_return": round(total_ret(cb3), 4),
            "combined_maxdd": round(maxdd(cb3), 4),
        }
        f3[code] = rec3
        mark = "★定稿" if code in pool.get("gold", []) else "  "
        print(f"  {mark} rank{rank} {code}  组合夏普 {s3:.4f} ({rec3['vs_baseline_pct']:+6.2f}%)  "
              f"收益 {rec3['combined_return']:+.2%}  回撤 {rec3['combined_maxdd']:.2%}  "
              f"{'✅' if s3 > base_sharpe else '❌'}")
    beats = sum(1 for v in f3.values() if v["vs_baseline_pct"] > 0)
    print(f"  → {beats}/{len(f3)} 个 gold 候选使组合 beat 基线")
    store["F3_gold_substitution"] = {"detail": f3, "beats": beats, "n": len(f3)}

    # ══════════════════════════════════════════════════════════════
    # F4 非重叠子窗口
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 78)
    print("F4 非重叠子窗口 —— 每一段都成立吗？")
    print("=" * 78)
    s_d, e_d = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
    span_days = (end - start).days
    sub_defs: List[Tuple[str, str, str]] = []
    if span_days >= 1000:  # 3 年
        sub_defs = [
            (s_d, f"{start.year + 1}-05-08", f"Y1_{start.year}-{start.year + 1}"),
            (f"{start.year + 1}-05-08", f"{start.year + 2}-05-08", f"Y2_{start.year + 1}-{start.year + 2}"),
            (f"{start.year + 2}-05-08", e_d, f"Y3_{start.year + 2}-{end.year}"),
            (s_d, f"{start.year + 2}-05-08", f"H_{start.year}-{start.year + 2}"),
            (s_d, e_d, f"FULL"),
        ]
    else:  # 2 年
        sub_defs = [
            (s_d, f"{start.year + 1}-05-08", f"Y1_{start.year}-{start.year + 1}"),
            (f"{start.year + 1}-05-08", e_d, f"Y2_{start.year + 1}-{end.year}"),
            (s_d, e_d, f"FULL"),
        ]
    f4: Dict[str, Any] = {}
    for s0, e0, label in sub_defs:
        a_sub = slice_with_base(a_nav, s0, e0)
        c_sub = slice_with_base(comb, s0, e0)
        if len(a_sub) < 30 or len(c_sub) < 30:
            continue
        sb, sc = sharpe(a_sub), sharpe(c_sub)
        rec4 = {
            "baseline_sharpe": round(sb, 4),
            "combo_sharpe": round(sc, 4),
            "delta": round(sc - sb, 4),
            "baseline_maxdd": round(maxdd(a_sub), 4),
            "combo_maxdd": round(maxdd(c_sub), 4),
            "beats": sc > sb,
            "n_days": len(c_sub),
        }
        f4[label] = rec4
        print(f"  {label:18s} 基线 {sb:7.4f}  组合 {sc:7.4f}  Δ{sc - sb:+7.4f}  "
              f"回撤 {maxdd(a_sub):7.2%}→{maxdd(c_sub):7.2%}  "
              f"{'✅' if sc > sb else '❌'}")
    non_full = [v for k, v in f4.items() if k != "FULL"]
    if non_full:
        w = sum(1 for v in non_full if v["beats"])
        print(f"  → {w}/{len(non_full)} 个非重叠子窗口 beat 基线")
        store["F4_subwindow_summary"] = {
            "beats": w, "n": len(non_full),
            "avg_delta": round(sum(v["delta"] for v in non_full) / len(non_full), 4),
        }
    store["F4_subwindows"] = f4

    # ══════════════════════════════════════════════════════════════
    # F5 滚动窗口胜率
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 78)
    print("F5 滚动窗口胜率 —— 在多少个滚动窗口里 beat A 股基线？")
    print("=" * 78)
    f5: Dict[str, Any] = {}
    for win_days, label in ((252, "滚动12月"), (504, "滚动24月")):
        a_vals = dict(a_nav)
        c_vals = dict(comb)
        dates = [d for d, _ in comb if d in a_vals]
        wins_s = wins_r = 0
        total = 0
        deltas: List[float] = []
        for i in range(win_days, len(dates)):
            d0, d1 = dates[i - win_days], dates[i]
            seg_a = [(d, a_vals[d]) for d in dates[i - win_days: i + 1]]
            seg_c = [(d, c_vals[d]) for d in dates[i - win_days: i + 1]]
            if len(seg_a) < win_days * 0.8:
                continue
            sb, sc = sharpe(seg_a), sharpe(seg_c)
            total += 1
            deltas.append(sc - sb)
            if sc > sb:
                wins_s += 1
            if c_vals[d1] / c_vals[d0] > a_vals[d1] / a_vals[d0]:
                wins_r += 1
        if total:
            f5[label] = {
                "n_windows": total,
                "sharpe_win_rate": round(wins_s / total, 4),
                "return_win_rate": round(wins_r / total, 4),
                "mean_sharpe_delta": round(sum(deltas) / len(deltas), 4),
            }
            print(f"  {label}: 夏普胜率 {wins_s}/{total} ({wins_s / total:.1%})  "
                  f"收益胜率 {wins_r}/{total} ({wins_r / total:.1%})  "
                  f"平均Δ夏普 {sum(deltas) / len(deltas):+.4f}")
    store["F5_rolling"] = f5

    # ══════════════════════════════════════════════════════════════
    # F6 gold 依赖度
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 78)
    print("F6 gold 依赖度 —— 组合里有多少是金价 beta？")
    print("=" * 78)
    # 剔除 gold 的定稿配置
    p6 = {c: pool[c] for c in FINAL_CLASSES if c != "gold" and pool.get(c)}
    cw6 = {c: 1.0 / len(p6) for c in p6}
    sl6, _, _ = build_sleeve(calendar, prices, p6, cw6, True)
    cb6 = combo(sl6)
    # gold 最差的 12 个月窗口
    g_vals = dict(gold_nav)
    g_dates = [d for d, _ in gold_nav]
    worst = None
    for i in range(260, len(g_dates)):
        ret = g_vals[g_dates[i]] / g_vals[g_dates[i - 260]] - 1
        if worst is None or ret < worst[0]:
            worst = (ret, g_dates[i - 260], g_dates[i])
    f6: Dict[str, Any] = {
        "corr_with_gold": round(corr(daily_rets(comb), daily_rets(gold_nav)), 4),
        "sleeve_corr_with_gold": round(corr(daily_rets(sleeve_nav), daily_rets(gold_nav)), 4),
        "sharpe_ex_gold": round(sharpe(cb6), 4),
        "sharpe_ex_gold_vs_base_pct": round((sharpe(cb6) / base_sharpe - 1) * 100, 2),
        "sharpe_ex_gold_minus_final": round(sharpe(cb6) - sharpe(comb), 4),
    }
    if worst:
        ret, d0, d1 = worst
        c_vals = dict(comb)
        f6["gold_worst_12m"] = {
            "gold_return": round(ret, 4),
            "window": [d0, d1],
            "combo_return": round(c_vals.get(d1, 0) / c_vals.get(d0, 1) - 1, 4) if d0 in c_vals and d1 in c_vals else None,
        }
        print(f"  gold 最差 12 月窗口 {d0}~{d1}: gold {ret:+.2%}  "
              f"组合 {f6['gold_worst_12m']['combo_return']:+.2%}")
    print(f"  与gold相关      组合 {f6['corr_with_gold']}  袖套 {f6['sleeve_corr_with_gold']}")
    print(f"  剔 gold 夏普    {f6['sharpe_ex_gold']:.4f} "
          f"({f6['sharpe_ex_gold_vs_base_pct']:+.2f}% vs 基线；"
          f"比定稿配置 {f6['sharpe_ex_gold_minus_final']:+.4f})")
    store["F6_gold_dependence"] = f6

    # ══════════════════════════════════════════════════════════════
    # F7 再平衡贡献（隔离）
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 78)
    print("F7 再平衡贡献 —— 同样持仓集，只加/去掉每月再平衡")
    print("=" * 78)
    f7: Dict[str, Any] = {}
    for label, reb in (("买入持有", False), ("每月再平衡", True)):
        sl7, rec7, _ = build_sleeve(calendar, prices, pool, cw, reb)
        cb7 = combo(sl7)
        s7 = sharpe(cb7)
        rec7 = {
            "combined_sharpe": round(s7, 4),
            "vs_baseline_pct": round((s7 / base_sharpe - 1) * 100, 2),
            "combined_return": round(total_ret(cb7), 4),
            "combined_maxdd": round(maxdd(cb7), 4),
            "combined_vol": round(ann_vol(cb7), 4),
            "turnover_ratio": rec7.get("turnover_ratio"),
            "commission": rec7.get("commission_total"),
        }
        f7[label] = rec7
        print(f"  {label:8s} 组合夏普 {s7:.4f} ({rec7['vs_baseline_pct']:+6.2f}%)  "
              f"收益 {rec7['combined_return']:+.2%}  回撤 {rec7['combined_maxdd']:.2%}  "
              f"波动 {rec7['combined_vol']:.4f}  换手 {rec7['turnover_ratio']}  "
              f"佣金 {rec7['commission']}")
    f7["rebal_sharpe_gain"] = round(
        f7["每月再平衡"]["combined_sharpe"] - f7["买入持有"]["combined_sharpe"], 4)
    f7["rebal_return_cost"] = round(
        f7["每月再平衡"]["combined_return"] - f7["买入持有"]["combined_return"], 4)
    print(f"  → 再平衡贡献 Δ夏普 {f7['rebal_sharpe_gain']:+.4f}，"
          f"收益代价 {f7['rebal_return_cost']:+.2%}")
    store["F7_rebalance_contribution"] = f7

    # ── 落盘 ──
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(store, fh, ensure_ascii=False, indent=1)
    print(f"\n[fr] ✓ 已写入 {out_path}", flush=True)


if __name__ == "__main__":
    main()
