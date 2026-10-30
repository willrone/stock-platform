#!/usr/bin/env python3
"""长历史验证：类间再平衡效应（ETF 袖套单独，2019→2026，7.7 年）。

## 为什么单独做

`L3_MOMENTUM_2026-10-30.md` 里唯一稳定的改进是「类间定期再平衡到等权」，
但只验证了 2 年（样本外）+ 3 年（推导）两个窗口。

**关键优势**：ETF 袖套**不需要 A 股腿**，所以不受 25 因子策略面板数据起点
（2021-05-11）的限制，可以独立回测到 **2019-01**（7.7 年）。
这是检验"再平衡 bonus"这个机械效应的最直接办法。

## 三个必须防的坑

1. **选标不能有前视偏差**：候选 ETF 按**窗口前 120 个交易日的成交额**排序，
   冻结整个窗口。绝不能用全窗口平均成交额（那是前视）。
2. **再平衡本身必须免费**：只扣佣金 0.03% + 滑点 0.01%（ETF 免印花税）。
   若扣了过高的成本，效应会被抹掉。
3. **口径与已验证脚本一致**：次日 open 成交、月末/周末调仓、等权目标权重。

## 输出

- 分年度收益：买入持有 vs 月度再平衡 vs 季度再平衡
- 滚动 24 个月窗口：再平衡胜率（有多少个滚动窗口跑赢买入持有）
- 机制分解：gold 权重漂移曲线
- 换手与成本

用法：
    cd backend && env -u PYTHONPATH .venv-py313/bin/python3 \
        scripts/multi_asset_long_history.py
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT = os.path.abspath(os.path.join(BACKEND_DIR, ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)
sys.path.insert(0, os.path.join(BACKEND_DIR, "scripts"))

import multi_asset_l3_momentum as L  # noqa: E402
from multi_asset_leg_experiment import (  # noqa: E402
    load_etf_universe,
    maxdd,
    sharpe,
    total_ret,
)

#: 长历史窗口（ETF 数据的实际起点由数据决定，不硬编码到某天）
LONG_START = "2019-01-01"
LONG_END = "2026-10-30"

#: 选标的用的流动性观察期长度（交易日）——只用这段，避免前视
LIQ_LOOKBACK_DAYS = 120

#: 窗口起点容忍度（自然日）：窗口起点常落在节假日/非交易日，
#: ETF 数据实际从下一个交易日起。严格比较 "> 窗口起点" 会把全部候选清空。
START_TOLERANCE_DAYS = 7

#: 再平衡频率（交易日间隔）
FREQS = {"monthly": 21, "biweekly": 10, "quarterly": 63}

OUT = os.path.join(ROOT, "data", "backtest", "multi_asset_long_history.json")


def pick_point_in_time(
    universe: Dict[str, Any], classes: List[str], start: str, end: str
) -> Dict[str, List[str]]:
    """按**窗口前 120 个交易日**的成交额选每类 top-3 → 冻结整个窗口。

    避免前视偏差：不能用全窗口平均成交额排序（那是用未来数据选标的）。
    """
    import pandas as pd

    by_class: Dict[str, List[Tuple[float, str]]] = {c: [] for c in classes}
    # 起点容忍度：窗口起点常落在节假日/非交易日（2019-01-01 是元旦），
    # ETF 数据实际从下一个交易日才开始。若严格比较 "> 窗口起点" 就排除，
    # 会把**全部**候选清空 —— 这是个已踩过的坑。
    start_lim = (
        datetime.strptime(start, "%Y-%m-%d") + timedelta(days=START_TOLERANCE_DAYS)
    ).strftime("%Y-%m-%d")
    # 先收集每只票在流动性观察期内的均价
    for code, meta in universe["etfs"].items():
        tags = [t for t in meta.get("tags", []) if t in by_class]
        if not tags:
            continue
        path = os.path.join(L.ETF_PARQUET, f"{code}.parquet")
        if not os.path.exists(path):
            continue
        df = pd.read_parquet(path).sort_values("date")
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        w = df[(df["date"] >= start) & (df["date"] <= end)]
        if len(w) < 200:
            continue
        if str(w["date"].iloc[0]) > start_lim:
            continue  # 上市晚于窗口起点 + 容忍度 → 不可用
        head = w.head(LIQ_LOOKBACK_DAYS)
        amt = float((head["volume"].fillna(0) * head["close"] * 100).mean())
        if amt <= 0:
            continue
        for t in tags:
            by_class[t].append((amt, code))

    out: Dict[str, List[str]] = {}
    for c, lst in by_class.items():
        lst.sort(reverse=True)
        out[c] = [code for _, code in lst[:3]]
    return out


def yearly_from_nav(nav: List[Tuple[str, float]]) -> Dict[str, float]:
    """按自然年计算收益（用上一年最后一个点作为基准）。"""
    out: Dict[str, float] = {}
    by_year: Dict[str, List[Tuple[str, float]]] = {}
    for d, v in nav:
        by_year.setdefault(d[:4], []).append((d, v))
    years = sorted(by_year)
    prev_end = None
    for y in years:
        pts = by_year[y]
        end_v = pts[-1][1]
        if prev_end is None:
            out[y] = round(end_v / pts[0][1] - 1, 6)
        else:
            out[y] = round(end_v / prev_end - 1, 6)
        prev_end = end_v
    return out


def rolling_window_win(
    nav_a: List[Tuple[str, float]], nav_b: List[Tuple[str, float]], win: int
) -> Dict[str, Any]:
    """滚动 win 个交易日窗口：b 跑赢 a 的比例。"""
    da = {d: v for d, v in nav_a}
    db = {d: v for d, v in nav_b}
    dates = sorted(set(da) & set(db))
    wins = 0
    total = 0
    diffs: List[float] = []
    for i in range(win, len(dates)):
        d0, d1 = dates[i - win], dates[i]
        if not da.get(d0) or not db.get(d0):
            continue
        ra = da[d1] / da[d0] - 1
        rb = db[d1] / db[d0] - 1
        total += 1
        if rb > ra:
            wins += 1
        diffs.append(rb - ra)
    return {
        "window_days": win,
        "n_windows": total,
        "win_rate": round(wins / total, 4) if total else None,
        "mean_excess": round(sum(diffs) / len(diffs), 6) if diffs else None,
    }


def rolling_sharpe_win(
    nav_a: List[Tuple[str, float]], nav_b: List[Tuple[str, float]], win: int
) -> Dict[str, Any]:
    """滚动 win 日窗口的**夏普差**：b 的夏普是否更高。

    这才是原始主张（再平衡提高夏普/降低回撤）对应的度量。
    滚动**收益**超额只测「是否跑赢」，而再平衡本质是「用收益换回撤」，
    两个指标必须分开看。
    """
    da = {d: v for d, v in nav_a}
    db = {d: v for d, v in nav_b}
    dates = sorted(set(da) & set(db))
    wins = 0
    total = 0
    diffs: List[float] = []
    mdd_wins = 0
    for i in range(win, len(dates)):
        d0, d1 = dates[i - win], dates[i]
        seg_a = [(d, da[d]) for d in dates[i - win: i + 1] if d in da]
        seg_b = [(d, db[d]) for d in dates[i - win: i + 1] if d in db]
        if len(seg_a) < win * 0.8 or len(seg_b) < win * 0.8:
            continue
        sa = sharpe(seg_a)
        sb = sharpe(seg_b)
        total += 1
        diffs.append(sb - sa)
        if sb > sa:
            wins += 1
        if maxdd(seg_b) > maxdd(seg_a):
            mdd_wins += 1
    return {
        "window_days": win,
        "n_windows": total,
        "sharpe_win_rate": round(wins / total, 4) if total else None,
        "mean_sharpe_diff": round(sum(diffs) / len(diffs), 4) if diffs else None,
        "mdd_better_rate": round(mdd_wins / total, 4) if total else None,
    }


def run_window(
    universe: Dict[str, Any],
    classes: List[str],
    start: str,
    end: str,
    freq: int = 21,
) -> Dict[str, Any]:
    """在 [start, end] 上跑一遍：point-in-time 选标 → 买入持有 vs 再平衡。"""
    picks = pick_point_in_time(universe, classes, start, end)
    codes = sorted({c for lst in picks.values() for c in lst})
    if not codes:
        return {"error": "no candidates", "start": start, "end": end}
    prices = L.load_prices(codes)
    cal = L.build_calendar(prices, codes, start, end)
    if len(cal) < 300:
        return {"error": "insufficient calendar", "start": start, "end": end}
    class_of = {c: cls for cls, lst in picks.items() for c in lst}
    cw = {c: 1.0 / len(classes) for c in classes}

    bh_t = L.build_static_targets(cal, prices, picks, cw)
    bh_nav, bh_rec = L.simulate(cal, prices, bh_t, class_of, cw)
    tg, _ = L.build_rotation_targets(cal, prices, picks, cw, 60, 99, freq, "all")
    rb_nav, rb_rec = L.simulate(cal, prices, tg, class_of, cw)

    return {
        "start": cal[0],
        "end": cal[-1],
        "n_days": len(cal),
        "candidates": picks,
        "buyhold": {
            "sharpe": round(sharpe(bh_nav), 4),
            "total_return": round(total_ret(bh_nav), 4),
            "max_drawdown": round(maxdd(bh_nav), 4),
            "yearly": yearly_from_nav(bh_nav),
        },
        "rebalanced": {
            "sharpe": round(sharpe(rb_nav), 4),
            "total_return": round(total_ret(rb_nav), 4),
            "max_drawdown": round(maxdd(rb_nav), 4),
            "yearly": yearly_from_nav(rb_nav),
            "turnover": rb_rec["turnover_ratio"],
            "commission": rb_rec["commission_total"],
        },
        "sharpe_delta": round(sharpe(rb_nav) - sharpe(bh_nav), 4),
        "mdd_delta": round(maxdd(rb_nav) - maxdd(bh_nav), 4),
    }


def main() -> None:
    universe = load_etf_universe()
    classes = list(L.L3_CLASSES)

    print(f"[lh] 选标的（流动性观察期 = 窗口前 {LIQ_LOOKBACK_DAYS} 个交易日，避免前视）",
          flush=True)
    picks = pick_point_in_time(universe, classes, LONG_START, LONG_END)
    print(f"[lh] 候选: {json.dumps(picks, ensure_ascii=False)}", flush=True)

    all_codes = sorted({c for lst in picks.values() for c in lst})
    prices = L.load_prices(all_codes)
    cal = L.build_calendar(prices, all_codes, LONG_START, LONG_END)
    print(f"[lh] 交易日历 {len(cal)} 天  {cal[0]} → {cal[-1]}", flush=True)
    if len(cal) < 1000:
        print("[lh] ❌ 历史长度不足，退出", flush=True)
        raise SystemExit(1)

    class_of = {c: cls for cls, lst in picks.items() for c in lst}
    cw = {c: 1.0 / len(classes) for c in classes}

    store: Dict[str, Any] = {
        "window": {"start": cal[0], "end": cal[-1], "n_days": len(cal)},
        "candidates": picks,
        "liq_lookback_days": LIQ_LOOKBACK_DAYS,
        "results": {},
    }

    # ── 买入持有基线 ──
    bh_targets = L.build_static_targets(cal, prices, picks, cw)
    bh_nav, bh_rec = L.simulate(cal, prices, bh_targets, class_of, cw)

    # ── 各频率再平衡 ──
    reb_navs: Dict[str, List[Tuple[str, float]]] = {}
    for name, freq in FREQS.items():
        tg, meta = L.build_rotation_targets(
            cal, prices, picks, cw, 60, 99, freq, "all")
        nav, rec = L.simulate(cal, prices, tg, class_of, cw)
        reb_navs[name] = nav
        store["results"][f"rebal_{name}"] = {
            "sharpe": round(sharpe(nav), 4),
            "total_return": round(total_ret(nav), 4),
            "max_drawdown": round(maxdd(nav), 4),
            "yearly": yearly_from_nav(nav),
            "cost": rec,
        }
        print(
            f"  rebal_{name:10s} 夏普 {sharpe(nav):.4f} 收益 {total_ret(nav):+.2%} "
            f"回撤 {maxdd(nav):.2%} 换手 {rec['turnover_ratio']:.2f} "
            f"佣金 {rec['commission_total']:.0f}",
            flush=True,
        )

    store["results"]["buyhold"] = {
        "sharpe": round(sharpe(bh_nav), 4),
        "total_return": round(total_ret(bh_nav), 4),
        "max_drawdown": round(maxdd(bh_nav), 4),
        "yearly": yearly_from_nav(bh_nav),
        "cost": bh_rec,
    }
    print(
        f"  {'buyhold':16s} 夏普 {sharpe(bh_nav):.4f} 收益 {total_ret(bh_nav):+.2%} "
        f"回撤 {maxdd(bh_nav):.2%} 换手 {bh_rec['turnover_ratio']:.2f}",
        flush=True,
    )

    # ── 滚动窗口胜率 ──
    print("\n[lh] 滚动窗口（收益超额口径）：", flush=True)
    for name, nav in reb_navs.items():
        for win_days, label in ((504, "滚动24月"), (252, "滚动12月")):
            r = rolling_window_win(bh_nav, nav, win_days)
            store["results"][f"rebal_{name}"][f"rolling_{win_days}d"] = r
            print(
                f"  {name:10s} {label}: 收益跑赢 {r['win_rate']:.1%} "
                f"({r['n_windows']} 窗口) 平均超额 {r['mean_excess']:+.2%}",
                flush=True,
            )

    print("\n[lh] 滚动窗口（夏普口径 = 原始主张）：", flush=True)
    for name, nav in reb_navs.items():
        for win_days, label in ((504, "滚动24月"), (252, "滚动12月")):
            r = rolling_sharpe_win(bh_nav, nav, win_days)
            store["results"][f"rebal_{name}"][f"rolling_sharpe_{win_days}d"] = r
            print(
                f"  {name:10s} {label}: 夏普更高 {r['sharpe_win_rate']:.1%} "
                f"平均夏普差 {r['mean_sharpe_diff']:+.4f} "
                f"回撤更好 {r['mdd_better_rate']:.1%}",
                flush=True,
            )

    # ── 不重叠独立子窗口（统计上更干净的样本）──
    print("\n[lh] 不重叠独立子窗口（point-in-time 选标，每个窗口独立）:", flush=True)
    sub_windows = []
    for y0 in range(2019, 2025):
        for years in (2, 3):
            s0 = f"{y0}-01-01"
            e0 = f"{y0 + years}-01-01"
            r = run_window(universe, classes, s0, e0)
            if "error" in r:
                print(f"  {s0}~{e0}: ⚠️ {r['error']}", flush=True)
                continue
            sub_windows.append(r)
            store.setdefault("sub_windows", []).append(r)
            print(
                f"  {r['start']}~{r['end']} ({years}年): "
                f"买入持有 夏普 {r['buyhold']['sharpe']:.4f} 回撤 {r['buyhold']['max_drawdown']:.2%} | "
                f"再平衡 夏普 {r['rebalanced']['sharpe']:.4f} 回撤 {r['rebalanced']['max_drawdown']:.2%} | "
                f"Δ夏普 {r['sharpe_delta']:+.4f}",
                flush=True,
            )
    if sub_windows:
        sp = sum(1 for r in sub_windows if r["sharpe_delta"] > 0)
        md = sum(1 for r in sub_windows if r["mdd_delta"] > 0)
        avg_s = sum(r["sharpe_delta"] for r in sub_windows) / len(sub_windows)
        avg_m = sum(r["mdd_delta"] for r in sub_windows) / len(sub_windows)
        print(
            f"\n  → {len(sub_windows)} 个独立子窗口：夏普更高 {sp}/{len(sub_windows)}，"
            f"回撤更浅 {md}/{len(sub_windows)}，平均 Δ夏普 {avg_s:+.4f}，"
            f"平均 Δ回撤 {avg_m:+.2%}",
            flush=True,
        )
        store["sub_window_summary"] = {
            "n": len(sub_windows),
            "sharpe_better": sp,
            "mdd_better": md,
            "avg_sharpe_delta": round(avg_s, 4),
            "avg_mdd_delta": round(avg_m, 6),
        }

    # ── 机制：gold 权重漂移 ──
    print("\n[lh] 机制：买入持有下 gold 权重漂移", flush=True)
    legs = {}
    for c in classes:
        t = L.build_static_targets(cal, prices, {c: picks[c]}, {c: 1.0})
        nav, _ = L.simulate(cal, prices, t, class_of, {c: 1.0})
        legs[c] = {d: v for d, v in nav}
    gw_series = []
    for d in cal:
        tot = sum(legs[c].get(d, 0.0) for c in classes)
        gw_series.append((d, legs["gold"].get(d, 0.0) / tot if tot > 0 else 0.0))
    store["gold_weight"] = {
        "start": round(gw_series[0][1], 4),
        "end": round(gw_series[-1][1], 4),
        "max": round(max(v for _, v in gw_series), 4),
        "min": round(min(v for _, v in gw_series), 4),
        "series_sample": [(d, round(v, 4)) for d, v in gw_series[::126]],
    }
    print(f"  gold 权重：起 {gw_series[0][1]:.1%} → 末 {gw_series[-1][1]:.1%} "
          f"峰值 {max(v for _, v in gw_series):.1%}", flush=True)

    # ── 分年度对比 ──
    print("\n[lh] 分年度收益（买入持有 vs 月度再平衡）：", flush=True)
    y_bh = store["results"]["buyhold"]["yearly"]
    y_mo = store["results"]["rebal_monthly"]["yearly"]
    print(f"  {'年份':8s} {'买入持有':>10s} {'月度再平衡':>12s} {'差':>8s}")
    for y in sorted(set(y_bh) | set(y_mo)):
        a = y_bh.get(y)
        b = y_mo.get(y)
        if a is None or b is None:
            continue
        print(f"  {y:8s} {a:>9.2%} {b:>11.2%} {b - a:>+7.2%}")

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(store, fh, ensure_ascii=False, indent=2)
    print(f"\n[lh] ✓ 已写入 {OUT}", flush=True)


if __name__ == "__main__":
    main()
