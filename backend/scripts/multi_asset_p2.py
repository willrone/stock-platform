#!/usr/bin/env python3
"""P2 —— 两个还没被证伪的类间/类内配置问题，一次做完。

## 为什么做这个

P1（跨类动量）已证伪，说明**类间权重的"择时"形式没有 alpha**。但类间权重的
**静态风险分配**形式还没被检验过 —— 这是本项目 ETF 袖套上唯一剩下的
"类间配置"自由度。

同时 L3 留了一个尾巴：商品类内部到底怎么选标。商品类只有 4 只候选，
其中 510170 在长历史上表现远好于其余三只，而定稿配置（top-3 流动性）
**恰好没有持有 510170**。这到底是"躲开了赢家"还是"碰巧"？必须分开看。

## 两个子实验

### P2a · 商品类内取舍（类内"策略分散"到底有没有价值）

commodity 类候选只有 4 只：159985（豆粕）、159981（能源化工）、159980（有色）、510170。
定稿持有 top-3 流动性 = 159985 + 159981 + 159980（**不含 510170**）。
本实验枚举类内持仓集，并给每只候选做**单只对照**（排除"碰巧选到赢家"）。

### P2b · 类间逆波动率配权（唯一没试过的类间配置形式）

权重 ∝ 1/类已实现波动率（用类内持有集合的等权收益序列），月度再平衡。
护栏（来自本项目风险平价踩过的坑）：单类权重上限 40% / 下限 8%，
年化波动率下限 2%（否则 bond 腿极低波会把权重吃光 → 伪夏普）。

对照：
- `cls_equal` 类间固定等权（定稿基线）
- `cls_rand_*` **每期重抽**的随机权重（Dirichlet(1)，同频率）→ 平凡解释排除
- `cls_rand_static_*` **全窗口固定**的随机配置（按月再平衡到固定权重）
  → 与等权**换手完全一致**的干净对照，排除"调权本身多转手"的混杂

## 口径

- 类内选标：除 P2a 外，全部固定 top-3 流动性（与定稿一致）
- 调仓：月度（21 交易日），信号日收盘 → 次日 open 成交
- 成本：佣金 0.03% + 滑点 0.01%（ETF 免印花税）
- A 股腿：50%（honest_composite），基线复现门不通过直接退出
- 长历史（`--window long`）：2019-01 → 2026-10，袖套单独（无 A 股腿），
  候选池按窗口前 120 交易日流动性冻结；滚动 504 日夏普胜率 + 12 个不重叠子窗口

## 用法
    cd backend && env -u PYTHONPATH BASELINE_LOG_LEVEL=ERROR .venv-py313/bin/python3 \
        scripts/multi_asset_p2.py --window main
    cd backend && env -u PYTHONPATH BASELINE_LOG_LEVEL=ERROR .venv-py313/bin/python3 \
        scripts/multi_asset_p2.py --window oos
    cd backend && env -u PYTHONPATH BASELINE_LOG_LEVEL=ERROR .venv-py313/bin/python3 \
        scripts/multi_asset_p2.py --window long
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
import os
import random
import statistics as st
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT = os.path.abspath(os.path.join(BACKEND_DIR, ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)
sys.path.insert(0, os.path.join(BACKEND_DIR, "scripts"))

import multi_asset_l3_momentum as L3  # noqa: E402
import multi_asset_long_history as LH  # noqa: E402
from multi_asset_leg_experiment import (  # noqa: E402
    END,
    OOS_END,
    OOS_START,
    START,
    load_etf_universe,
    maxdd,
    merge_by_date,
    sharpe,
    total_ret,
    yearly,
)

K_IN_CLASS = 3
REB_EVERY = 21
SEEDS = (20261030, 7, 99, 1234, 555, 8888)
CLASSES = ["gold", "bond", "commodity"]

VOL_FLOOR = 0.02          # 年化波动率下限 2%
WEIGHT_CAP = 0.40         # 单类权重上限
WEIGHT_FLOOR = 0.08       # 单类权重下限
TRADING_DAYS = 244

LONG_START = datetime(2019, 1, 1)
LONG_END = datetime(2026, 10, 30)


# ───────────────────────── 工具 ─────────────────────────

def nav_returns(nav: List[Tuple[str, float]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for i in range(1, len(nav)):
        p0, p1 = nav[i - 1][1], nav[i][1]
        if p0 > 0:
            out[nav[i][0]] = p1 / p0 - 1.0
    return out


def nav_vol(nav: List[Tuple[str, float]]) -> float:
    r = list(nav_returns(nav).values())
    if len(r) < 20:
        return 0.0
    mean = sum(r) / len(r)
    var = sum((x - mean) ** 2 for x in r) / (len(r) - 1)
    return math.sqrt(var) * math.sqrt(TRADING_DAYS)


def nav_corr(a: List[Tuple[str, float]], b: List[Tuple[str, float]]) -> Optional[float]:
    ra, rb = nav_returns(a), nav_returns(b)
    common = sorted(set(ra) & set(rb))
    if len(common) < 20:
        return None
    xa = [ra[d] for d in common]
    xb = [rb[d] for d in common]
    ma, mb = sum(xa) / len(xa), sum(xb) / len(xb)
    num = sum((xa[i] - ma) * (xb[i] - mb) for i in range(len(xa)))
    da = math.sqrt(sum((v - ma) ** 2 for v in xa))
    db = math.sqrt(sum((v - mb) ** 2 for v in xb))
    return round(num / (da * db), 4) if da > 0 and db > 0 else None


def class_vol(
    prices: Dict[str, Dict[str, Dict[str, float]]],
    codes: List[str],
    signal_date: str,
    lookback: int,
) -> Optional[float]:
    """类内持有集合的**等权日收益**序列在 signal_date 之前的年化波动率。"""
    idx = getattr(L3.load_prices, "index", {})
    per_code: List[List[float]] = []
    for c in codes:
        e = idx.get(c)
        if not e:
            return None
        ds = e["dates"]
        i = bisect.bisect_right(ds, signal_date) - 1
        if i < 0 or ds[i] != signal_date:
            return None
        if i - lookback < 0:
            return None
        per_code.append(e["close"][i - lookback: i + 1])
    n = len(per_code[0])
    rets: List[float] = []
    for k in range(1, n):
        r, ok = 0.0, True
        for s in per_code:
            p0, p1 = s[k - 1], s[k]
            if p0 <= 0:
                ok = False
                break
            r += p1 / p0 - 1.0
        if ok:
            rets.append(r / len(per_code))
    if len(rets) < 20:
        return None
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    return math.sqrt(var) * math.sqrt(TRADING_DAYS)


def cap_floor_weights(w: Dict[str, float]) -> Dict[str, float]:
    """把权重投影到 [floor, cap] 且和为 1（迭代投影）。"""
    keys = list(w.keys())
    if not keys:
        return {}
    n = len(keys)
    cap = min(WEIGHT_CAP, 1.0 - WEIGHT_FLOOR * (n - 1))
    floor = min(WEIGHT_FLOOR, 1.0 / n)
    if cap < floor:
        cap, floor = 1.0 / n, 1.0 / n
    w = dict(w)
    for _ in range(60):
        over = [k for k in keys if w[k] > cap + 1e-12]
        under = [k for k in keys if w[k] < floor - 1e-12]
        if not over and not under:
            break
        if over:
            excess = sum(w[k] - cap for k in over)
            for k in over:
                w[k] = cap
            rest = [k for k in keys if k not in over]
            if rest:
                s = sum(w[k] for k in rest)
                for k in rest:
                    w[k] += excess * (w[k] / s) if s > 0 else excess / len(rest)
        elif under:
            deficit = sum(floor - w[k] for k in under)
            for k in under:
                w[k] = floor
            rest = [k for k in keys if k not in under]
            s = sum(w[k] for k in rest)
            for k in rest:
                w[k] -= deficit * (w[k] / s) if s > 0 else deficit / len(rest)
    tot = sum(w.values()) or 1.0
    return {k: v / tot for k, v in w.items()}


def build_weighted_targets(
    calendar: List[str],
    by_class_held: Dict[str, List[str]],
    weight_fn,
    reb_every: int = REB_EVERY,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Any]]:
    """按 weight_fn(signal_date) -> {class: w} 生成调仓日目标权重（类内等权）。"""
    targets: Dict[str, Dict[str, float]] = {}
    log: List[Dict[str, Any]] = []
    for i in range(1, len(calendar)):
        if (i - 1) % reb_every != 0:
            continue
        signal_date, exec_date = calendar[i - 1], calendar[i]
        cw = weight_fn(signal_date)
        if not cw:
            continue
        t: Dict[str, float] = {}
        for c, w in cw.items():
            if w <= 0:
                continue
            codes = by_class_held.get(c) or []
            if not codes:
                continue
            for code in codes:
                t[code] = t.get(code, 0.0) + w / len(codes)
        if t:
            targets[exec_date] = t
            log.append({"date": signal_date, "exec": exec_date,
                        "class_weights": {k: round(v, 4) for k, v in cw.items()}})
    return targets, {"n_rebalances": len(log), "weight_log": log}


def fixed_weights(by_class_held: Dict[str, List[str]]):
    cls = [c for c in CLASSES if by_class_held.get(c)]
    w = {c: 1.0 / len(cls) for c in cls}
    return lambda _d: dict(w)


def rvol_weights(prices, by_class_held, lookback, use_cap=True):
    def fn(signal_date: str):
        vols = {}
        for c, codes in by_class_held.items():
            if not codes:
                continue
            v = class_vol(prices, codes, signal_date, lookback)
            if v is None or v <= 0:
                continue
            vols[c] = max(v, VOL_FLOOR)
        if len(vols) < 2:
            return fixed_weights(by_class_held)(signal_date)
        inv = {c: 1.0 / v for c, v in vols.items()}
        s = sum(inv.values())
        w = {c: v / s for c, v in inv.items()}
        return cap_floor_weights(w) if use_cap else w
    return fn


def random_weights(by_class_held, seed):
    """每期重抽的随机权重（Dirichlet(1)）。"""
    rng = random.Random(seed)
    def fn(_signal_date: str):
        cls = [c for c in CLASSES if by_class_held.get(c)]
        g = [rng.gammavariate(1.0, 1.0) for _ in cls]
        s = sum(g) or 1.0
        return {c: g[i] / s for i, c in enumerate(cls)}
    return fn


def static_random_weights(by_class_held, seed):
    """**全窗口固定**的随机配置：权重只抽一次，按月再平衡到它。

    → 与等权基线的换手完全一致，是"等权 vs 任意固定配置"的干净对照，
    排除"每期调权多转手"的成本混杂。
    """
    rng = random.Random(seed)
    cls = [c for c in CLASSES if by_class_held.get(c)]
    g = [rng.gammavariate(1.0, 1.0) for _ in cls]
    s = sum(g) or 1.0
    w = {c: g[i] / s for i, c in enumerate(cls)}
    return lambda _d: dict(w)


# ───────────────────────── 主流程 ─────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", default="main", choices=["main", "oos", "long"])
    ap.add_argument("--quick", action="store_true", help="读缓存的 A 股腿")
    args = ap.parse_args()

    import multi_asset_leg_experiment as M
    is_long = args.window == "long"
    if args.window == "main":
        start, end = START, END
    elif args.window == "oos":
        start, end = OOS_START, OOS_END
    else:
        start, end = LONG_START, LONG_END
    s_str, e_str = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

    out_path = os.path.join(ROOT, "data", "backtest", f"multi_asset_p2_{args.window}.json")
    nav_cache = os.path.join(ROOT, "data", "backtest", f"_a_share_nav_{args.window}.json")

    # ── A 股腿（长历史窗口没有 A 股腿：面板数据从 2021-05-11 起）──
    a_nav: List[Tuple[str, float]] = []
    a_rec: Dict[str, Any] = {}
    if not is_long:
        M.START, M.END = start, end   # ⚠️ 模块全局窗口，必须显式同步
        if args.quick and os.path.exists(nav_cache):
            with open(nav_cache, encoding="utf-8") as fh:
                cached = json.load(fh)
            a_nav = [(d, float(v)) for d, v in cached["a_share_nav"]]
            a_rec = cached.get("a_share_leg", {})
            print("[p2] 读缓存 A 股腿", flush=True)
        else:
            from verify_methods_engine import QLIB_OFFICIAL, universe_from_instruments
            codes = universe_from_instruments(QLIB_OFFICIAL)
            print(f"[p2] 跑 A 股腿（{len(codes)} 只）...", flush=True)
            import time as _t
            t0 = _t.perf_counter()
            a_rec, a_nav = M.run_a_share_leg(codes)
            a_rec["wall_s"] = round(_t.perf_counter() - t0, 1)
            ver = M.verify_baseline(a_rec)
            if args.window == "main":
                print(f"[p2] 基线复现门: {json.dumps(ver, ensure_ascii=False)}", flush=True)
                if ver.get("checked") and not ver.get("ok"):
                    raise SystemExit("[p2] ❌ 基线复现失败，退出")
        if not a_nav:
            raise SystemExit("[p2] ❌ A 股腿净值为空")
        if not (s_str <= a_nav[0][0] <= e_str):
            raise SystemExit(f"[p2] ❌ A 股腿起点 {a_nav[0][0]} 不在窗口 {s_str}~{e_str} 内")
        print(f"[p2] A 股腿 夏普 {a_rec.get('engine_sharpe')} "
              f"回撤 {a_rec.get('engine_max_drawdown')}", flush=True)

    # ── 候选池 / 持有集合 ──
    universe = load_etf_universe()
    if is_long:
        by_class_raw = LH.pick_point_in_time(
            universe, CLASSES, LONG_START.strftime("%Y-%m-%d"), LONG_END.strftime("%Y-%m-%d"))
    else:
        by_class_raw = L3.candidates_by_class(universe, start, end)
    held = {c: lst[:K_IN_CLASS] for c, lst in by_class_raw.items()}
    print(f"[p2] 候选: {json.dumps(by_class_raw, ensure_ascii=False)}", flush=True)
    print(f"[p2] 持有: {json.dumps(held, ensure_ascii=False)}", flush=True)

    all_codes = sorted({c for lst in by_class_raw.values() for c in lst})
    prices = L3.load_prices(all_codes)
    calendar = L3.build_calendar(prices, all_codes, s_str, e_str)
    print(f"[p2] 交易日历 {len(calendar)} 天 {calendar[0]} → {calendar[-1]}", flush=True)
    if len(calendar) < 300:
        raise SystemExit("[p2] ❌ 交易日历过短，退出")

    base_sharpe = sharpe(a_nav) if a_nav else None
    results: Dict[str, Any] = {}
    navs: Dict[str, List[Tuple[str, float]]] = {}
    store: Dict[str, Any] = {
        "window": args.window, "candidates": by_class_raw, "held_per_class": held,
        "calendar": {"n": len(calendar), "start": calendar[0], "end": calendar[-1]},
    }
    if a_rec:
        store["a_share_leg"] = a_rec

    def run_cfg(label: str, targets, meta, class_of):
        sleeve_nav, srec = L3.simulate(
            calendar, prices, targets, class_of, {c: 1.0 / len(CLASSES) for c in CLASSES})
        if not sleeve_nav:
            print(f"  {label}: ❌ 无净值", flush=True)
            return
        navs[label] = sleeve_nav
        rec: Dict[str, Any] = {
            "sleeve_sharpe": round(sharpe(sleeve_nav), 4),
            "sleeve_return": round(total_ret(sleeve_nav), 4),
            "sleeve_maxdd": round(maxdd(sleeve_nav), 4),
            "sleeve_vol": round(nav_vol(sleeve_nav), 4),
            "cost": srec,
            **meta,
        }
        line = (f"  {label:30s} 袖套夏普 {rec['sleeve_sharpe']:.4f}  "
                f"袖套波动 {rec['sleeve_vol']:.2%}  收益 {rec['sleeve_return']:+.2%}  "
                f"回撤 {rec['sleeve_maxdd']:.2%}")
        if a_nav:
            combined = merge_by_date([a_nav, sleeve_nav], [0.5, 0.5])
            cs = sharpe(combined)
            ng = [c for c in CLASSES if c != "gold"]
            ng_t = {}
            for d, t in targets.items():
                nt = {code: w * (1.0 / len(CLASSES)) / (1.0 / len(ng))
                      for code, w in t.items() if class_of.get(code) != "gold"}
                if nt:
                    ng_t[d] = nt
            ng_sleeve, _ = L3.simulate(calendar, prices, ng_t, class_of,
                                       {c: 1.0 / len(ng) for c in ng})
            ng_comb = merge_by_date([a_nav, ng_sleeve], [0.5, 0.5]) if ng_sleeve else []
            rec.update({
                "combined_sharpe": round(cs, 4),
                "vs_baseline": round(cs / base_sharpe - 1, 4) if base_sharpe else None,
                "combined_return": round(total_ret(combined), 4),
                "combined_maxdd": round(maxdd(combined), 4),
                "combined_yearly": yearly(combined),
                "sharpe_ex_gold": round(sharpe(ng_comb), 4) if ng_comb else None,
                "corr_sleeve_ashare": nav_corr(sleeve_nav, a_nav),
            })
            line = (f"  {label:30s} 组合夏普 {cs:.4f} ({rec['vs_baseline']:+.2%})  "
                    f"收益 {rec['combined_return']:+.2%}  回撤 {rec['combined_maxdd']:.2%}  "
                    f"| 袖套夏普 {rec['sleeve_sharpe']:.4f} 袖套波动 {rec['sleeve_vol']:.2%}  "
                    f"ρ(袖套,A) {rec['corr_sleeve_ashare']}  剔gold {rec['sharpe_ex_gold']}  "
                    f"换手 {srec['turnover_ratio']:.1f}x")
        results[label] = rec
        print(line, flush=True)

    # ═══════════ P2a · 商品类内取舍 ═══════════
    com = by_class_raw.get("commodity", [])
    print(f"\n[p2] === P2a · 商品类内持仓集（gold/bond 固定 top-3，类间等权 + 月度再平衡）===",
          flush=True)
    print(f"[p2] commodity 候选（按流动性）: {com}", flush=True)
    com_variants: List[Tuple[str, List[str]]] = [
        ("com_top1", com[:1]),
        ("com_top2", com[:2]),
        ("com_top3", com[:3]),          # ← 定稿现状
        ("com_all", com[:]),
    ]
    for c in com:
        com_variants.append((f"com_single_{c}", [c]))
    if len(com) >= 4:
        com_variants.append(("com_top2_plus_last", [com[0], com[3]]))
        com_variants.append(("com_top3_minus_first", com[1:3] + [com[3]]))
    for label, codes in com_variants:
        if not codes:
            continue
        h2 = dict(held)
        h2["commodity"] = codes
        tg, meta = build_weighted_targets(calendar, h2, fixed_weights(h2))
        c_of = {c: cls for cls, lst in h2.items() for c in lst}
        run_cfg(label, tg, {**meta, "rule": "commodity_inclass_variant",
                            "commodity_held": codes}, c_of)

    # 商品腿单标的自身表现
    print("\n[p2] --- 商品腿单标的自身表现 ---", flush=True)
    com_leg: Dict[str, Any] = {}
    for c in com:
        h1 = {"commodity": [c]}
        tg, _ = build_weighted_targets(calendar, h1, fixed_weights(h1))
        nav, rec = L3.simulate(calendar, prices, tg, {c: "commodity"}, {"commodity": 1.0})
        if nav:
            com_leg[c] = {"sleeve_sharpe": round(sharpe(nav), 4),
                          "sleeve_return": round(total_ret(nav), 4),
                          "sleeve_maxdd": round(maxdd(nav), 4),
                          "sleeve_vol": round(nav_vol(nav), 4)}
            print(f"  {c}: 夏普 {com_leg[c]['sleeve_sharpe']:.4f}  "
                  f"收益 {com_leg[c]['sleeve_return']:+.2%}  "
                  f"波动 {com_leg[c]['sleeve_vol']:.2%}  "
                  f"回撤 {com_leg[c]['sleeve_maxdd']:.2%}", flush=True)
    store["commodity_leg_single"] = com_leg

    # 商品类内相关性
    idx = getattr(L3.load_prices, "index", {})
    rets: Dict[str, List[float]] = {}
    for c in com:
        e = idx.get(c)
        if not e:
            continue
        cl = e["close"]
        rets[c] = [cl[k] / cl[k - 1] - 1.0 for k in range(1, len(cl)) if cl[k - 1] > 0]
    corr: Dict[str, Dict[str, float]] = {}
    if rets:
        n_min = min(len(v) for v in rets.values())
        for a in rets:
            corr[a] = {}
            for b in rets:
                xa, xb = rets[a][-n_min:], rets[b][-n_min:]
                ma, mb = sum(xa) / len(xa), sum(xb) / len(xb)
                num = sum((xa[k] - ma) * (xb[k] - mb) for k in range(len(xa)))
                da = math.sqrt(sum((v - ma) ** 2 for v in xa))
                db = math.sqrt(sum((v - mb) ** 2 for v in xb))
                corr[a][b] = round(num / (da * db), 4) if (da > 0 and db > 0) else 0.0
    store["commodity_correlation"] = corr
    print(f"[p2] 商品类内相关性: {json.dumps(corr, ensure_ascii=False)}", flush=True)

    # ═══════════ P2b · 类间权重 ═══════════
    print("\n[p2] === P2b · 类间权重（类内固定 top-3）===", flush=True)
    c_of = {c: cls for cls, lst in held.items() for c in lst}
    tg, meta = build_weighted_targets(calendar, held, fixed_weights(held))
    run_cfg("cls_equal", tg, {**meta, "rule": "class_equal_weight"}, c_of)

    for lb in (60, 120, 250):
        tg, meta = build_weighted_targets(calendar, held, rvol_weights(prices, held, lb, False))
        run_cfg(f"cls_rvol{lb}", tg, {**meta, "rule": "class_rvol_weight",
                                      "lookback": lb, "capped": False}, c_of)
    for lb in (60, 120, 250):
        tg, meta = build_weighted_targets(calendar, held, rvol_weights(prices, held, lb, True))
        run_cfg(f"cls_rvol{lb}_cap", tg, {**meta, "rule": "class_rvol_weight",
                                          "lookback": lb, "capped": True}, c_of)

    print("\n[p2] --- 随机权重对照 A：每期重抽（6 种子，Dirichlet(1)，同频率）---", flush=True)
    for seed in SEEDS:
        tg, meta = build_weighted_targets(calendar, held, random_weights(held, seed))
        run_cfg(f"cls_rand_s{seed}", tg, {**meta, "rule": "class_random_weight",
                                          "seed": seed}, c_of)

    print("\n[p2] --- 随机权重对照 B：全窗口固定（换手与等权一致，干净对照）---", flush=True)
    for seed in SEEDS:
        tg, meta = build_weighted_targets(calendar, held, static_random_weights(held, seed))
        run_cfg(f"cls_randstatic_s{seed}", tg, {**meta, "rule": "class_random_static",
                                                "seed": seed}, c_of)

    # 对照有效性自检
    for pref in ("cls_rand", "cls_randstatic"):
        vals = [results[f"{pref}_s{s}"]["sleeve_sharpe"] for s in SEEDS
                if f"{pref}_s{s}" in results]
        if len(set(vals)) < len(vals):
            raise SystemExit(f"[p2] ❌ 随机对照 {pref} 的种子结果相同 → 对照失效，退出。{vals}")
        print(f"[p2] ✓ {pref}: 6 种子互异（袖套夏普 {min(vals):.4f}~{max(vals):.4f}）", flush=True)

    # ── 长历史：滚动窗口 + 12 个不重叠子窗口 ──
    if is_long:
        base_nav = navs.get("cls_equal", [])
        print("\n[p2] === 长历史独立判定（相对 cls_equal）===", flush=True)
        for label, nav in navs.items():
            if label == "cls_equal":
                continue
            r504 = LH.rolling_sharpe_win(base_nav, nav, 504)
            r252 = LH.rolling_sharpe_win(base_nav, nav, 252)
            results[label]["rolling_sharpe_504d"] = r504
            results[label]["rolling_sharpe_252d"] = r252
            results[label]["delta_sharpe_vs_base"] = round(
                sharpe(nav) - sharpe(base_nav), 4)
            print(f"  {label:28s} Δ夏普 {results[label]['delta_sharpe_vs_base']:+.4f}  "
                  f"滚动504胜率 {r504['sharpe_win_rate']}  "
                  f"滚动252胜率 {r252['sharpe_win_rate']}", flush=True)

        n_sub = 12
        bounds = [calendar[min(int(i * len(calendar) / n_sub), len(calendar) - 1)]
                  for i in range(n_sub + 1)]
        sub_rows: Dict[str, List[Dict[str, Any]]] = {}
        for label, nav in navs.items():
            if label == "cls_equal":
                continue
            rows = []
            for i in range(n_sub):
                d0, d1 = bounds[i], bounds[i + 1]
                sb = [(d, v) for d, v in nav if d0 <= d < d1]
                sa = [(d, v) for d, v in base_nav if d0 <= d < d1]
                if len(sb) < 60 or len(sa) < 60:
                    continue
                rows.append({"i": i, "start": sb[0][0], "end": sb[-1][0],
                             "sharpe_base": round(sharpe(sa), 4),
                             "sharpe_cfg": round(sharpe(sb), 4),
                             "sharpe_delta": round(sharpe(sb) - sharpe(sa), 4)})
            sub_rows[label] = rows
            if rows:
                better = sum(1 for r in rows if r["sharpe_delta"] > 0)
                avg = sum(r["sharpe_delta"] for r in rows) / len(rows)
                print(f"  {label:28s} 12 子窗口夏普更好 {better}/{len(rows)}  "
                      f"平均 Δ {avg:+.4f}", flush=True)
        store["sub_windows"] = sub_rows

    store["results"] = results
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(store, fh, ensure_ascii=False, indent=2)
    print(f"\n[p2] ✓ 已写入 {out_path}", flush=True)

    if a_nav:
        with open(nav_cache, "w", encoding="utf-8") as fh:
            json.dump({"a_share_nav": a_nav, "a_share_leg": a_rec}, fh)


if __name__ == "__main__":
    main()
