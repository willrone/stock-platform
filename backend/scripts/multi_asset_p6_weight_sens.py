#!/usr/bin/env python3
"""P6：主腿权重敏感性 —— 「A 股腿占 50%」这个数字站得住吗？

## 为什么问这个

P3~P5 把资产侧榨干了：类内选标的、类间配权、扩充资产类全部证伪。
现在组合只剩一个可调维度没被检验过 —— **主腿（honest_composite A 股腿）的权重 w**。

- 当前定稿：w = 50%，袖套 50%（gold/commodity/bond 各 16.67%）
- 这个 50% 是**拍出来的**，不是推导出来的。

## 本实验的定位（重要）

**本实验的产出是「敏感性刻画」，不是「更优的 w」。**

原因是一个绕不开的结构性限制：

> honest_composite 的 A 股腿面板数据从 **2021-05-11** 起，OOS 窗口起点正是那一天。
> → **不存在"窗口前"的 A 股腿数据** → 无法做 point-in-time 的长历史 A 股腿。
> → w 的选择**没有独立于 main/oos 的第三个样本**。

所以任何"在 main 里找到更好的 w、再看 OOS 是不是也更好"的做法，本质上就是用 OOS 选参
（P1 跨类动量已经栽在同一个坑上：短窗口赢家在 7.7 年长历史上崩溃）。

## 预注册判据（跑之前写死，事后不得放宽）

采纳新 w **必须同时**满足：

- (a0) w 曲线必须有**内部极大值** —— 最优 w **不得落在网格端点**。
       落在端点 = 网格无效（曲线还没走完），这是"实验没做完"，不是"通过判据"。
       ⚠️ 第一版网格 [0.25, 0.75] 跑出来最优 w = 0.25（正好在左端点），
       若不加这条检测就会被误判成"最优 w 在 [0.40,0.60] 之外 → 通过 (a)"。
       → 网格必须扩到 [0.00, 1.00] 才能看到真实形状；
- (a) 最优 w **落在 [0.40, 0.60] 之外** —— 若最优点在区间内，说明 50% 已在最优附近，
      任何"改 w"都是噪声挖掘；
- (b) main 组合夏普 ≥ base + 0.05；
- (c) OOS 组合夏普 ≥ base − 0.02；
- (d) 4 个不重叠子窗口中 **≥ 3 个**指向同一 w 方向（曲线是尖峰还是稳定平台）；
- (e) 优于全部 6 个**随机 w** 对照的最大值（随机 w ~ U(0.30, 0.70)，固定种子）；
- (f) 有独立样本外证据支持。

**(f) 当前结构性不满足**（没有 A 股腿长历史）。
→ 因此本实验**默认不改定稿**；除非 (a)~(e) 全过，**且**另行补出长历史 A 股腿证据
（需要把 honest_composite 的因子面板往前扩到 2019，这是另一个工程量级的工作）。

即使 (a)~(e) 全过而 (f) 不满足，本实验的产出也只是**风险披露**
（"50% 站在悬崖边上"），不是 alpha 来源。

## 口径（与 P3 定稿完全一致，便于逐位复现）

- A 股腿 50%→w：honest_composite，信号日收盘 → 次日 open，佣金 0.03% + 滑点 0.01%
- 袖套：固定持仓集（gold/commodity 流动性 top-3 + bond 流动性达标池内 ρ 最低 top-3），
  类内等权、类间等权、每月 21 交易日再平衡回等权
- 合成：`merge_by_date([a_nav, sleeve_nav], [w, 1-w])` —— 每日在两腿之间再平衡回固定权重
- w 网格：0.00 ~ 1.00，步长 0.05（21 点，必须覆盖端点，否则看不到曲线拐点）
- 固定持仓集：**P5 定稿**（gold/commodity/bond 各 3 只）。⚠️ 不可用"窗口内流动性 top-3"
  口径 —— OOS 的流动性达标池只有 2 只，会让基线从 1.0685 漂到 1.0656
- 基线复现门：w = 0.5 必须逐位复现 P3 `bond_lowrho3` 的组合夏普

## 用法
    cd backend && env -u PYTHONPATH BASELINE_LOG_LEVEL=ERROR .venv-py313/bin/python3 \
        scripts/multi_asset_p6_weight_sens.py --window main
    cd backend && env -u PYTHONPATH BASELINE_LOG_LEVEL=ERROR .venv-py313/bin/python3 \
        scripts/multi_asset_p6_weight_sens.py --window oos
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT = os.path.abspath(os.path.join(BACKEND_DIR, ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)
sys.path.insert(0, os.path.join(BACKEND_DIR, "scripts"))

import multi_asset_l3_momentum as L3  # noqa: E402
import multi_asset_p2 as P2  # noqa: E402
import multi_asset_p3_bond as P3  # noqa: E402
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

CLASSES = ["gold", "bond", "commodity"]
K_IN_CLASS = 3
REB_EVERY = 21
SEEDS = (20261030, 7, 99, 1234, 555, 8888)

# w 网格：主腿权重
W_GRID = [round(0.05 * i, 2) for i in range(21)]
RAND_W_LO, RAND_W_HI = 0.30, 0.70
N_SUBWINDOWS = 4

# 基线复现门（P3 bond_lowrho3）
EXPECT_BASE = {}   # 运行时从 P3 产物读取（见 expected_base_from）

# P5 定稿的固定持仓集（用于交叉校验脚本算出的 held 是否一致）
P5_FIXED = {
    "gold": ["518880", "159934", "159937"],
    "commodity": ["159980", "159985", "159981"],
    "bond": ["511260", "511010", "511020"],
}



def expected_base_from(upstream_file: str, window: str, label: str) -> float:
    """基线期望值**读上游产物**，不硬编码数字。

    为什么：模拟引擎修 bug（如 L3.simulate 的现金约束顺序敏感性）后，
    所有下游数字都会变，硬编码的期望值会立刻把整条链卡死 —— 而它本来的作用
    是"防漂移"，不是"防数字变化"。改成读上游产物后，漂移照样能查出来
    （上游产物与本次结果不一致就报错），但数字更新不需要人肉同步 6 处。

    ⚠️ 调用方必须保证上游产物是**同一版引擎**跑出来的（重跑顺序要对）。
    """
    p = os.path.join(ROOT, "data", "backtest", upstream_file)
    if not os.path.exists(p):
        raise SystemExit(f"[gate] ❌ 找不到上游产物 {p}，无法建立基线门")
    with open(p, encoding="utf-8") as fh:
        d = json.load(fh)
    try:
        return float(d["results"]["base"]["combined_sharpe"])
    except (KeyError, TypeError):
        raise SystemExit(f"[gate] ❌ 上游产物 {upstream_file} 结构异常，无法取 base 组合夏普")


def nav_vol(nav) -> float:
    return P2.nav_vol(nav)


def subwindow_sharpes(navs: Dict[str, List[Tuple[str, float]]],
                      labels: List[str]) -> Dict[str, Dict[str, Any]]:
    """把窗口切成 N 段不重叠子段，各自算夏普 → 看曲线形状是否稳定。"""
    base = navs[labels[0]]
    dates = [d for d, _ in base]
    n = len(dates)
    edges = [n * i // N_SUBWINDOWS for i in range(N_SUBWINDOWS + 1)]
    out: Dict[str, Dict[str, Any]] = {}
    for si in range(N_SUBWINDOWS):
        lo, hi = edges[si], edges[si + 1]
        if hi - lo < 60:
            continue
        sub: Dict[str, Any] = {"start": dates[lo], "end": dates[hi - 1], "n": hi - lo}
        dset = set(dates[lo:hi])
        for lab in labels:
            seg = [(d, v) for d, v in navs[lab] if d in dset]
            sub[lab] = round(sharpe(seg), 4) if len(seg) > 30 else None
        # 该子段的最优 w
        vals = [(lab, sub[lab]) for lab in labels if sub.get(lab) is not None]
        if vals:
            sub["argmax_w"] = max(vals, key=lambda x: x[1])[0]
        out[f"sub{si}"] = sub
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", default="main", choices=["main", "oos"])
    ap.add_argument("--quick", action="store_true", help="读缓存的 A 股腿")
    args = ap.parse_args()

    import multi_asset_leg_experiment as M

    start, end = (START, END) if args.window == "main" else (OOS_START, OOS_END)
    s_str, e_str = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
    out_path = os.path.join(ROOT, "data", "backtest", f"multi_asset_p6_{args.window}.json")
    nav_cache = os.path.join(ROOT, "data", "backtest", f"_a_share_nav_{args.window}.json")

    # ── A 股腿 ──
    M.START, M.END = start, end   # ⚠️ 模块全局窗口，必须显式同步
    a_nav: List[Tuple[str, float]] = []
    a_rec: Dict[str, Any] = {}
    if args.quick and os.path.exists(nav_cache):
        with open(nav_cache, encoding="utf-8") as fh:
            cached = json.load(fh)
        a_nav = [(d, float(v)) for d, v in cached["a_share_nav"]]
        a_rec = cached.get("a_share_leg", {})
        print("[p6] 读缓存 A 股腿", flush=True)
    else:
        from verify_methods_engine import QLIB_OFFICIAL, universe_from_instruments
        codes = universe_from_instruments(QLIB_OFFICIAL)
        print(f"[p6] 跑 A 股腿（{len(codes)} 只）...", flush=True)
        a_rec, a_nav = M.run_a_share_leg(codes)
        ver = M.verify_baseline(a_rec)
        print(f"[p6] 基线复现门: {json.dumps(ver, ensure_ascii=False)}", flush=True)
        if ver.get("checked") and not ver.get("ok"):
            raise SystemExit("[p6] ❌ A 股腿基线复现失败，退出")
    if not a_nav:
        raise SystemExit("[p6] ❌ A 股腿净值为空")
    if not (s_str <= a_nav[0][0] <= e_str):
        raise SystemExit(f"[p6] ❌ A 股腿起点 {a_nav[0][0]} 不在窗口 {s_str}~{e_str} 内")
    base_sharpe = sharpe(a_nav)
    print(f"[p6] A 股腿 {len(a_nav)} 天，夏普 {base_sharpe:.4f}", flush=True)

    # ── 候选池 / 持仓集（与 P3 完全同口径，保证基线可逐位复现）──
    pool = P3.build_pool(start)
    universe = load_etf_universe()
    by_class_raw = L3.candidates_by_class(universe, start, end)
    held = {c: lst[:K_IN_CLASS] for c, lst in by_class_raw.items()}
    # ⚠️ 固定持仓集必须用 **P5 定稿**，不能用"窗口内流动性 top-3"：
    #    OOS 窗口的流动性达标池只有 2 只 → 用流动性口径会与定稿的 3 只不一致，
    #    基线从 1.0685 漂到 1.0656（实测踩到）。
    base_codes = sorted({c for lst in by_class_raw.values() for c in lst})
    held = {c: list(P5_FIXED[c]) for c in P5_FIXED}
    print(f"[p6] 固定持仓集(P5 定稿): {json.dumps(held, ensure_ascii=False)}", flush=True)
    # 交叉校验：P5 集合里的每只必须在候选池/价格索引里
    p5_match = {c: (sorted(held.get(c, [])) == sorted(P5_FIXED[c])) for c in P5_FIXED}
    _known = set(base_codes) | {r["code"] for r in pool}
    missing_codes = [c for lst in held.values() for c in lst if c not in _known]
    if not all(p5_match.values()) or missing_codes:
        raise SystemExit(f"[p6] ❌ 固定持仓集异常 match={p5_match} missing={missing_codes}")

    # ── 价格 / 交易日历（必须与 P3 同口径：日历按基线候选并集，价格索引含全池）──
    all_codes = sorted(set(base_codes) | {r["code"] for r in pool})
    prices = L3.load_prices(all_codes)
    calendar = L3.build_calendar(prices, base_codes, s_str, e_str)
    print(f"[p6] 交易日历 {len(calendar)} 天 {calendar[0]} → {calendar[-1]}", flush=True)
    if len(calendar) < 300:
        raise SystemExit("[p6] ❌ 交易日历过短，退出")

    # ── 袖套净值（只算一次；w 只影响两腿之间的合成比例）──
    for c, lst in held.items():
        if len(set(lst)) != len(lst):
            raise SystemExit(f"[p6] ❌ 类 {c} 持仓重复 {lst}")
    tg, tmeta = P3.build_targets(calendar, held, P3.fixed_weights_for(CLASSES))
    c_of = {c: cls for cls, lst in held.items() for c in lst}
    sleeve_nav, srec = L3.simulate(
        calendar, prices, tg, c_of, {c: 1.0 / len(CLASSES) for c in CLASSES})
    if not sleeve_nav:
        raise SystemExit("[p6] ❌ 袖套无净值")
    print(f"[p6] 袖套 {len(sleeve_nav)} 天，夏普 {sharpe(sleeve_nav):.4f}，"
          f"波动 {nav_vol(sleeve_nav):.2%}，ρ(袖套,A) {P2.nav_corr(sleeve_nav, a_nav)}，"
          f"换手 {srec['turnover_ratio']:.2f}x", flush=True)

    # ── w 网格 ──
    results: Dict[str, Any] = {}
    navs: Dict[str, List[Tuple[str, float]]] = {}

    def eval_w(w: float, label: str, meta: Optional[Dict[str, Any]] = None):
        comb = merge_by_date([a_nav, sleeve_nav], [w, 1.0 - w])
        cs = sharpe(comb)
        navs[label] = comb
        rec = {
            "w_a_share": round(w, 4),
            "combined_sharpe": round(cs, 4),
            "combined_return": round(total_ret(comb), 4),
            "combined_maxdd": round(maxdd(comb), 4),
            "combined_vol": round(nav_vol(comb), 4),
            "combined_yearly": yearly(comb),
            "sleeve_sharpe": round(sharpe(sleeve_nav), 4),
            "sleeve_vol": round(nav_vol(sleeve_nav), 4),
            "corr_sleeve_ashare": P2.nav_corr(sleeve_nav, a_nav),
            **(meta or {}),
        }
        results[label] = rec
        print(f"  {label:18s} w={w:.2f}  组合夏普 {cs:.4f}  收益 {rec['combined_return']:+.2%}  "
              f"回撤 {rec['combined_maxdd']:+.2%}  波动 {rec['combined_vol']:.2%}",
              flush=True)
        return rec

    print(f"\n[p6] === w 网格 ===", flush=True)
    for w in W_GRID:
        eval_w(w, f"w{int(round(w*100)):03d}")

    base_key = "w050"
    if base_key not in results:
        raise SystemExit("[p6] ❌ 缺 w=0.50 基线")
    base_cs = results[base_key]["combined_sharpe"]
    _p3p = os.path.join(ROOT, "data", "backtest", f"multi_asset_p3_{args.window}.json")
    if not os.path.exists(_p3p):
        raise SystemExit(f"[p6] ❌ 找不到 P3 产物 {_p3p}，无法建立基线门")
    with open(_p3p, encoding="utf-8") as fh:
        _p3d = json.load(fh)
    _p3res = _p3d.get("results", {})
    _p3key = ("bond_lowrho3_noliq" if "bond_lowrho3_noliq" in _p3res
              else "bond_lowrho3")
    exp = float(_p3res[_p3key]["combined_sharpe"])
    EXPECT_BASE[args.window] = exp
    print(f"[p6] 基线复现门: w=0.50 → {base_cs:.4f}（P3 定稿 {exp}）", flush=True)
    if abs(base_cs - exp) > 5e-4:
        raise SystemExit(f"[p6] ❌ 基线漂移 {base_cs} != {exp}，全部配置不可比，退出")

    # ── 随机 w 对照（隔离"w 是个可调参数"这件事本身）──
    print(f"\n[p6] === 随机 w 对照（U({RAND_W_LO}, {RAND_W_HI})，固定种子）===", flush=True)
    rnd_ws: List[float] = []
    for seed in SEEDS:
        rng = random.Random(seed)
        w = round(rng.uniform(RAND_W_LO, RAND_W_HI), 4)
        rnd_ws.append(w)
        eval_w(w, f"randw_s{seed}", {"note": f"随机 w={w}", "seed": seed})
    if len(set(rnd_ws)) != len(rnd_ws):
        raise SystemExit(f"[p6] ❌ 随机 w 退化（重复）: {rnd_ws}")

    # ── 曲线形状 ──
    grid = [(results[f"w{int(round(w*100)):03d}"]["combined_sharpe"], w)
            for w in W_GRID]
    best_s, best_w = max(grid)
    mid = [s for s, w in grid if 0.40 <= w <= 0.60]
    rng_mid = max(mid) - min(mid) if mid else None
    rand_max = max(results[f"randw_s{s}"]["combined_sharpe"] for s in SEEDS)
    rand_min = min(results[f"randw_s{s}"]["combined_sharpe"] for s in SEEDS)
    # 相邻网格点的最大跳变（尖峰检测）
    adj = [abs(grid[i + 1][0] - grid[i][0]) for i in range(len(grid) - 1)]
    max_jump = max(adj) if adj else None

    # ── 子窗口稳定性 ──
    print(f"\n[p6] === 子窗口稳定性（{N_SUBWINDOWS} 段不重叠）===", flush=True)
    labels = [f"w{int(round(w*100)):03d}" for w in W_GRID]
    subs = subwindow_sharpes(navs, labels)
    argmaxes = [v.get("argmax_w") for v in subs.values() if v.get("argmax_w")]
    mode_arg = max(set(argmaxes), key=argmaxes.count) if argmaxes else None
    mode_count = argmaxes.count(mode_arg) if mode_arg else 0
    for k, v in subs.items():
        print(f"  {k} {v['start']}~{v['end']} ({v['n']}天) 最优 w = {v.get('argmax_w')}  "
              f"w050 夏普 {v.get(base_key)}", flush=True)
    print(f"[p6] 子窗口最优 w 众数 = {mode_arg}（{mode_count}/{len(argmaxes)}）", flush=True)

    # ── 预注册判据判定 ──
    # ⚠️ 最优 w 落在**网格端点** = 网格无效（曲线还没走完），
    #    这不是"通过判据"，而是实验没做完 —— P1 踩过同类的坑。
    at_edge = (best_w <= min(W_GRID) + 1e-9) or (best_w >= max(W_GRID) - 1e-9)
    criteria = {
        "a0_curve_has_interior_max_not_edge": not at_edge,
        "a_best_w_outside_0.40_0.60": not (0.40 <= best_w <= 0.60),
        "b_main_sharpe_ge_base_plus_0.05": (best_s - base_cs) >= 0.05,
        "c_oos_sharpe_ge_base_minus_0.02": None,   # 另一个窗口跑完才判
        "d_subwindow_majority_same_w": mode_count >= 3,
        "e_beats_all_random_w": best_s > rand_max,
        "f_independent_oos_evidence": False,        # 无 A 股腿长历史 → 结构性不满足
    }
    passed = [k for k, v in criteria.items() if v is True]
    print(f"\n[p6] === 预注册判据（f 本窗口恒不满足）===", flush=True)
    for k, v in criteria.items():
        print(f"  {'✅' if v is True else ('—' if v is None else '❌')} {k} = {v}", flush=True)

    store: Dict[str, Any] = {
        "window": args.window,
        "start": s_str,
        "end": e_str,
        "held": held,
        "p5_fixed_match": p5_match,
        "a_share_sharpe": round(base_sharpe, 4),
        "sleeve": {"sharpe": round(sharpe(sleeve_nav), 4),
                   "vol": round(nav_vol(sleeve_nav), 4),
                   "corr_ashare": P2.nav_corr(sleeve_nav, a_nav),
                   "cost": srec,
                   "n_rebalances": tmeta["n_rebalances"]},
        "w_grid": W_GRID,
        "random_w": {str(s): rnd_ws[i] for i, s in enumerate(SEEDS)},
        "shape": {
            "best_w": best_w,
            "best_sharpe": round(best_s, 4),
            "best_w_at_grid_edge": at_edge,
            "grid_lo": min(W_GRID),
            "grid_hi": max(W_GRID),
            "mid_range_sharpe_span_040_060": round(rng_mid, 4) if rng_mid is not None else None,
            "max_adjacent_jump": round(max_jump, 4) if max_jump is not None else None,
            "random_w_sharpe_max": round(rand_max, 4),
            "random_w_sharpe_min": round(rand_min, 4),
            "subwindow_argmax_mode": mode_arg,
            "subwindow_mode_count": mode_count,
            "subwindow_n": len(argmaxes),
        },
        "criteria": criteria,
        "criteria_passed": passed,
        "results": results,
        "subwindows": subs,
        "note": ("本实验默认不改定稿：判据 (f) 需要独立样本外证据，"
                 "而 honest_composite 的 A 股腿面板从 2021-05-11 起，"
                 "不存在窗口前长历史 A 股腿 → (f) 结构性不满足。"
                 "即使 (a)~(e) 全过，产出也只是风险披露而非 alpha。"),
    }
    if a_rec:
        store["a_share_leg"] = a_rec
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(store, fh, ensure_ascii=False, indent=2)
    print(f"[p6] ✓ 已写入 {out_path}", flush=True)


if __name__ == "__main__":
    main()
