#!/usr/bin/env python3
"""P17 · 类内选标的的 oracle 上界。

问题（P13 留下的口子 + 台账 §四 未探分支 #1）：
  三类的**固定持仓集**（gold 518880/159934/159937、bond 511260/511010/511020、
  commodity 159980/159985/159981）之外的**类内选标的**还有多少空间？

方法（P15 铁律：先算 oracle 上界，再谈优化器）：
  阶段 0  基线复现门 —— 定稿持仓集的 sleeve/combined 夏普必须逐位复现 P13 产物
  阶段 1  全枚举筛选（零成本向量化近似，仅用于**排序**）
  阶段 2  精确验证 —— 对 top-N + 定稿 + 随机族，用 P13 完整撮合器跑精确数值

口径（与 P13/P3 定稿一致）：
  袖套 = 3 类各 1/3、类内等权、每月（21 交易日）再平衡
  组合 = merge_by_date([A腿, 袖套], [0.5, 0.5])（按日对齐归一化加权 = 每日再平衡）
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT = os.path.abspath(os.path.join(BACKEND_DIR, ".."))
OUT_DIR = os.path.join(ROOT, "data", "backtest")
sys.path.insert(0, BACKEND_DIR)

import multi_asset_l3_momentum as L3  # noqa: E402
import multi_asset_p3_bond as P3  # noqa: E402
import multi_asset_p13_asset_weight as P13  # noqa: E402
from multi_asset_leg_experiment import (  # noqa: E402
    merge_by_date,
    sharpe,
    maxdd,
)
from multi_asset_p13_asset_weight import (  # noqa: E402
    corr,
    nav_return,
    nav_vol,
)

REB_EVERY = 21          # 与 P3/P4/P5/P13 同口径
A_LEG_W = 0.5           # A 股腿权重（P12 已证不可调）

# ── 候选池（P5 性质审计 + P3 bond 勘察的**全池**，含不流动的）──
POOL_GOLD = ["518880", "159934", "159937", "518800", "159812", "159830", "159831",
             "159834", "518600", "518660", "518680", "518850", "518860", "518890"]
POOL_COMMODITY = ["159980", "159985", "159981", "510170"]
POOL_BOND = {
    "main": ["511260", "511010", "511020", "511270", "511360",
             "511060", "511220", "511380", "511180"],
    "oos": ["511260", "511010", "511020", "511270", "511060", "511220", "511380"],
}

# ── 定稿固定持仓集（P5 定稿 / P3 §7.1）──
P5_FIXED: Dict[str, List[str]] = {
    "gold": ["518880", "159934", "159937"],
    "bond": ["511260", "511010", "511020"],
    "commodity": ["159980", "159985", "159981"],
}

START = datetime(2023, 5, 8)
END = datetime(2026, 5, 8)
OOS_START = datetime(2021, 5, 11)
OOS_END = datetime(2023, 5, 8)


# ══════════════════════════ 工具 ══════════════════════════
def reb_flags(T: int) -> np.ndarray:
    """调仓日标记：**与 P13.build_targets_code 完全同口径**。

    build_targets_code: for i in 1..len(calendar)-1: if (i-1) % reb_every != 0: continue
    → exec_date = calendar[i]，i ∈ {1, 22, 43, ...}
    """
    return np.array([(i - 1) % REB_EVERY == 0 for i in range(1, T)], dtype=bool)


def class_navs_approx(px: np.ndarray, reb: np.ndarray) -> np.ndarray:
    """批量零成本月度等权再平衡净值。px: (S, T, 3) 收盘价 → (S, T) 净值。

    仅用于**排序筛选**，不含成本、不用 open 成交价 → 不与撮合器逐位相等，
    阶段 2 会用完整撮合器纠正。
    """
    S, T, _ = px.shape
    nav = np.ones(S)
    w = np.full((S, 3), 1.0 / 3.0)
    out = np.empty((S, T))
    out[:, 0] = 1.0
    for t in range(1, T):
        r = px[:, t, :] / px[:, t - 1, :] - 1.0
        pr = (w * r).sum(axis=1)
        nav = nav * (1.0 + pr)
        w = w * (1.0 + r) / (1.0 + pr)[:, None]
        if reb[t - 1]:
            w[:] = 1.0 / 3.0
        out[:, t] = nav
    return out


def sleeve_nav_approx(class_navs: List[np.ndarray], reb: np.ndarray) -> np.ndarray:
    """3 个类净值按类间等权、月度再平衡合成袖套净值（零成本近似）。

    class_navs: 3 个 (N, T) 数组（同组 N 组）
    """
    N, T = class_navs[0].shape
    nav = np.ones(N)
    w = np.full((N, 3), 1.0 / 3.0)
    out = np.empty((N, T))
    out[:, 0] = 1.0
    for t in range(1, T):
        r = np.stack([class_navs[c][:, t] / class_navs[c][:, t - 1] - 1.0
                      for c in range(3)], axis=1)      # (N,3)
        pr = (w * r).sum(axis=1)
        nav = nav * (1.0 + pr)
        w = w * (1.0 + r) / (1.0 + pr)[:, None]
        if reb[t - 1]:
            w[:] = 1.0 / 3.0
        out[:, t] = nav
    return out


def sharpe_np(nav: np.ndarray) -> np.ndarray:
    """向量化夏普（与 multi_asset_leg_experiment.sharpe 同口径：pstdev, √244）。"""
    r = nav[:, 1:] / nav[:, :-1] - 1.0
    mu = r.mean(axis=1)
    sd = r.std(axis=1)                                # numpy 默认 ddof=0 = pstdev ✅
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(sd > 0, mu / sd * np.sqrt(244.0), 0.0)
    return out


def maxdd_np(nav: np.ndarray) -> np.ndarray:
    peak = np.maximum.accumulate(nav, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = nav / peak - 1.0
    return dd.min(axis=1)


def nav_return_np(nav: np.ndarray) -> np.ndarray:
    return nav[:, -1] / nav[:, 0] - 1.0


# ══════════════════════════ 数据 ══════════════════════════
class Ctx:
    def __init__(self, window: str):
        if window == "main":
            self.start, self.end = START, END
        else:
            self.start, self.end = OOS_START, OOS_END
        self.window = window
        self.s_str = self.start.strftime("%Y-%m-%d")
        self.e_str = self.end.strftime("%Y-%m-%d")

        # ── A 股腿净值（缓存优先；P13 产出的 _a_share_nav_{window}.json）──
        cache = os.path.join(OUT_DIR, f"_a_share_nav_{window}.json")
        with open(cache, encoding="utf-8") as fh:
            blob = json.load(fh)
        self.a_nav: List[Tuple[str, float]] = [(d, float(v)) for d, v in blob["a_share_nav"]]

        # ── ETF 价格 / 日历 ──
        # ⚠️ P3 教训 4：价格索引覆盖全部候选，但**交易日历只用基线候选并集**，
        #    否则基线夏普会漂移。
        # ⚠️⚠️ 「基线候选」= L3.candidates_by_class 的各类候选池并集（**不是**
        #    P5_FIXED 展开的 9 只）。用 9 只当 base_codes 会让交易日历变短、
        #    调仓日错位 → 复现门必然失败（P13 里就是 by_class_raw）。
        pool = P3.build_pool(self.start)
        universe = L3.load_etf_universe()
        by_class_raw = L3.candidates_by_class(universe, self.start, self.end)
        base_codes = sorted({c for lst in by_class_raw.values() for c in lst})
        all_codes = sorted(set(base_codes) | {r["code"] for r in pool}
                          | set(POOL_GOLD) | set(POOL_COMMODITY) | set(POOL_BOND[window]))
        self.prices = L3.load_prices(all_codes)
        self.idx = P13.build_price_index(self.prices)
        self.calendar = L3.build_calendar(self.prices, base_codes, self.s_str, self.e_str)
        if len(self.calendar) < 300:
            raise SystemExit(f"[p17] ❌ 交易日历过短 {len(self.calendar)}")
        print(f"[p17] 基线候选并集 {len(base_codes)} 只 → 交易日历 {len(self.calendar)} 天",
              flush=True)
        self.T = len(self.calendar)
        self.reb = reb_flags(self.T)
        print(f"[p17] 窗口 {window}  日历 {self.T} 天 "
              f"{self.calendar[0]} → {self.calendar[-1]}  "
              f"调仓 {int(self.reb.sum())} 次", flush=True)

        # ── 价格矩阵（对齐到日历）──
        self.px: Dict[str, np.ndarray] = {}
        for c in sorted(set(POOL_GOLD) | set(POOL_COMMODITY) | set(POOL_BOND[window])):
            self.px[c] = self._price_vec(c)
        missing = [c for c, v in self.px.items() if v is None]
        if missing:
            print(f"[p17] ⚠️ 无行情（窗口前无价格/未上市）: {missing}")

    def _price_vec(self, code: str) -> Optional[np.ndarray]:
        node = self.idx.get(code)
        if not node:
            return None
        # build_price_index 的结构：dates + close/open 数组
        import bisect
        out = np.full(self.T, np.nan)
        for i, d in enumerate(self.calendar):
            j = bisect.bisect_right(node["dates"], d) - 1
            if j >= 0:
                out[i] = float(node["close"][j])
        return out

    def price_stack(self, combos: List[Sequence[str]]) -> np.ndarray:
        """combos → (S, T, 3) 价格张量；任一成分缺价则该组合置 NaN。"""
        S = len(combos)
        arr = np.full((S, self.T, 3), np.nan)
        for si, combo in enumerate(combos):
            for k, c in enumerate(combo):
                v = self.px.get(c)
                if v is None:
                    continue
                arr[si, :, k] = v
        return arr


# ══════════════════════════ 阶段 0：基线复现门 ══════════════════════════
def exact_sleeve(ctx: Ctx, holdings: Dict[str, List[str]],
                 reb_every: int = REB_EVERY) -> Tuple[List[Tuple[str, float]], Dict]:
    """用 P13 完整撮合器跑一个袖套持仓集 → (净值, 记录)。"""
    codes = [c for lst in holdings.values() for c in lst]
    class_of = {c: cl for cl, lst in holdings.items() for c in lst}
    class_list = list(holdings.keys())

    def weight_fn(_signal_date: str, _i: int) -> Optional[Dict[str, float]]:
        w = 1.0 / len(codes)
        return {c: w for c in codes}

    targets, log = P13.build_targets_code(ctx.calendar, weight_fn, reb_every=reb_every)
    nav, rec = P13.simulate_portfolio(ctx.calendar, ctx.idx, targets)
    return nav, {"rec": rec, "log": log, "codes": codes,
                 "class_of": class_of, "class_list": class_list}


def exact_metrics(ctx: Ctx, nav: List[Tuple[str, float]]) -> Dict[str, Any]:
    comb = merge_by_date([ctx.a_nav, nav], [A_LEG_W, 1.0 - A_LEG_W])
    return {
        "sleeve_sharpe": round(sharpe(nav), 4),
        "sleeve_return": round(nav_return(nav), 6),
        "sleeve_maxdd": round(maxdd(nav), 6),
        "sleeve_vol": round(nav_vol(nav), 6),
        "combined_sharpe": round(sharpe(comb), 4),
        "combined_return": round(nav_return(comb), 6),
        "combined_maxdd": round(maxdd(comb), 6),
        "rho_sleeve_ashare": corr(nav, ctx.a_nav),
    }


def baseline_gate(ctx: Ctx) -> Dict[str, Any]:
    """定稿持仓集 → 必须逐位复现 P13 产物 base_equal。"""
    nav, _ = exact_sleeve(ctx, P5_FIXED)
    m = exact_metrics(ctx, nav)
    ref_path = os.path.join(OUT_DIR, f"multi_asset_p13_{ctx.window}.json")
    ref = None
    if os.path.exists(ref_path):
        with open(ref_path, encoding="utf-8") as fh:
            ref = json.load(fh).get("results", {}).get("base_equal")
    print(f"[p17] 基线复现门: 复算 {json.dumps(m, ensure_ascii=False)}", flush=True)
    if ref:
        keys = ["sleeve_sharpe", "combined_sharpe", "rho_sleeve_ashare"]
        diffs = {k: (ref.get(k), m.get(k)) for k in keys if ref.get(k) != m.get(k)}
        if diffs:
            raise SystemExit(f"[p17] ❌ 基线复现失败: {json.dumps(diffs, ensure_ascii=False)}")
        print("[p17] ✅ 基线复现门通过（与 P13 base_equal 逐位一致）", flush=True)
    else:
        print("[p17] ⚠️ 无 P13 参考产物，跳过复现门", flush=True)
    return m


# ══════════════════════════ 阶段 1：全枚举筛选（近似，仅排序）══
def single_scan(ctx: Ctx, cls: str) -> List[Dict[str, Any]]:
    """精确扫描某一类的全部 C(pool,3) 持仓集（其余两类固定为定稿）。

    全部走 P13 完整撮合器 → 与基线逐位同口径，**不做任何近似**。
    """
    pool = POOL_BOND[ctx.window] if cls == "bond" else \
        (POOL_GOLD if cls == "gold" else POOL_COMMODITY)
    out: List[Dict[str, Any]] = []
    for combo in itertools.combinations(pool, 3):
        hold = {c: list(P5_FIXED[c]) for c in P5_FIXED}
        hold[cls] = list(combo)
        nav, _ = exact_sleeve(ctx, hold)
        m = exact_metrics(ctx, nav)
        out.append({"holdings": hold, "combo": list(combo), **m})
        print(f"[p17] {cls} {list(combo)} 袖套 {m['sleeve_sharpe']:.4f} "
              f"组合 {m['combined_sharpe']:.4f} ρ {m['rho_sleeve_ashare']:.4f}",
              flush=True)
    return out


# ══════════════════════════ 阶段 2：精确验证 ══════════════════════════
def verify_phase(ctx: Ctx, cand: List[Dict[str, List[str]]]) -> Dict[str, Any]:
    out = []
    for i, hold in enumerate(cand):
        nav, _ = exact_sleeve(ctx, hold)
        m = exact_metrics(ctx, nav)
        out.append({"holdings": hold, **m})
        print(f"[p17] {i+1}/{len(cand)} {json.dumps(hold, ensure_ascii=False)} "
              f"袖套 {m['sleeve_sharpe']:.4f} 组合 {m['combined_sharpe']:.4f} "
              f"ρ {m['rho_sleeve_ashare']:.4f}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", default="main", choices=["main", "oos"])
    ap.add_argument("--phase", default="single",
                    choices=["gate", "single", "combine", "freq"])
    args = ap.parse_args()

    ctx = Ctx(args.window)
    base = baseline_gate(ctx)

    if args.phase == "gate":
        print(json.dumps(base, ensure_ascii=False, indent=2))
        return

    single_path = os.path.join(OUT_DIR, f"multi_asset_p17_single_{args.window}.json")

    if args.phase == "single":
        res: Dict[str, Any] = {"window": args.window, "baseline": base, "classes": {}}
        for cls in ["gold", "commodity", "bond"]:
            rows = single_scan(ctx, cls)
            rows.sort(key=lambda r: r["combined_sharpe"], reverse=True)
            res["classes"][cls] = {
                "n": len(rows),
                "fixed_other_classes": {c: list(P5_FIXED[c]) for c in P5_FIXED if c != cls},
                "best": rows[0],
                "worst": rows[-1],
                "rows": rows,
            }
            best = rows[0]
            print(f"[p17] {cls}: 最优 {best['combo']} 组合夏普 {best['combined_sharpe']:.4f} "
                  f"(基线 {base['combined_sharpe']:.4f}, "
                  f"增益 {best['combined_sharpe'] - base['combined_sharpe']:+.4f})", flush=True)
        with open(single_path, "w", encoding="utf-8") as fh:
            json.dump(res, fh, ensure_ascii=False)
        print(f"[p17] ✓ 已写入 {single_path}")
        return

    if args.phase == "freq":
        # 袖套调仓频率扫描：定稿持仓集固定，只改调仓周期
        # 买入持有 = 只在第一个可执行日调一次 → reb_every 极大
        res3 = []
        for label, re_ in [("weekly_5", 5), ("biweekly_10", 10), ("monthly_21", 21),
                           ("bimonthly_42", 42), ("quarterly_63", 63),
                           ("buyhold", 10 ** 6)]:
            nav, info = exact_sleeve(ctx, {c: list(P5_FIXED[c]) for c in P5_FIXED},
                                     reb_every=re_)
            m = exact_metrics(ctx, nav)
            m["label"] = label
            m["reb_every"] = re_
            m["n_rebalances"] = info["log"]["n_rebalances"]
            m["turnover_ratio"] = info["rec"]["turnover_ratio"]
            res3.append(m)
            print(f"[p17] {label:12s} 调仓 {m['n_rebalances']:3d} 次 "
                  f"换手 {m['turnover_ratio']:.2f}x  袖套 {m['sleeve_sharpe']:.4f} "
                  f"组合 {m['combined_sharpe']:.4f} ρ {m['rho_sleeve_ashare']:.4f}",
                  flush=True)
        out_path = os.path.join(OUT_DIR, f"multi_asset_p17_freq_{args.window}.json")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump({"window": args.window, "baseline": base, "results": res3},
                      fh, ensure_ascii=False)
        print(f"[p17] ✓ 已写入 {out_path}")
        return

    # combine：三个单类 oracle 组合 + 定稿 + 随机族
    with open(single_path, encoding="utf-8") as fh:
        single = json.load(fh)
    cands: List[Dict[str, List[str]]] = []
    labels: List[str] = []
    for cls, combo in [(c, single["classes"][c]["best"]["combo"]) for c in
                       ["gold", "commodity", "bond"]]:
        hold = {c: list(P5_FIXED[c]) for c in P5_FIXED}
        hold[cls] = list(combo)
        cands.append(hold)
        labels.append(f"single_oracle_{cls}")
    allor = {c: list(single["classes"][c]["best"]["combo"]) for c in
             ["gold", "commodity", "bond"]}
    cands.append(allor)
    labels.append("all_three_oracles")
    cands.append({c: list(P5_FIXED[c]) for c in P5_FIXED})
    labels.append("finalized_fixed")
    import random
    rng = random.Random(20261101)
    pools = {"gold": POOL_GOLD, "commodity": POOL_COMMODITY, "bond": POOL_BOND[args.window]}
    for s_i in range(20):
        cands.append({cl: sorted(rng.sample(pool, 3)) for cl, pool in pools.items()})
        labels.append(f"rand_s{s_i}")

    res2 = []
    for i, hold in enumerate(cands):
        nav, _ = exact_sleeve(ctx, hold)
        m = exact_metrics(ctx, nav)
        res2.append({"label": labels[i], "holdings": hold, **m})
        print(f"[p17] {labels[i]:26s} 袖套 {m['sleeve_sharpe']:.4f} "
              f"组合 {m['combined_sharpe']:.4f} ρ {m['rho_sleeve_ashare']:.4f}", flush=True)
    out_path = os.path.join(OUT_DIR, f"multi_asset_p17_verify_{args.window}.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({"window": args.window, "baseline": base, "results": res2},
                  fh, ensure_ascii=False)
    print(f"[p17] ✓ 已写入 {out_path}")


if __name__ == "__main__":
    main()
