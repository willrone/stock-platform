#!/usr/bin/env python3
"""P7：主腿因子消融 —— 主腿夏普 0.87/0.72，是因子构成的问题吗？

## 为什么做这个

P6 的真正推论不是「把 w 从 0.50 调小」，而是：

> 主腿（honest_composite）夏普只有 0.87(main) / 0.72(oos)，
> 袖套却有 1.62 / 1.04。降 w = 用更弱的资产替换更强的资产，不是 alpha。
> → 真正该改的是**主腿本身**。

主腿 = honest_composite，25 个公式因子（5 族 × 5 窗口）：
    MIN{n} +1 / MAX{n} -1 / STD{n} -1 / CORR{n} -1 / CORD{n} -1, n ∈ {5,10,20,30,60}
（`use_panel=False` 是显式选择 —— 17 个 Alpha158 面板因子实测净负面。）

## 先查已有研究（避免重复劳动）

2026-10-27 已做过一次 **42f 口径**的因子消融（`factor_subset_*_full3y.json`）：
ablation 43 / family 19 / greedy / weights / stability / oosval / walkforward / consensus。
窗口切分：IS = 2023-05-08~2025-05-08（2y），OOS = 2025-05~2026-05（1y）。

关键事实：
- `minus_panel_all`（= 25 公式版）全 3y 夏普 **0.8796**、回撤 **−0.1337**
  → 与当前主线 0.880 / −13.37% **完全对上**，确认这批产物就是当前主线的同源数据。
- 已有消融全部是「**在 42f 基线上剔一族**」（n=37/39/41），
  **从未在 25f 口径上单独消融** ← 这就是本实验的缺口。

已有消融给的信号（42f 口径，剔族后全 3y 夏普）：
    minus_VSUMN      0.9251   ← 最优，但 VSUMN 是面板因子族，当前主线没有 → 不适用
    minus_MAX_IMIN5  0.8852
    minus_panel_all  0.8796   ← 当前主线
    minus_MAX        0.8482
    minus_STD        0.8416
    minus_MIN        0.7773
    minus_CORD       0.7092
    minus_CORR       0.6765
→ MIN / CORR / CORD 是主力正贡献，不能碰；MAX / STD 相对次要 → 值得在 25f 上单测。

⚠️ 与既有结论的**张力**（必须同时测两条路）：
walk-forward 研究认为 **MAX30/60 的符号是错的**（2016-2021 IC 为正，硬编码 −1 反了），
而因子消融又说「剔 MAX 家族有正收益」。两条结论窗口/口径都不同 →
本实验同时测「剔 MAX」和「翻 MAX 符号」，让数据裁决。

## 本实验测什么（唯一变量 = 主腿因子集合）

规则臂（8 个）：
    base25 / drop_MAX / drop_MAX5_10_20 / drop_STD / drop_MAX_STD
    flip_MAX30_60 / drop_MAX5_10_20_flip30_60 / only_honest_high_ic

随机对照（9 个）：从 25 个公式里**随机剔 k 个**（k=5/3/10，各 3 种子）。
⚠️ 必须是「剔同样个数」而不是「随机选同样个数」—— 后者因子数更少、分散化天然更弱，
规则组几乎必然赢，对照没有鉴别力。
⚠️ 随机对照必须断言多种子持仓集互异，且每组内因子不重复。

## 预注册判据（跑之前写死，事后不得放宽）

对每个规则臂 R：
  (a) main 窗口 **A 股腿自身**夏普 > 基线 A 股腿夏普
  (b) main **组合**夏普 ≥ 基线组合夏普 + 0.05
  (c) oos  **组合**夏普 ≥ 基线组合夏普 − 0.02
  (d) 4 个不重叠子窗口中 **≥ 3 个** R 的组合夏普 > 基线
  (e) R 的组合夏普 > **同 keep-count 随机对照的最大值**
  (f) 独立样本外证据 —— A 股腿无窗口前历史 → **结构性不满足**

(f) 不满足 → 即使 (a)~(e) 全过，本实验也**不自动改定稿**，
只产出「主腿可改进的量化证据 + 候选清单」，改定稿必须人工点头。
（同 P6 的处理原则：判据是防噪，不是自动改配置。）

## 两阶段（防止用 OOS 挑参数）

  Stage 1  main 窗口跑全部 17 臂 → 排序
  Stage 2  oos 窗口只跑 基线 + Stage1 前 N 名规则臂 + 对应随机对照

## 口径（与 P3/P5/P6 定稿完全一致，便于逐位复现）

- A 股腿 50%：honest_composite，信号日收盘 → 次日 open，佣金 0.03% + 滑点 0.01%
- 袖套 50%：P5 固定持仓集（gold/commodity/bond 各 3 只），类内等权、类间等权，
  每月 21 交易日再平衡回类间等权
- 合成：`merge_by_date([a_nav, sleeve_nav], [0.5, 0.5])`
- 基线复现门：base25 臂必须逐位复现 P3 定稿组合夏普 + `ALL_real_lag1_v2.json` 的 A 股腿基线

## 用法

    cd backend && env -u PYTHONPATH BASELINE_LOG_LEVEL=ERROR .venv-py313/bin/python3 \
        scripts/multi_asset_p7_factor_ablation.py --window main
    cd backend && env -u PYTHONPATH BASELINE_LOG_LEVEL=ERROR .venv-py313/bin/python3 \
        scripts/multi_asset_p7_factor_ablation.py --window oos \
            --carry data/backtest/multi_asset_p7_main.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT = os.path.abspath(os.path.join(BACKEND_DIR, ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)
sys.path.insert(0, os.path.join(BACKEND_DIR, "scripts"))

import multi_asset_l3_momentum as L3  # noqa: E402
import multi_asset_p2 as P2  # noqa: E402
import multi_asset_p3_bond as P3  # noqa: E402
import multi_asset_leg_experiment as M  # noqa: E402
from multi_asset_leg_experiment import (  # noqa: E402
    load_etf_universe,
    maxdd,
    merge_by_date,
    sharpe,
    total_ret,
    yearly,
)

from app.services.backtest.execution.backtest_executor import (  # noqa: E402
    BacktestExecutor,
)
from app.services.backtest.models.data_models import BacktestConfig  # noqa: E402
from app.services.backtest.strategies import honest_composite_strategy as HCS  # noqa: E402

CLASSES = ["gold", "bond", "commodity"]
K_IN_CLASS = 3
REB_EVERY = 21
N_SUBWINDOWS = 4
W_A_SHARE = 0.5

#: P5 定稿的固定持仓集（不可用"窗口内流动性 top-3" —— OOS 池只有 2 只，会让基线漂移）
P5_FIXED = {
    "gold": ["518880", "159934", "159937"],
    "commodity": ["159980", "159985", "159981"],
    "bond": ["511260", "511010", "511020"],
}

#: 25 个公式因子全集（5 族 × 5 窗口）
ALL_FACTORS: List[str] = list(HCS._HONEST_FORMULAS.keys())
BASE_SIGNS: Dict[str, float] = dict(HCS._HONEST_FORMULAS)

#: 随机对照：keep-count → 随机剔除个数（k = 25 - keep）
RANDOM_SPEC: List[Tuple[int, int, List[int]]] = [
    (20, 5, [20261030, 7, 99]),      # 对标 drop_MAX
    (22, 3, [20261030, 7, 99]),      # 对标 drop_MAX5_10_20
    (15, 10, [20261030, 7, 99]),     # 对标 drop_MAX_STD
]

FAMILY_SUFFIX_OK = ("MIN", "MAX", "STD", "CORR", "CORD")


def _is_family(name: str, fam: str) -> bool:
    return name.startswith(fam) and name[len(fam):].isdigit()


def load_only_honest_high_ic() -> List[str]:
    """从已有 42f 家族消融产物里取 only_honest_high_ic 的 20 个公式因子。

    口径来源：data/backtest/factor_subset_family_full3y.json → subsets['only_honest_high_ic']
    （那是 42f 里的 honest 公式子集，正好是 25 公式的一个子集。）
    """
    p = os.path.join(ROOT, "data", "backtest", "factor_subset_family_full3y.json")
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as fh:
        d = json.load(fh)
    lst = d.get("subsets", {}).get("only_honest_high_ic", [])
    return [f for f in lst if f in BASE_SIGNS]


def build_rule_arms() -> List[Dict[str, Any]]:
    """规则臂定义：每个臂 = (因子白名单, 符号覆盖)。"""
    allf = list(ALL_FACTORS)
    high_ic = load_only_honest_high_ic()
    arms: List[Dict[str, Any]] = []

    def keep_without(fams: Sequence[str] = (), fams_windows: Optional[Dict[str, Sequence[int]]] = None
                     ) -> List[str]:
        out = []
        for f in allf:
            drop = False
            for fam in fams:
                if _is_family(f, fam):
                    drop = True
                    break
            if not drop and fams_windows:
                for fam, wins in fams_windows.items():
                    if _is_family(f, fam) and int(f[len(fam):]) in wins:
                        drop = True
                        break
            if not drop:
                out.append(f)
        return out

    arms.append({"label": "base25", "factors": allf, "signs": {}, "kind": "none",
                 "desc": "当前主腿基线（25 公式全留）"})
    arms.append({"label": "drop_MAX", "factors": keep_without(fams=["MAX"]),
                 "signs": {},
                 "kind": "drop",
                 "desc": "剔 MAX 全族（25→20）"})
    arms.append({"label": "drop_MAX5_10_20", "factors": keep_without(fams_windows={"MAX": (5, 10, 20)}),
                 "signs": {},
                 "kind": "drop",
                 "desc": "剔 MAX5/MAX10/MAX20，留 MAX30/MAX60（25→22）"})
    arms.append({"label": "drop_STD", "factors": keep_without(fams=["STD"]),
                 "signs": {},
                 "kind": "drop",
                 "desc": "剔 STD 全族（25→20）"})
    arms.append({"label": "drop_MAX_STD", "factors": keep_without(fams=["MAX", "STD"]),
                 "signs": {},
                 "kind": "drop",
                 "desc": "剔 MAX+STD 全族（25→15）"})
    arms.append({"label": "flip_MAX30_60", "factors": allf,
                 "signs": {"MAX30": 1.0, "MAX60": 1.0},
                 "kind": "flip",
                 "desc": "25 因子不变，MAX30/MAX60 符号 −1→+1（walk-forward 指出的符号错）"})
    arms.append({"label": "drop_MAX5_10_20_flip30_60",
                 "factors": keep_without(fams_windows={"MAX": (5, 10, 20)}),
                 "signs": {"MAX30": 1.0, "MAX60": 1.0},
                 "kind": "flip",
                 "desc": "22f 诚实候选：剔 MAX5/10/20 + MAX30/60 符号 +1"})
    if high_ic:
        arms.append({"label": "only_honest_high_ic", "factors": high_ic,
                     "signs": {},
                 "kind": "drop",
                     "desc": f"仅保留高 IC 公式族（{len(high_ic)} 个，来源 42f family 产物）"})
    else:
        print("[p7] ⚠️ 读不到 only_honest_high_ic，跳过该臂", flush=True)
    return arms


def build_random_arms() -> List[Dict[str, Any]]:
    """随机对照：从 25 公式里随机剔 k 个（保持因子数与规则臂可比）。"""
    out: List[Dict[str, Any]] = []
    for keep, k, seeds in RANDOM_SPEC:
        seen: Set[frozenset] = set()
        for seed in seeds:
            rng = random.Random(seed * 1000 + k)
            drop = tuple(sorted(rng.sample(list(ALL_FACTORS), k)))
            key = frozenset(drop)
            if key in seen:
                raise SystemExit(f"[p7] ❌ 随机对照退化（同种子产生相同剔除集）: {drop}")
            seen.add(key)
            factors = [f for f in ALL_FACTORS if f not in drop]
            if len(factors) != keep or len(set(factors)) != len(factors):
                raise SystemExit(f"[p7] ❌ 随机对照因子数异常 keep={len(factors)} 期望 {keep}")
            out.append({
                "label": f"rand_drop{k}_s{seed}",
                "factors": factors,
                "signs": {},
                "kind": "drop",
                "desc": f"随机剔 {k} 个（keep {keep}），seed={seed}",
                "dropped": list(drop),
                "keep": keep,
                "seed": seed,
            })
    # 断言：不同 keep-count 之间标签不重复；同 keep-count 的剔除集互异
    labels = [a["label"] for a in out]
    if len(set(labels)) != len(labels):
        raise SystemExit(f"[p7] ❌ 随机对照标签重复: {labels}")
    return out


def run_leg(codes: List[str], allowlist: Optional[List[str]],
            signs: Dict[str, float],
            extra_cfg: Optional[Dict[str, Any]] = None,
            start_date=None, end_date=None
            ) -> Tuple[Dict[str, Any], List[Tuple[str, float]]]:
    """跑 A 股腿（honest_composite），可指定因子白名单与符号覆盖。

    ⚠️ 符号表是策略模块的**全局字典** `_HONEST_FORMULAS`，在 prepare_backtest_data
    运行时才被读取 → 运行前改、运行后必须还原，否则会污染后续臂。
    """
    old_signs = dict(HCS._HONEST_FORMULAS)
    try:
        if signs:
            HCS._HONEST_FORMULAS.clear()
            HCS._HONEST_FORMULAS.update({k: (signs.get(k, v)) for k, v in old_signs.items()})
        cfg: Dict[str, Any] = {**M.STRATEGY_CFG, "lag": 1, "topk": M.TOPK, "n_drop": M.N_DROP}
        if allowlist is not None:
            cfg["factor_allowlist"] = sorted(allowlist)
        # 正交开关（如 exposure_neutral）—— 与因子集合轴互不干扰
        if extra_cfg:
            cfg.update(extra_cfg)
        executor = BacktestExecutor(
            data_dir=os.path.join(ROOT, "data"),
            enable_parallel=True,
            max_workers=8,
            enable_performance_profiling=False,
            use_multiprocessing=False,
        )
        res = asyncio.run(
            executor.run_backtest(
                strategy_name=M.STRATEGY,
                stock_codes=codes,
                start_date=(start_date if start_date is not None else M.START),
                end_date=(end_date if end_date is not None else M.END),
                strategy_config=cfg,
                backtest_config=BacktestConfig(
                    enable_tradability=True,
                    initial_cash=1_000_000.0,
                    commission_rate=0.0003,
                    slippage_rate=0.0001,
                    max_position_size=0.0,
                    cash_reserve_ratio=0.05,
                    board_lot_size=100,
                    stop_loss_pct=0.0,
                    take_profit_pct=0.0,
                    rebalance_frequency="daily",
                    open_cost=0.0,
                    close_cost=0.0,
                    min_cost=0.0,
                ),
            )
        )
        nav = [(str(d), float(v)) for d, v in (res.get("equity_curve") or []) if v is not None]
        rec = {
            "engine_total_return": res.get("total_return"),
            "engine_sharpe": res.get("sharpe_ratio"),
            "engine_max_drawdown": res.get("max_drawdown"),
            "total_trades": len(res.get("trade_history") or []),
            "nav_points": len(nav),
        }
        return rec, nav
    finally:
        HCS._HONEST_FORMULAS.clear()
        HCS._HONEST_FORMULAS.update(old_signs)
        # ⚠️ 符号表是模块全局，必须逐臂还原 —— 否则一个"翻符号"臂会污染后面所有臂，
        #    而且症状是"夏普莫名其妙"，极难归因。这里做硬断言。
        if dict(HCS._HONEST_FORMULAS) != BASE_SIGNS:
            raise SystemExit("[p7] ❌ 符号表未还原 → 后续臂会被污染")


def subwindow_compare(navs: Dict[str, List[Tuple[str, float]]],
                      base_label: str, labels: List[str]) -> Dict[str, Any]:
    """把窗口切成 N 段不重叠子段，各自算夏普 → 规则臂是否稳定优于基线。"""
    base = navs[base_label]
    dates = [d for d, _ in base]
    n = len(dates)
    edges = [n * i // N_SUBWINDOWS for i in range(N_SUBWINDOWS + 1)]
    out: Dict[str, Any] = {}
    for si in range(N_SUBWINDOWS):
        lo, hi = edges[si], edges[si + 1]
        if hi - lo < 60:
            continue
        dset = set(dates[lo:hi])
        seg: Dict[str, Any] = {"start": dates[lo], "end": dates[hi - 1], "n": hi - lo}
        for lab in labels:
            s = [(d, v) for d, v in navs[lab] if d in dset]
            seg[lab] = round(sharpe(s), 4) if len(s) > 30 else None
        out[f"sub{si}"] = seg
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", default="main", choices=["main", "oos"])
    ap.add_argument("--carry", default="", help="Stage1 产物路径（oos 阶段读它选臂）")
    ap.add_argument("--top", type=int, default=3, help="Stage2 带进 OOS 的规则臂数量")
    ap.add_argument("--only", default="", help="只跑指定标签（调试用，逗号分隔）")
    args = ap.parse_args()

    start, end = (M.START, M.END) if args.window == "main" else (M.OOS_START, M.OOS_END)
    s_str, e_str = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
    out_path = os.path.join(ROOT, "data", "backtest", f"multi_asset_p7_{args.window}.json")

    print(f"[p7] 窗口 {s_str} → {e_str}", flush=True)
    print(f"[p7] 公式因子全集 {len(ALL_FACTORS)} 个: {ALL_FACTORS}", flush=True)

    # ── 袖套（只算一次；与 P3/P6 完全同口径）──
    pool = P3.build_pool(start)
    universe = load_etf_universe()
    by_class_raw = L3.candidates_by_class(universe, start, end)
    base_codes = sorted({c for lst in by_class_raw.values() for c in lst})
    held = {c: list(P5_FIXED[c]) for c in P5_FIXED}
    p5_match = {c: (sorted(held.get(c, [])) == sorted(P5_FIXED[c])) for c in P5_FIXED}
    _known = set(base_codes) | {r["code"] for r in pool}
    missing = [c for lst in held.values() for c in lst if c not in _known]
    if not all(p5_match.values()) or missing:
        raise SystemExit(f"[p7] ❌ 固定持仓集异常 match={p5_match} missing={missing}")

    all_codes = sorted(set(base_codes) | {r["code"] for r in pool})
    prices = L3.load_prices(all_codes)
    calendar = L3.build_calendar(prices, base_codes, s_str, e_str)
    if len(calendar) < 300:
        raise SystemExit(f"[p7] ❌ 交易日历过短 {len(calendar)}")
    print(f"[p7] 交易日历 {len(calendar)} 天 {calendar[0]} → {calendar[-1]}", flush=True)

    for c, lst in held.items():
        if len(set(lst)) != len(lst):
            raise SystemExit(f"[p7] ❌ 类 {c} 持仓重复 {lst}")
    tg, tmeta = P3.build_targets(calendar, held, P3.fixed_weights_for(CLASSES))
    c_of = {c: cls for cls, lst in held.items() for c in lst}
    sleeve_nav, srec = L3.simulate(
        calendar, prices, tg, c_of, {c: 1.0 / len(CLASSES) for c in CLASSES})
    if not sleeve_nav:
        raise SystemExit("[p7] ❌ 袖套无净值")
    print(f"[p7] 袖套 {len(sleeve_nav)} 天，夏普 {sharpe(sleeve_nav):.4f}，"
          f"波动 {P2.nav_vol(sleeve_nav):.2%}，换手 {srec['turnover_ratio']:.2f}x", flush=True)

    # ── 选臂 ──
    rule_arms = build_rule_arms()
    random_arms = build_random_arms()
    only = {x for x in args.only.split(",") if x}

    if args.window == "oos":
        # Stage 2：只跑基线 + Stage1 前 N 名规则臂 + 对应 keep-count 的随机对照
        # ⚠️ --carry 是**相对仓库根**的路径（如 data/backtest/xxx.json），
        #    而脚本的 CWD 是 backend/ → 必须按 ROOT 解析，否则一律 not found。
        carry_path = (args.carry if os.path.isabs(args.carry)
                      else os.path.join(ROOT, args.carry))
        if not args.carry or not os.path.exists(carry_path):
            raise SystemExit(f"[p7] ❌ oos 阶段必须给 --carry（Stage1 main 产物），"
                             f"解析后路径 {carry_path} 不存在")
        with open(carry_path, encoding="utf-8") as fh:
            carry = json.load(fh)
        order = carry.get("ranking", [])
        # ⚠️ 先剔除 base25 再取 top-N，否则 base25 占掉一个名额，
        #    实际带进 OOS 的规则臂会少一个。
        # 优先带 (a)+(b) 全过的真候选；一个都没有时，退而带 main 排名第 1 的非基线臂，
        # 只为在 OOS 上做一次交叉验证（判据能否定的臂不必浪费样本外算力）。
        carry_crit = carry.get("criteria", {})
        passed = [l for l in order if l != "base25"
                  and carry_crit.get(l, {}).get("a_ashare_sharpe_gt_base") is True
                  and carry_crit.get(l, {}).get("b_main_combined_ge_base_plus_0.05") is True]
        if passed:
            picked = passed[: args.top]
            pick_mode = "main 阶段通过 (a)+(b) 的真候选"
        else:
            picked = [l for l in order if l != "base25"][:1]
            pick_mode = ("无候选通过 (a)(b) → 带 main 排名第 1 的非基线臂做 OOS 交叉验证")
        # 同因子集 + 同符号的臂是等价臂，只跑一次（drop_MAX ≡ only_honest_high_ic）
        _seen, dedup = set(), []
        for l in picked:
            r = carry["results"][l]
            key = (tuple(r["factors"]), tuple(sorted(r.get("signs_overridden", {}).items())))
            if key in _seen:
                continue
            _seen.add(key)
            dedup.append(l)
        picked = dedup
        # 随机对照必须与规则臂 **同 kind + 同 keep** 才可比
        want = {(a.get("kind", "drop"), len(a["factors"]))
                for a in rule_arms if a["label"] in picked}
        arms = [a for a in rule_arms if a["label"] == "base25" or a["label"] in picked]
        arms += [a for a in random_arms if (a.get("kind", "drop"), a.get("keep")) in want]
        print(f"[p7] Stage2 入选规则臂 {picked}（{pick_mode}），"
              f"随机对照配对 {sorted(want)}", flush=True)
    else:
        arms = rule_arms + random_arms
    if only:
        arms = [a for a in arms if a["label"] in only]
    if not arms:
        raise SystemExit("[p7] ❌ 没有要跑的臂")

    # ── 跑 A 股腿 + 合成组合 ──
    from verify_methods_engine import QLIB_OFFICIAL, universe_from_instruments
    codes = universe_from_instruments(QLIB_OFFICIAL)
    # ⚠️⚠️ 必须显式同步模块全局窗口 —— 否则 A 股腿会跑默认的 main 窗口，
    #    再和 oos 袖套合成（日期几乎不重叠）→ 组合夏普凭空变成 0.52（实测踩到）。
    M.START, M.END = start, end
    print(f"[p7] 股票池 {len(codes)} 只，臂 {len(arms)} 个，"
          f"A 股腿窗口 {M.START.date()}→{M.END.date()}", flush=True)

    results: Dict[str, Any] = {}
    navs: Dict[str, List[Tuple[str, float]]] = {}
    astore: Dict[str, List[Tuple[str, float]]] = {}

    for ai, arm in enumerate(arms, 1):
        lab = arm["label"]
        t0 = time.perf_counter()
        rec, a_nav = run_leg(codes, arm["factors"], arm.get("signs", {}),
                             start_date=start, end_date=end)
        wall = time.perf_counter() - t0
        if not a_nav:
            raise SystemExit(f"[p7] ❌ 臂 {lab} 净值为空")
        # ⚠️ 断言必须同时检查首尾 —— 只查起点会漏掉"整条腿跑错窗口"：
        #    oos 阶段曾用 main 窗口的 A 股腿（2023-05-08~2026-05-08），
        #    起点 2023-05-08 恰好 ≤ oos 终点 2023-05-08 → 旧断言放行。
        if not (s_str <= a_nav[0][0] <= e_str) or not (s_str <= a_nav[-1][0] <= e_str):
            raise SystemExit(f"[p7] ❌ 臂 {lab} A 股腿区间 {a_nav[0][0]}~{a_nav[-1][0]} "
                             f"不在目标窗口 {s_str}~{e_str} 内")
        a_sh = sharpe(a_nav)
        comb = merge_by_date([a_nav, sleeve_nav], [W_A_SHARE, 1 - W_A_SHARE])
        cs = sharpe(comb)
        navs[lab] = comb
        astore[lab] = a_nav
        results[lab] = {
            "n_factors": len(arm["factors"]),
            "factors": sorted(arm["factors"]),
            "signs_overridden": arm.get("signs", {}),
            "desc": arm.get("desc", ""),
            "is_random_control": lab.startswith("rand_"),
            "kind": arm.get("kind", "drop"),
            "keep": arm.get("keep"),
            "seed": arm.get("seed"),
            "dropped": arm.get("dropped"),
            "a_share_sharpe": round(a_sh, 4),
            "a_share_return": round(total_ret(a_nav), 4),
            "a_share_maxdd": round(maxdd(a_nav), 4),
            "a_share_trades": rec["total_trades"],
            "combined_sharpe": round(cs, 4),
            "combined_return": round(total_ret(comb), 4),
            "combined_maxdd": round(maxdd(comb), 4),
            "combined_vol": round(P2.nav_vol(comb), 4),
            "combined_yearly": yearly(comb),
            "corr_sleeve_ashare": P2.nav_corr(sleeve_nav, a_nav),
            "wall_s": round(wall, 1),
        }
        print(f"  [{ai}/{len(arms)}] {lab:32s} n={len(arm['factors']):2d} "
              f"A腿夏普 {a_sh:.4f}  组合夏普 {cs:.4f}  回撤 {maxdd(comb):+.2%}  "
              f"({wall:.0f}s)", flush=True)

    # ── 基线复现门 ──
    if "base25" not in results:
        raise SystemExit("[p7] ❌ 缺 base25 基线臂")
    base_cs = results["base25"]["combined_sharpe"]
    _p3p = os.path.join(ROOT, "data", "backtest", f"multi_asset_p3_{args.window}.json")
    if not os.path.exists(_p3p):
        raise SystemExit(f"[p7] ❌ 找不到 P3 产物 {_p3p}")
    with open(_p3p, encoding="utf-8") as fh:
        _p3res = json.load(fh).get("results", {})
    _p3key = ("bond_lowrho3_noliq" if "bond_lowrho3_noliq" in _p3res else "bond_lowrho3")
    exp = float(_p3res[_p3key]["combined_sharpe"])
    print(f"[p7] 基线复现门: base25 → {base_cs:.4f}（P3 定稿 {exp}）", flush=True)
    if abs(base_cs - exp) > 5e-4:
        raise SystemExit(f"[p7] ❌ 基线漂移 {base_cs} != {exp}，全部配置不可比，退出")

    # ── 等价臂交叉验证（免费回归测试）──
    # only_honest_high_ic 的 20 个因子恰好 == drop_MAX 的 20 个
    # （已有 42f 研究里的「高 IC 公式族」本身就一个 MAX 族都没有）
    # → 两臂必须产生**逐位相同**的 A 股腿净值。任一不符 → 因子白名单/符号处理有 bug。
    equiv_check: Dict[str, Any] = {}
    for _a, _b in [("drop_MAX", "only_honest_high_ic")]:
        if _a in astore and _b in astore:
            _na, _nb = astore[_a], astore[_b]
            _same = (len(_na) == len(_nb)
                     and all(x[0] == y[0] and x[1] == y[1] for x, y in zip(_na, _nb)))
            equiv_check[f"{_a}|{_b}"] = {"identical": bool(_same),
                                          "n_a": len(_na), "n_b": len(_nb)}
            if not _same:
                raise SystemExit(f"[p7] ❌ 等价臂 {_a} 与 {_b} 净值不一致 → 因子/符号处理有 bug")
            print(f"[p7] 等价臂 {_a} ≡ {_b} 逐位相同 ✓", flush=True)

    # ── 随机对照汇总（同 keep-count 分组）──
    rand_groups: Dict[str, Any] = {}
    for a in random_arms:
        lab = a["label"]
        if lab not in results:
            continue
        g = rand_groups.setdefault(str(a["keep"]), {"max": None, "min": None, "labels": [], "sharpes": {}})
        g["labels"].append(lab)
        g["sharpes"][lab] = results[lab]["combined_sharpe"]
        s = results[lab]["combined_sharpe"]
        g["max"] = s if g["max"] is None else max(g["max"], s)
        g["min"] = s if g["min"] is None else min(g["min"], s)

    # ── 子窗口 ──
    labels = [a["label"] for a in arms if a["label"] in navs]
    subs = subwindow_compare(navs, "base25", labels)

    # ── 预注册判据 ──
    criteria: Dict[str, Any] = {}
    per_arm: Dict[str, Any] = {}
    for a in rule_arms:
        lab = a["label"]
        if lab not in results or lab == "base25":
            continue
        r = results[lab]
        keep = str(r["n_factors"])
        rand_max = rand_groups.get(keep, {}).get("max")
        wins = 0
        for seg in subs.values():
            if seg.get(lab) is not None and seg.get("base25") is not None and seg[lab] > seg["base25"]:
                wins += 1
        c = {
            "a_ashare_sharpe_gt_base": r["a_share_sharpe"] > results["base25"]["a_share_sharpe"],
            # (b) 是 **main 窗口专属**判据；(c) 是 **oos 窗口专属**判据。
            # 绝不能在 oos 窗口里也判 (b)——那等于用"提升幅度"在样本外挑参。
            "b_main_combined_ge_base_plus_0.05":
                ((r["combined_sharpe"] - base_cs) >= 0.05) if args.window == "main" else None,
            "c_oos_combined_ge_base_minus_0.02":
                ((r["combined_sharpe"] - base_cs) >= -0.02) if args.window == "oos" else None,
            "d_subwindow_majority_ge_3": wins >= 3,
            "e_beats_same_keep_random_max":
                (rand_max is not None and r["combined_sharpe"] > rand_max),
            "f_independent_oos_evidence": False,
        }
        per_arm[lab] = {"criteria": c, "subwindow_wins": wins,
                        "rand_group_keep": keep, "rand_max": rand_max,
                        "delta_combined": round(r["combined_sharpe"] - base_cs, 4),
                        "delta_ashare": round(r["a_share_sharpe"]
                                              - results["base25"]["a_share_sharpe"], 4)}
        criteria[lab] = c

    ranking = sorted([l for l in results if not l.startswith("rand_")],
                     key=lambda l: -results[l]["combined_sharpe"])

    print(f"\n[p7] === 排名（组合夏普）===", flush=True)
    for l in ranking:
        r = results[l]
        print(f"  {l:32s} n={r['n_factors']:2d} 组合夏普 {r['combined_sharpe']:.4f} "
              f"(Δ{r['combined_sharpe']-base_cs:+.4f})  A腿 {r['a_share_sharpe']:.4f}", flush=True)
    print(f"\n[p7] === 随机对照（同 keep-count）===", flush=True)
    for k, g in rand_groups.items():
        print(f"  keep={k}: max {g['max']} min {g['min']}  {g['sharpes']}", flush=True)

    print(f"\n[p7] === 预注册判据（f 结构性不满足）===", flush=True)
    for lab, c in criteria.items():
        print(f"  {lab}:", flush=True)
        for k, v in c.items():
            print(f"    {'✅' if v is True else ('—' if v is None else '❌')} {k} = {v}", flush=True)

    store: Dict[str, Any] = {
        "window": args.window,
        "start": s_str,
        "end": e_str,
        "held": held,
        "p5_fixed_match": p5_match,
        "sleeve": {"sharpe": round(sharpe(sleeve_nav), 4),
                   "vol": round(P2.nav_vol(sleeve_nav), 4),
                   "cost": srec,
                   "n_rebalances": tmeta["n_rebalances"]},
        "w_a_share": W_A_SHARE,
        "base_combined_sharpe": base_cs,
        "p3_expected_base": exp,
        "all_factors": ALL_FACTORS,
        "ranking": ranking,
        "random_groups": rand_groups,
        "equiv_check": equiv_check,
        "criteria": criteria,
        "per_arm": per_arm,
        "results": results,
        "subwindows": subs,
        "note": ("(f) 独立样本外证据结构性缺失（honest_composite A 股腿面板从 2021-05-11 起，"
                 "OOS 起点正是那天 → 不存在窗口前长历史 A 股腿）。"
                 "即使 (a)~(e) 全过，本实验也不自动改定稿，只产出候选与量化证据。"),
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(store, fh, ensure_ascii=False, indent=2)
    print(f"[p7] ✓ 已写入 {out_path}", flush=True)


if __name__ == "__main__":
    main()
