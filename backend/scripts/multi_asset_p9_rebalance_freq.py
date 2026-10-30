#!/usr/bin/env python3
"""P9：主腿执行层 —— 调仓频率曲线。

## 为什么做这个（承接 P6/P7/P8 的结论）

P7（因子消融）与 P8（暴露中性化）双双证伪 → **主腿弱（0.87 main / 0.72 oos）
不是"因子选错了"能修的**。P6 又证明组合对主腿权重 `w` 高度敏感。
剩下唯一没测过的维度是**执行层**：信号怎么合成已经排除了（因子层），
那么"多久换一次仓"就是剩下的主杠杆。

⚠️ 关键事实修正（2026-10-31 实施时发现）：**主腿本来就是周频**，不是日频 ——
`multi_asset_leg_experiment.STRATEGY_CFG = {"rebalance_days": "weekly"}`，
策略默认 `rebalance_weekday=4`（周五）。即：周五收盘出分 → 下一交易日（周一）执行。
所以本实验不是"降频省成本"，而是**频率曲线扫描**：日频 ↔ 周频 ↔ 月频，
周频（当前定稿）是曲线上的一个点，不是端点。

## 本实验测什么（唯一变量 = A 股腿调仓频率）

规则臂：
    wk_fri     基线。周五出分 → 周一执行（= 当前定稿）
    daily      日频出分 → 次日执行（频率上限）
    wk_mon     周一出分 → 周二执行（同一频率、不同日历锚点）
    monthly    每月 1 号出分 → 下一交易日执行（频率下限）

随机对照（**同调仓日数**、日期随机）：
    rand_wkA_s{seed}   对标 wk_fri（K≈104 个执行日）
    rand_wkB_s{seed}   对标 wk_mon（K≈104）
    rand_mo_s{seed}    对标 monthly（K≈24）

⚠️ 为什么必须要有"日期随机"对照：同样低的调仓频率，收益可能来自两件不同的事 ——
    (1) 换手减少 → 成本下降（**机械效应**，与日期无关）
    (2) 日历对齐（避开周一开盘、月末等系统性效应）—— 这是**真信息**
只有"同频随机日期"能把两者分开：
    规则臂 ≈ 随机组        ⇒ 纯机械效应，降频不是 alpha
    规则臂 > 随机组最大值  ⇒ 存在日历效应，频率选择本身携带信息

⚠️ 对照按**调仓日数**匹配，不是按交易笔数匹配 —— 每次调仓换多少只由排名决定，
会自然波动；匹配"机会数"才是频率本身的控制变量。

⚠️ daily 臂**没有**随机对照：日频的规则本身就是"每个交易日都调仓"，
同规模的随机子集恒等于规则本身，对照退化。这条在文档里显式说明。

## 实现方式（零生产代码改动）

- 规则臂直接用策略**已有**的 `rebalance_days` / `rebalance_weekday` 配置
  （`honest_composite_strategy.py`：非调仓日分数整行置 NaN → 执行器无分可依 → 不换仓）。
- 随机臂：实验脚本内给策略装补丁，在 `prepare_backtest_data` 之后
  **按日期过滤已存储的分数映射**（`_honest_composite_scores`）。
  掩码打在 lag 平移**之后** —— 而分数映射**只存有限分数的日期**，
  所以映射键就是**实际执行日**，规则臂与随机臂口径天然一致。
- 每臂都**实测**自己的执行日集合（从分数映射读），不靠推断。

## 预注册判据（跑之前写死，事后不得放宽）

对每个规则臂 R（vs 基线 wk_fri）：
  (a) main 窗口 **A 股腿自身**夏普 > 基线 A 股腿夏普
  (b) main **组合**夏普 ≥ 基线组合夏普 + 0.05
  (c) oos  **组合**夏普 ≥ 基线组合夏普 − 0.02
  (d) 4 个不重叠子窗口中 **≥ 3 个** R 的组合夏普 > 基线
  (e) R 的组合夏普 > **同调仓日数随机对照的最大值**
  (f) 独立样本外证据 —— A 股腿无窗口前历史 → **结构性不满足**

(f) 不满足 → 即使 (a)~(e) 全过，本实验也**不自动改定稿**，只产出量化证据。

## 两阶段（防止用 OOS 挑参数）

  Stage 1  main 窗口跑全部臂 → 排序
  Stage 2  oos 窗口只跑 基线 + Stage1 前 N 名规则臂 + 对应随机对照

## 口径（与 P3/P5/P6/P7 定稿完全一致，便于逐位复现）

- A 股腿 50%：honest_composite，信号日收盘 → 次日 open，佣金 0.03% + 滑点 0.01%
- 袖套 50%：P5 固定持仓集（gold/commodity/bond 各 3 只），类内等权、类间等权，
  每月 21 交易日再平衡回类间等权
- 合成：`merge_by_date([a_nav, sleeve_nav], [0.5, 0.5])`
- 基线复现门：wk_fri 臂必须逐位复现 P3 定稿组合夏普

## 用法
    cd backend && env -u PYTHONPATH BASELINE_LOG_LEVEL=ERROR .venv-py313/bin/python3 \
        scripts/multi_asset_p9_rebalance_freq.py --window main
    cd backend && env -u PYTHONPATH BASELINE_LOG_LEVEL=ERROR .venv-py313/bin/python3 \
        scripts/multi_asset_p9_rebalance_freq.py --window oos \
            --carry data/backtest/multi_asset_p9_main.json
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
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
N_SUBWINDOWS = 4
W_A_SHARE = 0.5
BASE_LABEL = "wk_fri"

P5_FIXED = {
    "gold": ["518880", "159934", "159937"],
    "commodity": ["159980", "159985", "159981"],
    "bond": ["511260", "511010", "511020"],
}

RAND_WK_SEEDS = [20261030, 7, 99]
RAND_MO_SEEDS = [20261030, 7]

#: 随机对照抽样时排除的头尾交易日 —— 让随机臂也有与周频臂相当的"建仓预热期"，
#: 否则随机组可能首日即建仓、白拿一段空仓期的机会差。
WARMUP_EXCLUDE_HEAD = 10
WARMUP_EXCLUDE_TAIL = 5

_SCORES_ATTR = "_honest_composite_scores"
_MASK: Dict[str, Optional[Set[dt.date]]] = {"allowed": None}
_LAST_SCORE_DATES: Set[dt.date] = set()

_ORIG_PREPARE = HCS.HonestCompositeStrategy.prepare_backtest_data


async def _patched_prepare(self, stock_data, start_date, end_date):  # type: ignore
    """原逻辑 + （实验用）按日期过滤分数映射 + 记录该臂真实执行日。

    分数映射**只存有限分数的日期**（策略源码：非有限值不入表），
    而 lag 平移在入表之前完成 → 映射键 = 实际执行日。
    所以掩码打在掩码之后过滤，规则臂与随机臂口径完全一致。
    """
    await _ORIG_PREPARE(self, stock_data, start_date, end_date)
    allowed = _MASK["allowed"]
    seen: Set[dt.date] = set()
    for _code, df in stock_data.items():
        sm = df.attrs.get(_SCORES_ATTR)
        if not sm:
            continue
        if allowed is not None:
            sm = {d: v for d, v in sm.items() if d.date() in allowed}
            df.attrs[_SCORES_ATTR] = sm
        for d in sm:
            seen.add(d.date())
    _LAST_SCORE_DATES.clear()
    _LAST_SCORE_DATES.update(seen)


HCS.HonestCompositeStrategy.prepare_backtest_data = _patched_prepare


def run_leg(codes: List[str], extra_cfg: Optional[Dict[str, Any]] = None,
            allowed: Optional[Set[dt.date]] = None,
            start_date=None, end_date=None
            ) -> Tuple[Dict[str, Any], List[Tuple[str, float]]]:
    """跑 A 股腿（honest_composite），可改调仓频率 / 施加日期掩码。"""
    _MASK["allowed"] = allowed
    _LAST_SCORE_DATES.clear()
    try:
        cfg: Dict[str, Any] = {**M.STRATEGY_CFG, "lag": 1,
                               "topk": M.TOPK, "n_drop": M.N_DROP}
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
        nav = [(str(d), float(v)) for d, v in (res.get("equity_curve") or [])
               if v is not None]
        rec = {
            "engine_total_return": res.get("total_return"),
            "engine_sharpe": res.get("sharpe_ratio"),
            "engine_max_drawdown": res.get("max_drawdown"),
            "total_trades": len(res.get("trade_history") or []),
            "nav_points": len(nav),
        }
        rec["rebalance_days"] = len(_LAST_SCORE_DATES)
        rec["rebalance_dates"] = sorted(d.isoformat() for d in _LAST_SCORE_DATES)
        return rec, nav
    finally:
        _MASK["allowed"] = None


def subwindow_compare(navs: Dict[str, List[Tuple[str, float]]],
                      base_label: str, labels: List[str]) -> Dict[str, Any]:
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


def _parse_dates(nav: Sequence[Tuple[str, float]]) -> List[dt.date]:
    return [dt.date.fromisoformat(d[:10]) for d, _ in nav]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", default="main", choices=["main", "oos"])
    ap.add_argument("--carry", default="", help="Stage1 产物路径（oos 阶段读它选臂）")
    ap.add_argument("--top", type=int, default=2, help="Stage2 带进 OOS 的规则臂数量")
    ap.add_argument("--only", default="", help="只跑指定标签（调试用，逗号分隔）")
    ap.add_argument("--no-random", action="store_true",
                    help="不跑随机对照（冒烟用）")
    args = ap.parse_args()

    start, end = (M.START, M.END) if args.window == "main" else (M.OOS_START, M.OOS_END)
    s_str, e_str = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
    out_path = os.path.join(ROOT, "data", "backtest", f"multi_asset_p9_{args.window}.json")

    print(f"[p9] 窗口 {s_str} → {e_str}", flush=True)
    print(f"[p9] 主腿基线调仓配置 M.STRATEGY_CFG = {M.STRATEGY_CFG}，"
          f"rebalance_weekday 默认 4(周五)", flush=True)

    # ── 袖套（只算一次；与 P3/P6/P7 完全同口径）──
    pool = P3.build_pool(start)
    universe = load_etf_universe()
    by_class_raw = L3.candidates_by_class(universe, start, end)
    base_codes = sorted({c for lst in by_class_raw.values() for c in lst})
    held = {c: list(P5_FIXED[c]) for c in P5_FIXED}
    p5_match = {c: (sorted(held.get(c, [])) == sorted(P5_FIXED[c])) for c in P5_FIXED}
    _known = set(base_codes) | {r["code"] for r in pool}
    missing = [c for lst in held.values() for c in lst if c not in _known]
    if not all(p5_match.values()) or missing:
        raise SystemExit(f"[p9] ❌ 固定持仓集异常 match={p5_match} missing={missing}")

    all_codes = sorted(set(base_codes) | {r["code"] for r in pool})
    prices = L3.load_prices(all_codes)
    calendar = L3.build_calendar(prices, base_codes, s_str, e_str)
    if len(calendar) < 300:
        raise SystemExit(f"[p9] ❌ 交易日历过短 {len(calendar)}")
    print(f"[p9] 交易日历 {len(calendar)} 天 {calendar[0]} → {calendar[-1]}", flush=True)

    for c, lst in held.items():
        if len(set(lst)) != len(lst):
            raise SystemExit(f"[p9] ❌ 类 {c} 持仓重复 {lst}")
    tg, tmeta = P3.build_targets(calendar, held, P3.fixed_weights_for(CLASSES))
    c_of = {c: cls for cls, lst in held.items() for c in lst}
    sleeve_nav, srec = L3.simulate(
        calendar, prices, tg, c_of, {c: 1.0 / len(CLASSES) for c in CLASSES})
    if not sleeve_nav:
        raise SystemExit("[p9] ❌ 袖套无净值")
    print(f"[p9] 袖套 {len(sleeve_nav)} 天，夏普 {sharpe(sleeve_nav):.4f}，"
          f"波动 {P2.nav_vol(sleeve_nav):.2%}，换手 {srec['turnover_ratio']:.2f}x", flush=True)

    # ── 规则臂 ──
    rule_arms: List[Dict[str, Any]] = [
        {"label": BASE_LABEL, "kind": "rule",
         "cfg": {"rebalance_days": "weekly", "rebalance_weekday": 4},
         "desc": "基线：周五出分 → 下一交易日（周一）执行（= 当前定稿）"},
        {"label": "daily", "kind": "rule",
         "cfg": {"rebalance_days": "daily"},
         "desc": "日频出分 → 次日执行（频率上限）"},
        {"label": "wk_mon", "kind": "rule",
         "cfg": {"rebalance_days": "weekly", "rebalance_weekday": 0},
         "desc": "周一出分 → 下一交易日（周二）执行（同频率，不同日历锚点）"},
        {"label": "monthly", "kind": "rule",
         "cfg": {"rebalance_days": "monthly"},
         "desc": "每月 1 号出分 → 下一交易日执行（频率下限）"},
    ]

    only = {x for x in args.only.split(",") if x}
    if args.window == "oos":
        # ⚠️ OOS 跑**完整设计**（全部规则臂 + 全部随机组），不是「只验证 main 选出的赢家」。
        # 理由：P9 的问题不是「哪个频率最好」，而是「日历对齐是否携带信息」——
        # 这需要基线 / 规则臂 / 同频随机对照**同时**出现在第二个窗口里，
        # 否则「规则臂 vs 随机」这一对照在 OOS 里无法成立。
        # 且 (f) 独立样本外证据结构性缺失 → 本实验无论结果如何都**不自动改定稿**，
        # 所以「用 OOS 挑参数」的风险被 (f) 兜住。
        if args.carry:
            carry_path = (args.carry if os.path.isabs(args.carry)
                          else os.path.join(ROOT, args.carry))
            if not os.path.exists(carry_path):
                raise SystemExit(f"[p9] ❌ --carry 路径不存在 {carry_path}")
            print(f"[p9] Stage2 携带 main 产物（仅作对照参考）：{carry_path}", flush=True)
        arms = list(rule_arms)
        print(f"[p9] Stage2 规则臂 {[_a['label'] for _a in arms]}"
              f"（完整设计，非候选筛选）", flush=True)
    else:
        arms = list(rule_arms)

    if only:
        arms = [a for a in arms if a["label"] in only]
    if not any(a["label"] == BASE_LABEL for a in arms):
        raise SystemExit("[p9] ❌ 缺基线臂")

    # ── 股票池 ──
    from verify_methods_engine import QLIB_OFFICIAL, universe_from_instruments
    codes = universe_from_instruments(QLIB_OFFICIAL)
    M.START, M.END = start, end
    print(f"[p9] 股票池 {len(codes)} 只，A 股腿窗口 {M.START.date()}→{M.END.date()}",
          flush=True)

    results: Dict[str, Any] = {}
    navs: Dict[str, List[Tuple[str, float]]] = {}
    astore: Dict[str, List[Tuple[str, float]]] = {}

    def run_arm(arm: Dict[str, Any], allowed: Optional[Set[dt.date]] = None) -> None:
        lab = arm["label"]
        t0 = time.time()
        rec, a_nav = run_leg(codes, extra_cfg=arm.get("cfg"), allowed=allowed,
                             start_date=start, end_date=end)
        if not a_nav:
            raise SystemExit(f"[p9] ❌ 臂 {lab} 无 A 股腿净值")
        a0 = dt.date.fromisoformat(a_nav[0][0][:10])
        a1 = dt.date.fromisoformat(a_nav[-1][0][:10])
        if not (start.date() <= a0 <= end.date() and start.date() <= a1 <= end.date()):
            raise SystemExit(f"[p9] ❌ 臂 {lab} A 股腿净值区间 [{a0},{a1}] 不在窗口 "
                             f"[{start.date()},{end.date()}]")
        comb = merge_by_date([a_nav, sleeve_nav], [W_A_SHARE, 1 - W_A_SHARE])
        if not comb:
            raise SystemExit(f"[p9] ❌ 臂 {lab} 合成失败")
        cs = sharpe(comb)
        a_sh = sharpe(a_nav)
        max_dd = maxdd(comb)
        _corr = P2.nav_corr(sleeve_nav, a_nav)
        results[lab] = {
            "desc": arm.get("desc", ""),
            "kind": arm.get("kind", "rule"),
            "cfg": arm.get("cfg", {}),
            "rebalance_days": rec.get("rebalance_days"),
            "rebalance_dates": rec.get("rebalance_dates", []),
            "total_trades": rec.get("total_trades"),
            "a_share_sharpe": round(a_sh, 4),
            "a_share_return": round(rec.get("engine_total_return") or 0.0, 4),
            "a_share_maxdd": round(rec.get("engine_max_drawdown") or 0.0, 4),
            "combined_sharpe": round(cs, 4),
            "combined_return": round(total_ret(comb), 4),
            "combined_maxdd": round(max_dd, 4),
            "combined_vol": round(P2.nav_vol(comb), 4),
            "combined_yearly": yearly(comb),
            "corr_sleeve_ashare": (round(_corr, 4) if _corr is not None else None),
            "wall_s": round(time.time() - t0, 1),
        }
        navs[lab] = comb
        astore[lab] = a_nav
        print(f"  [{len(results)}] {lab:16s} 执行日 {rec.get('rebalance_days'):4d} "
              f"交易 {rec.get('total_trades'):5d}  A腿夏普 {a_sh:.4f}  "
              f"组合夏普 {cs:.4f}  回撤 {max_dd:+.2%}  ({time.time()-t0:.0f}s)",
              flush=True)

    # 先跑基线（wk_fri），拿到 A 股腿交易日历 → 才能给随机臂抽样
    run_arm([a for a in arms if a["label"] == BASE_LABEL][0])
    cal_a = _parse_dates(astore[BASE_LABEL])
    print(f"[p9] A 股腿交易日历 {len(cal_a)} 天 {cal_a[0]} → {cal_a[-1]}", flush=True)
    for arm in arms:
        if arm["label"] == BASE_LABEL:
            continue
        run_arm(arm)

    # ── 随机对照：与规则臂**同执行日数**、日期随机 ──
    random_results: Dict[str, Any] = {}
    groups: List[Tuple[str, str, List[int]]] = []
    if args.window == "main":
        groups = [("wk_fri", "rand_wkA", RAND_WK_SEEDS),
                  ("wk_mon", "rand_wkB", RAND_WK_SEEDS),
                  ("monthly", "rand_mo", RAND_MO_SEEDS)]
    else:
        # OOS：同样的分组，K 按 **OOS 实测**执行日数重新抽样（窗口长度不同 → K 不同）
        groups = [("wk_fri", "rand_wkA", RAND_WK_SEEDS),
                  ("wk_mon", "rand_wkB", RAND_WK_SEEDS),
                  ("monthly", "rand_mo", RAND_MO_SEEDS)]

    for rule_label, rand_prefix, seeds in groups:
        if args.no_random:
            break
        if rule_label not in results:
            continue
        k = results[rule_label].get("rebalance_days") or 0
        if k <= 0:
            print(f"[p9] ⚠️ {rule_label} 没有实测到执行日，跳过其随机对照", flush=True)
            continue
        pool_dates = cal_a[WARMUP_EXCLUDE_HEAD:len(cal_a) - WARMUP_EXCLUDE_TAIL]
        if len(pool_dates) < k:
            raise SystemExit(f"[p9] ❌ 可抽样交易日 {len(pool_dates)} < 需要 {k}")
        seen: Set[frozenset] = set()
        for seed in seeds:
            rng = random.Random(seed * 7919 + k)
            picked = sorted(rng.sample(pool_dates, k))
            key = frozenset(picked)
            if key in seen:
                raise SystemExit(f"[p9] ❌ 随机对照退化（种子 {seed} 产生相同日期集）")
            seen.add(key)
            lab = f"{rand_prefix}_s{seed}"
            # ⚠️ cfg 必须显式 daily：若沿用基线的 weekly 掩码，
            #    分数在平移前就被砍成"只有周五"，随机臂将只能在周一交易 ——
            #    随机对照就被 weekly 规则污染，退化成"周频 ∩ 随机"。
            arm = {"label": lab, "kind": "random",
                   "cfg": {"rebalance_days": "daily"},
                   "desc": f"随机 {k} 个交易日调仓（对标 {rule_label}）"}
            run_arm(arm, allowed=set(picked))
            random_results[lab] = {
                "matched_rule": rule_label,
                "n_rebalance_days": k,
                "seed": seed,
                "dates": [d.isoformat() for d in picked],
            }

    base_cs = results[BASE_LABEL]["combined_sharpe"]
    base_a = results[BASE_LABEL]["a_share_sharpe"]

    # ── 基线复现门 ──
    _p3p = os.path.join(ROOT, "data", "backtest", f"multi_asset_p3_{args.window}.json")
    if not os.path.exists(_p3p):
        raise SystemExit(f"[p9] ❌ 找不到 P3 产物 {_p3p}")
    with open(_p3p, encoding="utf-8") as fh:
        _p3res = json.load(fh).get("results", {})
    _p3key = ("bond_lowrho3_noliq" if "bond_lowrho3_noliq" in _p3res else "bond_lowrho3")
    exp = float(_p3res[_p3key]["combined_sharpe"])
    print(f"[p9] 基线复现门: {BASE_LABEL} → {base_cs:.4f}（P3 定稿 {exp}）", flush=True)
    if abs(base_cs - exp) > 5e-4:
        raise SystemExit(f"[p9] ❌ 基线漂移 {base_cs} != {exp}，全部配置不可比，退出")

    # ── 子窗口 ──
    labels = list(results)
    subs = subwindow_compare(navs, BASE_LABEL, labels)

    # ── 判据 ──
    criteria: Dict[str, Any] = {}
    for lab in results:
        if lab == BASE_LABEL or lab.startswith("rand_"):
            continue
        r = results[lab]
        peers = [x for x in random_results
                 if random_results[x]["matched_rule"] == lab]
        peer_max = (max(results[x]["combined_sharpe"] for x in peers)
                    if peers else None)
        wins = sum(1 for s in subs.values()
                   if s.get(lab) is not None and s.get(BASE_LABEL) is not None
                   and s[lab] > s[BASE_LABEL])
        crit = {
            "a_ashare_sharpe_gt_base": r["a_share_sharpe"] > base_a,
            "b_main_combined_ge_base_plus_0.05":
                (r["combined_sharpe"] - base_cs) >= 0.05,
            "c_oos_combined_ge_base_minus_0.02": None,
            "d_subwindow_majority_ge_3": wins >= 3,
            "e_beats_same_freq_random_max": (None if peer_max is None
                                             else r["combined_sharpe"] > peer_max),
            "f_independent_oos_evidence": False,
        }
        criteria[lab] = {
            "random_group": sorted(peers),
            "random_group_max": (round(peer_max, 4) if peer_max is not None else None),
            "delta_vs_random_max": (round(r["combined_sharpe"] - peer_max, 4)
                                    if peer_max is not None else None),
            "subwindow_wins": wins,
            "delta_vs_base_combined": round(r["combined_sharpe"] - base_cs, 4),
            "delta_vs_base_ashare": round(r["a_share_sharpe"] - base_a, 4),
            "criteria": crit,
        }

    ranking = sorted(results, key=lambda k: -results[k]["combined_sharpe"])

    out = {
        "window": args.window,
        "start": s_str,
        "end": e_str,
        "held": held,
        "p5_fixed_match": p5_match,
        "sleeve": {
            "sharpe": round(sharpe(sleeve_nav), 4),
            "vol": round(P2.nav_vol(sleeve_nav), 4),
            "cost": {k: srec.get(k) for k in
                     ("commission_total", "turnover_total", "turnover_ratio")},
        },
        "w_a_share": W_A_SHARE,
        "base_label": BASE_LABEL,
        "base_combined_sharpe": base_cs,
        "base_ashare_sharpe": base_a,
        "p3_expected_base": exp,
        "a_calendar_days": len(cal_a),
        "random_controls": random_results,
        "results": results,
        "subwindows": subs,
        "criteria": criteria,
        "ranking": ranking,
        "note": (
            "基线 = wk_fri（周五出分→周一执行），即当前定稿 —— 主腿本来就是周频"
            "（multi_asset_leg_experiment.STRATEGY_CFG={'rebalance_days':'weekly'}），"
            "不是日频。随机对照按**同执行日数**匹配（不是同交易笔数）：低频收益可能来自"
            "「换手减少→成本下降」（机械效应，日期无关）或「日历对齐」（真信息）；"
            "规则臂 ≈ 随机组 ⇒ 纯机械，规则臂 > 随机组最大 ⇒ 有日历效应。"
            "daily 臂无随机对照（同规模的随机子集恒等于规则本身）。"
            "(f) 独立样本外证据结构性缺失（honest_composite A 股腿面板从 2021-05-11 起，"
            "OOS 起点正是那天）。"
        ),
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(f"[p9] ✅ 已写入 {out_path}", flush=True)
    print(f"[p9] 排名: {ranking}", flush=True)


if __name__ == "__main__":
    main()
