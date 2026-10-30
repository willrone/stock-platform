#!/usr/bin/env python3
"""P1 跨类动量 —— 在 gold/bond/commodity **三类之间**做战术配置。

## 为什么做这个

L3 的结论是「ETF **类内**动量没有 alpha」，而且后续头对头更狠：
同 L/K/频率配对下，动量选标只赢随机 **4/18**，L60_K1_monthly 甚至 **0/6**；
随机 K=1 比流动性 top1 高 +0.1258(main) / +0.0806(oos) —— 即**类内选标这一层
不但无 alpha，动量/流动性排名还不如乱选**。

这说明 ETF 腿的收益来源不在「类内选哪只」，而在：
1. A 股腿本身
2. **类间配置**（gold/bond/commodity 的低相关）
3. **定期再平衡**

那么真正没被检验过、且历史上真正有效的一层是 **类间配置**：
能不能按动量在三类之间切换权重，而不是固定 1/3-1/3-1/3？

## 实验设计（严格隔离每一层）

| 层 | 本实验怎么处理 |
|---|---|
| 类内选哪只 ETF | **全部固定为 top-3 流动性**（与定稿配置一致）→ 只测类间配置，不掺入类内选标 |
| 调仓机制 | 全部月度（21 交易日），次日 open 成交 |
| A 股腿 | 固定 50%，honest_composite，不动 |
| 袖套权重 | **唯一变量**：类间怎么分 |

配置矩阵：

- `xclass_mom_topM`（M=1,2）：按类动量选最好的 M 个类，等权
- `xclass_mom_weighted`：三类全留，权重 ∝ max(0, 类动量)
- `xclass_mom_top1_abs`：动量最好的类若绝对动量 < 0 → 转防御（债券）
- `xclass_rand_topM`（M=1,2，6 个种子）：**随机选 M 个类** → 平凡解释排除
- `xclass_rand_weighted`（6 个种子）：随机权重（Dirichlet）→ 同上

**random 对照的硬要求**（L3 踩过的坑）：同频率、同持仓类数、同类内规则，
多种子。否则「持仓集中度」和「选标能力」混淆。

## 判据（事前声明）

只有**同时**满足下面三条才算 P1 成立：
1. 动量版本在**两个窗口**都优于**同参数随机对照的均值**；
2. 动量版本在**两个窗口**都优于**定稿基线** static_top3_rebal_monthly；
3. 换手/佣金没有失控（不能靠高换手赌出来的夏普）。

任何一条不满足 → P1 证伪，类间固定等权 + 每月再平衡保持定稿。

## 用法
    cd backend && env -u PYTHONPATH BASELINE_LOG_LEVEL=ERROR .venv-py313/bin/python3 \
        scripts/multi_asset_xclass.py --window main
    cd backend && env -u PYTHONPATH BASELINE_LOG_LEVEL=ERROR .venv-py313/bin/python3 \
        scripts/multi_asset_xclass.py --window oos
"""
from __future__ import annotations

import argparse
import json
import os
import random
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
    END,
    OOS_END,
    OOS_START,
    START,
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

#: 本实验的资产类别（与定稿配置一致）
XCLASSES = ["gold", "bond", "commodity"]

#: 类内固定持仓数（= 定稿配置的 top-3 流动性）。**本实验不测类内选标**。
K_IN_CLASS = 3

#: 调仓频率（交易日间隔）
REB_EVERY = 21

#: 随机对照种子
SEEDS = (20261030, 7, 99, 1234, 555, 8888)

#: 动量回看窗口
LOOKBACKS = (60, 120, 250)


# ---------------------------------------------------------------------------
# 类间配置的目标权重生成
# ---------------------------------------------------------------------------
def class_momentum_scores(
    prices: Dict[str, Dict[str, Dict[str, float]]],
    by_class_held: Dict[str, List[str]],
    signal_date: str,
    lookback: int,
) -> Dict[str, float]:
    """每类的动量 = 该类**实际持有集合**（top3 流动性）动量的等权平均。

    用持有集合而不是全候选池：这样类间信号与最终持仓严格一致，
    不会出现「用没买的票算信号」。
    """
    scores: Dict[str, float] = {}
    for c, codes in by_class_held.items():
        ms = []
        for code in codes:
            m = L3.momentum(prices, code, signal_date, lookback)
            if m is not None:
                ms.append(m)
        if ms:
            scores[c] = sum(ms) / len(ms)
    return scores


def build_xclass_targets(
    calendar: List[str],
    prices: Dict[str, Dict[str, Dict[str, float]]],
    by_class_held: Dict[str, List[str]],
    lookback: int,
    top_m: Optional[int],
    weighted: bool,
    reb_every: int,
    mode: str,
    seed: int = 20261030,
    abs_filter: bool = False,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Any]]:
    """生成「每个调仓日的袖套目标权重」。

    mode:
      - `mom`  按类动量挑类
      - `rand` 随机挑类（平凡解释排除）

    top_m:     选前 M 个类（M=None 时用于 weighted 模式，保留全部类）
    weighted:  True → 权重 ∝ max(0, 动量)；False → 选中的类等权
    abs_filter: True → 动量最好的类若绝对动量 < 0，全部转防御（bond）
    """
    rng = random.Random(seed)
    targets: Dict[str, Dict[str, float]] = {}
    picks_log: List[Dict[str, Any]] = []
    n_exec = 0
    n_abs_switch = 0

    for i in range(1, len(calendar)):
        if (i - 1) % reb_every != 0:
            continue
        signal_date = calendar[i - 1]
        exec_date = calendar[i]

        scores = class_momentum_scores(prices, by_class_held, signal_date, lookback)
        if not scores:
            continue

        if mode == "rand":
            pool = sorted(scores.keys())
            rng.shuffle(pool)
            ranked = pool
        else:
            ranked = sorted(scores.keys(), key=lambda c: -scores[c])

        if weighted:
            if mode == "rand":
                # ⚠️ 随机对照必须**真的随机**：从单纯形上均匀采样（Dirichlet(1)）。
                # 曾经误用 scores（动量）→ 6 个种子结果几乎相同 → 对照失效。
                wts = [rng.gammavariate(1.0, 1.0) for _ in ranked]
                s = sum(wts) or 1.0
                cls_w = {c: wts[k] / s for k, c in enumerate(ranked)}
            else:
                # 全部类保留，权重 ∝ max(0, 动量)；动量全负 → 等权（不退化成空仓）
                w_raw = {c: max(0.0, scores.get(c, 0.0)) for c in ranked}
                s = sum(w_raw.values())
                if s <= 0:
                    w_raw = {c: 1.0 for c in ranked}
                    s = float(len(ranked))
                cls_w = {c: w_raw[c] / s for c in ranked}
            sel = list(ranked)
        else:
            m = top_m or 1
            sel = ranked[:m]
            if abs_filter and mode == "mom":
                # 绝对动量为正则转防御（债券）
                if scores.get(sel[0], 0.0) < 0 and "bond" in by_class_held:
                    sel = ["bond"]
                    n_abs_switch += 1
            cls_w = {c: 1.0 / len(sel) for c in sel}

        t: Dict[str, float] = {}
        for c in sel:
            codes = by_class_held.get(c) or []
            if not codes:
                continue
            w = cls_w[c] / len(codes)
            for code in codes:
                t[code] = t.get(code, 0.0) + w

        if t:
            targets[exec_date] = t
            n_exec += 1
            picks_log.append({
                "date": signal_date,
                "exec": exec_date,
                "classes": sel,
                "class_weights": {c: round(cls_w[c], 4) for c in sel},
                "scores": {c: round(v, 4) for c, v in scores.items()},
            })

    return targets, {"n_exec": n_exec, "n_abs_switch": n_abs_switch, "picks": picks_log}


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", default="main", choices=["main", "oos"])
    ap.add_argument("--quick", action="store_true", help="A 股腿读缓存（需已有 nav 缓存）")
    args = ap.parse_args()

    if args.window == "oos":
        start, end = OOS_START, OOS_END
        out_path = os.path.join(ROOT, "data", "backtest", "multi_asset_xclass_oos.json")
        nav_cache = os.path.join(ROOT, "data", "backtest", "multi_asset_xclass_navs_oos.json")
    else:
        start, end = START, END
        out_path = os.path.join(ROOT, "data", "backtest", "multi_asset_xclass.json")
        nav_cache = os.path.join(ROOT, "data", "backtest", "multi_asset_xclass_navs.json")

    print(f"[xclass] 窗口 {start.date()} → {end.date()}", flush=True)

    # ⚠️ run_a_share_leg 读 MA 模块的**全局** START/END，必须显式同步，
    # 否则 OOS 会静默用 main 的 A 股腿（L3 踩过）。
    MA.START, MA.END = start, end

    store: Dict[str, Any] = {"window_mode": args.window}

    # ── A 股腿 ──
    a_nav: List[Tuple[str, float]] = []
    a_rec: Dict[str, Any] = {}
    if args.quick and os.path.exists(nav_cache):
        with open(nav_cache, encoding="utf-8") as fh:
            cached = json.load(fh)
        a_nav = [(d, float(v)) for d, v in cached["a_share_nav"]]
        a_rec = cached.get("a_share_leg", {})
        print("[xclass] 读缓存 A 股腿", flush=True)
    else:
        from verify_methods_engine import QLIB_OFFICIAL, universe_from_instruments
        codes = universe_from_instruments(QLIB_OFFICIAL)
        print(f"[xclass] 跑 A 股腿（{len(codes)} 只）...", flush=True)
        t0 = time.perf_counter()
        a_rec, a_nav = run_a_share_leg(codes)
        a_rec["wall_s"] = round(time.perf_counter() - t0, 1)
        ver = verify_baseline(a_rec)
        if args.window == "main":
            print(f"[xclass] 基线复现门: {json.dumps(ver, ensure_ascii=False)}", flush=True)
            if ver.get("checked") and not ver.get("ok"):
                raise SystemExit(1)
        store["a_share_leg"] = a_rec
    print(f"  A 股腿: 夏普 {a_rec.get('engine_sharpe')}  回撤 {a_rec.get('engine_max_drawdown')}",
          flush=True)

    # 护栏：A 股腿净值必须完全落在本窗口内
    if not a_nav:
        raise SystemExit("[xclass] ❌ A 股腿净值为空，退出")
    a_first, a_last = a_nav[0][0], a_nav[-1][0]
    s_str, e_str = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
    if not (s_str <= a_first <= e_str):
        raise SystemExit(f"[xclass] ❌ A 股腿起点 {a_first} 不在窗口 {s_str}~{e_str} 内 → 退出")
    if a_last < e_str:
        print(f"[xclass] ⚠️ A 股腿终点 {a_last} 早于窗口终点 {e_str}", flush=True)

    # ── 候选池：类内固定 top-3 流动性（本实验不测类内选标）──
    universe = load_etf_universe()
    by_class_raw = L3.candidates_by_class(universe, start, end)
    by_class_held = {c: lst[:K_IN_CLASS] for c, lst in by_class_raw.items()}
    print(f"[xclass] 类内固定持仓（top-{K_IN_CLASS} 流动性）: "
          f"{json.dumps(by_class_held, ensure_ascii=False)}", flush=True)

    # ⚠️ 日历必须用**全部候选**（不是只有被持有的 top-3）的交易日并集，
    # 否则与 L3 / final_robustness 的日历不同 → 基线夏普对不上、结果不可比。
    all_codes = sorted({c for lst in by_class_raw.values() for c in lst})
    prices = L3.load_prices(all_codes)
    calendar = L3.build_calendar(prices, all_codes, s_str, e_str)
    print(f"[xclass] 交易日历 {len(calendar)} 天  {calendar[0]} → {calendar[-1]}", flush=True)

    class_of = {c: cls for cls, lst in by_class_held.items() for c in lst}
    class_weights = {c: 1.0 / len(XCLASSES) for c in XCLASSES}

    # gold 单独一条腿（供依赖度度量）
    gold_targets = L3.build_static_targets(
        calendar, prices, {"gold": by_class_held.get("gold", [])}, {"gold": 1.0})
    gold_nav, _ = L3.simulate(calendar, prices, gold_targets, class_of, {"gold": 1.0})

    results: Dict[str, Any] = {}

    def run_cfg(label: str, targets: Dict[str, Dict[str, float]], meta: Dict[str, Any]) -> None:
        sleeve_nav, srec = L3.simulate(calendar, prices, targets, class_of, class_weights)
        if not sleeve_nav:
            print(f"  {label}: ❌ 无净值", flush=True)
            return
        combined = merge_by_date([a_nav, sleeve_nav], [0.5, 0.5])

        # 剔 gold 版（去掉 gold 类，其余类维持原绝对权重，空出留现金 = 中性口径）
        ng_classes = [c for c in XCLASSES if c != "gold"]
        ng_targets: Dict[str, Dict[str, float]] = {}
        for d, t in targets.items():
            nt = {code: w * (1.0 / len(XCLASSES)) / (1.0 / len(ng_classes))
                  for code, w in t.items() if class_of.get(code) != "gold"}
            if nt:
                ng_targets[d] = nt
        ng_sleeve, _ = L3.simulate(
            calendar, prices, ng_targets, class_of,
            {c: 1.0 / len(ng_classes) for c in ng_classes})
        ng_combined = merge_by_date([a_nav, ng_sleeve], [0.5, 0.5]) if ng_sleeve else []

        base_sharpe = sharpe(a_nav)
        cs = sharpe(combined)
        rec: Dict[str, Any] = {
            "sleeve_sharpe": round(sharpe(sleeve_nav), 4),
            "sleeve_return": round(total_ret(sleeve_nav), 4),
            "sleeve_maxdd": round(maxdd(sleeve_nav), 4),
            "combined_sharpe": round(cs, 4),
            "vs_baseline": round(cs / base_sharpe - 1, 4) if base_sharpe else None,
            "combined_return": round(total_ret(combined), 4),
            "combined_maxdd": round(maxdd(combined), 4),
            "combined_yearly": yearly(combined),
            "sharpe_ex_gold": round(sharpe(ng_combined), 4) if ng_combined else None,
            "cost": srec,
            **meta,
        }
        rec.update(L3.gold_dependence(combined, gold_nav, a_nav, sleeve_nav, class_weights))
        results[label] = rec
        print(f"  {label:42s} 夏普 {cs:.4f} ({rec['vs_baseline']:+.2%})  "
              f"收益 {rec['combined_return']:+.2%}  回撤 {rec['combined_maxdd']:.2%}  "
              f"| 剔gold {rec['sharpe_ex_gold']}  换手 {srec['turnover_ratio']:.1f}x "
              f"佣金 {srec['commission_total']:.0f}", flush=True)

    # ── 基线 ──
    print("\n[xclass] === 基线（类间固定等权）===", flush=True)
    tg, meta = L3.build_rotation_targets(
        calendar, prices, by_class_held, class_weights, 60, 99, REB_EVERY, "all")
    run_cfg("base_top3_rebal_monthly", tg,
            {**meta, "rule": "static_top3_rebalanced", "freq": "monthly"})
    run_cfg("base_top3_buyhold",
            L3.build_static_targets(calendar, prices, by_class_held, class_weights),
            {"rule": "static_top3_by_liquidity_buyhold"})

    # ── 跨类动量 ──
    print("\n[xclass] === 跨类动量（类内固定 top-3 流动性）===", flush=True)
    for lookback in LOOKBACKS:
        for top_m in (1, 2):
            tg, meta = build_xclass_targets(
                calendar, prices, by_class_held, lookback, top_m, False, REB_EVERY, "mom")
            run_cfg(f"xclass_mom_L{lookback}_top{top_m}", tg,
                    {**meta, "rule": "xclass_momentum", "lookback": lookback,
                     "top_m": top_m, "freq": "monthly"})
        tg, meta = build_xclass_targets(
            calendar, prices, by_class_held, lookback, None, True, REB_EVERY, "mom")
        run_cfg(f"xclass_mom_L{lookback}_weighted", tg,
                {**meta, "rule": "xclass_momentum_weighted", "lookback": lookback,
                 "freq": "monthly"})

    # 绝对动量过滤（动量最好的类若 <0 → 转债券）
    for lookback in (60, 120):
        tg, meta = build_xclass_targets(
            calendar, prices, by_class_held, lookback, 1, False, REB_EVERY, "mom",
            abs_filter=True)
        run_cfg(f"xclass_mom_L{lookback}_top1_abs", tg,
                {**meta, "rule": "xclass_momentum_abs_filter", "lookback": lookback,
                 "top_m": 1, "freq": "monthly"})

    # ── 随机对照（平凡解释排除）──
    print("\n[xclass] === 随机对照（同频率、同持仓类数、同类内规则，6 种子）===", flush=True)
    for top_m in (1, 2):
        for seed in SEEDS:
            tg, meta = build_xclass_targets(
                calendar, prices, by_class_held, 60, top_m, False, REB_EVERY, "rand",
                seed=seed)
            run_cfg(f"xclass_rand_top{top_m}_s{seed}", tg,
                    {**meta, "rule": "xclass_random_control", "seed": seed,
                     "top_m": top_m, "freq": "monthly"})
    for seed in SEEDS:
        tg, meta = build_xclass_targets(
            calendar, prices, by_class_held, 60, None, True, REB_EVERY, "rand", seed=seed)
        run_cfg(f"xclass_rand_weighted_s{seed}", tg,
                {**meta, "rule": "xclass_random_weighted", "seed": seed,
                 "freq": "monthly"})

    # ── 对照有效性自检 ──
    # 随机对照的各个种子必须给出**不同**结果。若完全相同 → 对照其实没随机
    # （曾发生过：加权模式下误用动量分数 → 6 个种子结果一致 → 对照失效）。
    for grp in ("top1", "top2", "weighted"):
        vals = [results[f"xclass_rand_{grp}_s{s}"]["combined_sharpe"] for s in SEEDS
                if f"xclass_rand_{grp}_s{s}" in results]
        if len(set(vals)) < len(vals):
            raise SystemExit(
                f"[xclass] ❌ 随机对照 {grp} 的 {len(SEEDS)} 个种子结果完全相同 "
                f"→ 对照没有真正随机，实验无效，退出。值: {vals}"
            )
        print(f"[xclass] ✓ 随机对照 {grp}: {len(vals)} 个种子结果互异 "
              f"(夏普 {min(vals):.4f}~{max(vals):.4f})", flush=True)

    store["candidates"] = by_class_raw
    store["held_per_class"] = by_class_held
    store["calendar"] = {"n": len(calendar), "start": calendar[0], "end": calendar[-1]}
    store["results"] = results
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(store, fh, ensure_ascii=False, indent=2)
    print(f"\n[xclass] ✓ 已写入 {out_path}", flush=True)

    with open(nav_cache, "w", encoding="utf-8") as fh:
        json.dump({"a_share_nav": a_nav, "a_share_leg": a_rec}, fh)


if __name__ == "__main__":
    main()
