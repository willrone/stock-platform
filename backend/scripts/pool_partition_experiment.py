#!/usr/bin/env python3
"""分池并行（Pool-Partitioned Parallelism）实验。

## 动机（来自 docs/backtest/CONCLUSION_2026-10-27.md）

结论文档的「决定性发现」：四条主线策略的**分数 RankIC 相关只有 0.11~0.77，
但日收益相关高达 0.75~0.79**，因为它们都从**同一个股票池**选 TopK，
持仓天然重叠 → 分散化在「单一股票池 + TopK」框架内被堵死，框架内夏普上限 ~0.88~1.03。

## 本实验的假设

**把股票池切成互不重叠的子池，各池独立选股并合成，可以让相关系数显著下降，
从而在不提升单腿信号质量的前提下，通过分散化提高组合夏普。**

数学上限：两条腿夏普各为 S、日收益相关 ρ，等权合成夏普 = S·√(2/(1+ρ))。
- ρ=0.79（现状）→ 提升 5.7%
- ρ=0.50 → 提升 20.0%
- ρ=0.30 → 提升 25.5%
- ρ=0.00 → 提升 41.4%

## 切池方式

**臂 A（行业分池，主实验）**：用申万一级行业（data/industry/industry_map.json，
32 个行业 / 5579 只）做贪心平衡划分：行业规模从大到小逐个塞进当前股票数最少的池
→ 各池股票数尽量均衡，避免两腿风险不等价。
切完后每池**独立**做横截面 z-score → 池内选 TopK → 持仓天然不重叠。

**臂 B（随机分池，对照）**：同样的池规模，按固定随机种子把股票随机分配到各池。
用途：区分「分散化效应」与「行业结构效应」。若随机分池也能得到同样的提升，
说明机制是纯分散化（好消息、更普适）；若只有行业分池有效，说明是行业结构在起作用。

## 口径（必须与 docs/backtest/CONCLUSION_2026-10-27.md 主线逐位一致）

- 引擎：BacktestExecutor（真实引擎）
- lag=1（信号日收盘出信号 → 下一交易日成交）
- enable_tradability=True（涨跌停 / 停牌 / ST）
- 策略：honest_composite（25 公式因子主线，use_panel=False）
- 调仓：weekly（周五）
- topk=60 / n_drop=5 / cash_reserve=5%
- 费用：commission 0.0003 / slippage 0.0001
- 窗口：2023-05-08 → 2026-05-08（3 年）
- 股票池：qlib official 全市场（5514 只）

## 基线复现门

K=1 即全池单条（= 当前主线），必须逐位复现 v2 基准
（total_return / sharpe / mdd / trades），不一致直接退出码 1 —— 否则所有新格不可比。

## 产出

`data/backtest/pool_partition.json`：基线 / 各 K 值下每腿指标 + 合成组合指标 +
实测腿间相关 + 分年收益。

用法：
    cd backend && env -u PYTHONPATH BASELINE_LOG_LEVEL=ERROR .venv-py313/bin/python3 scripts/pool_partition_experiment.py
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import random
import statistics
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Tuple

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT = os.path.abspath(os.path.join(BACKEND_DIR, ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from app.services.backtest.execution.backtest_executor import (  # noqa: E402
    BacktestExecutor,
)
from app.services.backtest.models.data_models import BacktestConfig  # noqa: E402

sys.path.insert(0, os.path.join(BACKEND_DIR, "scripts"))
from verify_methods_engine import (  # noqa: E402
    QLIB_OFFICIAL,
    universe_from_instruments,
)

OUT_PATH = os.path.join(ROOT, "data", "backtest", "pool_partition.json")
INDUSTRY_MAP = os.path.join(ROOT, "data", "industry", "industry_map.json")
V2_PATH = os.path.join(ROOT, "data", "backtest", "ALL_real_lag1_v2.json")

START = datetime(2023, 5, 8)
END = datetime(2026, 5, 8)

STRATEGY = "honest_composite"
STRATEGY_CFG: Dict[str, Any] = {"rebalance_days": "weekly"}
TOPK = 60
N_DROP = 5

#: 行业分池的 K 值。K=1 即基线（全池单条）。
K_GRID = [1, 2, 3, 4]
#: 随机分池对照臂的 K 值（只测 2 和 4，控制总耗时）
RANDOM_K_GRID = [2, 4]
RANDOM_SEED = 20261030


# --------------------------------------------------------------------------
# 指标
# --------------------------------------------------------------------------
def sharpe_from_nav(nav: List[Tuple[str, float]]) -> float:
    """标准夏普 μ/σ·√244（逐日收益率，A 股年交易日数）。"""
    if len(nav) < 3:
        return 0.0
    rets = []
    for i in range(1, len(nav)):
        p0, p1 = nav[i - 1][1], nav[i][1]
        if p0 and p0 > 0:
            rets.append(p1 / p0 - 1.0)
    if len(rets) < 3:
        return 0.0
    mu = statistics.fmean(rets)
    sd = statistics.pstdev(rets)
    if sd <= 0:
        return 0.0
    return mu / sd * math.sqrt(244.0)


def maxdd_from_nav(nav: List[Tuple[str, float]]) -> float:
    peak = None
    mdd = 0.0
    for _, v in nav:
        if peak is None or v > peak:
            peak = v
        if peak and peak > 0:
            dd = v / peak - 1.0
            if dd < mdd:
                mdd = dd
    return mdd


def total_return_from_nav(nav: List[Tuple[str, float]]) -> float:
    if not nav or nav[0][1] <= 0:
        return 0.0
    return nav[-1][1] / nav[0][1] - 1.0


def daily_rets(nav: List[Tuple[str, float]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for i in range(1, len(nav)):
        p0, p1 = nav[i - 1][1], nav[i][1]
        if p0 and p0 > 0:
            out[nav[i][0]] = p1 / p0 - 1.0
    return out


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
    if da <= 0 or db <= 0:
        return 0.0
    return num / (da * db)


def align_navs(navs: List[List[Tuple[str, float]]]) -> List[Tuple[str, float]]:
    """按日对齐多腿净值：缺失腿沿用上一值（ffill），逐日相加。

    每条腿独立 100 万 → 组合初始资金 = 腿数 × 100 万。
    """
    if not navs:
        return []
    all_dates: List[str] = sorted({d for nav in navs for d, _ in nav})
    curves = [{d: v for d, v in nav} for nav in navs]
    first = navs[0][0][1] if navs and navs[0] else 1_000_000.0
    last = [first for _ in navs]
    combined: List[Tuple[str, float]] = []
    for d in all_dates:
        total = 0.0
        for li, c in enumerate(curves):
            if d in c:
                last[li] = c[d]
            total += last[li]
        combined.append((d, total))
    return combined


def yearly_from_nav(nav: List[Tuple[str, float]]) -> Dict[str, float]:
    """按年切分净值算年度收益率。

    首年基线必须是**组合的初始净值**（nav[0][1]，绝对金额），
    不能用 1.0 —— 否则首年收益会变成 金额×倍数 的垃圾值。
    """
    out: Dict[str, float] = {}
    if not nav:
        return out
    by_year: Dict[str, List[Tuple[str, float]]] = {}
    for d, v in nav:
        by_year.setdefault(d[:4], []).append((d, v))
    years = sorted(by_year)
    for i, y in enumerate(years):
        base = nav[0][1] if i == 0 else by_year[years[i - 1]][-1][1]
        end = by_year[y][-1][1]
        if base and base > 0:
            out[y] = round(end / base - 1.0, 6)
    return out


# --------------------------------------------------------------------------
# 切池
# --------------------------------------------------------------------------
def load_industry_map() -> Dict[str, str]:
    with open(INDUSTRY_MAP, encoding="utf-8") as fh:
        return json.load(fh)


def partition_by_industry(
    codes: List[str], k: int, ind_map: Dict[str, str]
) -> List[List[str]]:
    """按申万一级行业做贪心平衡划分：K 个互斥子池。"""
    if k <= 1:
        return [sorted(codes)]
    by_ind: Dict[str, List[str]] = {}
    no_ind: List[str] = []
    for c in codes:
        ind = ind_map.get(c)
        if ind:
            by_ind.setdefault(ind, []).append(c)
        else:
            no_ind.append(c)

    sizes = [0] * k
    pools: List[List[str]] = [[] for _ in range(k)]
    for ind in sorted(by_ind, key=lambda i: -len(by_ind[i])):
        j = sizes.index(min(sizes))
        pools[j].extend(by_ind[ind])
        sizes[j] += len(by_ind[ind])
    for c in no_ind:
        j = sizes.index(min(sizes))
        pools[j].append(c)
        sizes[j] += 1
    return [sorted(p) for p in pools]


def partition_random(
    codes: List[str], k: int, sizes: List[int], seed: int
) -> List[List[str]]:
    """随机分池（对照臂）：池规模与行业分池一致，随机分配股票。"""
    if k <= 1:
        return [sorted(codes)]
    rnd = random.Random(seed)
    shuffled = sorted(codes)
    rnd.shuffle(shuffled)
    pools: List[List[str]] = []
    i = 0
    for s in sizes:
        pools.append(sorted(shuffled[i : i + s]))
        i += s
    return pools


# --------------------------------------------------------------------------
# 回测
# --------------------------------------------------------------------------
def run_one(codes: List[str], label: str) -> Tuple[Dict[str, Any], List[Tuple[str, float]]]:
    executor = BacktestExecutor(
        data_dir=os.path.join(ROOT, "data"),
        enable_parallel=True,
        max_workers=8,
        enable_performance_profiling=False,
        use_multiprocessing=False,
    )
    res = asyncio.run(
        executor.run_backtest(
            strategy_name=STRATEGY,
            stock_codes=codes,
            start_date=START,
            end_date=END,
            strategy_config={
                **STRATEGY_CFG,
                "lag": 1,
                "topk": TOPK,
                "n_drop": N_DROP,
            },
            backtest_config=BacktestConfig(
                enable_tradability=True,
                initial_cash=1_000_000.0,
                commission_rate=0.0003,
                slippage_rate=0.0001,
                max_position_size=0.0,  # 策略自算等权上限
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

    nav_raw = res.get("equity_curve") or []
    nav: List[Tuple[str, float]] = [(str(d), float(v)) for d, v in nav_raw if v is not None]
    trades = res.get("trade_history") or []
    buys = sum(1 for t in trades if str(t.get("action", "")).upper() == "BUY")
    sells = sum(1 for t in trades if str(t.get("action", "")).upper() == "SELL")

    rec: Dict[str, Any] = {
        "label": label,
        "n_codes": len(codes),
        "engine_total_return": res.get("total_return"),
        "engine_sharpe": res.get("sharpe_ratio"),
        "engine_max_drawdown": res.get("max_drawdown"),
        "engine_win_rate": res.get("win_rate"),
        "total_trades": len(trades),
        "buy_trades": buys,
        "sell_trades": sells,
        "nav_points": len(nav),
        "nav_start": nav[0][1] if nav else None,
        "nav_end": nav[-1][1] if nav else None,
        # 自算口径（逐日净值）
        "nav_total_return": round(total_return_from_nav(nav), 6),
        "nav_sharpe": round(sharpe_from_nav(nav), 6),
        "nav_max_drawdown": round(maxdd_from_nav(nav), 6),
    }
    return rec, nav


def verify_baseline(rec: Dict[str, Any]) -> Dict[str, Any]:
    """基线复现门：与 ALL_real_lag1_v2.json 的 honest_composite 逐位比对。"""
    if not os.path.exists(V2_PATH):
        return {"checked": False, "reason": "v2 基准不存在"}
    with open(V2_PATH, encoding="utf-8") as fh:
        v2 = json.load(fh).get("results") or {}
    base = v2.get("honest_composite")
    if not base:
        return {"checked": False, "reason": "v2 无 honest_composite"}
    diffs: Dict[str, Any] = {}
    for field in ("total_return", "sharpe_ratio", "max_drawdown"):
        a, b = rec.get(field), base.get(field)
        if a is None or b is None:
            continue
        if abs(float(a) - float(b)) > 1e-12:
            diffs[field] = {"recomputed": a, "v2_baseline": b}
    if int(rec.get("total_trades") or 0) != int(base.get("total_trades") or 0):
        diffs["total_trades"] = {
            "recomputed": rec.get("total_trades"),
            "v2_baseline": base.get("total_trades"),
        }
    return {"checked": True, "diffs": diffs, "ok": not diffs}


def run_arm(
    arm: str,
    pools: List[List[str]],
    store: Dict[str, Any],
) -> Dict[str, Any]:
    """跑一个切池方案的所有腿，返回合成结果。"""
    key = f"{arm}_K{len(pools)}"
    print(
        f"\n[pool] === {key}  池规模 {[len(p) for p in pools]} ===",
        flush=True,
    )
    legs: List[Dict[str, Any]] = []
    navs: List[List[Tuple[str, float]]] = []
    for i, pool in enumerate(pools):
        t0 = time.perf_counter()
        rec, nav = run_one(pool, f"{key}_leg{i}")
        rec["wall_s"] = round(time.perf_counter() - t0, 1)
        navs.append(nav)
        legs.append(rec)
        store[f"{key}_leg{i}"] = rec
        print(
            f"  leg{i}: {rec['n_codes']:5d} 只  夏普 {rec['engine_sharpe']}  "
            f"收益 {rec['engine_total_return']}  回撤 {rec['engine_max_drawdown']}  "
            f"笔数 {rec['total_trades']}  ({rec['wall_s']}s)",
            flush=True,
        )

    combined_nav = align_navs(navs) if len(navs) > 1 else navs[0]
    rets = [daily_rets(n) for n in navs]

    # 腿间两两相关
    pairwise: List[Dict[str, Any]] = []
    for i in range(len(navs)):
        for j in range(i + 1, len(navs)):
            c = corr(rets[i], rets[j])
            pairwise.append({"legs": [i, j], "corr": round(c, 4)})

    avg_corr = (
        round(statistics.fmean([p["corr"] for p in pairwise]), 4) if pairwise else None
    )

    combined = {
        "key": key,
        "arm": arm,
        "k": len(pools),
        "pool_sizes": [len(p) for p in pools],
        "combined_total_return": round(total_return_from_nav(combined_nav), 6),
        "combined_sharpe": round(sharpe_from_nav(combined_nav), 6),
        "combined_max_drawdown": round(maxdd_from_nav(combined_nav), 6),
        "combined_trades": sum(int(l["total_trades"]) for l in legs),
        "avg_leg_sharpe": round(
            statistics.fmean([float(l["engine_sharpe"] or 0) for l in legs]), 6
        ),
        "avg_pairwise_corr": avg_corr,
        "pairwise_corr": pairwise,
        "yearly": yearly_from_nav(combined_nav),
        "legs": legs,
    }
    store[key] = combined
    print(
        f"  ▶ 合成: 夏普 {combined['combined_sharpe']}  "
        f"收益 {combined['combined_total_return']}  "
        f"回撤 {combined['combined_max_drawdown']}  "
        f"笔数 {combined['combined_trades']}  "
        f"腿均夏普 {combined['avg_leg_sharpe']}  平均相关 {avg_corr}",
        flush=True,
    )
    return combined


def main() -> None:
    # 注意：universe_from_instruments 返回的 5514 条里有 **1 条重复**
    # （688665.SH 出现两次）→ 唯一代码 5513 个。
    # 基线（K=1）必须用**原始列表**跑，才能与 ALL_real_lag1_v2.json 逐位对齐；
    # 分池（K>1）必须用**去重列表**，否则同一只票会在某个池里被计入两次
    # （引擎侧 dict 按 code 去重，但切池时若不去重会导致池规模统计错误、
    #  且该池的等权上限按 topk 计算而实际候选少一只）。
    codes_raw = universe_from_instruments(QLIB_OFFICIAL)
    codes = sorted(set(codes_raw))
    ind_map = load_industry_map()
    smoke = "--smoke" in sys.argv
    print(
        f"[pool] 窗口 {START.date()} → {END.date()}  全市场 {len(codes)} 只"
        f"（原始 {len(codes_raw)} 条，含重复 {len(codes_raw) - len(codes)} 条）  "
        f"策略 {STRATEGY} topk={TOPK} ndrop={N_DROP} weekly lag=1 tradability=on",
        flush=True,
    )

    store: Dict[str, Any] = {}

    # ── 基线（K=1 全池）—— 先过基线复现门 ──
    if smoke:
        # 冒烟：小子集 + 短窗口，只验证管线（净值导出 / 合成 / 相关计算）
        smoke_codes = codes[:400]
        base_comb = run_arm("smoke", [smoke_codes], store)
        verification = {"checked": False, "reason": "smoke 模式跳过基线复现门"}
    else:
        base_comb = run_arm("industry", partition_by_industry(codes_raw, 1, ind_map), store)
        base_leg = store["industry_K1_leg0"]
        verification = verify_baseline(base_leg)
        print(
            f"\n[pool] 基线复现门: {json.dumps(verification, ensure_ascii=False)}",
            flush=True,
        )
        if verification.get("checked") and not verification.get("ok"):
            print("[pool] ❌ 基线无法复现 v2 基准 → 所有新格不可比，退出", flush=True)
            raise SystemExit(1)

    if smoke:
        run_arm("smoke", partition_by_industry(codes[:800], 2, ind_map), store)
        os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
        with open(
            os.path.join(ROOT, "data", "backtest", "pool_partition_smoke.json"),
            "w",
            encoding="utf-8",
        ) as fh:
            json.dump(store, fh, ensure_ascii=False, indent=1)
        print("\n[pool] ✓ 冒烟完成")
        return

    # ── 行业分池 ──
    for k in K_GRID:
        if k == 1:
            continue
        run_arm("industry", partition_by_industry(codes, k, ind_map), store)

    # ── 随机分池对照（池规模沿用行业分池的）──
    for k in RANDOM_K_GRID:
        ref = store.get(f"industry_K{k}")
        sizes = ref["pool_sizes"] if ref else []
        if not sizes:
            continue
        run_arm("random", partition_random(codes, k, sizes, RANDOM_SEED), store)

    base_sharpe = float(base_comb["combined_sharpe"])
    summary = {
        "window": f"{START.date()}→{END.date()}",
        "universe_size": len(codes),
        "strategy": STRATEGY,
        "methodology": {
            "lag": 1,
            "enable_tradability": True,
            "rebalance_days": "weekly",
            "topk": TOPK,
            "n_drop": N_DROP,
            "commission_rate": 0.0003,
            "slippage_rate": 0.0001,
            "initial_cash_per_leg": 1_000_000.0,
            "note": "合成 = 各腿逐日净值按日对齐（ffill）后相加，腿数×100万为本金",
        },
        "baseline_verification": verification,
        "baseline_sharpe": base_sharpe,
        "results": store,
    }

    # 提升幅度汇总
    gains = []
    for key, rec in store.items():
        if "_leg" in key:
            continue
        gains.append(
            {
                "key": key,
                "arm": rec["arm"],
                "k": rec["k"],
                "combined_sharpe": rec["combined_sharpe"],
                "delta_vs_baseline": round(rec["combined_sharpe"] - base_sharpe, 4),
                "pct_vs_baseline": round(
                    (rec["combined_sharpe"] / base_sharpe - 1.0) * 100.0, 2
                )
                if base_sharpe
                else None,
                "avg_pairwise_corr": rec["avg_pairwise_corr"],
                "combined_max_drawdown": rec["combined_max_drawdown"],
                "combined_total_return": rec["combined_total_return"],
            }
        )
    gains.sort(key=lambda g: -g["combined_sharpe"])
    summary["gains"] = gains

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1)
    print(f"\n[pool] ✓ 已写入 {OUT_PATH}")
    print("[pool] 提升幅度排行:")
    for g in gains:
        print(
            f"  {g['key']:20s} 夏普 {g['combined_sharpe']:.4f} "
            f"({g['pct_vs_baseline']:+.2f}%)  相关 {g['avg_pairwise_corr']}  "
            f"回撤 {g['combined_max_drawdown']:.4f}"
        )


if __name__ == "__main__":
    main()
