#!/usr/bin/env python3
"""L3 动量轮动 —— 给 ETF 腿加 alpha，目标是**降低对 gold 单一资产的依赖**。

## 为什么做这个

`MULTI_ASSET_LEG_2026-10-30.md` 的结论：推荐配置（A股腿 50% + 黄金/债券/商品
各 16.67%）把夏普从 0.880 提到 1.326（推导窗口）/ +24.5%（样本外窗口），回撤
−13.4% → −9.1%。

**但这个组合有致命依赖**：去掉 gold，夏普 0.793，比纯 A 股基线还差 8.4%。
组合夏普 1.33 里有相当一部分是**金价 beta**，不是可复制的 alpha。

→ P0 任务：给 ETF 腿加主动 alpha，看能否在**保住分散化收益**的同时
**削掉对 gold 的依赖**。

## 本实验做什么

在**同一个候选池、同一套撮合引擎**下，只改**选标规则**，做对照：

| 配置 | 选标规则 | 作用 |
|---|---|---|
| `static_top3` | 每类成交额 top-3，买入持有 | **基线**（= L2，等价于已定稿的推荐配置） |
| `static_all` | 每类全部候选等权，买入持有 | 隔离「换池」效应 |
| `momentum` | 每类按动量选 top-K，每月轮动 | **L3 主目标** |
| `dual` | 相对动量选标 + 绝对动量为正则转防御（债券 ETF） | 削回撤专用 |
| `random` | 每类随机选 K 只（固定种子） | **平凡解释排除** |

**必须有 random 对照**：否则「任何轮动都能改善」无法排除。历史上本项目
多次被这种平凡解释坑到（现金对照、宽基 ETF 对照）。

## 黄金依赖度的度量（核心指标）

1. `corr_with_gold`：组合日收益与 gold 腿日收益的相关
2. `gold_crash_window`：gold 腿最差的 12 个月窗口内，组合的收益 vs gold 的收益
3. `avg_gold_weight`：组合中 gold 的平均实际权重（轮动会动态变化）
4. `sharpe_ex_gold`：剔除 gold 后组合的夏普（直接测「去掉 gold 会不会崩」）

## 口径（沿用已验证的踩坑结论）

- 成交价：ETF 用 **次日 open**（信号用前一日收盘价 → lag=1，与 A 股腿一致）
- 成本：佣金 0.03% + 滑点 0.01%（ETF 免印花税）
- 换手成本按**每次调仓的成交额**扣，不是按组合价值
- 覆盖门槛：候选 ETF 必须**上市早于窗口起点**且窗口内样本 ≥ 200 天
- 动量所需的**窗口前历史**从 parquet 全量读取（不截断），否则 L=250 在
  窗口起点就无信号可用

## 参数纪律

动量窗口 L ∈ {20, 60, 120, 250}、每类持仓 K ∈ {1, 2, 3}、调仓频率 ∈ {月度, 双周}。
**先在推导窗口看，再在样本外窗口验证**；参数若在两窗口不一致地最优，
按 `MULTI_ASSET_LEG_2026-10-30.md` 的教训处理（宁可选规则简单、
两窗口方向一致的参数，不选单一窗口最优的刀锋参数）。

用法：
    cd backend && env -u PYTHONPATH BASELINE_LOG_LEVEL=ERROR .venv-py313/bin/python3 \
        scripts/multi_asset_l3_momentum.py --window main
    cd backend && env -u PYTHONPATH BASELINE_LOG_LEVEL=ERROR .venv-py313/bin/python3 \
        scripts/multi_asset_l3_momentum.py --window oos
"""
from __future__ import annotations

import argparse
import bisect
import json
import os
import random
import sys
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT = os.path.abspath(os.path.join(BACKEND_DIR, ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)
sys.path.insert(0, os.path.join(BACKEND_DIR, "scripts"))

from multi_asset_leg_experiment import (  # noqa: E402
    END,
    ETF_PARQUET,
    ETF_UNIVERSE,
    OOS_END,
    OOS_START,
    START,
    STRATEGY,
    STRATEGY_CFG,
    TOPK,
    N_DROP,
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
import multi_asset_leg_experiment as MA  # noqa: E402
from verify_methods_engine import QLIB_OFFICIAL, universe_from_instruments  # noqa: E402

#: 本实验关注的类别集合 = 已定稿的推荐配置
L3_CLASSES = ["gold", "bond", "commodity"]

#: 调仓频率：交易日间隔
REB_FREQ = {"monthly": 21, "biweekly": 10}

COMMISSION = 0.0003
SLIPPAGE = 0.0001

MIN_SAMPLES = 200
#: 候选池上限（每类）。太多会让轮动噪声变大，也拖慢撮合。
MAX_CANDIDATES_PER_CLASS = 12
#: 窗口起点容忍度（自然日）：窗口起点常落在节假日/非交易日，ETF 数据实际从
#: 下一个交易日起。严格比较 "> 窗口起点" 会把**全部**候选清空。
START_TOLERANCE_DAYS = 7


# --------------------------------------------------------------------------
# 数据
# --------------------------------------------------------------------------
def load_prices(codes: List[str]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """读 parquet → {code: {date: {open, close}}}，保留全量历史（动量需要）。

    同时建索引 `{code: {"dates": [...], "open": [...], "close": [...]}}`，
    供 bisect O(log n) 前向填充查找用（simulate 每天每只票都要查价，
    线性查会退化到 O(days² × holdings)）。
    """
    import pandas as pd

    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    idx: Dict[str, Dict[str, Any]] = {}
    for code in codes:
        path = os.path.join(ETF_PARQUET, f"{code}.parquet")
        if not os.path.exists(path):
            continue
        df = pd.read_parquet(path).sort_values("date")
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        rows: Dict[str, Dict[str, float]] = {}
        for _, r in df.iterrows():
            rows[str(r["date"])] = {
                "open": float(r["open"]),
                "close": float(r["close"]),
            }
        if not rows:
            continue
        out[code] = rows
        ds = sorted(rows)
        idx[code] = {
            "dates": ds,
            "open": [rows[d]["open"] for d in ds],
            "close": [rows[d]["close"] for d in ds],
        }
    load_prices.index = idx  # type: ignore[attr-defined]
    return out


def _px(
    idx: Dict[str, Dict[str, Any]],
    code: str,
    d: str,
    field: str,
    exact: bool = False,
) -> Optional[float]:
    """取 code 在日期 d 的价格；缺失时前向填充（用最近的历史收盘价）。"""
    e = idx.get(code)
    if not e:
        return None
    ds: List[str] = e["dates"]
    i = bisect.bisect_right(ds, d) - 1
    if i < 0:
        return None
    if exact and ds[i] != d:
        return None
    return e[field][i]


def _momentum(
    idx: Dict[str, Dict[str, Any]], code: str, signal_date: str, lookback: int
) -> Optional[float]:
    """signal_date 收盘价 / (前 lookback 交易日) 收盘价 - 1（要求当日有行情）。"""
    e = idx.get(code)
    if not e:
        return None
    ds: List[str] = e["dates"]
    i = bisect.bisect_right(ds, signal_date) - 1
    if i < 0 or ds[i] != signal_date:
        return None
    if i - lookback < 0:
        return None
    c0 = e["close"][i]
    c1 = e["close"][i - lookback]
    if c1 <= 0:
        return None
    return c0 / c1 - 1.0


def _realized_vol(
    idx: Dict[str, Dict[str, Any]], code: str, signal_date: str, win: int
) -> Optional[float]:
    e = idx.get(code)
    if not e:
        return None
    ds: List[str] = e["dates"]
    i = bisect.bisect_right(ds, signal_date) - 1
    if i - win < 0:
        return None
    cl = e["close"][i - win: i + 1]
    rets = [cl[k] / cl[k - 1] - 1 for k in range(1, len(cl)) if cl[k - 1] > 0]
    if len(rets) < 2:
        return None
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    return var ** 0.5 * (252 ** 0.5)


def candidates_by_class(
    universe: Dict[str, Any], start: datetime, end: datetime
) -> Dict[str, List[str]]:
    """每类的候选池：**覆盖全窗口** + 流动性排序取前 N。

    覆盖门槛（沿用已验证的踩坑结论）：数据起点 ≤ 窗口起点、窗口内样本 ≥ 200 天。
    否则共同日历交集会把整条腿的起点推后 → 该腿夏普只在部分窗口成立，
    与其他腿不可比（bond 腿曾因此只剩 1 年数据、算出「夏普 4.84」的样本噪声）。
    """
    import pandas as pd

    s_str = start.strftime("%Y-%m-%d")
    e_str = end.strftime("%Y-%m-%d")
    # 起点容忍度：窗口起点常落在节假日/非交易日，ETF 数据实际从下一个交易日
    # 才开始。严格比较 "> 窗口起点" 会把**全部**候选清空（2019-01-01 踩过）。
    start_lim = (start + timedelta(days=START_TOLERANCE_DAYS)).strftime("%Y-%m-%d")
    by_class: Dict[str, List[Tuple[float, str]]] = {c: [] for c in L3_CLASSES}

    for code, meta in universe["etfs"].items():
        tags = [t for t in meta.get("tags", []) if t in by_class]
        if not tags:
            continue
        path = path_of(code)
        if not os.path.exists(path):
            continue
        df = pd.read_parquet(path).sort_values("date")
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        w = df[(df["date"] >= s_str) & (df["date"] <= e_str)]
        if len(w) < MIN_SAMPLES:
            continue
        if str(w["date"].iloc[0]) > start_lim:
            continue  # 上市晚于窗口起点 + 容忍度 → 不可用
        amt = float((w["volume"].fillna(0) * w["close"] * 100).mean())
        if amt <= 0:
            continue
        for t in tags:
            by_class[t].append((amt, code))

    out: Dict[str, List[str]] = {}
    for c, lst in by_class.items():
        lst.sort(reverse=True)
        out[c] = [code for _, code in lst[:MAX_CANDIDATES_PER_CLASS]]
    return out


def path_of(code: str) -> str:
    return os.path.join(ETF_PARQUET, f"{code}.parquet")


# --------------------------------------------------------------------------
# 撮合引擎：给定「每个调仓日的目标权重」→ 净值曲线
# --------------------------------------------------------------------------
def simulate(
    calendar: List[str],
    prices: Dict[str, Dict[str, Dict[str, float]]],
    targets: Dict[str, Dict[str, float]],
    class_of: Dict[str, str],
    class_weights: Dict[str, float],
    capital: float = 1_000_000.0,
) -> Tuple[List[Tuple[str, float]], Dict[str, Any]]:
    """按调仓日目标权重撮合。

    - `targets[date]` = {code: 目标权重}（相对整条腿）
    - 成交价 = 该日 open；买入滑点上浮、卖出滑点下压
    - 佣金 0.03%（ETF 免印花税），按成交额扣
    - 未在目标里的持仓 → 全部卖出
    """
    idx = getattr(load_prices, "index", {})
    cash = capital
    shares: Dict[str, float] = {}
    nav: List[Tuple[str, float]] = []
    total_cost = 0.0
    total_turnover = 0.0
    n_rebal = 0

    def px(code: str, d: str, field: str) -> Optional[float]:
        return _px(idx, code, d, field)

    for d in calendar:
        tgt = targets.get(d)
        if tgt:
            n_rebal += 1
            # 1) 用开盘价估值
            val = cash
            for code, sh in shares.items():
                p = px(code, d, "open")
                if p is not None:
                    val += sh * p
            # 2) 先卖后买（释放现金）
            for code in list(shares.keys()):
                if code in tgt:
                    continue
                p = px(code, d, "open")
                if p is None:
                    continue
                sell_px = p * (1 - SLIPPAGE)
                proceeds = shares[code] * sell_px
                cash += proceeds
                total_cost += proceeds * COMMISSION
                total_turnover += proceeds
                del shares[code]
            # 3) 买（按目标权重）
            for code, w in tgt.items():
                if w <= 0:
                    continue
                p = px(code, d, "open")
                if p is None:
                    continue
                buy_px = p * (1 + SLIPPAGE)
                want = val * w
                cur = shares.get(code, 0) * p
                if cur >= want:
                    continue
                amount = want - cur
                cost = amount * COMMISSION
                if cash < amount + cost:
                    amount = max(0.0, cash - cost)
                if amount <= 0:
                    continue
                cash -= amount + cost
                total_cost += cost
                total_turnover += amount
                shares[code] = shares.get(code, 0) + amount / buy_px
            # 4) 权重再平衡到精确目标（卖出超配的部分）
            for code in list(shares.keys()):
                if code not in tgt:
                    continue
                p = px(code, d, "open")
                if p is None:
                    continue
                cur = shares[code] * p
                want = val * tgt[code]
                if cur <= want:
                    continue
                sell_px = p * (1 - SLIPPAGE)
                proceeds = min(shares[code] * sell_px, cur - want)
                if proceeds <= 0:
                    continue
                cash += proceeds
                total_cost += proceeds * COMMISSION
                total_turnover += proceeds
                shares[code] -= proceeds / sell_px

        # 收盘 mark-to-market
        val = cash
        for code, sh in shares.items():
            p = px(code, d, "close")
            if p is not None:
                val += sh * p
        nav.append((d, val))

    rec = {
        "commission_total": round(total_cost, 2),
        "turnover_total": round(total_turnover, 2),
        "turnover_ratio": round(total_turnover / capital, 4),
        "n_rebalances": n_rebal,
    }
    return nav, rec


def build_calendar(
    prices: Dict[str, Dict[str, Dict[str, float]]],
    codes: List[str],
    start: str,
    end: str,
) -> List[str]:
    """所有候选 ETF 交易日的并集（缺失日用前向填充近似处理）。"""
    s: set = set()
    for c in codes:
        s |= {d for d in prices.get(c, {}) if start <= d <= end}
    return sorted(s)


def momentum(
    prices: Dict[str, Dict[str, Dict[str, float]]],
    code: str,
    signal_date: str,
    lookback: int,
) -> Optional[float]:
    """signal_date 收盘价 / (signal_date - lookback 交易日) 收盘价 - 1。"""
    return _momentum(getattr(load_prices, "index", {}), code, signal_date, lookback)


def realized_vol(
    prices: Dict[str, Dict[str, Dict[str, float]]], code: str, signal_date: str, win: int
) -> Optional[float]:
    return _realized_vol(getattr(load_prices, "index", {}), code, signal_date, win)


# --------------------------------------------------------------------------
# 各选标规则 → 目标权重表
# --------------------------------------------------------------------------
def build_static_targets(
    calendar: List[str],
    prices: Dict[str, Dict[str, Dict[str, float]]],
    by_class: Dict[str, List[str]],
    class_weights: Dict[str, float],
) -> Dict[str, Dict[str, float]]:
    """买入持有：窗口第一个交易日建仓，之后不动。"""
    targets: Dict[str, Dict[str, float]] = {}
    if not calendar:
        return targets
    first = calendar[1] if len(calendar) > 1 else calendar[0]
    t: Dict[str, float] = {}
    for c, codes in by_class.items():
        if not codes:
            continue
        w = class_weights.get(c, 0.0) / len(codes)
        for code in codes:
            t[code] = t.get(code, 0.0) + w
    if t:
        targets[first] = t
    return targets


def build_rotation_targets(
    calendar: List[str],
    prices: Dict[str, Dict[str, Dict[str, float]]],
    by_class: Dict[str, List[str]],
    class_weights: Dict[str, float],
    lookback: int,
    top_k: int,
    reb_every: int,
    mode: str,
    seed: int = 20261030,
    vol_window: int = 60,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Any]]:
    """轮动：每 reb_every 个交易日重选一次。

    mode:
      - `momentum`   每类按动量选 top_k
      - `dual`       同 momentum，但**绝对动量**（宽基指数动量）< 0 的类别
                     转入防御资产（bond 类里夏普最高的一只）
      - `random`     每类随机选 top_k（固定种子）→ 平凡解释排除
      - `rvol`       每类按 动量/波动 选 top_k（风险调整动量）
      - `all`        每类**全选**（= 静态买入持有的持仓集），但**定期再平衡**
                     → 隔离「再平衡到等权」本身的效果，不含任何选标 alpha
    """
    rng = random.Random(seed)
    targets: Dict[str, Dict[str, float]] = {}
    picks_log: List[Dict[str, Any]] = []
    n_exec = 0

    # 防御资产：bond 类里第一个候选
    defense = by_class.get("bond", [None])[0] if by_class.get("bond") else None

    for i in range(1, len(calendar)):
        if (i - 1) % reb_every != 0:
            continue
        signal_date = calendar[i - 1]
        exec_date = calendar[i]
        t: Dict[str, float] = {}
        log: Dict[str, Any] = {"date": signal_date, "exec": exec_date, "picks": {}}

        for c, codes in by_class.items():
            if not codes:
                continue
            cw = class_weights.get(c, 0.0)
            if cw <= 0:
                continue

            if mode == "all":
                sel = list(codes)
            elif mode == "random":
                pool = list(codes)
                rng.shuffle(pool)
                sel = pool[:top_k]
            else:
                scored: List[Tuple[float, str]] = []
                for code in codes:
                    m = momentum(prices, code, signal_date, lookback)
                    if m is None:
                        continue
                    if mode == "rvol":
                        v = realized_vol(prices, code, signal_date, vol_window)
                        if v is None or v <= 0:
                            continue
                        scored.append((m / v, code))
                    else:
                        scored.append((m, code))
                scored.sort(reverse=True)
                sel = [code for _, code in scored[:top_k]]

            # dual momentum：绝对动量为负 → 转防御
            if mode == "dual" and sel:
                abs_m = momentum(prices, sel[0], signal_date, lookback)
                if abs_m is not None and abs_m < 0 and defense and defense in prices:
                    sel = [defense]

            if not sel:
                continue
            w = cw / len(sel)
            for code in sel:
                t[code] = t.get(code, 0.0) + w
            log["picks"][c] = sel

        if t:
            targets[exec_date] = t
            n_exec += 1
            picks_log.append(log)

    return targets, {"n_exec": n_exec, "picks": picks_log}


# --------------------------------------------------------------------------
# 黄金依赖度
# --------------------------------------------------------------------------
def gold_dependence(
    combined_nav: List[Tuple[str, float]],
    gold_nav: List[Tuple[str, float]],
    a_nav: List[Tuple[str, float]],
    sleeve_nav: List[Tuple[str, float]],
    class_weights: Dict[str, float],
) -> Dict[str, Any]:
    c_rets = daily_rets(combined_nav)
    g_rets = daily_rets(gold_nav)
    a_rets = daily_rets(a_nav)
    s_rets = daily_rets(sleeve_nav)

    # gold 腿最差的滚动 12 个月窗口
    g_dates = [d for d, _ in gold_nav]
    g_vals = {d: v for d, v in gold_nav}
    c_vals = {d: v for d, v in combined_nav}
    worst = None
    if len(g_dates) > 260:
        for i in range(260, len(g_dates)):
            ret = g_vals[g_dates[i]] / g_vals[g_dates[i - 260]] - 1
            if worst is None or ret < worst[0]:
                worst = (ret, g_dates[i - 260], g_dates[i])
    out: Dict[str, Any] = {
        "corr_with_gold": round(corr(c_rets, g_rets), 4) if g_rets else None,
        "corr_with_a_share": round(corr(c_rets, a_rets), 4),
        "corr_sleeve_with_gold": round(corr(s_rets, g_rets), 4) if g_rets else None,
    }
    if worst:
        ret, d0, d1 = worst
        c0 = c_vals.get(d0)
        c1 = c_vals.get(d1)
        out["gold_worst_12m"] = {
            "gold_return": round(ret, 4),
            "window": [d0, d1],
            "combined_return": round(c1 / c0 - 1, 4) if c0 and c1 else None,
        }
    return out


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", default="main", choices=["main", "oos"])
    ap.add_argument("--quick", action="store_true", help="跳过 A 股腿（读缓存）")
    args = ap.parse_args()

    if args.window == "oos":
        start, end = OOS_START, OOS_END
        out_path = os.path.join(ROOT, "data", "backtest", "multi_asset_l3_oos.json")
        nav_cache = os.path.join(ROOT, "data", "backtest", "multi_asset_l3_navs_oos.json")
    else:
        start, end = START, END
        out_path = os.path.join(ROOT, "data", "backtest", "multi_asset_l3.json")
        nav_cache = os.path.join(ROOT, "data", "backtest", "multi_asset_l3_navs.json")

    print(f"[l3] 窗口 {start.date()} → {end.date()}", flush=True)

    # ⚠️ `run_a_share_leg()` 读的是 multi_asset_leg_experiment 模块的**全局**
    # START/END，不是传参。必须显式改模块全局，否则 OOS 窗口会静默地
    # 用**主窗口**的 A 股腿（曾因此把 OOS 的 ETF 腿和 main 的 A 股腿混算，
    # 得出「OOS 全军覆没」的假结论）。
    MA.START, MA.END = start, end

    store: Dict[str, Any] = {"window_mode": args.window}

    # ── A 股腿 ──
    a_nav: List[Tuple[str, float]] = []
    a_rec: Dict[str, Any] = {}
    if os.path.exists(nav_cache) and args.quick:
        with open(nav_cache, encoding="utf-8") as fh:
            cached = json.load(fh)
        a_nav = [(d, float(v)) for d, v in cached["a_share_nav"]]
        a_rec = cached.get("a_share_leg", {})
        print("[l3] 读缓存 A 股腿", flush=True)
    else:
        codes = universe_from_instruments(QLIB_OFFICIAL)
        print(f"[l3] 跑 A 股腿（{len(codes)} 只）...", flush=True)
        t0 = time.perf_counter()
        a_rec, a_nav = run_a_share_leg(codes)
        a_rec["wall_s"] = round(time.perf_counter() - t0, 1)
        ver = verify_baseline(a_rec)
        if args.window == "main":
            print(f"[l3] 基线复现门: {json.dumps(ver, ensure_ascii=False)}", flush=True)
            if ver.get("checked") and not ver.get("ok"):
                raise SystemExit(1)
        store["a_share_leg"] = a_rec
    print(
        f"  A 股腿: 夏普 {a_rec.get('engine_sharpe')}  回撤 {a_rec.get('engine_max_drawdown')}",
        flush=True,
    )
    # 护栏：A 股腿的净值起点必须落在本窗口内。
    # 曾因模块全局 START/END 没改，导致 OOS 窗口静默用了 main 的 A 股腿，
    # 混算后得出「OOS 全军覆没」的假结论。这条断言让同类错误当场暴露。
    a_first = a_nav[0][0] if a_nav else ""
    a_last = a_nav[-1][0] if a_nav else ""
    if not (start.strftime("%Y-%m-%d") <= a_first <= end.strftime("%Y-%m-%d")):
        raise SystemExit(
            f"[l3] ❌ A 股腿起点 {a_first} 不在窗口 "
            f"{start.date()}~{end.date()} 内 → 模块全局 START/END 没同步，退出"
        )
    if a_last < end.strftime("%Y-%m-%d"):
        print(
            f"[l3] ⚠️ A 股腿终点 {a_last} 早于窗口终点 {end.date()}（数据可用性限制）",
            flush=True,
        )

    # ── 候选池 ──
    universe = load_etf_universe()
    by_class = candidates_by_class(universe, start, end)
    print(f"[l3] 候选池: {json.dumps(by_class, ensure_ascii=False)}", flush=True)

    all_codes = sorted({c for lst in by_class.values() for c in lst})
    prices = load_prices(all_codes)
    s_str, e_str = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
    calendar = build_calendar(prices, all_codes, s_str, e_str)
    print(f"[l3] 交易日历 {len(calendar)} 天  {calendar[0]} → {calendar[-1]}", flush=True)

    class_weights = {c: 1.0 / len(L3_CLASSES) for c in L3_CLASSES}
    class_of = {c: cls for cls, lst in by_class.items() for c in lst}

    # ── 静态基线（gold 单独一条腿，供依赖度度量）──
    gold_targets = build_static_targets(
        calendar, prices, {"gold": by_class.get("gold", [])}, {"gold": 1.0}
    )
    gold_nav, _ = simulate(calendar, prices, gold_targets, class_of, {"gold": 1.0})

    results: Dict[str, Any] = {}

    def run_cfg(label: str, targets: Dict[str, Dict[str, float]], meta: Dict[str, Any]) -> None:
        sleeve_nav, srec = simulate(calendar, prices, targets, class_of, class_weights)
        if not sleeve_nav:
            print(f"  {label}: ❌ 无净值", flush=True)
            return
        # 50% A 股腿 + 50% ETF 腿（类间等权）
        combined = merge_by_date([a_nav, sleeve_nav], [0.5, 0.5])
        # 剔除 gold 的版本（直接测「去掉 gold 会不会崩」）
        ng_classes = [c for c in L3_CLASSES if c != "gold"]
        ng_t = {k: v for k, v in targets.items()}
        ng_targets = {}
        for d, t in ng_t.items():
            nt = {code: w * (1.0 / len(L3_CLASSES)) / (1.0 / len(ng_classes))
                  for code, w in t.items() if class_of.get(code) != "gold"}
            if nt:
                ng_targets[d] = nt
        ng_sleeve, _ = simulate(calendar, prices, ng_targets, class_of,
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
        rec.update(gold_dependence(combined, gold_nav, a_nav, sleeve_nav, class_weights))
        results[label] = rec
        print(
            f"  {label:38s} 组合夏普 {cs:.4f} ({rec['vs_baseline']:+.2%})  "
            f"收益 {rec['combined_return']:+.2%}  回撤 {rec['combined_maxdd']:.2%}  "
            f"| 剔gold夏普 {rec['sharpe_ex_gold']}  与gold相关 {rec['corr_with_gold']}",
            flush=True,
        )

    # ── 配置组 ──
    print("\n[l3] === 静态对照 ===", flush=True)
    run_cfg(
        "static_all_buyhold",
        build_static_targets(calendar, prices, by_class, class_weights),
        {"rule": "static_all_candidates_equal_buyhold"},
    )
    # top-3 流动性子集（= 已定稿的推荐配置，用于锚定复现）
    by_class_top3 = {c: lst[:3] for c, lst in by_class.items()}
    run_cfg(
        "static_top3_buyhold",
        build_static_targets(calendar, prices, by_class_top3, class_weights),
        {"rule": "static_top3_by_liquidity_buyhold"},
    )

    print("\n[l3] === 再平衡效应隔离（持仓集不变，只加再平衡） ===", flush=True)
    # top-1：类内无分化（每类只有 1 只）→ static_top1_buyhold vs static_top1_rebal
    #        = **纯类间再平衡效应**；再与 top3 版对比 = 类内再平衡的增量
    by_class_top1 = {c: lst[:1] for c, lst in by_class.items()}
    run_cfg(
        "static_top1_buyhold",
        build_static_targets(calendar, prices, by_class_top1, class_weights),
        {"rule": "static_top1_buyhold"},
    )
    for freq_name, freq in REB_FREQ.items():
        tg, meta = build_rotation_targets(
            calendar, prices, by_class_top1, class_weights, 60, 99, freq, "all")
        run_cfg(
            f"static_top1_rebal_{freq_name}",
            tg,
            {**meta, "rule": "static_top1_rebalanced", "freq": freq_name},
        )
    for freq_name, freq in REB_FREQ.items():
        tg, meta = build_rotation_targets(
            calendar, prices, by_class, class_weights, 60, 99, freq, "all")
        run_cfg(
            f"static_all_rebal_{freq_name}",
            tg,
            {**meta, "rule": "static_all_rebalanced", "freq": freq_name},
        )
        tg, meta = build_rotation_targets(
            calendar, prices, by_class_top3, class_weights, 60, 99, freq, "all")
        run_cfg(
            f"static_top3_rebal_{freq_name}",
            tg,
            {**meta, "rule": "static_top3_rebalanced", "freq": freq_name},
        )

    print("\n[l3] === top1 选股稳健性（换成每类第 2/3 名的 ETF） ===", flush=True)
    # 若 top1 的优势只是「碰巧选中了表现好的那只」，换成第 2/3 名就该明显变差。
    for alt in (1, 2):
        pool = {c: lst[alt: alt + 1] for c, lst in by_class.items()}
        run_cfg(
            f"static_top1_alt{alt + 1}_buyhold",
            build_static_targets(calendar, prices, pool, class_weights),
            {"rule": f"static_top1_liq_rank{alt + 1}_buyhold",
             "codes": {c: v[0] for c, v in pool.items()}},
        )
        tg, meta = build_rotation_targets(
            calendar, prices, pool, class_weights, 60, 99, REB_FREQ["monthly"], "all")
        run_cfg(
            f"static_top1_alt{alt + 1}_rebal_monthly",
            tg,
            {**meta, "rule": f"static_top1_liq_rank{alt + 1}_rebalanced",
             "codes": {c: v[0] for c, v in pool.items()}},
        )

    print("\n[l3] === L3 动量轮动 ===", flush=True)
    for lookback in (20, 60, 120, 250):
        for top_k in (1, 2, 3):
            for freq_name, freq in REB_FREQ.items():
                tg, meta = build_rotation_targets(
                    calendar, prices, by_class, class_weights, lookback, top_k, freq, "momentum")
                run_cfg(
                    f"mom_L{lookback}_K{top_k}_{freq_name}",
                    tg,
                    {**meta, "rule": "momentum", "lookback": lookback, "top_k": top_k, "freq": freq_name},
                )

    print("\n[l3] === 双动量（绝对动量过滤→防御） ===", flush=True)
    for lookback in (60, 120):
        for freq_name, freq in REB_FREQ.items():
            tg, meta = build_rotation_targets(
                calendar, prices, by_class, class_weights, lookback, 1, freq, "dual")
            run_cfg(
                f"dual_L{lookback}_K1_{freq_name}",
                tg,
                {**meta, "rule": "dual", "lookback": lookback, "top_k": 1, "freq": freq_name},
            )

    print("\n[l3] === 风险调整动量（动量/波动） ===", flush=True)
    for lookback in (60, 120):
        tg, meta = build_rotation_targets(
            calendar, prices, by_class, class_weights, lookback, 1, REB_FREQ["monthly"], "rvol")
        run_cfg(
            f"rvol_L{lookback}_K1_monthly",
            tg,
            {**meta, "rule": "rvol", "lookback": lookback, "top_k": 1, "freq": "monthly"},
        )

    print("\n[l3] === 随机对照（平凡解释排除） ===", flush=True)
    # K 必须与动量配置对齐：否则「持仓集中度」与「选标能力」混淆
    for lookback in (60,):
        for top_k in (1, 2, 3):
            for seed in (20261030, 7, 99, 1234, 555, 8888):
                tg, meta = build_rotation_targets(
                    calendar, prices, by_class, class_weights, lookback, top_k,
                    REB_FREQ["monthly"], "random", seed=seed)
                run_cfg(
                    f"random_L{lookback}_K{top_k}_monthly_s{seed}",
                    tg,
                    {**meta, "rule": "random_control", "seed": seed, "top_k": top_k},
                )

    store["candidates"] = by_class
    store["calendar"] = {"n": len(calendar), "start": calendar[0], "end": calendar[-1]}
    store["results"] = results
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(store, fh, ensure_ascii=False, indent=2)
    print(f"\n[l3] ✓ 已写入 {out_path}", flush=True)

    # 缓存 A 股腿供 --quick 使用
    with open(nav_cache, "w", encoding="utf-8") as fh:
        json.dump({"a_share_nav": a_nav, "a_share_leg": a_rec}, fh)


if __name__ == "__main__":
    main()
