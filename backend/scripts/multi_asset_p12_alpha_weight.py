#!/usr/bin/env python3
"""P12：组合层 α/β 分解 + 用 α 夏普重跑 w 曲线 + 长历史第三样本。

## 为什么做这个

P11 的结论把整条研究线的地基改写了：

1. **「主腿弱」废止** —— 主腿 α 夏普 5/5 个窗口为正（0.4937~1.4169）。
   P6 那个「主腿 0.87 / 袖套 1.62 → 该改主腿不是调 w」的推论**前提没了**。
2. **P6 用组合绝对夏普挑 w，而绝对夏普会系统性惩罚低 β 的主腿** →
   「最优 w = 0.20~0.25」很可能是口径产物，不是真实结论。
   P11 §8.2 明确点出：「应当改用组合相对国证A指的 α 夏普重跑 w 曲线」。

## P12 的三个子问题

| # | 问题 | 窗口 | 产出 |
|---|---|---|---|
| **P12-A** | 定稿组合（w=0.5）的 α/β 分解是什么？ | oos + main | 组合相对 4 个基准口径的 α/β/ρ/IR |
| **P12-B** | 用 **α 夏普**（相对国证A指）重跑 w 曲线，最优 w 是多少？ | oos + main | 与 P6 的绝对夏普曲线对照，看结论是否翻转 |
| **P12-C** | 组合层的**长历史**第三样本（终于能做吗？） | w2013 + w2016 | 降级配置（gold+bond 两类袖套）的组合 α/β |

## P12-C 为什么现在才做得成

P6 判据 (f)「有独立样本外证据」一直结构性不满足，两个原因：

1. A 股腿面板从 2021-05-11 起 → **P10 已解决**（`features_full/` 全历史）
2. **ETF 袖套本地数据只回溯到 2019-01** → **本轮解决**：
   `fetch_etf_long_history.py` 实测腾讯源可把 gold/bond 拉到 2013-03，
   但 **commodity 三只（159980/159985/159981）物理上不存在**（2019 底才上市的期货 ETF）
   → 长历史只能用 **gold+bond 两类袖套**（降级配置），必须在文档里显式声明

## 预注册判据（跑之前写死，事后不得放宽）

采纳新 w **必须同时**满足（与 P6 同构，但换评价口径）：

- (a0) α 夏普曲线的最优 w **不得落在网格端点**（端点最优 = 网格无效 = 实验没做完）
- (a) 最优 w **落在 [0.40, 0.60] 之外**
- (b) main 的 α 夏普 ≥ w=0.5 基线 + 0.05
- (c) OOS 的 α 夏普 ≥ w=0.5 基线 − 0.02
- (d) 4 个不重叠子窗口中 ≥ 3 个指向同一 w 方向
- (e) 优于全部 6 个随机 w 对照的最大值
- (f) **独立样本外证据** → 本轮由 P12-C 的长历史窗口提供

**(f) 是否满足由 P12-C 决定；若 P12-C 不可用则仍不满足 → 不改定稿。**

**无论 (a)~(f) 怎么判，定稿（w=0.50）都不自动变更** —— 改 w 必须同时给出
「为什么绝对夏普口径是错的」和「为什么 α 夏普口径是对的」两个论证。

## 口径（与 P3/P6 定稿完全一致，便于逐位复现）

- A 股腿：honest_composite 25 公式版，信号日收盘 → 次日 open，佣金 0.03% + 滑点 0.01%
- 袖套：P5 固定持仓集，类内等权、类间等权、每月 21 交易日再平衡回等权
- 合成：`merge_by_date([a_nav, sleeve_nav], [w, 1-w])`
- ETF 成交：信号日后下一交易日 open，佣金 0.03% + 滑点 0.01%，ETF 免印花税
- 基准：**买入持有等权**（P11 主口径）+ 国证A指 + 沪深300 + 日频再平衡（反例）
- w 网格：0.00 ~ 1.00，步长 0.05（21 点，必须覆盖端点）

## 基线复现门

- w=0.50 的**绝对夏普**必须逐位复现 P3 `bond_lowrho3`（main 1.4114 / oos 1.0685）
- w=0.50 的**组合净值首末日期**必须落在目标窗口内（防模块全局 START/END 静默漂移）

## 用法
    cd backend && env -u PYTHONPATH BASELINE_LOG_LEVEL=ERROR .venv-py313/bin/python3 \\
        scripts/multi_asset_p12_alpha_weight.py --phase ab --window main
    cd backend && env -u PYTHONPATH BASELINE_LOG_LEVEL=ERROR .venv-py313/bin/python3 \\
        scripts/multi_asset_p12_alpha_weight.py --phase ab --window oos
    cd backend && env -u PYTHONPATH BASELINE_LOG_LEVEL=ERROR .venv-py313/bin/python3 \\
        scripts/multi_asset_p12_alpha_weight.py --phase long
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT = os.path.abspath(os.path.join(BACKEND_DIR, ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)
sys.path.insert(0, os.path.join(BACKEND_DIR, "scripts"))

import multi_asset_l3_momentum as L3  # noqa: E402
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

ANN = 244
CLASSES = ["gold", "bond", "commodity"]
K_IN_CLASS = 3
REB_EVERY = 21
SEEDS = (20261030, 7, 99, 1234, 555, 8888)
W_GRID = [round(0.05 * i, 2) for i in range(21)]
RAND_W_LO, RAND_W_HI = 0.30, 0.70
N_SUBWINDOWS = 4

# P5 定稿的固定持仓集
P5_FIXED = {
    "gold": ["518880", "159934", "159937"],
    "commodity": ["159980", "159985", "159981"],
    "bond": ["511260", "511010", "511020"],
}

# P12-C 的降级配置：commodity 物理上不存在 2019 前的数据 → 只有 gold + bond
LONG_CLASSES = ["gold", "bond"]
LONG_FIXED = {
    "gold": ["518880", "159934", "159937"],
    "bond": ["511260", "511010", "511020"],
}

# 长历史 ETF 数据目录（fetch_etf_long_history.py 产出）
ETF_LONG_DIR = os.path.join(ROOT, "data", "etf", "parquet_long")

OUT_DIR = os.path.join(ROOT, "data", "backtest")


# ───────────────────────── 基础统计（与 P11 同口径） ─────────────────────────


def sharpe_from_rets(rets: List[float]) -> float:
    if len(rets) < 3:
        return 0.0
    mu = statistics.fmean(rets)
    sd = statistics.pstdev(rets)
    return mu / sd * math.sqrt(ANN) if sd > 0 else 0.0


def maxdd_from_rets(rets: List[float]) -> float:
    v = 1.0
    peak = 1.0
    mdd = 0.0
    for r in rets:
        v *= (1.0 + r)
        if v > peak:
            peak = v
        if peak > 0:
            dd = v / peak - 1.0
            if dd < mdd:
                mdd = dd
    return mdd


def nav_to_rets(nav: List[Tuple[str, float]]) -> List[float]:
    out = []
    for i in range(1, len(nav)):
        p0, p1 = nav[i - 1][1], nav[i][1]
        out.append(p1 / p0 - 1.0 if (p0 and p0 > 0) else 0.0)
    return out


def decompose(leg_nav: List[Tuple[str, float]],
              bench_nav: List[Tuple[str, float]]) -> Dict[str, Any]:
    """α/β 分解（实现与 P11 完全一致，避免两套口径互为盲区）。"""
    bmap = {d: v for d, v in bench_nav}
    lmap = {d: v for d, v in leg_nav}
    common = sorted(set(lmap) & set(bmap))
    if len(common) < 30:
        raise SystemExit(f"[p12] ❌ 共同日期过少：{len(common)}")
    lr = nav_to_rets([(d, lmap[d]) for d in common])
    br = nav_to_rets([(d, bmap[d]) for d in common])
    ex = [a - b for a, b in zip(lr, br)]
    n = len(lr)
    mb, ml = statistics.fmean(br), statistics.fmean(lr)
    vb = statistics.pvariance(br)
    cov = sum((x - ml) * (y - mb) for x, y in zip(lr, br)) / n
    beta = (cov / vb) if vb > 0 else 0.0
    vl = statistics.pvariance(lr)
    rho = (cov / math.sqrt(vb * vl)) if vb > 0 and vl > 0 else 0.0
    ex_an = [a - beta * b for a, b in zip(lr, br)]
    sd_ex = statistics.pstdev(ex)
    ann_ex = statistics.fmean(ex) * ANN
    return {
        "n_common_days": n,
        "leg_sharpe": sharpe_from_rets(lr),
        "leg_return_ann": statistics.fmean(lr) * ANN,
        "leg_maxdd": maxdd_from_rets(lr),
        "bench_sharpe": sharpe_from_rets(br),
        "bench_return_ann": statistics.fmean(br) * ANN,
        "bench_maxdd": maxdd_from_rets(br),
        "excess_sharpe": sharpe_from_rets(ex),
        "excess_return_ann": ann_ex,
        "alpha_sharpe": sharpe_from_rets(ex_an),
        "beta": beta,
        "corr": rho,
        "ir": (ann_ex / (sd_ex * math.sqrt(ANN))) if sd_ex > 0 else 0.0,
        "excess_maxdd": maxdd_from_rets(ex),
    }


# ───────────────────────── 基准构建 ─────────────────────────


def build_equal_weight_buyhold(wide: Any, leg_dates: List[str]) -> Dict[str, Any]:
    """等权**买入持有**基准：portfolio(t) = mean_i[ P_i(t) / P_i(t0) ]。

    成分与权重全程固定、**不做任何再平衡** → 这是零技巧基准。
    （P11 §2.2：写成 mean_t[mean_i r_i] 再连乘 = 日频等权再平衡 = 给对手塞策略。）
    """
    import pandas as pd

    dates = pd.DatetimeIndex([pd.Timestamp(d) for d in leg_dates])
    cw = wide.reindex(dates)
    alive = cw.notna()
    p = cw.ffill()
    members = [c for c in cw.columns if bool(alive.iloc[0].get(c, False))]
    if not members:
        raise SystemExit("[p12] ❌ 窗口起点无成分股")
    norm = p[members] / p[members].iloc[0]
    port = norm.mean(axis=1)
    rets = port.pct_change(fill_method=None).fillna(0.0)
    rr = [float(x) for x in rets.tolist()]
    rr[0] = 0.0
    nav: List[Tuple[str, float]] = []
    v = 1.0
    for d, x in zip(leg_dates, rr):
        v *= (1.0 + x)
        nav.append((d, v))
    return {"nav": nav, "n_members": len(members)}


def build_equal_weight_rebalance(wide: Any, leg_dates: List[str]) -> Dict[str, Any]:
    """日频等权再平衡基准（P11 §2.1 的反例口径，保留做敏感性）。"""
    import pandas as pd

    dates = pd.DatetimeIndex([pd.Timestamp(d) for d in leg_dates])
    cw = wide.reindex(dates)
    alive = cw.notna()
    p = cw.ffill()
    r = p.pct_change(fill_method=None)
    members = [c for c in cw.columns if bool(alive.iloc[0].get(c, False))]
    sub = r[members].where(alive[members])
    n_alive = sub.notna().sum(axis=1)
    br = (sub.sum(axis=1) / n_alive.replace(0, float("nan"))).fillna(0.0)
    rr = [float(x) for x in br.tolist()]
    rr[0] = 0.0
    nav: List[Tuple[str, float]] = []
    v = 1.0
    for d, x in zip(leg_dates, rr):
        v *= (1.0 + x)
        nav.append((d, v))
    return {"nav": nav, "n_members": len(members)}


def build_index_benchmark(series: Any, leg_dates: List[str],
                          min_coverage: float = 0.8) -> Optional[Dict[str, Any]]:
    """官方指数基准。**覆盖率闸门**：覆盖 < 80% → 不可用。

    （P11 §2.3：510300 本地数据只有 2019 起，在 w2016 窗口 ffill 成 0 收益，
      凭空造出「夏普 0.868」这种熊市里不可能的数字。）
    """
    import pandas as pd

    if series is None or len(series) == 0:
        return None
    dates = pd.DatetimeIndex([pd.Timestamp(d) for d in leg_dates])
    cov = int(series.reindex(dates).notna().sum())
    if cov < min_coverage * len(dates):
        return None
    s = series.reindex(dates).ffill()
    r = s.pct_change(fill_method=None).fillna(0.0)
    rr = [float(x) for x in r.tolist()]
    rr[0] = 0.0
    nav: List[Tuple[str, float]] = []
    v = 1.0
    for d, x in zip(leg_dates, rr):
        v *= (1.0 + x)
        nav.append((d, v))
    return {"nav": nav}


def load_market_proxies() -> Dict[str, Any]:
    """官方指数序列：优先 data/benchmarks/*.parquet，回退 fixtures。"""
    import pandas as pd

    out: Dict[str, Any] = {}
    bench_dir = os.path.join(ROOT, "data", "benchmarks")
    fix_dir = os.path.join(BACKEND_DIR, "fixtures", "market_benchmarks")

    def _ingest(name: str, df) -> None:
        if "trade_date" not in df.columns or "close" not in df.columns:
            return
        s = pd.Series(df["close"].to_numpy(), index=pd.DatetimeIndex(df["trade_date"]))
        out[name] = s[~s.index.duplicated(keep="last")].sort_index()

    if os.path.isdir(bench_dir):
        for fn in sorted(os.listdir(bench_dir)):
            if not fn.endswith(".parquet"):
                continue
            try:
                _ingest(fn[:-len(".parquet")],
                        pd.read_parquet(os.path.join(bench_dir, fn)))
            except Exception:
                continue
    if os.path.isdir(fix_dir):
        for fn in sorted(os.listdir(fix_dir)):
            if not fn.endswith(".csv"):
                continue
            key = fn[:-len(".csv")]
            if key in out:
                continue
            try:
                _ingest(key, pd.read_csv(os.path.join(fix_dir, fn)))
            except Exception:
                continue
    return out


# ───────────────────────── 长历史 ETF 价格 ─────────────────────────


def load_long_prices(codes: List[str]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """从 `data/etf/parquet_long/` 读长历史 ETF 价格。

    ⚠️ **不复用 L3.load_prices**：它读的是 `data/etf/parquet/`（2019 起），
    且会把索引塞进全局单例 `load_prices.index` → 与 P12 的长历史口径混淆。
    这里独立实现，P12-C 专用。
    """
    import pandas as pd

    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for code in codes:
        path = os.path.join(ETF_LONG_DIR, f"{code}.parquet")
        if not os.path.exists(path):
            continue
        df = pd.read_parquet(path).sort_values("date")
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        rows: Dict[str, Dict[str, float]] = {}
        for _, r in df.iterrows():
            rows[str(r["date"])] = {"open": float(r["open"]), "close": float(r["close"])}
        if rows:
            out[code] = rows
    return out


def build_calendar_from(prices: Dict[str, Dict[str, Dict[str, float]]],
                        codes: List[str], start: str, end: str) -> List[str]:
    s: set = set()
    for c in codes:
        s |= {d for d in prices.get(c, {}) if start <= d <= end}
    return sorted(s)


def simulate_long(calendar: List[str],
                  prices: Dict[str, Dict[str, Dict[str, float]]],
                  targets: Dict[str, Dict[str, float]],
                  capital: float = 1_000_000.0) -> Tuple[List[Tuple[str, float]], Dict[str, Any]]:
    """长历史版撮合（独立实现，逻辑与 L3.simulate 一致：信号日前一日出分、
    次日 open 成交、佣金 0.03% + 滑点 0.01%、ETF 免印花税、pro-rata 缩放）。

    为什么独立实现而不是 monkeypatch L3：
    - L3.simulate 从全局单例 `load_prices.index` 取价，而 P12-C 要用**另一套目录**
      的价格（parquet_long）→ 直接改 L3.ETF_PARQUET 会污染同进程里其他实验。
    - P3 的 build_targets 需要 `by_class_held` 的类映射，长历史只有 2 类。
    """
    import bisect

    COMMISSION = 0.0003
    SLIPPAGE = 0.0001

    idx: Dict[str, Dict[str, Any]] = {}
    for code, rows in prices.items():
        ds = sorted(rows)
        idx[code] = {
            "dates": ds,
            "open": [rows[d]["open"] for d in ds],
            "close": [rows[d]["close"] for d in ds],
        }

    def px(code: str, d: str, field: str) -> Optional[float]:
        node = idx.get(code)
        if not node:
            return None
        i = bisect.bisect_right(node["dates"], d) - 1
        if i < 0:
            return None
        return float(node[field][i])

    missing = set()
    for _d, t in targets.items():
        missing |= {c for c in t if c not in idx}
    if missing:
        raise SystemExit(f"[p12] ❌ 长历史价格索引缺少 {len(missing)} 个代码: "
                         f"{sorted(missing)[:5]}")

    cash = capital
    shares: Dict[str, float] = {}
    nav: List[Tuple[str, float]] = []
    total_cost = 0.0
    total_turnover = 0.0
    n_rebal = 0

    for d in calendar:
        tgt = targets.get(d)
        if tgt:
            n_rebal += 1
            val = cash
            for code, sh in shares.items():
                p = px(code, d, "open")
                if p is not None:
                    val += sh * p
            for code in list(shares.keys()):
                if code in tgt:
                    continue
                p = px(code, d, "open")
                if p is None:
                    continue
                proceeds = shares[code] * p * (1 - SLIPPAGE)
                cash += proceeds
                total_cost += proceeds * COMMISSION
                total_turnover += proceeds
                del shares[code]
            # pro-rata 缩放（顺序无关；L3 已修，P12-C 独立实现保持同口径）
            need = 0.0
            plan: List[Tuple[str, float]] = []
            for code, w in tgt.items():
                p = px(code, d, "open")
                if p is None or w <= 0:
                    continue
                amount = val * w
                buy_px = p * (1 + SLIPPAGE)
                need += amount / buy_px * buy_px
                plan.append((code, amount))
            if need > 0:
                scale = 1.0
                if cash < need * (1 + COMMISSION):
                    scale = cash / (need * (1 + COMMISSION))
                for code, amount in plan:
                    p = px(code, d, "open")
                    if p is None:
                        continue
                    buy_px = p * (1 + SLIPPAGE)
                    amt = amount * scale
                    cost = amt * COMMISSION
                    total_cost += cost
                    total_turnover += amt
                    sh = amt / buy_px
                    shares[code] = shares.get(code, 0.0) + sh
                    cash -= amt + cost
        # mark-to-market
        v = cash
        for code, sh in shares.items():
            p = px(code, d, "close")
            if p is not None:
                v += sh * p
        nav.append((d, v))

    rec = {
        "commission_total": round(total_cost, 4),
        "turnover_total": round(total_turnover, 4),
        "turnover_ratio": round(total_turnover / capital, 6),
        "n_rebalances": n_rebal,
    }
    return nav, rec


# ───────────────────────── 袖套构建 ─────────────────────────


def build_sleeve(calendar: List[str],
                 prices: Dict[str, Dict[str, Dict[str, float]]],
                 held: Dict[str, List[str]],
                 classes: List[str],
                 reb_every: int = REB_EVERY) -> Tuple[List[Tuple[str, float]], Dict[str, Any]]:
    """袖套净值：类内等权、类间等权、每 reb_every 个交易日再平衡回类间等权。

    目标权重表构造与 P3.build_targets 完全同构：
    第 i 个交易日（i 从 1 起）若 (i-1) % reb_every == 0 → 信号日 = calendar[i-1]，
    执行日 = calendar[i]。
    """
    targets: Dict[str, Dict[str, float]] = {}
    cw = {c: 1.0 / len(classes) for c in classes}
    for i in range(1, len(calendar)):
        if (i - 1) % reb_every != 0:
            continue
        exec_date = calendar[i]
        t: Dict[str, float] = {}
        for c, w in cw.items():
            codes = held.get(c) or []
            if not codes:
                continue
            for code in codes:
                t[code] = t.get(code, 0.0) + w / len(codes)
        if t:
            targets[exec_date] = t
    nav, rec = simulate_long(calendar, prices, targets)
    return nav, rec


# ───────────────────────── A 股腿 ─────────────────────────


def load_a_share_nav(window: str, start: datetime, end: datetime,
                     quick: bool) -> Tuple[List[Tuple[str, float]], Dict[str, Any]]:
    """A 股腿净值。

    - quick=True：读 `data/backtest/_a_share_nav_{window}.json` 缓存
    - quick=False：跑真实引擎（5513 只，慢）

    ⚠️ **模块全局 START/END 必须同步**（`multi_asset_leg_experiment` 的
    `run_a_share_leg` 读全局，不传参）→ 不改会让 OOS 静默用 main 的腿。
    """
    import multi_asset_leg_experiment as M

    cache = os.path.join(OUT_DIR, f"_a_share_nav_{window}.json")
    if quick and os.path.exists(cache):
        with open(cache, encoding="utf-8") as fh:
            blob = json.load(fh)
        nav = [(d, float(v)) for d, v in blob["a_share_nav"]]
        print(f"[p12] 读缓存 A 股腿 {window}: {len(nav)} 点", flush=True)
        return nav, blob.get("a_share_leg", {})

    M.START, M.END = start, end
    from verify_methods_engine import QLIB_OFFICIAL, universe_from_instruments

    codes = universe_from_instruments(QLIB_OFFICIAL)
    print(f"[p12] 跑 A 股腿 {window}（{len(codes)} 只）...", flush=True)
    rec, nav = M.run_a_share_leg(codes)
    ver = M.verify_baseline(rec)
    print(f"[p12] 基线复现门: {json.dumps(ver, ensure_ascii=False)}", flush=True)
    if ver.get("checked") and not ver.get("ok"):
        raise SystemExit("[p12] ❌ A 股腿基线复现失败，退出")
    with open(cache, "w", encoding="utf-8") as fh:
        json.dump({"a_share_nav": nav, "a_share_leg": rec}, fh, ensure_ascii=False)
    return nav, rec


def assert_window(nav: List[Tuple[str, float]], s_str: str, e_str: str, what: str) -> None:
    """净值首末日期必须**都**落在目标窗口内。

    只断言起点会被边界日巧合放行（P7 踩过）。
    """
    if not nav:
        raise SystemExit(f"[p12] ❌ {what} 净值为空")
    d0, d1 = nav[0][0], nav[-1][0]
    if not (s_str <= d0 <= e_str and s_str <= d1 <= e_str):
        raise SystemExit(f"[p12] ❌ {what} 净值区间 {d0}~{d1} 不在窗口 {s_str}~{e_str} 内")


# ───────────────────────── 阶段 A/B ─────────────────────────


def run_ab(window: str, sleeve_mode: str = "full") -> Dict[str, Any]:
    """P12-A（组合层分解）+ P12-B（α 夏普重跑 w 曲线）。

    sleeve_mode:
      "full"     = 定稿三类配置（gold/commodity/bond），P5 固定持仓集
      "degraded" = 降级两类配置（gold/bond），与 P12-D 长历史同口径
                   → 用于**配置敏感性检验**：配置一换，w 曲线结论会不会翻？
    """
    import multi_asset_p11_alpha_beta as P11

    start, end = (START, END) if window == "main" else (OOS_START, OOS_END)
    s_str, e_str = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
    out_path = os.path.join(OUT_DIR, f"multi_asset_p12_{window}.json") if sleeve_mode == "full" \
        else os.path.join(OUT_DIR, f"multi_asset_p12_{window}_{sleeve_mode}.json")

    # ── A 股腿 ──
    a_nav, a_rec = load_a_share_nav(window, start, end, quick=True)
    assert_window(a_nav, s_str, e_str, "A 股腿")
    print(f"[p12] A 股腿 {len(a_nav)} 天，夏普 {sharpe(a_nav):.4f}", flush=True)

    # ── 候选池 / 固定持仓集（与 P6 同口径）──
    pool = P3.build_pool(start)
    universe = load_etf_universe()
    by_class_raw = L3.candidates_by_class(universe, start, end)
    base_codes = sorted({c for lst in by_class_raw.values() for c in lst})
    if sleeve_mode == "degraded":
        # 配置敏感性：与 P12-D 长历史同口径的 gold+bond 两类
        held = {c: list(LONG_FIXED[c]) for c in LONG_CLASSES}
        use_classes = list(LONG_CLASSES)
        p5_match = {c: (sorted(held.get(c, [])) == sorted(LONG_FIXED[c])) for c in LONG_CLASSES}
    else:
        held = {c: list(P5_FIXED[c]) for c in P5_FIXED}
        use_classes = list(CLASSES)
        p5_match = {c: (sorted(held.get(c, [])) == sorted(P5_FIXED[c])) for c in P5_FIXED}
    _known = set(base_codes) | {r["code"] for r in pool}
    missing_codes = [c for lst in held.values() for c in lst if c not in _known]
    if not all(p5_match.values()) or missing_codes:
        raise SystemExit(f"[p12] ❌ 固定持仓集异常 match={p5_match} missing={missing_codes}")
    print(f"[p12] 固定持仓集(P5 定稿): {json.dumps(held, ensure_ascii=False)}", flush=True)

    # ── 价格 / 日历 ──
    all_codes = sorted(set(base_codes) | {r["code"] for r in pool})
    prices = L3.load_prices(all_codes)
    calendar = L3.build_calendar(prices, base_codes, s_str, e_str)
    print(f"[p12] 交易日历 {len(calendar)} 天 {calendar[0]} → {calendar[-1]}", flush=True)
    if len(calendar) < 300:
        raise SystemExit("[p12] ❌ 交易日历过短，退出")

    # ── 袖套 ──
    for c, lst in held.items():
        if len(set(lst)) != len(lst):
            raise SystemExit(f"[p12] ❌ 类 {c} 持仓重复 {lst}")
    tg, tmeta = P3.build_targets(calendar, held, P3.fixed_weights_for(use_classes))
    c_of = {c: cls for cls, lst in held.items() for c in lst}
    sleeve_nav, srec = L3.simulate(calendar, prices, tg, c_of,
                                    {c: 1.0 / len(use_classes) for c in use_classes})
    if not sleeve_nav:
        raise SystemExit("[p12] ❌ 袖套无净值")
    print(f"[p12] 袖套 {len(sleeve_nav)} 天，夏普 {sharpe(sleeve_nav):.4f} "
          f"换手 {srec['turnover_ratio']:.2f}x", flush=True)

    # ── 基准 ──
    wide = P11.load_close_panel()
    proxies = load_market_proxies()

    leg_dates = [d for d, _ in a_nav]

    def bench_bundle(nav: List[Tuple[str, float]]) -> Dict[str, Any]:
        """为一条腿构建全部基准口径。"""
        dates = [d for d, _ in nav]
        out: Dict[str, Any] = {}
        bh = build_equal_weight_buyhold(wide, dates)
        out["equal_weight_buyhold"] = {
            "n_members": bh["n_members"],
            "decomp": decompose(nav, bh["nav"]),
        }
        rb = build_equal_weight_rebalance(wide, dates)
        out["equal_weight_rebalance"] = {
            "n_members": rb["n_members"],
            "decomp": decompose(nav, rb["nav"]),
            "note": "P11 §2.1 反例口径：每日等权再平衡 = 含反转技巧，不该用作基准",
        }
        for key, series in proxies.items():
            bm = build_index_benchmark(series, dates)
            if bm is None:
                out[key] = {"available": False, "reason": "覆盖率不足 80%"}
                continue
            out[key] = {"available": True, "decomp": decompose(nav, bm["nav"])}
        return out

    # ── P12-A：组合层分解（w=0.50 定稿）──
    comb_base = merge_by_date([a_nav, sleeve_nav], [0.5, 0.5])
    assert_window(comb_base, s_str, e_str, "组合(w=0.5)")
    combo_bench = bench_bundle(comb_base)
    print(f"[p12] === P12-A 组合层分解 (w=0.50) ===", flush=True)
    for k, v in combo_bench.items():
        d = v.get("decomp")
        if not d:
            print(f"[p12]   {k}: 不可用 ({v.get('reason')})", flush=True)
            continue
        print(f"[p12]   {k}: 组合夏普 {d['leg_sharpe']:.4f} | 基准夏普 {d['bench_sharpe']:.4f} | "
              f"超额夏普 {d['excess_sharpe']:.4f} | α夏普 {d['alpha_sharpe']:.4f} | "
              f"β {d['beta']:.4f} | ρ {d['corr']:.4f}", flush=True)

    # ── 基线复现门：w=0.50 绝对夏普必须逐位复现 P3 ──
    base_abs_sharpe = sharpe(comb_base)
    p3p = os.path.join(OUT_DIR, f"multi_asset_p3_{window}.json")
    if sleeve_mode == "degraded":
        # 降级配置的袖套与 P3 定稿不同 → P3 基线不适用，改为「袖套自身可复现」自检
        exp = None
        print(f"[p12] ⚠️ 降级配置模式：跳过 P3 基线门（袖套配置不同）", flush=True)
        base_ok = None
    elif not os.path.exists(p3p):
        raise SystemExit(f"[p12] ❌ 找不到 P3 产物 {p3p}，无法建立基线门")
    p3key = None
    if sleeve_mode != "degraded":
        with open(p3p, encoding="utf-8") as fh:
            p3d = json.load(fh)
        p3res = p3d.get("results", {})
        p3key = ("bond_lowrho3_noliq" if "bond_lowrho3_noliq" in p3res else "bond_lowrho3")
        exp = float(p3res[p3key]["combined_sharpe"])
        print(f"[p12] 基线复现门: w=0.50 绝对夏普 {base_abs_sharpe:.4f}（P3 定稿 {exp}）", flush=True)
        if abs(base_abs_sharpe - exp) > 5e-4:
            raise SystemExit(f"[p12] ❌ 基线漂移 {base_abs_sharpe} != {exp}，退出")

    # ── P12-B：w 曲线 × 多评价口径 ──
    # 主口径：**相对国证A指（IDX_399317）的 α 夏普**（P11 §8.2 指定）
    # 对照：等权买入持有的 α 夏普、绝对夏普（P6 原口径）
    print(f"\n[p12] === P12-B w 曲线（21 点）× 4 评价口径 ===", flush=True)
    results: Dict[str, Any] = {}
    navs: Dict[str, List[Tuple[str, float]]] = {}
    w_values: Dict[str, float] = {}

    def eval_w(w: float, label: str, meta: Optional[Dict[str, Any]] = None):
        comb = merge_by_date([a_nav, sleeve_nav], [w, 1.0 - w])
        navs[label] = comb
        w_values[label] = w
        rec: Dict[str, Any] = {
            "w_a_share": round(w, 4),
            "combined_sharpe": round(sharpe(comb), 4),
            "combined_return": round(total_ret(comb), 4),
            "combined_maxdd": round(maxdd(comb), 4),
            "combined_vol": round(P2_vol(comb), 4),
            "combined_yearly": yearly(comb),
            **(meta or {}),
        }
        # 三个基准口径下的 α 夏普
        for bkey in ("IDX_399317", "equal_weight_buyhold", "IDX_000300"):
            series = proxies.get(bkey)
            if bkey == "equal_weight_buyhold":
                bm = build_equal_weight_buyhold(wide, [d for d, _ in comb])
            else:
                bm = build_index_benchmark(series, [d for d, _ in comb]) if series is not None else None
            if bm is None:
                rec[f"alpha_sharpe_{bkey}"] = None
                rec[f"beta_{bkey}"] = None
                continue
            d = decompose(comb, bm["nav"])
            rec[f"alpha_sharpe_{bkey}"] = round(d["alpha_sharpe"], 4)
            rec[f"excess_sharpe_{bkey}"] = round(d["excess_sharpe"], 4)
            rec[f"beta_{bkey}"] = round(d["beta"], 4)
            rec[f"corr_{bkey}"] = round(d["corr"], 4)
        results[label] = rec
        print(f"  {label:10s} w={w:.2f} | 绝对夏普 {rec['combined_sharpe']:.4f} | "
              f"α夏普(国证A指) {rec.get('alpha_sharpe_IDX_399317')} | "
              f"α夏普(等权持有) {rec.get('alpha_sharpe_equal_weight_buyhold')}", flush=True)
        return rec

    for w in W_GRID:
        eval_w(w, f"w{int(round(w * 100)):03d}")

    # ── 随机 w 对照（隔离「w 是个可调参数」这件事本身）──
    print(f"\n[p12] === 随机 w 对照（U({RAND_W_LO}, {RAND_W_HI})，{len(SEEDS)} 个种子）===", flush=True)
    rnd_ws: List[float] = []
    for seed in SEEDS:
        rng = random.Random(seed)
        w = round(rng.uniform(RAND_W_LO, RAND_W_HI), 4)
        rnd_ws.append(w)
        eval_w(w, f"randw_s{seed}", {"note": f"随机 w={w}", "seed": seed})
    if len(set(rnd_ws)) != len(rnd_ws):
        raise SystemExit(f"[p12] ❌ 随机 w 退化（重复）: {rnd_ws}")

    # ── 曲线形状 & 判据 ──
    metrics = {
        "abs_sharpe": "combined_sharpe",
        "alpha_gz": "alpha_sharpe_IDX_399317",
        "alpha_ew": "alpha_sharpe_equal_weight_buyhold",
        "alpha_hs300": "alpha_sharpe_IDX_000300",
    }
    curves: Dict[str, Any] = {}
    for mname, mfield in metrics.items():
        grid = [(results[f"w{int(round(w * 100)):03d}"].get(mfield), w)
                for w in W_GRID]
        grid = [(s, w) for s, w in grid if s is not None]
        if not grid:
            continue
        best_s, best_w = max(grid)
        at_edge = (best_w <= min(W_GRID) + 1e-9) or (best_w >= max(W_GRID) - 1e-9)
        rnd_vals = [results[f"randw_s{s}"].get(mfield) for s in SEEDS]
        rnd_vals = [v for v in rnd_vals if v is not None]
        rand_max = max(rnd_vals) if rnd_vals else None
        adj = [abs(grid[i + 1][0] - grid[i][0]) for i in range(len(grid) - 1)]
        # 子窗口：每个子段的最优 w（**必须用该指标自己的基准**，不能恒用等权持有）
        base = navs[f"w{int(round(W_GRID[0] * 100)):03d}"]
        dates = [d for d, _ in base]
        n = len(dates)
        edges = [n * i // N_SUBWINDOWS for i in range(N_SUBWINDOWS + 1)]
        argmaxes: List[float] = []
        sub_detail: Dict[str, Any] = {}
        for si in range(N_SUBWINDOWS):
            lo, hi = edges[si], edges[si + 1]
            if hi - lo < 60:
                continue
            dset = set(dates[lo:hi])
            vals = []
            for w in W_GRID:
                lab = f"w{int(round(w * 100)):03d}"
                seg = [(d, v) for d, v in navs[lab] if d in dset]
                if len(seg) < 30:
                    continue
                seg_dates = [d for d, _ in seg]
                if mfield == "combined_sharpe":
                    vals.append((sharpe(seg), w))
                    continue
                bkey = mfield.replace("alpha_sharpe_", "")
                if bkey == "equal_weight_buyhold":
                    bm = build_equal_weight_buyhold(wide, seg_dates)
                else:
                    ser = proxies.get(bkey)
                    bm = build_index_benchmark(ser, seg_dates) if ser is not None else None
                if bm is None:
                    continue
                dd = decompose(seg, bm["nav"])
                vals.append((dd["alpha_sharpe"], w))
            if vals:
                # vals 的元组顺序是 (指标值, w) → max 后 [0]=值 [1]=w
                bval, bw = max(vals)
                argmaxes.append(bw)
                sub_detail[f"sub{si}"] = {
                    "start": dates[lo], "end": dates[hi - 1], "n": hi - lo,
                    "argmax_w": bw, "argmax_value": round(bval, 4),
                }
        mode_arg = max(set(argmaxes), key=argmaxes.count) if argmaxes else None
        # ⚠️ 不能用 `if mode_arg` —— w=0.0 是 falsy → mode_count 被误判成 0
        mode_count = argmaxes.count(mode_arg) if mode_arg is not None else 0
        curves[mname] = {
            "field": mfield,
            "best_w": best_w,
            "best_value": round(best_s, 4),
            "best_w_at_grid_edge": at_edge,
            "random_max": round(rand_max, 4) if rand_max is not None else None,
            "random_min": round(min(rnd_vals), 4) if rnd_vals else None,
            "max_adjacent_jump": round(max(adj), 4) if adj else None,
            "subwindow_argmax_mode": mode_arg,
            "subwindow_mode_count": mode_count,
            "subwindow_n": len(argmaxes),
            "subwindow_argmaxes": argmaxes,
            "subwindow_detail": sub_detail,
        }
        print(f"[p12] 口径 {mname}: 最优 w={best_w} 值 {best_s:.4f} 端点最优={at_edge} "
              f"随机最大 {curves[mname]['random_max']} 子窗口众数 {mode_arg}"
              f"({mode_count}/{len(argmaxes)})", flush=True)

    # ── 预注册判据（用主口径 alpha_gz 判定）──
    primary = "alpha_gz"
    cur = curves.get(primary) or {}
    base_val = results["w050"].get(metrics[primary])
    best_val = cur.get("best_value")
    at_edge = cur.get("best_w_at_grid_edge")
    best_w = cur.get("best_w")
    criteria = {
        "a0_curve_has_interior_max_not_edge": (not at_edge) if at_edge is not None else None,
        "a_best_w_outside_0.40_0.60": (not (0.40 <= best_w <= 0.60)) if best_w is not None else None,
        "b_alpha_sharpe_ge_base_plus_0.05": (best_val - base_val >= 0.05)
        if (best_val is not None and base_val is not None) else None,
        "c_oos_alpha_sharpe_ge_base_minus_0.02": None,   # 另一个窗口跑完才判
        "d_subwindow_majority_same_w": (cur.get("subwindow_mode_count", 0) >= 3)
        if cur.get("subwindow_n") else None,
        "e_beats_all_random_w": (best_val > cur.get("random_max"))
        if (best_val is not None and cur.get("random_max") is not None) else None,
        "f_independent_oos_evidence": False,   # 由 P12-C 决定
    }
    passed = [k for k, v in criteria.items() if v is True]
    print(f"\n[p12] === 预注册判据（主口径 {primary}）===", flush=True)
    for k, v in criteria.items():
        print(f"[p12]   {'✅' if v is True else ('—' if v is None else '❌')} {k} = {v}", flush=True)

    store: Dict[str, Any] = {
        "phase": "ab",
        "window": window,
        "start": s_str,
        "end": e_str,
        "held": held,
        "p5_fixed_match": p5_match,
        "a_share_sharpe": round(sharpe(a_nav), 4),
        "a_share_leg": a_rec,
        "sleeve": {
            "sharpe": round(sharpe(sleeve_nav), 4),
            "vol": round(P2_vol(sleeve_nav), 4),
            "cost": srec,
            "n_rebalances": tmeta["n_rebalances"],
        },
        "baseline_gate": {
            "w050_abs_sharpe": round(base_abs_sharpe, 4),
            "p3_expected": exp,
            "p3_key": p3key if sleeve_mode != "degraded" else None,
            "ok": (abs(base_abs_sharpe - exp) <= 5e-4) if exp is not None else None,
            "sleeve_mode": sleeve_mode,
        },
        "primary_metric": primary,
        "combo_decomposition_w050": combo_bench,
        "curves": curves,
        "criteria": criteria,
        "criteria_passed": passed,
        "results": results,
        "random_w": {str(s): rnd_ws[i] for i, s in enumerate(SEEDS)},
        "note": ("评价口径换了（绝对夏普 → 相对国证A指的 α 夏普），"
                 "但**定稿 w=0.50 不自动变更**：改 w 需要同时论证「绝对夏普口径是错的」"
                 "与「α 夏普口径是对的」。判据 (f) 由 P12-C 的长历史提供。"),
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(store, fh, ensure_ascii=False, indent=2)
    print(f"[p12] ✓ 写出 {out_path}", flush=True)
    return store


def P2_vol(nav: List[Tuple[str, float]]) -> float:
    """年化波动（与 P2.nav_vol 同口径）。"""
    import multi_asset_p2 as P2

    return P2.nav_vol(nav)


# ───────────────────────── 阶段 C：长历史第三样本 ─────────────────────────


def run_long() -> Dict[str, Any]:
    """P12-C：长历史（gold+bond 两类袖套）的组合 α/β 分解。

    ⚠️ **降级配置**：commodity 三只 ETF 物理上不存在 2019 前的数据 →
    长历史只有 gold + bond 两类袖套（类间等权 50/50），**不是定稿的三类配置**。
    这个窗口不能交叉验证定稿的 commodity 类，只能验证「组合层 α 是否跨窗口为正」
    与「w 曲线在长历史上的形状」。
    """
    import multi_asset_p11_alpha_beta as P11

    out_path = os.path.join(OUT_DIR, "multi_asset_p12_long.json")
    # 长历史窗口：与 P10/P11 的 A 股腿窗口对齐
    WINDOWS = [
        ("w2013", datetime(2013, 5, 8), datetime(2016, 5, 8)),
        ("w2016", datetime(2016, 5, 8), datetime(2019, 5, 8)),
        ("w2019", datetime(2019, 5, 8), datetime(2021, 5, 11)),
    ]
    results: Dict[str, Any] = {}
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as fh:
            results = (json.load(fh).get("results") or {})

    long_codes = sorted({c for lst in LONG_FIXED.values() for c in lst})
    prices = load_long_prices(long_codes)
    missing = [c for c in long_codes if c not in prices]
    if missing:
        raise SystemExit(f"[p12] ❌ 长历史 ETF 缺数据: {missing}")
    print(f"[p12] 长历史 ETF 载入 {len(prices)} 只: {sorted(prices)}", flush=True)

    wide = P11.load_close_panel()
    proxies = load_market_proxies()

    for label, start, end in WINDOWS:
        s_str, e_str = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
        print(f"[p12] 长历史窗口 {label}: {s_str}~{e_str}", flush=True)

        # ── A 股腿（复用 P10 的长历史产物）──
        p11p = os.path.join(OUT_DIR, "multi_asset_p11_alpha.json")
        if not os.path.exists(p11p):
            raise SystemExit("[p12] ❌ 找不到 P11 产物（内含长历史 A 腿净值）")
        with open(p11p, encoding="utf-8") as fh:
            p11 = json.load(fh)
        v = (p11.get("results") or {}).get(label)
        if not v or not v.get("leg_nav_pairs"):
            raise SystemExit(f"[p12] ❌ P11 产物缺窗口 {label} 的 A 腿净值")
        a_nav = [(d, float(x)) for d, x in v["leg_nav_pairs"]]
        assert_window(a_nav, s_str, e_str, f"A 股腿({label})")
        print(f"[p12]   A 股腿 {len(a_nav)} 天，夏普 {sharpe(a_nav):.4f}", flush=True)

        # ── 袖套（gold+bond 两类，长历史价格）──
        calendar = build_calendar_from(prices, long_codes, s_str, e_str)
        print(f"[p12]   日历 {len(calendar)} 天 {calendar[0]} → {calendar[-1]}", flush=True)
        if len(calendar) < 300:
            raise SystemExit(f"[p12] ❌ 日历过短，退出")
        # 类内可用性：某些标的可能在该窗口内才上市 → 逐窗口披露实际 held
        avail: Dict[str, List[str]] = {}
        for c, lst in LONG_FIXED.items():
            ok = [x for x in lst
                  if any(d in prices[x] for d in calendar)]
            if not ok:
                raise SystemExit(f"[p12] ❌ 类 {c} 在窗口 {label} 内无任何可用标的")
            avail[c] = ok
        sleeve_nav, srec = build_sleeve(calendar, prices, avail, LONG_CLASSES)
        print(f"[p12]   袖套(降级 gold+bond) {len(sleeve_nav)} 天，夏普 {sharpe(sleeve_nav):.4f} "
              f"held={json.dumps(avail, ensure_ascii=False)}", flush=True)

        # ── 组合分解（w=0.50）──
        comb = merge_by_date([a_nav, sleeve_nav], [0.5, 0.5])
        dates = [d for d, _ in comb]
        bh = build_equal_weight_buyhold(wide, dates)
        d_bh = decompose(comb, bh["nav"])
        entry: Dict[str, Any] = {
            "window": [s_str, e_str],
            "config": "DEGRADED gold+bond only (commodity 无 2019 前数据)",
            "held": avail,
            "n_calendar": len(calendar),
            "a_share_sharpe": round(sharpe(a_nav), 4),
            "sleeve_sharpe": round(sharpe(sleeve_nav), 4),
            "sleeve_vol": round(P2_vol(sleeve_nav), 4),
            "sleeve_cost": srec,
            "combo_sharpe": round(sharpe(comb), 4),
            "combo_return": round(total_ret(comb), 4),
            "combo_maxdd": round(maxdd(comb), 4),
            "decomp_ew_buyhold": d_bh,
            "benchmarks": {},
        }
        for key, series in proxies.items():
            bm = build_index_benchmark(series, dates)
            if bm is None:
                entry["benchmarks"][key] = {"available": False, "reason": "覆盖率不足 80%"}
                continue
            entry["benchmarks"][key] = {"available": True, "decomp": decompose(comb, bm["nav"])}
        print(f"[p12]   [等权持有] 组合夏普 {d_bh['leg_sharpe']:.4f} | 基准 {d_bh['bench_sharpe']:.4f} | "
              f"超额夏普 {d_bh['excess_sharpe']:.4f} | α夏普 {d_bh['alpha_sharpe']:.4f} | "
              f"β {d_bh['beta']:.4f} | ρ {d_bh['corr']:.4f}", flush=True)
        for k, pv in entry["benchmarks"].items():
            if pv.get("available"):
                dd = pv["decomp"]
                print(f"[p12]   [{k}] α夏普 {dd['alpha_sharpe']:.4f} | 超额夏普 {dd['excess_sharpe']:.4f} "
                      f"| β {dd['beta']:.4f}", flush=True)

        results[label] = entry
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump({"phase": "long", "results": results}, fh, ensure_ascii=False, indent=2)

    # ── 汇总：α 是否跨窗口同号 ──
    summary: Dict[str, Any] = {}
    for m in ("decomp_ew_buyhold",):
        vals = []
        for label in results:
            d = results[label].get(m)
            if d:
                vals.append(d["alpha_sharpe"])
        if vals:
            summary[m] = {
                "n": len(vals),
                "n_positive": sum(1 for v in vals if v > 0),
                "min": round(min(vals), 4),
                "max": round(max(vals), 4),
                "mean": round(statistics.fmean(vals), 4),
                "median": round(statistics.median(vals), 4),
            }
    print(f"\n[p12] === 长历史汇总（组合 α 夏普，等权持有基准）===", flush=True)
    for k, v in summary.items():
        print(f"[p12]   {k}: {v}", flush=True)

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({"phase": "long", "results": results, "summary": summary},
                  fh, ensure_ascii=False, indent=2)
    print(f"[p12] ✓ 写出 {out_path}", flush=True)
    return {"results": results, "summary": summary}



# ───────────────────────── 阶段 D：长历史 w 曲线 ─────────────────────────

_BM_CACHE: Dict[Tuple[str, str, str, int], Optional[Dict[str, Any]]] = {}


def _bench_cached(kind: str, wide: Any, proxies: Dict[str, Any],
                  dates: List[str]) -> Optional[Dict[str, Any]]:
    """基准按 (kind, 起, 止, 天数) 缓存：同一窗口内所有 w 的日期集合相同 → 只算一次。"""
    if not dates:
        return None
    key = (kind, dates[0], dates[-1], len(dates))
    if key in _BM_CACHE:
        return _BM_CACHE[key]
    if kind == "equal_weight_buyhold":
        bm = build_equal_weight_buyhold(wide, dates)
    else:
        series = proxies.get(kind)
        bm = build_index_benchmark(series, dates) if series is not None else None
    _BM_CACHE[key] = bm
    return bm


def run_longw() -> Dict[str, Any]:
    """P12-D：长历史三个窗口的 w 曲线（降级 gold+bond 袖套）× 4 评价口径。

    目的：给判据 (f) 提供 **独立于 main/oos** 的证据。
    主问题不是「哪个 w 最好」，而是「α 夏普口径下的最优 w 在 5 个独立窗口里
    是否稳定落在 [0.40, 0.60]」：
      - main  α_gz best_w = 0.25（区间外）
      - oos   α_gz best_w = 0.55（区间内）  ← 两者已经打架
    若长历史三个窗口再加进来仍然分散 → 「α 口径应该改 w」是噪声驱动 → 定稿 w=0.50 不动。

    ⚠️ 降级配置：commodity 无 2019 前数据 → 袖套只有 gold + bond 两类（类间等权）。
       所以这里能验证的是「最优 w 的跨窗口一致性」，不能验证定稿的三类配置。
    """
    import multi_asset_p11_alpha_beta as P11

    out_path = os.path.join(OUT_DIR, "multi_asset_p12_longw.json")
    WINDOWS = [
        ("w2013", datetime(2013, 5, 8), datetime(2016, 5, 8)),
        ("w2016", datetime(2016, 5, 8), datetime(2019, 5, 8)),
        ("w2019", datetime(2019, 5, 8), datetime(2021, 5, 11)),
    ]
    METRICS = {
        "abs_sharpe": None,
        "alpha_gz": "IDX_399317",
        "alpha_ew": "equal_weight_buyhold",
        "alpha_hs300": "IDX_000300",
    }

    long_codes = sorted({c for lst in LONG_FIXED.values() for c in lst})
    prices = load_long_prices(long_codes)
    missing = [c for c in long_codes if c not in prices]
    if missing:
        raise SystemExit(f"[p12] ❌ 长历史 ETF 缺数据: {missing}")

    wide = P11.load_close_panel()
    proxies = load_market_proxies()
    p11p = os.path.join(OUT_DIR, "multi_asset_p11_alpha.json")
    if not os.path.exists(p11p):
        raise SystemExit("[p12] ❌ 找不到 P11 产物")
    with open(p11p, encoding="utf-8") as fh:
        p11 = json.load(fh)

    results: Dict[str, Any] = {}
    for label, start, end in WINDOWS:
        s_str, e_str = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
        print(f"\n[p12] === P12-D 长历史 w 曲线：{label} {s_str}~{e_str} ===", flush=True)

        v = (p11.get("results") or {}).get(label)
        if not v or not v.get("leg_nav_pairs"):
            raise SystemExit(f"[p12] ❌ P11 产物缺窗口 {label}")
        a_nav = [(d, float(x)) for d, x in v["leg_nav_pairs"]]
        assert_window(a_nav, s_str, e_str, f"A 股腿({label})")

        calendar = build_calendar_from(prices, long_codes, s_str, e_str)
        if len(calendar) < 300:
            raise SystemExit(f"[p12] ❌ 日历过短 {label}")
        avail: Dict[str, List[str]] = {}
        for c, lst in LONG_FIXED.items():
            ok = [x for x in lst if any(d in prices[x] for d in calendar)]
            if not ok:
                raise SystemExit(f"[p12] ❌ 类 {c} 在 {label} 无可用标的")
            avail[c] = ok
        sleeve_nav, srec = build_sleeve(calendar, prices, avail, LONG_CLASSES)
        print(f"[p12]   袖套(降级) 夏普 {sharpe(sleeve_nav):.4f} held={json.dumps(avail, ensure_ascii=False)}",
              flush=True)

        recs: Dict[str, Any] = {}
        navs: Dict[str, List[Tuple[str, float]]] = {}

        def eval_w(w: float, lab: str) -> Dict[str, Any]:
            comb = merge_by_date([a_nav, sleeve_nav], [w, 1.0 - w])
            navs[lab] = comb
            dates = [d for d, _ in comb]
            rec: Dict[str, Any] = {
                "w_a_share": round(w, 4),
                "combined_sharpe": round(sharpe(comb), 4),
                "combined_return": round(total_ret(comb), 4),
                "combined_maxdd": round(maxdd(comb), 4),
            }
            for mname, bkey in METRICS.items():
                if mname == "abs_sharpe":
                    continue
                bm = _bench_cached(bkey, wide, proxies, dates)
                if bm is None:
                    rec[f"alpha_sharpe_{bkey}"] = None
                    rec[f"beta_{bkey}"] = None
                    continue
                dd = decompose(comb, bm["nav"])
                rec[f"alpha_sharpe_{bkey}"] = round(dd["alpha_sharpe"], 4)
                rec[f"beta_{bkey}"] = round(dd["beta"], 4)
                rec[f"corr_{bkey}"] = round(dd["corr"], 4)
            recs[lab] = rec
            return rec

        for w in W_GRID:
            eval_w(w, f"w{int(round(w * 100)):03d}")
        rnd_ws: List[float] = []
        for seed in SEEDS:
            rng = random.Random(seed)
            w = round(rng.uniform(RAND_W_LO, RAND_W_HI), 4)
            rnd_ws.append(w)
            eval_w(w, f"randw_s{seed}")
        if len(set(rnd_ws)) != len(rnd_ws):
            raise SystemExit(f"[p12] ❌ 随机 w 退化: {rnd_ws}")

        curves: Dict[str, Any] = {}
        for mname, bkey in METRICS.items():
            field = "combined_sharpe" if mname == "abs_sharpe" else f"alpha_sharpe_{bkey}"
            grid = [(recs[f"w{int(round(w * 100)):03d}"].get(field), w) for w in W_GRID]
            grid = [(s, w) for s, w in grid if s is not None]
            if not grid:
                continue
            best_s, best_w = max(grid)
            rnd = [recs[f"randw_s{s}"].get(field) for s in SEEDS]
            rnd = [x for x in rnd if x is not None]
            rand_max = max(rnd) if rnd else None
            # 子窗口（用该口径自己的基准）
            base = navs[f"w{int(round(W_GRID[0] * 100)):03d}"]
            dates_all = [d for d, _ in base]
            n = len(dates_all)
            edges = [n * i // N_SUBWINDOWS for i in range(N_SUBWINDOWS + 1)]
            argmaxes: List[float] = []
            detail: Dict[str, Any] = {}
            for si in range(N_SUBWINDOWS):
                lo, hi = edges[si], edges[si + 1]
                if hi - lo < 60:
                    continue
                dset = set(dates_all[lo:hi])
                vals: List[Tuple[float, float]] = []
                for w in W_GRID:
                    lab = f"w{int(round(w * 100)):03d}"
                    seg = [(d, x) for d, x in navs[lab] if d in dset]
                    if len(seg) < 30:
                        continue
                    if mname == "abs_sharpe":
                        vals.append((sharpe(seg), w))
                        continue
                    bm = _bench_cached(bkey, wide, proxies, [d for d, _ in seg])
                    if bm is None:
                        continue
                    vals.append((decompose(seg, bm["nav"])["alpha_sharpe"], w))
                if vals:
                    bval, bw = max(vals)
                    argmaxes.append(bw)
                    detail[f"sub{si}"] = {
                        "start": dates_all[lo], "end": dates_all[hi - 1],
                        "argmax_w": bw, "argmax_value": round(bval, 4),
                    }
            mode_arg = max(set(argmaxes), key=argmaxes.count) if argmaxes else None
            mode_count = argmaxes.count(mode_arg) if mode_arg is not None else 0
            curves[mname] = {
                "field": field,
                "best_w": best_w,
                "best_value": round(best_s, 4),
                "best_w_at_grid_edge": (best_w <= min(W_GRID) + 1e-9) or (best_w >= max(W_GRID) - 1e-9),
                "random_max": round(rand_max, 4) if rand_max is not None else None,
                "random_min": round(min(rnd), 4) if rnd else None,
                "subwindow_argmax_mode": mode_arg,
                "subwindow_mode_count": mode_count,
                "subwindow_n": len(argmaxes),
                "subwindow_detail": detail,
            }
            print(f"[p12]   口径 {mname}: 最优 w={best_w} 值 {best_s:.4f} "
                  f"随机最大 {rand_max} 子窗口众数 {mode_arg}({mode_count}/{len(argmaxes)})", flush=True)

        results[label] = {
            "window": [s_str, e_str],
            "config": "DEGRADED gold+bond only",
            "held": avail,
            "a_share_sharpe": round(sharpe(a_nav), 4),
            "sleeve_sharpe": round(sharpe(sleeve_nav), 4),
            "sleeve_cost": srec,
            "curves": curves,
            "results": recs,
            "random_w": {str(s): rnd_ws[i] for i, s in enumerate(SEEDS)},
        }
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump({"phase": "longw", "results": results}, fh, ensure_ascii=False, indent=2)

    # ── 跨窗口汇总：5 个窗口（main/oos 来自 P12-B 产物）──
    print(f"\n[p12] === P12-D 跨窗口汇总（最优 w）===", flush=True)
    table: Dict[str, Dict[str, Any]] = {}
    for mname in METRICS:
        row: Dict[str, Any] = {}
        for wname in ("main", "oos"):
            fp = os.path.join(OUT_DIR, f"multi_asset_p12_{wname}.json")
            if not os.path.exists(fp):
                continue
            with open(fp, encoding="utf-8") as fh:
                dd = json.load(fh)
            c = (dd.get("curves") or {}).get(mname)
            if c:
                row[wname] = {"best_w": c["best_w"], "best_value": c["best_value"],
                              "subwindow_mode": c.get("subwindow_argmax_mode"),
                              "subwindow_count": c.get("subwindow_mode_count")}
        for lname, entry in results.items():
            c = (entry.get("curves") or {}).get(mname)
            if c:
                row[lname] = {"best_w": c["best_w"], "best_value": c["best_value"],
                              "subwindow_mode": c.get("subwindow_argmax_mode"),
                              "subwindow_count": c.get("subwindow_mode_count")}
        if not row:
            continue
        ws = [r["best_w"] for r in row.values()]
        in_band = sum(1 for x in ws if 0.40 <= x <= 0.60)
        mode_w = max(set(ws), key=ws.count) if ws else None
        mode_n = ws.count(mode_w) if mode_w is not None else 0
        table[mname] = {
            "per_window": row,
            "n_windows": len(ws),
            "n_in_0.40_0.60": in_band,
            "mode_best_w": mode_w,
            "mode_count": mode_n,
            "spread": round(max(ws) - min(ws), 4) if ws else None,
        }
        print(f"[p12]   {mname}: {json.dumps({k: v['best_w'] for k, v in row.items()}, ensure_ascii=False)} "
              f"| 落 [0.40,0.60]: {in_band}/{len(ws)} | 众数 {mode_w}({mode_n}/{len(ws)}) "
              f"| 跨度 {table[mname]['spread']}", flush=True)

    # 判据 (f)：α 夏普口径下，最优 w 落 [0.40,0.60] 的窗口数 ≥ 4/5 且众数计数 ≥ 3
    crit_f = None
    gz = table.get("alpha_gz")
    if gz and gz["n_windows"] >= 5:
        crit_f = (gz["n_in_0.40_0.60"] >= 4) and (gz["mode_count"] >= 3)
    print(f"\n[p12] === 判据 (f) 跨窗口独立证据 ===", flush=True)
    print(f"[p12]   f_alpha_gz_best_w_consistent_5w = {crit_f}", flush=True)

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({"phase": "longw", "results": results, "cross_window": table,
                   "criteria": {"f_alpha_gz_best_w_consistent_5w": crit_f}},
                  fh, ensure_ascii=False, indent=2)
    print(f"[p12] ✓ 写出 {out_path}", flush=True)
    return {"results": results, "cross_window": table, "criteria": {"f": crit_f}}

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["ab", "long", "longw"], default="ab")
    ap.add_argument("--window", default="main", choices=["main", "oos"])
    ap.add_argument("--sleeve", default="full", choices=["full", "degraded"])
    args = ap.parse_args()
    if args.phase == "longw":
        run_longw()
        return 0
    if args.phase == "ab":
        run_ab(args.window, args.sleeve)
    else:
        run_long()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
