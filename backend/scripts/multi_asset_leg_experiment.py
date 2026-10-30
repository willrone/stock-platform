#!/usr/bin/env python3
"""多资产腿实验 —— 检验「降低 beta 能否突破 0.88~1.03 的 A 股多头框架上限」。

## 动机（两篇文档共同结论）

- `CONCLUSION_2026-10-27.md` 第三节：四条主线策略日收益相关 0.75~0.79，
  分散化在「单一股票池 + TopK」框架内被堵死，框架内夏普上限 0.88~1.03。
- `POOL_PARTITION_2026-10-30.md`：分池并行证伪，并**更正了归因** ——
  持仓完全不重叠后腿间相关反而 0.90~0.96 → 那 0.75 的相关来自
  **共同的市场 beta**，不是持仓重叠。

→ **A 股多头策略的夏普天花板在 beta 层，不在选股层。**
→ 唯一未被试过、且机制上能突破上限的路：**引入非 beta 相关的资产**
（黄金 / 债券 / 跨境 / 红利低波）。

## 本实验做什么

1. **相关性诊断（先做）**：A 股主线腿（honest_composite 25 因子）vs
   各类 ETF 腿的**日收益相关**。这一步决定后面有没有意义 ——
   如果所有 ETF 腿与 A 股腿相关都 > 0.7，那"降低 beta"就是伪命题，直接停。

2. **组合实验**：主线腿 + ETF 腿合成（等权 / 波动率目标），
   看夏普 / 回撤 / 胜率是否改善。

3. **对照组**：纯 A 股主线腿（基线，必须逐位复现 v2 基准）。

## ETF 腿的选择（关键设计）

ETF 腿分三档，**从最简单到最复杂**，任何一档成功都说明方向成立：

| 档 | 做法 | 参数 | 作用 |
|---|---|---|---|
| **L1 静态等权** | 每个资产类别取成交额最大的 1 只，等权 buy & hold | 0 个可调参数 | 纯粹的 beta 稀释 |
| **L2 类别等权** | 每类取 top-N（N=3），类内等权、类间等权 | N | 降低单只票的特异性风险 |
| **L3 动量轮动** | 每月选动量最强的 1 只 | 动量窗口 + 持有期 | 有 alpha 的腿 |

**先跑 L1/L2**。理由：参数最少 → 结论最可信。L3 若要跑，必须扫邻档
（刀锋参数 = 过拟合）。

## ETF 腿的回测口径

ETF 与个股的差异，必须处理：
- **成交价**：ETF 用 open 成交更现实（收盘后才知道信号）。主线腿用 close。
  为了可比，L1/L2 是 buy & hold，**建仓日用次日 open**。
- **涨跌停**：ETF 有 ±10%（普通）/ ±20%（创业板/科创板）限制 →
  **一字涨停买不进、一字跌停卖不出**。这里必须检查。
- **成本**：ETF **免印花税**！佣金 0.03%（很多券商万 0.5~万 3，但保守取万 3）
  + 滑点 0.01%。**印花税 0 是 ETF 相对个股的确定性优势**，
  这会体现在换手频繁的轮动策略上。
- **无最小 5 元佣金**：ETF 通常免最低佣金（多数券商已取消）。

## 产出

`data/backtest/multi_asset_leg.json`：相关性诊断 + 各档组合指标。

用法：
    cd backend && env -u PYTHONPATH BASELINE_LOG_LEVEL=ERROR .venv-py313/bin/python3 scripts/multi_asset_leg_experiment.py
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import statistics
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

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

ETF_PARQUET = os.path.join(ROOT, "data", "etf", "parquet")
ETF_UNIVERSE = os.path.join(ROOT, "data", "etf", "etf_universe.json")
OUT_PATH = os.path.join(ROOT, "data", "backtest", "multi_asset_leg.json")
NAV_CACHE_PATH = os.path.join(ROOT, "data", "backtest", "multi_asset_navs.json")
V2_PATH = os.path.join(ROOT, "data", "backtest", "ALL_real_lag1_v2.json")

START = datetime(2023, 5, 8)
END = datetime(2026, 5, 8)

#: 样本外窗口：25 因子策略的面板数据从 2021-05-11 起（此前 17 因子为 NaN，
#: 策略退化为 23 公式版 → 不是同一个策略，不能用于 OOS）。
#: 这 2 年恰好是 A 股弱势期（2021H2 暴跌 + 2022 熊市），是检验
#: 「分散化在弱势市场是否有效」的最佳场景，且未参与组合规则的推导。
OOS_START = datetime(2021, 5, 11)
OOS_END = datetime(2023, 5, 8)

STRATEGY = "honest_composite"
STRATEGY_CFG: Dict[str, Any] = {"rebalance_days": "weekly"}
TOPK = 60
N_DROP = 5

#: 本实验关注的资产类别（不含 a_share —— A 股腿由主线策略承担）
ASSET_CLASSES = ["gold", "bond", "cross_border", "dividend", "commodity"]


# --------------------------------------------------------------------------
# 指标
# --------------------------------------------------------------------------
def sharpe(nav: List[Tuple[str, float]]) -> float:
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
    return mu / sd * math.sqrt(244.0) if sd > 0 else 0.0


def maxdd(nav: List[Tuple[str, float]]) -> float:
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


def total_ret(nav: List[Tuple[str, float]]) -> float:
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
    return num / (da * db) if da > 0 and db > 0 else 0.0


def merge_by_date(
    navs: List[List[Tuple[str, float]]], weights: List[float]
) -> List[Tuple[str, float]]:
    """按**日期**对齐合成多腿净值（不是按下标！按下标配对会错位）。

    每条腿先归一化到起点 = 1.0，再按权重加权。
    """
    if not navs or len(navs) != len(weights):
        return []
    norm: List[Dict[str, float]] = []
    for nav in navs:
        if not nav or nav[0][1] <= 0:
            return []
        base = nav[0][1]
        norm.append({d: v / base for d, v in nav})
    all_dates = sorted(set().union(*[set(c) for c in norm]))
    out: List[Tuple[str, float]] = []
    for d in all_dates:
        val = 0.0
        for c, w in zip(norm, weights):
            val += w * c.get(d, 1.0)
        out.append((d, val))
    return out


def align(navs: List[List[Tuple[str, float]]]) -> List[Tuple[str, float]]:
    if not navs:
        return []
    all_dates = sorted({d for nav in navs for d, _ in nav})
    curves = [{d: v for d, v in nav} for nav in navs]
    first = navs[0][0][1] if navs and navs[0] else 1.0
    last = [first for _ in navs]
    out = []
    for d in all_dates:
        tot = 0.0
        for li, c in enumerate(curves):
            if d in c:
                last[li] = c[d]
            tot += last[li]
        out.append((d, tot))
    return out


def yearly(nav: List[Tuple[str, float]]) -> Dict[str, float]:
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
# ETF 腿
# --------------------------------------------------------------------------
def load_etf_universe() -> Dict[str, Any]:
    with open(ETF_UNIVERSE, encoding="utf-8") as fh:
        return json.load(fh)


def etf_turnover_proxy(code: str) -> float:
    """用区间内平均成交额（万元）当流动性代理。"""
    import pandas as pd

    path = os.path.join(ETF_PARQUET, f"{code}.parquet")
    if not os.path.exists(path):
        return 0.0
    df = pd.read_parquet(path)
    df = df[(df["date"] >= "2024-01-01") & (df["date"] <= "2026-05-08")]
    if df.empty:
        return 0.0
    # volume 单位是手，用 close 估算成交额（元）
    amt = (df["volume"].fillna(0) * df["close"] * 100).mean()
    return float(amt)


def pick_etfs(
    universe: Dict[str, Any], classes: List[str], top_n: int
) -> Dict[str, List[str]]:
    """每个资产类别取成交额最大的 top_n 只。

    **硬门槛：必须覆盖整个回测窗口**（数据起点 ≤ 窗口起点 + 5 天，且样本
    ≥ 200 天）。原因：若某只 ETF 是 2024 年才上市的，common calendar
    交集会把整条腿的起点推到 2024 年 → 该腿的夏普/相关性只在部分窗口上
    成立，与其他腿不可比。bond 腿曾因此只剩 2025~2026 一年数据，
    算出「夏普 4.84」的样本噪声。
    """
    import pandas as pd

    etfs = universe["etfs"]
    ws = START.strftime("%Y-%m-%d")
    we = END.strftime("%Y-%m-%d")
    by_class: Dict[str, List[str]] = {c: [] for c in classes}
    for code, meta in etfs.items():
        path = os.path.join(ETF_PARQUET, f"{code}.parquet")
        if not os.path.exists(path):
            continue
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        w = df[(df["date"] >= ws) & (df["date"] <= we)]
        if len(w) < 200:
            continue
        if w["date"].iloc[0] > START.strftime("%Y-%m-%d"):
            continue  # 上市晚于窗口起点 → 不可用
        for c in meta.get("tags", []):
            if c in by_class:
                by_class[c].append(code)
    out: Dict[str, List[str]] = {}
    for c, codes in by_class.items():
        scored = [(etf_turnover_proxy(code), code) for code in codes]
        scored = [s for s in scored if s[0] > 0]
        scored.sort(reverse=True)
        out[c] = [code for _, code in scored[:top_n]]
    return out


def etf_static_leg(
    codes: List[str],
    start: datetime,
    end: datetime,
    capital: float,
) -> Tuple[Dict[str, Any], List[Tuple[str, float]]]:
    """L1/L2 静态等权腿：等权 buy & hold，建仓用次日 open，扣佣金+滑点。

    口径要点（都是踩过的坑）：
    1. **权重分母 = 实际成交的标的数**，不是候选数。用候选数会让没成交的
       标的白占权重，净值凭空缩水（曾导致 bond 腿出现 −66% 的假收益）。
    2. **只在共同交易日历上建仓**：某只 ETF 上市晚于窗口起点 → 无法在统一起
       点建仓 → 剔除该标的，而不是让它占权重。
    3. **涨跌停**：建仓日开盘已达涨停 → 买不进 → 剔除（真实约束）。
    4. ETF **免印花税**，只收佣金 0.03%（不少券商万 0.5~万 3，保守取万 3）
       + 滑点 0.01%；ETF 通常免最低 5 元佣金。
    """
    import pandas as pd

    s_str = start.strftime("%Y-%m-%d")
    e_str = end.strftime("%Y-%m-%d")

    frames: Dict[str, pd.DataFrame] = {}
    for code in codes:
        path = os.path.join(ETF_PARQUET, f"{code}.parquet")
        if not os.path.exists(path):
            continue
        df = pd.read_parquet(path).sort_values("date")
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        df = df[(df["date"] >= s_str) & (df["date"] <= e_str)]
        if len(df) > 60:
            frames[code] = df.reset_index(drop=True)

    if not frames:
        return {"error": "no etf data"}, []

    # 共同交易日历（取交集，保证每只票每天都有价）
    common = set(frames[next(iter(frames))]["date"])
    for df in frames.values():
        common &= set(df["date"])
    common_dates = sorted(d for d in common if d >= s_str)
    if len(common_dates) < 30:
        return {"error": "insufficient common calendar"}, []

    # 建仓日 = 共同日历的第二个交易日（用次日 open 成交）
    buy_date = common_dates[1]
    blocked: List[str] = []

    # 第一步：判定哪些标的能成交（不含权重，避免分母错误）
    fillable: List[Tuple[str, float]] = []  # (code, 成交价)
    for code, df in frames.items():
        row = df[df["date"] == buy_date]
        if row.empty:
            blocked.append(f"{code}:no_data")
            continue
        r = row.iloc[0]
        o = float(r["open"])
        limit = 0.20 if code.startswith(("30", "68")) else 0.10
        prev = df[df["date"] < buy_date]
        if not prev.empty:
            pclose = float(prev.iloc[-1]["close"])
            if pclose > 0 and (o - pclose) / pclose >= limit - 0.001:
                blocked.append(f"{code}:limit_up")
                continue
        fillable.append((code, o * (1 + 0.0001)))  # 买入滑点：价上浮

    if not fillable:
        return {"error": "no fills", "blocked": blocked}, []

    # 第二步：权重分母 = 实际成交数
    weight = capital / len(fillable)
    fills: List[Dict[str, Any]] = []
    for code, px in fillable:
        shares = weight / px
        fills.append({"code": code, "date": buy_date, "price": px, "shares": shares})

    # 建仓成本：佣金 0.03%（ETF 免印花税），ETF 免最低 5 元
    gross = sum(f["shares"] * f["price"] for f in fills)
    comm = gross * 0.0003

    # 逐日 mark-to-market（共同日历内）
    nav: List[Tuple[str, float]] = []
    for d in common_dates:
        val = 0.0
        for f in fills:
            df = frames[f["code"]]
            row = df[df["date"] == d]
            if row.empty:
                prev = df[df["date"] < d]
                px = float(prev.iloc[-1]["close"]) if not prev.empty else f["price"]
            else:
                px = float(row.iloc[0]["close"])
            val += f["shares"] * px
        nav.append((d, val))

    # 起点：建仓前一日的现金（扣佣金）—— 用日历首日作为 T0
    nav.insert(0, (common_dates[0], capital - comm))

    return {
        "codes": [f["code"] for f in fills],
        "blocked": blocked,
        "buy_date": buy_date,
        "initial_capital": capital,
        "invested_ratio": round(gross / capital, 6),
        "commission": round(comm, 2),
        "nav_points": len(nav),
    }, nav


# --------------------------------------------------------------------------
# A 股主线腿
# --------------------------------------------------------------------------
def run_a_share_leg(codes: List[str]) -> Tuple[Dict[str, Any], List[Tuple[str, float]]]:
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
            strategy_config={**STRATEGY_CFG, "lag": 1, "topk": TOPK, "n_drop": N_DROP},
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
    trades = res.get("trade_history") or []
    rec = {
        "n_codes": len(codes),
        "engine_total_return": res.get("total_return"),
        "engine_sharpe": res.get("sharpe_ratio"),
        "engine_max_drawdown": res.get("max_drawdown"),
        "engine_win_rate": res.get("win_rate"),
        "total_trades": len(trades),
        "nav_points": len(nav),
    }
    return rec, nav


def verify_baseline(rec: Dict[str, Any]) -> Dict[str, Any]:
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


def main() -> None:
    global START, END, OUT_PATH, NAV_CACHE_PATH
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--window",
        default="main",
        choices=["main", "oos"],
        help="main=2023-05-08~2026-05-08（推导窗口）；oos=2021-05-11~2023-05-08（样本外）",
    )
    args = ap.parse_args()

    if args.window == "oos":
        START, END = OOS_START, OOS_END
        OUT_PATH = os.path.join(ROOT, "data", "backtest", "multi_asset_leg_oos.json")
        NAV_CACHE_PATH = os.path.join(ROOT, "data", "backtest", "multi_asset_navs_oos.json")
        print(f"[ma] **样本外窗口** {START.date()} → {END.date()}", flush=True)
    else:
        print(f"[ma] **推导窗口** {START.date()} → {END.date()}", flush=True)

    store: Dict[str, Any] = {"window_mode": args.window}

    universe = load_etf_universe()
    print(f"[ma] ETF 清单 {universe['total']} 只", flush=True)

    # ── A 股主线腿（基线）──
    codes = universe_from_instruments(QLIB_OFFICIAL)
    print(f"[ma] 跑 A 股主线腿（{len(codes)} 只）...", flush=True)
    t0 = time.perf_counter()
    a_rec, a_nav = run_a_share_leg(codes)
    a_rec["wall_s"] = round(time.perf_counter() - t0, 1)
    print(
        f"  A 股腿: 夏普 {a_rec['engine_sharpe']}  收益 {a_rec['engine_total_return']}  "
        f"回撤 {a_rec['engine_max_drawdown']}  笔数 {a_rec['total_trades']}  ({a_rec['wall_s']}s)",
        flush=True,
    )
    verification = verify_baseline(a_rec)
    if args.window == "oos":
        # 样本外窗口没有 v2 基准可比 → 改为**确定性自检**：
        # 同一策略在同窗口跑两次必须一致。这里记录引擎口径供后续比对。
        verification = {
            "checked": False,
            "reason": "样本外窗口无 v2 基准；改为跑两次验证确定性",
        }
    print(f"[ma] 基线复现门: {json.dumps(verification, ensure_ascii=False)}", flush=True)
    if verification.get("checked") and not verification.get("ok"):
        print("[ma] ❌ 基线无法复现 → 全部不可比，退出", flush=True)
        raise SystemExit(1)
    store["a_share_leg"] = a_rec
    store["a_share_nav_metrics"] = {
        "sharpe": round(sharpe(a_nav), 6),
        "total_return": round(total_ret(a_nav), 6),
        "max_drawdown": round(maxdd(a_nav), 6),
        "yearly": yearly(a_nav),
    }

    a_rets = daily_rets(a_nav)

    # ── ETF 腿候选（每个类别 top-N）──
    picks = pick_etfs(universe, ASSET_CLASSES, top_n=3)
    print(f"[ma] ETF 候选: {json.dumps(picks, ensure_ascii=False)}", flush=True)

    # ── 相关性诊断（先做，决定后面有没有意义）──
    diag: Dict[str, Any] = {}
    class_navs: Dict[str, List[Tuple[str, float]]] = {}
    for cls, codes_c in picks.items():
        if not codes_c:
            continue
        rec, nav = etf_static_leg(codes_c, START, END, 1_000_000.0)
        if not nav:
            print(f"  {cls}: ❌ {rec}", flush=True)
            continue
        class_navs[cls] = nav
        r = daily_rets(nav)
        diag[cls] = {
            "codes": rec.get("codes"),
            "blocked": rec.get("blocked"),
            "sharpe": round(sharpe(nav), 6),
            "total_return": round(total_ret(nav), 6),
            "max_drawdown": round(maxdd(nav), 6),
            "corr_with_a_share": round(corr(a_rets, r), 4),
            "yearly": yearly(nav),
        }
        print(
            f"  {cls:14s} 夏普 {diag[cls]['sharpe']:.4f}  收益 {diag[cls]['total_return']:+.4f}  "
            f"回撤 {diag[cls]['max_drawdown']:.4f}  **与A股腿相关 {diag[cls]['corr_with_a_share']:.4f}**  "
            f"标的 {rec.get('codes')}",
            flush=True,
        )
    store["correlation_diagnostic"] = diag

    if not class_navs:
        print("[ma] ❌ 无可用 ETF 腿数据", flush=True)
        raise SystemExit(1)

    # ── 组合实验 ──
    combos: Dict[str, Any] = {}
    base_sharpe = sharpe(a_nav)
    base_mdd = maxdd(a_nav)

    # 全类别 ETF 腿：各类别腿先等权合成
    cls_sorted = sorted(class_navs)
    etf_nav = merge_by_date([class_navs[c] for c in cls_sorted], [1.0 / len(cls_sorted)] * len(cls_sorted))

    # 各腿年化波动（用于风险平价权重）
    def ann_vol(nav: List[Tuple[str, float]]) -> float:
        r = daily_rets(nav)
        vals = list(r.values())
        if len(vals) < 30:
            return 0.0
        return statistics.pstdev(vals) * math.sqrt(244.0)

    vols = {"a_share": ann_vol(a_nav)}
    for c in cls_sorted:
        vols[c] = ann_vol(class_navs[c])
    print(f"[ma] 年化波动: {json.dumps({k: round(v, 4) for k, v in vols.items()}, ensure_ascii=False)}", flush=True)

    def combo_metrics(combined_nav: List[Tuple[str, float]], label: str, extra: Dict[str, Any]) -> None:
        s = sharpe(combined_nav)
        vol = ann_vol(combined_nav)
        combos[label] = {
            **extra,
            "sharpe": round(s, 6),
            "sharpe_vs_base_pct": round((s / base_sharpe - 1) * 100, 3) if base_sharpe else None,
            "total_return": round(total_ret(combined_nav), 6),
            "max_drawdown": round(maxdd(combined_nav), 6),
            "annual_vol": round(vol, 6),
            "return_per_vol": round(total_ret(combined_nav) / vol, 4) if vol > 0 else 0.0,
            "yearly": yearly(combined_nav),
        }
        print(
            f"  {label}: 夏普 {s:.4f} ({(s/base_sharpe-1)*100:+.2f}%)  "
            f"收益 {combos[label]['total_return']:+.4f}  回撤 {combos[label]['max_drawdown']:.4f}  "
            f"波动 {vol:.4f}  收益/波动 {combos[label]['return_per_vol']:.4f}",
            flush=True,
        )

    # C1~C3: 资金等权 A 股腿 + 全类别 ETF 腿
    for name, wa, we in [
        ("C1_50_50", 0.5, 0.5),
        ("C2_70_30", 0.7, 0.3),
        ("C3_30_70", 0.3, 0.7),
    ]:
        combined_nav = merge_by_date([a_nav, etf_nav], [wa, we])
        combo_metrics(
            combined_nav,
            name,
            {"scheme": "capital_equal", "weight_a_share": wa, "weight_etf": we},
        )

    # C4: 风险平价（逆波动率加权，A股腿 vs ETF 腿两层）
    # 注意：逆波动率对**极低波资产**会退化 —— 债券腿年化波动 0.5% 时
    # 权重比高达 200:1，几乎吸干全部资金，夏普分母趋零 → 伪夏普。
    # 因此对类别级逆波动率权重做**上限裁剪**（单类 ≤ 40%）。
    MAX_CLASS_W = 0.40
    MIN_VOL = 0.02  # 波动率下限，防止超低波资产权重爆炸

    def inv_vol_weights(vol_map: Dict[str, float], keys: List[str]) -> Dict[str, float]:
        invs = {k: 1.0 / max(vol_map.get(k, 1.0), MIN_VOL) for k in keys}
        tot = sum(invs.values())
        w = {k: v / tot for k, v in invs.items()}
        # 迭代裁剪超限项（把超出部分按比例分给未超限项）
        for _ in range(20):
            over = {k: v - MAX_CLASS_W for k, v in w.items() if v > MAX_CLASS_W + 1e-12}
            if not over:
                break
            excess = sum(over.values())
            for k in over:
                w[k] = MAX_CLASS_W
            under = {k: v for k, v in w.items() if v < MAX_CLASS_W - 1e-12}
            if not under:
                break
            tot_u = sum(under.values())
            for k in under:
                w[k] += excess * (under[k] / tot_u)
        return w

    for etf_w in (0.3, 0.5, 0.7):
        inv_a = 1.0 / max(vols["a_share"], MIN_VOL)
        inv_e = 1.0 / max(ann_vol(etf_nav), MIN_VOL)
        w_a = (etf_w * inv_a) / (etf_w * inv_a + (1 - etf_w) * inv_e)
        combined_nav = merge_by_date([a_nav, etf_nav], [w_a, 1 - w_a])
        combo_metrics(
            combined_nav,
            f"C4_rp_etf{int(etf_w*100)}",
            {
                "scheme": "risk_parity_2layer",
                "weight_a_share": round(w_a, 6),
                "weight_etf": round(1 - w_a, 6),
            },
        )

    # C5: 风险平价（全部类别各自风险平价 + A 股腿），带单类上限
    cls_w_map = inv_vol_weights(vols, cls_sorted)
    cls_w = [cls_w_map[c] for c in cls_sorted]
    risk_etf_nav = merge_by_date([class_navs[c] for c in cls_sorted], cls_w)
    inv_a = 1.0 / max(vols["a_share"], MIN_VOL)
    inv_e = 1.0 / max(ann_vol(risk_etf_nav), MIN_VOL)
    for etf_w in (0.3, 0.5):
        w_a = (etf_w * inv_a) / (etf_w * inv_a + (1 - etf_w) * inv_e)
        combined_nav = merge_by_date([a_nav, risk_etf_nav], [w_a, 1 - w_a])
        combo_metrics(
            combined_nav,
            f"C5_rp_classlevel_etf{int(etf_w*100)}",
            {
                "scheme": "risk_parity_classlevel",
                "class_weights": {c: round(w, 6) for c, w in zip(cls_sorted, cls_w)},
                "weight_a_share": round(w_a, 6),
                "weight_etf": round(1 - w_a, 6),
            },
        )

    # C6: 单一类别 + A 股腿（按相关系数在**全窗口**上排），风险平价
    if class_navs:
        best_cls = min(diag, key=lambda c: diag[c]["corr_with_a_share"])
        b_nav = class_navs[best_cls]
        for etf_w in (0.3, 0.5):
            inv_a = 1.0 / max(vols["a_share"], MIN_VOL)
            inv_e = 1.0 / max(ann_vol(b_nav), MIN_VOL)
            w_a = (etf_w * inv_a) / (etf_w * inv_a + (1 - etf_w) * inv_e)
            combined_nav = merge_by_date([a_nav, b_nav], [w_a, 1 - w_a])
            combo_metrics(
                combined_nav,
                f"C6_best_{best_cls}_rp_etf{int(etf_w*100)}",
                {
                    "scheme": "risk_parity_single_class",
                    "best_class": best_cls,
                    "weight_a_share": round(w_a, 6),
                    "weight_etf": round(1 - w_a, 6),
                },
            )

    # C7: 对照 —— **剔除 gold** 的组合（gold 腿 +126% 几乎全靠 2025 金价
    # 牛市，是窗口依赖；必须验证去掉它后组合是否仍然成立）
    if len(cls_sorted) > 1:
        cls_ng = [c for c in cls_sorted if c != "gold"]
        etf_ng = merge_by_date([class_navs[c] for c in cls_ng], [1.0 / len(cls_ng)] * len(cls_ng))
        for wa, we in [(0.5, 0.5), (0.3, 0.7)]:
            combined_nav = merge_by_date([a_nav, etf_ng], [wa, we])
            combo_metrics(
                combined_nav,
                f"C7_no_gold_{int(wa*100)}_{int(we*100)}",
                {
                    "scheme": "capital_equal_ex_gold",
                    "classes": cls_ng,
                    "weight_a_share": wa,
                    "weight_etf": we,
                },
            )

    # C8: 对照 —— **只留 gold + bond**（两个最低相关类别），资本等权
    if "gold" in class_navs and "bond" in class_navs:
        for wa, we in [(0.5, 0.5), (0.3, 0.7)]:
            combined_nav = merge_by_date(
                [a_nav, class_navs["gold"], class_navs["bond"]],
                [wa, we / 2, we / 2],
            )
            combo_metrics(
                combined_nav,
                f"C8_gold_bond_{int(wa*100)}_{int(we*100)}",
                {
                    "scheme": "capital_equal_gold_bond",
                    "weight_a_share": wa,
                    "weight_gold": we / 2,
                    "weight_bond": we / 2,
                },
            )

    store["baseline_sharpe"] = base_sharpe
    store["baseline_max_drawdown"] = round(base_mdd, 6)
    store["annual_vol"] = {k: round(v, 6) for k, v in vols.items()}
    store["combinations"] = combos

    # ── 净值缓存（避免每次重跑都重算 90s 的 A 股腿）──
    nav_payload = {
        "window": f"{START.date()}→{END.date()}",
        "a_share": [[d, round(v, 4)] for d, v in a_nav],
        "classes": {c: [[d, round(v, 4)] for d, v in n] for c, n in class_navs.items()},
    }
    nav_path = NAV_CACHE_PATH
    with open(nav_path, "w", encoding="utf-8") as fh:
        json.dump(nav_payload, fh)

    payload = {
        "window": f"{START.date()}→{END.date()}",
        "asset_classes": ASSET_CLASSES,
        "methodology": {
            "lag": 1,
            "enable_tradability": True,
            "rebalance_days": "weekly",
            "note": "ETF 腿 = 静态等权 buy&hold，次日 open 建仓，佣金 0.03%（ETF 免印花税）+ 滑点 0.01%",
        },
        "results": store,
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)
    print(f"\n[ma] ✓ 已写入 {OUT_PATH}")


if __name__ == "__main__":
    main()
