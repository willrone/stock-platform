#!/usr/bin/env python3
"""组合层波动率择时 A/B (stochastic + rsi 双腿)。

背景 (sharpe_overlay_experiment 结论):
  - rsi+vol25 夏普 0.31→0.66 是真的, 但阈值刀锋(30% 崩 -0.19) + 一半是空仓工程 → 不单独注册
  - 推荐下一步: 把 vol 开关从"单策略刀锋"升级为"组合风控"——
    stochastic(真实 alpha, 夏普 0.77-1.28) 主力 + rsi 副仓, 组合 vol 超阈值整体降仓。

引擎约束: 一个回测实例只能跑一个策略 → 用两个 50 万子组合(各半仓)模拟一个账户,
组合净值 = 两腿净值之和(各从 50 万起, 组合从 100 万起)。

变体 (全部 500 只宇宙 × 2023-05-12→2026-05-08, 口径同基线, 确定性引擎):
  V1  静态 50/50, 无择时                        → 组合基线
  V2  各腿自触发 vol25 (各腿自己的 20d vol)      → 组合
  V3  组合路径触发 vol {25%,30%}: 用 V1 组合基线路径预计算防御日集合,
      两腿按日期集合全进全出 (预承诺规则, d 日判定只用 <=d 数据, 无未来函数)
  C1  对照: stochastic 单腿 100万
  C2  对照: rsi 单腿 100万

产物: data/strategy_baseline/portfolio_vol_experiment.json
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import sys
import time
from collections import deque
from datetime import datetime

import numpy as np
import pandas as pd

BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BACKEND_ROOT)

from app.core.config import settings  # noqa: E402
from app.services.backtest import BacktestExecutor  # noqa: E402
from app.services.backtest.models import (  # noqa: E402
    BacktestConfig,
    SignalType,
    TradingSignal,
)
from app.services.backtest.core import portfolio_manager_array as pma_module  # noqa: E402

DATA_ROOT = "/Users/rone/projects/stock-platform/data"
OUT_JSON = os.path.join(DATA_ROOT, "strategy_baseline", "portfolio_vol_experiment.json")

START, END = datetime(2023, 5, 12), datetime(2026, 5, 8)
VOL_WINDOW = 20
LEG_CASH = 500_000.0

LEG_STOCH = "stochastic"
LEG_RSI = "rsi"


class ComboOverlay:
    """双腿共用一个防御状态机。

    mode:
      None   → 只 stash 实例引用(拿 equity_curve)
      "vol"  → 各腿自触发: 该腿自己的 20d 已实现 vol > trigger 防御, < 0.7x 恢复
      dates  → 预计算防御日集合(d 日 in 集合 → 清仓+禁买)
    """

    def __init__(self, mode=None, trigger=0.25, defense_dates=None, trigger30=None):
        self.mode = mode
        self.trigger = trigger
        self.defense_dates = set(defense_dates or ())
        self.in_defense = False
        self.active = mode is not None
        self.last_instance = None
        self.n_defense_days = 0
        self.leg_vol: float | None = None  # 该腿最近一次已实现 vol(调试)

    @property
    def eff_trigger(self):
        return self.trigger


_orig_snapshot = pma_module.PortfolioManagerArray.record_portfolio_snapshot
_orig_buy = pma_module.PortfolioManagerArray._execute_buy


def _realized_vol(equity, window=VOL_WINDOW) -> float | None:
    if len(equity) <= window:
        return None
    vals = np.array([v for _, v in equity[-(window + 1):]], dtype=float)
    base = np.where(vals[:-1] > 0, vals[:-1], 1.0)
    rr = np.diff(vals) / base
    v = float(rr.std() * math.sqrt(252))
    return v if v > 1e-9 else None


def install(o: ComboOverlay) -> None:
    if not o.active:

        def _stash(self, date, current_prices):
            o.last_instance = self
            _orig_snapshot(self, date, current_prices)

        pma_module.PortfolioManagerArray.record_portfolio_snapshot = _stash
        pma_module.PortfolioManagerArray._execute_buy = _orig_buy
        return

    def patched_buy(self, idx, stock_code, price, original_price, slip, signal, current_prices):
        if o.in_defense:
            return None, "组合防御态: 风险过高"
        return _orig_buy(self, idx, stock_code, price, original_price, slip, signal, current_prices)

    def patched_snapshot(self, date, current_prices):
        # 防御状态机
        d = date.date() if hasattr(date, "date") else date
        if o.mode == "dates":
            if d in o.defense_dates and not o.in_defense:
                o.in_defense = True
            elif d not in o.defense_dates and o.in_defense:
                o.in_defense = False
        elif o.mode == "vol":
            rv = _realized_vol(self.equity_curve)
            o.leg_vol = rv
            if rv is not None:
                if rv > o.trigger and not o.in_defense:
                    o.in_defense = True
                elif rv < o.trigger * 0.7 and o.in_defense:
                    o.in_defense = False
        # 防御态 → 清仓(全进全出)
        if o.in_defense:
            o.n_defense_days += 1
            for i in np.nonzero(self.quantities > 0)[0]:
                qi = int(i)
                code = self.stock_codes[qi]
                price = current_prices.get(code, self.avg_costs[qi])
                if not price or price <= 0:
                    continue
                self.execute_signal(
                    TradingSignal(
                        timestamp=date,
                        stock_code=code,
                        signal_type=SignalType.SELL,
                        strength=1.0,
                        price=float(price),
                        reason="combo_defense",
                    ),
                    current_prices,
                )
        o.last_instance = self
        _orig_snapshot(self, date, current_prices)

    pma_module.PortfolioManagerArray._execute_buy = patched_buy
    pma_module.PortfolioManagerArray.record_portfolio_snapshot = patched_snapshot


def _ret_stats(equity):
    if not equity or len(equity) < 30:
        return None
    vals = np.array([v for _, v in equity], dtype=float)
    base = np.where(vals[:-1] > 0, vals[:-1], 1.0)
    rets = np.diff(vals) / base
    years = max(1.0, len(equity) / 252.0)
    total = vals[-1] / vals[0] - 1.0
    ann = (1 + total) ** (1 / years) - 1 if total > -1 else -1.0
    vol = float(rets.std() * math.sqrt(252))
    sharpe = float(rets.mean() / rets.std() * math.sqrt(252)) if rets.std() > 0 else 0.0
    peak = np.maximum.accumulate(vals)
    mdd = float(((vals - peak) / peak).min())
    return {
        "total_return": round(total, 6),
        "annualized_return": round(ann, 6),
        "volatility": round(vol, 6),
        "sharpe_ratio": round(sharpe, 4),
        "max_drawdown": round(mdd, 6),
    }


def _yearly(equity):
    by = {}
    for d, v in equity:
        dd = d.date() if hasattr(d, "date") else d
        by.setdefault(dd.year, []).append(float(v))
    out, prev = {}, None
    for y in sorted(by):
        vals = np.array(by[y])
        if len(vals) < 20:
            continue
        r0 = prev if prev else vals[0]
        out[str(y)] = round(float(vals[-1] / r0 - 1.0), 4)
        prev = float(vals[-1])
    return out


def _combine(equities: list) -> list:
    """两腿净值逐日相加(日期对齐, 缺失腿沿用上一值)。输入各从 LEG_CASH 起。"""
    maps = []
    for eq in equities:
        m = {}
        for d, v in eq:
            dd = d.date() if hasattr(d, "date") else d
            m[dd] = float(v)
        maps.append(m)
    dates = sorted(set().union(*[set(m) for m in maps]))
    out = []
    last = [0.0] * len(maps)
    seen = [False] * len(maps)
    for d in dates:
        for i, m in enumerate(maps):
            if d in m:
                last[i] = m[d]
                seen[i] = True
        if not all(seen):
            continue
        out.append((pd.Timestamp(d), float(sum(last))))
    return out


def load_universe() -> list:
    uni_file = os.path.join(DATA_ROOT, "strategy_baseline", "baseline_universe_500.txt")
    codes = []
    with open(uni_file) as f:
        for line in f:
            c = line.strip()
            if not c or c.startswith("#") or "." not in c:
                continue
            if c not in codes:
                codes.append(c)
    return codes


def make_executor() -> BacktestExecutor:
    return BacktestExecutor(
        data_dir=str(settings.DATA_ROOT_PATH),
        enable_parallel=True,
        max_workers=8,
        enable_performance_profiling=False,
        use_multiprocessing=False,
    )


def leg_config() -> BacktestConfig:
    return BacktestConfig(
        initial_cash=LEG_CASH,
        commission_rate=0.0003,
        slippage_rate=0.0001,
        max_position_size=0.2,
        cash_reserve_ratio=0.05,
        board_lot_size=100,
        stop_loss_pct=0.05,
        take_profit_pct=0.15,
        rebalance_frequency="daily",
        open_cost=0.0,
        close_cost=0.0,
        min_cost=0.0,
    )


def run_leg(ex, strat: str, codes: list, cfg, o: ComboOverlay):
    install(o)
    try:
        res = asyncio.run(
            ex.run_backtest(
                strategy_name=strat,
                stock_codes=codes,
                start_date=START,
                end_date=END,
                strategy_config={},
                backtest_config=cfg,
            )
        )
        eq = getattr(o.last_instance, "equity_curve", None)
        return res, eq
    finally:
        install(ComboOverlay())


def _defense_dates_from_combo(combo_eq, trigger: float):
    """从组合基线路径预计算防御日集合(20d vol > trigger 离场, < 0.7x 恢复)。"""
    vals = [(d, float(v)) for d, v in combo_eq]
    dates = [d for d, _ in vals]
    vs = np.array([v for _, v in vals])
    defense = set()
    in_def = False
    for i in range(len(vs)):
        if i > VOL_WINDOW:
            window = vs[i - VOL_WINDOW : i + 1]
            base = np.where(window[:-1] > 0, window[:-1], 1.0)
            rv = float((np.diff(window) / base).std() * math.sqrt(252))
            if rv > trigger and not in_def:
                in_def = True
            elif rv < trigger * 0.7 and in_def:
                in_def = False
        if in_def:
            dd = dates[i].date() if hasattr(dates[i], "date") else dates[i]
            defense.add(dd)
    return defense, len(vs)


def main() -> None:
    codes = load_universe()
    print(f"[universe] {len(codes)} 只, 腿资金 {LEG_CASH:,.0f}", flush=True)
    ex = make_executor()
    cfg = leg_config()

    results: dict = {}

    def save():
        payload = {
            "meta": {
                "generated_at": datetime.now().isoformat(),
                "window": [str(START), str(END)],
                "universe_size": len(codes),
                "leg_cash": LEG_CASH,
                "note": "组合层vol择时: stochastic+rsi 50/50; V1静态 V2自触发V3组合路径触发",
            },
            "results": results,
        }
        os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
        with open(OUT_JSON, "w") as f:
            json.dump(payload, f, ensure_ascii=False, indent=1, default=str)

    # ---- Phase 1: V1 静态 50/50 (两腿无 overlay) ----
    eqs = {}
    for strat in (LEG_STOCH, LEG_RSI):
        o = ComboOverlay()
        t0 = time.perf_counter()
        res, eq = run_leg(ex, strat, codes, cfg, o)
        eqs[strat] = eq
        print(f"[V1] {strat} 腿: n={len(eq) if eq else 0} wall={time.perf_counter()-t0:.0f}s", flush=True)

    combo_v1 = _combine([eqs[LEG_STOCH], eqs[LEG_RSI]])
    rs_v1 = _ret_stats(combo_v1)
    results["V1_静态5050"] = {
        "combo": rs_v1,
        "legs": {s: _ret_stats(e) for s, e in eqs.items()},
        "yearly_combo": _yearly(combo_v1),
    }
    if rs_v1:
        print(f"[V1] 组合: sharpe={rs_v1['sharpe_ratio']} ann={rs_v1['annualized_return']} mdd={rs_v1['max_drawdown']} tr={rs_v1['total_return']}", flush=True)
    else:
        print("[V1] 组合净值不足, 无法计算", flush=True)
    save()

    # ---- Phase 2: V2 各腿自触发 vol25 ----
    eqs2 = {}
    for strat in (LEG_STOCH, LEG_RSI):
        o = ComboOverlay(mode="vol", trigger=0.25)
        res, eq = run_leg(ex, strat, codes, cfg, o)
        eqs2[strat] = eq
        print(f"[V2] {strat} 腿: defense={o.n_defense_days}d lastvol={o.leg_vol}", flush=True)
    combo_v2 = _combine([eqs2[LEG_STOCH], eqs2[LEG_RSI]])
    rs_v2 = _ret_stats(combo_v2)
    results["V2_自触发vol25"] = {
        "combo": rs_v2,
        "legs": {s: _ret_stats(e) for s, e in eqs2.items()},
        "yearly_combo": _yearly(combo_v2),
    }
    if rs_v2:
        print(f"[V2] 组合: sharpe={rs_v2['sharpe_ratio']} ann={rs_v2['annualized_return']} mdd={rs_v2['max_drawdown']}", flush=True)
    save()

    # ---- Phase 3: V3 组合路径触发 (25% / 30%) ----
    for trig in (0.25, 0.30):
        tag = f"V3_组合vol{int(trig*100)}"
        defense, n = _defense_dates_from_combo(combo_v1, trig)
        print(f"[{tag}] 防御日 {len(defense)}/{n}", flush=True)
        eqs3 = {}
        for strat in (LEG_STOCH, LEG_RSI):
            o = ComboOverlay(mode="dates", defense_dates=defense)
            res, eq = run_leg(ex, strat, codes, cfg, o)
            eqs3[strat] = eq
        combo_v3 = _combine([eqs3[LEG_STOCH], eqs3[LEG_RSI]])
        rs_v3 = _ret_stats(combo_v3)
        results[tag] = {
            "trigger": trig,
            "defense_days": len(defense),
            "combo": rs_v3,
            "legs": {s: _ret_stats(e) for s, e in eqs3.items()},
            "yearly_combo": _yearly(combo_v3),
        }
        if rs_v3:
            print(f"[{tag}] 组合: sharpe={rs_v3['sharpe_ratio']} ann={rs_v3['annualized_return']} mdd={rs_v3['max_drawdown']} tr={rs_v3['total_return']}", flush=True)
        save()

    # ---- Phase 4: 对照 单腿 100万 ----
    for strat in (LEG_STOCH, LEG_RSI):
        tag = f"C_{strat}单腿"
        o = ComboOverlay()
        res, eq = run_leg(ex, strat, codes, BacktestConfig(
            initial_cash=1_000_000.0, commission_rate=0.0003, slippage_rate=0.0001,
            max_position_size=0.2, cash_reserve_ratio=0.05, board_lot_size=100,
            stop_loss_pct=0.05, take_profit_pct=0.15, rebalance_frequency="daily",
            open_cost=0.0, close_cost=0.0, min_cost=0.0,
        ), o)
        rs = _ret_stats(eq)
        results[tag] = {"single": rs}
        if rs:
            print(f"[{tag}] sharpe={rs['sharpe_ratio']} ann={rs['annualized_return']} mdd={rs['max_drawdown']}", flush=True)
        save()

    print(f"[saved] {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
