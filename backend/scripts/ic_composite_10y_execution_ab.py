#!/usr/bin/env python3
"""P0 执行层 A/B: lag0 vs lag1, 日频 vs 周频, 回测成本 vs 实盘成本.

背景: 42f topk30/ndrop3 的 0.8865 是 lag0 + 回测成本 (佣金 0.03% + 滑点 0.01%,
无印花税)。实盘收盘后才知道信号, 只能次日执行 = lag1; 卖出还有 0.05% 印花税,
小盘滑点 0.1% 更现实。本实验回答两个问题:
  1. lag1 会损失多少?
  2. 周频调仓 (换手降 ~5 倍) 能不能把成本省回来, 净收益是否更高?

四个单元 (同 42f / PIT 4329 / topk30 / ndrop3):
  daily_lag0_origcost   日频, lag0, 回测成本 (复现 0.8865 的 sanity check)
  daily_lag0_livecost   日频, lag0, 实盘成本 (佣金 0.03% + 卖出印花税 0.05% + 滑点 0.1%)
  daily_lag1_livecost   日频, lag1, 实盘成本 (真实"收盘后出信号, 次日执行")
  weekly_lag1_livecost  周频 (周一调仓, 用上周五分数), lag1, 实盘成本

成本口径:
  回测: commission 0.03% 双边, slippage 0.01% 双边, 无印花税
  实盘: open_cost=0.03%, close_cost=0.08% (佣金+印花税), slippage=0.1% 双边
  (引擎无印花税, 用 close_cost 显式补齐; min_cost=0 因为 3.17% 仓位的佣金 > 5 元)

输出: data/strategy_baseline/ic_composite_10y_execution_ab.json
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

BACKEND_ROOT = "/Users/rone/Projects/stock-platform/backend"
sys.path.insert(0, BACKEND_ROOT)

_spec = importlib.util.spec_from_file_location(
    "ds", os.path.join(BACKEND_ROOT, "scripts", "daily_signal.py"))
ds = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ds)

_spec2 = importlib.util.spec_from_file_location(
    "xleg", os.path.join(BACKEND_ROOT, "scripts", "expanded_legs_10y.py"))
m = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(m)

from app.core.config import settings  # noqa: E402
from app.services.backtest import BacktestExecutor  # noqa: E402
from app.services.backtest.models import BacktestConfig  # noqa: E402
from app.services.backtest.strategies.factor_ranking_strategy import (  # noqa: E402
    FactorRankingStrategy, _SCORES_ATTR)

OUT_JSON = "/Users/rone/Projects/stock-platform/data/strategy_baseline/ic_composite_10y_execution_ab.json"
START, END = m.START, m.END
OOS_SPLIT = pd.Timestamp("2021-05-11")
TOPK, NDROP = 30, 3

t0 = time.time()
m.patch_loader()

frames_all, panel, master_all = ds.load_data(m.list_universe())
comp_all = ds.compute_composite(frames_all, panel, master_all)
print(f"[composite] {len(frames_all)} stocks, {len(master_all)} days {time.time()-t0:.0f}s", flush=True)

frames = {}
for c, f in frames_all.items():
    x = f.loc[START:END]
    if len(x) > 60:
        x = x[~x.index.duplicated(keep="first")]
        x.attrs["stock_code"] = c
        x.attrs["from_precomputed"] = True
        frames[c] = x
codes = list(frames)
master = pd.DatetimeIndex(sorted(set().union(*[set(f.index) for f in frames.values()])))
comp = comp_all.loc[START:END, codes].reindex(master)
print(f"[window] {len(codes)} stocks {len(master)} days {START.date()}->{END.date()} {time.time()-t0:.0f}s", flush=True)

_orig_prepare = FactorRankingStrategy.prepare_backtest_data
_orig_snap = m.pma_module.PortfolioManagerArray.record_portfolio_snapshot


async def _noop_prepare(self, *a, **k):
    return None


def prep_scores(lag: int, rebalance: str) -> pd.DataFrame:
    """按 lag / 调仓频率生成分数矩阵。lag 是交易日行位移 (近似 horizon runner 的逐股 shift)。"""
    c = comp.shift(lag) if lag else comp.copy()
    if rebalance == "weekly_mon":
        # 周一用上一交易日(通常周五)的分数; shift(1) 已把周五值移到周一
        mask = pd.Series(c.index.dayofweek == 0, index=c.index)
        c = c.where(mask, np.nan)
    elif rebalance == "weekly_fri":
        mask = pd.Series(c.index.dayofweek == 4, index=c.index)
        c = c.where(mask, np.nan)
    return c


def inject(comp_cell: pd.DataFrame) -> None:
    for c in codes:
        col = comp_cell[c].to_numpy(dtype=np.float64)
        smap = {master[k].to_pydatetime(): float(v) for k, v in enumerate(col) if np.isfinite(v)}
        frames[c].attrs["stock_code"] = c
        frames[c].attrs["from_precomputed"] = True
        frames[c].attrs[_SCORES_ATTR] = smap


async def run_cell(name: str, lag: int, rebalance: str, live_cost: bool):
    inject(prep_scores(lag, rebalance))
    ex = BacktestExecutor(data_dir=str(settings.DATA_ROOT_PATH),
                          enable_parallel=True, max_workers=8,
                          enable_performance_profiling=False,
                          use_multiprocessing=False)
    ex.data_loader.load_multiple_stocks = lambda *a, **k: frames
    FactorRankingStrategy.prepare_backtest_data = _noop_prepare
    stash = {}

    def patched(self, date, current_prices):
        stash["pm"] = self
        _orig_snap(self, date, current_prices)

    m.pma_module.PortfolioManagerArray.record_portfolio_snapshot = patched
    try:
        if live_cost:
            cfg = BacktestConfig(
                initial_cash=1_000_000.0, commission_rate=0.0003, slippage_rate=0.001,
                open_cost=0.0003, close_cost=0.0008, min_cost=0.0,
                max_position_size=round(0.95 / TOPK, 6), cash_reserve_ratio=0.05,
                board_lot_size=100, stop_loss_pct=0.0, take_profit_pct=0.0,
                rebalance_frequency="daily",
            )
        else:
            cfg = BacktestConfig(
                initial_cash=1_000_000.0, commission_rate=0.0003, slippage_rate=0.0001,
                max_position_size=round(0.95 / TOPK, 6), cash_reserve_ratio=0.05,
                board_lot_size=100, stop_loss_pct=0.0, take_profit_pct=0.0,
                rebalance_frequency="daily",
            )
        await ex.run_backtest(
            strategy_name="factor_ranking", stock_codes=codes,
            start_date=START, end_date=END,
            strategy_config={"topk": TOPK, "n_drop": NDROP, "hold_thresh": 0},
            backtest_config=cfg,
        )
    finally:
        m.pma_module.PortfolioManagerArray.record_portfolio_snapshot = _orig_snap
        FactorRankingStrategy.prepare_backtest_data = _orig_prepare

    pm = stash.get("pm")
    eq = [(pd.Timestamp(d), float(v)) for d, v in pm.equity_curve]
    vals = np.array([v for _, v in eq])
    rets = np.diff(vals) / vals[:-1]
    stats = m._ret_stats(rets)
    yearly = {}
    for d, v in eq:
        yearly.setdefault(d.year, []).append(v)
    yr = {str(y): round((vs[-1] / vs[0] - 1) * 100, 1) for y, vs in sorted(yearly.items()) if len(vs) > 1}
    subs = {}
    for sname, lo, hi in [("oos_16_21", START, OOS_SPLIT), ("in_21_26", OOS_SPLIT, END)]:
        mask = np.array([(d >= lo) and (d <= hi) for d, _ in eq])
        if mask.sum() > 30:
            sv = vals[mask]
            sr = m._ret_stats(np.diff(sv) / sv[:-1])
            subs[sname] = {k: round(float(v), 4) for k, v in sr.items()
                           if isinstance(v, (int, float))}
    print(f"[{name}] sharpe={stats['sharpe_ratio']:.3f} ret={stats['total_return']*100:.0f}% "
          f"mdd={stats['max_drawdown']*100:.1f}% ann={stats['annualized_return']*100:.1f}% "
          f"trades={len(pm.trades)} yearly={yr}", flush=True)
    return {"name": name, "lag": lag, "rebalance": rebalance, "live_cost": live_cost,
            "stats": {k: float(v) for k, v in stats.items() if isinstance(v, (int, float))},
            "subperiods": subs, "yearly": yr, "n_trades": len(pm.trades)}


CELLS = [
    ("daily_lag0_origcost", 0, "daily", False),
    ("daily_lag0_livecost", 0, "daily", True),
    ("daily_lag1_livecost", 1, "daily", True),
    ("weekly_lag1_livecost", 1, "weekly_mon", True),
]


def save_results(cells, partial: bool) -> None:
    payload = {"meta": {
        "window": [str(START.date()), str(END.date())],
        "universe": "point-in-time 4329", "factor_set": "honest_composite 42f",
        "topk": TOPK, "n_drop": NDROP,
        "cost_model": {"orig": "commission 0.03% + slippage 0.01%, no stamp tax",
                       "live": "open_cost 0.03%, close_cost 0.08% (incl 0.05% stamp), slippage 0.1%"},
        "note": "lag1=收盘后信号次日执行; weekly_mon=周一调仓用上一交易日分数",
        "partial": partial,
        "generated_at": datetime.now().isoformat(),
        "elapsed_s": round(time.time() - t0)},
        "cells": cells}
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1, default=str)


async def main():
    out = []
    for name, lag, rebalance, live in CELLS:
        out.append(await run_cell(name, lag, rebalance, live))
        save_results(out, partial=True)  # 每个单元跑完立即落盘, 崩溃不丢已完成结果
    return out


cells = asyncio.run(main())
save_results(cells, partial=False)
print(f"[done] {time.time()-t0:.0f}s -> {OUT_JSON}", flush=True)
