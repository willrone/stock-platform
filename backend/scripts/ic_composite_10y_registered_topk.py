#!/usr/bin/env python3
"""注册版 honest_composite (42f) 的 topk 集中度扫描: 60/40/30/20.

为什么单独写:
  ic_composite_10y_expand2.py 依赖 /tmp/factor_formula_map.json (31 个已验证因子),
  该临时文件已丢失; 子代理重建的 25 项版本因子集不同 → topk60 只有 0.8206,
  不能与历史 0.976 比。本脚本不依赖任何临时映射, 直接用
  backend/scripts/daily_signal.py 的 compute_composite — 与注册策略
  honest_composite_strategy.py 的 42 因子定义逐行一致:
    23 基础公式 (MIN/MAX/STD/CORR/CORD × 5/10/20/30/60)
    + 17 Alpha158 面板因子 (VSUMP/VSUMN/VSUMD/QTLD × 5/10/20 + IMAX/IMIN/WVMA/VMA/VSTD × 5)
  等权截面 z-score, 涨停掩码 9.5%。

输出: data/strategy_baseline/ic_composite_10y_registered_topk.json
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

# 复用 daily_signal 的精确 42f 打分 (不依赖 /tmp 映射)
_spec = importlib.util.spec_from_file_location(
    "ds", os.path.join(BACKEND_ROOT, "scripts", "daily_signal.py"))
ds = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ds)

# 复用 expanded_legs 的 10y 窗口/数据层/统计函数
_spec2 = importlib.util.spec_from_file_location(
    "xleg", os.path.join(BACKEND_ROOT, "scripts", "expanded_legs_10y.py"))
m = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(m)

from app.core.config import settings  # noqa: E402
from app.services.backtest import BacktestExecutor  # noqa: E402
from app.services.backtest.models import BacktestConfig  # noqa: E402
from app.services.backtest.strategies.factor_ranking_strategy import (  # noqa: E402
    FactorRankingStrategy, _SCORES_ATTR)

OUT_JSON = "/Users/rone/Projects/stock-platform/data/strategy_baseline/ic_composite_10y_registered_topk.json"
START, END = m.START, m.END
OOS_SPLIT = pd.Timestamp("2021-05-11")

t0 = time.time()
m.patch_loader()

# 1) 42f 复合打分 (全历史算完再切窗口, 保证滚动窗预热)
#    宇宙必须用 expanded_legs 的 point-in-time 固定池 (4329 只, 2021-05 时点),
#    不能用"当前全部文件" (5595, 含 1266 只 2021 后上市) — 否则截面 z-score 与
#    历史基线不可比。
frames_all, panel, master_all = ds.load_data(m.list_universe())
comp_all = ds.compute_composite(frames_all, panel, master_all)
print(f"[composite] {len(frames_all)} stocks, {len(master_all)} days, "
      f"valid={np.isfinite(comp_all.to_numpy()).mean():.1%} {time.time()-t0:.0f}s", flush=True)

# 2) 切到 10y 窗口
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
n_d, n_s = len(master), len(codes)
print(f"[window] {n_s} stocks {n_d} days {START.date()}->{END.date()} {time.time()-t0:.0f}s", flush=True)

# 3) 分数注入 attrs
for j, c in enumerate(codes):
    col = comp[c].to_numpy(dtype=np.float64)
    smap = {}
    for k in range(n_d):
        v = col[k]
        if np.isfinite(v):
            smap[master[k].to_pydatetime()] = float(v)
    frames[c].attrs["stock_code"] = c
    frames[c].attrs["from_precomputed"] = True
    frames[c].attrs[_SCORES_ATTR] = smap

_orig_prepare = FactorRankingStrategy.prepare_backtest_data
_orig_snap = m.pma_module.PortfolioManagerArray.record_portfolio_snapshot


async def _noop_prepare(self, *a, **k):
    return None


async def run_cell(name, topk, ndrop):
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
        cfg = BacktestConfig(
            initial_cash=1_000_000.0, commission_rate=0.0003, slippage_rate=0.0001,
            max_position_size=round(0.95 / topk, 6), cash_reserve_ratio=0.05,
            board_lot_size=100, stop_loss_pct=0.0, take_profit_pct=0.0,
            rebalance_frequency="daily",
        )
        await ex.run_backtest(
            strategy_name="factor_ranking", stock_codes=codes,
            start_date=START, end_date=END,
            strategy_config={"topk": topk, "n_drop": ndrop, "hold_thresh": 0},
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
    subs = {}
    for sname, lo, hi in [("oos_16_21", START, OOS_SPLIT), ("in_21_26", OOS_SPLIT, END)]:
        mask = np.array([(d >= lo) and (d <= hi) for d, _ in eq])
        if mask.sum() > 30:
            sv = vals[mask]
            sr = m._ret_stats(np.diff(sv) / sv[:-1])
            subs[sname] = {k: round(float(v), 4) for k, v in sr.items()
                           if isinstance(v, (int, float))}
    print(f"[{name}] sharpe={stats['sharpe_ratio']:.3f} ret={stats['total_return']*100:.0f}% "
          f"mdd={stats['max_drawdown']*100:.1f}% oos={subs.get('oos_16_21',{}).get('sharpe_ratio')} "
          f"in={subs.get('in_21_26',{}).get('sharpe_ratio')} trades={len(pm.trades)}", flush=True)
    return {"name": name, "topk": topk, "n_drop": ndrop,
            "stats": {k: float(v) for k, v in stats.items() if isinstance(v, (int, float))},
            "subperiods": subs, "n_trades": len(pm.trades)}


CELLS = [
    ("42f_topk60_ndrop5", 60, 5),
    ("42f_topk40_ndrop4", 40, 4),
    ("42f_topk30_ndrop3", 30, 3),
    ("42f_topk20_ndrop2", 20, 2),
]


async def main():
    out = []
    for name, topk, ndrop in CELLS:
        out.append(await run_cell(name, topk, ndrop))
    return out


cells = asyncio.run(main())
out = {"meta": {
    "window": [str(START.date()), str(END.date())],
    "factor_set": "honest_composite 42f (daily_signal.compute_composite)",
    "note": "注册策略精确 42f; 不依赖 /tmp/factor_formula_map.json",
    "generated_at": datetime.now().isoformat(),
    "elapsed_s": round(time.time() - t0)},
    "cells": cells}
os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
with open(OUT_JSON, "w") as fh:
    json.dump(out, fh, ensure_ascii=False, indent=1, default=str)
print(f"[done] {time.time()-t0:.0f}s -> {OUT_JSON}", flush=True)
