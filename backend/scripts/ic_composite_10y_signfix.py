#!/usr/bin/env python3
"""42f 因子符号修正 + 集中度 A/B 扫描（10 年窗口, PIT 宇宙）。

背景:
  记忆库实证（2026-10-19 滚动 walk-forward）: 22f 固定版的 edge 来自
  **MAX 符号修正** —— 2016-2021 期间 MAX30/MAX60 的 IC 为正, 而 42f
  把 MAX5/10/20/30/60 全部硬编码为 -1。walk-forward 用 +1 时
  2021-2026 滚动 OOS 夏普 0.685, all23 只有 0.348。
  22f 固定版 = MIN +1, STD/CORR/CORD -1, MAX30/60 +1, 剔 MAX5/10/20。

本脚本在**同一 10y 窗口、同一 PIT 宇宙、同一费率、同一执行模型**下，
只改因子符号/集合，回答：
  A. 符号修正（MAX30/60 → +1、剔 MAX5/10/20）是否真的优于现状 42f？
  B. 若成立，它与 17 个 Alpha158 面板因子叠加后是否仍然成立？
  C. 集中度（topk/ndrop）在这个新因子上是否需要重扫？

设计: 复用 registered_topk 的架构（预计算 composite → 注入 attrs → 跑引擎），
唯一变量是 (因子符号表, topk, ndrop)。每个 cell 独立跑一次引擎。

输出: data/strategy_baseline/ic_composite_10y_signfix.json
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

BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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

OUT_JSON = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "strategy_baseline", "ic_composite_10y_signfix.json")
START, END = m.START, m.END
OOS_SPLIT = pd.Timestamp("2021-05-11")

t0 = time.time()
m.patch_loader()

# ── 因子符号变体 ──────────────────────────────────────────────────────
# 基线：注册版 42f（MAX 全 -1）
BASE_FORMULAS = dict(ds._HONEST_FORMULAS)
BASE_PANEL = dict(ds._ALPHAPANEL_FACTORS)

# 变体 A「符号修正」：MAX30/60 → +1, 剔 MAX5/10/20（22f 固定版的 edge）
SIGNFIX_FORMULAS = {k: v for k, v in BASE_FORMULAS.items()
                    if k not in ("MAX5", "MAX10", "MAX20")}
SIGNFIX_FORMULAS["MAX30"] = 1.0
SIGNFIX_FORMULAS["MAX60"] = 1.0

# 变体 B「仅符号修正、保留全部窗口」：MAX30/60 → +1，MAX5/10/20 仍 -1
MAX30_60_FORMULAS = dict(BASE_FORMULAS)
MAX30_60_FORMULAS["MAX30"] = 1.0
MAX30_60_FORMULAS["MAX60"] = 1.0

VARIANTS = [
    ("base_42f", BASE_FORMULAS, BASE_PANEL),
    ("max3060_plus", MAX30_60_FORMULAS, BASE_PANEL),
    ("signfix_22f_core", SIGNFIX_FORMULAS, BASE_PANEL),
    ("signfix_22f_core_noPanel", SIGNFIX_FORMULAS, {}),
]


def compute_composite_variant(frames, panel, master, formulas, panel_factors):
    """按给定符号表算复合分（与 daily_signal.compute_composite 同构）。"""
    n_d, n_s = len(master), len(frames)
    close = ds.to_mat(frames, master, "close")
    vol = ds.to_mat(frames, master, "volume")
    with np.errstate(divide="ignore", invalid="ignore"):
        ret = close.pct_change(fill_method=None)
        vchg = vol.pct_change(fill_method=None)
        cands = {}
        for n in (5, 10, 20, 30, 60):
            cands[f"MIN{n}"] = close.rolling(n).min() / close - 1.0
            cands[f"MAX{n}"] = close.rolling(n).max() / close - 1.0
            cands[f"STD{n}"] = close.rolling(n).std() / close
            cands[f"CORR{n}"] = close.rolling(n).corr(vol)
            cands[f"CORD{n}"] = ret.rolling(n).corr(vchg)

    acc = np.zeros((n_d, n_s))
    wsum = np.zeros((n_d, n_s))
    for fname, sign in formulas.items():
        ds.zscore_add(acc, wsum, cands[fname].to_numpy(), sign, ds.MIN_CROSS)
    for fname, sign in panel_factors.items():
        ds.zscore_add(acc, wsum, ds.to_panel_mat(panel, master, fname),
                      sign, ds.MIN_CROSS)

    composite = np.where(wsum > 0, acc / np.where(wsum > 0, wsum, 1.0), np.nan)
    limit_up = close.pct_change(fill_method=None).to_numpy() >= ds.LIMIT_UP
    composite[limit_up] = np.nan
    return pd.DataFrame(composite, index=master, columns=list(frames))


# 1) 全历史加载（一次），算完再切窗口
frames_all, panel, master_all = ds.load_data(m.list_universe())
print(f"[load] {len(frames_all)} stocks, {len(master_all)} days, {time.time()-t0:.0f}s",
      flush=True)

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
n_d, n_s = len(master), len(codes)
print(f"[window] {n_s} stocks {n_d} days {START.date()}->{END.date()} "
      f"{time.time()-t0:.0f}s", flush=True)

_orig_prepare = FactorRankingStrategy.prepare_backtest_data
_orig_snap = m.pma_module.PortfolioManagerArray.record_portfolio_snapshot


async def _noop_prepare(self, *a, **k):
    return None


async def run_cell(name, topk, ndrop, composite):
    # 分数注入 attrs
    comp = composite.reindex(master)
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
          f"mdd={stats['max_drawdown']*100:.1f}% "
          f"oos={subs.get('oos_16_21',{}).get('sharpe_ratio')} "
          f"in={subs.get('in_21_26',{}).get('sharpe_ratio')} trades={len(pm.trades)}",
          flush=True)
    return {"name": name, "topk": topk, "n_drop": ndrop,
            "stats": {k: float(v) for k, v in stats.items() if isinstance(v, (int, float))},
            "subperiods": subs, "n_trades": len(pm.trades)}


# 每个符号变体 → 扫 topk 集中度
TOPK_GRID = [(60, 5), (30, 3)]


async def main():
    cells = []
    for vname, formulas, panels in VARIANTS:
        t1 = time.time()
        comp_all = compute_composite_variant(
            frames_all, panel, master_all, formulas, panels)
        comp_win = comp_all.loc[START:END, codes].reindex(master)
        valid = np.isfinite(comp_win.to_numpy()).mean()
        print(f"[variant] {vname}: n_factors={len(formulas)+len(panels)} "
              f"valid={valid:.1%} {time.time()-t1:.0f}s", flush=True)
        for topk, ndrop in TOPK_GRID:
            cell = await run_cell(f"{vname}_topk{topk}_ndrop{ndrop}",
                                  topk, ndrop, comp_win)
            cell["variant"] = vname
            cell["n_factors"] = len(formulas) + len(panels)
            cells.append(cell)
            # 增量落盘（每个 cell 跑完即写）
            out = {"meta": {
                "window": [str(START.date()), str(END.date())],
                "note": "符号修正 A/B: 只改因子符号表, 窗口/宇宙/费率/执行全同",
                "generated_at": datetime.now().isoformat(),
                "elapsed_s": round(time.time() - t0)},
                "cells": cells}
            os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
            with open(OUT_JSON, "w") as fh:
                json.dump(out, fh, ensure_ascii=False, indent=1, default=str)
    return cells


cells = asyncio.run(main())
print(f"\n===== 符号修正 A/B 汇总 =====", flush=True)
print(f"{'cell':<44}{'sharpe':>9}{'ret':>9}{'mdd':>9}{'oos':>8}{'in':>8}")
for c in cells:
    s = c["stats"]
    print(f"{c['name']:<44}{s['sharpe_ratio']:>9.4f}{s['total_return']:>9.1%}"
          f"{s['max_drawdown']:>9.1%}"
          f"{(c['subperiods'].get('oos_16_21') or {}).get('sharpe_ratio', float('nan')):>8.4f}"
          f"{(c['subperiods'].get('in_21_26') or {}).get('sharpe_ratio', float('nan')):>8.4f}")
print(f"\n[saved] {OUT_JSON}  ({time.time()-t0:.0f}s)", flush=True)
