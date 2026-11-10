#!/usr/bin/env python3
"""池子卫生 A/B: 剔除北交所/低价/低流动性/次新 → 10 年回测 topk60 vs topk30.

动机: 42f 复合的 -37% 回撤主要是 2018 一年砸出来的 (-30.9%)。因子池里的 MIN
(接近 60 日低点) 天然偏爱超跌小盘/低价股, 这类票在崩盘里跌得最狠、还有退市/
流动性风险。本实验检验: 只做"池子卫生"这个非择时、非因子改动的结构过滤,
能否显著降低回撤而不明显损失收益。

四类过滤 (全部 point-in-time, 从现有 OHLCV 精确算):
  1. 北交所 (.BJ, 30% 涨跌停 + 流动性差) — 静态剔除
  2. 收盘价 < 2.0 元 (低价/退市风险)
  3. 20 日均成交额 < 5000 万元 (低流动性)
  4. 上市不足 60 个交易日 (次新)

ST 未纳入: 项目内无历史 ST 标记 (qlib 面板无 isST, tdx 原始包仅 108 天)。
ST 多为低价+低流动性, 会被 2/3 过滤掉一部分, 但不是精确剔除 — 后续用 Baostock
isST 单独补。

四个单元 (同 point-in-time 4329 宇宙, 同 42f 打分):
  base_topk60_ndrop5      无过滤, 基线
  base_topk30_ndrop3      无过滤
  hygiene_topk60_ndrop5   过滤
  hygiene_topk30_ndrop3   过滤

判据: 同 topk 下, hygiene vs base 的 Sharpe/收益/回撤/OOS 对比。
输出: data/strategy_baseline/ic_composite_10y_hygiene.json
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

OUT_JSON = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "strategy_baseline", "ic_composite_10y_hygiene.json")
START, END = m.START, m.END
OOS_SPLIT = pd.Timestamp("2021-05-11")

MIN_PRICE = 2.0
MIN_AMOUNT_20D = 50_000_000.0
MIN_LISTED_DAYS = 60


def compute_composite_masked(frames, panel, master, eligible: pd.DataFrame | None = None):
    """与 ds.compute_composite 相同的 42f 逻辑, 但 z-score 截面只统计 eligible 股票。"""
    n_d, n_s = len(master), len(frames)
    close = ds.to_mat(frames, master, "close")
    vol = ds.to_mat(frames, master, "volume")
    with np.errstate(divide="ignore", invalid="ignore"):
        ret = close.pct_change(fill_method=None)
        vchg = vol.pct_change(fill_method=None)
        cands: dict[str, pd.DataFrame] = {}
        for n in (5, 10, 20, 30, 60):
            cands[f"MIN{n}"] = close.rolling(n).min() / close - 1.0
            cands[f"MAX{n}"] = close.rolling(n).max() / close - 1.0
            cands[f"STD{n}"] = close.rolling(n).std() / close
            cands[f"CORR{n}"] = close.rolling(n).corr(vol)
            cands[f"CORD{n}"] = ret.rolling(n).corr(vchg)

    elig = None if eligible is None else eligible.to_numpy(dtype=bool)
    acc = np.zeros((n_d, n_s))
    wsum = np.zeros((n_d, n_s))

    def zadd(mat: np.ndarray, sign: float) -> None:
        if elig is not None:
            mat = np.where(elig, mat, np.nan)
        ds.zscore_add(acc, wsum, mat, sign, ds.MIN_CROSS)

    for fname, sign in ds._HONEST_FORMULAS.items():
        zadd(cands[fname].to_numpy(), sign)
    for fname, sign in ds._ALPHAPANEL_FACTORS.items():
        zadd(ds.to_panel_mat(panel, master, fname), sign)

    composite = np.where(wsum > 0, acc / np.where(wsum > 0, wsum, 1.0), np.nan)
    if elig is not None:
        composite = np.where(elig, composite, np.nan)
    limit_up = close.pct_change(fill_method=None).to_numpy() >= ds.LIMIT_UP
    composite[limit_up] = np.nan
    return pd.DataFrame(composite, index=master, columns=list(frames))


def build_eligibility(frames, master) -> pd.DataFrame:
    close = ds.to_mat(frames, master, "close")
    vol = ds.to_mat(frames, master, "volume")
    amount = close * vol * 100.0                      # volume 手 → 股
    avg_amt20 = amount.rolling(20, min_periods=10).mean()
    listed = close.notna().cumsum()
    elig = (close >= MIN_PRICE) & (avg_amt20 >= MIN_AMOUNT_20D) & (listed >= MIN_LISTED_DAYS)
    for c in elig.columns:
        if c.endswith(".BJ"):
            elig[c] = False
    return elig


t0 = time.time()
m.patch_loader()

# 1) point-in-time 固定池 + 全历史数据
frames_all, panel, master_all = ds.load_data(m.list_universe())
print(f"[load] {len(frames_all)} stocks {len(master_all)} days {time.time()-t0:.0f}s", flush=True)

# 2) 两份复合: 无过滤 / 有过滤
comp_base = compute_composite_masked(frames_all, panel, master_all, None)
elig_all = build_eligibility(frames_all, master_all)
comp_hyg = compute_composite_masked(frames_all, panel, master_all, elig_all)
print(f"[composite] base_valid={np.isfinite(comp_base.to_numpy()).mean():.1%} "
      f"hyg_valid={np.isfinite(comp_hyg.to_numpy()).mean():.1%} "
      f"elig={elig_all.to_numpy().mean():.1%} {time.time()-t0:.0f}s", flush=True)

# 3) 切 10y 窗口 + 注入分数
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
comp_base = comp_base.loc[START:END, codes].reindex(master)
comp_hyg = comp_hyg.loc[START:END, codes].reindex(master)
n_d = len(master)
print(f"[window] {len(codes)} stocks {n_d} days {START.date()}->{END.date()} {time.time()-t0:.0f}s", flush=True)

_orig_prepare = FactorRankingStrategy.prepare_backtest_data
_orig_snap = m.pma_module.PortfolioManagerArray.record_portfolio_snapshot


async def _noop_prepare(self, *a, **k):
    return None


def inject(comp: pd.DataFrame) -> None:
    for c in codes:
        col = comp[c].to_numpy(dtype=np.float64)
        smap = {master[k].to_pydatetime(): float(v) for k, v in enumerate(col) if np.isfinite(v)}
        frames[c].attrs["stock_code"] = c
        frames[c].attrs["from_precomputed"] = True
        frames[c].attrs[_SCORES_ATTR] = smap


async def run_cell(name, comp, topk, ndrop):
    inject(comp)
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
          f"mdd={stats['max_drawdown']*100:.1f}% oos={subs.get('oos_16_21',{}).get('sharpe_ratio')} "
          f"in={subs.get('in_21_26',{}).get('sharpe_ratio')} trades={len(pm.trades)} yearly={yr}",
          flush=True)
    return {"name": name, "topk": topk, "n_drop": ndrop,
            "stats": {k: float(v) for k, v in stats.items() if isinstance(v, (int, float))},
            "subperiods": subs, "yearly": yr, "n_trades": len(pm.trades)}


CELLS = [
    ("base_topk60_ndrop5", False, 60, 5),
    ("base_topk30_ndrop3", False, 30, 3),
    ("hygiene_topk60_ndrop5", True, 60, 5),
    ("hygiene_topk30_ndrop3", True, 30, 3),
]


async def main():
    out = []
    for name, use_hyg, topk, ndrop in CELLS:
        out.append(await run_cell(name, comp_hyg if use_hyg else comp_base, topk, ndrop))
    return out


cells = asyncio.run(main())
out = {"meta": {
    "window": [str(START.date()), str(END.date())],
    "universe": "point-in-time 4329",
    "factor_set": "honest_composite 42f",
    "filters": {"exclude_bj": True, "min_price": MIN_PRICE,
                "min_amount_20d": MIN_AMOUNT_20D, "min_listed_days": MIN_LISTED_DAYS,
                "st": "not available (no historical isST)"},
    "generated_at": datetime.now().isoformat(),
    "elapsed_s": round(time.time() - t0)},
    "cells": cells}
os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
with open(OUT_JSON, "w") as fh:
    json.dump(out, fh, ensure_ascii=False, indent=1, default=str)
print(f"[done] {time.time()-t0:.0f}s -> {OUT_JSON}", flush=True)
