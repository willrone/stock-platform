"""面板复合因子 5y 验证第二步: 对照组 + 邻域格子 (三门纪律的门1/门2).

A) control: 官方 factor_ranking 默认因子 (momentum+low_vol+low_amp) 同窗同宇宙同成本
B) neighbors: topk=40/60, n_drop=3/7 (复合因子分数不变)
"""
import asyncio
import importlib.util
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

BACKEND_ROOT = "/Users/rone/projects/stock-platform/backend"
sys.path.insert(0, BACKEND_ROOT)
_spec = importlib.util.spec_from_file_location(
    "xleg", os.path.join(BACKEND_ROOT, "scripts", "expanded_legs_10y.py"))
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)

from app.core.config import settings
from app.services.backtest import BacktestExecutor
from app.services.backtest.models import BacktestConfig
from app.services.backtest.strategies.factor_ranking_strategy import (
    FactorRankingStrategy, _SCORES_ATTR)

FEAT_DIR = "/Users/rone/projects/stock-platform/data/qlib_data/features/day"
SURV_JSON = "/Users/rone/projects/stock-platform/data/strategy_baseline/factor_ic_study_5y.json"
OUT_JSON = "/Users/rone/projects/stock-platform/data/strategy_baseline/ic_composite_panel5y_cells.json"
B2 = "CNTP5"
START = pd.Timestamp("2021-05-11")

t0 = time.time()
m.patch_loader()
codes = m.list_universe()
END = m.END

study = json.load(open(SURV_JSON))
survivors = study["survivors"]
SIGNS = {c: (1.0 if study["results"][c]["fwd1"]["ic_mean"] > 0 else -1.0) for c in survivors}

# ---- 面板分数 (与 ic_composite_panel5y 相同) ----
feat_files = [c.replace(".", "_") + ".parquet" for c in codes]
files = [f for f in feat_files if os.path.exists(os.path.join(FEAT_DIR, f))]
date_sets = []
for f in files[::60]:
    x = pd.read_parquet(os.path.join(FEAT_DIR, f), columns=["$close"])
    idx = pd.DatetimeIndex(x.index.get_level_values("date"))
    date_sets.append(set(idx[x["$close"].notna().to_numpy()]))
dates = pd.DatetimeIndex(sorted(set().union(*date_sets)))
dates = dates[(dates >= START) & (dates <= END)]
n_d = len(dates)
pos_of = {d: i for i, d in enumerate(dates)}

feat_mats = {c: np.full((n_d, len(files)), np.nan, dtype=np.float32) for c in survivors}
for j, f in enumerate(files):
    try:
        p = pd.read_parquet(os.path.join(FEAT_DIR, f))
    except Exception:
        continue
    idx = pd.DatetimeIndex(p.index.get_level_values("date"))
    p = p.copy(); p.index = idx
    b2 = p[p[B2].notna()]
    if b2.empty:
        continue
    pos = np.array([pos_of[d] for d in b2.index if d in pos_of], dtype=np.int64)
    for c in survivors:
        if c in p.columns:
            v = b2[c].to_numpy(dtype=np.float32)
            n = min(len(pos), len(v))
            feat_mats[c][pos[:n], j] = v[:n]

acc = np.zeros((n_d, len(files)), dtype=np.float64)
wsum = np.zeros((n_d, len(files)), dtype=np.float64)
for c, mat in feat_mats.items():
    M = mat.astype(np.float64) * SIGNS[c]
    ok = np.isfinite(M)
    cnt = ok.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore"):
        mu = np.nanmean(np.where(ok, M, np.nan), axis=1, keepdims=True)
        sd = np.nanstd(np.where(ok, M, np.nan), axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (M - mu) / np.where(sd > 1e-12, sd, np.nan)
    z = np.where(cnt >= 30, z, np.nan)
    good = np.isfinite(z)
    acc[good] += np.nan_to_num(z)[good]
    wsum[good] += 1
composite = np.where(wsum > 0, acc / np.where(wsum > 0, wsum, 1), np.nan)

# ---- OHLCV frames ----
stock_data = {}
for c in codes:
    f = c.replace(".", "_") + ".parquet"
    try:
        df = pd.read_parquet(os.path.join(m.STOCK_DATA_DIR, f))
    except Exception:
        continue
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df = df.loc[START:END]
    df = df[~df.index.duplicated(keep="first")]
    if len(df) < 60:
        continue
    df.attrs["stock_code"] = c
    df.attrs["from_precomputed"] = True
    stock_data[c] = df
usable = [c for c in codes if c in stock_data]
col_of = {f[:-8].replace("_", "."): j for j, f in enumerate(files)}

def inject_panel_scores():
    n = 0
    for c in usable:
        j = col_of.get(c)
        if j is None:
            continue
        smap = {}
        for k in range(n_d):
            v = composite[k, j]
            if np.isfinite(v):
                smap[dates[k].to_pydatetime()] = float(v)
        stock_data[c].attrs[_SCORES_ATTR] = smap
        n += len(smap)
    return n

print(f"[setup] stocks={len(usable)} {time.time()-t0:.0f}s", flush=True)

_orig_prepare = FactorRankingStrategy.prepare_backtest_data
_orig_snap = m.pma_module.PortfolioManagerArray.record_portfolio_snapshot

async def _noop_prepare(self, *a, **k):
    return None

async def run_cell(name, cfg_kwargs, use_panel_scores, weights):
    ex = BacktestExecutor(data_dir=str(settings.DATA_ROOT_PATH),
                          enable_parallel=True, max_workers=8,
                          enable_performance_profiling=False,
                          use_multiprocessing=False)
    ex.data_loader.load_multiple_stocks = lambda *a, **k: stock_data
    if use_panel_scores:
        inject_panel_scores()
        FactorRankingStrategy.prepare_backtest_data = _noop_prepare
    else:
        # 控制组: 必须清掉注入的面板分数, 否则 calculate_indicators 会优先用 _SCORES_ATTR 污染对照
        for df in stock_data.values():
            df.attrs.pop(_SCORES_ATTR, None)
        FactorRankingStrategy.prepare_backtest_data = _orig_prepare
    stash = {}

    def patched(self, date, current_prices):
        stash["pm"] = self
        _orig_snap(self, date, current_prices)
    m.pma_module.PortfolioManagerArray.record_portfolio_snapshot = patched
    try:
        cfg = BacktestConfig(
            initial_cash=1_000_000.0, commission_rate=0.0003, slippage_rate=0.0001,
            max_position_size=0.019, cash_reserve_ratio=0.05, board_lot_size=100,
            stop_loss_pct=0.0, take_profit_pct=0.0, rebalance_frequency="daily",
        )
        sc = {"topk": 50, "n_drop": 5, "hold_thresh": 0}
        sc.update(cfg_kwargs)
        if weights:
            sc["weights"] = weights
        await ex.run_backtest(
            strategy_name="factor_ranking",
            stock_codes=usable, start_date=START, end_date=END,
            strategy_config=sc, backtest_config=cfg)
    finally:
        m.pma_module.PortfolioManagerArray.record_portfolio_snapshot = _orig_snap
        FactorRankingStrategy.prepare_backtest_data = _orig_prepare
    pm = stash.get("pm")
    eq = [(d, float(v)) for d, v in pm.equity_curve]
    vals = np.array([v for _, v in eq])
    rets = np.diff(vals) / vals[:-1]
    stats = m._ret_stats(rets)
    yearly = {}
    for d, v in eq:
        yearly.setdefault(pd.Timestamp(d).year, []).append(v)
    yr = {y: round((vs[-1] / vs[0] - 1) * 100, 1) for y, vs in sorted(yearly.items()) if len(vs) > 1}
    print(f"[{name}] sharpe={stats['sharpe_ratio']:.3f} ret={stats['total_return']*100:.1f}% "
          f"mdd={stats['max_drawdown']*100:.1f}% yearly={yr}", flush=True)
    return {"name": name, "stats": {k: float(v) for k, v in stats.items() if isinstance(v, (int, float))},
            "yearly": yr}

async def main():
    results = []
    # 门1: 官方默认因子对照 (同窗同宇宙)
    results.append(await run_cell("control_official", {}, False,
                                  {"momentum": 1.0, "low_vol": 1.0, "low_amp": 1.0}))
    # 门2: 邻域格子 (panel 分数)
    results.append(await run_cell("panel_topk40", {"topk": 40}, True, SIGNS))
    results.append(await run_cell("panel_topk60", {"topk": 60}, True, SIGNS))
    results.append(await run_cell("panel_ndrop3", {"n_drop": 3}, True, SIGNS))
    results.append(await run_cell("panel_ndrop7", {"n_drop": 7}, True, SIGNS))
    return results

results = asyncio.run(main())
out = {"meta": {"window": [str(START.date()), str(END.date())],
                "universe": len(usable), "generated_at": datetime.now().isoformat()},
       "cells": results}
os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
with open(OUT_JSON, "w") as fh:
    json.dump(out, fh, ensure_ascii=False, indent=1, default=str)
print(f"[done] {time.time()-t0:.0f}s -> {OUT_JSON}", flush=True)
