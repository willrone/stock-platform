"""10y 复合因子稳健性电池: 滞后执行 / 滑点加倍 / 邻域格子 / 硬性检查.

背景: ic_composite_10y_v2 基准 Sharpe 1.75 (OOS 2016-21: 1.36) — 远超历史所有腿,
必须按"异常好=先怀疑"处理。核心风险: 因子全部含当日 close (MIN/STD/蜡烛), 而成交价
也是当日 close → 同 bar 信号+成交。本电池测:
  lag1     : 分数滞后一天执行 (T-1 收盘信号, T 收盘成交) — 决定性检验
  slip2x   : 滑点加倍 0.0002
  topk40/60, ndrop3/7 : 邻域格子
输出: data/strategy_baseline/ic_composite_10y_cells.json
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

SURV_JSON = "/Users/rone/projects/stock-platform/data/strategy_baseline/factor_ic_study_5y.json"
FORMULA_MAP = "/tmp/factor_formula_map.json"
OUT_JSON = "/Users/rone/projects/stock-platform/data/strategy_baseline/ic_composite_10y_cells.json"

t0 = time.time()
m.patch_loader()
codes = m.list_universe()
START, END = m.START, m.END

study = json.load(open(SURV_JSON))
fmap = json.load(open(FORMULA_MAP))
use = {c: {"formula": fmap[c]["formula"],
           "sign": (1.0 if study["results"][c]["fwd1"]["ic_mean"] > 0 else -1.0)}
       for c in study["survivors"] if c in fmap}

frames = {}
for c in codes:
    try:
        df = pd.read_parquet(os.path.join(m.STOCK_DATA_DIR, c.replace(".", "_") + ".parquet"))
    except Exception:
        continue
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df = df.loc[START:END]
    df = df[~df.index.duplicated(keep="first")]
    if len(df) > 60:
        frames[c] = df
codes = list(frames)
n_s = len(codes)
master = pd.DatetimeIndex(sorted(set().union(*[set(f.index) for f in frames.values()])))
n_d = len(master)
print(f"[load] {n_s} stocks {n_d} days {time.time()-t0:.0f}s", flush=True)

def to_mat(col):
    out = np.full((n_d, n_s), np.nan)
    for j, c in enumerate(codes):
        pos = master.get_indexer(pd.DatetimeIndex(frames[c].index))
        ok = pos >= 0
        out[pos[ok], j] = pd.to_numeric(frames[c][col], errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)[ok]
    return out

close = pd.DataFrame(to_mat("close"), index=master)
high = pd.DataFrame(to_mat("high"), index=master)
low = pd.DataFrame(to_mat("low"), index=master)
opn = pd.DataFrame(to_mat("open"), index=master)
vol = pd.DataFrame(to_mat("volume"), index=master)

with np.errstate(divide="ignore", invalid="ignore"):
    ret = close.pct_change(fill_method=None)
    vchg = vol.pct_change(fill_method=None)
    cands = {}
    for n in (5, 10, 20, 30, 60):
        cands[f"MIN{n}"] = close.rolling(n).min() / close - 1
        cands[f"MAX{n}"] = close.rolling(n).max() / close - 1
        cands[f"STD{n}"] = close.rolling(n).std() / close
        cands[f"CORR{n}"] = close.rolling(n).corr(vol)
        cands[f"CORD{n}"] = ret.rolling(n).corr(vchg)
        cands[f"RSQR{n}"] = close.rolling(n).corr(
            pd.Series(np.arange(n_d, dtype=float), index=master)) ** 2
    hi_oc = pd.DataFrame(np.maximum(opn.to_numpy(), close.to_numpy()), index=master)
    lo_oc = pd.DataFrame(np.minimum(opn.to_numpy(), close.to_numpy()), index=master)
    cands["KLEN"] = (high - low) / (opn + 1e-12)
    cands["KUP"] = (high - hi_oc) / (opn + 1e-12)
    cands["KLOW"] = (lo_oc - low) / (opn + 1e-12)
    cands["LOW0_a"] = np.log(low / close)
    cands["HIGH0_a"] = np.log(high / close)

seen_formula = {}
for feat, spec in use.items():
    seen_formula.setdefault(spec["formula"], []).append(feat)

acc = np.zeros((n_d, n_s))
wsum = np.zeros((n_d, n_s))
for fname, feats in seen_formula.items():
    mat = cands[fname].to_numpy(dtype=np.float64)
    s = use[feats[0]]["sign"]
    ok = np.isfinite(mat)
    cnt = ok.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore"):
        mu = np.nanmean(np.where(ok, mat, np.nan), axis=1, keepdims=True)
        sd = np.nanstd(np.where(ok, mat, np.nan), axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (mat - mu) / np.where(sd > 1e-12, sd, np.nan) * s
    z = np.where(cnt >= 30, z, np.nan)
    good = np.isfinite(z)
    acc[good] += np.nan_to_num(z)[good]
    wsum[good] += 1
composite = np.where(wsum > 0, acc / np.where(wsum > 0, wsum, 1), np.nan)
print(f"[composite] valid={np.isfinite(composite).mean():.1%} {time.time()-t0:.0f}s", flush=True)

# master 位置映射 (用于 lag 变体)
col_index = {}
for j, c in enumerate(codes):
    pos = master.get_indexer(pd.DatetimeIndex(frames[c].index))
    col_index[c] = pos[pos >= 0]

_orig_prepare = FactorRankingStrategy.prepare_backtest_data
_orig_snap = m.pma_module.PortfolioManagerArray.record_portfolio_snapshot

async def _noop_prepare(self, *a, **k):
    return None

def build_score_map(j, lag):
    """lag=0: T 日分数 T 日用; lag=1: T-1 日分数 T 日用 (在 master 上移位)."""
    col = composite[:, j]
    pos = col_index[codes[j]]
    smap = {}
    for idx_pos in pos:
        src = idx_pos - lag
        if src < 0:
            continue
        v = col[src]
        if np.isfinite(v):
            smap[master[idx_pos].to_pydatetime()] = float(v)
    return smap

async def run_cell(name, cfg_kwargs, lag, slip):
    for j, c in enumerate(codes):
        frames[c].attrs["stock_code"] = c
        frames[c].attrs["from_precomputed"] = True
        frames[c].attrs[_SCORES_ATTR] = build_score_map(j, lag)
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
            initial_cash=1_000_000.0, commission_rate=0.0003, slippage_rate=slip,
            max_position_size=0.019, cash_reserve_ratio=0.05, board_lot_size=100,
            stop_loss_pct=0.0, take_profit_pct=0.0, rebalance_frequency="daily",
        )
        sc = {"topk": 50, "n_drop": 5, "hold_thresh": 0,
              "weights": {f: 1.0 for f in use}}
        sc.update(cfg_kwargs)
        await ex.run_backtest(strategy_name="factor_ranking",
                              stock_codes=codes, start_date=START, end_date=END,
                              strategy_config=sc, backtest_config=cfg)
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
    yr = {y: round((vs[-1] / vs[0] - 1) * 100, 1) for y, vs in sorted(yearly.items()) if len(vs) > 1}
    oos_mask = np.array([d < pd.Timestamp("2021-05-11") for d, _ in eq])
    oos = m._ret_stats(np.diff(vals[oos_mask]) / vals[oos_mask][:-1]) if oos_mask.sum() > 30 else {}
    print(f"[{name}] sharpe={stats['sharpe_ratio']:.3f} ret={stats['total_return']*100:.0f}% "
          f"mdd={stats['max_drawdown']*100:.1f}% oos_sharpe={oos.get('sharpe_ratio', float('nan')):.3f} "
          f"yearly_neg={[y for y, r in yr.items() if r < 0]}", flush=True)
    return {"name": name, "lag": lag, "slippage": slip,
            "stats": {k: float(v) for k, v in stats.items() if isinstance(v, (int, float))},
            "oos_stats": {k: float(v) for k, v in oos.items() if isinstance(v, (int, float))},
            "yearly": yr}

async def main():
    out = []
    out.append(await run_cell("lag1", {}, 1, 0.0001))       # 决定性检验
    out.append(await run_cell("slip2x", {}, 0, 0.0002))
    out.append(await run_cell("topk40", {"topk": 40}, 0, 0.0001))
    out.append(await run_cell("topk60", {"topk": 60}, 0, 0.0001))
    out.append(await run_cell("ndrop3", {"n_drop": 3}, 0, 0.0001))
    out.append(await run_cell("ndrop7", {"n_drop": 7}, 0, 0.0001))
    return out

cells = asyncio.run(main())
out = {"meta": {"window": [str(START.date()), str(END.date())],
                "n_factors": len(use), "generated_at": datetime.now().isoformat()},
       "cells": cells}
os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
with open(OUT_JSON, "w") as fh:
    json.dump(out, fh, ensure_ascii=False, indent=1, default=str)
print(f"[done] {time.time()-t0:.0f}s -> {OUT_JSON}", flush=True)
