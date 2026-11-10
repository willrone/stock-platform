"""43-factor composite (25 existing + 18 Alpha158 panel factors) → 10y backtest, 5 cells.

New factors from qlib Alpha158 panel (../data/qlib_data/features/day/):
  VSUMP5/10/20, VSUMN5/10/20, VSUMD5/10/20,
  QTLD5/10/20, IMAX5, IMIN5, WVMA5, VMA5, VSTD5
All 18 exist with exact names (verified).

5 cells: topk40, topk50, topk60, ndrop3, ndrop7 (ndrop uses topk60).
Baseline: 25-factor equal-weight zscore, same 5 cells.

Output: data/strategy_baseline/ic_composite_10y_expanded.json
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

BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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

SURV_JSON = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "strategy_baseline", "factor_ic_study_5y.json")
FORMULA_MAP = "/tmp/factor_formula_map.json"
PANEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "qlib_data", "features", "day")
OUT_JSON = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "strategy_baseline", "ic_composite_10y_expanded.json")
INTRA = {"KLEN", "KUP", "KLOW", "LOW0_a", "HIGH0_a", "RSQR5"}
LIMIT = 0.095
OOS_SPLIT = pd.Timestamp("2021-05-11")

t0 = time.time()
m.patch_loader()
codes = m.list_universe()
START, END = m.START, m.END

study = json.load(open(SURV_JSON))
fmap = json.load(open(FORMULA_MAP))
use = {c: {"formula": fmap[c]["formula"],
           "sign": (1.0 if study["results"][c]["fwd1"]["ic_mean"] > 0 else -1.0)}
       for c in study["survivors"] if c in fmap and fmap[c]["formula"] not in INTRA}
print(f"[setup] {len(codes)} stocks, {len(use)} honest factors", flush=True)

frames = {}
panel = {}
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
        try:
            pf = pd.read_parquet(os.path.join(PANEL_DIR, c.replace(".", "_") + ".parquet"))
            # MultiIndex (stock_code, date) → make date the index
            pf = pf.reset_index()
            pf = pf.set_index("date").sort_index()
            pf = pf.loc[START:END]
            pf = pf[~pf.index.duplicated(keep="first")]
            panel[c] = pf
        except Exception:
            panel[c] = None
codes = list(frames)
n_s = len(codes)
master = pd.DatetimeIndex(sorted(set().union(*[set(f.index) for f in frames.values()])))
n_d = len(master)
n_panel = sum(1 for c in codes if panel.get(c) is not None)
print(f"[load] {n_s} stocks {n_d} days, {n_panel} panel {time.time()-t0:.0f}s", flush=True)

def to_mat(col):
    out = np.full((n_d, n_s), np.nan)
    for j, c in enumerate(codes):
        pos = master.get_indexer(pd.DatetimeIndex(frames[c].index))
        ok = pos >= 0
        out[pos[ok], j] = pd.to_numeric(frames[c][col], errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)[ok]
    return out

def to_panel_mat(col):
    out = np.full((n_d, n_s), np.nan)
    for j, c in enumerate(codes):
        if panel.get(c) is None:
            continue
        pos = master.get_indexer(pd.DatetimeIndex(panel[c].index))
        ok = pos >= 0
        out[pos[ok], j] = pd.to_numeric(panel[c][col], errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)[ok]
    return out

close = pd.DataFrame(to_mat("close"), index=master)
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

day_ret = close.pct_change(fill_method=None).to_numpy()
limit_up = day_ret >= LIMIT

# 18 new panel factors with their signs (from study fwd1 IC)
# 9 families x 2 windows (5, 10)
NEW_FACTORS = {
    "VSUMP5": -1.0, "VSUMP10": -1.0,
    "VSUMN5": 1.0, "VSUMN10": 1.0,
    "VSUMD5": -1.0, "VSUMD10": -1.0,
    "QTLD5": 1.0, "QTLD10": 1.0,
    "IMAX5": -1.0, "IMAX10": -1.0,
    "IMIN5": -1.0, "IMIN10": 1.0,
    "WVMA5": -1.0, "WVMA10": -1.0,
    "VMA5": 1.0, "VMA10": 1.0,
    "VSTD5": -1.0, "VSTD10": -1.0,
}

# --- 25-factor composite (baseline) ---
seen = {}
for feat, spec in use.items():
    seen.setdefault(spec["formula"], []).append(feat)

def zscore_composite(factor_mats):
    """factor_mats: dict formula->(matrix, sign). Returns composite array."""
    acc = np.zeros((n_d, n_s))
    wsum = np.zeros((n_d, n_s))
    for fname, (mat, s) in factor_mats.items():
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
    return np.where(wsum > 0, acc / np.where(wsum > 0, wsum, 1), np.nan)

# 25-factor: 25 formulas from cands
f25 = {fname: (cands[fname].to_numpy(dtype=np.float64), use[feats[0]]["sign"])
       for fname, feats in seen.items()}
comp25 = zscore_composite(f25)
comp25[limit_up] = np.nan
print(f"[25f] valid={np.isfinite(comp25).mean():.1%} {time.time()-t0:.0f}s", flush=True)

# 43-factor: 25 + 18 new
f43 = dict(f25)
for fname, sign in NEW_FACTORS.items():
    f43[fname] = (to_panel_mat(fname), sign)
comp43 = zscore_composite(f43)
comp43[limit_up] = np.nan
print(f"[43f] valid={np.isfinite(comp43).mean():.1%} {time.time()-t0:.0f}s", flush=True)

col_index = {}
for j, c in enumerate(codes):
    pos = master.get_indexer(pd.DatetimeIndex(frames[c].index))
    col_index[c] = pos[pos >= 0]

_orig_prepare = FactorRankingStrategy.prepare_backtest_data
_orig_snap = m.pma_module.PortfolioManagerArray.record_portfolio_snapshot

async def _noop_prepare(self, *a, **k):
    return None

async def run_cell(name, composite, n_factors, extra=None):
    for j, c in enumerate(codes):
        col = composite[:, j]
        pos = col_index[c]
        smap = {}
        for idx_pos in pos:
            v = col[idx_pos]
            if np.isfinite(v):
                smap[master[idx_pos].to_pydatetime()] = float(v)
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
        topk_val = (extra or {}).get("topk", 60)
        cfg = BacktestConfig(
            initial_cash=1_000_000.0, commission_rate=0.0003, slippage_rate=0.0001,
            max_position_size=round(0.95 / topk_val, 6), cash_reserve_ratio=0.05,
            board_lot_size=100, stop_loss_pct=0.0, take_profit_pct=0.0,
            rebalance_frequency="daily",
        )
        await ex.run_backtest(strategy_name="factor_ranking",
                              stock_codes=codes, start_date=START, end_date=END,
                              strategy_config={**{"topk": 60, "n_drop": 5, "hold_thresh": 0,
                                               "weights": {f: 1.0 for f in use}}, **(extra or {})},
                              backtest_config=cfg)
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
    subs = {}
    for sname, lo, hi in [("full", START, END), ("oos_16_21", START, OOS_SPLIT),
                          ("in_21_26", OOS_SPLIT, END)]:
        mask = np.array([(d >= lo) and (d <= hi) for d, _ in eq])
        if mask.sum() > 30:
            sv = vals[mask]
            sr = m._ret_stats(np.diff(sv) / sv[:-1])
            subs[sname] = {k: round(float(v), 4) for k, v in sr.items() if isinstance(v, (int, float))}
    print(f"[{name}] sharpe={stats['sharpe_ratio']:.3f} ret={stats['total_return']*100:.0f}% "
          f"mdd={stats['max_drawdown']*100:.1f}% oos={subs.get('oos_16_21',{}).get('sharpe_ratio'):.3f} "
          f"in={subs.get('in_21_26',{}).get('sharpe_ratio'):.3f} n_trades={len(pm.trades)}", flush=True)
    return {"name": name, "n_factors": n_factors,
            "stats": {k: float(v) for k, v in stats.items() if isinstance(v, (int, float))},
            "subperiods": subs, "yearly": yr, "n_trades": len(pm.trades)}

CELLS = [
    ("topk40", {"topk": 40}),
    ("topk50", {"topk": 50}),
    ("topk60", {"topk": 60}),
    ("ndrop3", {"n_drop": 3}),
    ("ndrop7", {"n_drop": 7}),
]

async def main():
    out = []
    for label, extra in CELLS:
        out.append(await run_cell(f"25_{label}", comp25, 25, extra))
        out.append(await run_cell(f"43_{label}", comp43, 43, extra))
    return out

cells = asyncio.run(main())
out = {"meta": {"window": [str(START.date()), str(END.date())],
                "n_factors_25": 25, "n_factors_43": 43,
                "new_factors": sorted(NEW_FACTORS.keys()),
                "limit_up_mask": LIMIT,
                "note": "25-factor baseline vs 43-factor expanded (18 Alpha158 panel factors added)",
                "generated_at": datetime.now().isoformat(),
                "elapsed_s": round(time.time() - t0)},
       "cells": cells}
os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
with open(OUT_JSON, "w") as fh:
    json.dump(out, fh, ensure_ascii=False, indent=1, default=str)
print(f"[done] {time.time()-t0:.0f}s -> {OUT_JSON}", flush=True)

# Comparison table
print("\n" + "=" * 90)
print(f"{'cell':<10} {'n_f':>4} {'sharpe':>8} {'total_ret':>10} {'mdd':>8} {'oos_sharpe':>12} {'in_sharpe':>10}")
print("=" * 90)
for cell in cells:
    s = cell["stats"]
    oos = cell["subperiods"].get("oos_16_21", {}).get("sharpe_ratio", 0)
    inn = cell["subperiods"].get("in_21_26", {}).get("sharpe_ratio", 0)
    label = cell["name"].split("_", 1)[0]
    cl = cell["name"].split("_", 1)[1]
    print(f"{cl:<10} {cell['n_factors']:>4} {s['sharpe_ratio']:>8.3f} {s['total_return']*100:>9.0f}% {s['max_drawdown']*100:>7.1f}% {oos:>12.3f} {inn:>10.3f}")
print("=" * 90)
