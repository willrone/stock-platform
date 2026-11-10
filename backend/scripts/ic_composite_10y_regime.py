"""Regime 门 A/B: 42f topk60 无门 vs 熊市清仓 vs 熊市减半 → 10y 回测.

Regime 门: 等权全市场指数 (所有票 close 均值) < MA200 → 熊市.
  - gate_cash: 熊市 → 全部现金 (所有分数置 NaN, 不持仓)
  - gate_half: 熊市 → 只持 top30 (同 mps, 仓位减半, 50% 现金)
  - 门用当日信息 (与 lag0 同义), MA200 需要 200 日前数据, 前 200 天无门
目标: 补 OOS (2016-2021 熊市) + 降回撤, 不伤窗内收益.
对照: 无门 42f topk60 = 0.976 / oos 0.607 / in 1.275 / mdd -37.8%
输出: data/strategy_baseline/ic_composite_10y_regime.json
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
OUT_JSON = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "strategy_baseline", "ic_composite_10y_regime.json")
INTRA = {"KLEN", "KUP", "KLOW", "LOW0_a", "HIGH0_a", "RSQR5"}
LIMIT = 0.095
OOS_SPLIT = pd.Timestamp("2021-05-11")
MA_WIN = 200

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
            pf = pf.reset_index().set_index("date").sort_index()
            pf = pf.loc[START:END]
            pf = pf[~pf.index.duplicated(keep="first")]
            panel[c] = pf
        except Exception:
            panel[c] = None
codes = list(frames)
n_s = len(codes)
master = pd.DatetimeIndex(sorted(set().union(*[set(f.index) for f in frames.values()])))
n_d = len(master)
print(f"[load] {n_s} stocks {n_d} days, panel={sum(1 for c in codes if panel.get(c) is not None)} {time.time()-t0:.0f}s", flush=True)

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

# 市场指数 = 等权 close 均值, MA200 之下为熊市
mkt = close.mean(axis=1, skipna=True)
ma200 = mkt.rolling(MA_WIN).mean()
bear = (mkt < ma200).to_numpy()
bear[~np.isfinite(ma200.to_numpy())] = False
print(f"[regime] bear days={bear.sum()} ({bear.mean():.0%}) {time.time()-t0:.0f}s", flush=True)

F42 = {
    "VSUMP5": -1.0, "VSUMP10": -1.0, "VSUMP20": -1.0,
    "VSUMN5": 1.0, "VSUMN10": 1.0, "VSUMN20": 1.0,
    "VSUMD5": -1.0, "VSUMD10": -1.0, "VSUMD20": -1.0,
    "QTLD5": 1.0, "QTLD10": 1.0, "QTLD20": 1.0,
    "IMAX5": -1.0, "IMIN5": -1.0, "WVMA5": -1.0,
    "VMA5": 1.0, "VSTD5": -1.0,
}
seen = {}
for feat, spec in use.items():
    seen.setdefault(spec["formula"], []).append(feat)

acc = np.zeros((n_d, n_s)); wsum = np.zeros((n_d, n_s))
def zscore_add(mat, s):
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

for fname, feats in seen.items():
    zscore_add(cands[fname].to_numpy(dtype=np.float64), use[feats[0]]["sign"])
for f, s in F42.items():
    zscore_add(to_panel_mat(f), s)
comp42 = np.where(wsum > 0, acc / np.where(wsum > 0, wsum, 1), np.nan)
comp42[limit_up] = np.nan
print(f"[composite] valid={np.isfinite(comp42).mean():.1%} {time.time()-t0:.0f}s", flush=True)

col_index = {}
for j, c in enumerate(codes):
    pos = master.get_indexer(pd.DatetimeIndex(frames[c].index))
    col_index[c] = pos[pos >= 0]

_orig_prepare = FactorRankingStrategy.prepare_backtest_data
_orig_snap = m.pma_module.PortfolioManagerArray.record_portfolio_snapshot

async def _noop_prepare(self, *a, **k):
    return None

async def run_cell(name, gate):
    """gate: None=无门, 'cash'=熊市清仓, 'half'=熊市减半(只持top30)."""
    for j, c in enumerate(codes):
        col = comp42[:, j]
        pos = col_index[c]
        smap = {}
        for idx_pos in pos:
            v = col[idx_pos]
            if not np.isfinite(v):
                continue
            if bear[idx_pos] and gate == "cash":
                continue  # 熊市全部清仓
            smap[master[idx_pos].to_pydatetime()] = float(v)
        frames[c].attrs["stock_code"] = c
        frames[c].attrs["from_precomputed"] = True
        frames[c].attrs[_SCORES_ATTR] = smap
    # half 门全局处理: 对每个交易日, 取全部票分数排序, 只保留前 30
    if gate == "half":
        for t in sorted(set().union(*[set(frames[c].attrs[_SCORES_ATTR]) for c in codes])):
            ts = pd.Timestamp(t)
            k = master.get_loc(ts)
            if not bear[k]:
                continue
            day_vals = []
            for c in codes:
                sm = frames[c].attrs[_SCORES_ATTR]
                if t in sm:
                    day_vals.append((sm[t], c))
            day_vals.sort(reverse=True)
            keep = {c for _, c in day_vals[:30]}
            for c in codes:
                sm = frames[c].attrs[_SCORES_ATTR]
                if t in sm and c not in keep:
                    del sm[t]
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
            max_position_size=round(0.95 / 60, 6), cash_reserve_ratio=0.05,
            board_lot_size=100, stop_loss_pct=0.0, take_profit_pct=0.0,
            rebalance_frequency="daily",
        )
        await ex.run_backtest(strategy_name="factor_ranking",
                              stock_codes=codes, start_date=START, end_date=END,
                              strategy_config={"topk": 60, "n_drop": 5, "hold_thresh": 0,
                                               "weights": {f: 1.0 for f in use}},
                              backtest_config=cfg)
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
            subs[sname] = {k: round(float(v), 4) for k, v in sr.items() if isinstance(v, (int, float))}
    print(f"[{name}] sharpe={stats['sharpe_ratio']:.3f} ret={stats['total_return']*100:.0f}% "
          f"mdd={stats['max_drawdown']*100:.1f}% oos={subs.get('oos_16_21',{}).get('sharpe_ratio'):.3f} "
          f"in={subs.get('in_21_26',{}).get('sharpe_ratio'):.3f}", flush=True)
    return {"name": name,
            "stats": {k: float(v) for k, v in stats.items() if isinstance(v, (int, float))},
            "subperiods": subs, "n_trades": len(pm.trades)}

async def main():
    out = []
    out.append(await run_cell("42f_nogate", None))
    out.append(await run_cell("42f_bearcash", "cash"))
    out.append(await run_cell("42f_bearhalf", "half"))
    return out

cells = asyncio.run(main())
out = {"meta": {"window": [str(START.date()), str(END.date())],
                "ma_window": MA_WIN,
                "note": "Regime 门 A/B: 42f topk60 无门 vs 熊市(MA200下)清仓 vs 熊市减半",
                "generated_at": datetime.now().isoformat(),
                "elapsed_s": round(time.time() - t0)},
       "cells": cells}
os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
with open(OUT_JSON, "w") as fh:
    json.dump(out, fh, ensure_ascii=False, indent=1, default=str)
print(f"[done] {time.time()-t0:.0f}s -> {OUT_JSON}", flush=True)
