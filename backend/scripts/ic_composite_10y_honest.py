"""诚实版复合因子 → 10y 回测 (诊断8 结论落地).

诊断8 发现: v2 复合的头条 Sharpe 1.75 大部分是"当日涨停锁死"伪迹 —
  内盘因子 (HIGH0=log(high/close)/KLEN/KUP/KLOW/LOW0/RSQR5) 的 top50 全是当日涨停票,
  排除当日涨停后 HIGH0 +161.7%→-16.1%, KLEN +118.2%→+2.5%, MAX 族 -30~-59%。
  这些票在收盘价根本买不进 (涨停锁死无卖盘), 引擎未建模锁死 → 伪收益。

本版三条纪律:
  1. 剔除全部内盘 OHLC 因子 (KLEN/KUP/KLOW/LOW0/HIGH0/RSQR5)
  2. 涨停掩码: 当日涨幅 >= 9.5% 的票分数置 NaN (当日不可在收盘买入)
  3. 保留 close/volume 族 (MIN/STD/CORR/CORD/MAX), MAX 亦被掩码约束
同时跑 lag0 / lag1 两变体。
输出: data/strategy_baseline/ic_composite_10y_honest.json
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
OUT_JSON = "/Users/rone/projects/stock-platform/data/strategy_baseline/ic_composite_10y_honest.json"
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
print(f"[setup] {len(codes)} stocks, {len(use)} honest factors (intraday excluded)", flush=True)

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
limit_up = day_ret >= LIMIT  # 当日涨停 (T 收盘不可买入)

seen = {}
for feat, spec in use.items():
    seen.setdefault(spec["formula"], []).append(feat)

acc = np.zeros((n_d, n_s)); wsum = np.zeros((n_d, n_s))
for fname, feats in seen.items():
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
composite[limit_up] = np.nan  # 涨停掩码: 当日不可买
print(f"[composite] valid={np.isfinite(composite).mean():.1%} {time.time()-t0:.0f}s", flush=True)

col_index = {}
for j, c in enumerate(codes):
    pos = master.get_indexer(pd.DatetimeIndex(frames[c].index))
    col_index[c] = pos[pos >= 0]

_orig_prepare = FactorRankingStrategy.prepare_backtest_data
_orig_snap = m.pma_module.PortfolioManagerArray.record_portfolio_snapshot

async def _noop_prepare(self, *a, **k):
    return None

async def run_cell(name, lag):
    for j, c in enumerate(codes):
        col = composite[:, j]
        pos = col_index[c]
        smap = {}
        for idx_pos in pos:
            src = idx_pos - lag
            if src < 0:
                continue
            v = col[src]
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
        cfg = BacktestConfig(
            initial_cash=1_000_000.0, commission_rate=0.0003, slippage_rate=0.0001,
            max_position_size=0.019, cash_reserve_ratio=0.05, board_lot_size=100,
            stop_loss_pct=0.0, take_profit_pct=0.0, rebalance_frequency="daily",
        )
        await ex.run_backtest(strategy_name="factor_ranking",
                              stock_codes=codes, start_date=START, end_date=END,
                              strategy_config={"topk": 50, "n_drop": 5, "hold_thresh": 0,
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
          f"in={subs.get('in_21_26',{}).get('sharpe_ratio'):.3f} yearly={yr}", flush=True)
    return {"name": name, "lag": lag,
            "stats": {k: float(v) for k, v in stats.items() if isinstance(v, (int, float))},
            "subperiods": subs, "yearly": yr, "n_trades": len(pm.trades)}

async def main():
    r0 = await run_cell("honest_lag0", 0)
    r1 = await run_cell("honest_lag1", 1)
    return [r0, r1]

cells = asyncio.run(main())
out = {"meta": {"window": [str(START.date()), str(END.date())],
                "n_factors": len(use),
                "excluded_intraday": sorted(INTRA),
                "limit_up_mask": LIMIT,
                "note": "诚实版: 剔除内盘因子 + 当日涨停不可买掩码",
                "generated_at": datetime.now().isoformat(),
                "elapsed_s": round(time.time() - t0)},
       "cells": cells}
os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
with open(OUT_JSON, "w") as fh:
    json.dump(out, fh, ensure_ascii=False, indent=1, default=str)
print(f"[done] {time.time()-t0:.0f}s -> {OUT_JSON}", flush=True)
