"""B2 Horizon 对齐: 按因子窗口长短拆分复合 → 10y 回测 4 cell.

B2 假设: 42 因子 IC 集中在 fwd1, 但 topk60/ndrop5 持有多天.
  当前策略把所有因子混在一个 composite 里等权, 没有区分 horizon.
  B2: 拆成 short (n=5 窗) 和 long (n≥10 窗) 两组, 分别回测, 看哪组更强.

因子窗口拆分 (从公式名取 n):
  SHORT (n=5):  MIN5 MAX5 STD5 CORR5 CORD5 + VSUMP5 VSUMN5 VSUMD5 QTLD5 + IMAX5 IMIN5 WVMA5 VMA5 VSTD5
  LONG  (n≥10): MIN10 MAX10 STD10 CORR10 CORD10 + MIN20...CORR20 CORD20 + MIN30...CORD30 + MIN60...CORD60
               + VSUMP10 VSUMN10 VSUMD10 QTLD10 + VSUMP20 VSUMN20 VSUMD20 QTLD20

4 cell (topk=60, ndrop=5, 与当前注册版同参数):
  h_short_lag0  →  14 short factors, lag=0
  h_long_lag0   →  28 long factors, lag=0
  h_long_lag1   →  28 long factors, lag=1  (检验 lag 对长 horizon 的影响)
  baseline_42f  →  42f 等权, lag=0  (= 当前注册版, 预期 ~0.976)

输出: data/strategy_baseline/ic_composite_10y_horizon.json
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

BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
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

SURV_JSON = os.path.join(os.path.dirname(BACKEND_ROOT), "data", "strategy_baseline", "factor_ic_study_5y.json")
FORMULA_MAP = "/tmp/factor_formula_map.json"
PANEL_DIR = os.path.join(os.path.dirname(BACKEND_ROOT), "data", "qlib_data", "features", "day")
OUT_JSON = os.path.join(os.path.dirname(BACKEND_ROOT), "data", "strategy_baseline", "ic_composite_10y_horizon.json")
INTRA = {"KLEN", "KUP", "KLOW", "LOW0_a", "HIGH0_a", "RSQR5"}
LIMIT = 0.095
OOS_SPLIT = pd.Timestamp("2021-05-11")

t0 = time.time()
m.patch_loader()
codes = m.list_universe()
START, END = m.START, m.END

study = json.load(open(SURV_JSON))
fmap = json.load(open(FORMULA_MAP))
use = {
    c: {"formula": fmap[c]["formula"],
        "sign": 1.0 if study["results"][c]["fwd1"]["ic_mean"] > 0 else -1.0}
    for c in study["survivors"]
    if c in fmap and fmap[c]["formula"] not in INTRA
}
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

PANEL_SIGNS = {
    "VSUMP5": -1.0, "VSUMP10": -1.0, "VSUMP20": -1.0,
    "VSUMN5": 1.0, "VSUMN10": 1.0, "VSUMN20": 1.0,
    "VSUMD5": -1.0, "VSUMD10": -1.0, "VSUMD20": -1.0,
    "QTLD5": 1.0, "QTLD10": 1.0, "QTLD20": 1.0,
    "IMAX5": -1.0, "IMIN5": -1.0, "WVMA5": -1.0,
    "VMA5": 1.0, "VSTD5": -1.0,
}

# ── B2 核心: 按窗口 n 拆因子 ─────────────────────────────────────────
# 短 horizon (n=5): 对次日收益最敏感
SHORT_BASE = {
    "MIN5": use.get("MIN5", (use[[c for c in use if "MIN5" in use[c]["formula"]][0]] if any("MIN5" in use[c]["formula"] for c in use) else None)),
}
# 直接从 use dict 取: 按公式名的窗口数字分类
seen = {}
for feat, spec in use.items():
    seen.setdefault(spec["formula"], []).append(feat)

def _sign(fname):
    feats = seen.get(fname, [])
    if feats:
        return use[feats[0]]["sign"]
    return 1.0

BASE_FORMULAS = set(seen.keys())
SHORT_BASE_SET = {f for f in BASE_FORMULAS if f.endswith("5")}
LONG_BASE_SET = {f for f in BASE_FORMULAS if f.endswith(("10", "20", "30", "60"))}
SHORT_PANEL = ["VSUMP5", "VSUMN5", "VSUMD5", "QTLD5", "IMAX5", "IMIN5", "WVMA5", "VMA5", "VSTD5"]
LONG_PANEL = ["VSUMP10", "VSUMP20", "VSUMN10", "VSUMN20", "VSUMD10", "VSUMD20", "QTLD10", "QTLD20"]
ALL_BASE = SHORT_BASE_SET | LONG_BASE_SET
ALL_PANEL = SHORT_PANEL + LONG_PANEL

print(f"[horizon] short_base={sorted(SHORT_BASE_SET)} long_base={sorted(LONG_BASE_SET)}", flush=True)
print(f"[horizon] short_panel={SHORT_PANEL} long_panel={LONG_PANEL}", flush=True)
print(f"[horizon] total: short={len(SHORT_BASE_SET)+len(SHORT_PANEL)} long={len(LONG_BASE_SET)+len(LONG_PANEL)} all={len(ALL_BASE)+len(ALL_PANEL)}", flush=True)

def zscore_composite(base_set, panel_list):
    acc = np.zeros((n_d, n_s)); wsum = np.zeros((n_d, n_s))
    for fname in base_set:
        mat = cands[fname].to_numpy(dtype=np.float64)
        s = _sign(fname)
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
    for fname in panel_list:
        mat = to_panel_mat(fname)
        s = PANEL_SIGNS.get(fname, 1.0)
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
    comp = np.where(wsum > 0, acc / np.where(wsum > 0, wsum, 1), np.nan)
    comp[limit_up] = np.nan
    return comp

comp_short = zscore_composite(SHORT_BASE_SET, SHORT_PANEL)
comp_long = zscore_composite(LONG_BASE_SET, LONG_PANEL)
comp_all = zscore_composite(ALL_BASE, ALL_PANEL)

print(f"[composite] short={np.isfinite(comp_short).mean():.1%} long={np.isfinite(comp_long).mean():.1%} all={np.isfinite(comp_all).mean():.1%} {time.time()-t0:.0f}s", flush=True)

# 也验证: 当前注册版 = comp_all 应该 ≈ 0.976
# 检查 short/long 因子的 fwd1 vs fwd5 IC 分布
fwd1_ic = {}
fwd5_ic = {}
for c, spec in use.items():
    fname = spec["formula"]
    fwd1_ic[fname] = study["results"][c]["fwd1"]["ic_mean"]
    fwd5_ic[fname] = study["results"][c].get("fwd5", {}).get("ic_mean", float("nan"))

print("[fwd_ic] short:", {f: round(fwd1_ic.get(f, float('nan')), 4) for f in sorted(SHORT_BASE_SET)}, flush=True)
print("[fwd_ic] long:", {f: round(fwd1_ic.get(f, float('nan')), 4) for f in sorted(LONG_BASE_SET)}, flush=True)

col_index = {}
for j, c in enumerate(codes):
    pos = master.get_indexer(pd.DatetimeIndex(frames[c].index))
    col_index[c] = pos[pos >= 0]

_orig_prepare = FactorRankingStrategy.prepare_backtest_data
_orig_snap = m.pma_module.PortfolioManagerArray.record_portfolio_snapshot

async def _noop_prepare(self, *a, **k):
    return None

async def run_cell(name, composite, lag=0):
    # 应用 lag
    if lag > 0:
        lagged = np.full_like(composite, np.nan)
        if lag < n_d:
            lagged[lag:] = composite[: n_d - lag]
        composite = lagged
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
        cfg = BacktestConfig(
            initial_cash=1_000_000.0, commission_rate=0.0003, slippage_rate=0.0001,
            max_position_size=round(0.95 / 60, 6), cash_reserve_ratio=0.05,
            board_lot_size=100, stop_loss_pct=0.0, take_profit_pct=0.0,
            rebalance_frequency="daily",
        )
        await ex.run_backtest(
            strategy_name="factor_ranking",
            stock_codes=codes, start_date=START, end_date=END,
            strategy_config={"topk": 60, "n_drop": 5, "hold_thresh": 0,
                            "weights": {f: 1.0 for f in use}, "lag": lag},
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
          f"mdd={stats['max_drawdown']*100:.1f}% oos={subs.get('oos_16_21',{}).get('sharpe_ratio',float('nan')):.3f} "
          f"in={subs.get('in_21_26',{}).get('sharpe_ratio',float('nan')):.3f}", flush=True)
    return {"name": name, "lag": lag,
            "stats": {k: float(v) for k, v in stats.items() if isinstance(v, (int, float))},
            "subperiods": subs, "n_trades": len(pm.trades)}

async def main():
    out = []
    out.append(await run_cell("h_short_lag0", comp_short, lag=0))
    out.append(await run_cell("h_long_lag0", comp_long, lag=0))
    out.append(await run_cell("h_long_lag1", comp_long, lag=1))
    out.append(await run_cell("baseline_42f_lag0", comp_all, lag=0))
    return out

cells = asyncio.run(main())
out = {"meta": {
        "window": [str(START.date()), str(END.date())],
        "short_base": sorted(SHORT_BASE_SET), "long_base": sorted(LONG_BASE_SET),
        "short_panel": SHORT_PANEL, "long_panel": LONG_PANEL,
        "note": "B2 horizon 对齐: 短窗因子(n=5) vs 长窗因子(n>=10) vs 全因子等权",
        "generated_at": datetime.now().isoformat(),
        "elapsed_s": round(time.time() - t0)},
       "cells": cells}
os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
with open(OUT_JSON, "w") as fh:
    json.dump(out, fh, ensure_ascii=False, indent=1, default=str)
print(f"[done] {time.time()-t0:.0f}s -> {OUT_JSON}", flush=True)
