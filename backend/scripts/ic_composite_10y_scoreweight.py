"""C 按分数加权: 每只票的买入金额 ∝ z分数 (不是等权) → 10y 回测 3 cell.

之前 dyn_k 崩到 0.18 是掩码只能控制"选不选", 控制不了"买多少".
真正分数加权 = 每只票买入金额 ∝ z分数 (归一化到平均 1):
  z=2 的票买 2 份, z=0.5 的票买 0.5 份.
  平均仓位不变 (还是 95% 可投), 但向高信号票集中.

实现 (monkey-patch _execute_buy 一行):
  available_cash_for_stock = min(max_position_value × weight_i, reserve_cash)
  weight_i = z_i / mean(z_topk)  (归一化到平均 1, 保证平均仓位不变)

3 cell (ndrop=5, mps=0.95/60, commission 0.03% + slippage 0.01%):
  eq        →  等权 (baseline, 预期 0.976)
  z_lin     →  weight_i = z_i / mean(z), 线性 z 加权
  z_sq      →  weight_i = z_i² / mean(z²), 平方 z 加权 (更集中于高分票)

权重只对 topk60 里的票归一化 (mean=1), 不影响总仓位.
输出: data/strategy_baseline/ic_composite_10y_scoreweight.json
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
OUT_JSON = os.path.join(os.path.dirname(BACKEND_ROOT), "data", "strategy_baseline", "ic_composite_10y_scoreweight.json")
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

seen = {}
for feat, spec in use.items():
    seen.setdefault(spec["formula"], []).append(feat)
def _sign(fname):
    feats = seen.get(fname, [])
    return use[feats[0]]["sign"] if feats else 1.0
BASE_FORMULAS = set(seen.keys())
ALL_PANEL = ["VSUMP5", "VSUMP10", "VSUMP20", "VSUMN5", "VSUMN10", "VSUMN20",
             "VSUMD5", "VSUMD10", "VSUMD20", "QTLD5", "QTLD10", "QTLD20",
             "IMAX5", "IMIN5", "WVMA5", "VMA5", "VSTD5"]

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
        acc[good] += np.nan_to_num(z)[good]; wsum[good] += 1
    for fname in panel_list:
        mat = to_panel_mat(fname); s = PANEL_SIGNS.get(fname, 1.0)
        ok = np.isfinite(mat)
        cnt = ok.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore"):
            mu = np.nanmean(np.where(ok, mat, np.nan), axis=1, keepdims=True)
            sd = np.nanstd(np.where(ok, mat, np.nan), axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            z = (mat - mu) / np.where(sd > 1e-12, sd, np.nan) * s
        z = np.where(cnt >= 30, z, np.nan)
        good = np.isfinite(z)
        acc[good] += np.nan_to_num(z)[good]; wsum[good] += 1
    comp = np.where(wsum > 0, acc / np.where(wsum > 0, wsum, 1), np.nan)
    comp[limit_up] = np.nan
    return comp

comp = zscore_composite(BASE_FORMULAS, ALL_PANEL)
print(f"[composite] 42f={np.isfinite(comp).mean():.1%} {time.time()-t0:.0f}s", flush=True)

# 权重矩阵: 对 topk60 内的票, weight_i = z_i / mean(z_topk) (归一化到平均 1)
# z_lin:  用 z 本身 (可能负 → 取 max(z, 0) 防止负权重)
# z_sq:   用 z² (全正, 更集中于高分票)
def build_weight_maps(weight_type):
    """返回 [date -> {code: weight}] 的 list. weight 归一化到 top60 平均 1."""
    maps = [{} for _ in range(n_s)]
    for d in range(n_d):
        row = comp[d]
        if not np.isfinite(row).any():
            continue
        order = np.argsort(-row, kind="stable")
        valid = order[np.isfinite(row[order])]
        top = valid[:60]
        if not top.size:
            continue
        z = row[top]
        if weight_type == "z_lin":
            w = np.clip(z, 0.05, None)          # 防负权重 (z 可能负)
        else:  # z_sq
            w = z ** 2
        w = w / w.mean()                          # 归一化到平均 1
        for idx, j in enumerate(top):
            maps[j][master[d].to_pydatetime()] = float(w[idx])
    return maps

# monkey-patch _execute_buy: 用当日权重缩放 max_position_value
# _weight_by_day: {date_pydatetime: {code: weight}}
_weight_by_day: dict = {}
_orig_execute_buy = m.pma_module.PortfolioManagerArray._execute_buy

def _weighted_execute_buy(self, idx, stock_code, price, original_price, slippage_cost_per_share, signal, current_prices):
    day_w = _weight_by_day.get(signal.timestamp, {})
    w = day_w.get(stock_code, 1.0)
    old_mps = self.config.max_position_size
    self.config.max_position_size = old_mps * w
    try:
        return _orig_execute_buy(self, idx, stock_code, price, original_price,
                                slippage_cost_per_share, signal, current_prices)
    finally:
        self.config.max_position_size = old_mps

col_index = {}
for j, c in enumerate(codes):
    pos = master.get_indexer(pd.DatetimeIndex(frames[c].index))
    col_index[c] = pos[pos >= 0]

_orig_prepare = FactorRankingStrategy.prepare_backtest_data
_orig_snap = m.pma_module.PortfolioManagerArray.record_portfolio_snapshot

async def _noop_prepare(self, *a, **k):
    return None

async def run_cell(name, score_maps, weight_maps=None):
    """score_maps: [date -> {code: score}] for ranking.
    weight_maps: [date -> {code: weight}] for position sizing. None = 等权."""
    global _weight_by_day
    _weight_by_day = {}
    if weight_maps is not None:
        for d in range(n_d):
            dt = master[d].to_pydatetime()
            day_w = {}
            for j in range(n_s):
                w = weight_maps[j].get(dt)
                if w is not None:
                    day_w[codes[j]] = w
            if day_w:
                _weight_by_day[dt] = day_w

    # 让 executor 知道当前日期, 从而取当天权重
    m.pma_module.PortfolioManagerArray._execute_buy = _weighted_execute_buy

    for j, c in enumerate(codes):
        frames[c].attrs["stock_code"] = c
        frames[c].attrs["from_precomputed"] = True
        frames[c].attrs[_SCORES_ATTR] = score_maps[j]
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
                            "weights": {f: 1.0 for f in use}},
            backtest_config=cfg)
    finally:
        m.pma_module.PortfolioManagerArray._execute_buy = _orig_execute_buy
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
            subs[sname] = {k2: round(float(v), 4) for k2, v in sr.items() if isinstance(v, (int, float))}
    print(f"[{name}] sharpe={stats['sharpe_ratio']:.3f} ret={stats['total_return']*100:.0f}% "
          f"mdd={stats['max_drawdown']*100:.1f}% oos={subs.get('oos_16_21',{}).get('sharpe_ratio',float('nan')):.3f} "
          f"in={subs.get('in_21_26',{}).get('sharpe_ratio',float('nan')):.3f}", flush=True)
    return {"name": name,
            "stats": {k2: float(v) for k2, v in stats.items() if isinstance(v, (int, float))},
            "subperiods": subs, "n_trades": len(pm.trades)}

# 等权 score_maps
maps_eq = [{} for _ in range(n_s)]
for j in range(n_s):
    pos = col_index[codes[j]]
    for idx_pos in pos:
        v = comp[idx_pos, j]
        if np.isfinite(v):
            maps_eq[j][master[idx_pos].to_pydatetime()] = float(v)

# 加权 score_maps
maps_zlin = build_weight_maps("z_lin")
maps_zsq = build_weight_maps("z_sq")

async def main():
    out = []
    out.append(await run_cell("eq_baseline", maps_eq, weight_maps=None))   # 等权, 预期 0.976
    out.append(await run_cell("z_lin", maps_eq, weight_maps=maps_zlin))   # z 加权
    out.append(await run_cell("z_sq", maps_eq, weight_maps=maps_zsq))     # z² 加权
    return out

cells = asyncio.run(main())
out = {"meta": {
        "window": [str(START.date()), str(END.date())],
        "note": "C 按分数加权: 买入金额 ∝ z分数 (归一化到 top60 平均 1), 对比等权",
        "generated_at": datetime.now().isoformat(),
        "elapsed_s": round(time.time() - t0)},
       "cells": cells}
os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
with open(OUT_JSON, "w") as fh:
    json.dump(out, fh, ensure_ascii=False, indent=1, default=str)
print(f"[done] {time.time()-t0:.0f}s -> {OUT_JSON}", flush=True)
