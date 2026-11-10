"""自适应 topk: 按当日截面信号分散度动态调持仓数 → 10y 回测 4 cell.

C 方案: 固定 topk=60 把弱信号和强信号一起买进来.
  改为按当日截面分散度 (composite 的截面 std) 动态调 k:
    分散度低 → 头部与尾部差距小 → 只拿头部 (k 小, 仓位低, 防守)
    分散度高 → 排名可信 → 多拿 (k 大, 仓位满, 进攻)

实现 (不碰 executor, 用分数掩码技巧):
  1. 算 42f composite (同 baseline)
  2. 对每个交易日, 算截面 std = dispersion_day
  3. rolling_pctile = dispersion_day 在过去 60 天里的分位数 [0,1] (因果, 无前视)
  4. k_day = round(k_min + (k_max - k_min) * rolling_pctile), k_min=20 k_max=60
  5. 把当日 composite 排名 > k_day 的票分数置 NaN → executor (topk=60, mps=0.95/60)
     只会选中 k_day 只票, 每只仓位 = 0.95/60 组合市值 → 总仓位 = k_day/60 * 95%
     弱信号 (k 小) → 低仓位防守; 强信号 (k 大) → 满仓进攻. 信号强度仓位管理免费到手.

3 cell (ndrop=5, mps=0.95/60 固定, commission 0.03% + slippage 0.01%):
  fixed_k60   →  固定 60 (baseline, 预期 0.976)
  dyn_k20_60  →  k 在 20~60 之间随分散度动 (rolling-60d 分位数, 因果)
  dyn_k30_60  →  k 在 30~60 之间随分散度动 (更保守的下限)
参考点 fixed_k40 已有: expand2 五格 topk40 = 0.969.

输出: data/strategy_baseline/ic_composite_10y_dyntopk.json
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
OUT_JSON = os.path.join(os.path.dirname(BACKEND_ROOT), "data", "strategy_baseline", "ic_composite_10y_dyntopk.json")
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

# ── 自适应 k: 截面分散度 → rolling-60d 分位数 → k_day ─────────────────
# 每日截面 std (有效分数, 去 NaN)
disp = np.nanstd(np.where(np.isfinite(comp), comp, np.nan), axis=1)
disp_s = pd.Series(disp, index=master)
# rolling 60d 分位数: 当日分散度在过去 60 天里的排名 [0,1] (因果, 无前视)
roll_rank = disp_s.rolling(60, min_periods=20).apply(
    lambda w: (w[:-1] <= w[-1]).mean() if len(w) > 1 else 0.5, raw=True)
roll_rank = roll_rank.fillna(0.5)
print(f"[disp] std of dispersion: min={disp.min():.4f} max={disp.max():.4f} mean={disp.mean():.4f}", flush=True)
print(f"[roll_rank] min={roll_rank.min():.2f} max={roll_rank.max():.2f} mean={roll_rank.mean():.2f}", flush=True)

def k_of_day(roll_rank_val, k_min, k_max):
    return int(round(k_min + (k_max - k_min) * roll_rank_val))

# 生成各 cell 的 (date -> {code: score}) 掩码
def build_score_map(mask_k=None, k_min=20, k_max=60, const_rank=None):
    """mask_k: None=固定 60 (不掩码), 否则按 k_day 把排名>k_day 的票分数置 NaN.
    const_rank: 给定则忽略 roll_rank, k 恒为 k_of_day(const_rank). 用于验证 mask 机制."""
    maps = [{} for _ in range(n_s)]
    for d in range(n_d):
        row = comp[d]
        if mask_k is None:
            keep = np.isfinite(row)
        else:
            r = const_rank if const_rank is not None else roll_rank[d]
            k = k_of_day(r, k_min, k_max)
            order = np.argsort(-row, kind="stable")
            valid_order = order[np.isfinite(row[order])]
            keep = np.zeros(n_s, dtype=bool)
            if valid_order.size:
                keep[valid_order[:k]] = True
        for j in range(n_s):
            if keep[j]:
                maps[j][master[d].to_pydatetime()] = float(row[j])
    return maps

col_index = {}
for j, c in enumerate(codes):
    pos = master.get_indexer(pd.DatetimeIndex(frames[c].index))
    col_index[c] = pos[pos >= 0]

_orig_prepare = FactorRankingStrategy.prepare_backtest_data
_orig_snap = m.pma_module.PortfolioManagerArray.record_portfolio_snapshot

async def _noop_prepare(self, *a, **k):
    return None

async def run_cell(name, score_maps, fixed_topk=60, n_drop=5):
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
        # 仓位固定 0.95/60, 动态 k 通过掩码自然缩放总仓位
        cfg = BacktestConfig(
            initial_cash=1_000_000.0, commission_rate=0.0003, slippage_rate=0.0001,
            max_position_size=round(0.95 / 60, 6), cash_reserve_ratio=0.05,
            board_lot_size=100, stop_loss_pct=0.0, take_profit_pct=0.0,
            rebalance_frequency="daily",
        )
        await ex.run_backtest(
            strategy_name="factor_ranking",
            stock_codes=codes, start_date=START, end_date=END,
            strategy_config={"topk": fixed_topk, "n_drop": n_drop, "hold_thresh": 0,
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
            subs[sname] = {k2: round(float(v), 4) for k2, v in sr.items() if isinstance(v, (int, float))}
    print(f"[{name}] sharpe={stats['sharpe_ratio']:.3f} ret={stats['total_return']*100:.0f}% "
          f"mdd={stats['max_drawdown']*100:.1f}% oos={subs.get('oos_16_21',{}).get('sharpe_ratio',float('nan')):.3f} "
          f"in={subs.get('in_21_26',{}).get('sharpe_ratio',float('nan')):.3f}", flush=True)
    return {"name": name,
            "stats": {k2: float(v) for k2, v in stats.items() if isinstance(v, (int, float))},
            "subperiods": subs, "n_trades": len(pm.trades)}

maps_fixed = build_score_map(mask_k=None)              # 固定 60
maps_const40 = build_score_map(mask_k=1, k_min=20, k_max=60, const_rank=1.0/3)  # 恒定 k=40 验证 mask 机制
maps_dyn_2060 = build_score_map(mask_k=1, k_min=20, k_max=60)   # k 20~60
maps_dyn_3060 = build_score_map(mask_k=1, k_min=30, k_max=60)   # k 30~60
maps_dyn_2060_nd30 = maps_dyn_2060                       # 同掩码, 但 n_drop=30 即时跟上 k

async def main():
    out = []
    out.append(await run_cell("fixed_k60", maps_fixed, fixed_topk=60))
    out.append(await run_cell("const_k40", maps_const40, fixed_topk=60))   # 验证: 应复现 0.969
    out.append(await run_cell("dyn_k20_60", maps_dyn_2060, fixed_topk=60))
    out.append(await run_cell("dyn_k30_60", maps_dyn_3060, fixed_topk=60))
    out.append(await run_cell("dyn_k20_60_nd30", maps_dyn_2060_nd30, fixed_topk=60, n_drop=30))
    return out

cells = asyncio.run(main())
# k_day 分布统计
k_days = [k_of_day(roll_rank[d], 20, 60) for d in range(n_d)]
k_days_30 = [k_of_day(roll_rank[d], 30, 60) for d in range(n_d)]
out = {"meta": {
        "window": [str(START.date()), str(END.date())],
        "note": "C 自适应 topk: 按截面分散度 (rolling-60d 分位数) 动态调持仓数 20~60 / 30~60",
        "k2060_dist": {"min": min(k_days), "max": max(k_days),
                       "mean": round(float(np.mean(k_days)), 1),
                       "pct_at_min": round(sum(1 for k in k_days if k == 20) / n_d, 3),
                       "pct_at_max": round(sum(1 for k in k_days if k == 60) / n_d, 3)},
        "k3060_dist": {"min": min(k_days_30), "max": max(k_days_30),
                       "mean": round(float(np.mean(k_days_30)), 1),
                       "pct_at_min": round(sum(1 for k in k_days_30 if k == 30) / n_d, 3),
                       "pct_at_max": round(sum(1 for k in k_days_30 if k == 60) / n_d, 3)},
        "generated_at": datetime.now().isoformat(),
        "elapsed_s": round(time.time() - t0)},
       "cells": cells}
os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
with open(OUT_JSON, "w") as fh:
    json.dump(out, fh, ensure_ascii=False, indent=1, default=str)
print(f"[done] {time.time()-t0:.0f}s -> {OUT_JSON}", flush=True)
