"""ICIR 加权复合因子 → 10y 回测 (topk60).

基于诚实版等权复合 (ic_composite_10y_honest.py) 改造:
  等权: composite = sum_f(z_f * s_f) / n_factors          (s_f = 研究符号)
  ICIR: composite = sum_f(ICIR_f * z_f) / sum_f|ICIR_f|   (ICIR_f 带符号, 取绝对值归一)

ICIR 纪律 (避免前视):
  - RankIC[t,f] = 截面 spearman(因子[t,:], 前向收益[t,:])  (前向 = fwd5, 5 日收益)
  - ICIR[t,f]  = mean(RankIC[t-59..t,f]) / std(RankIC[t-59..t,f])  滚动 60 天窗口
  - 权重符号由 ICIR 自带 (因子与前向收益正/负相关), 不再乘研究符号 (避免符号双重施加)
  - 取绝对值做分母归一化, 保留各因子相对大小 (ICIR 0.15 的权重 = ICIR 0.05 的 3 倍)

TopK60 / Dropout5 配置保持不变 (与 baseline 一致), 仅改复合权重方式。
输出: data/strategy_baseline/ic_composite_10y_icir.json
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
OUT_JSON = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "strategy_baseline", "ic_composite_10y_icir.json")
INTRA = {"KLEN", "KUP", "KLOW", "LOW0_a", "HIGH0_a", "RSQR5"}
LIMIT = 0.095
OOS_SPLIT = pd.Timestamp("2021-05-11")
IC_WIN = 60          # 滚动 ICIR 窗口
IC_MINP = 20         # 最少样本
FWD = 5             # 前向收益周期 (fwd5)
TOPK = 60           # TopK 选股数 (与 baseline 一致)
N_DROP = 5          # Dropout 数 (与 baseline 一致)

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
        out[pos[ok], j] = pd.to_numeric(frames[c][col], errors="coerce").to_numpy(
            dtype=np.float64, na_value=np.nan)[ok]
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
limit_up = day_ret >= LIMIT  # 当日涨停 (T 收盘不可买入)

# 前向收益 (fwd5): close[t+5]/close[t] - 1
fwd = (close.shift(-FWD) / close - 1).to_numpy(dtype=np.float64)

seen = {}
for feat, spec in use.items():
    seen.setdefault(spec["formula"], []).append(feat)


# ---- 截面 RankIC (按行, 因子 vs 前向收益, spearman) ----
def cross_rank_ic(mat):
    a = pd.DataFrame(mat, index=master, columns=codes).rank(axis=1)
    b = pd.DataFrame(fwd, index=master, columns=codes).rank(axis=1)
    a_c = a.sub(a.mean(axis=1), axis=0)
    b_c = b.sub(b.mean(axis=1), axis=0)
    num = (a_c * b_c).sum(axis=1, skipna=True)
    den = np.sqrt((a_c ** 2).sum(axis=1, skipna=True) * (b_c ** 2).sum(axis=1, skipna=True))
    with np.errstate(invalid="ignore", divide="ignore"):
        ic = (num / den).to_numpy(dtype=np.float64)
    return ic


def rolling_icir(rank_ic, win=IC_WIN, minp=IC_MINP):
    s = pd.Series(rank_ic)
    mu = s.rolling(win, min_periods=minp).mean()
    sd = s.rolling(win, min_periods=minp).std()
    with np.errstate(invalid="ignore", divide="ignore"):
        return (mu / sd).to_numpy(dtype=np.float64)


# 每个 formula 的 (z_raw, signed ICIR)
formula_mats = {}
formula_icir = {}
for fname, feats in seen.items():
    mat = cands[fname].to_numpy(dtype=np.float64)
    ok = np.isfinite(mat)
    cnt = ok.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore"):
        mu = np.nanmean(np.where(ok, mat, np.nan), axis=1, keepdims=True)
        sd = np.nanstd(np.where(ok, mat, np.nan), axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        z_raw = (mat - mu) / np.where(sd > 1e-12, sd, np.nan)  # 不带符号 (符号由 ICIR 携带)
    z_raw = np.where(cnt >= 30, z_raw, np.nan)
    formula_mats[fname] = z_raw
    rank_ic = cross_rank_ic(mat)
    formula_icir[fname] = rolling_icir(rank_ic)
    print(f"[icir] {fname:8s} mean_rankIC={np.nanmean(rank_ic):+.4f} "
          f"icir_tail={formula_icir[fname][-1]:+.3f} "
          f"study_sign={use[feats[0]]['sign']:+.0f}", flush=True)


def make_composite(weight_mode):
    """weight_mode='eq' 等权(带研究符号) / 'icir' ICIR 加权."""
    acc = np.zeros((n_d, n_s))
    wsum = np.zeros((n_d, n_s))
    for fname in formula_mats:
        z = formula_mats[fname]
        if weight_mode == "eq":
            w = np.full(n_d, use[seen[fname][0]]["sign"])  # 等权: 带研究符号
        else:
            w = formula_icir[fname]  # 带符号 ICIR
        good = np.isfinite(z) & np.isfinite(w[:, None])
        term = np.where(good, np.nan_to_num(z) * w[:, None], 0.0)
        acc += term
        wsum += np.abs(w[:, None]) * good
    comp = np.where(wsum > 0, acc / np.where(wsum > 0, wsum, 1.0), np.nan)
    comp[limit_up] = np.nan  # 涨停掩码: 当日不可买
    return comp


composite_eq = make_composite("eq")
composite_icir = make_composite("icir")
print(f"[composite] eq_valid={np.isfinite(composite_eq).mean():.1%} "
      f"icir_valid={np.isfinite(composite_icir).mean():.1%} {time.time()-t0:.0f}s", flush=True)

col_index = {}
for j, c in enumerate(codes):
    pos = master.get_indexer(pd.DatetimeIndex(frames[c].index))
    col_index[c] = pos[pos >= 0]

_orig_prepare = FactorRankingStrategy.prepare_backtest_data
_orig_snap = m.pma_module.PortfolioManagerArray.record_portfolio_snapshot


async def _noop_prepare(self, *a, **k):
    return None


async def run_cell(name, composite, lag):
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
        # 单仓上限须与 topk 一致: topk × max_position_size ≈ (1 - cash_reserve),
        # 否则 topk>50 时 topk×0.019>95% → 现金不够 → 买单失败 → 组合冻结.
        topk_val = TOPK
        cfg = BacktestConfig(
            initial_cash=1_000_000.0, commission_rate=0.0003, slippage_rate=0.0001,
            max_position_size=round(0.95 / topk_val, 6), cash_reserve_ratio=0.05,
            board_lot_size=100, stop_loss_pct=0.0, take_profit_pct=0.0,
            rebalance_frequency="daily",
        )
        await ex.run_backtest(strategy_name="factor_ranking",
                              stock_codes=codes, start_date=START, end_date=END,
                              strategy_config={"topk": TOPK, "n_drop": N_DROP, "hold_thresh": 0,
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
          f"mdd={stats['max_drawdown']*100:.1f}% oos={subs.get('oos_16_21', {}).get('sharpe_ratio'):.3f} "
          f"in={subs.get('in_21_26', {}).get('sharpe_ratio'):.3f} yearly={yr}", flush=True)
    return {"name": name, "lag": lag,
            "stats": {k: float(v) for k, v in stats.items() if isinstance(v, (int, float))},
            "subperiods": subs, "yearly": yr, "n_trades": len(pm.trades)}


async def main():
    re = await run_cell("icir_eq_baseline_lag0", composite_eq, 0)
    ri = await run_cell("icir_icir_lag0", composite_icir, 0)
    return [re, ri]


cells = asyncio.run(main())
out = {"meta": {"window": [str(START.date()), str(END.date())],
                "n_factors": len(use),
                "n_unique_formulas": len(formula_mats),
                "excluded_intraday": sorted(INTRA),
                "limit_up_mask": LIMIT,
                "icir": {"window_days": IC_WIN, "min_periods": IC_MINP,
                         "forward": f"fwd{FWD}", "note": "ICIR 带符号, 滚动窗口, 截面 RankIC vs fwd5"},
                "topk": TOPK, "n_drop": N_DROP,
                "note": "ICIR 加权复合 vs 等权复合 (同 topk60/dropout5)",
                "generated_at": datetime.now().isoformat(),
                "elapsed_s": round(time.time() - t0)},
       "cells": cells}
os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
with open(OUT_JSON, "w") as fh:
    json.dump(out, fh, ensure_ascii=False, indent=1, default=str)
print(f"[done] {time.time()-t0:.0f}s -> {OUT_JSON}", flush=True)

# ---- 对比表 ----
eq = cells[0]
ic = cells[1]
print("\n===== 对比表 (topk60 / dropout5, 诚实版: 剔除内盘 + 涨停掩码) =====", flush=True)
print(f"{'配置':<24} {'Sharpe':>8} {'总收益':>8} {'回撤':>8} {'OOS夏普':>8} {'窗内夏普':>8}")
print(f"{'等权 baseline (lag0)':<24} {eq['stats']['sharpe_ratio']:>8.3f} "
      f"{eq['stats']['total_return']*100:>+7.0f}% {eq['stats']['max_drawdown']*100:>7.1f}% "
      f"{eq['subperiods']['oos_16_21']['sharpe_ratio']:>8.3f} "
      f"{eq['subperiods']['in_21_26']['sharpe_ratio']:>8.3f}")
print(f"{'ICIR 加权 (lag0)':<24} {ic['stats']['sharpe_ratio']:>8.3f} "
      f"{ic['stats']['total_return']*100:>+7.0f}% {ic['stats']['max_drawdown']*100:>7.1f}% "
      f"{ic['subperiods']['oos_16_21']['sharpe_ratio']:>8.3f} "
      f"{ic['subperiods']['in_21_26']['sharpe_ratio']:>8.3f}")
d = ic['stats']['sharpe_ratio'] - eq['stats']['sharpe_ratio']
print(f"\nSharpe 变化: {d:+.3f}  (目标 ≥ +0.05 即 ICIR Sharpe ≥ 0.72)", flush=True)
print(f"OOS 仍 > 0.6: {ic['subperiods']['oos_16_21']['sharpe_ratio'] > 0.6}", flush=True)
print(f"窗内 > 0.7: {ic['subperiods']['in_21_26']['sharpe_ratio'] > 0.7}", flush=True)
