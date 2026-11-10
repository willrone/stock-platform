"""Stacking 元模型 A/B: 等权42f vs ICIR vs Logistic vs LightGBM → 10y 回测 topk60.

元模型在 42f 因子 z 分数上学习 5 日前向收益, 按 (股票, 日) 月度滚动重训 (walk-forward, 无前视):
  - logistic: L2 岭回归, 学线性因子权重
  - lightgbm: 深度 3, 100 树, 能学因子交互/非线性
  面板因子缺失 (2016-2021) 时 z 填 0 并加 presence 哑变量, logistic 权重自动学出"缺失"语义.
对照 (已有): 等权42f topk60 Sharpe 0.976 / oos 0.607 / in 1.275
            ICIR 加权 0.485 (否决)
TopK60/Dropout5 执行端与 baseline 一致, 仅改打分.
输出: data/strategy_baseline/ic_composite_10y_stacking.json
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
OUT_JSON = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "strategy_baseline", "ic_composite_10y_stacking.json")
INTRA = {"KLEN", "KUP", "KLOW", "LOW0_a", "HIGH0_a", "RSQR5"}
LIMIT = 0.095
OOS_SPLIT = pd.Timestamp("2021-05-11")
FWD = 5

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
fwd_ret = (close.shift(-FWD) / close - 1.0).to_numpy()  # 5 日前向收益 (标签)

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

# 42 个因子原始矩阵 (n_d, n_s) + 是否面板因子
FACTOR_NAMES = list(seen.keys()) + list(F42.keys())
raw = []
is_panel = []
for fname, feats in seen.items():
    raw.append(cands[fname].to_numpy(dtype=np.float64))
    is_panel.append(False)
for fname, s in F42.items():
    raw.append(to_panel_mat(fname))
    is_panel.append(True)
n_fac = len(FACTOR_NAMES)
print(f"[factors] {n_fac} factors ({n_fac - len(F42)} base + {len(F42)} panel)", flush=True)

# 截面 z 分数 + presence
z_all = np.zeros((n_d, n_s, n_fac))
pre_all = np.zeros((n_d, n_s, n_fac))
for k in range(n_fac):
    mat = raw[k]
    ok = np.isfinite(mat)
    cnt = ok.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = np.nanmean(np.where(ok, mat, np.nan), axis=1, keepdims=True)
        sd = np.nanstd(np.where(ok, mat, np.nan), axis=1, keepdims=True)
        z = (mat - mu) / np.where(sd > 1e-12, sd, np.nan)
    z = np.where(cnt >= 30, z, 0.0)
    z = np.where(np.isfinite(z), z, 0.0)
    z_all[:, :, k] = z
    pre_all[:, :, k] = ok

# 等权42f (对照基准, 与研究符号一致)
acc = np.zeros((n_d, n_s)); wsum = np.zeros((n_d, n_s))
for (fname, feats), k in zip(list(seen.items()) + list(F42.items()), range(n_fac)):
    s = use[feats[0]]["sign"] if fname in seen else F42[fname]
    ok = np.isfinite(raw[k])
    acc[ok] += z_all[ok, k] * s
    wsum[ok] += 1
comp_eq = np.where(wsum > 0, acc / np.where(wsum > 0, wsum, 1), np.nan)
comp_eq[limit_up] = np.nan
print(f"[equi-weight] valid={np.isfinite(comp_eq).mean():.1%} {time.time()-t0:.0f}s", flush=True)

# ── walk-forward stacking: 按月滚动重训 ─────────────────────────────
# 月份切分: 每月第一个交易日; 训 t 月用 [start, t月前一天], 测 t 月
month_starts = []
prev = None
for k, ts in enumerate(master):
    key = (ts.year, ts.month)
    if key != prev:
        month_starts.append((k, ts))
        prev = key
warm = 2  # 前 2 个月无训练数据, 用等权
print(f"[stacking] {len(month_starts)} months, warmup={warm}", flush=True)

def make_feat_block(k0, k1):
    """[k0:k1] 的特征块 (N, n_fac*2) 和标签向量."""
    z = z_all[k0:k1].reshape(-1, n_fac)
    p = pre_all[k0:k1].reshape(-1, n_fac)
    lab = fwd_ret[k0:k1].reshape(-1)
    valid = np.isfinite(lab)
    X = np.column_stack([z[valid], p[valid]])
    return X, lab[valid]

def stack_run(model_kind):
    comp = np.full((n_d, n_s), np.nan)
    for mi, (k0, ts0) in enumerate(month_starts):
        k1 = month_starts[mi + 1][0] if mi + 1 < len(month_starts) else n_d
        if mi < warm:
            comp[k0:k1] = comp_eq[k0:k1]
            continue
        Xtr, ytr = make_feat_block(0, k0)
        if len(Xtr) < 500:
            comp[k0:k1] = comp_eq[k0:k1]
            continue
        if model_kind == "logistic":
            from sklearn.linear_model import Ridge
            mdl = Ridge(alpha=1.0).fit(Xtr, ytr)
            w = mdl.coef_
        elif model_kind == "gbm":
            import lightgbm as lgb
            mdl = lgb.LGBMRegressor(n_estimators=100, max_depth=3, learning_rate=0.05,
                                    num_leaves=15, subsample=0.8, colsample_bytree=0.8,
                                    random_state=42, verbosity=-1).fit(Xtr, ytr)
            w = None
        for k in range(k0, k1):
            # 当日全部股票各是一个样本: Xk = (n_s, 2*n_fac)
            Xk = np.column_stack([z_all[k], pre_all[k]])
            if model_kind == "logistic":
                score = Xk @ w
            else:
                score = mdl.predict(Xk)
            comp[k] = score
    comp[limit_up] = np.nan
    return comp

comp_log = stack_run("logistic")
comp_gbm = stack_run("gbm")
print(f"[stacking] logistic valid={np.isfinite(comp_log).mean():.1%} gbm valid={np.isfinite(comp_gbm).mean():.1%} {time.time()-t0:.0f}s", flush=True)

col_index = {}
for j, c in enumerate(codes):
    pos = master.get_indexer(pd.DatetimeIndex(frames[c].index))
    col_index[c] = pos[pos >= 0]

_orig_prepare = FactorRankingStrategy.prepare_backtest_data
_orig_snap = m.pma_module.PortfolioManagerArray.record_portfolio_snapshot

async def _noop_prepare(self, *a, **k):
    return None

async def run_cell(name, composite):
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
    out.append(await run_cell("equi42f", comp_eq))
    out.append(await run_cell("stack_log", comp_log))
    out.append(await run_cell("stack_gbm", comp_gbm))
    return out

cells = asyncio.run(main())
out = {"meta": {"window": [str(START.date()), str(END.date())],
                "n_factors": n_fac,
                "label": f"fwd{FWD} return",
                "note": "Stacking A/B: 等权42f vs Ridge-logistic vs LightGBM, walk-forward 月度重训 (无前视), presence 哑变量处理面板缺失",
                "generated_at": datetime.now().isoformat(),
                "elapsed_s": round(time.time() - t0)},
       "cells": cells}
os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
with open(OUT_JSON, "w") as fh:
    json.dump(out, fh, ensure_ascii=False, indent=1, default=str)
print(f"[done] {time.time()-t0:.0f}s -> {OUT_JSON}", flush=True)
