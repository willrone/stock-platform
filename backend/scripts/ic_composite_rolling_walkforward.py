#!/usr/bin/env python3
"""滚动 walk-forward: 每年用过去 5 年重估 IC + 符号, 下一年验证.

与单次 split 的区别:
  - 单次: 2016-2021 选一次 → 2021-2026 一条曲线 (只有一个 OOS 期)
  - 滚动: 2021 用 2016-2020 选, 2022 用 2017-2021 选, ..., 2026 用 2021-2025 选
    每年重新选因子 + 重估符号, 拼成 2021-2026 连续 OOS 曲线
这检验的是"滚动重估"这个流程本身有没有 edge, 而不是某个固定因子表。

三个单元 (topk30/ndrop3, 周频周一 + lag1 + 实盘成本):
  rolling_wf   每年重选因子+符号 (真滚动)
  fixed_22f    2016-2021 选一次, 固定用到 2026 (单次 split 的对照)
  all23        23 公式全上, 符号也按当年训练窗估 (无因子选择, 只滚动符号)

输出: data/strategy_baseline/ic_composite_rolling_walkforward.json
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

BACKEND_ROOT = "/Users/rone/Projects/stock-platform/backend"
sys.path.insert(0, BACKEND_ROOT)

_spec = importlib.util.spec_from_file_location(
    "ds", os.path.join(BACKEND_ROOT, "scripts", "daily_signal.py"))
ds = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ds)

_spec2 = importlib.util.spec_from_file_location(
    "xleg", os.path.join(BACKEND_ROOT, "scripts", "expanded_legs_10y.py"))
m = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(m)

from app.core.config import settings  # noqa: E402
from app.services.backtest import BacktestExecutor  # noqa: E402
from app.services.backtest.models import BacktestConfig  # noqa: E402
from app.services.backtest.strategies.factor_ranking_strategy import (  # noqa: E402
    FactorRankingStrategy, _SCORES_ATTR)

OUT_JSON = "/Users/rone/Projects/stock-platform/data/strategy_baseline/ic_composite_rolling_walkforward.json"
START, END = m.START, m.END
TEST_START = pd.Timestamp("2021-01-01")
TOPK, NDROP = 30, 3

t0 = time.time()
m.patch_loader()
frames_all, panel, master_all = ds.load_data(m.list_universe())
print(f"[load] {len(frames_all)} stocks {len(master_all)} days {time.time()-t0:.0f}s", flush=True)

frames = {}
for c, f in frames_all.items():
    x = f.loc[START:END]
    if len(x) > 60:
        x = x[~x.index.duplicated(keep="first")]
        x.attrs["stock_code"] = c
        x.attrs["from_precomputed"] = True
        frames[c] = x
codes = list(frames)
master = pd.DatetimeIndex(sorted(set().union(*[set(f.index) for f in frames.values()])))
print(f"[window] {len(codes)} stocks {len(master)} days {time.time()-t0:.0f}s", flush=True)

close = ds.to_mat(frames, master, "close")
vol = ds.to_mat(frames, master, "volume")
with np.errstate(divide="ignore", invalid="ignore"):
    ret = close.pct_change(fill_method=None)
    vchg = vol.pct_change(fill_method=None)
    cands: dict[str, pd.DataFrame] = {}
    for n in (5, 10, 20, 30, 60):
        cands[f"MIN{n}"] = close.rolling(n).min() / close - 1.0
        cands[f"MAX{n}"] = close.rolling(n).max() / close - 1.0
        cands[f"STD{n}"] = close.rolling(n).std() / close
        cands[f"CORR{n}"] = close.rolling(n).corr(vol)
        cands[f"CORD{n}"] = ret.rolling(n).corr(vchg)
fwd5 = close.shift(-5) / close - 1.0
limit_up = close.pct_change(fill_method=None).to_numpy() >= ds.LIMIT_UP
print(f"[factors] {len(cands)} base formulas {time.time()-t0:.0f}s", flush=True)


def rank_ic(mat: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.Series:
    f = mat.loc[dates]
    r = fwd5.loc[dates]
    fr = f.rank(axis=1)
    rr = r.rank(axis=1)
    fm = fr.sub(fr.mean(axis=1), axis=0)
    rm = rr.sub(rr.mean(axis=1), axis=0)
    num = (fm * rm).sum(axis=1)
    den = np.sqrt((fm ** 2).sum(axis=1) * (rm ** 2).sum(axis=1))
    return (num / den).replace([np.inf, -np.inf], np.nan).dropna()


def select_factors(train_dates: pd.DatetimeIndex) -> dict:
    if len(train_dates) < 100:
        return {}
    half = len(train_dates) // 2
    h1, h2 = train_dates[:half], train_dates[half:]
    stats = {}
    for name, mat in cands.items():
        ic = rank_ic(mat, train_dates)
        if len(ic) < 100:
            continue
        mean_ic = float(ic.mean())
        t = mean_ic / (float(ic.std()) / np.sqrt(len(ic)))
        ic1 = float(rank_ic(mat, h1).mean())
        ic2 = float(rank_ic(mat, h2).mean())
        stats[name] = {"ic": mean_ic, "t": t, "ic_h1": ic1, "ic_h2": ic2,
                       "same_sign": np.sign(ic1) == np.sign(ic2)}
    selected = [n for n, s in stats.items()
                if abs(s["ic"]) > 0.01 and abs(s["t"]) > 2.0 and s["same_sign"]]
    if len(selected) < 5:
        selected = sorted(stats, key=lambda n: abs(stats[n]["t"]), reverse=True)[:10]
    return {"selected": selected, "stats": stats}


def build_segment(names: list[str], stats: dict, dates: pd.DatetimeIndex) -> pd.DataFrame:
    acc = np.zeros((len(dates), len(codes)))
    wsum = np.zeros((len(dates), len(codes)))
    for n in names:
        sign = 1.0 if stats[n]["ic"] > 0 else -1.0
        mat = cands[n].loc[dates].to_numpy(dtype=np.float64)
        ds.zscore_add(acc, wsum, mat, sign, ds.MIN_CROSS)
    comp = np.where(wsum > 0, acc / np.where(wsum > 0, wsum, 1.0), np.nan)
    lu = limit_up[np.searchsorted(master, dates)]
    comp[lu] = np.nan
    return pd.DataFrame(comp, index=dates, columns=codes)


years = [2021, 2022, 2023, 2024, 2025, 2026]
segments = []
yearly = {}
for y in years:
    train_lo = pd.Timestamp(f"{y-5}-01-01")
    train_hi = pd.Timestamp(f"{y-1}-12-31")
    test_lo = pd.Timestamp(f"{y}-01-01")
    test_hi = min(END, pd.Timestamp(f"{y}-12-31"))
    train_dates = master[(master >= train_lo) & (master <= train_hi)][:-5]
    test_dates = master[(master >= test_lo) & (master <= test_hi)]
    if len(train_dates) < 200 or len(test_dates) < 20:
        continue
    sel = select_factors(train_dates)
    if not sel:
        continue
    seg = build_segment(sel["selected"], sel["stats"], test_dates)
    segments.append(seg)
    yearly[str(y)] = {
        "train": [str(train_dates[0].date()), str(train_dates[-1].date())],
        "test": [str(test_dates[0].date()), str(test_dates[-1].date())],
        "selected": sel["selected"],
        "signs": {n: (1 if sel["stats"][n]["ic"] > 0 else -1) for n in sel["selected"]},
        "n_selected": len(sel["selected"]),
    }
    print(f"[year {y}] train={train_dates[0].date()}~{train_dates[-1].date()} "
          f"selected={len(sel['selected'])} {sel['selected']}", flush=True)

comp_rolling = pd.concat(segments).sort_index()
print(f"[rolling] {len(comp_rolling)} days, valid={np.isfinite(comp_rolling.to_numpy()).mean():.1%} "
      f"{time.time()-t0:.0f}s", flush=True)

# 对照: 2016-2021 选一次, 固定用到 2026
base_train = master[(master >= START) & (master <= pd.Timestamp("2021-05-11"))][:-5]
base_sel = select_factors(base_train)
comp_fixed = build_segment(base_sel["selected"], base_sel["stats"],
                           master[master >= TEST_START])
print(f"[fixed] {len(base_sel['selected'])} factors {base_sel['selected']}", flush=True)
# all23: 符号也按 2016-2021 估
comp_all23 = build_segment(list(cands), base_sel["stats"], master[master >= TEST_START])

_orig_prepare = FactorRankingStrategy.prepare_backtest_data
_orig_snap = m.pma_module.PortfolioManagerArray.record_portfolio_snapshot


async def _noop_prepare(self, *a, **k):
    return None


def inject(comp: pd.DataFrame) -> None:
    for c in codes:
        col = comp[c].to_numpy(dtype=np.float64)
        smap = {comp.index[k].to_pydatetime(): float(v)
                for k, v in enumerate(col) if np.isfinite(v)}
        frames[c].attrs["stock_code"] = c
        frames[c].attrs["from_precomputed"] = True
        frames[c].attrs[_SCORES_ATTR] = smap


async def run_cell(name: str, comp: pd.DataFrame):
    c = comp.shift(1)
    mask = pd.Series(c.index.dayofweek == 0, index=c.index)
    c = c.where(mask, np.nan)
    inject(c)
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
            initial_cash=1_000_000.0, commission_rate=0.0003, slippage_rate=0.001,
            open_cost=0.0003, close_cost=0.0008, min_cost=0.0,
            max_position_size=round(0.95 / TOPK, 6), cash_reserve_ratio=0.05,
            board_lot_size=100, stop_loss_pct=0.0, take_profit_pct=0.0,
            rebalance_frequency="daily",
        )
        await ex.run_backtest(
            strategy_name="factor_ranking", stock_codes=codes,
            start_date=TEST_START, end_date=END,
            strategy_config={"topk": TOPK, "n_drop": NDROP, "hold_thresh": 0},
            backtest_config=cfg,
        )
    finally:
        m.pma_module.PortfolioManagerArray.record_portfolio_snapshot = _orig_snap
        FactorRankingStrategy.prepare_backtest_data = _orig_prepare
    pm = stash.get("pm")
    eq = [(pd.Timestamp(d), float(v)) for d, v in pm.equity_curve]
    vals = np.array([v for _, v in eq])
    rets = np.diff(vals) / vals[:-1]
    st = m._ret_stats(rets)
    yearly_ret = {}
    for d, v in eq:
        yearly_ret.setdefault(d.year, []).append(v)
    yr = {str(y): round((vs[-1] / vs[0] - 1) * 100, 1)
          for y, vs in sorted(yearly_ret.items()) if len(vs) > 1}
    print(f"[{name}] sharpe={st['sharpe_ratio']:.3f} ret={st['total_return']*100:.0f}% "
          f"mdd={st['max_drawdown']*100:.1f}% ann={st['annualized_return']*100:.1f}% "
          f"trades={len(pm.trades)} yearly={yr}", flush=True)
    return {"name": name, "stats": {k: float(v) for k, v in st.items()
                                    if isinstance(v, (int, float))},
            "yearly": yr, "n_trades": len(pm.trades)}


async def main():
    out = []
    for name, comp in [("rolling_wf", comp_rolling),
                       ("fixed_22f", comp_fixed),
                       ("all23", comp_all23)]:
        out.append(await run_cell(name, comp))
    return out


cells = asyncio.run(main())
out = {"meta": {
    "window": [str(TEST_START.date()), str(END.date())],
    "universe": "point-in-time 4329",
    "factors": "23 base formulas",
    "execution": "weekly Monday, lag1, live cost (stamp 0.05% + slippage 0.1%)",
    "rolling": "每年用过去5年重估 IC+符号, 只对下一年打分",
    "yearly_selection": yearly,
    "fixed_selected": base_sel["selected"],
    "generated_at": datetime.now().isoformat(),
    "elapsed_s": round(time.time() - t0)},
    "cells": cells}
os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
with open(OUT_JSON, "w") as fh:
    json.dump(out, fh, ensure_ascii=False, indent=1, default=str)
print(f"[done] {time.time()-t0:.0f}s -> {OUT_JSON}", flush=True)
