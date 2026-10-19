#!/usr/bin/env python3
"""Walk-forward 因子重选: 2016-2021 选因子 → 2021-2026 验证.

问题: 当前 42f 的因子是用 2021-2026 的 IC 选出来的, 所以 2021-2026 是 in-sample。
2016-2021 才是真 OOS, 而周频 OOS 夏普只有 0.041。本实验检验:
  "用 2016-2021 的 IC 选因子, 在 2021-2026 还能不能赚钱?"
如果 selected_test 夏普 ~0 → 因子选择过程没有 OOS 预测力, 当前 42f 是拟合。
如果 >0.5 → 有真 edge, 值得继续。

范围: 23 个基础公式 (MIN/MAX/STD/CORR/CORD × 5/10/20/30/60), 因为:
  - 它们能从 stock_data 完整重算到 2016;
  - 17 个 Alpha158 面板因子只有 2021-05 起的数据, 无法做 2016-2021 训练。

四个单元 (topk30/ndrop3, 周频周一调仓 + lag1 + 实盘成本):
  selected_train  2016-2021, 用 2016-2021 IC 选出的因子 (in-sample)
  selected_test   2021-2026, 同一批因子 (真 OOS)
  all23_train     2016-2021, 全部 23 公式等权 (对照)
  all23_test      2021-2026, 全部 23 公式等权 (对照)

选因子规则:
  - 训练窗 RankIC (fwd5) 均值 |IC| > 0.01
  - |t| > 2
  - 训练窗前后两半 IC 同号
  若合格因子 < 5, 退化为按 |t| 取前 10。

输出: data/strategy_baseline/ic_composite_walkforward.json
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

OUT_JSON = "/Users/rone/Projects/stock-platform/data/strategy_baseline/ic_composite_walkforward.json"
START, END = m.START, m.END
TRAIN_END = pd.Timestamp("2021-05-11")
TEST_START = pd.Timestamp("2021-05-12")
TOPK, NDROP = 30, 3

t0 = time.time()
m.patch_loader()
frames_all, panel, master_all = ds.load_data(m.list_universe())
print(f"[load] {len(frames_all)} stocks {len(master_all)} days {time.time()-t0:.0f}s", flush=True)

# 切 2016-2026 窗口
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
print(f"[factors] {len(cands)} base formulas {time.time()-t0:.0f}s", flush=True)

train_mask = master <= TRAIN_END
test_mask = master >= TEST_START
# 去掉训练窗最后 5 个交易日: 它们的 fwd5 标签会用到测试窗数据 (边界前视)
train_dates = master[train_mask][:-5]
half = len(train_dates) // 2
h1, h2 = train_dates[:half], train_dates[half:]


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


stats = {}
for name, mat in cands.items():
    ic = rank_ic(mat, train_dates)
    if len(ic) < 100:
        continue
    mean_ic = float(ic.mean())
    t = mean_ic / (float(ic.std()) / np.sqrt(len(ic)))
    ic1 = rank_ic(mat, h1)
    ic2 = rank_ic(mat, h2)
    same_sign = np.sign(float(ic1.mean())) == np.sign(float(ic2.mean()))
    stats[name] = {"ic": mean_ic, "t": t, "ic_h1": float(ic1.mean()),
                   "ic_h2": float(ic2.mean()), "same_sign": bool(same_sign),
                   "n": len(ic)}
    print(f"[ic] {name:<6} ic={mean_ic:+.4f} t={t:+.2f} "
          f"h1={ic1.mean():+.4f} h2={ic2.mean():+.4f} same={same_sign}", flush=True)

selected = [n for n, s in stats.items()
            if abs(s["ic"]) > 0.01 and abs(s["t"]) > 2.0 and s["same_sign"]]
if len(selected) < 5:
    selected = sorted(stats, key=lambda n: abs(stats[n]["t"]), reverse=True)[:10]
print(f"[select] {len(selected)} factors: {selected}", flush=True)


def build_composite(names: list[str]) -> pd.DataFrame:
    acc = np.zeros((len(master), len(codes)))
    wsum = np.zeros((len(master), len(codes)))
    for n in names:
        sign = 1.0 if stats[n]["ic"] > 0 else -1.0
        ds.zscore_add(acc, wsum, cands[n].to_numpy(dtype=np.float64), sign, ds.MIN_CROSS)
    comp = np.where(wsum > 0, acc / np.where(wsum > 0, wsum, 1.0), np.nan)
    limit_up = close.pct_change(fill_method=None).to_numpy() >= ds.LIMIT_UP
    comp[limit_up] = np.nan
    return pd.DataFrame(comp, index=master, columns=codes)


comp_sel = build_composite(selected)
comp_all = build_composite(list(cands))
print(f"[composite] selected={np.isfinite(comp_sel.to_numpy()).mean():.1%} "
      f"all={np.isfinite(comp_all.to_numpy()).mean():.1%} {time.time()-t0:.0f}s", flush=True)

_orig_prepare = FactorRankingStrategy.prepare_backtest_data
_orig_snap = m.pma_module.PortfolioManagerArray.record_portfolio_snapshot


async def _noop_prepare(self, *a, **k):
    return None


def inject(comp: pd.DataFrame) -> None:
    for c in codes:
        col = comp[c].to_numpy(dtype=np.float64)
        smap = {master[k].to_pydatetime(): float(v) for k, v in enumerate(col) if np.isfinite(v)}
        frames[c].attrs["stock_code"] = c
        frames[c].attrs["from_precomputed"] = True
        frames[c].attrs[_SCORES_ATTR] = smap


async def run_cell(name: str, comp: pd.DataFrame, lo, hi):
    # 周频: 周一用上一交易日分数
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
            start_date=lo, end_date=hi,
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
    print(f"[{name}] sharpe={st['sharpe_ratio']:.3f} ret={st['total_return']*100:.0f}% "
          f"mdd={st['max_drawdown']*100:.1f}% ann={st['annualized_return']*100:.1f}% "
          f"trades={len(pm.trades)}", flush=True)
    return {"name": name, "period": [str(lo.date()), str(hi.date())],
            "stats": {k: float(v) for k, v in st.items() if isinstance(v, (int, float))},
            "n_trades": len(pm.trades)}


CELLS = [
    ("selected_train", comp_sel, START, TRAIN_END),
    ("selected_test", comp_sel, TEST_START, END),
    ("all23_train", comp_all, START, TRAIN_END),
    ("all23_test", comp_all, TEST_START, END),
]


async def main():
    out = []
    for name, comp, lo, hi in CELLS:
        out.append(await run_cell(name, comp, lo, hi))
    return out


cells = asyncio.run(main())
out = {"meta": {
    "window": [str(START.date()), str(END.date())],
    "train": [str(START.date()), str(TRAIN_END.date())],
    "test": [str(TEST_START.date()), str(END.date())],
    "universe": "point-in-time 4329",
    "factors": "23 base formulas (MIN/MAX/STD/CORR/CORD x 5/10/20/30/60)",
    "selected": selected, "ic_stats": stats,
    "execution": "weekly Monday, lag1, live cost (stamp 0.05% + slippage 0.1%)",
    "note": "2016-2021 选因子 → 2021-2026 验证; 17 panel factors unavailable pre-2021",
    "generated_at": datetime.now().isoformat(),
    "elapsed_s": round(time.time() - t0)},
    "cells": cells}
os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
with open(OUT_JSON, "w") as fh:
    json.dump(out, fh, ensure_ascii=False, indent=1, default=str)
print(f"[done] {time.time()-t0:.0f}s -> {OUT_JSON}", flush=True)
