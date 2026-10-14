"""[已废弃 2026-10-14] 因子定义与 qlib 面板错位 (MIN 取倒数/CORR 对象错), -0.53 为伪证。
保留仅供追溯, 请用 ic_composite_10y_honest.py。

IC 幸存因子复合 → 10 年 TopK/Dropout 回测 (信号侧研究弧第二步).

因子选择依据: factor_ic_study_5y.json 三门幸存者 (49/156) 的家族结构:
  - STD5-60 全负 IC 且两子期几乎同幅 (最稳定家族) → low_vol = -std(ret,20)
  - MIN5-60 全正 IC + MAX5-60 全负 IC → 同一短期反转效应两面:
    near_low = close/min(close,20)-1; dist_high = close/max(close,20)-1
  - CORR5-60 全负 IC (与市场相关性高 → 未来差) → corr_neg = -corr(ret,mkt,20)
等权复合 4 因子 (zscore 截面), 不做权重搜索 (避免同窗过拟合).

方法对齐 risk_parity_10y / expanded_legs_10y:
  窗口 2016-05-12 → 2026-05-08, point-in-time 固定宇宙, 引擎停牌估值修复后口径.
  TopK/Dropout 官方配置 topk=50/n_drop=5, max_position=0.019.
输出: data/strategy_baseline/ic_composite_10y.json
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
from app.services.backtest.core import portfolio_manager_array as pma_module

OUT_JSON = "/Users/rone/projects/stock-platform/data/strategy_baseline/ic_composite_10y.json"
WIN_START = os.environ.get("IC_COMP_START")  # >0 = 覆盖窗口起点 (5y 窗内对照)
SCORES_ATTR = "_factor_ranking_scores"

t0 = time.time()
if WIN_START:
    from datetime import datetime as _dt
    m.START = _dt.fromisoformat(WIN_START)
    OUT_JSON = OUT_JSON.replace(".json", "_5y.json")
m.patch_loader()
codes = m.list_universe()
print(f"[universe] {len(codes)} stocks, window {m.START.date()} -> {m.END.date()}", flush=True)

# ---------------- 载入 OHLCV → 矩阵 ----------------
files = {c: c.replace(".", "_") + ".parquet" for c in codes}
data_dir = m.STOCK_DATA_DIR
frames = {}
for c in codes:
    try:
        df = pd.read_parquet(os.path.join(data_dir, files[c]),
                             columns=["date", "close"])
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        df = df.loc[m.START:m.END]
        df = df[~df.index.duplicated(keep="first")]
        frames[c] = df
    except Exception:
        pass
codes = [c for c in codes if c in frames and len(frames[c]) > 60]
print(f"[load] {len(codes)} stocks usable, {time.time()-t0:.0f}s", flush=True)

master = sorted(set().union(*[set(f.index) for f in frames.values()]))
master = pd.DatetimeIndex(master)
n_d, n_s = len(master), len(codes)
close = np.full((n_d, n_s), np.nan)
for j, c in enumerate(codes):
    pos = master.get_indexer(pd.DatetimeIndex(frames[c].index))
    ok = pos >= 0
    close[pos[ok], j] = frames[c]["close"].to_numpy(dtype=np.float64)[ok]
close_df = pd.DataFrame(close, index=master)

# ---------------- 因子 (全部 OHLCV 可算, Alpha158 同名口径) ----------------
with np.errstate(divide="ignore", invalid="ignore"):
    ret = close_df.pct_change()
    mkt_ret = ret.mean(axis=1)  # 宇宙等权市场收益代理
    low_vol = -ret.rolling(20, min_periods=20).std()
    near_low = close_df / close_df.rolling(20, min_periods=20).min() - 1.0
    dist_high = close_df / close_df.rolling(20, min_periods=20).max() - 1.0
    # CORR: 与市场收益的20日相关 (Alpha158 同构)
    corr = ret.rolling(20, min_periods=20).corr(
        pd.Series(mkt_ret.to_numpy(), index=master))
    corr_neg = -corr

# dist_high = close/max-1 的 IC 为负 (MAX20 ic=-0.0165) → 复合取负
FACTOR_SIGN = {"low_vol": 1.0, "near_low": 1.0, "dist_high": -1.0, "corr_neg": 1.0}
factors = {"low_vol": low_vol, "near_low": near_low,
           "dist_high": dist_high, "corr_neg": corr_neg}
for k, v in factors.items():
    v_np = v.to_numpy(dtype=np.float64)
    print(f"[factor] {k}: valid={np.isfinite(v_np).sum()/max(1,np.isfinite(v_np).shape[0]*np.isfinite(v_np).shape[1]):.1%}", flush=True)

# ---------------- 复合: 截面 zscore 等权 ----------------
composite = np.full((n_d, n_s), np.nan)
weight_sum = np.zeros((n_d, n_s))
acc = np.zeros((n_d, n_s))
min_cross = min(30, max(2, n_s // 10))
for name, v in factors.items():
    mat = v.to_numpy(dtype=np.float64) * FACTOR_SIGN[name]
    valid = np.isfinite(mat)
    counts = valid.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(np.where(valid, mat, np.nan), axis=1, keepdims=True)
        std = np.nanstd(np.where(valid, mat, np.nan), axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (mat - mean) / np.where(std > 1e-12, std, np.nan)
    z = np.where(counts >= min_cross, z, np.nan)
    ok = np.isfinite(z)
    acc[ok] += np.nan_to_num(z)[ok]
    weight_sum[ok] += 1.0
composite = np.where(weight_sum > 0, acc / np.where(weight_sum > 0, weight_sum, 1), np.nan)
print(f"[composite] valid={np.isfinite(composite).mean():.1%} "
      f"nonzero_weights={(weight_sum>0).mean():.1%}", flush=True)

# ---------------- 注入 scores → 回测 ----------------
_orig_prepare = None
from app.services.backtest.strategies.factor_ranking_strategy import FactorRankingStrategy
from app.services.backtest.strategies.factor_ranking_strategy import _SCORES_ATTR

def inject_scores(stock_data):
    stored = 0
    for j, c in enumerate(codes):
        if c not in stock_data:
            continue
        col = composite[:, j]
        score_map = {}
        for k in range(n_d):
            v = col[k]
            if np.isfinite(v):
                score_map[master[k].to_pydatetime()] = float(v)
        stock_data[c].attrs[_SCORES_ATTR] = score_map
        stored += len(score_map)
    print(f"[inject] scores stored={stored}", flush=True)

async def _noop_prepare(self, *a, **k):
    return None

async def run_one():
    ex = BacktestExecutor(data_dir=str(settings.DATA_ROOT_PATH),
                          enable_parallel=True, max_workers=8,
                          enable_performance_profiling=False,
                          use_multiprocessing=False)
    stock_data = {}
    for c in codes:
        df = frames[c].copy()
        df["open"] = df["close"]
        df["high"] = df["close"]
        df["low"] = df["close"]
        df["volume"] = 1e6
        df.attrs["stock_code"] = c
        df.attrs["from_precomputed"] = True
        stock_data[c] = df
    inject_scores(stock_data)
    # 直接短路加载器, 注入带分数的 frames
    ex.data_loader.load_multiple_stocks = lambda *a, **k: stock_data
    FactorRankingStrategy.prepare_backtest_data = _noop_prepare
    stash = {}
    _orig_snap = pma_module.PortfolioManagerArray.record_portfolio_snapshot
    def patched(self, date, current_prices):
        stash["pm"] = self
        _orig_snap(self, date, current_prices)
    pma_module.PortfolioManagerArray.record_portfolio_snapshot = patched
    try:
        cfg = BacktestConfig(
            initial_cash=1_000_000.0, commission_rate=0.0003, slippage_rate=0.0001,
            max_position_size=0.019, cash_reserve_ratio=0.05, board_lot_size=100,
            stop_loss_pct=0.0, take_profit_pct=0.0, rebalance_frequency="daily",
        )
        await ex.run_backtest(
            strategy_name="factor_ranking",
            stock_codes=codes, start_date=m.START, end_date=m.END,
            strategy_config={"topk": 50, "n_drop": 5, "hold_thresh": 1,
                             "weights": FACTOR_SIGN},
            backtest_config=cfg)
    finally:
        pma_module.PortfolioManagerArray.record_portfolio_snapshot = _orig_snap
        FactorRankingStrategy.prepare_backtest_data = _orig_prepare or _noop_prepare
    return stash.get("pm")

pm = asyncio.run(run_one())
assert pm is not None, "no PM captured"
eq = [(d, float(v)) for d, v in pm.equity_curve]
vals = np.array([v for _, v in eq])
rets = np.diff(vals) / vals[:-1]
stats = m._ret_stats(rets)
print(f"[result] {json.dumps({k: round(float(v), 4) for k, v in stats.items() if isinstance(v, (int, float))}, ensure_ascii=False)}", flush=True)

# 年度稳定性
yearly = {}
for d, v in eq:
    y = pd.Timestamp(d).year
    yearly.setdefault(y, []).append(v)
yr = {y: round((vs[-1] / vs[0] - 1) * 100, 1) for y, vs in sorted(yearly.items()) if len(vs) > 1}
print(f"[yearly] {yr}", flush=True)

out = {
    "meta": {
        "window": [str(m.START.date()), str(m.END.date())],
        "universe": f"point-in-time-fixed-{len(codes)}",
        "factors": list(FACTOR_SIGN.keys()),
        "composite": "equal-weight zscore",
        "config": {"topk": 50, "n_drop": 5, "max_position": 0.019},
        "generated_at": datetime.now().isoformat(),
        "elapsed_s": round(time.time() - t0),
    },
    "stats": stats,
    "yearly_return_pct": yr,
    "n_trades": len(pm.trades),
}
os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
with open(OUT_JSON, "w") as fh:
    json.dump(out, fh, ensure_ascii=False, indent=1, default=str)
print(f"[done] {time.time()-t0:.0f}s -> {OUT_JSON}", flush=True)
