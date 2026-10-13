"""面板原生复合因子 → 引擎回测 (信号侧研究弧第二步, 修正版).

上一版 ic_composite_10y.py 用 OHLCV 重算因子, 但定义与 qlib 面板不一致, 方向错位:
  - 面板 MIN20 = min/close-1 (rc -0.94 vs 我的 close/min-1) → 买"远离低点"= 反转输家侧
  - 面板 MAX20 = max/close-1 (rc -0.96 vs close/max-1)
  - 面板 CORR20 = corr(close, volume) rc 0.95, 不是 corr(ret, mkt)
  → 上一版 -0.53 Sharpe 实为方向错位的伪证.

本版: 分数直接取面板原生特征 (49 个三门幸存者, 研究 IC 符号, 截面 zscore 等权),
不重算任何因子。窗口 2021-05-11→2026-05-08 (面板覆盖), TopK/Dropout topk=50/n_drop=5,
真实成本 (佣金 0.0003 + 滑点 0.0001)。
输出: data/strategy_baseline/ic_composite_panel5y.json
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

FEAT_DIR = "/Users/rone/projects/stock-platform/data/qlib_data/features/day"
SURV_JSON = "/Users/rone/projects/stock-platform/data/strategy_baseline/factor_ic_study_5y.json"
OUT_JSON = "/Users/rone/projects/stock-platform/data/strategy_baseline/ic_composite_panel5y.json"
B2 = "CNTP5"

t0 = time.time()
m.patch_loader()
codes = m.list_universe()
START = pd.Timestamp("2021-05-11")
END = m.END  # 2026-05-08

study = json.load(open(SURV_JSON))
survivors = study["survivors"]
SIGNS = {c: (1.0 if study["results"][c]["fwd1"]["ic_mean"] > 0 else -1.0) for c in survivors}
print(f"[setup] {len(codes)} stocks, {len(survivors)} survivors, window {START.date()}->{END.date()}", flush=True)

# ---- 载入面板特征 → 每只股票的分数序列 (对齐面板交易日历) ----
code_set = set(codes)
feat_files = [c.replace(".", "_") + ".parquet" for c in codes]
files = [f for f in feat_files if os.path.exists(os.path.join(FEAT_DIR, f))]
print(f"[load] {len(files)} panel files exist for universe", flush=True)

date_sets = []
for f in files[::60]:
    x = pd.read_parquet(os.path.join(FEAT_DIR, f), columns=["$close"])
    idx = pd.DatetimeIndex(x.index.get_level_values("date"))
    date_sets.append(set(idx[x["$close"].notna().to_numpy()]))
dates = pd.DatetimeIndex(sorted(set().union(*date_sets)))
dates = dates[(dates >= START) & (dates <= END)]
n_d = len(dates)
pos_of = {d: i for i, d in enumerate(dates)}
print(f"[cal] {n_d} days {dates[0].date()}->{dates[-1].date()}", flush=True)

feat_mats = {c: np.full((n_d, len(files)), np.nan, dtype=np.float32) for c in survivors}
for j, f in enumerate(files):
    try:
        p = pd.read_parquet(os.path.join(FEAT_DIR, f))
    except Exception:
        continue
    idx = pd.DatetimeIndex(p.index.get_level_values("date"))
    p = p.copy(); p.index = idx
    b2 = p[p[B2].notna()]
    if b2.empty:
        continue
    pos = np.array([pos_of[d] for d in b2.index if d in pos_of], dtype=np.int64)
    for c in survivors:
        if c in p.columns:
            v = b2[c].to_numpy(dtype=np.float32)
            n = min(len(pos), len(v))
            feat_mats[c][pos[:n], j] = v[:n]
    if (j + 1) % 1000 == 0:
        print(f"[load] {j+1}/{len(files)} {time.time()-t0:.0f}s", flush=True)

# ---- 复合: 截面 zscore × IC 符号, 等权 ----
acc = np.zeros((n_d, len(files)), dtype=np.float64)
wsum = np.zeros((n_d, len(files)), dtype=np.float64)
min_cross = 30
for c, mat in feat_mats.items():
    M = mat.astype(np.float64) * SIGNS[c]
    ok = np.isfinite(M)
    cnt = ok.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore"):
        mu = np.nanmean(np.where(ok, M, np.nan), axis=1, keepdims=True)
        sd = np.nanstd(np.where(ok, M, np.nan), axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (M - mu) / np.where(sd > 1e-12, sd, np.nan)
    z = np.where(cnt >= min_cross, z, np.nan)
    good = np.isfinite(z)
    acc[good] += np.nan_to_num(z)[good]
    wsum[good] += 1
composite = np.where(wsum > 0, acc / np.where(wsum > 0, wsum, 1), np.nan)
print(f"[composite] valid={np.isfinite(composite).mean():.1%}", flush=True)

# ---- 注入引擎 (与 ic_composite_10y 相同路径) ----
def load_stock_frames():
    data_dir = m.STOCK_DATA_DIR
    out = {}
    for c in codes:
        f = c.replace(".", "_") + ".parquet"
        try:
            df = pd.read_parquet(os.path.join(data_dir, f))
        except Exception:
            continue
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        df = df.loc[START:END]
        df = df[~df.index.duplicated(keep="first")]
        if len(df) < 60:
            continue
        df.attrs["stock_code"] = c
        df.attrs["from_precomputed"] = True
        out[c] = df
    return out

stock_data = load_stock_frames()
usable = [c for c in codes if c in stock_data]
print(f"[frames] {len(usable)} stocks with OHLCV", flush=True)
col_of = {f[:-8].replace("_", "."): j for j, f in enumerate(files)}

stored = 0
for c in usable:
    j = col_of.get(c)
    if j is None:
        continue
    smap = {}
    for k in range(n_d):
        v = composite[k, j]
        if np.isfinite(v):
            smap[dates[k].to_pydatetime()] = float(v)
    stock_data[c].attrs[_SCORES_ATTR] = smap
    stored += len(smap)
print(f"[inject] score entries={stored}", flush=True)

async def _noop_prepare(self, *a, **k):
    return None

async def run_one():
    ex = BacktestExecutor(data_dir=str(settings.DATA_ROOT_PATH),
                          enable_parallel=True, max_workers=8,
                          enable_performance_profiling=False,
                          use_multiprocessing=False)
    ex.data_loader.load_multiple_stocks = lambda *a, **k: stock_data
    FactorRankingStrategy.prepare_backtest_data = _noop_prepare
    stash = {}
    _orig = m.pma_module.PortfolioManagerArray.record_portfolio_snapshot
    def patched(self, date, current_prices):
        stash["pm"] = self
        _orig(self, date, current_prices)
    m.pma_module.PortfolioManagerArray.record_portfolio_snapshot = patched
    try:
        cfg = BacktestConfig(
            initial_cash=1_000_000.0, commission_rate=0.0003, slippage_rate=0.0001,
            max_position_size=0.019, cash_reserve_ratio=0.05, board_lot_size=100,
            stop_loss_pct=0.0, take_profit_pct=0.0, rebalance_frequency="daily",
        )
        await ex.run_backtest(
            strategy_name="factor_ranking",
            stock_codes=usable, start_date=START, end_date=END,
            strategy_config={"topk": 50, "n_drop": 5, "hold_thresh": 0, "weights": SIGNS},
            backtest_config=cfg)
    finally:
        m.pma_module.PortfolioManagerArray.record_portfolio_snapshot = _orig
    return stash.get("pm")

pm = asyncio.run(run_one())
assert pm is not None
eq = [(d, float(v)) for d, v in pm.equity_curve]
vals = np.array([v for _, v in eq])
rets = np.diff(vals) / vals[:-1]
stats = m._ret_stats(rets)
print(f"[result] {json.dumps({k: round(float(v), 4) for k, v in stats.items() if isinstance(v, (int, float))}, ensure_ascii=False)}", flush=True)

yearly = {}
for d, v in eq:
    yearly.setdefault(pd.Timestamp(d).year, []).append(v)
yr = {y: round((vs[-1] / vs[0] - 1) * 100, 1) for y, vs in sorted(yearly.items()) if len(vs) > 1}
print(f"[yearly] {yr}", flush=True)

out = {
    "meta": {
        "window": [str(START.date()), str(END.date())],
        "n_scores": stored,
        "n_stocks": len(usable),
        "factors": f"panel-native {len(survivors)} three-gate survivors, IC-signed zscore equal-weight",
        "config": {"topk": 50, "n_drop": 5, "max_position": 0.019,
                   "commission": 0.0003, "slippage": 0.0001},
        "note": "修正版: 分数直接用 qlib 面板原生特征 (上一版 OHLCV 重算因子定义错位已废弃)",
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
