"""IC 幸存因子复合 → 10 年回测 v2 (信号侧研究弧第三步, 定义修正版).

v1 (ic_composite_10y.py) 废弃: OHLCV 重算因子与 qlib 面板定义错位 (MIN/MAX 取倒数、
CORR 用错对象), 方向反转, -0.53 Sharpe 是伪证。

本版: 31/49 幸存者已通过公式匹配验证 (rank-corr ≥ 0.85 且方向保持, 见
/tmp/factor_formula_map.json), 用匹配公式从 stock_data (真实 OHLCV) 重算。
关键识别力: IC 符号选自 2021-05→2026-05 面板窗口, 因此
  - 2021-05→2026-05 = 选择窗内 (in-selection)
  - 2016-05→2021-05 = 真正样本外 (OOS, IC 选择从未见过)
门3 判据: OOS 子期方向必须一致且显著。

窗口 2016-05-12→2026-05-08, point-in-time 固定宇宙, TopK/Dropout topk=50/n_drop=5,
成本 0.0003+0.0001。输出: data/strategy_baseline/ic_composite_10y_v2.json
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
OUT_JSON = "/Users/rone/projects/stock-platform/data/strategy_baseline/ic_composite_10y_v2.json"
OOS_SPLIT = pd.Timestamp("2021-05-11")

t0 = time.time()
m.patch_loader()
codes = m.list_universe()
START, END = m.START, m.END

study = json.load(open(SURV_JSON))
fmap = json.load(open(FORMULA_MAP))
# 仅用已验证公式 (方向保持 rc>=0.85), 符号取研究 IC 符号
use = {c: {"formula": fmap[c]["formula"], "rc": fmap[c]["rc"],
           "sign": (1.0 if study["results"][c]["fwd1"]["ic_mean"] > 0 else -1.0)}
       for c in study["survivors"] if c in fmap}
print(f"[setup] {len(codes)} stocks, {len(use)} verified factors, {START.date()}->{END.date()}", flush=True)

# ---- 载入真实 OHLCV ----
frames = {}
for c in codes:
    f = c.replace(".", "_") + ".parquet"
    try:
        df = pd.read_parquet(os.path.join(m.STOCK_DATA_DIR, f))
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
opn = pd.DataFrame(to_mat("open"), index=master)
vol = pd.DataFrame(to_mat("volume"), index=master)

with np.errstate(divide="ignore", invalid="ignore"):
    ret = close.pct_change()
    vchg = vol.pct_change()
    ratio = close / close.shift(1)
    cands = {}
    for n in (5, 10, 20, 30, 60):
        cands[f"MIN{n}"] = close.rolling(n).min() / close - 1
        cands[f"MAX{n}"] = close.rolling(n).max() / close - 1
        cands[f"STD{n}"] = close.rolling(n).std() / close
        cands[f"CORR{n}"] = close.rolling(n).corr(vol)
        cands[f"CORD{n}"] = ret.rolling(n).corr(vchg)
        cands[f"RSQR{n}"] = close.rolling(n).corr(
            pd.Series(np.arange(n_d, dtype=float), index=master)) ** 2
    cands["KLEN"] = (high - low) / (opn + 1e-12)
    # 注意: 不要用 pd.concat(...).max(axis=1) — 同名列会触发列 join 笛卡尔展开
    hi_oc = pd.DataFrame(np.maximum(opn.to_numpy(), close.to_numpy()), index=master)
    lo_oc = pd.DataFrame(np.minimum(opn.to_numpy(), close.to_numpy()), index=master)
    cands["KUP"] = (high - hi_oc) / (opn + 1e-12)
    cands["KLOW"] = (lo_oc - low) / (opn + 1e-12)
    cands["LOW0"] = np.log(low / close)
    cands["HIGH0"] = np.log(high / close)
    cands["LOW0_a"] = cands["LOW0"]
    cands["HIGH0_a"] = cands["HIGH0"]

# ---- 复合: 每因子截面 zscore × IC 符号, 等权 (公式级去重: 同公式多特征只算一次) ----
seen_formula = {}
for feat, spec in use.items():
    seen_formula.setdefault(spec["formula"], []).append(feat)
print(f"[dedup] {len(use)} features -> {len(seen_formula)} unique formulas", flush=True)

acc = np.zeros((n_d, n_s))
wsum = np.zeros((n_d, n_s))
for fname, feats in seen_formula.items():
    mat = cands[fname].to_numpy(dtype=np.float64)
    # 同公式多窗特征: 平均 IC 符号冲突时按多数决; 此处公式名带窗, 不会冲突
    signs = [use[f]["sign"] for f in feats]
    s = signs[0]
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
print(f"[composite] valid={np.isfinite(composite).mean():.1%} {time.time()-t0:.0f}s", flush=True)

# ---- 注入引擎 ----
for j, c in enumerate(codes):
    smap = {}
    col = composite[:, j]
    for k in range(n_d):
        v = col[k]
        if np.isfinite(v):
            smap[master[k].to_pydatetime()] = float(v)
    frames[c].attrs["stock_code"] = c
    frames[c].attrs["from_precomputed"] = True
    frames[c].attrs[_SCORES_ATTR] = smap

async def _noop_prepare(self, *a, **k):
    return None

async def run_one():
    ex = BacktestExecutor(data_dir=str(settings.DATA_ROOT_PATH),
                          enable_parallel=True, max_workers=8,
                          enable_performance_profiling=False,
                          use_multiprocessing=False)
    ex.data_loader.load_multiple_stocks = lambda *a, **k: frames
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
            stock_codes=codes, start_date=START, end_date=END,
            strategy_config={"topk": 50, "n_drop": 5, "hold_thresh": 0,
                             "weights": {f: 1.0 for f in use}},
            backtest_config=cfg)
    finally:
        m.pma_module.PortfolioManagerArray.record_portfolio_snapshot = _orig
    return stash.get("pm")

pm = asyncio.run(run_one())
assert pm is not None
eq = [(pd.Timestamp(d), float(v)) for d, v in pm.equity_curve]
vals = np.array([v for _, v in eq])
rets = np.diff(vals) / vals[:-1]
stats = m._ret_stats(rets)
print(f"[result] {json.dumps({k: round(float(v), 4) for k, v in stats.items() if isinstance(v, (int, float))}, ensure_ascii=False)}", flush=True)

# ---- 子期: OOS 2016-05→2021-05 vs in-selection 2021-05→2026-05 ----
sub = {}
for name, lo, hi in [("full", START, END), ("oos_2016_2021", START, OOS_SPLIT),
                     ("inselect_2021_2026", OOS_SPLIT, END)]:
    mask = [(d >= lo) and (d <= hi) for d, _ in eq]
    sv = vals[np.array(mask)]
    if len(sv) > 30:
        sr = np.diff(sv) / sv[:-1]
        sub[name] = {k: round(float(v), 4) for k, v in m._ret_stats(sr).items()
                     if isinstance(v, (int, float))}
        sub[name]["days"] = int(len(sv))
        print(f"[sub:{name}] {json.dumps(sub[name], ensure_ascii=False)}", flush=True)

yearly = {}
for d, v in eq:
    yearly.setdefault(d.year, []).append(v)
yr = {y: round((vs[-1] / vs[0] - 1) * 100, 1) for y, vs in sorted(yearly.items()) if len(vs) > 1}
print(f"[yearly] {yr}", flush=True)

out = {
    "meta": {
        "window": [str(START.date()), str(END.date())],
        "n_factors": len(use),
        "factor_map": {k: v for k, v in use.items()},
        "config": {"topk": 50, "n_drop": 5, "max_position": 0.019},
        "oos_split": str(OOS_SPLIT.date()),
        "note": "v2 定义修正: 31 幸存者经 rc>=0.85 公式匹配后从真实 OHLCV 重算; OOS=2016-2021",
        "generated_at": datetime.now().isoformat(),
        "elapsed_s": round(time.time() - t0),
    },
    "stats": stats,
    "subperiods": sub,
    "yearly_return_pct": yr,
    "n_trades": len(pm.trades),
}
os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
with open(OUT_JSON, "w") as fh:
    json.dump(out, fh, ensure_ascii=False, indent=1, default=str)
print(f"[done] {time.time()-t0:.0f}s -> {OUT_JSON}", flush=True)
