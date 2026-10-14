"""修复后官方基线重跑 (方案 A 验收): 5y 窗对照组 — 应从 2021-12 起有交易."""
import asyncio
import importlib.util
import json
import os
import sys
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
from app.services.backtest.strategies.factor_ranking_strategy import FactorRankingStrategy, _SCORES_ATTR

START = pd.Timestamp(os.environ.get("FR_START", "2021-05-11"))
OUT_JSON = os.environ.get(
    "FR_OUT",
    "/Users/rone/projects/stock-platform/data/strategy_baseline/factor_ranking_control_5y_fixed.json")

m.patch_loader()
codes = m.list_universe()
END = m.END

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
        df.attrs["stock_code"] = c
        df.attrs["from_precomputed"] = True
        frames[c] = df
codes = list(frames)
print(f"[setup] {len(codes)} stocks {START.date()}->{END.date()}", flush=True)

_orig_snap = m.pma_module.PortfolioManagerArray.record_portfolio_snapshot

async def run():
    ex = BacktestExecutor(data_dir=str(settings.DATA_ROOT_PATH),
                          enable_parallel=True, max_workers=8,
                          enable_performance_profiling=False,
                          use_multiprocessing=False)
    ex.data_loader.load_multiple_stocks = lambda *a, **k: frames
    stash = {}

    def patched(self, date, current_prices):
        stash["pm"] = self
        _orig_snap(self, date, current_prices)
    m.pma_module.PortfolioManagerArray.record_portfolio_snapshot = patched
    try:
        cfg = BacktestConfig(
            initial_cash=1_000_000.0, commission_rate=0.0003, slippage_rate=0.0001,
            max_position_size=0.019, cash_reserve_ratio=0.05, board_lot_size=100,
            stop_loss_pct=0.0, take_profit_pct=0.0, rebalance_frequency="daily",
        )
        await ex.run_backtest(strategy_name="factor_ranking", stock_codes=codes,
                              start_date=START, end_date=END,
                              strategy_config={"topk": 50, "n_drop": 5, "hold_thresh": 0,
                                               "weights": {"momentum": 1.0, "low_vol": 1.0,
                                                           "low_amp": 1.0}},
                              backtest_config=cfg)
    finally:
        m.pma_module.PortfolioManagerArray.record_portfolio_snapshot = _orig_snap
    pm = stash["pm"]
    # 分数覆盖年份 (修复是否生效)
    cover = {}
    for c in codes[:200]:
        for d in frames[c].attrs.get(_SCORES_ATTR, {}):
            cover.setdefault(pd.Timestamp(d).year, set()).add(c)
    print(f"[scores] valid-score stocks by year: {{{', '.join(f'{y}: {len(s)}' for y, s in sorted(cover.items()))}}}", flush=True)
    eq = [(pd.Timestamp(d), float(v)) for d, v in pm.equity_curve]
    vals = np.array([v for _, v in eq])
    rets = np.diff(vals) / vals[:-1]
    stats = m._ret_stats(rets)
    yearly = {}
    for d, v in eq:
        yearly.setdefault(d.year, []).append(v)
    yr = {y: round((vs[-1] / vs[0] - 1) * 100, 1) for y, vs in sorted(yearly.items()) if len(vs) > 1}
    print(f"[result] sharpe={stats['sharpe_ratio']:.3f} ret={stats['total_return']*100:.1f}% "
          f"mdd={stats['max_drawdown']*100:.1f}% n_trades={len(pm.trades)}", flush=True)
    print(f"[yearly] {yr}", flush=True)
    out = {"meta": {"window": [str(START.date()), str(END.date())],
                    "note": "NaN 传播修复后重跑的官方三因子 5y 对照组",
                    "generated_at": datetime.now().isoformat()},
           "stats": stats, "yearly_return_pct": yr, "n_trades": len(pm.trades)}
    with open(OUT_JSON, "w") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1, default=str)
    print(f"[done] -> {OUT_JSON}", flush=True)

asyncio.run(run())
