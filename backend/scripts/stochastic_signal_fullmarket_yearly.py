#!/usr/bin/env python3
"""stochastic 信号变体 全市场分年验证 (刀锋 vs 平台 + 分年稳定)。

上一轮全市场单跑发现 G3(35/65 k10) 夏普 0.924 跑赢默认(0.807), 但它是从
500宇宙网格挑选的 → 选择偏差未排除。本脚本在全市场 5513 上对
默认 + G3 + 两个邻格 做**分年**对比, 一次性回答三关:
  (a) G3 是否每个子年度都稳定跑赢默认(而非单窗口聚合假象)
  (b) 邻格(35/65 k14, 30/70 k10)是否同向改善 → 平台 or 刀锋
  (c) 综合判定 G3 能否注册为正式策略变体

口径: 全市场 5513, 2023-05-12→2026-05-08, 同基线费率/仓位, 确定性引擎。
产物: data/strategy_baseline/stochastic_signal_fullmarket_yearly.json
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import json
import math
import os
import sys
import time
from datetime import datetime

import numpy as np

BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BACKEND_ROOT)

_spec = importlib.util.spec_from_file_location("rsb", os.path.join(BACKEND_ROOT, "scripts", "run_strategy_baseline.py"))
assert _spec is not None
rsb = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(rsb)

from app.core.config import settings  # noqa: E402
from app.services.backtest import BacktestExecutor  # noqa: E402
from app.services.backtest.models import BacktestConfig  # noqa: E402
from app.services.backtest.core import portfolio_manager_array as pma_module  # noqa: E402

DATA_ROOT = "/Users/rone/projects/stock-platform/data"
OUT_JSON = os.path.join(DATA_ROOT, "strategy_baseline", "stochastic_signal_fullmarket_yearly.json")
START, END = datetime(2023, 5, 12), datetime(2026, 5, 8)

GRID = [
    ("G0_默认20_80_k14", {"oversold": 20, "overbought": 80, "k_period": 14, "d_period": 3}),
    ("G3_35_65_k10", {"oversold": 35, "overbought": 65, "k_period": 10, "d_period": 3}),
    ("N1_35_65_k14", {"oversold": 35, "overbought": 65, "k_period": 14, "d_period": 3}),
    ("N2_30_70_k10", {"oversold": 30, "overbought": 70, "k_period": 10, "d_period": 3}),
]

_orig_snapshot = pma_module.PortfolioManagerArray.record_portfolio_snapshot


def install_stash(stash: dict) -> None:
    def patched(self, date, current_prices):
        stash["pm"] = self
        _orig_snapshot(self, date, current_prices)

    pma_module.PortfolioManagerArray.record_portfolio_snapshot = patched


def uninstall() -> None:
    pma_module.PortfolioManagerArray.record_portfolio_snapshot = _orig_snapshot


def _ret_stats(equity):
    if not equity or len(equity) < 30:
        return None
    vals = np.array([v for _, v in equity], dtype=float)
    base = np.where(vals[:-1] > 0, vals[:-1], 1.0)
    rets = np.diff(vals) / base
    years = max(1.0, len(equity) / 252.0)
    total = vals[-1] / vals[0] - 1.0
    ann = (1 + total) ** (1 / years) - 1 if total > -1 else -1.0
    vol = float(rets.std() * math.sqrt(252))
    sharpe = float(rets.mean() / rets.std() * math.sqrt(252)) if rets.std() > 0 else 0.0
    peak = np.maximum.accumulate(vals)
    mdd = float(((vals - peak) / peak).min())
    return {
        "total_return": round(total, 6),
        "annualized_return": round(ann, 6),
        "sharpe_ratio": round(sharpe, 4),
        "max_drawdown": round(mdd, 6),
    }


def _yearly(equity):
    by = {}
    for d, v in equity:
        dd = d.date() if hasattr(d, "date") else d
        by.setdefault(dd.year, []).append(float(v))
    out, prev = {}, None
    for y in sorted(by):
        vals = np.array(by[y])
        if len(vals) < 20:
            continue
        r0 = prev if prev else vals[0]
        out[str(y)] = round(float(vals[-1] / r0 - 1.0), 4)
        prev = float(vals[-1])
    return out


def main() -> None:
    codes = rsb.list_full_universe()
    print(f"[universe] 全市场 {len(codes)} 只, {len(GRID)} 组分年验证", flush=True)
    ex = BacktestExecutor(
        data_dir=str(settings.DATA_ROOT_PATH),
        enable_parallel=True, max_workers=8,
        enable_performance_profiling=False, use_multiprocessing=False,
    )
    cfg = BacktestConfig(
        initial_cash=1_000_000.0, commission_rate=0.0003, slippage_rate=0.0001,
        max_position_size=0.2, cash_reserve_ratio=0.05, board_lot_size=100,
        stop_loss_pct=0.05, take_profit_pct=0.15, rebalance_frequency="daily",
        open_cost=0.0, close_cost=0.0, min_cost=0.0,
    )
    results = {}
    for tag, scfg in GRID:
        stash: dict = {}
        install_stash(stash)
        t0 = time.perf_counter()
        try:
            res = asyncio.run(
                ex.run_backtest(
                    strategy_name="stochastic", stock_codes=codes,
                    start_date=START, end_date=END,
                    strategy_config=scfg, backtest_config=cfg,
                )
            )
            pm = stash.get("pm")
            eq = getattr(pm, "equity_curve", None) if pm else None
            rs = _ret_stats(eq) if eq else None
            yearly = _yearly(eq) if eq else {}
            results[tag] = {
                "config": scfg,
                "metrics_reported": {k: res.get(k) for k in ("total_return", "sharpe_ratio", "max_drawdown")},
                "engine": {k: res.get(k) for k in ("total_return", "annualized_return", "sharpe_ratio", "max_drawdown", "win_rate", "profit_factor", "total_trades")},
                "yearly": yearly,
                "wall_s": round(time.perf_counter() - t0, 1),
            }
            eng = results[tag]["engine"]
            print(
                f"[{tag}] sharpe={eng.get('sharpe_ratio')} ann={eng.get('annualized_return')} "
                f"trades={eng.get('total_trades')} yearly={yearly} wall={results[tag]['wall_s']}s",
                flush=True,
            )
        finally:
            uninstall()
        payload = {
            "meta": {
                "generated_at": datetime.now().isoformat(),
                "window": [str(START), str(END)],
                "universe": "full-market-5513",
                "note": "stochastic 信号变体全市场分年验证: 刀锋vs平台 + 分年稳定",
            },
            "results": results,
        }
        os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
        with open(OUT_JSON, "w") as f:
            json.dump(payload, f, ensure_ascii=False, indent=1, default=str)
    print(f"[saved] {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
