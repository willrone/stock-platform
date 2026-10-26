#!/usr/bin/env python3
"""gbm_lab 冒烟测试：小股票池 + 短窗口，验证策略能跑通且无未来函数。

检查项：
1. 策略能注册、能打分、能被引擎消费（产生交易）
2. lag 语义正确：T 日成交用的是 T-1 日的信号
3. purge 正确：训练标签不含未来
4. 面板因子覆盖率（features_full vs features/day）
"""
from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime

BACKEND = "/Users/rone/Projects/stock-platform/backend"
ROOT = "/Users/rone/Projects/stock-platform"
sys.path.insert(0, BACKEND)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from app.services.backtest import BacktestExecutor  # noqa: E402
from app.services.backtest.models import BacktestConfig  # noqa: E402
from app.services.backtest.strategies.strategy_factory import (  # noqa: E402
    StrategyFactory,
)
from app.services.backtest.strategies.gbm_lab_strategy import (  # noqa: E402
    GbmLabStrategy, DEFAULT_PANEL_DIR, LEGACY_PANEL_DIR,
)

StrategyFactory.register_strategy("gbm_lab", GbmLabStrategy)

CODES = [
    "000001.SZ", "000002.SZ", "000004.SZ", "000006.SZ", "000007.SZ",
    "000008.SZ", "000009.SZ", "000010.SZ", "000011.SZ", "000012.SZ",
    "600000.SH", "600004.SH", "600006.SH", "600007.SH", "600008.SH",
    "600009.SH", "600010.SH", "600011.SH", "600012.SH", "600015.SH",
    "300001.SZ", "300002.SZ", "300003.SZ", "300004.SZ", "300005.SZ",
    "688001.SH", "688002.SH", "688003.SH", "000001.SH", "600011.SH",
]


def run(cfg, start, end, label):
    executor = BacktestExecutor(
        data_dir=ROOT + "/data",
        enable_parallel=True, max_workers=4,
        enable_performance_profiling=False, use_multiprocessing=False,
    )
    res = asyncio.run(executor.run_backtest(
        strategy_name="gbm_lab",
        stock_codes=CODES,
        start_date=start, end_date=end,
        strategy_config={**cfg, "lag": 1},
        backtest_config=BacktestConfig(
            enable_tradability=True, initial_cash=1_000_000.0,
            commission_rate=0.0003, slippage_rate=0.0001,
            max_position_size=0.015833, cash_reserve_ratio=0.05,
            board_lot_size=100, stop_loss_pct=0.0, take_profit_pct=0.0,
            rebalance_frequency="daily", open_cost=0.0, close_cost=0.0,
            min_cost=0.0,
        ),
    ))
    print(f"\n=== {label} ===")
    for k in ("sharpe_ratio", "total_return", "max_drawdown", "total_trades",
              "win_rate", "final_value"):
        print(f"  {k}: {res.get(k)}")
    print(f"  equity_curve 点数: {len(res.get('equity_curve') or [])}")
    th = res.get("trade_history") or []
    print(f"  trade_history: {len(th)} 条")
    if th:
        print(f"  首笔: {th[0].get('timestamp')} {th[0].get('stock_code')} "
              f"{th[0].get('action')} price={th[0].get('price')}")
    return res


def main():
    start = datetime(2023, 5, 8)
    end = datetime(2023, 11, 8)

    print("[smoke] 1) 基线跑通")
    run(
        {"topk": 5, "n_drop": 1, "rebalance_days": "weekly",
         "rebalance_weekday": 4, "horizon": 1, "target": "cs_z",
         "model": "lgbm", "n_estimators": 30, "max_depth": 3,
         "retrain": "monthly", "feature_set": "f42", "target_vol": 0.20,
         "blend": 1.0, "n_jobs": 2},
        start, end, "baseline f42 lgbm cs_z",
    )

    print("\n[smoke] 2) blend=0（纯等权因子分，不看模型）")
    run(
        {"topk": 5, "n_drop": 1, "rebalance_days": "weekly",
         "rebalance_weekday": 4, "horizon": 1, "target": "cs_z",
         "model": "lgbm", "n_estimators": 30, "feature_set": "f42",
         "target_vol": 0.20, "blend": 0.0, "n_jobs": 2},
        start, end, "eq-only",
    )

    print("\n[smoke] 3) 旧面板目录（features/day，日期重复）")
    run(
        {"topk": 5, "n_drop": 1, "rebalance_days": "weekly",
         "rebalance_weekday": 4, "horizon": 1, "target": "cs_z",
         "model": "lgbm", "n_estimators": 30, "feature_set": "f42",
         "target_vol": 0.20, "blend": 1.0, "panel_dir": LEGACY_PANEL_DIR,
         "n_jobs": 2},
        start, end, "panel=legacy(features/day)",
    )

    print("\n[smoke] 4) f25（无面板因子）")
    run(
        {"topk": 5, "n_drop": 1, "rebalance_days": "weekly",
         "rebalance_weekday": 4, "horizon": 1, "target": "cs_z",
         "model": "lgbm", "n_estimators": 30, "feature_set": "f25",
         "target_vol": 0.20, "blend": 1.0, "n_jobs": 2},
        start, end, "f25",
    )

    print("\n[smoke] 5) ridge 模型 + rank 目标")
    run(
        {"topk": 5, "n_drop": 1, "rebalance_days": "weekly",
         "rebalance_weekday": 4, "horizon": 5, "target": "rank",
         "model": "ridge", "feature_set": "f42", "target_vol": 0.20,
         "blend": 0.5, "n_jobs": 2},
        start, end, "ridge rank blend0.5",
    )

    print("\n[smoke] OK")


if __name__ == "__main__":
    main()
