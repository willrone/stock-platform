#!/usr/bin/env python3
"""stochastic 信号变体 全市场(5513)外推验证。

500 只网格选出最优格 ov15_ob85_k14 (夏普 0.87 / 收益 +196.8%), 但 500 只是
统计套利专用宇宙(排序尾部选取), 非随机样本 → 存在宇宙过拟合风险。
本脚本在**与基线相同的全市场 5513 宇宙**上重跑候选格, 检验结论外推性:
  G0  默认 20/80 k14   (基线格, 参照)
  G1  15/85 k14        (网格最优: 夏普 0.87)
  G2  35/65 k21        (网格次优: 夏普 0.83, 高频稳健候选)
  G3  35/65 k10        (网格第三: 夏普 0.74)
口径: 同基线(100万/费率/仓位/窗口 2023-05-12→2026-05-08), 确定性引擎。
产物: data/strategy_baseline/stochastic_signal_fullmarket.json
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sys
import time
from datetime import datetime

BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BACKEND_ROOT)

import importlib  # noqa: E402

# 复用基线 runner 的宇宙/配置函数(保证与基线完全同口径)
_spec = importlib.util.spec_from_file_location("rsb", os.path.join(BACKEND_ROOT, "scripts", "run_strategy_baseline.py"))
rsb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rsb)

from app.core.config import settings  # noqa: E402
from app.services.backtest import BacktestExecutor  # noqa: E402
from app.services.backtest.models import BacktestConfig  # noqa: E402

DATA_ROOT = "/Users/rone/projects/stock-platform/data"
OUT_JSON = os.path.join(DATA_ROOT, "strategy_baseline", "stochastic_signal_fullmarket.json")
START, END = datetime(2023, 5, 12), datetime(2026, 5, 8)

GRID = [
    ("G0_默认20_80_k14", {"oversold": 20, "overbought": 80, "k_period": 14, "d_period": 3}),
    ("G1_15_85_k14", {"oversold": 15, "overbought": 85, "k_period": 14, "d_period": 3}),
    ("G2_35_65_k21", {"oversold": 35, "overbought": 65, "k_period": 21, "d_period": 3}),
    ("G3_35_65_k10", {"oversold": 35, "overbought": 65, "k_period": 10, "d_period": 3}),
]


def main() -> None:
    codes = rsb.list_full_universe()
    print(f"[universe] 全市场 {len(codes)} 只, 网格 {len(GRID)} 组", flush=True)
    ex = BacktestExecutor(
        data_dir=str(settings.DATA_ROOT_PATH),
        enable_parallel=True,
        max_workers=8,
        enable_performance_profiling=False,
        use_multiprocessing=False,
    )
    cfg = BacktestConfig(
        initial_cash=1_000_000.0,
        commission_rate=0.0003,
        slippage_rate=0.0001,
        max_position_size=0.2,
        cash_reserve_ratio=0.05,
        board_lot_size=100,
        stop_loss_pct=0.05,
        take_profit_pct=0.15,
        rebalance_frequency="daily",
        open_cost=0.0,
        close_cost=0.0,
        min_cost=0.0,
    )
    results = {}
    for tag, scfg in GRID:
        t0 = time.perf_counter()
        try:
            res = asyncio.run(
                ex.run_backtest(
                    strategy_name="stochastic",
                    stock_codes=codes,
                    start_date=START,
                    end_date=END,
                    strategy_config=scfg,
                    backtest_config=cfg,
                )
            )
        except Exception as e:  # noqa: BLE001
            print(f"[{tag}] FAILED: {e}", flush=True)
            results[tag] = {"config": scfg, "error": str(e)}
            continue
        results[tag] = {
            "config": scfg,
            "universe_size": len(codes),
            "metrics": {k: res.get(k) for k in (
                "total_return", "annualized_return", "sharpe_ratio",
                "max_drawdown", "win_rate", "profit_factor", "total_trades",
            )},
            "wall_s": round(time.perf_counter() - t0, 1),
        }
        m = results[tag]["metrics"]
        print(
            f"[{tag}] ret={m.get('total_return')} ann={m.get('annualized_return')} "
            f"sharpe={m.get('sharpe_ratio')} mdd={m.get('max_drawdown')} "
            f"trades={m.get('total_trades')} win={m.get('win_rate')} pf={m.get('profit_factor')} "
            f"wall={results[tag]['wall_s']}s",
            flush=True,
        )
        payload = {
            "meta": {
                "generated_at": datetime.now().isoformat(),
                "window": [str(START), str(END)],
                "universe": "full-market",
                "note": "stochastic 信号变体全市场外推验证",
            },
            "results": results,
        }
        os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
        with open(OUT_JSON, "w") as f:
            json.dump(payload, f, ensure_ascii=False, indent=1, default=str)
    print(f"[saved] {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
