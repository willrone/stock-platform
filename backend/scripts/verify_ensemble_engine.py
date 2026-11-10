#!/usr/bin/env python3
"""用**真实引擎**验证合成策略的候选配置（不修改注册表）。

## 为什么需要这一步
快速模拟器与引擎有已知的 −0.05 级偏差（42f 腿），单腿结论可靠，
但「合成后夏普跳到 1.3」这种大幅提升必须用引擎复核，否则可能是模拟器
执行语义的假象。

## 做法
在本进程内 `StrategyFactory.register_strategy("ensemble", EnsembleStrategy)`
注册新策略（**只改内存里的注册表**，不改 strategy_factory.py 文件），
然后用 BacktestExecutor 跑真实盘口径。

用法:
  env -u PYTHONPATH -u PYTHONHOME backend/.venv-py313/bin/python3 \
    backend/scripts/verify_ensemble_engine.py \
      --case "42f+mf|winsor3|k30|d5" --legs 42f mf --normalize winsor3 \
      --topk 30 --n-drop 5
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(BACKEND)
sys.path.insert(0, BACKEND)
sys.path.insert(0, os.path.join(BACKEND, "scripts"))

import ensemble_lab as L  # noqa: E402

END = datetime(2026, 5, 8)
START = datetime(2023, 5, 8)


def register_ensemble() -> None:
    """在内存里注册 ensemble 策略（不改 strategy_factory.py）。"""
    from app.services.backtest.strategies.ensemble_strategy import EnsembleStrategy
    from app.services.backtest.strategies.strategy_factory import StrategyFactory
    StrategyFactory.register_strategy("ensemble", EnsembleStrategy)
    print("[reg] ensemble 已注册（仅本进程内存）", flush=True)


def run_case(case: Dict[str, Any], codes: List[str]) -> Dict[str, Any]:
    from app.services.backtest import BacktestExecutor
    from app.services.backtest.models import BacktestConfig

    cfg: Dict[str, Any] = {
        "topk": case["topk"],
        "n_drop": case["n_drop"],
        "rebalance_days": case.get("rebalance_days", "weekly"),
        "rebalance_weekday": 4,
        "lag": 1,
        "legs": case["legs"],
        "normalize": case.get("normalize", "z"),
        "weight_mode": case.get("weight_mode", "fixed"),
    }
    if case.get("weights"):
        cfg["weights"] = case["weights"]
    if case.get("residualize"):
        cfg["residualize"] = True
    if case.get("weight_mode") == "icir":
        cfg["icir_window"] = case.get("icir_window", 120)
        cfg["icir_smooth"] = case.get("icir_smooth", 20)

    executor = BacktestExecutor(data_dir=ROOT + "/data", enable_parallel=True,
                                max_workers=8, enable_performance_profiling=False,
                                use_multiprocessing=False)
    t0 = time.perf_counter()
    res = asyncio.run(
        executor.run_backtest(
            strategy_name="ensemble",
            stock_codes=codes,
            start_date=START,
            end_date=END,
            strategy_config=cfg,
            backtest_config=BacktestConfig(
                enable_tradability=True,
                initial_cash=1_000_000.0,
                commission_rate=0.0003,
                slippage_rate=0.0001,
                max_position_size=case["max_position_size"],
                cash_reserve_ratio=0.05,
                board_lot_size=100,
                stop_loss_pct=0.0,
                take_profit_pct=0.0,
                rebalance_frequency="daily",
                open_cost=0.0,
                close_cost=0.0,
                min_cost=0.0,
            ),
        )
    )
    out = {
        "case": case,
        "wall_s": round(time.perf_counter() - t0, 1),
        "total_return": res.get("total_return"),
        "sharpe_ratio": res.get("sharpe_ratio"),
        "max_drawdown": res.get("max_drawdown"),
        "total_trades": res.get("total_trades"),
        "win_rate": res.get("win_rate"),
        "volatility": res.get("volatility"),
        "yearly": res.get("yearly"),
        "monthly": res.get("monthly"),
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True,
                    help="JSON 文件：[{name, legs, normalize, weight_mode, "
                         "weights, topk, n_drop, max_position_size}, ...]")
    ap.add_argument("--out", default="ensemble_engine.json")
    args = ap.parse_args()

    with open(args.cases) as fh:
        cases = json.load(fh)
    register_ensemble()
    codes = L.universe_from_instruments()
    print(f"[setup] universe {len(codes)}", flush=True)

    results = []
    for case in cases:
        print(f"\n=== {case['name']} ===", flush=True)
        r = run_case(case, codes)
        results.append(r)
        print(f"  SR={r['sharpe_ratio']} ret={r['total_return']:+.2%} "
              f"mdd={r['max_drawdown']:.2%} n={r['total_trades']} "
              f"({r['wall_s']}s)", flush=True)
        print(f"  yearly: {r['yearly']}", flush=True)

    path = os.path.join(L.OUT_DIR, args.out)
    with open(path, "w") as fh:
        json.dump({"results": results}, fh, ensure_ascii=False, indent=1,
                  default=str)
    print(f"\n写入 {path}")


if __name__ == "__main__":
    main()
