#!/usr/bin/env python3
"""新策略 `methods_composite` 的**真实引擎**验证（真实盘口径）。

## 为什么不用 verify_10y_pit.py 直接跑
`methods_composite` 是**新策略文件**，按任务要求**不改注册表**
（`strategy_factory.py` / `verify_10y_pit.py` 都不动）。
本脚本用 `StrategyFactory.register_strategy()`（公开 API）在**运行时**
注册，然后调用**与 verify_10y_pit.py 完全相同的引擎路径与配置**：
  - 股票池 data/qlib_official_data/instruments/all.txt（PIT 上市/退市日）
  - BacktestConfig: commission 0.0003 / slippage 0.0001 / 整手 100 /
    现金预留 5% / enable_tradability=True / max_position_size=(1-0.05)/topk
  - lag=1
  - 窗口 2023-05-08 → 2026-05-08（--years 3）

## 跑哪些变体
  1. methods_composite(exposure_neutral="none")   ← 与 honest_composite 对照
  2. methods_composite(exposure_neutral="size+illiq")
  3. methods_composite(exposure_neutral="size+illiq", turnover_blend=1.0)
  4. honest_composite（引擎原生基线，用于口径核对）

## 用法
  env -u PYTHONPATH -u PYTHONHOME backend/.venv-py313/bin/python3 \
    backend/scripts/verify_methods_engine.py --years 3 --lag 1
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

BACKEND = "/Users/rone/Projects/stock-platform/backend"
ROOT = "/Users/rone/Projects/stock-platform"
sys.path.insert(0, BACKEND)

from app.services.backtest import BacktestExecutor  # noqa: E402
from app.services.backtest.models import BacktestConfig  # noqa: E402
from app.services.backtest.strategies.strategy_factory import (  # noqa: E402
    StrategyFactory,
)
from app.services.backtest.strategies.methods_composite_strategy import (  # noqa: E402
    MethodsCompositeStrategy,
)

QLIB_OFFICIAL = os.path.join(ROOT, "data", "qlib_official_data")
OUT_DIR = os.path.join(ROOT, "data", "backtest")

# 运行时注册（不改 strategy_factory.py）
StrategyFactory.register_strategy("methods_composite", MethodsCompositeStrategy)

VARIANTS: List[Dict[str, Any]] = [
    {
        "name": "methods_composite_neutral_none",
        "strategy": "methods_composite",
        "cfg": {"topk": 60, "n_drop": 5, "rebalance_days": "weekly",
                "rebalance_weekday": 4, "exposure_neutral": "none",
                "turnover_blend": 0.0},
        "mps": 0.015833,
        "label": "新策略对照组（未中性化）",
    },
    {
        "name": "methods_composite_neutral_size_illiq",
        "strategy": "methods_composite",
        "cfg": {"topk": 60, "n_drop": 5, "rebalance_days": "weekly",
                "rebalance_weekday": 4, "exposure_neutral": "size+illiq",
                "turnover_blend": 0.0},
        "mps": 0.015833,
        "label": "42 因子 + 规模/流动性中性化",
    },
    {
        "name": "methods_composite_neutral_si_to1",
        "strategy": "methods_composite",
        "cfg": {"topk": 60, "n_drop": 5, "rebalance_days": "weekly",
                "rebalance_weekday": 4, "exposure_neutral": "size+illiq",
                "turnover_blend": 1.0},
        "mps": 0.015833,
        "label": "42 因子 + 中性化 + 低换手因子(w=1)",
    },
    {
        "name": "methods_composite_neutral_si_to1_k40",
        "strategy": "methods_composite",
        "cfg": {"topk": 40, "n_drop": 3, "rebalance_days": "weekly",
                "rebalance_weekday": 4, "exposure_neutral": "size+illiq",
                "turnover_blend": 1.0},
        "mps": 0.02375,
        "label": "同上 topk=40",
    },
    {
        "name": "honest_composite_baseline",
        "strategy": "honest_composite",
        "cfg": {"topk": 60, "n_drop": 5, "rebalance_days": "weekly",
                "rebalance_weekday": 4},
        "mps": 0.015833,
        "label": "引擎原生基线（42 因子）",
    },
]


def universe_from_instruments(instruments_dir: str) -> List[str]:
    path = os.path.join(instruments_dir, "instruments", "all.txt")
    codes: List[str] = []
    with open(path) as fh:
        for line in fh:
            parts = line.strip().split("\t")
            if not parts:
                continue
            name = parts[0]
            if "_" not in name:
                continue
            code, _, mkt = name.partition("_")
            if mkt.upper() in ("SH", "SZ", "BJ") and code.isdigit():
                codes.append(f"{code}.{mkt.upper()}")
    return sorted(codes)


def run_one(
    data_root: str,
    codes: List[str],
    strategy: str,
    cfg: Dict[str, Any],
    mps: float,
    start: datetime,
    end: datetime,
    lag: int,
) -> Dict[str, Any]:
    executor = BacktestExecutor(
        data_dir=data_root, enable_parallel=True, max_workers=8,
        enable_performance_profiling=False, use_multiprocessing=False,
    )
    t0 = time.perf_counter()
    try:
        res = asyncio.run(
            executor.run_backtest(
                strategy_name=strategy,
                stock_codes=codes,
                start_date=start,
                end_date=end,
                strategy_config={**cfg, "lag": lag},
                backtest_config=BacktestConfig(
                    enable_tradability=True,
                    initial_cash=1_000_000.0,
                    commission_rate=0.0003,
                    slippage_rate=0.0001,
                    max_position_size=mps,
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
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        return {"strategy": strategy, "error": f"{type(e).__name__}: {e}",
                "wall_s": round(time.perf_counter() - t0, 1)}

    out: Dict[str, Any] = {
        "strategy": strategy,
        "wall_s": round(time.perf_counter() - t0, 1),
        "requested_stocks": len(codes),
    }
    for k in ("total_return", "annual_return", "sharpe_ratio", "max_drawdown",
              "total_trades", "win_rate", "volatility", "final_value",
              "avg_positions", "calmar_ratio", "annualized_return"):
        if k in res:
            out[k] = res[k]
    yearly = res.get("yearly_returns") or res.get("yearly") or {}
    if isinstance(yearly, dict) and yearly:
        out["yearly"] = {str(k): round(float(v), 6) for k, v in yearly.items()}
    yd = res.get("yearly_returns_detail")
    if isinstance(yd, list) and yd:
        out["yearly"] = {
            str(int(x["year"])): round(float(x["return"]), 6) for x in yd}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=3)
    ap.add_argument("--lag", type=int, default=1)
    ap.add_argument("--out", default="methods_engine.json")
    ap.add_argument("--only", nargs="*", default=[])
    args = ap.parse_args()

    end = datetime(2026, 5, 8)
    start = datetime(end.year - args.years, end.month, end.day)
    data_root = ROOT + "/data"
    codes = universe_from_instruments(QLIB_OFFICIAL)
    print(f"[engine] 窗口 {start.date()} → {end.date()}  "
          f"股票池 {len(codes)} 只  lag={args.lag}", flush=True)

    todo = args.only or [v["name"] for v in VARIANTS]
    results: Dict[str, Any] = {}
    for v in VARIANTS:
        if v["name"] not in todo:
            continue
        print(f"\n[engine] === {v['name']} ({v['label']}) ===", flush=True)
        r = run_one(data_root, codes, v["strategy"], v["cfg"], v["mps"],
                    start, end, args.lag)
        r["label"] = v["label"]
        r["config"] = v["cfg"]
        results[v["name"]] = r
        print(f"  SR={r.get('sharpe_ratio')} ret={r.get('total_return')} "
              f"mdd={r.get('max_drawdown')} trades={r.get('total_trades')} "
              f"win={r.get('win_rate')} ({r.get('wall_s')}s)", flush=True)
        if r.get("yearly"):
            print(f"  分年: {r['yearly']}", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, args.out)
    with open(path, "w") as fh:
        json.dump(
            {"window": f"{start.date()}→{end.date()}", "lag": args.lag,
             "n_codes": len(codes), "results": results},
            fh, ensure_ascii=False, indent=1, default=str)
    print(f"\n[engine] 已写入 {path}")


if __name__ == "__main__":
    main()
