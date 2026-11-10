
"""引擎 OOS 分段验证：对指定配置，用真实引擎分别跑 IS(2023-05→2025-05) 与
OOS(2025-05→2026-05) 两段，报告两段的夏普/收益/回撤/笔数。

用法:
  env -u PYTHONPATH -u PYTHONHOME backend/.venv-py313/bin/python3 \
    backend/scripts/verify_ensemble_engine_oos.py --cases /tmp/ens_cases.json
"""
from __future__ import annotations
import os

import argparse, asyncio, json, os, sys, time
from datetime import datetime
from typing import Any, Dict, List

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(BACKEND)
sys.path.insert(0, BACKEND); sys.path.insert(0, os.path.join(BACKEND, "scripts"))
import ensemble_lab as L

IS_START, IS_END = datetime(2023, 5, 8), datetime(2025, 5, 8)
OOS_START, OOS_END = datetime(2025, 5, 8), datetime(2026, 5, 8)


def register_ensemble() -> None:
    from app.services.backtest.strategies.ensemble_strategy import EnsembleStrategy
    from app.services.backtest.strategies.strategy_factory import StrategyFactory
    StrategyFactory.register_strategy("ensemble", EnsembleStrategy)


def run(cfg: Dict[str, Any], codes, start, end) -> Dict[str, Any]:
    from app.services.backtest import BacktestExecutor
    from app.services.backtest.models import BacktestConfig
    ex = BacktestExecutor(data_dir=ROOT + "/data", enable_parallel=True,
                          max_workers=8, enable_performance_profiling=False,
                          use_multiprocessing=False)
    t0 = time.perf_counter()
    res = asyncio.run(ex.run_backtest(
        strategy_name="ensemble", stock_codes=codes,
        start_date=start, end_date=end, strategy_config=cfg,
        backtest_config=BacktestConfig(
            enable_tradability=True, initial_cash=1_000_000.0,
            commission_rate=0.0003, slippage_rate=0.0001,
            max_position_size=cfg["_mps"], cash_reserve_ratio=0.05,
            board_lot_size=100, stop_loss_pct=0.0, take_profit_pct=0.0,
            rebalance_frequency="daily", open_cost=0.0, close_cost=0.0, min_cost=0.0),
    ))
    return {"sharpe_ratio": res.get("sharpe_ratio"), "total_return": res.get("total_return"),
            "max_drawdown": res.get("max_drawdown"), "total_trades": res.get("total_trades"),
            "yearly": res.get("yearly"), "wall_s": round(time.perf_counter()-t0,1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--out", default="ensemble_engine_oos.json")
    args = ap.parse_args()
    cases = json.load(open(args.cases))
    register_ensemble()
    codes = L.universe_from_instruments()
    print(f"[setup] universe {len(codes)}", flush=True)
    results = []
    for case in cases:
        cfg = {"topk": case["topk"], "n_drop": case["n_drop"],
               "rebalance_days": case.get("rebalance_days","weekly"),
               "rebalance_weekday": 4, "lag": 1, "legs": case["legs"],
               "normalize": case.get("normalize","z"),
               "weight_mode": case.get("weight_mode","fixed")}
        if case.get("weights"): cfg["weights"] = case["weights"]
        if case.get("residualize"): cfg["residualize"] = True
        if case.get("weight_mode") == "icir":
            cfg["icir_window"] = case.get("icir_window",120); cfg["icir_smooth"]=case.get("icir_smooth",20)
        cfg["_mps"] = case["max_position_size"]
        print(f"\n=== {case['name']} ===", flush=True)
        isr = run(cfg, codes, IS_START, IS_END)
        print(f"  IS  SR={isr['sharpe_ratio']} ret={isr['total_return']:+.2%} "
              f"mdd={isr['max_drawdown']:.2%} n={isr['total_trades']} ({isr['wall_s']}s)", flush=True)
        oosr = run(cfg, codes, OOS_START, OOS_END)
        print(f"  OOS SR={oosr['sharpe_ratio']} ret={oosr['total_return']:+.2%} "
              f"mdd={oosr['max_drawdown']:.2%} n={oosr['total_trades']} ({oosr['wall_s']}s)", flush=True)
        results.append({"case": case, "IS": isr, "OOS": oosr})
    path = os.path.join(L.OUT_DIR, args.out)
    json.dump({"results": results}, open(path,"w"), ensure_ascii=False, indent=1, default=str)
    print(f"\n写入 {path}")

if __name__ == "__main__":
    main()
