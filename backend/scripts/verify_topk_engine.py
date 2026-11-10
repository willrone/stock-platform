#!/usr/bin/env python3
"""真实引擎交叉验证：topk 敏感性 + 各腿 lag=1 参考值。

## 为什么需要
快速模拟器（ensemble_lab.py）跑几十个变体，但它的结论必须用**真实引擎**
复核，否则模拟器的保真度问题会被当成 alpha。

本脚本用 `BacktestExecutor`（与 verify_10y_pit.py 完全同源）跑：
  1. 各策略在 **lag=1** 下的参考值（基线文档里的 rsi 1.015 是 lag=0 的数）
  2. honest_composite 的 **topk 扫描**（模拟器说 topk=20 → 夏普 1.03）
  3. 面板目录口径对照（features/day 两段拼接 vs features_full 交易日格）

## 口径
窗口 2023-05-08 → 2026-05-08 / PIT 股票池 / lag=1 /
enable_tradability=True / TopK-Dropout 等权 / 佣金 0.03% + 滑点 0.01% /
整手 100 股 / 现金预留 5%

## 用法
  env -u PYTHONPATH -u PYTHONHOME backend/.venv-py313/bin/python3 \
    backend/scripts/verify_topk_engine.py --which topk refs
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(BACKEND)
sys.path.insert(0, BACKEND)

from app.services.backtest import BacktestExecutor  # noqa: E402
from app.services.backtest.models import BacktestConfig  # noqa: E402

QLIB_OFFICIAL = os.path.join(ROOT, "data", "qlib_official_data")
OUT_DIR = os.path.join(ROOT, "data", "backtest")


def universe_from_instruments(instruments_dir: str = QLIB_OFFICIAL) -> List[str]:
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
    codes: List[str],
    strategy: str,
    cfg: Dict[str, Any],
    mps: float,
    start: datetime,
    end: datetime,
    lag: int = 1,
    enable_tradability: bool = True,
) -> Dict[str, Any]:
    executor = BacktestExecutor(
        data_dir=os.path.join(ROOT, "data"),
        enable_parallel=True,
        max_workers=8,
        enable_performance_profiling=False,
        use_multiprocessing=False,
    )
    t0 = time.perf_counter()
    try:
        res = asyncio.run(
            executor.run_backtest(
                strategy_name=strategy,
                stock_codes=codes,
                start_date=start,
                end_date=end,
                strategy_config={**dict(cfg), "lag": lag},
                backtest_config=BacktestConfig(
                    enable_tradability=enable_tradability,
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
    except Exception as exc:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        return {"strategy": strategy, "error": f"{type(exc).__name__}: {exc}",
                "wall_s": round(time.perf_counter() - t0, 1)}

    out: Dict[str, Any] = {
        "strategy": strategy,
        "strategy_config": {**dict(cfg), "lag": lag},
        "max_position_size": mps,
        "wall_s": round(time.perf_counter() - t0, 1),
    }
    for k in ("total_return", "annual_return", "sharpe_ratio", "max_drawdown",
              "total_trades", "win_rate", "volatility", "final_value",
              "annualized_return", "avg_positions"):
        if k in res:
            out[k] = res[k]
    yd = res.get("yearly_returns_detail")
    if isinstance(yd, list) and yd:
        out["yearly"] = {str(int(x["year"])): round(float(x["return"]), 6)
                         for x in yd}
    elif isinstance(res.get("yearly"), dict) and res.get("yearly"):
        out["yearly"] = {str(k): round(float(v), 6)
                         for k, v in res["yearly"].items()}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", nargs="*", default=["topk", "refs"],
                    choices=["topk", "refs", "panel"])
    ap.add_argument("--out", default="engine_topk_verify.json")
    args = ap.parse_args()

    codes = universe_from_instruments()
    end = datetime(2026, 5, 8)
    start = datetime(2023, 5, 8)
    print(f"[engine] universe {len(codes)}  window {start.date()}→{end.date()}",
          flush=True)

    results: Dict[str, Any] = {
        "window": f"{start.date()}→{end.date()}",
        "caliber": {"lag": 1, "enable_tradability": True,
                    "commission": 0.0003, "slippage": 0.0001,
                    "cash_reserve_ratio": 0.05, "board_lot_size": 100},
        "n_codes": len(codes),
        "runs": [],
    }

    if "refs" in args.which:
        print("\n=== A. 各策略 lag=1 参考值（真实引擎）===", flush=True)
        refs = [
            ("honest_composite", {"topk": 60, "n_drop": 5,
                                  "rebalance_days": "weekly",
                                  "rebalance_weekday": 4}, 0.015833),
            ("rsi_71421", {"topk": 30, "n_drop": 3, "rebalance_days": "weekly",
                           "rebalance_weekday": 4}, 0.031667),
            ("factor_ranking", {"topk": 50, "n_drop": 5,
                                "rebalance_days": "weekly",
                                "rebalance_weekday": 4}, 0.019),
        ]
        for strat, cfg, mps in refs:
            r = run_one(codes, strat, cfg, mps, start, end, lag=1)
            results["runs"].append(r)
            print(f"  {strat:20s} SR={r.get('sharpe_ratio')} "
                  f"ret={r.get('total_return')} mdd={r.get('max_drawdown')} "
                  f"n={r.get('total_trades')} yearly={r.get('yearly')} "
                  f"({r.get('wall_s')}s)", flush=True)

    if "topk" in args.which:
        print("\n=== B. honest_composite topk 扫描（lag=1, 真实引擎）===",
              flush=True)
        for topk in (20, 30, 40, 60, 100):
            n_drop = max(2, topk // 12)
            mps = (1 - 0.05) / topk
            r = run_one(codes, "honest_composite",
                        {"topk": topk, "n_drop": n_drop,
                         "rebalance_days": "weekly", "rebalance_weekday": 4},
                        mps, start, end, lag=1)
            results["runs"].append(r)
            print(f"  topk={topk:3d} n_drop={n_drop} mps={mps:.6f} "
                  f"SR={r.get('sharpe_ratio')} ret={r.get('total_return')} "
                  f"mdd={r.get('max_drawdown')} n={r.get('total_trades')} "
                  f"yearly={r.get('yearly')} ({r.get('wall_s')}s)", flush=True)

    if "panel" in args.which:
        print("\n=== C. 面板目录口径对照（诚实：features/day 两段拼接 vs "
              "features_full 交易日格）===", flush=True)
        for pdir in (os.path.join(ROOT, "data", "qlib_data", "features", "day"),
                     os.path.join(ROOT, "data", "qlib_data", "features_full")):
            r = run_one(codes, "honest_composite",
                        {"topk": 60, "n_drop": 5, "rebalance_days": "weekly",
                         "rebalance_weekday": 4, "panel_dir": pdir},
                        0.015833, start, end, lag=1)
            r["panel_dir"] = pdir
            results["runs"].append(r)
            print(f"  panel_dir={os.path.basename(pdir):16s} "
                  f"SR={r.get('sharpe_ratio')} ret={r.get('total_return')} "
                  f"mdd={r.get('max_drawdown')} n={r.get('total_trades')} "
                  f"yearly={r.get('yearly')} ({r.get('wall_s')}s)", flush=True)

    path = os.path.join(OUT_DIR, args.out)
    with open(path, "w") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=1, default=str)
    print(f"\n写入 {path}")


if __name__ == "__main__":
    main()
