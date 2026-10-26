#!/usr/bin/env python3
"""用**平台原生 BacktestExecutor** 独立验证「42 因子 + 行业中性」的效果。

## 为什么要有这个脚本
子代理的实验用的是自建快速模拟器（已与原生引擎逐笔对账过基线），
但那条变体代码在 scratch 里、没进项目。而本次要把行业中性落地成
**正式策略线**，必须用项目自己的引擎端到端跑通，不能只靠复刻版结论。

## 对比
  A. honest_composite（基线，industry_neutral=False）
  B. honest_composite（industry_neutral=True）

两者共用同一份数据、同一套成本参数、同一执行器，**唯一差异是行业中性开关**。

用法：
  env -u PYTHONPATH backend/.venv-py313/bin/python3 \
    backend/scripts/verify_industry_neutral.py
"""
from __future__ import annotations

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

# 注意：loader 内部还会拼 "parquet/stock_data"，所以这里要给**项目根的 data 目录**
DATA_ROOT = os.path.join(ROOT, "data")
# 引擎的股票代码来自 Qlib 预计算目录（文件名 <code>_<SH|SZ|BJ>.parquet）
QLIB_DAY = os.path.join(ROOT, "data", "qlib_data", "features", "day")
OUT = os.path.join(ROOT, "data", "backtest", "industry_neutral_verify.json")


def universe() -> List[str]:
    """引擎口径的股票池：Qlib 预计算目录下的 <code>_<SH|SZ|BJ>.parquet。"""
    if os.path.isdir(QLIB_DAY):
        codes = []
        for f in sorted(os.listdir(QLIB_DAY)):
            if not f.endswith(".parquet"):
                continue
            stem = f[: -len(".parquet")]
            if "_" not in stem:
                continue
            code, _, market = stem.partition("_")
            if market in ("SH", "SZ", "BJ") and code.isdigit():
                codes.append(f"{code}.{market}")
        if codes:
            return sorted(codes)
    # 兜底：从 stock_data 目录扫
    codes = []
    for f in sorted(os.listdir(os.path.join(DATA_ROOT, "parquet", "stock_data"))):
        if not f.endswith(".parquet"):
            continue
        stem = f[: -len(".parquet")]
        if "_" not in stem:
            continue
        code, _, market = stem.partition("_")
        if market in ("SH", "SZ", "BJ") and code.isdigit():
            codes.append(f"{code}.{market}")
    return sorted(codes)


def run_one(codes: List[str], neutral: bool) -> Dict[str, Any]:
    executor = BacktestExecutor(
        data_dir=DATA_ROOT,
        enable_parallel=True,
        max_workers=8,
        enable_performance_profiling=False,
        use_multiprocessing=False,
    )
    cfg = {
        "topk": 60,
        "n_drop": 5,
        "rebalance_days": "weekly",
        "rebalance_weekday": 4,
        "industry_neutral": neutral,
    }
    t0 = time.perf_counter()
    res = asyncio.run(
        executor.run_backtest(
            strategy_name="honest_composite",
            stock_codes=codes,
            # 必须是 datetime 对象 —— 传字符串会让引擎的日期覆盖校验失败
            # （_is_data_valid 拿字符串和 Timestamp 比 → 全部判为无效数据）
            start_date=datetime(2023, 5, 12),
            end_date=datetime(2026, 5, 8),
            strategy_config=cfg,
            backtest_config=BacktestConfig(
                initial_cash=1_000_000.0,
                commission_rate=0.0003,
                slippage_rate=0.0001,
                max_position_size=(1.0 - 0.05) / 60,
                cash_reserve_ratio=0.05,
                board_lot_size=100,
                stop_loss_pct=0.05,
                take_profit_pct=0.15,
                rebalance_frequency="daily",
                open_cost=0.0,
                close_cost=0.0,
                min_cost=0.0,
            ),
        )
    )
    wall = time.perf_counter() - t0
    out = {
        "industry_neutral": neutral,
        "wall_s": round(wall, 1),
    }
    for k in ("total_return", "annual_return", "sharpe_ratio", "max_drawdown",
              "total_trades", "win_rate", "volatility", "final_value",
              "avg_positions", "calmar_ratio"):
        if k in res:
            out[k] = res[k]
    pb = res.get("perf_breakdown") or {}
    if isinstance(pb, dict):
        out["perf_breakdown"] = {k: v for k, v in pb.items()
                                 if not isinstance(v, (list, dict))}
    yearly = res.get("yearly_returns") or res.get("yearly") or {}
    if isinstance(yearly, dict):
        out["yearly"] = {str(k): v for k, v in yearly.items()}
    eq = res.get("equity_curve") or res.get("equity") or []
    if eq:
        out["equity_points"] = len(eq)
    return out


def main() -> None:
    codes = universe()
    print(f"[verify] universe={len(codes)}", flush=True)
    results = {}
    for neutral in (False, True):
        label = "行业中性" if neutral else "基线"
        print(f"\n[verify] 跑 {label} …", flush=True)
        try:
            r = run_one(codes, neutral)
        except Exception as e:
            import traceback

            traceback.print_exc()
            r = {"industry_neutral": neutral, "error": f"{type(e).__name__}: {e}"}
        results[label] = r
        print(f"[verify] {label} 完成 ({r.get('wall_s')}s): "
              f"sharpe={r.get('sharpe_ratio')} ret={r.get('total_return')} "
              f"mdd={r.get('max_drawdown')} trades={r.get('total_trades')}",
              flush=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=1, default=str)
    print(f"\n[verify] 已写入 {OUT}", flush=True)


if __name__ == "__main__":
    main()
