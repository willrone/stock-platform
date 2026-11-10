"""moving_average 参数修复的全市场复验（500 只是排序尾部选取，必须全市场验）。

诊断结论（500 只 / 3 年）：
  base(5/20, 阈值0.5%)   -5.6% / 0.187 / -66.2% / 690 笔
  成本归零               -5.2% / 0.189 / -66.1% / 725 笔   ← 不是成本问题
  阈值=0                +22.1% / 0.390 / -65.5% / 1974 笔
  长窗口 20/60          +61.3% / 0.622 / -46.6% /   79 笔  ← 根因：5/20 太短
  反向(死叉买)          +36.2% / 0.470 / -51.2% /    8 笔

→ 根因是**信号质量**（短周期均线交叉在 A 股是反向/噪音信号），不是交易成本。
→ 修复方向：拉长均线窗口 + 放宽阈值，把换手从 690 笔降到 ~100 笔。

本脚本在全市场 5513 只上对比 4 格：
  base_5_20_thr005      现状
  thr0                 只放宽阈值
  ma_20_60             只拉长窗口
  ma_20_60_thr0        两者都做（候选修复）
  ma_10_60_thr0        折中
"""
import asyncio
import json
import os
import sys
import time
from datetime import datetime

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND)
os.chdir(BACKEND)

from loguru import logger  # noqa

logger.remove()
logger.add(sys.stderr, level="ERROR")

from app.services.backtest import BacktestExecutor  # noqa: E402
from app.services.backtest.models import BacktestConfig  # noqa: E402

QLIB = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "qlib_data", "features", "day")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "strategy_baseline", "moving_average_fix_3y.json")


def universe():
    files = sorted(f[: -len(".parquet")] for f in os.listdir(QLIB) if f.endswith(".parquet"))
    return sorted(c.replace("_", ".") for c in files)


async def run(codes, sc, start, end):
    ex = BacktestExecutor(
        data_dir=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data"),
        enable_parallel=True, max_workers=8,
        enable_performance_profiling=False, use_multiprocessing=False,
    )
    t0 = time.perf_counter()
    res = await ex.run_backtest(
        strategy_name="moving_average", stock_codes=codes,
        start_date=start, end_date=end, strategy_config=sc,
        backtest_config=BacktestConfig(
            initial_cash=1_000_000.0, commission_rate=0.0003, slippage_rate=0.0001,
            max_position_size=0.2, cash_reserve_ratio=0.05, board_lot_size=100,
            stop_loss_pct=0.0, take_profit_pct=0.0, rebalance_frequency="daily",
        ),
    )
    m = res.get("metrics", {})
    hist = res.get("portfolio_history") or []
    yearly = {}
    if hist:
        import pandas as pd

        s = pd.Series(
            [h.get("portfolio_value") for h in hist],
            index=pd.to_datetime([h.get("date") for h in hist]),
        )
        first = float(s.iloc[0])
        yearly = {str(k)[:4]: round(float(v) / first - 1, 4)
                  for k, v in s.resample("YE").last().items()}
    return {
        "ret": m.get("total_return"), "sharpe": m.get("sharpe_ratio"),
        "mdd": m.get("max_drawdown"), "trades": m.get("total_trades"),
        "stocks": m.get("stocks_traded"), "yearly": yearly,
        "wall_s": round(time.perf_counter() - t0, 1),
    }


CELLS = [
    ("base_5_20_thr005", {"short_window": 5, "long_window": 20, "signal_threshold": 0.005}),
    ("thr0_5_20", {"short_window": 5, "long_window": 20, "signal_threshold": 0.0}),
    ("ma_20_60_thr005", {"short_window": 20, "long_window": 60, "signal_threshold": 0.005}),
    ("ma_20_60_thr0", {"short_window": 20, "long_window": 60, "signal_threshold": 0.0}),
    ("ma_10_60_thr0", {"short_window": 10, "long_window": 60, "signal_threshold": 0.0}),
]


async def main():
    codes = universe()
    print(f"universe={len(codes)}", flush=True)
    start, end = datetime(2023, 5, 12), datetime(2026, 5, 8)
    out = {"meta": {"window": [str(start.date()), str(end.date())],
                    "universe": len(codes)},
           "cells": []}
    for label, sc in CELLS:
        print(f"\n===== {label} =====", flush=True)
        r = await run(codes, sc, start, end)
        r["label"] = label
        r["config"] = sc
        out["cells"].append(r)
        print(f"{label}: ret={r['ret']} sharpe={r['sharpe']} mdd={r['mdd']} "
              f"trades={r['trades']} yearly={r['yearly']} wall={r['wall_s']}s",
              flush=True)
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        with open(OUT, "w") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=1, default=str)


asyncio.run(main())
