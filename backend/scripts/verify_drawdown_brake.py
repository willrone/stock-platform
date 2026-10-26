#!/usr/bin/env python3
"""回撤刹车（组合层择时）10 年 A/B 验证 —— 2026-10-26

## 动机
10 年 PIT 回测里**所有策略 2018 年都亏 24%~42%** —— 没有一条线能躲开
系统性熊市。42f 十年回撤 −39.4%，这个数字放到实盘用户不会接受。

## 测什么
在选股之上加一层「满仓还是减仓」，不预测涨跌，只在市场状态差时砍仓：
  - `breadth`  市场宽度（当日上涨股票占比）< 阈值 → 减仓
  - `ma`       等权指数收盘 < 其 N 日均线 → 减仓
减仓方式：把超出保留比例的分数置 NaN（目标权重不变 → 剩余资金留现金）

## 防过拟合
1. 变体尽量少、参数取**自然值**（宽度 30%/40%、均线 20/60 日）
2. 必看**分年**结果 —— 只在某一年变好的不算数
3. 不看"夏普最高"就选，要看**回撤是否真的降**且**收益没塌**

用法：
  env -u PYTHONPATH backend/.venv-py313/bin/python3 \
    backend/scripts/verify_drawdown_brake.py
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

QLIB_OFFICIAL = os.path.join(ROOT, "data", "qlib_official_data")
OUT = os.path.join(ROOT, "data", "backtest", "drawdown_brake_10y.json")

#: 变体：(标签, strategy_config 覆盖, backtest_config 覆盖)
#: 2026-10-26 第一轮：市场状态择时（宽度 / 均线）**全部变差** → 已证伪
#:   宽度<40%: 夏普 0.549 / 宽度<30%: 0.576 / MA20: 0.559 / MA60: 0.584
#:   （基线 0.670，回撤几乎没改善，收益掉 25-40%）
#:   原因：宽度与指数均线反映的是**大盘 beta**，而 2018 年杀的是小盘估值；
#:   减仓期间反而错过 2019 年的报复性反弹。
#: 第二轮：改测**引擎原生的个股止损/止盈**（BacktestConfig 一直支持，
#:   之前基线配置里传的是 0 = 关闭）
VARIANTS: List[tuple] = [
    ("基线（无止损无择时）", {}, {}),
    ("个股止损5%", {}, {"stop_loss_pct": 0.05, "take_profit_pct": 0.0}),
    ("个股止损8%", {}, {"stop_loss_pct": 0.08, "take_profit_pct": 0.0}),
    ("个股止损10%", {}, {"stop_loss_pct": 0.10, "take_profit_pct": 0.0}),
    ("个股止损5%+止盈15%", {}, {"stop_loss_pct": 0.05, "take_profit_pct": 0.15}),
    ("个股止损10%+止盈20%", {}, {"stop_loss_pct": 0.10, "take_profit_pct": 0.20}),
]


def universe_from_instruments() -> List[str]:
    path = os.path.join(QLIB_OFFICIAL, "instruments", "all.txt")
    codes = []
    if os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                parts = line.strip().split("\t")
                if not parts or "_" not in parts[0]:
                    continue
                code, _, mkt = parts[0].partition("_")
                if mkt.upper() in ("SH", "SZ", "BJ") and code.isdigit():
                    codes.append(f"{code}.{mkt.upper()}")
    return sorted(codes)


def run_one(codes, cfg_extra, start, end, bt_extra=None) -> Dict[str, Any]:
    cfg = {"topk": 60, "n_drop": 5, "rebalance_days": "weekly",
           "rebalance_weekday": 4}
    cfg.update(cfg_extra)
    executor = BacktestExecutor(
        data_dir=os.path.join(ROOT, "data"),
        enable_parallel=True, max_workers=8,
        enable_performance_profiling=False, use_multiprocessing=False,
    )
    t0 = time.perf_counter()
    try:
        res = asyncio.run(
            executor.run_backtest(
                strategy_name="honest_composite",
                stock_codes=codes,
                start_date=start, end_date=end,
                strategy_config=cfg,
                backtest_config=BacktestConfig(
                    initial_cash=1_000_000.0,
                    commission_rate=0.0003, slippage_rate=0.0001,
                    max_position_size=0.015833,
                    cash_reserve_ratio=0.05, board_lot_size=100,
                    stop_loss_pct=(bt_extra or {}).get("stop_loss_pct", 0.0),
                    take_profit_pct=(bt_extra or {}).get("take_profit_pct", 0.0),
                    rebalance_frequency="daily",
                    open_cost=0.0, close_cost=0.0, min_cost=0.0,
                ),
            )
        )
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        return {"error": f"{type(e).__name__}: {e}",
                "wall_s": round(time.perf_counter() - t0, 1)}

    out: Dict[str, Any] = {"wall_s": round(time.perf_counter() - t0, 1)}
    for k in ("total_return", "annual_return", "sharpe_ratio", "max_drawdown",
              "total_trades", "win_rate", "volatility", "final_value",
              "calmar_ratio"):
        if k in res:
            out[k] = res[k]
    yd = res.get("yearly_returns_detail")
    if isinstance(yd, list) and yd:
        out["yearly"] = {str(int(x["year"])): round(float(x["return"]), 6)
                         for x in yd}
    pb = res.get("perf_breakdown") or {}
    if isinstance(pb, dict):
        out["avg_positions"] = pb.get("avg_positions")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=3)
    ap.add_argument("--only", nargs="*", default=[])
    args = ap.parse_args()

    end = datetime(2026, 5, 8)
    start = datetime(end.year - args.years, end.month, end.day)
    codes = universe_from_instruments()
    print(f"[brake] 窗口 {start.date()} → {end.date()}  股票池 {len(codes)} 只",
          flush=True)

    results: Dict[str, Any] = {}
    for label, cfg_extra, bt_extra in VARIANTS:
        if args.only and not any(o in label for o in args.only):
            continue
        print(f"\n[brake] === {label} ===", flush=True)
        r = run_one(codes, cfg_extra, start, end, bt_extra)
        results[label] = r
        if "error" in r:
            print(f"  ERROR {r['error'][:120]}", flush=True)
            continue
        print(f"  夏普={r.get('sharpe_ratio'):.3f}  收益={r.get('total_return')*100:+.2f}%  "
              f"回撤={r.get('max_drawdown')*100:.2f}%  笔数={r.get('total_trades')}  "
              f"均仓={r.get('avg_positions')}  ({r.get('wall_s')}s)", flush=True)
        y = r.get("yearly") or {}
        if y:
            print("  分年: " + "  ".join(
                f"{k[-2:]}:{v*100:+.1f}%" for k, v in sorted(y.items())), flush=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump({"window": f"{start.date()}→{end.date()}",
                   "results": results}, fh, ensure_ascii=False, indent=1,
                  default=str)
    print(f"\n[brake] 已写入 {OUT}", flush=True)


if __name__ == "__main__":
    main()
