#!/usr/bin/env python3
"""ICIR 加权扫描（2026-10-27）—— 攻夏普的主攻方向

## 动机（基于因子级 IC 拆解的硬数据）
22 个公式因子的 RankIC(fwd5) **全部显著**（|t| 4.13~18.54）：
| 因子 | meanIC | ICIR | t |
|---|---|---|---|
| CORR10 | −0.0673 | **−0.691** | −18.5 |
| CORR5  | −0.0583 | −0.670 | −18.0 |
| MIN30  | +0.0671 | +0.482 | +13.0 |
| MAX60  | −0.0706 | −0.436 | −11.7 |
| STD5   | −0.0595 | −0.364 | −9.8 |

**但 42f 组合的 ICIR 只有 0.43 —— 低于单因子 CORR10 的 0.69。**
说明**等权平均把强因子的信号稀释了**（22 个因子各 1/22 权重，
而 CORR 族只有 3/22；42 因子版更被压到 3/42）。

→ **按 |ICIR| 加权**应该能把组合 ICIR 拉回接近最强因子的水平。

## 扫的参数
- factor_weight_power: 0（等权基线）/ 0.5 / 1.0 / 1.5 / 2.0
  权重 = min(|ICIR|^power, cap) / mean
- factor_weight_cap: 3.0（单因子权重上限，避免过度集中）
- 25 公式因子版（加速）；**候选必须用 42 因子版复测**

用法：
  env -u PYTHONPATH -u PYTHONHOME backend/.venv-py313/bin/python3 \
    backend/scripts/scan_icir_weight.py
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime

BACKEND = "/Users/rone/Projects/stock-platform/backend"
ROOT = "/Users/rone/Projects/stock-platform"
sys.path.insert(0, BACKEND)

from app.services.backtest import BacktestExecutor  # noqa: E402
from app.services.backtest.models import BacktestConfig  # noqa: E402

OUT = os.path.join(ROOT, "data", "backtest", "icir_weight_scan.json")
ICIR_FILE = os.path.join(ROOT, "data", "backtest", "factor_ic_decompose.json")

VARIANTS = [
    (0.0, 3.0, 60, 5),      # 等权基线
    (0.5, 3.0, 60, 5),
    (1.0, 3.0, 60, 5),
    (1.5, 3.0, 60, 5),
    (2.0, 3.0, 60, 5),
    (1.0, 2.0, 60, 5),
    (1.0, 1.5, 60, 5),
    (1.0, 3.0, 40, 3),
    (1.0, 3.0, 30, 3),
]


def universe() -> list:
    path = os.path.join(ROOT, "data", "qlib_official_data", "instruments", "all.txt")
    codes = []
    with open(path) as fh:
        for line in fh:
            parts = line.strip().split("\t")
            if parts and "_" in parts[0]:
                c, _, m = parts[0].partition("_")
                if m.upper() in ("SH", "SZ", "BJ") and c.isdigit():
                    codes.append(f"{c}.{m.upper()}")
    return sorted(codes)


def run_one(codes, power, cap, topk, n_drop, start, end, use_panel=False) -> dict:
    ex = BacktestExecutor(
        data_dir=os.path.join(ROOT, "data"),
        enable_parallel=True, max_workers=8,
        enable_performance_profiling=False, use_multiprocessing=False,
    )
    cfg = {
        "topk": topk, "n_drop": n_drop,
        "rebalance_days": "weekly", "rebalance_weekday": 4, "lag": 1,
        "panel_dir": (os.path.join(ROOT, "data", "qlib_data", "features", "day")
                      if use_panel else "/nonexistent/panel_off"),
    }
    if power > 0:
        cfg["factor_icir_file"] = ICIR_FILE
        cfg["factor_weight_power"] = power
        cfg["factor_weight_cap"] = cap
    t0 = time.perf_counter()
    res = asyncio.run(
        ex.run_backtest(
            strategy_name="honest_composite",
            stock_codes=codes,
            start_date=start, end_date=end,
            strategy_config=cfg,
            backtest_config=BacktestConfig(
                initial_cash=1_000_000.0,
                commission_rate=0.0003, slippage_rate=0.0001,
                max_position_size=0.95 / topk,
                cash_reserve_ratio=0.05, board_lot_size=100,
                stop_loss_pct=0.0, take_profit_pct=0.0,
                rebalance_frequency="daily",
                open_cost=0.0, close_cost=0.0, min_cost=0.0,
                enable_tradability=True,
            ),
        )
    )
    yd = res.get("yearly_returns_detail")
    yearly = ({str(int(x["year"])): round(float(x["return"]), 6) for x in yd}
              if isinstance(yd, list) else {})
    return {
        "sharpe_ratio": res.get("sharpe_ratio"),
        "total_return": res.get("total_return"),
        "max_drawdown": res.get("max_drawdown"),
        "total_trades": res.get("total_trades"),
        "win_rate": res.get("win_rate"),
        "yearly": yearly,
        "wall_s": round(time.perf_counter() - t0, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=3)
    ap.add_argument("--panel", action="store_true")
    ap.add_argument("--start", default="")
    ap.add_argument("--end", default="")
    args = ap.parse_args()

    end = datetime.strptime(args.end, "%Y-%m-%d") if args.end else datetime(2026, 5, 8)
    start = (datetime.strptime(args.start, "%Y-%m-%d") if args.start
             else datetime(end.year - args.years, end.month, end.day))
    codes = universe()
    print(f"[icir] 窗口 {start.date()}→{end.date()}  股票池 {len(codes)}", flush=True)

    results = {}
    for power, cap, topk, n_drop in VARIANTS:
        label = f"p{power}_cap{cap}_topk{topk}_nd{n_drop}"
        print(f"\n[icir] === {label} ===", flush=True)
        try:
            r = run_one(codes, power, cap, topk, n_drop, start, end, args.panel)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            r = {"error": f"{type(e).__name__}: {e}"}
        results[label] = r
        if "error" in r:
            print(f"  ERROR {r['error'][:100]}", flush=True)
            continue
        wr = r.get("win_rate")
        print(f"  夏普={r['sharpe_ratio']:.3f}  收益={r['total_return']*100:+.2f}%  "
              f"回撤={r['max_drawdown']*100:.2f}%  笔数={r['total_trades']}  "
              f"胜率={(wr*100 if wr is not None else 0):.1f}%  ({r['wall_s']}s)", flush=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump({
            "window": f"{start.date()}→{end.date()}",
            "methodology": {
                "lag": 1, "enable_tradability": True, "realistic": True,
                "note": "ICIR 加权扫描（25 公式因子版加速）；候选需 42 因子版复测",
            },
            "results": results,
        }, fh, ensure_ascii=False, indent=1, default=str)
    print(f"\n[icir] 已写入 {OUT}", flush=True)
    ok = {k: v for k, v in results.items() if "error" not in v}
    if ok:
        bs = max(ok.items(), key=lambda kv: kv[1].get("sharpe_ratio") or 0)
        print(f"[icir] 夏普最高: {bs[0]} ({bs[1].get('sharpe_ratio'):.3f}, "
              f"胜率 {(bs[1].get('win_rate') or 0)*100:.1f}%, "
              f"回撤 {bs[1].get('max_drawdown')*100:.2f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
