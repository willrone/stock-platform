#!/usr/bin/env python3
"""趋势加分参数扫描（2026-10-27）

## 目标
用户的三个目标之一是**逐笔交易胜率 ≥ 75%**（当前 67.1%）。
夏普和胜率是**不同的杠杆**：
- 夏普 = 收益 / 波动 → 靠「收益更大」或「波动更小」
- 胜率 = 盈利笔数占比 → 靠「少做亏损交易」

本脚本攻胜率：把「close / MA_n - 1」作为**加分项**加进复合分
（不是过滤 —— 过滤会让持仓跌破均线就被踢出，变成止损，实测有害）。

## 口径
真实盘口径：`--lag 1` + 可成交性检查开 + 5238 只股票池 + 3 年窗口。
扫描时用 `--no-panel`（25 公式因子版）加速：每次 ~80s vs 42 因子版 ~536s。
**筛出候选后必须用 42 因子版复测**（面板因子真生效）才算数。

用法：
  env -u PYTHONPATH -u PYTHONHOME backend/.venv-py313/bin/python3 \
    backend/scripts/scan_trend_bonus.py
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(BACKEND)
sys.path.insert(0, BACKEND)

from app.services.backtest import BacktestExecutor  # noqa: E402
from app.services.backtest.models import BacktestConfig  # noqa: E402

OUT = os.path.join(ROOT, "data", "backtest", "trend_bonus_scan.json")

#: (trend_ma_window, trend_weight)
VARIANTS = [
    (0, 0.0),        # 基线（无趋势加分）
    (20, 0.3),
    (20, 0.6),
    (20, 1.0),
    (60, 0.3),
    (60, 0.6),
    (60, 1.0),
    (120, 0.6),
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


def run_one(codes, cfg_extra, start, end) -> dict:
    ex = BacktestExecutor(
        data_dir=os.path.join(ROOT, "data"),
        enable_parallel=True, max_workers=8,
        enable_performance_profiling=False, use_multiprocessing=False,
    )
    cfg = {"topk": 60, "n_drop": 5, "rebalance_days": "weekly",
           "rebalance_weekday": 4, "lag": 1,
           "panel_dir": "/nonexistent/panel_off"}
    cfg.update(cfg_extra)
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
                max_position_size=0.015833,
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
        "volatility": res.get("volatility"),
        "yearly": yearly,
        "wall_s": round(time.perf_counter() - t0, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=3)
    ap.add_argument("--only", nargs="*", default=None,
                    help="只跑这些 (ma,weight)，格式 '20:0.6'")
    args = ap.parse_args()

    end = datetime(2026, 5, 8)
    start = datetime(end.year - args.years, end.month, end.day)
    codes = universe()
    print(f"[trend] 窗口 {start.date()}→{end.date()}  股票池 {len(codes)}  "
          f"口径=lag1+可成交性+25因子(加速)", flush=True)

    variants = VARIANTS
    if args.only:
        want = {tuple(int(x) for x in o.split(":")) for o in args.only}
        variants = [v for v in VARIANTS if (v[0], v[1]) in want
                    or (v[0] == 0 and (0, 0.0) in want)]

    results = {}
    for ma, w in variants:
        label = f"MA{ma}_w{w}" if ma else "baseline"
        print(f"\n[trend] === {label} ===", flush=True)
        try:
            r = run_one(codes, {"trend_ma_window": ma, "trend_weight": w},
                        start, end)
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
              f"胜率={(wr*100 if wr is not None else 0):.1f}%  ({r['wall_s']}s)",
              flush=True)
        if r.get("yearly"):
            print("  分年: " + "  ".join(
                f"{k[-2:]}:{v*100:+.1f}%" for k, v in sorted(r["yearly"].items())),
                flush=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump({
            "window": f"{start.date()}→{end.date()}",
            "methodology": {
                "lag": 1, "enable_tradability": True, "realistic": True,
                "note": "25 公式因子版（加速扫描用）；候选需用 42 因子版复测",
            },
            "results": results,
        }, fh, ensure_ascii=False, indent=1, default=str)
    print(f"\n[trend] 已写入 {OUT}", flush=True)

    # 排名
    ok = {k: v for k, v in results.items() if "error" not in v}
    if ok:
        best_wr = max(ok.items(), key=lambda kv: kv[1].get("win_rate") or 0)
        best_sr = max(ok.items(), key=lambda kv: kv[1].get("sharpe_ratio") or 0)
        print(f"\n[trend] 胜率最高: {best_wr[0]} "
              f"({(best_wr[1].get('win_rate') or 0)*100:.1f}%, "
              f"夏普 {best_wr[1].get('sharpe_ratio'):.3f})")
        print(f"[trend] 夏普最高: {best_sr[0]} "
              f"({best_sr[1].get('sharpe_ratio'):.3f}, "
              f"胜率 {(best_sr[1].get('win_rate') or 0)*100:.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
