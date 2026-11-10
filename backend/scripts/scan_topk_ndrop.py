#!/usr/bin/env python3
"""持仓集中度 / 换手参数扫描（2026-10-27）—— 攻「逐笔胜率 75%」

## 为什么扫这个
夏普和胜率是**不同的杠杆**：
- 夏普 = 收益/波动 → 靠「收益更大」或「波动更小」
- 逐笔胜率 = 盈利笔数占比 → 靠「少做亏损交易」

周频 TopK60 的胜率 65.7%（25 因子版）。要提到 75%，最直接的杠杆是
**减少交易次数 + 集中持仓**：
- topk 变小 → 每笔仓位更大、只做最有信心的票 → 胜率通常上升
- n_drop 变小 → 换手更低 → 避免「刚买就跌」的摩擦型亏损

子代理在做因子子集/合成/模型线/开源方法，没覆盖这个维度。

## 口径
真实盘口径：`lag=1` + 可成交性检查开 + 5238 只股票池 + 3 年窗口。
用 25 公式因子版加速（`panel_dir` 指向不存在的目录 → 面板因子全 NaN）。
**候选必须再用 42 因子版复测**才算数。

用法：
  env -u PYTHONPATH -u PYTHONHOME backend/.venv-py313/bin/python3 \
    backend/scripts/scan_topk_ndrop.py
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

OUT = os.path.join(ROOT, "data", "backtest", "topk_ndrop_scan.json")

#: (topk, n_drop) —— 单仓上限按等权口径：0.95 / topk
VARIANTS = [
    (60, 5),   # 基线
    (60, 2),
    (60, 3),
    (40, 3),
    (40, 5),
    (30, 3),
    (30, 2),
    (20, 2),
    (20, 3),
    (80, 5),
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


def run_one(codes, topk, n_drop, start, end, use_panel=False) -> dict:
    ex = BacktestExecutor(
        data_dir=os.path.join(ROOT, "data"),
        enable_parallel=True, max_workers=8,
        enable_performance_profiling=False, use_multiprocessing=False,
    )
    mps = 0.95 / topk
    t0 = time.perf_counter()
    res = asyncio.run(
        ex.run_backtest(
            strategy_name="honest_composite",
            stock_codes=codes,
            start_date=start, end_date=end,
            strategy_config={
                "topk": topk, "n_drop": n_drop,
                "rebalance_days": "weekly", "rebalance_weekday": 4,
                "lag": 1,
                "panel_dir": (os.path.join(ROOT, "data", "qlib_data", "features", "day")
                              if use_panel else "/nonexistent/panel_off"),
            },
            backtest_config=BacktestConfig(
                initial_cash=1_000_000.0,
                commission_rate=0.0003, slippage_rate=0.0001,
                max_position_size=mps,
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
    ap.add_argument("--panel", action="store_true",
                    help="用 42 因子版（面板因子真生效）复测")
    ap.add_argument("--start", default="", help="窗口起点 YYYY-MM-DD（OOS 用）")
    ap.add_argument("--end", default="", help="窗口终点 YYYY-MM-DD")
    ap.add_argument("--only", nargs="*", default=None,
                    help="只跑这些 topk:ndrop")
    args = ap.parse_args()

    end = datetime.strptime(args.end, "%Y-%m-%d") if args.end else datetime(2026, 5, 8)
    start = (datetime.strptime(args.start, "%Y-%m-%d") if args.start
             else datetime(end.year - args.years, end.month, end.day))
    codes = universe()
    print(f"[topk] 窗口 {start.date()}→{end.date()}  股票池 {len(codes)}", flush=True)

    results = {}
    variants = VARIANTS
    if args.only:
        want = {tuple(int(x) for x in o.split(":")) for o in args.only}
        variants = [v for v in VARIANTS if v in want]
    for topk, n_drop in variants:
        label = f"topk{topk}_ndrop{n_drop}"
        print(f"\n[topk] === {label} ===", flush=True)
        try:
            r = run_one(codes, topk, n_drop, start, end, use_panel=args.panel)
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

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump({
            "window": f"{start.date()}→{end.date()}",
            "methodology": {
                "lag": 1, "enable_tradability": True, "realistic": True,
                "note": "25 公式因子版（加速扫描）；候选需 42 因子版复测",
            },
            "results": results,
        }, fh, ensure_ascii=False, indent=1, default=str)
    print(f"\n[topk] 已写入 {OUT}", flush=True)

    ok = {k: v for k, v in results.items() if "error" not in v}
    if ok:
        bw = max(ok.items(), key=lambda kv: kv[1].get("win_rate") or 0)
        bs = max(ok.items(), key=lambda kv: kv[1].get("sharpe_ratio") or 0)
        print(f"[topk] 胜率最高: {bw[0]} ({(bw[1].get('win_rate') or 0)*100:.1f}%, "
              f"夏普 {bw[1].get('sharpe_ratio'):.3f})")
        print(f"[topk] 夏普最高: {bs[0]} ({bs[1].get('sharpe_ratio'):.3f}, "
              f"胜率 {(bs[1].get('win_rate') or 0)*100:.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
