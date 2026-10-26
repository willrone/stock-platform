#!/usr/bin/env python3
"""因子白名单扫描（2026-10-27）—— 「因子减法」的直接实现

## 依据
因子级 RankIC 拆解（`factor_ic_decompose.py`）显示 22 个公式因子的 IC 全部显著，
但 |ICIR| 差 4.5 倍（CORR10 0.69 → STD60 0.15）。

**ICIR 加权已被证伪**（因子高度相关时加权反而更差）→ 正确做法是
**把弱因子直接去掉**，让强因子在等权下占更大比重。

本脚本按 |ICIR| 排序生成「取前 N 个因子」的候选集，逐个跑真实盘口径。

## 口径
`lag=1` + 可成交性开 + 5238 只 + 3 年；25 公式因子版（加速）。

用法：
  env -u PYTHONPATH -u PYTHONHOME backend/.venv-py313/bin/python3 \
    backend/scripts/scan_factor_subset.py
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

OUT = os.path.join(ROOT, "data", "backtest", "factor_subset_scan.json")
ICIR_FILE = os.path.join(ROOT, "data", "backtest", "factor_ic_decompose.json")

#: 要试的子集大小
SIZES = [3, 5, 8, 12, 16, 22]


def load_icir_order() -> list:
    with open(ICIR_FILE) as fh:
        d = json.load(fh)
    rows = d.get("factors") or []
    rows.sort(key=lambda r: -abs(r.get("icir") or 0))
    return [r["factor"] for r in rows]


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


def run_one(codes, allowlist, topk, n_drop, start, end, use_panel=False) -> dict:
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
    if allowlist is not None:
        cfg["factor_allowlist"] = list(allowlist)
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
    ap.add_argument("--topk", type=int, default=60)
    ap.add_argument("--ndrop", type=int, default=5)
    args = ap.parse_args()

    end = datetime.strptime(args.end, "%Y-%m-%d") if args.end else datetime(2026, 5, 8)
    start = (datetime.strptime(args.start, "%Y-%m-%d") if args.start
             else datetime(end.year - args.years, end.month, end.day))
    codes = universe()
    order = load_icir_order()
    print(f"[subset] 窗口 {start.date()}→{end.date()}  股票池 {len(codes)}")
    print(f"[subset] ICIR 排序: {' > '.join(order)}", flush=True)

    results = {}
    for n in SIZES:
        allow = order[:n] if n < len(order) else None
        label = f"top{n}" + ("_all" if allow is None else "")
        print(f"\n[subset] === {label} ===", flush=True)
        try:
            r = run_one(codes, allow, args.topk, args.ndrop, start, end, args.panel)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            r = {"error": f"{type(e).__name__}: {e}"}
        results[label] = r
        results[label]["factors"] = allow if allow is not None else order
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
                "note": f"因子白名单（按|ICIR|排序取前N），topk={args.topk} "
                        f"ndrop={args.ndrop}；候选需分段/42因子复测",
            },
            "icir_order": order,
            "results": results,
        }, fh, ensure_ascii=False, indent=1, default=str)
    print(f"\n[subset] 已写入 {OUT}", flush=True)
    ok = {k: v for k, v in results.items() if "error" not in v}
    if ok:
        bs = max(ok.items(), key=lambda kv: kv[1].get("sharpe_ratio") or 0)
        print(f"[subset] 夏普最高: {bs[0]} ({bs[1].get('sharpe_ratio'):.3f}, "
              f"回撤 {bs[1].get('max_drawdown')*100:.2f}%, "
              f"胜率 {(bs[1].get('win_rate') or 0)*100:.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
