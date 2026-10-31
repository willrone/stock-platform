#!/usr/bin/env python3
"""诊断：用**产品代码**（rebalance_sleeve）跑袖套，区分两种「类权重偏离」。

parity 回测报的是「类权重最大偏离 均值 2.85pt / 最差 7.10pt」，
但那个数是**按周**采样的，而袖套**每 21 个交易日才调一次仓** →
绝大多数采样点根本不是调仓日，偏离只是**两次调仓之间的价格漂移**。

必须分开看：

  - **调仓日偏离**：调完仓那一刻三类权重离 1/3 有多远 → 衡量「撮合器有没有把
    类间等权做出来」
  - **区间最大偏离**：两次调仓之间漂到多远 → 衡量「月度再平衡的漂移」，
    这是**策略设计的固有成本**，不是实现 bug

两者的处置完全不同：前者是 bug，要修；后者是取舍，要么改承诺，
要么把调仓频率提上去（代价：换手上升 + 整手误差变大）。

本脚本只用产品撮合器跑袖套（无 A 腿）→ 秒级。
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app.services.trading.multi_asset_service import (  # noqa: E402
    ETF_PARQUET_FROZEN,
    SLEEVE_CLASSES,
    SLEEVE_HELD,
    load_etf_bars,
    rebalance_sleeve,
    sleeve_class_weights_now,
    sleeve_target_weights,
)

ALL_CODES: List[str] = sorted({c for lst in SLEEVE_HELD.values() for c in lst})
CAPITAL = 500_000.0
REB_EVERY = 21


def main() -> int:
    bars = load_etf_bars(ALL_CODES, base_dir=ETF_PARQUET_FROZEN)
    closes = {}
    for c in ALL_CODES:
        df = bars.frames[c]
        closes[c] = df.set_index("date")["close"].astype(float)
    px_df = pd.DataFrame(closes).sort_index()
    valid = px_df.dropna()
    lo, hi = valid.index[0], valid.index[-1]
    raw = px_df.loc[lo:hi]
    # ⚠️ 只保留「所有 9 只都有原始 bar」的交易日：rebalance_sleeve 的
    #    max_stale_days=3 会拒绝「价格落后超过 3 个自然日」的标的。
    #    节假日错位会让某些日期只有部分 ETF 有 bar → 调仓直接抛错。
    ok_days = raw.index[raw.notna().all(axis=1)]
    if len(ok_days) == 0:
        raise SystemExit("❌ 没有 9 只 ETF 全部有 bar 的交易日")
    px_df = raw.loc[ok_days]
    print(f"[diag] 共同区间 {px_df.index[0].date()} → {px_df.index[-1].date()}"
          f"  {len(px_df)} 天（剔除单边缺价日 {len(raw) - len(px_df)} 天）")

    dates = list(px_df.index)
    target = sleeve_target_weights()

    positions: Dict[str, Dict[str, Any]] = {}
    cash = CAPITAL
    anchor = 0
    rows: List[Dict[str, Any]] = []
    last_reb: Dict[str, Any] = {}

    for i, d in enumerate(dates):
        reb = (i >= anchor) and ((i - anchor) % REB_EVERY == 0)
        if reb:
            positions, cash, buys, sells = rebalance_sleeve(
                positions, cash, bars, d, target,
            )
            cw = sleeve_class_weights_now(positions, bars, d)
            dev = max(abs(cw[c] - 1 / 3) for c in SLEEVE_CLASSES)
            pos_val = sum(
                float(pp.get("shares", 0)) * float(px_df.at[d, c])
                for c, pp in positions.items() if c in px_df.columns
            )
            invested = pos_val / (cash + pos_val) if (cash + pos_val) > 0 else 0.0
            last_reb = {
                "date": str(d.date()), "dev": dev,
                "cw": cw, "invested": invested,
                "n_buys": len(buys), "n_sells": len(sells),
            }
        else:
            cw = sleeve_class_weights_now(positions, bars, d)
            dev = max(abs(cw[c] - 1 / 3) for c in SLEEVE_CLASSES)
        rows.append({
            "date": str(d.date()), "reb": reb, "dev": dev,
            "cw": cw, "cash": cash,
        })

    reb_rows = [r for r in rows if r["reb"]]
    # 区间内（两次调仓之间）的最大偏离
    seg_max: List[float] = []
    cur = 0.0
    for r in rows:
        cur = max(cur, r["dev"])
        if r["reb"]:
            seg_max.append(cur)
            cur = 0.0

    def stat(xs):
        return {
            "mean": round(float(np.mean(xs)), 4) if xs else None,
            "max": round(float(np.max(xs)), 4) if xs else None,
            "p90": round(float(np.percentile(xs, 90)), 4) if xs else None,
        }

    print()
    print(f"[diag] 调仓次数 {len(reb_rows)}（每 {REB_EVERY} 交易日）")
    print(f"[diag] ① 调仓日类权重偏离  {stat([r['dev'] for r in reb_rows])}  (pt)")
    print(f"[diag] ② 区间内最大偏离    {stat(seg_max)}  (pt)")
    print(f"[diag] ③ 全部采样点偏离    {stat([r['dev'] for r in rows])}  (pt)")

    # 逐年
    print("\n[diag] 逐年：调仓日偏离 / 区间最大偏离")
    by_year: Dict[str, List[Any]] = {}
    for r in reb_rows:
        by_year.setdefault(r["date"][:4], []).append(r["dev"])
    for y in sorted(by_year):
        ds = by_year[y]
        print(f"        {y}: 调仓日 均 {np.mean(ds)*100:5.2f}pt 最大 {np.max(ds)*100:5.2f}pt"
              f"  (n={len(ds)})")

    out = os.path.join(ROOT, "scripts", "diag_sleeve_drift.json")
    with open(out, "w") as fh:
        json.dump({
            "window": [str(dates[0].date()), str(dates[-1].date())],
            "reb_every": REB_EVERY,
            "rebalance_day_dev": stat([r["dev"] for r in reb_rows]),
            "interval_max_dev": stat(seg_max),
            "all_samples_dev": stat([r["dev"] for r in rows]),
            "by_year": {y: stat(v) for y, v in by_year.items()},
            "rows": rows,
        }, fh, ensure_ascii=False, indent=1)
    print(f"\n[diag] 产物 → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
