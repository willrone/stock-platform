#!/usr/bin/env python3
"""信号质量追踪 —— 回答「这个信号到底准不准」（2026-10-25）

## 为什么做
平台卖的是信号。用户凭什么信？不能只给一条收益曲线，得给出
**信号本身的可验证质量**：
  - 每次买入信号发出后，T+1 / T+5 / T+10 / T+20 的超额收益
  - 胜率、中位数、最差的一次
  - 对比基准（全市场等权）

这既是给用户的信任凭证，也是我们自己判断策略是否还有效的体检指标。

## 数据来源
  - data/paper/<strategy>_ledger.csv  历史成交流水（date, action, code, ...）
  - data/parquet/baostock_daily/        交易所口径涨跌幅（已除权调整）
  - 后复权价格由 pct_chg 累积合成（见 market_data.load_adjusted_close）

## 输出
  data/paper/<strategy>_signal_quality.json
  {
    "strategy": "...", "updated_at": "...",
    "horizons": {"T+5": {"n": 120, "win_rate": 0.58, "avg_excess": 0.012,
                          "median_excess": 0.008, "worst": -0.09}, ...},
    "by_signal": [ {date, code, name, excess_T5, ...}, ... ],
    "recent_10": [...]
  }

用法:
  env -u PYTHONPATH backend/.venv-py313/bin/python3 backend/scripts/signal_quality.py
  ... --strategies honest_42f rsi_71421
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

BACKEND = "/Users/rone/Projects/stock-platform/backend"
ROOT = "/Users/rone/Projects/stock-platform"
sys.path.insert(0, BACKEND)
sys.path.insert(0, os.path.join(BACKEND, "scripts"))

from scripts.market_data import load_adjusted_close  # noqa: E402

PAPER_DIR = os.path.join(ROOT, "data", "paper")
HORIZONS = (1, 5, 10, 20)


def read_ledger(strategy: str) -> list:
    path = os.path.join(PAPER_DIR, f"{strategy}_ledger.csv")
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return list(csv.DictReader(fh))


def load_names() -> dict:
    raw_dir = os.path.join(ROOT, "data", "tdx_daily", "raw")
    if not os.path.isdir(raw_dir):
        return {}
    dates = sorted(d for d in os.listdir(raw_dir) if d.startswith("date="))
    if not dates:
        return {}
    try:
        df = pd.read_parquet(
            os.path.join(raw_dir, dates[-1], "part.parquet"),
            columns=["ts_code", "name"],
        )
        return dict(zip(df["ts_code"].astype(str), df["name"].astype(str)))
    except Exception:
        return {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategies", nargs="*", default=[])
    ap.add_argument("--master-days", type=int, default=400,
                    help="回看多少个交易日（够覆盖 ledger 里的信号）")
    args = ap.parse_args()

    strategies = args.strategies or [
        f[:-len("_ledger.csv")]
        for f in sorted(os.listdir(PAPER_DIR))
        if f.endswith("_ledger.csv")
    ]
    if not strategies:
        print("没有可分析的 ledger")
        return

    names = load_names()

    # 收集所有出现过的股票代码 + 需要的日期范围
    all_codes, all_dates = set(), []
    ledgers = {}
    for s in strategies:
        rows = read_ledger(s)
        ledgers[s] = rows
        for r in rows:
            if r.get("action") != "BUY":
                continue
            c = r.get("code", "")
            if c:
                all_codes.add(c)
            d = r.get("date", "")
            if d:
                all_dates.append(pd.Timestamp(d))
    if not all_codes:
        print("ledger 里没有买入记录")
        return

    min_d = min(all_dates)
    # 起点往前留出 1 天，最大留出 horizon 天
    start = min_d - pd.Timedelta(days=10)
    end = max(all_dates) + pd.Timedelta(days=45)
    master = pd.bdate_range(start=start, end=end)
    codes = sorted(all_codes)

    print(f"[signal_quality] 分析 {len(strategies)} 条线，"
          f"{len(codes)} 只股票，{len(master)} 个日期", flush=True)

    adj = load_adjusted_close(codes, master)
    # 基准：全市场等权（用后复权涨跌幅的均值近似）
    bench_ret = adj.pct_change(fill_method=None)
    bench_daily = bench_ret.mean(axis=1, skipna=True)

    results = {}
    for s in strategies:
        rows = ledgers[s]
        signals = []
        for r in rows:
            if r.get("action") != "BUY":
                continue
            code = r.get("code", "")
            try:
                d = pd.Timestamp(r.get("date", ""))
            except Exception:
                continue
            if code not in adj.columns or d not in adj.index:
                continue
            p0 = adj.at[d, code]
            if not np.isfinite(p0) or p0 <= 0:
                continue
            item = {
                "date": str(d.date()),
                "code": code,
                "name": names.get(code, ""),
                "price": float(r.get("price") or 0),
            }
            for h in HORIZONS:
                # 找 h 个交易日后的日期
                idx = master.get_indexer([d])[0]
                if idx < 0 or idx + h >= len(master):
                    item[f"excess_T{h}"] = None
                    item[f"ret_T{h}"] = None
                    continue
                d2 = master[idx + h]
                p2 = adj.at[d2, code]
                if not np.isfinite(p2) or p2 <= 0:
                    item[f"excess_T{h}"] = None
                    item[f"ret_T{h}"] = None
                    continue
                ret = p2 / p0 - 1.0
                b = bench_daily.iloc[idx + 1: idx + h + 1]
                b = b[np.isfinite(b)]
                bret = float((1.0 + b).prod() - 1.0) if len(b) else 0.0
                item[f"ret_T{h}"] = round(ret, 6)
                item[f"excess_T{h}"] = round(ret - bret, 6)
            signals.append(item)

        horizons = {}
        for h in HORIZONS:
            vals = [x[f"excess_T{h}"] for x in signals
                    if x.get(f"excess_T{h}") is not None]
            if not vals:
                continue
            arr = np.array(vals, dtype=float)
            horizons[f"T+{h}"] = {
                "n": int(len(arr)),
                "win_rate": round(float((arr > 0).mean()), 4),
                "avg_excess": round(float(arr.mean()), 6),
                "median_excess": round(float(np.median(arr)), 6),
                "worst": round(float(arr.min()), 6),
                "best": round(float(arr.max()), 6),
            }

        out = {
            "strategy": s,
            "updated_at": datetime.now().isoformat(),
            "n_signals": len(signals),
            "horizons": horizons,
            "recent_10": signals[-10:][::-1],
        }
        results[s] = out
        path = os.path.join(PAPER_DIR, f"{s}_signal_quality.json")
        with open(path, "w") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=1)

        print(f"\n[{s}] {len(signals)} 个买入信号", flush=True)
        for k, v in horizons.items():
            print(
                f"  {k}: 胜率 {v['win_rate']*100:.1f}%  "
                f"平均超额 {v['avg_excess']*100:+.2f}%  "
                f"中位 {v['median_excess']*100:+.2f}%  "
                f"最差 {v['worst']*100:+.1f}%",
                flush=True,
            )

    print(f"\n已写入 {PAPER_DIR}/<strategy>_signal_quality.json", flush=True)


if __name__ == "__main__":
    main()
