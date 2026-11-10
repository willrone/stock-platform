#!/usr/bin/env python3
"""汇总 features_full 的覆盖统计，用于回测可行性判断。

    env -u PYTHONPATH -u PYTHONHOME backend/.venv-py313/bin/python3 \
        backend/scripts/full_history_features_stats.py
"""
from __future__ import annotations

import os
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
FULL = ROOT / "data" / "qlib_data" / "features_full"
DAY = ROOT / "data" / "qlib_data" / "features" / "day"
PROGRESS = ROOT / "data" / "backtest" / "full_history_features_progress.json"
OUT = ROOT / "data" / "backtest" / "full_history_features_stats.json"

CUTOFF = pd.Timestamp("2021-05-11")  # features/day 增量管道的起点


def main() -> None:
    progress: dict = {}
    if PROGRESS.exists():
        progress = json.loads(PROGRESS.read_text(encoding="utf-8"))

    ok = {k: v for k, v in progress.items() if v.get("status") == "ok"}
    starts = pd.Series([pd.Timestamp(v["start"]) for v in ok.values()])
    ends = pd.Series([pd.Timestamp(v["end"]) for v in ok.values()])
    rows = pd.Series([v["rows"] for v in ok.values()])

    stats: dict = {
        "n_stocks_total_targets": 5513,
        "n_stocks_ok": len(ok),
        "n_stocks_failed": sum(1 for v in progress.values()
                               if v.get("status", "").startswith("error")),
        "n_stocks_no_data": sum(1 for v in progress.values()
                                if v.get("status") == "no_data"),
        "start_date": {
            "min": str(starts.min().date()) if len(starts) else None,
            "median": str(starts.median().date()) if len(starts) else None,
            "mean": str(starts.mean().date()) if len(starts) else None,
            "max": str(starts.max().date()) if len(starts) else None,
        },
        "end_date": {
            "min": str(ends.min().date()) if len(ends) else None,
            "median": str(ends.median().date()) if len(ends) else None,
            "max": str(ends.max().date()) if len(ends) else None,
        },
        "n_stocks_starting_before_2021_05_11": int((starts < CUTOFF).sum()),
        "n_stocks_starting_before_2000": int(
            (starts < pd.Timestamp("2000-01-01")).sum()),
        "n_stocks_starting_before_2010": int(
            (starts < pd.Timestamp("2010-01-01")).sum()),
        "pct_start_before_2021_05_11": round(
            float((starts < CUTOFF).mean()) * 100, 2) if len(starts) else None,
        "rows_per_stock": {
            "min": int(rows.min()) if len(rows) else None,
            "median": int(rows.median()) if len(rows) else None,
            "mean": round(float(rows.mean()), 1) if len(rows) else None,
            "max": int(rows.max()) if len(rows) else None,
            "total": int(rows.sum()) if len(rows) else None,
        },
        "n_files_on_disk": len(list(FULL.glob("*.parquet"))),
        "n_files_in_features_day": len(list(DAY.glob("*.parquet"))),
    }

    # 与 features/day 的起始日期对比
    day_starts = []
    for p in sorted(DAY.glob("*.parquet")):
        try:
            idx = pd.read_parquet(p, columns=[]).index
            dt = pd.DatetimeIndex(idx.get_level_values(1))
            day_starts.append(dt.min())
        except Exception:
            continue
    if day_starts:
        ds = pd.Series(day_starts)
        stats["features_day_start_date"] = {
            "min": str(ds.min().date()),
            "median": str(ds.median().date()),
            "n_start_before_2021_05_11": int((ds < CUTOFF).sum()),
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(stats, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.exit(main())
