"""把通达信盘后包增量数据合并进 data/parquet/stock_data。

目标格式（与现有 5513 个 *_SH/_SZ/_BJ.parquet 完全一致）:
    列: ts_code, date, open, high, low, close, volume
    - ts_code: 600000.SH 风格
    - date:    datetime64[ns]
    - OHLC:    不复权价（float64）
    - volume:  **手**（float64）— 现有 stock_data 是手
              实测 600000 2026-05-08: stock_data=892690.33,
              通达信原值=89269033 股 → 需 /100

已验证: 现有 stock_data 600000 2026-05-08 OHLC = 9.08/9.13/9.05/9.07,
与通达信不复权完全一致 → 存不复权价口径正确。

用法:
    cd ~/Projects/stock-platform
    env -u PYTHONPATH backend/.venv/bin/python3 \
        back_test_data_service/tdx_pipeline/merge_to_stock_data.py --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from typing import Dict, List

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from back_test_data_service.tdx_pipeline.pipeline import RAW_DIR
from back_test_data_service.tdx_pipeline import loaded_dates
from back_test_data_service.tdx_pipeline.ashare import filter_ashare

STOCK_DATA_DIR = "/Users/rone/Projects/stock-platform/data/parquet/stock_data"

# ts_code 后缀 → 文件名后缀
SUFFIX = {"SH": "SH", "SZ": "SZ", "BJ": "BJ"}


def stock_data_path(ts_code: str) -> str:
    """600000.SH → stock_data/600000_SH.parquet"""
    code, suffix = ts_code.split(".")
    return os.path.join(STOCK_DATA_DIR, f"{code}_{suffix}.parquet")


def load_new_rows(start: str, end: str) -> pd.DataFrame:
    """加载 [start, end] 的 A 股日线，转成 stock_data 口径。

    转换:
    - volume 股 → 手 (/100)
    - 丢掉辅助列（market/code/name/prev_close/amount/is_suspended）
    """
    dates = [d for d in loaded_dates() if start <= d <= end]
    if not dates:
        return pd.DataFrame(columns=["ts_code", "date", "open", "high", "low", "close", "volume"])

    frames = []
    for d in dates:
        part = os.path.join(RAW_DIR, f"date={d}", "part.parquet")
        if not os.path.exists(part):
            continue
        df = pd.read_parquet(part)
        a = filter_ashare(df)
        if len(a) == 0:
            continue
        frames.append(a[["ts_code", "date", "open", "high", "low", "close", "volume"]])

    if not frames:
        return pd.DataFrame(columns=["ts_code", "date", "open", "high", "low", "close", "volume"])

    out = pd.concat(frames, ignore_index=True)
    # volume 股 → 手
    out["volume"] = (out["volume"] / 100.0).round(2)
    out["date"] = pd.to_datetime(out["date"])
    # 停牌日 close 可能是 0 或停牌前价，但 volume=0 — 保留（回测引擎自己处理）
    return out


def merge(dry_run: bool = True, start: str = "2026-05-09", end: str = "2026-10-16",
          backup: bool = True) -> Dict[str, int]:
    """把增量合并进 stock_data 各 parquet。

    Args:
        dry_run: 只统计不写盘
        start:   起始日
        end:     结束日
        backup:  写盘前备份原文件到 .bak

    Returns:
        {"updated": N, "created": N, "skipped": N, "rows_added": N}
    """
    new = load_new_rows(start, end)
    if len(new) == 0:
        print("无新数据")
        return {"updated": 0, "created": 0, "skipped": 0, "rows_added": 0}

    by_code: Dict[str, pd.DataFrame] = dict(tuple(new.groupby("ts_code")))
    stats = {"updated": 0, "created": 0, "skipped": 0, "rows_added": 0}

    for ts_code, new_df in sorted(by_code.items()):
        path = stock_data_path(ts_code)
        exists = os.path.exists(path)

        if exists:
            old = pd.read_parquet(path)
            # 去重：只保留 new 中不在 old 的日期
            old_dates = set(pd.to_datetime(old["date"]).dt.date)
            fresh = new_df[~new_df["date"].dt.date.isin(old_dates)]
            if len(fresh) == 0:
                stats["skipped"] += 1
                continue
            merged = pd.concat([old, fresh], ignore_index=True)
            merged = merged.sort_values("date").reset_index(drop=True)
            merged["date"] = pd.to_datetime(merged["date"])
            action = "updated"
            n_added = len(fresh)
            stats["updated"] += 1
        else:
            merged = new_df.sort_values("date").reset_index(drop=True)
            action = "created"
            n_added = len(new_df)
            stats["created"] += 1

        stats["rows_added"] += n_added

        if not dry_run:
            if backup and exists:
                os.replace(path, path + ".bak")
            merged.to_parquet(path, index=False, compression="snappy")

        done = stats["updated"] + stats["created"]
        if done <= 5 or done % 1000 == 0:
            print(f"  [{action}] {ts_code}: +{n_added} 行 → 总 {len(merged)} 行")

    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="合并通达信增量到 stock_data")
    ap.add_argument("--dry-run", action="store_true", help="只统计不写盘")
    ap.add_argument("--start", default="2026-05-09")
    ap.add_argument("--end", default="2026-10-16")
    ap.add_argument("--no-backup", action="store_true")
    args = ap.parse_args()

    print(f"模式: {'DRY-RUN（不写盘）' if args.dry_run else '实际写盘'}")
    print(f"区间: {args.start} → {args.end}")
    print(f"stock_data 目录: {STOCK_DATA_DIR}")
    print("=" * 60)

    stats = merge(dry_run=args.dry_run, start=args.start, end=args.end,
                  backup=not args.no_backup)

    print("=" * 60)
    print(f"结果: 更新 {stats['updated']} 只, 新建 {stats['created']} 只, "
          f"跳过 {stats['skipped']} 只, 新增行 {stats['rows_added']}")


if __name__ == "__main__":
    main()
