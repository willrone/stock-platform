#!/usr/bin/env python3
"""校验基准指数的交易日历一致性（跨数据源交叉验证门）。

## 为什么需要这道门

基准指数来自**不同数据源**（东方财富 / Baostock），
如果某个源的数据有缺口/错位，`build_single_asset_benchmark` 的 ffill 会把缺口填成"零收益"，
**伪造出一条平滑但虚假的净值曲线**（历史上 510300 在 2016 窗口被填出"夏普 0.87"的假象就是这么来的）。

所以：**所有基准指数的交易日历必须逐日完全一致**，不一致就不许进基准集。

## 判定

- 参考轴 = `IDX_000985`（中证全指）
- 每个指数：行数、首末日期、日期集合差异（缺/多）必须为 0

## 用法

    cd ~/Projects/stock-platform && env -u PYTHONPATH -u PYTHONHOME \\
        backend/.venv-py313/bin/python3 backend/scripts/verify_bench_calendars.py
"""
from __future__ import annotations

import os
import sys

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
OUT = os.path.join(ROOT, "data", "benchmarks")
FIX = os.path.join(ROOT, "backend", "fixtures", "market_benchmarks")
REFERENCE = "IDX_000985"


def main() -> int:
    if not os.path.isdir(OUT):
        print(f"✗ 无基准目录 {OUT}")
        return 2
    ref_path = os.path.join(OUT, f"{REFERENCE}.parquet")
    if not os.path.exists(ref_path):
        print(f"✗ 无参考基准 {ref_path}")
        return 2
    ref = pd.read_parquet(ref_path)
    ref_dates = set(pd.to_datetime(ref["trade_date"].astype(str)).dt.strftime("%Y-%m-%d"))
    print(f"参考轴 {REFERENCE}: {len(ref_dates)} 天 {min(ref_dates)} → {max(ref_dates)}")

    bad = 0
    for fn in sorted(os.listdir(OUT)):
        if not fn.endswith(".parquet"):
            continue
        p = os.path.join(OUT, fn)
        df = pd.read_parquet(p)
        df["trade_date"] = pd.to_datetime(df["trade_date"].astype(str)).dt.strftime("%Y-%m-%d")
        df = df.sort_values("trade_date").reset_index(drop=True)
        df.to_parquet(p, index=False)
        dates = set(df["trade_date"])
        miss = len(ref_dates - dates)
        extra = len(dates - ref_dates)
        ok = miss == 0 and extra == 0
        if not ok:
            bad += 1
        print(f"  {fn:24s} {len(df):5d} 根 {df['trade_date'].iloc[0]}→{df['trade_date'].iloc[-1]} "
              f"{'日期轴一致 ✓' if ok else f'不一致 ✗ (缺{miss}/多{extra})'}")
        csv = os.path.join(FIX, fn[:-len(".parquet")] + ".csv")
        os.makedirs(FIX, exist_ok=True)
        df.to_csv(csv, index=False)
    print(f"\n结果：{len(os.listdir(OUT))} 个基准，{'全部一致 ✓' if bad == 0 else f'{bad} 个不一致 ✗'}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
