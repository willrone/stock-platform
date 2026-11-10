#!/usr/bin/env python3
"""用 Baostock 抓取官方指数日线，作为**无幸存者偏差**的市场基准。

## 为什么需要第二条通道

`fetch_market_index_benchmarks.py` 走**东方财富** `push2his` K 线接口。
实测 2026-11-01：东财对本机出口**整段封禁**（curl 与浏览器同为 `ERR_EMPTY_RESPONSE`），
且 Tushare `index_daily` 限频 1次/小时、`fund_daily` **直接无接口权限** → 两条路都不通。

**Baostock（证券宝）**：无需 key、无频次限制、可直接取指数日线。

## 覆盖

    sh.000905 中证500
    sz.399006 创业板指
    sh.000001 上证指数

（东财通道已抓到：IDX_000300 沪深300 / IDX_000985 中证全指 / IDX_399317 国证A指）

## 交叉验证

抓完必须跑 `verify_bench_calendars.py`：
六个指数的**交易日历必须逐日完全一致**（各 3355 天，2013-01-04→2026-10-30），
否则说明某个源的数据有问题，不能直接进基准集。

## 输出

    data/benchmarks/IDX_<6位代码>.parquet        （运行态，被 .gitignore）
    backend/fixtures/market_benchmarks/IDX_<6位代码>.csv   （离线快照，纳入 git）

## 用法

    cd backend && env -u PYTHONPATH -u PYTHONHOME .venv-py313/bin/python3 \\
        scripts/fetch_index_benchmarks_baostock.py
"""
from __future__ import annotations

import os
import sys

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
OUT_DIR = os.path.join(ROOT, "data", "benchmarks")
FIXTURES = os.path.join(ROOT, "backend", "fixtures", "market_benchmarks")

TARGETS = [
    ("sh.000905", "IDX_000905", "中证500"),
    ("sz.399006", "IDX_399006", "创业板指"),
    ("sh.000001", "IDX_000001", "上证指数"),
]

START = "2013-01-01"
END = "2026-10-31"
FIELDS = "trade_date,open,close,high,low,volume"


def fetch(code: str) -> pd.DataFrame | None:
    import baostock as bs
    rs = bs.query_history_k_data_plus(
        code, "date,open,close,high,low,volume",
        start_date=START, end_date=END, frequency="d", adjustflag="3")
    rows = []
    while (rs.error_code == "0") and (rs.next()):
        rows.append(rs.get_row_data())
    if not rows:
        print(f"  ✗ {code}: {rs.error_code} {rs.error_msg}")
        return None
    df = pd.DataFrame(rows, columns=["trade_date", "open", "close", "high", "low", "volume"])
    for c in ("open", "close", "high", "low", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["close"]).reset_index(drop=True)
    # 统一成 YYYY-MM-DD（与东财通道产物同格式，避免同一目录两种日期写法）
    df["trade_date"] = pd.to_datetime(df["trade_date"].str.replace("-", "", regex=False)).dt.strftime("%Y-%m-%d")
    return df.sort_values("trade_date").reset_index(drop=True)


def main() -> int:
    try:
        import baostock as bs
    except ImportError:
        print("✗ 未安装 baostock")
        return 2

    lg = bs.login()
    if lg.error_code != "0":
        print(f"✗ baostock 登录失败: {lg.error_msg}")
        return 3

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIXTURES, exist_ok=True)
    ok = 0
    try:
        for code, key, name in TARGETS:
            df = fetch(code)
            if df is None:
                continue
            p = os.path.join(OUT_DIR, f"{key}.parquet")
            df.to_parquet(p, index=False)
            csv = os.path.join(FIXTURES, f"{key}.csv")
            df.to_csv(csv, index=False)
            print(f"  ✓ {code} {name}: {len(df)} 根 "
                  f"{df['trade_date'].iloc[0]}→{df['trade_date'].iloc[-1]} "
                  f"close {df['close'].iloc[0]}→{df['close'].iloc[-1]}")
            print(f"      → {p}")
            print(f"      → {csv}（离线快照）")
            ok += 1
    finally:
        bs.logout()
    print(f"完成：{ok} 个指数")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
