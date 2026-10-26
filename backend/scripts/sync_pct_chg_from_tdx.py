#!/usr/bin/env python3
"""从 TDX raw 的 `prev_close` 同步精确涨跌幅到 baostock_daily 目录 —— 2026-10-26

## 为什么做
1. **北交所缺口**：baostock 不支持 `bj.` 前缀，354 只 `920xxx.BJ` 拉不到数据 →
   它们在纸面盘里只能走「涨跌幅不可靠」回退路径 → 涨跌停判断失真。
   而 **TDX 的 `prev_close` 是交易所口径的前收盘**（已做除权除息调整），
   实测 600000.SH 的 close/prev_close-1 = -0.94%，与 baostock 的 -0.9356% 一致。
2. **每日增量**：TDX raw 每天新增一个分区，可以把当日精确涨跌幅补进去
   （比每次用 baostock 重拉 5240 只快两个数量级）。

## 数据流
  TDX raw 分区 (date, ts_code, open, high, low, close, prev_close, volume, amount)
      → pct_chg = close / prev_close - 1
      → 合并进 data/parquet/baostock_daily/<code>_<SH|SZ|BJ>.parquet
         （已有文件：更新对应日期的行；baostock 没有的：新建）

## 边界
- TDX raw 目前只回溯到 2026-05-11（更早的分区已被清理），
  所以 2026-05-11 之前的历史仍依赖 baostock（北交所那段缺失）。
- 本地已有 baostock 行时**只更新 pct_chg**，不覆盖其它字段
  （成交额/换手率/ST 标记只有 baostock 有）。

用法：
  env -u PYTHONPATH backend/.venv-py313/bin/python3 \
    backend/scripts/sync_pct_chg_from_tdx.py            # 全量同步所有分区
  ... --since 2026-10-01                               # 只同步指定日期之后
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = "/Users/rone/Projects/stock-platform"
RAW_DIR = os.path.join(ROOT, "data", "tdx_daily", "raw")
OUT_DIR = os.path.join(ROOT, "data", "parquet", "baostock_daily")
COLS = [
    "date", "code", "open", "high", "low", "close", "volume", "amount",
    "pct_chg", "turn", "is_st", "trade_status", "pe_ttm", "pb_mrq", "ps_ttm",
]


def raw_partitions(since: str = "") -> List[Tuple[str, str]]:
    out = []
    for name in sorted(os.listdir(RAW_DIR)):
        if not name.startswith("date="):
            continue
        day = name.split("=", 1)[1]
        if since and day < since:
            continue
        out.append((day, os.path.join(RAW_DIR, name, "part.parquet")))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="", help="只同步该日期之后的分区")
    args = ap.parse_args()

    parts = raw_partitions(args.since)
    if not parts:
        print("没有可用的 TDX raw 分区")
        return
    print(f"[sync] TDX 分区 {len(parts)} 个："
          f"{parts[0][0]} → {parts[-1][0]}", flush=True)

    # 白名单：stock_data 目录里的代码（= 我们真正的股票池）
    stock_dir = os.path.join(ROOT, "data", "parquet", "stock_data")
    whitelist = set()
    for f in os.listdir(stock_dir):
        if not f.endswith(".parquet"):
            continue
        stem = f[: -len(".parquet")]
        if "_" not in stem:
            continue
        code, _, market = stem.partition("_")
        if market in ("SH", "SZ", "BJ") and code.isdigit():
            whitelist.add(f"{code}.{market}")
    print(f"[sync] 白名单（stock_data）{len(whitelist)} 只", flush=True)
    if not whitelist:
        print("[sync] 白名单为空，终止")
        return

    # 1) 汇总所有分区的 (code, date) → pct_chg / OHLCV
    rows: Dict[str, Dict[str, dict]] = {}
    t0 = time.time()
    for day, path in parts:
        try:
            df = pd.read_parquet(
                path,
                columns=["ts_code", "date", "open", "high", "low", "close",
                         "prev_close", "volume", "amount", "is_suspended"],
            )
        except Exception as e:  # noqa: BLE001
            print(f"  跳过 {day}: {e}")
            continue
        df["ts_code"] = df["ts_code"].astype(str)
        df = df[df["ts_code"].str.contains(r"\.(SH|SZ|BJ)$", regex=True)]
        # 白名单：只要我们自己的股票池。
        # TDX raw 一天有 5 万+ 条记录（指数、基金、可转债…），
        # 不加白名单会误建 5 万个文件。
        df = df[df["ts_code"].isin(whitelist)]
        for r in df.itertuples(index=False):
            code = r.ts_code
            stem = code.replace(".", "_")
            prev = float(r.prev_close) if r.prev_close == r.prev_close else 0.0
            close = float(r.close) if r.close == r.close else float("nan")
            if prev <= 0 or not np.isfinite(close) or close <= 0:
                continue
            pct = close / prev - 1.0
            rec = {
                "date": str(r.date)[:10],
                "open": float(r.open) if r.open == r.open else np.nan,
                "high": float(r.high) if r.high == r.high else np.nan,
                "low": float(r.low) if r.low == r.low else np.nan,
                "close": close,
                "volume": float(r.volume) if r.volume == r.volume else np.nan,
                "amount": float(r.amount) if r.amount == r.amount else np.nan,
                "pct_chg": pct * 100.0,   # 与 baostock 一致用百分数
                "suspended": int(r.is_suspended) if r.is_suspended == r.is_suspended else 0,
            }
            rows.setdefault(stem, {})[rec["date"]] = rec
        print(f"  {day}: 累计 {len(rows)} 只", flush=True)

    print(f"[sync] 汇总 {len(rows)} 只，{time.time()-t0:.0f}s", flush=True)

    # 2) 合并进 baostock_daily
    os.makedirs(OUT_DIR, exist_ok=True)
    updated = created = 0
    for stem, recs in rows.items():
        path = os.path.join(OUT_DIR, f"{stem}.parquet")
        new_rows = []
        for d, rec in sorted(recs.items()):
            new_rows.append({
                "date": pd.Timestamp(d),
                "code": stem.replace("_", "."),
                "open": rec["open"], "high": rec["high"], "low": rec["low"],
                "close": rec["close"], "volume": rec["volume"],
                "amount": rec["amount"], "pct_chg": rec["pct_chg"],
                "turn": np.nan, "is_st": 0, "trade_status": 0 if rec["suspended"] else 1,
                "pe_ttm": np.nan, "pb_mrq": np.nan, "ps_ttm": np.nan,
            })
        new_df = pd.DataFrame(new_rows)

        if os.path.exists(path):
            try:
                old = pd.read_parquet(path)
            except Exception:
                old = pd.DataFrame(columns=COLS)
            old["date"] = pd.to_datetime(old["date"])
            # 以 TDX 的 pct_chg 为准更新同日期行；TDX 没有的日期保留 baostock 的
            merged = old.merge(
                new_df[["date", "pct_chg", "trade_status"]],
                on="date", how="left", suffixes=("", "_tdx"),
            )
            mask = merged["pct_chg_tdx"].notna()
            merged.loc[mask, "pct_chg"] = merged.loc[mask, "pct_chg_tdx"]
            merged.loc[mask, "trade_status"] = merged.loc[mask, "trade_status_tdx"]
            out = merged[[c for c in COLS if c in merged.columns]].drop(
                columns=[c for c in merged.columns if c.endswith("_tdx")], errors="ignore"
            )
            updated += 1
        else:
            out = new_df[COLS]
            created += 1
        out = out.sort_values("date").reset_index(drop=True)
        tmp = path + ".tmp"
        out.to_parquet(tmp, index=False)
        os.replace(tmp, path)

    print(f"[sync] 更新 {updated} 个文件，新建 {created} 个文件", flush=True)
    print(f"[sync] 输出目录 {OUT_DIR}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
