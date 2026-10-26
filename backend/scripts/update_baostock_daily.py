#!/usr/bin/env python3
"""baostock 每日增量更新 —— 2026-10-26

## 为什么需要
全量拉取是一次性的（2026-10-25 拉了 5594 只，其中 5240 成功）。
之后**每天都有新股上市**，新股票在 `data/parquet/baostock_daily/` 里没有文件 →
纸面盘走到它时会走「涨跌幅不可靠」回退路径 → 涨跌停判断失真。

## 做法
1. `baostock.query_all_stock(day=今天)` 拿当日全部股票代码
2. 与本地已有目录对比 → 只对**新增**的股票拉数据
3. 同时重试上次失败的（北交所 baostock 不支持，失败就跳过）
4. 顺带把**当天**的日线更新到已有股票（增量行）

用法：
  env -u PYTHONPATH backend/.venv-py313/bin/python3 backend/scripts/update_baostock_daily.py
  ... --date 2026-10-26     # 指定日期（默认今天）
  ... --full-refresh        # 重拉全部（数据源修好后用）
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List

BACKEND = "/Users/rone/Projects/stock-platform/backend"
ROOT = "/Users/rone/Projects/stock-platform"
sys.path.insert(0, BACKEND)

OUT_DIR = os.path.join(ROOT, "data", "parquet", "baostock_daily")
START = "20210101"

FIELDS = (
    "date,code,open,high,low,close,volume,amount,"
    "pctChg,turn,isST,tradestatus,peTTM,pbMRQ,psTTM"
)
COLS = [
    "date", "code", "open", "high", "low", "close", "volume", "amount",
    "pct_chg", "turn", "is_st", "trade_status", "pe_ttm", "pb_mrq", "ps_ttm",
]


def local_files() -> set:
    if not os.path.isdir(OUT_DIR):
        return set()
    return {f[:-8] for f in os.listdir(OUT_DIR) if f.endswith(".parquet")}


def bs_code(code: str, market: str) -> str:
    """baostock 的代码格式：sh.600000 / sz.000001 / bj.920001"""
    return f"{ {'SH': 'sh.', 'SZ': 'sz.', 'BJ': 'bj.'}[market] }{code}"


def fetch_one(bs_symbol: str, stem: str) -> str:
    import baostock as bs
    import pandas as pd

    out_path = os.path.join(OUT_DIR, f"{stem}.parquet")
    rs = bs.query_history_k_data_plus(
        bs_symbol, FIELDS, start_date=START, end_date="20261231",
        frequency="d", adjustflag="3",
    )
    if rs.error_code != "0":
        return f"fail:{rs.error_code}"
    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    if not rows:
        return "empty"
    df = pd.DataFrame(rows, columns=rs.fields).rename(
        columns={
            "pctChg": "pct_chg", "isST": "is_st", "tradestatus": "trade_status",
            "peTTM": "pe_ttm", "pbMRQ": "pb_mrq", "psTTM": "ps_ttm",
        }
    )[COLS].copy()
    for c in ("open", "high", "low", "close", "volume", "amount", "pct_chg",
              "turn", "pe_ttm", "pb_mrq", "ps_ttm"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("is_st", "trade_status"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    tmp = out_path + ".tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, out_path)
    return "ok"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="", help="YYYY-MM-DD，默认今天")
    ap.add_argument("--full-refresh", action="store_true",
                    help="重拉全部（忽略已有文件）")
    ap.add_argument("--new-only", action="store_true",
                    help="只拉新股票，不更新已有股票的当日行")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    import baostock as bs

    lg = bs.login()
    if lg.error_code != "0":
        print(f"❌ baostock 登录失败: {lg.error_msg}")
        return

    day = args.date or time.strftime("%Y-%m-%d")

    # 股票池来源：**我们自己的 stock_data 目录**（每天由 TDX 增量更新），
    # 而不是 baostock 的 query_all_stock —— 后者的股票列表接口滞后
    # （实测到 2026-10-16，而日线数据已到 10-23）。
    have = local_files()
    stock_dir = os.path.join(ROOT, "data", "parquet", "stock_data")
    all_stems = []
    for f in sorted(os.listdir(stock_dir)):
        if not f.endswith(".parquet"):
            continue
        stem = f[: -len(".parquet")]
        if "_" not in stem:
            continue
        code, _, market = stem.partition("_")
        if market in ("SH", "SZ", "BJ") and code.isdigit():
            all_stems.append((code, market))
    print(f"[update] stock_data 池 {len(all_stems)} 只；"
          f"baostock 已有 {len(have)} 只", flush=True)

    # 已知 baostock 拉不到的（北交所：baostock 不支持 bj. 前缀）→ 不重复试
    known_bad = set()
    fail_path = os.path.join(OUT_DIR, "_failures.txt")
    if os.path.exists(fail_path):
        with open(fail_path) as fh:
            for line in fh:
                parts = line.split()
                if parts:
                    known_bad.add(parts[0])

    todo = []
    skipped = 0
    for code, market in all_stems:
        stem = f"{code}_{market}"
        if stem in known_bad and not args.full_refresh:
            skipped += 1
            continue
        if args.full_refresh or stem not in have:
            todo.append((bs_code(code, market), stem))
    if skipped:
        print(f"[update] 跳过已知不支持 {skipped} 只（北交所）", flush=True)

    if args.new_only:
        # 只更新已有股票的当日行
        todo = [(s, t) for s, t in todo if t in have]

    print(f"[update] 待拉取 {len(todo)} 只"
          f"{'（全量重拉）' if args.full_refresh else '（仅新增）'}",
          flush=True)

    ok = fail = 0
    t0 = time.time()
    for i, (sym, stem) in enumerate(todo, 1):
        try:
            r = fetch_one(sym, stem)
        except Exception as e:  # noqa: BLE001
            r = f"fail:{type(e).__name__}"
        if r == "ok":
            ok += 1
        else:
            fail += 1
        if i % 100 == 0:
            print(f"  [{i}/{len(todo)}] ok={ok} fail={fail} "
                  f"{time.time()-t0:.0f}s", flush=True)

    print(f"\n[update] 完成 ok={ok} fail={fail} 用时 {time.time()-t0:.0f}s",
          flush=True)
    print(f"[update] 本地文件数 {len(local_files())}", flush=True)


if __name__ == "__main__":
    main()
