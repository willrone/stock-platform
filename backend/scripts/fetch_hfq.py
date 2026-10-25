#!/usr/bin/env python3
"""拉取全市场后复权(hfq)日线 —— 2026-10-25

## 为什么需要
stock_data 存的是**未复权真实价**（复权因子恒为 1.0，已实测）。
好处是金额/股数可以直接照着下单；坏处是**除权除息日价格跳空**：
  - 实测 000001.SZ 有 6 天 |涨跌幅| > 30%（1991 年 -44%、1993 年 -50%）
  - 后果 1：除权日被误判成「跌停卖不出」→ 持仓卡住
  - 后果 2：MIN/MAX 因子在除权日出现假极值 → 选股被污染
  - 后果 3：持有期跨除权日的收益被系统性**低估**（真实收益没这么差）

## 为什么用后复权(hfq)而不是前复权(qfq)
qfq 的基准是「最新价」，每多一天数据，所有历史价都会变 →
回测结果不可复现。hfq 基准固定，历史结果永远可复现。

## 输出
data/parquet/stock_data_hfq/<code>_<SH|SZ|BJ>.parquet
  列：date, open, high, low, close, volume, amount（成交额）

用法：python3 fetch_hfq.py [--workers 4] [--limit N]
断点续传：已存在的文件跳过。
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

ROOT = "/Users/rone/Projects/stock-platform"
STOCK_DIR = os.path.join(ROOT, "data", "parquet", "stock_data")
OUT_DIR = os.path.join(ROOT, "data", "parquet", "stock_data_hfq")
START = "20210101"
END = "20261231"

COLS = ["日期", "开盘", "收盘", "最高", "最低", "成交量", "成交额"]


def universe() -> list[str]:
    codes = []
    for f in sorted(os.listdir(STOCK_DIR)):
        if not f.endswith(".parquet"):
            continue
        stem = f[: -len(".parquet")]
        if "_" not in stem:
            continue
        code, _, market = stem.partition("_")
        if market in ("SH", "SZ", "BJ") and code.isdigit():
            codes.append(f"{code}.{market}")
    return codes


def fetch_one(symbol: str, retries: int = 3) -> tuple[str, str]:
    """symbol 形如 000001.SZ。返回 (symbol, 状态)。"""
    stem = symbol.replace(".", "_")
    out_path = os.path.join(OUT_DIR, f"{stem}.parquet")
    if os.path.exists(out_path):
        return symbol, "skip"
    plain = symbol.split(".")[0]
    last_err = ""
    for attempt in range(retries):
        try:
            import akshare as ak

            df = ak.stock_zh_a_hist(
                symbol=plain,
                period="daily",
                start_date=START,
                end_date=END,
                adjust="hfq",
            )
            if df is None or len(df) == 0:
                return symbol, "empty"
            d = df.rename(
                columns={
                    "日期": "date",
                    "开盘": "open",
                    "收盘": "close",
                    "最高": "high",
                    "最低": "low",
                    "成交量": "volume",
                    "成交额": "amount",
                }
            )[["date", "open", "high", "low", "close", "volume", "amount"]].copy()
            d["date"] = pd.to_datetime(d["date"])
            d = d.sort_values("date").reset_index(drop=True)
            tmp = out_path + ".tmp"
            d.to_parquet(tmp, index=False)
            os.replace(tmp, out_path)
            return symbol, "ok"
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(1.5 * (attempt + 1))
    return symbol, f"fail:{last_err[:80]}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="0 = 全部")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    codes = universe()
    if args.limit:
        codes = codes[: args.limit]
    print(f"[fetch_hfq] 目标 {len(codes)} 只，线程 {args.workers}", flush=True)

    stats = {"ok": 0, "skip": 0, "empty": 0, "fail": 0}
    failures: list[str] = []
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_one, c): c for c in codes}
        done = 0
        for fut in as_completed(futures):
            sym, status = fut.result()
            done += 1
            key = "fail" if status.startswith("fail") else status
            stats[key] = stats.get(key, 0) + 1
            if status.startswith("fail"):
                failures.append(f"{sym} {status}")
            if done % 200 == 0:
                el = time.time() - t0
                rate = done / el if el else 0
                eta = (len(codes) - done) / rate if rate else 0
                print(
                    f"[{done}/{len(codes)}] ok={stats['ok']} skip={stats['skip']} "
                    f"empty={stats['empty']} fail={stats['fail']} "
                    f"{rate:.1f}/s ETA {eta/60:.1f}min",
                    flush=True,
                )

    print(f"\n[完成] {stats}  用时 {(time.time()-t0)/60:.1f} 分钟", flush=True)
    if failures:
        fp = os.path.join(OUT_DIR, "_failures.txt")
        with open(fp, "w") as fh:
            fh.write("\n".join(failures))
        print(f"失败清单已写入 {fp}（{len(failures)} 条）", flush=True)
    print(f"输出目录：{OUT_DIR}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
