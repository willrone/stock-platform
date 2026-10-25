#!/usr/bin/env python3
"""用 baostock 拉全市场日线（不复权 + 交易所口径涨跌幅）—— 2026-10-25

## 拉什么
baostock 的 `query_history_k_data_plus` 一次能拿到我们缺的全部字段：
  - close/open/high/low  adjustflag=3 → **未复权真实价**（可照着下单算股数金额）
  - pctChg              → **交易所口径涨跌幅，已做除权除息调整**（不会因送转跳空）
  - tradestatus         → 1=正常交易 0=停牌（可成交性过滤）
  - isST                → 当时是否 ST（避免用今天的 ST 名单过滤历史 → 生存偏差）
  - turn / amount       → 换手率 / 成交额（流动性约束）
  - peTTM/pbMRQ/psTTM   → 估值（后续可选因子）

## 为什么用 pctChg 合成后复权
pctChg 是交易所公布的、已除权调整的真实涨跌幅。用它从数据起点累积：
    adj[t] = raw[t] × Π(1 + pctChg[k]/100)
得到**后复权序列** —— 基准固定（数据起点），历史结果永远可复现；
而前复权会随每天新增数据变化，导致回测不可复现。

## 输出
data/parquet/baostock_daily/<code>_<SH|SZ|BJ>.parquet
  列：date, open, high, low, close, volume, amount, pct_chg, turn, is_st, trade_status

用法：python3 fetch_baostock.py --workers 6 [--limit N]
断点续传：已存在的跳过。
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from multiprocessing import Pool

import pandas as pd

ROOT = "/Users/rone/Projects/stock-platform"
STOCK_DIR = os.path.join(ROOT, "data", "parquet", "stock_data")
OUT_DIR = os.path.join(ROOT, "data", "parquet", "baostock_daily")
START = "2021-01-01"
END = "2026-12-31"

FIELDS = (
    "date,code,open,high,low,close,volume,amount,"
    "pctChg,turn,isST,tradestatus,peTTM,pbMRQ,psTTM"
)
COLS = [
    "date", "code", "open", "high", "low", "close", "volume", "amount",
    "pct_chg", "turn", "is_st", "trade_status", "pe_ttm", "pb_mrq", "ps_ttm",
]


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
            codes.append((code, market))
    return codes


def _to_bs(code: str, market: str) -> str:
    prefix = {"SH": "sh.", "SZ": "sz.", "BJ": "bj."}[market]
    return f"{prefix}{code}"


def _init_worker() -> None:
    import baostock as bs

    bs.login()
    os.environ["BAOSTOCK_SOCKET_TIMEOUT"] = "30"


def fetch_one(job: tuple[str, str]) -> tuple[str, str]:
    code, market = job
    stem = f"{code}_{market}"
    out_path = os.path.join(OUT_DIR, f"{stem}.parquet")
    if os.path.exists(out_path):
        return stem, "skip"
    try:
        import baostock as bs

        rs = bs.query_history_k_data_plus(
            _to_bs(code, market),
            FIELDS,
            start_date=START,
            end_date=END,
            frequency="d",
            adjustflag="3",   # 3 = 不复权（真实价）
        )
        if rs.error_code != "0":
            return stem, f"fail:{rs.error_code}:{rs.error_msg[:60]}"
        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
        if not rows:
            return stem, "empty"
        df = pd.DataFrame(rows, columns=rs.fields)
        # rs.fields 是 baostock 的原始字段名（pctChg / isST / tradestatus ...），
        # 先重命名再选列
        df = df.rename(
            columns={
                "pctChg": "pct_chg",
                "isST": "is_st",
                "tradestatus": "trade_status",
                "peTTM": "pe_ttm",
                "pbMRQ": "pb_mrq",
                "psTTM": "ps_ttm",
            }
        )
        df = df[COLS].copy()
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
        return stem, "ok"
    except Exception as e:  # noqa: BLE001
        return stem, f"fail:{type(e).__name__}:{str(e)[:60]}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    jobs = universe()
    if args.limit:
        jobs = jobs[: args.limit]
    print(f"[fetch_baostock] 目标 {len(jobs)} 只，进程 {args.workers}", flush=True)

    stats = {"ok": 0, "skip": 0, "empty": 0, "fail": 0}
    failures: list[str] = []
    t0 = time.time()

    with Pool(args.workers, initializer=_init_worker) as pool:
        done = 0
        for stem, status in pool.imap_unordered(fetch_one, jobs, chunksize=8):
            done += 1
            key = "fail" if status.startswith("fail") else status
            stats[key] = stats.get(key, 0) + 1
            if status.startswith("fail"):
                failures.append(f"{stem} {status}")
            if done % 200 == 0:
                el = time.time() - t0
                rate = done / el if el else 0
                eta = (len(jobs) - done) / rate if rate else 0
                print(
                    f"[{done}/{len(jobs)}] ok={stats['ok']} skip={stats['skip']} "
                    f"empty={stats['empty']} fail={stats['fail']} "
                    f"{rate:.1f}/s ETA {eta/60:.1f}min",
                    flush=True,
                )

    print(f"\n[完成] {stats}  用时 {(time.time()-t0)/60:.1f} 分钟", flush=True)
    if failures:
        fp = os.path.join(OUT_DIR, "_failures.txt")
        with open(fp, "w") as fh:
            fh.write("\n".join(failures))
        print(f"失败清单：{fp}（{len(failures)} 条）", flush=True)
    print(f"输出目录：{OUT_DIR}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
