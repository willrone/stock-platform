#!/usr/bin/env python3
"""补长窗口 + 补幸存者偏差：用 baostock 拉 2012-01-01 至今的**全 A 股（含已退市）**日线。

为什么要这一步（2026-11-10，§12.83 的前置）：
- 现役 `data/parquet/baostock_daily` 起点是 2021-01-01（`fetch_baostock.py` 里写死 START="2021-01-01"）
  → 3 年持有的入场队列只剩 2022、2023 两期，K=20 只有 40 只样本，命中率标准误 ≈7.8pp。
- 更要命的是**宇宙只从存续股目录枚举** → 池里 0 只退市股，所有"买了迟早会涨"的命中率都是上界。
- 实测：`baostock.query_stock_basic()` 给出 type=1 共 5,565 只，其中 **341 只 status=0（已退市）**，
  且历史日线可取（sz.000018 神城A退 末根 2020-01-07 收 0.27；sz.300104 乐视退 末根 2020-07-21 收 0.18）。
  peTTM/pbMRQ 自 2012-01-04 起覆盖 100%（实测 600000 2012-2018 全非空）。

口径与现役层一致（不另起炉灶）：
- `adjustflag="3"` → OHLC 是**未复权真实价**（可照着下单算手数金额）；
- `pctChg` 是交易所口径、**已做除权除息调整**的涨跌幅 → 后复权总回报链 `adj = cumprod(1 + pctChg/100)`；
- `isST` / `tradestatus` 是**当时**的 ST 与停牌状态（不许拿今天的 ST 名单去过滤历史）；
- 基准固定（起点 2012-01-01）= 后复权，可复现；前复权会随新数据漂移（§12.81 教训）。

输出（新目录，不动现役层，现有回测/测试不受影响）：
- data/parquet/baostock_full/<num>_<SH|SZ>.parquet    列同 baostock_daily
- data/parquet/baostock_full/_registry.parquet        code/name/ipoDate/outDate/status（退市可见性就靠这张表）
- data/parquet/baostock_full/_failures.txt            断点续传用：已存在的代码默认跳过

用法：
    cd backend && env -u PYTHONPATH .venv-py313/bin/python3 scripts/fetch_baostock_full_history.py --workers 6
    先小样：... --limit 12   重跑失败项：... --only-failures
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from multiprocessing import Pool

import pandas as pd

ROOT = "/Users/rone/Projects/stock-platform"
OUT_DIR = os.path.join(ROOT, "data", "parquet", "baostock_full")
REGISTRY = os.path.join(OUT_DIR, "_registry.parquet")
FAILURES = os.path.join(OUT_DIR, "_failures.txt")
START = "2012-01-01"
END = "2026-12-31"

FIELDS = ("date,code,open,high,low,close,volume,amount,"
          "pctChg,turn,isST,tradestatus,peTTM,pbMRQ,psTTM")
COLS = ["date", "code", "open", "high", "low", "close", "volume", "amount",
        "pct_chg", "turn", "is_st", "trade_status", "pe_ttm", "pb_mrq", "ps_ttm"]

_STATE: dict = {}


def _out_name(bs_code: str) -> str:
    mkt, num = bs_code.split(".")
    return f"{num}_{mkt.upper()}.parquet"


def _init_worker() -> None:
    import baostock as bs
    bs.login()
    _STATE["bs"] = bs


def fetch_one(bs_code: str) -> tuple[str, int, str]:
    """拉一只；baostock 连接会随机 RemoteDisconnected，重试三次；返回 (code, 行数, 错误)。"""
    import baostock as bs
    path = os.path.join(OUT_DIR, _out_name(bs_code))
    last_err = ""
    for attempt in range(3):
        try:
            rs = bs.query_history_k_data_plus(bs_code, FIELDS, start_date=START, end_date=END,
                                              frequency="d", adjustflag="3")
            if rs.error_code != "0":
                last_err = f"{rs.error_code}:{rs.error_msg}"
                time.sleep(1.0 + attempt)
                continue
            rows = []
            while rs.next():
                rows.append(rs.get_row_data())
            if not rows:
                return (bs_code, 0, "no_rows")
            df = pd.DataFrame(rows, columns=rs.fields)
            for c in COLS:
                if c not in df.columns:
                    df[c] = None
            df = df[COLS]
            df["date"] = pd.to_datetime(df["date"])
            for c in ("open", "high", "low", "close", "volume", "amount", "pct_chg",
                      "turn", "is_st", "trade_status", "pe_ttm", "pb_mrq", "ps_ttm"):
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.sort_values("date").drop_duplicates("date")
            tmp = path + ".tmp"
            df.to_parquet(tmp, index=False)
            os.replace(tmp, path)
            return (bs_code, len(df), "")
        except Exception as exc:                      # 连接层错误是常态，必须重试而不是整批放弃
            last_err = f"{type(exc).__name__}:{str(exc)[:80]}"
            try:                                       # 连接断了要重新登录，否则后续全部失败
                bs.logout()
                bs.login()
            except Exception:
                pass
            time.sleep(1.5 + attempt * 1.5)
    return (bs_code, 0, last_err)


def load_registry(force: bool = False) -> pd.DataFrame:
    """注册表：type=1 的全部 A 股，含 status=0（已退市）与 outDate。幸存者偏差的解药就是这张表。"""
    if os.path.exists(REGISTRY) and not force:
        return pd.read_parquet(REGISTRY)
    import baostock as bs
    lg = bs.login()                                  # 主进程也要登录，否则 query_stock_basic 直接 "you don't login"
    if lg.error_code != "0":
        raise SystemExit(f"baostock login 失败: {lg.error_code} {lg.error_msg}")
    rs = bs.query_stock_basic()
    if rs.error_code != "0":
        raise SystemExit(f"query_stock_basic 失败: {rs.error_code} {rs.error_msg}")
    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    df = pd.DataFrame(rows, columns=rs.fields)
    df = df[df["type"] == "1"].copy()                 # 1=股票；剔指数(type=2)
    df["status"] = pd.to_numeric(df["status"], errors="coerce")
    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_parquet(REGISTRY, index=False)
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 只（冒烟用）")
    ap.add_argument("--force", action="store_true", help="重拉已存在的代码")
    ap.add_argument("--only-failures", action="store_true", help="只重试 _failures.txt 里的代码")
    args = ap.parse_args()

    reg = load_registry(force=False)
    delisted = reg[reg["status"] != 1]
    print(f"注册表：A 股 {len(reg)} 只（上市 {(reg['status']==1).sum()}，已退市 {len(delisted)}）")

    todo = list(reg["code"])
    if args.only_failures and os.path.exists(FAILURES):
        bad = [ln.split(" ")[0] for ln in open(FAILURES) if ln.strip()]
        todo = [c for c in reg["code"] if c in set(bad)]
    if not args.force:
        todo = [c for c in todo if not os.path.exists(os.path.join(OUT_DIR, _out_name(c)))]
    if args.limit:
        todo = todo[: args.limit]
    print(f"待拉 {len(todo)} 只，起点 {START}，{args.workers} 进程")
    if not todo:
        print("全部已存在，无需拉取")
        return 0

    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()
    ok_rows = fails = 0
    delisted_ok = 0
    delisted_set = set(delisted["code"])
    failures: list[str] = []
    with Pool(processes=args.workers, initializer=_init_worker) as pool:
        for i, (code, n, err) in enumerate(pool.imap_unordered(fetch_one, todo, chunksize=8), 1):
            if err:
                fails += 1
                failures.append(f"{code} fail:{err}")
            else:
                ok_rows += n
                if code in delisted_set:
                    delisted_ok += 1
            if i % 100 == 0 or i == len(todo):
                el = time.time() - t0
                print(f"[{i}/{len(todo)}] 行数 {ok_rows} 失败 {fails} 退市股已拉 {delisted_ok} "
                      f"用时 {el:.0f}s 速率 {i/max(el,1):.2f}/s", flush=True)
    with open(FAILURES, "w") as fh:
        fh.write("\n".join(failures) + ("\n" if failures else ""))
    try:
        import baostock as bs
        bs.logout()
    except Exception:
        pass

    print(f"\n完成：成功 {len(todo)-fails}/{len(todo)}，行数 {ok_rows}，失败 {fails}，退市股成功 {delisted_ok}/{len(delisted)}")
    if failures:
        print(f"失败清单 → {FAILURES}")
    if fails > len(todo) * 0.1:
        print("警告：失败率 >10%，先 --only-failures 重试再采信覆盖率数字")
    return 0 if fails <= len(todo) * 0.1 else 1


if __name__ == "__main__":
    sys.exit(main())
