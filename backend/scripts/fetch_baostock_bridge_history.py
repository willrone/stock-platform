#!/usr/bin/env python3
"""baostock 补 2012-01-01 → 2020-12-31 全宇宙日线（含退市股）——§12.84 唯一可用的收益口径来源。

为什么必须这一层（实测结论，不是猜）：
- 东财(akshare) `adjust="hfq"` 后复权序列**对含配股的票用减法复权**，会出现负价：
  600608 后复权序列里有 15 根 ≤0、末值 −262.0；该股 2021-2026 无分红事件，
  真实总回报 = 未复权价比 = −95.90%，baostock pctChg 链给出 −95.90%（完全吻合），hfq 给出 −155.87%。
- 腾讯 hfq 同病：600608 给出 −128.74%，比未复权还差 32.8pp。
- akshare 未复权日线里的「涨跌幅」**既不是未复权也不是全复权**（东财内部按前复权算），
  链式累计与 baostock pctChg 链中位差 5.5pp、p90 差 60pp → 不能当总回报。
- 唯一与「分红明细手工地面真相」吻合的口径：**baostock pctChg 链**（§12.83 已独立验证过一次）。

所以：价格/手数层用 akshare_full（未复权 close，已对腾讯现价核对）；
总回报层用 baostock pctChg 链（2021 起已有，本脚本补 2012-2020，并把 PE/PB 一起带回来）。

速度实测：单会话单只 2012→今 14–17 秒。本脚本每线程只登录一次并复用会话（每次查询都重登录会触发
10001011「您还没有登录」），6 线程 + 断点续传。

用法：
    cd backend && env -u PYTHONPATH .venv-py313/bin/python3 scripts/fetch_baostock_bridge_history.py --limit 8
    cd backend && env -u PYTHONPATH .venv-py313/bin/python3 scripts/fetch_baostock_bridge_history.py --workers 6
输出：data/parquet/baostock_bridge/<num>_<SH|SZ>.parquet，失败清单 _failures.txt
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(ROOT, "data", "parquet", "baostock_bridge")
REGISTRY = os.path.join(ROOT, "data", "parquet", "baostock_full", "_registry.parquet")
FAILURES = os.path.join(OUT_DIR, "_failures.txt")
START, END = "2012-01-01", "2020-12-31"
FIELDS = "date,code,open,high,low,close,volume,amount,turn,pctChg,peTTM,pbMRQ,tradestatus,isST"
COLS = ["date", "open", "high", "low", "close", "volume", "amount", "turnover",
        "pct_chg", "pe_ttm", "pb_mrq", "tradestatus", "is_st"]
_relogin = threading.local()


def out_name(num: str, mkt: str) -> str:
    return f"{num}_{mkt.upper()}.parquet"


def _ensure_login(bs):
    st = getattr(_relogin, "state", None)
    if st is None:
        lg = bs.login()
        if lg.error_code != "0":
            raise RuntimeError(f"login {lg.error_code} {lg.error_msg}")
        _relogin.state = True
        _relogin.login_ts = time.time()
    elif time.time() - getattr(_relogin, "login_ts", 0) > 1800:      # 会话 30 分钟强制续登
        bs.logout()
        _relogin.state = None
        return _ensure_login(bs)


def fetch_one(code: str) -> tuple[str, int, str]:
    import baostock as bs
    for attempt in range(4):
        try:
            _ensure_login(bs)
            rs = bs.query_history_k_data_plus(code, FIELDS, start_date=START, end_date=END,
                                              frequency="d", adjustflag="3")
            rows = []
            while rs.next():
                rows.append(rs.get_row_data())
            if rs.error_code == "0":
                if not rows:
                    return (code, 0, "empty")
                df = pd.DataFrame(rows, columns=rs.fields)
                df = df.rename(columns={"turn": "turnover", "pctChg": "pct_chg",
                                        "peTTM": "pe_ttm", "pbMRQ": "pb_mrq", "isST": "is_st"})
                df["date"] = pd.to_datetime(df["date"])
                df = df.sort_values("date").drop_duplicates("date")
                for c in COLS[1:]:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                num, mkt = code.split(".")[1], code.split(".")[0]
                path = os.path.join(OUT_DIR, out_name(num, mkt))
                tmp = path + ".tmp"
                df[COLS].to_parquet(tmp, index=False)
                os.replace(tmp, path)
                return (code, len(df), "")
            if rs.error_code in ("10001011", "10002007"):            # 会话掉了：续登再来
                bs.logout()
                _relogin.state = None
                time.sleep(2 + 2 * attempt)
                continue
            return (code, 0, f"err {rs.error_code} {rs.error_msg}")
        except Exception as e:
            _relogin.state = None
            time.sleep(2 + 2 * attempt)
            last = repr(e)[:90]
    return (code, 0, "retry exhausted")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only-failures", action="store_true")
    args = ap.parse_args()

    reg = pd.read_parquet(REGISTRY)
    reg["status"] = pd.to_numeric(reg["status"], errors="coerce")
    reg["num"] = reg["code"].str.split(".").str[1]
    ipo = reg["ipoDate"].astype(str)
    todo = [c for c, i in zip(reg["code"], ipo) if i == "" or i >= START]   # 2012 前上市才有 2012-2020 bar
    if args.only_failures and os.path.exists(FAILURES):
        bad = {ln.split(" ")[0] for ln in open(FAILURES) if ln.strip()}
        todo = [c for c in todo if c in bad]
    if not args.force:
        done = set()
        if os.path.isdir(OUT_DIR):
            for f in os.listdir(OUT_DIR):
                if f.endswith(".parquet"):
                    num, mkt = f[: -len(".parquet")].rsplit("_", 1)
                    done.add(f"{mkt.lower()}.{num}")
        todo = [c for c in todo if c not in done]
    if args.limit:
        todo = todo[: args.limit]
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"待拉 {len(todo)} 只（{START}→{END}），{args.workers} 线程，每线程单会话复用")
    if not todo:
        print("全部已存在")
        return 0

    t0 = time.time()
    rows = fails = 0
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(fetch_one, c): c for c in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            code, n, err = fut.result()
            if err:
                fails += 1
                failures.append(f"{code} {err}")
            else:
                rows += n
            if i % 100 == 0 or i == len(todo):
                el = time.time() - t0
                print(f"[{i}/{len(todo)}] 行数 {rows} 失败 {fails} 用时 {el:.0f}s "
                      f"速率 {i/max(el,1):.2f}/s 外推剩余 {(len(todo)-i)/max(i/max(el,1),0.01)/60:.0f} 分钟", flush=True)
    with open(FAILURES, "w") as fh:
        fh.write("\n".join(failures) + ("\n" if failures else ""))
    print(f"\n完成：成功 {len(todo)-fails}/{len(todo)}，行数 {rows}，失败 {fails}")
    if failures:
        print(f"失败清单 → {FAILURES}")
    return 0 if fails <= max(5, len(todo) * 0.02) else 1


if __name__ == "__main__":
    sys.exit(main())
