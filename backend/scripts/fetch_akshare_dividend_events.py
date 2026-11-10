#!/usr/bin/env python3
"""抓全 A 股（含退市）的分红送配明细——§12.84 自建总回报的唯一依据。

为什么不用现成复权序列（实测，见台账 §12.84 口径审计）：
- 东财 hfq：分红股累计少 15~67pp，含配股的票直接出现负复权价（600608 末值 −262）；
- 腾讯 hfq：600608 给 −128.74%，比未复权还差 32.8pp；
- akshare 未复权「涨跌幅」：链式累计等于未复权价比，分红送转全漏；
- baostock pctChg：分红口径对（±1~4pp），但送转虚增（301018 +739pp、300963 +232pp）。
→ 只能用「未复权价 + 分红送配明细」自己算。

语义（写死，研究里不许漂移）：**持有人不参与配股，现金分红按除息日收盘再投**。
这是散户真实行为，也是唯一能用原始价+事件表复算的口径。

只保留「已实施」的方案：预案/股东大会通过/不实施的都不算钱。
退市股在东财分红接口上多数返回空 → 记入覆盖率披露（这类票的早期分红会被漏计，方向是低估其回报）。

用法：cd backend && env -u PYTHONPATH .venv-py313/bin/python3 scripts/fetch_akshare_dividend_events.py [--limit 8] [--workers 6]
输出：data/parquet/dividend_events/<num>_<SH|SZ>.parquet + _coverage.json + _failures.txt
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(ROOT, "data", "parquet", "dividend_events")
REGISTRY = os.path.join(ROOT, "data", "parquet", "baostock_full", "_registry.parquet")
FAILURES = os.path.join(OUT_DIR, "_failures.txt")
KEEP_STATUS = ("实施分配", "完成实施", "已实施", "实施")
COLS = ["code", "report_date", "announce_date", "record_date", "ex_date",
        "dps", "song_ratio", "eps", "bps", "plan_status"]


def out_name(num: str, mkt: str) -> str:
    return f"{num}_{mkt.upper()}.parquet"


def fetch_one(num: str) -> tuple[str, int, str]:
    import akshare as ak
    last_err = ""
    for attempt in range(3):
        try:
            df = ak.stock_fhps_detail_em(symbol=num)
            if df is None:
                return (num, 0, "empty")
            if len(df) == 0:
                return (num, 0, "empty")
            rows = []
            for _, r in df.iterrows():
                status = str(r.get("方案进度") or "")
                if not any(k in status for k in KEEP_STATUS) or "不实施" in status or "未实施" in status:
                    continue
                ex = r.get("除权除息日")
                if ex is None or (isinstance(ex, float) and pd.isna(ex)):
                    continue
                def f(k):
                    v = r.get(k)
                    try:
                        return None if v is None or pd.isna(v) else float(v) / 10.0
                    except Exception:
                        return None
                rows.append({"report_date": pd.Timestamp(r["报告期"]) if not pd.isna(r.get("报告期")) else None,
                             "announce_date": pd.Timestamp(r["最新公告日期"]) if not pd.isna(r.get("最新公告日期")) else None,
                             "record_date": pd.Timestamp(r["股权登记日"]) if not pd.isna(r.get("股权登记日")) else None,
                             "ex_date": pd.Timestamp(ex),
                             "dps": f("现金分红-现金分红比例"),
                             "song_ratio": f("送转股份-送转总比例"),
                             "eps": None if pd.isna(r.get("每股收益")) else float(r.get("每股收益")),
                             "bps": None if pd.isna(r.get("每股净资产")) else float(r.get("每股净资产")),
                             "plan_status": status})
            if not rows:
                return (num, 0, "no-implemented-plan")
            out = pd.DataFrame(rows, columns=COLS)
            out["code"] = num
            out = out.sort_values("ex_date").drop_duplicates(["ex_date"])
            path = os.path.join(OUT_DIR, out_name(num, "SH" if num.startswith(("6", "9")) else "SZ"))
            tmp = path + ".tmp"
            out.to_parquet(tmp, index=False)
            os.replace(tmp, path)
            return (num, len(out), "")
        except Exception as e:
            last_err = repr(e)[:90]
            time.sleep(1 + attempt + random.random())
    return (num, 0, f"fail {last_err}")


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
    reg = reg[~reg["code"].str.startswith("bj.")]
    done: set[str] = set()
    todo = list(reg["num"])
    if args.only_failures and os.path.exists(FAILURES):
        bad = {ln.split(" ")[0] for ln in open(FAILURES) if ln.strip()}
        todo = [n for n in reg["num"] if n in bad]
    if not args.force:
        if os.path.isdir(OUT_DIR):
            for f in os.listdir(OUT_DIR):
                if f.endswith(".parquet"):
                    done.add(f[: -len(".parquet")].rsplit("_", 1)[0])
        todo = [n for n in todo if n not in done]
    if args.limit:
        todo = todo[: args.limit]
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"待抓 {len(todo)} 只分红送配明细，{args.workers} 线程")
    if not todo:
        print("全部已存在")
        return 0

    t0 = time.time()
    rows = fails = 0
    failures, delisted_ok, delisted_total = [], 0, 0
    delisted_nums = set(reg[reg["status"] != 1]["num"])
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(fetch_one, n): n for n in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            num, n, err = fut.result()
            if err:
                fails += 1
                failures.append(f"{num} {err}")
            else:
                rows += n
                if num in delisted_nums:
                    delisted_ok += 1
            if i % 300 == 0 or i == len(todo):
                el = time.time() - t0
                print(f"[{i}/{len(todo)}] 事件行 {rows} 空/失败 {fails} 用时 {el:.0f}s "
                      f"速率 {i/max(el,1):.2f}/s", flush=True)
            time.sleep(0.04 + random.random() * 0.08)
    with open(FAILURES, "w") as fh:
        fh.write("\n".join(failures) + ("\n" if failures else ""))
    delisted_total = len(delisted_nums)
    cov = {"stocks_requested": len(todo),
           "files_now": len([f for f in os.listdir(OUT_DIR) if f.endswith(".parquet")]),
           "event_rows_total_est": rows, "empty_or_failed": fails,
           "delisted_total": delisted_total, "delisted_with_events": delisted_ok,
           "note": "退市股分红接口多为空 → 其早期现金分红会被漏计，方向是低估退市股回报（对『买了迟早会涨』是保守偏差）"}
    json.dump(cov, open(os.path.join(OUT_DIR, "_coverage.json"), "w"), ensure_ascii=False, indent=2)
    print(f"\n完成：事件行 {rows}，空/失败 {fails}/{len(todo)}，退市股有分红记录 {delisted_ok}/{delisted_total}")
    print(json.dumps(cov, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
