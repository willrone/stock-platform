#!/usr/bin/env python3
"""抓取全市场股票名称（用于识别 ST / 退市整理期）—— 2026-10-26

## 为什么
P0-2「涨跌停阈值按板块动态」有两个反向错误：
  - 20% 板（创业板/科创板）股票涨 9.5%~20% 会被误判涨停剔除 → 砍 alpha
  - **ST 股涨停只有 5%**，用 10% 阈值就拦不住 → 买入即亏损

要判 ST 就必须有股票名称表。`qlib_official_data/instruments/all.txt` 只有
symbol/start/end 三列，**没有名称**；`data/parquet/stock_data` 也没有。
baostock 的 `query_stock_basic` 返回 `code_name`（含 "ST"/"退"），是本地库。

输出：`data/stocks/st_names.json`  {code: {"name": str, "is_st": bool, "is_delisting": bool}}
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "data", "stocks", "st_names.json")


def main() -> int:
    import baostock as bs

    lg = bs.login()
    if lg.error_code != "0":
        print(f"[st] baostock 登录失败: {lg.error_code} {lg.error_msg}", file=sys.stderr)
        return 1
    try:
        rs = bs.query_stock_basic()
        if rs.error_code != "0":
            print(f"[st] query_stock_basic 失败: {rs.error_msg}", file=sys.stderr)
            return 1
        out: dict = {}
        n = 0
        while rs.next():
            row = rs.get_row_data()
            # (code, code_name, ipoDate, outDate, type, status)
            code, name = row[0], row[1]
            if not code or not name:
                continue
            upper = name.upper().replace(" ", "")
            is_st = "ST" in upper or "退" in name
            is_del = name.strip().endswith("退")
            out[code] = {
                "name": name,
                "is_st": bool(is_st and not is_del),
                "is_delisting": bool(is_del),
            }
            n += 1
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        with open(OUT, "w") as fh:
            json.dump(out, fh, ensure_ascii=False)
        n_st = sum(1 for v in out.values() if v["is_st"])
        n_del = sum(1 for v in out.values() if v["is_delisting"])
        print(f"[st] 抓到 {n} 只，其中 ST {n_st} 只、退市整理 {n_del} 只 → {OUT}")
        return 0
    finally:
        bs.logout()


if __name__ == "__main__":
    raise SystemExit(main())
