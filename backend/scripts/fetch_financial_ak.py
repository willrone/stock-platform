#!/usr/bin/env python3
"""抓取财务指标并计算 RankIC（2026-10-27 04:50）

## 为什么改用 akshare
baostock 的财务接口实测单次查询 20+ 秒（21600 次查询 = 120 小时，不可行）。
akshare 的 `stock_financial_analysis_indicator` 单只 <1 秒（17 it/s），
一次给 100+ 个财务指标的历史序列。

## PIT 对齐（重要）
akshare 的「日期」是**报告期**，不是公布日。A 股财报公布有法定截止日：
年报 4/30、一季报 4/30、半年报 8/31、三季报 10/31。
本脚本按 **报告期 + 120 天** 作为「最早可得日」→ 覆盖所有法定截止日
（年报 12-31 + 120 = 4-30 ✓），对季报偏保守（滞后 2~3 个月）。
**偏保守会让 IC 偏低而不会虚高** —— 可以接受。

## 用法
  env -u PYTHONPATH -u PYTHONHOME backend/.venv-py313/bin/python3 \
    backend/scripts/fetch_financial_ak.py --limit 800 --workers 6
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(ROOT, "data", "fundamentals")

#: akshare 列名 → 输出列名（挑 A 股横截面上最有信息量的一批）
KEEP = {
    "净资产收益率(%)": "roe",
    "主营业务利润率(%)": "gross_margin",
    "总资产净利润率(%)": "roa",
    "销售净利率(%)": "net_margin",
    "营业利润率(%)": "operating_margin",
    "成本费用利润率(%)": "cost_margin",
    "净资产增长率(%)": "roe_yoy",
    "总资产增长率(%)": "asset_yoy",
    "主营业务收入增长率(%)": "revenue_yoy",
    "净利润增长率(%)": "profit_yoy",
    "资产负债率(%)": "debt_ratio",
    "流动比率": "current_ratio",
    "速动比率": "quick_ratio",
    "总资产周转率(次)": "asset_turnover",
    "存货周转率(次)": "inventory_turnover",
    "应收账款周转率(次)": "receivable_turnover",
    "每股收益_调整后(元)": "eps_adj",
    "每股净资产_调整前(元)": "bps",
    "每股经营性现金流(元)": "cfps",
    "主营业务收入(元)": "revenue",
    "净利润(元)": "net_profit",
    "净资产收益率_调整(%)": "roe_adj",
    "总资产周转率": "asset_turnover_raw",
    "经营现金流/营业收入(%)": "cfo_to_revenue",
    "经营现金流/净利润(%)": "cfo_to_profit",
}

#: 报告期 → 最早可得日滞后（自然日）。120 天覆盖所有法定公布截止日。
AVAIL_LAG_DAYS = 120


def fetch_one(code: str, start_year: str) -> List[list]:
    """返回 [[avail_date, {factor: value}], ...]"""
    try:
        import akshare as ak
        df = ak.stock_financial_analysis_indicator(symbol=code, start_year=start_year)
    except Exception:
        return []
    if df is None or len(df) == 0:
        return []
    date_col = "日期" if "日期" in df.columns else df.columns[0]
    out = []
    for _, row in df.iterrows():
        d = row[date_col]
        try:
            ts = pd.Timestamp(str(d))
        except Exception:
            continue
        if ts is pd.NaT or ts is None:
            continue
        avail = (ts + pd.Timedelta(days=AVAIL_LAG_DAYS)).strftime("%Y-%m-%d")
        vals = {}
        for src, dst in KEEP.items():
            if src not in df.columns:
                continue
            v = row[src]
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if fv == fv and abs(fv) < 1e12:
                vals[dst] = fv
        if vals:
            out.append([avail, vals])
    return out


def main() -> int:
    global pd
    import pandas as pd  # noqa: F811

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=800)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--start-year", default="2018")
    ap.add_argument("--out", default="financial_ak.json")
    args = ap.parse_args()

    codes: List[str] = []
    inst = os.path.join(ROOT, "data", "qlib_official_data", "instruments", "all.txt")
    with open(inst) as fh:
        for line in fh:
            parts = line.strip().split("\t")
            if parts and "_" in parts[0]:
                c, _, m = parts[0].partition("_")
                if m.upper() in ("SH", "SZ", "BJ") and c.isdigit():
                    codes.append(c)   # akshare 用 6 位代码
    codes = sorted(codes)[: args.limit]
    print(f"[fin] {len(codes)} 只，{args.workers} 线程并发，起始 {args.start_year}",
          flush=True)

    t0 = time.perf_counter()
    data: Dict[str, List[list]] = {}
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_one, c, args.start_year): c for c in codes}
        for fut in as_completed(futs):
            c = futs[fut]
            try:
                recs = fut.result()
            except Exception:
                recs = []
            if recs:
                data[c] = recs
            done += 1
            if done % 100 == 0:
                el = time.perf_counter() - t0
                print(f"  {done}/{len(codes)}  有数据 {len(data)}  用时 {el:.0f}s  "
                      f"预计剩余 {el/done*(len(codes)-done):.0f}s", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, args.out)
    with open(path, "w") as fh:
        json.dump(data, fh, ensure_ascii=False)
    n_fac = len({f for recs in data.values() for _, v in recs for f in v})
    print(f"[fin] 完成：{len(data)}/{len(codes)} 只有数据，{n_fac} 个因子 → {path}  "
          f"（{time.perf_counter()-t0:.0f}s）", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
