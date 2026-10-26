#!/usr/bin/env python3
"""抓取 baostock 财务数据并构建 PIT 因子面板（2026-10-27 03:25）

## 动机（今晚最重要的结论导向）
「单一股票池 + TopK」框架内，所有对现有 42 个价量因子的操作
（加权 / 减法 / 加趋势 / 缩 topk / 中性化 / 止损 / 择时）**全部证伪**。
唯一还能带来**新信息**的路径是引入**财务因子**（质量 / 成长 / 盈利能力）——
而且它与价量因子天然低相关，不增加持仓重叠。

## PIT 正确性
baostock 的财务数据带 **`pubDate`（实际公布日）** —— 不是报告期。
因子值从 `pubDate` 当天才可用 → 严格防未来函数。
（用 statDate 对齐是常见错误，会让回测看到「当时还没公布的财报」。）

## 因子
- 质量：roeAvg、dupontROE、npMargin、gpMargin、CFOToNP、CFOToOR、ebitToInterest
- 成长：YOYNI、YOYEPSBasic、YOYPNI、YOYEquity、YOYAsset
- 杠杆/流动性：liabilityToAsset、currentRatio、quickRatio、YOYLiability
- 规模/结构：liqaShare(流通占比)、dupontAssetStoEquity、dupontAssetTurn

## 用法
  env -u PYTHONPATH -u PYTHONHOME backend/.venv-py313/bin/python3 \
    backend/scripts/fetch_fundamentals.py --limit 300
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List

ROOT = "/Users/rone/Projects/stock-platform"
OUT_DIR = os.path.join(ROOT, "data", "fundamentals")

#: (接口名, 字段名) → 输出列名
FACTORS = {
    "profit": {
        "roeAvg": "roe_avg",
        "npMargin": "np_margin",
        "gpMargin": "gp_margin",
        "epsTTM": "eps_ttm",
        "MBRevenue": "revenue",
        "liqaShare": "float_share",
    },
    "dupont": {
        "dupontROE": "dupont_roe",
        "dupontAssetStoEquity": "equity_multiplier",
        "dupontAssetTurn": "asset_turnover",
        "dupontPnitoni": "dupont_net_margin",
        "dupontTaxBurden": "tax_burden",
        "dupontIntburden": "int_burden",
    },
    "balance": {
        "currentRatio": "current_ratio",
        "quickRatio": "quick_ratio",
        "YOYLiability": "yoy_liability",
        "liabilityToAsset": "liability_to_asset",
        "assetToEquity": "asset_to_equity",
    },
    "cashflow": {
        "CAToAsset": "ca_to_asset",
        "tangibleAssetToAsset": "tangible_to_asset",
        "ebitToInterest": "ebit_to_interest",
        "CFOToOR": "cfo_to_revenue",
        "CFOToNP": "cfo_to_netprofit",
        "CFOToGr": "cfo_to_grossprofit",
    },
    "growth": {
        "YOYEquity": "yoy_equity",
        "YOYAsset": "yoy_asset",
        "YOYNI": "yoy_ni",
        "YOYEPSBasic": "yoy_eps",
        "YOYPNI": "yoy_pni",
    },
}


def bs_code(code: str) -> str:
    """600000.SH → sh.600000"""
    if "." not in code:
        return code
    num, mkt = code.split(".")
    return f"{mkt.lower()}.{num}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--start-year", type=int, default=2018)
    ap.add_argument("--end-year", type=int, default=2026)
    ap.add_argument("--out", default="fundamentals_pit.json")
    args = ap.parse_args()

    import baostock as bs

    # 股票池：与回测同一份
    codes: List[str] = []
    inst = os.path.join(ROOT, "data", "qlib_official_data", "instruments", "all.txt")
    with open(inst) as fh:
        for line in fh:
            parts = line.strip().split("\t")
            if parts and "_" in parts[0]:
                c, _, m = parts[0].partition("_")
                if m.upper() in ("SH", "SZ", "BJ") and c.isdigit():
                    codes.append(f"{c}.{m.upper()}")
    codes = sorted(codes)[: args.limit]
    print(f"[fund] 股票 {len(codes)} 只  年份 {args.start_year}~{args.end_year}", flush=True)

    lg = bs.login()
    if lg.error_code != "0":
        print(f"[fund] baostock 登录失败: {lg.error_msg}", file=sys.stderr)
        return 1

    # records: {code: [(pubDate, statDate, {factor: value})]}
    records: Dict[str, List[tuple]] = {}
    api = {
        "profit": bs.query_profit_data,
        "dupont": bs.query_dupont_data,
        "balance": bs.query_balance_data,
        "cashflow": bs.query_cash_flow_data,
        "growth": bs.query_growth_data,
    }
    t0 = time.perf_counter()
    n_q = 0
    try:
        for i, code in enumerate(codes):
            bc = bs_code(code)
            for year in range(args.start_year, args.end_year + 1):
                for q in (1, 2, 3, 4):
                    for kind, fn in api.items():
                        try:
                            rs = fn(code=bc, year=year, quarter=q)
                            n_q += 1
                        except Exception:
                            continue
                        if rs is None or rs.error_code != "0":
                            continue
                        while rs.next():
                            row = rs.get_row_data()
                            d = dict(zip(rs.fields, row))
                            pub = d.get("pubDate") or ""
                            stat = d.get("statDate") or ""
                            if not pub:
                                continue
                            vals = {}
                            for src, dst in FACTORS[kind].items():
                                v = d.get(src)
                                if v in (None, ""):
                                    continue
                                try:
                                    fv = float(v)
                                except (TypeError, ValueError):
                                    continue
                                if fv == fv:  # not NaN
                                    vals[dst] = fv
                            if vals:
                                records.setdefault(code, []).append(
                                    (pub, stat, vals))
            if (i + 1) % 50 == 0:
                el = time.perf_counter() - t0
                print(f"  {i+1}/{len(codes)}  查询 {n_q} 次  用时 {el:.0f}s  "
                      f"有数据股票 {len(records)}", flush=True)
    finally:
        bs.logout()

    # 去重：同一 (code, pubDate) 保留最后一条（后公布的可能覆盖）
    out: Dict[str, List[list]] = {}
    for code, recs in records.items():
        by_pub: Dict[str, dict] = {}
        stat_by_pub: Dict[str, str] = {}
        for pub, stat, vals in recs:
            by_pub[pub] = vals
            stat_by_pub[pub] = stat
        out[code] = [[pub, stat_by_pub[pub], vals] for pub, vals in
                      sorted(by_pub.items())]

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, args.out)
    with open(path, "w") as fh:
        json.dump(out, fh, ensure_ascii=False)
    n_fac = len({f for recs in out.values() for _, _, v in recs for f in v})
    print(f"[fund] 完成：{len(out)} 只股票有财务数据，{n_fac} 个因子，"
          f"写入 {path}", flush=True)
    print(f"[fund] 用时 {time.perf_counter()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
