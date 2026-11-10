#!/usr/bin/env python3
"""P12 前置：把 gold / bond 类 ETF 的历史往前拉到 2013（commodity 拉不动，见下）。

## 为什么需要

P6 的预注册判据 (f)「有独立样本外证据」一直结构性不满足，根因是：
- A 股腿长历史此前没有（P10 已解决）
- **ETF 袖套的数据只回溯到 2019-01**（本地 parquet 起点）→ 组合层仍缺长历史

本脚本探测并实测：腾讯源对 gold / bond 类 ETF **可以**往前拉到 2013~2017，
只有 commodity 三只（159980/159985/159981）物理上不存在（2019 底才上市的期货 ETF）。

→ 因此 P12 的长历史验证必须**分两层**：
1. **三袖套 + A 股腿**（定稿配置）：只能到 2019-01 之后
2. **gold+bond 两类袖套 + A 股腿**（删掉 commodity 的降级配置）：可到 2013-03
   → 这是**长历史唯一可能的组合配置**

## 拉取策略

- 腾讯源是「结束端条数上限」语义 → 必须分段倒推（复用 etf_data_pipeline 的实现）
- 落到 `data/etf/parquet_long/`，**不覆盖**现有 `data/etf/parquet/`（定稿口径必须原样保留）
- 每只拉完后做**分段拼接复权一致性校验**：与现有 parquet 的重叠区间逐日比对 close
  （容差 0.5%）→ 抓伪跳变

## 用法
    cd backend && env -u PYTHONPATH .venv-py313/bin/python3 \\
        scripts/fetch_etf_long_history.py --codes 518880,511010,511260,159934,159937,511020
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime
from typing import Any, Dict, List, Optional

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT = os.path.abspath(os.path.join(BACKEND_DIR, ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)
sys.path.insert(0, os.path.join(BACKEND_DIR, "scripts"))

import etf_data_pipeline as E  # noqa: E402

OUT_DIR = os.path.join(ROOT, "data", "etf", "parquet_long")
REPORT = os.path.join(ROOT, "data", "etf", "long_history_fetch_report.json")
OVERLAP_TOL = 0.005  # 0.5% 相对容差


def _load(path: str):
    import pandas as pd

    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    col = "date" if "date" in df.columns else df.columns[0]
    s = pd.Series(df["close"].to_numpy(), index=pd.DatetimeIndex(df[col]))
    return s[~s.index.duplicated(keep="last")].sort_index()


def cross_check(code: str, new: Any) -> Dict[str, Any]:
    """与现有 parquet 的重叠区间逐日比对 close（抓分段拼接的伪跳变）。"""
    old = _load(os.path.join(E.PARQUET_DIR, f"{code}.parquet"))
    if old is None or new is None:
        return {"checked": False, "reason": "无现有 parquet"}
    common = old.index.intersection(new.index)
    if len(common) == 0:
        return {"checked": False, "reason": "无重叠区间"}
    a = old.loc[common].astype(float)
    b = new.loc[common].astype(float)
    rel = (b - a).abs() / a.abs().replace(0.0, float("nan"))
    worst_idx = rel.idxmax()
    worst = float(rel.max())
    return {
        "checked": True,
        "n_overlap": int(len(common)),
        "overlap_start": str(common[0])[:10],
        "overlap_end": str(common[-1])[:10],
        "max_rel_diff": round(worst, 8),
        "worst_date": str(worst_idx)[:10],
        "ok": bool(worst <= OVERLAP_TOL),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", required=True)
    ap.add_argument("--start", default="2012-01-01")
    ap.add_argument("--end", default=date.today().isoformat())
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    codes = [c.strip() for c in args.codes.split(",") if c.strip()]
    report: Dict[str, Any] = {"codes": {}, "out_dir": OUT_DIR}
    for i, code in enumerate(codes, 1):
        print(f"[p12fetch] [{i}/{len(codes)}] {code} ...", flush=True)
        recs = E._fetch_tencent_full(code, datetime.strptime(args.start, "%Y-%m-%d").date(),
                                     datetime.strptime(args.end, "%Y-%m-%d").date(),
                                     max_segments=24)
        if not recs:
            print(f"[p12fetch] {code} ❌ 无数据", flush=True)
            report["codes"][code] = {"ok": False, "reason": "无数据"}
            continue
        import pandas as pd

        df = pd.DataFrame(recs)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last")
        path = os.path.join(OUT_DIR, f"{code}.parquet")
        df.to_parquet(path, index=False)
        new = pd.Series(df["close"].to_numpy(), index=pd.DatetimeIndex(df["date"]))
        cc = cross_check(code, new)
        # 跳变扫描：|日收益| > 11% 的行（金价/商品期货 ETF 涨跌停板 ±10%，11% 留余量）
        r = new.pct_change().abs()
        jumps = int((r > 0.11).sum())
        report["codes"][code] = {
            "ok": True,
            "n_rows": int(len(df)),
            "first_date": str(df["date"].iloc[0])[:10],
            "last_date": str(df["date"].iloc[-1])[:10],
            "cross_check": cc,
            "n_jumps_gt_11pct": jumps,
        }
        print(f"[p12fetch] {code} ✅ {len(df)} 行 {report['codes'][code]['first_date']}~"
              f"{report['codes'][code]['last_date']} | 拼接校验 {cc.get('max_rel_diff')} "
              f"ok={cc.get('ok')} | >11% 跳变 {jumps}", flush=True)
        with open(REPORT, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)

    bad = [c for c, v in report["codes"].items() if not v.get("ok")
           or (v.get("cross_check", {}).get("checked") and not v["cross_check"].get("ok"))]
    report["failed"] = bad
    with open(REPORT, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print(f"[p12fetch] 写出 {REPORT}；失败/校验不通过: {bad}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
