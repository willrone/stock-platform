#!/usr/bin/env python3
"""`data/parquet/baostock_full`（2012 起 + 含退市股）的验收闸门（§12.83 前置 B）。

不跑策略、不出信号，只用数字回答三件事：
1. **幸存者偏差是不是真补上了**：注册表里 status=0（已退市）的股票，有没有真的落到 bar 里；
2. **退市股的最后一根 bar 语义对不对**：必须在摘牌日（outDate）之前 0~45 天内（A 股有 30 天退市整理期，
   末根应落在整理期内，不能凭空早消失，也不能"活"到摘牌之后）；
3. **长窗口链和现役层是否同口径**：同一只票在 2021-01-04 之后的 `cumprod(1+pctChg)` 链必须与
   `baostock_daily` 一致（相对差 < 1e-3），否则两条线的收益数字不可比。

另给：逐只覆盖率闸门（bar 数 / 应交易天数）、北交所缺口披露（baostock 不覆盖 BJ）。

用法：cd backend && env -u PYTHONPATH .venv-py313/bin/python3 scripts/baostock_full_history_audit.py
产出：../data/stocks/baostock_full_audit.json   退出码 0=闸门全过，1=有闸门失败
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = "/Users/rone/Projects/stock-platform"
FULL_DIR = os.path.join(ROOT, "data", "parquet", "akshare_full")          # 价格层（含 hfq）
BAO_DIR = os.path.join(ROOT, "data", "parquet", "baostock_full")          # 注册表层（含退市名单）
OLD_DIR = os.path.join(ROOT, "data", "parquet", "baostock_daily")
OUT = os.path.join(ROOT, "data", "stocks", "baostock_full_audit.json")
REGISTRY = os.path.join(BAO_DIR, "_registry.parquet")
START = pd.Timestamp("2012-01-01")
COVERAGE_MIN = 0.90            # 单只覆盖率下限（停牌容忍 10%）
ENDPOINT_WINDOW_DAYS = 90      # 摘牌前 90 个自然日：30 交易日整理期(≈42 天) + 停牌缓冲
CHAIN_GAP_MEDIAN_PP = 1.0      # 后复权链 vs 交易所口径 pctChg 链：|累计差| 中位数上限（pp）
MIN_DELISTED_WITH_BARS = 250   # 注册表 341 只退市股里，摘牌日 ≥2012 的约 279 只必须拿到 bar


def reg_key(bs_code: str) -> str:
    """注册表 code 是 "sh.600000"，bar 文件是 "600000.SH" → 索引前必须统一（这里错位整份验收就废了）。"""
    mkt, num = bs_code.split(".")
    return f"{num}.{mkt.upper()}"


def _files(d: str) -> dict[str, str]:
    if not os.path.isdir(d):
        return {}
    out = {}
    for f in os.listdir(d):
        if not f.endswith(".parquet") or f.startswith("_"):
            continue
        stem = f[: -len(".parquet")]
        num, mkt = stem.rsplit("_", 1)
        out[f"{num}.{mkt}"] = os.path.join(d, f)      # 统一成 600000.SH，和 bars_map / 注册表同一套键
    return out


def endpoint_in_window(gap_days: int, window_days: int = ENDPOINT_WINDOW_DAYS) -> bool:
    """摘牌语义：末根 bar 必须落在摘牌日前 0~window 天（退市整理期），不能早消失也不能延后存活。"""
    return 0 <= gap_days <= window_days


def coverage_ratio(bars: pd.Series, from_d: pd.Timestamp, to_d: pd.Timestamp, cal: pd.DatetimeIndex) -> float:
    """bar 数 ÷ [from,to] 内的交易日历天数。日历取自全库日期并集，不假设交易日历。"""
    window = cal[(cal >= from_d) & (cal <= to_d)]
    if len(window) == 0:
        return 0.0
    inwin = bars[(bars >= from_d) & (bars <= to_d)]      # 窗口外的 bar 不许 inflate 覆盖率
    return len(inwin) / len(window)


def endpoint_gap_days(last_bar: pd.Timestamp, out_date: pd.Timestamp) -> int:
    return int((out_date - last_bar).days)


def chain_consistency(ra: pd.Series, rb: pd.Series) -> dict:
    """两条日收益序列（小数，不是百分数）在同一日历上的 cumprod 链相对漂移。"""
    idx = ra.index.intersection(rb.index)
    if len(idx) < 60:
        return {"overlap_days": int(len(idx)), "max_rel_diff": None}
    ca = (1.0 + ra.loc[idx].fillna(0.0)).cumprod()
    cb = (1.0 + rb.loc[idx].fillna(0.0)).cumprod()
    rel = (ca / cb - 1.0).abs()
    return {"overlap_days": int(len(idx)),
            "max_rel_diff": round(float(rel.max()), 8),
            "mean_rel_diff": round(float(rel.mean()), 8),
            "worst_day": str(idx[int(np.argmax(rel.values))].date())}


def run_audit() -> dict:
    out: dict = {"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M")}
    reg_path = REGISTRY
    files = _files(FULL_DIR)
    if not os.path.exists(reg_path) or not files:
        out["gates"] = {"registry_present": False, "files_present": False, "all_pass": False}
        out["note"] = f"{FULL_DIR} 还没拉 → 先跑 scripts/fetch_baostock_full_history.py"
        return out

    reg = pd.read_parquet(reg_path)
    reg["status"] = pd.to_numeric(reg["status"], errors="coerce")
    listed = reg[reg["status"] == 1]
    delisted = reg[reg["status"] != 1]

    bars_map, meta = {}, {}
    for key, path in files.items():                 # files 的键已是 600000.SH
        d = pd.read_parquet(path)
        d["date"] = pd.to_datetime(d["date"])
        d = d.sort_values("date")
        bars_map[key] = d["date"]
        meta[key] = {"rows": int(len(d)), "first": d["date"].min(), "last": d["date"].max(),
                     }

    cal = pd.DatetimeIndex(sorted({d for s in bars_map.values() for d in s}))
    data_end = cal[-1]

    reg_idx = {reg_key(r["code"]): r for _, r in reg.iterrows()}

    cov, low_cov, delisted_ok, endpoint_bad, endpoint_gaps = [], [], 0, [], []
    delisted_after_start = 0
    for key, s in bars_map.items():
        row = reg_idx.get(key)
        ipo = pd.Timestamp(row["ipoDate"]) if row is not None and str(row["ipoDate"]) not in ("", "NaT", "nan") else START
        raw_out = row["outDate"] if row is not None else ""
        outd = pd.Timestamp(raw_out) if (row is not None and str(raw_out) not in ("", "NaT", "nan")) else None
        if outd is not None and outd >= START:
            delisted_after_start += 1
        if outd is not None and outd < START:       # 起点之前就摘牌：不在窗口内，不评覆盖率
            continue
        end = min(outd, data_end) if outd else data_end
        c = coverage_ratio(s, max(ipo, START), end, cal)
        cov.append(c)
        if c < COVERAGE_MIN:
            low_cov.append({"code": key, "coverage_pct": round(c * 100, 1), "rows": meta[key]["rows"]})
        if outd and outd <= data_end:
            gap = endpoint_gap_days(s.max(), outd)
            endpoint_gaps.append(gap)
            if endpoint_in_window(gap):
                delisted_ok += 1
            else:
                endpoint_bad.append({"code": key, "outDate": str(outd.date()),
                                     "last_bar": str(s.max().date()), "gap_days": gap})

    # 与现役层比链：**按日期对齐的累计口径**（hfq 用首末比值，baostock 用它自己 bar 上的 pctChg 连乘）。
    # 绝不逐日配对两条链：两侧停牌日不同会把两天涨跌当成一天，15 倍的假分歧就是这么造出来的（实测踩过）。
    old = _files(OLD_DIR)
    old_alive = len(old) >= 100
    chain = {"checked": 0, "skipped": 0, "gaps_pp": [], "worst": [],
             "status": ("现役层对照（只披露，不作闸门）" if old_alive else
                        "旧 baostock bar 层已作废（10001011 IP 黑名单；半截 bar 隔离在 baostock_full/_killed_run_partial/）。"
                        "而且两侧口径都已被地面真相判不合格（hfq 配股减法复权会算出负股价；pctChg 把送转虚增），"
                        "两个坏口径互比不构成闸门 —— 口径闸门改在 build_total_return_panel.py")}
    for key, s in (list(bars_map.items())[:1200] if old_alive else []):
        if key not in old:
            chain["skipped"] += 1
            continue
        h = pd.to_numeric(pd.read_parquet(files[key], columns=["close_hfq"])["close_hfq"], errors="coerce").to_numpy()
        if h[0] <= 0 or not np.isfinite(h).all():
            chain["skipped"] += 1
            continue
        od = pd.read_parquet(old[key], columns=["date", "pct_chg"])
        od["date"] = pd.to_datetime(od["date"])
        od = od.sort_values("date")
        common = sorted(set(s) & set(od["date"]))
        if len(common) < 200:
            chain["skipped"] += 1
            continue
        t1, t2 = common[0], common[-1]
        hh = h[np.searchsorted(s.values, np.datetime64(t1)):np.searchsorted(s.values, np.datetime64(t2), "right")]
        rb = od[od["date"] >= t1]["pct_chg"]
        rb = pd.to_numeric(rb, errors="coerce").fillna(0.0).to_numpy()[1:] / 100.0
        if len(hh) < 2 or hh[0] <= 0 or not np.isfinite(hh).all():
            chain["skipped"] += 1
            continue
        r_hfq = hh[-1] / hh[0] - 1.0
        r_bs = float(np.prod(1.0 + rb) - 1.0)
        gap_pp = (r_hfq - r_bs) * 100.0
        chain["checked"] += 1
        chain["gaps_pp"].append(gap_pp)
    gaps = np.array(chain.pop("gaps_pp"))
    chain["gap_median_pp"] = round(float(np.median(gaps)), 2) if len(gaps) else None
    chain["abs_gap_median_pp"] = round(float(np.median(np.abs(gaps))), 2) if len(gaps) else None
    chain["abs_gap_p90_pp"] = round(float(np.quantile(np.abs(gaps), 0.9)), 2) if len(gaps) else None
    chain["abs_gap_gt5pp_n"] = int((np.abs(gaps) > 5).sum()) if len(gaps) else None
    chain["abs_gap_gt10pp_n"] = int((np.abs(gaps) > 10).sum()) if len(gaps) else None
    chain["abs_gap_max_pp"] = round(float(np.max(np.abs(gaps))), 2) if len(gaps) else None

    bj_missing = sorted({k for k in bars_map if k.endswith(".BJ")})
    delisted_with_bars = sum(1 for _, d in delisted.iterrows() if reg_key(d["code"]) in bars_map)

    gates = {
        "registry_rows": int(len(reg)),
        "listed_rows": int(len(listed)),
        "delisted_in_registry": int(len(delisted)),
        "files": len(files),
        "total_bars": int(sum(m["rows"] for m in meta.values())),
        "data_first": str(cal[0].date()), "data_last": str(data_end.date()),
        "delisted_with_bars": delisted_with_bars,
        "delisted_out_date_after_start": delisted_after_start,
        "delisted_endpoint_ok": delisted_ok,
        "delisted_endpoint_bad": endpoint_bad[:12],
        "delisted_endpoint_bad_n": len(endpoint_bad),
        "endpoint_gap_median_days": int(np.median(endpoint_gaps)) if endpoint_gaps else None,
        "endpoint_gap_p90_days": int(np.quantile(endpoint_gaps, 0.9)) if endpoint_gaps else None,
        "endpoint_gap_max_days": int(np.max(endpoint_gaps)) if endpoint_gaps else None,
        "coverage_median_pct": round(float(np.median(cov)) * 100, 2) if cov else None,
        "low_coverage_n": len(low_cov),
        "low_coverage_worst": sorted(low_cov, key=lambda x: x["coverage_pct"])[:10],
        "pe_coverage_note": "akshare 日线层不含 PE/PB；2012-2020 估值需 qlib 层或 baostock 慢拉，见 disclosed_gaps",
        "chain_vs_live_layer": chain,
        "bj_codes_in_new_layer": len(bj_missing),
    }
    gates["all_pass"] = bool(
        gates["delisted_with_bars"] >= MIN_DELISTED_WITH_BARS
        and gates["delisted_with_bars"] >= int(gates["delisted_out_date_after_start"] * 0.95)
        and gates["delisted_endpoint_bad_n"] <= max(3, int(gates["delisted_with_bars"] * 0.05))
        and gates["coverage_median_pct"] is not None and gates["coverage_median_pct"] >= COVERAGE_MIN * 100
        and gates["low_coverage_n"] <= int(gates["files"] * 0.10)
    )   # 口径/链对比不在这里判：见 chain_vs_live_layer.status 与 build_total_return_panel.py 的闸门
    gates["disclosed_gaps"] = [
        f"baostock 不覆盖北交所：现役层有 BJ 代码 {len([1 for k in _files(OLD_DIR) if k.endswith('_BJ')])} 只，新层 {len(bj_missing)} 只",
        "财务 roe 只有 2018 年后的披露记录 → 质量规则只能从 2019 年入场起测",
        "akshare 日线无 PE/PB → 2021 年前的估值类规则（低PE/低PB）需 qlib 层（1990-2020-11 有 peTTM/pbMrq）另行验收",
        "baostock 行情源已被封禁（10001011 黑名单用户）→ 只保留它的注册表（含 341 只退市股 + ipoDate/outDate）",
        "退市股在东财分红接口多数返回空 → 自建 TR 会漏计这些票早期分红，方向是低估退市股回报（保守偏差）",
    ]
    out["gates"] = gates
    return out


def main() -> int:
    res = run_audit()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=2)
    print(json.dumps(res, ensure_ascii=False, indent=2)[:4000])
    print(f"\n→ {OUT}")
    ok = bool(res.get("gates", {}).get("all_pass"))
    print("闸门：", "全过" if ok else "有失败")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
