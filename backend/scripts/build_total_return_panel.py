#!/usr/bin/env python3
"""自建总回报层（§12.84 唯一可信口径）：未复权价 × 现金分红再投 × 送转扩股 = 总回报指数。

语义（写死）：持有人 **不参与配股**（配股的除权跌幅按真实损失计入），**现金分红按除息日收盘再投**，
送转/转增按扩股计入。这三条是唯一能用「原始价 + 分红送配明细」独立复算的口径，
也是散户真实行为；现成的复权序列（东财 hfq / 腾讯 hfq / akshare 涨跌幅 / baostock pctChg）
实测都不符合这个口径（偏差 −740pp ~ +739pp，见台账 §12.84 口径审计）。

指数定义：TR[i] = shares[i] × close[i]，shares 从 1 股起、只在除息日变化：
    shares[i] = shares[i-1] × (1 + song_ratio + dps / close[i])
纯乘性递推 ⇒ TR[b]/TR[a] 与起点无关（可任意入场日切片）。

产出：data/parquet/tr_panel/<num>_<SH|SZ>.parquet
列：date, close(未复权), tr_index, tr_ret, amount, pe_ttm(2021 起), pb_mrq(2021 起), has_dividend_event
闸门（跑完自动检查，不合格就红）：
1. tr_ret 只在除息/除权日不为 0（其他日子必须等于未复权涨跌幅 ±1e-6）；
2. tr_index 与未复权 close 的比值必须单调不减（复权因子不许倒退）；
3. 与东财 hfq 累计对比：|差|>15pp 的只数占比必须披露；
4. 退市股必须仍在面板里（不许被静默剔除）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PRICE_DIR = os.path.join(ROOT, "data", "parquet", "akshare_full")
EVENT_DIR = os.path.join(ROOT, "data", "parquet", "dividend_events")
VAL_DIR = os.path.join(ROOT, "data", "parquet", "baostock_daily")          # 2021 起有 pe/pb
OUT_DIR = os.path.join(ROOT, "data", "parquet", "tr_panel")
REGISTRY = os.path.join(ROOT, "data", "parquet", "baostock_full", "_registry.parquet")
AUDIT_OUT = os.path.join(ROOT, "data", "stocks", "tr_panel_audit.json")
COLS = ["date", "close", "tr_index", "tr_ret", "amount", "pe_ttm", "pb_mrq", "has_dividend_event"]


def build_one(path: str, events_path: str | None, val_path: str | None) -> dict:
    stem = os.path.basename(path)[: -len(".parquet")]
    num, mkt = stem.rsplit("_", 1)
    d = pd.read_parquet(path, columns=["date", "close", "amount", "pct_chg_raw", "close_hfq"])
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date").reset_index(drop=True)
    close = pd.to_numeric(d["close"], errors="coerce").astype(float)
    if (close <= 0).any():
        return {"code": stem, "skipped": "非正收盘价"}

    ex_map: dict[pd.Timestamp, tuple[float, float]] = {}
    if events_path:
        ev = pd.read_parquet(events_path)
        for _, r in ev.iterrows():
            dps = 0.0 if (r["dps"] is None or pd.isna(r["dps"])) else float(r["dps"])
            song = 0.0 if (r["song_ratio"] is None or pd.isna(r["song_ratio"])) else float(r["song_ratio"])
            if dps == 0.0 and song == 0.0:
                continue
            ex_map[pd.Timestamp(r["ex_date"])] = (dps, song)

    mult = np.ones(len(d))
    has_ev = np.zeros(len(d), dtype=bool)
    dates = d["date"].to_numpy()
    for i, dt in enumerate(dates):
        key = pd.Timestamp(dt)
        if key in ex_map:
            dps, song = ex_map[key]
            c = close.iloc[i]
            if not np.isfinite(c) or c <= 0:
                return {"code": stem, "skipped": "除息日无有效收盘价"}
            mult[i] = 1.0 + song + dps / c
            has_ev[i] = True
    tr_index = close.to_numpy() * np.cumprod(mult)
    tr_ret = np.concatenate([[0.0], tr_index[1:] / tr_index[:-1] - 1.0])

    out = pd.DataFrame({"date": d["date"], "close": close.to_numpy(),
                        "tr_index": tr_index, "tr_ret": tr_ret,
                        "amount": pd.to_numeric(d["amount"], errors="coerce").to_numpy(),
                        "has_dividend_event": has_ev})

    # 闸门 1（真闸门）：没有事件的日子，未复权涨跌幅必须等于东财涨跌幅。
    # 不相等 = 东财那天用了除权基准（分红/配股/送转），而我们的事件表里没有这条事件 → 漏计，必须曝光。
    raw_ret = np.concatenate([[0.0], close.to_numpy()[1:] / close.to_numpy()[:-1] - 1.0])
    ak_pct = pd.to_numeric(d["pct_chg_raw"], errors="coerce").to_numpy() / 100.0
    plain = (~has_ev) & np.isfinite(ak_pct) & np.isfinite(raw_ret)
    plain[0] = False        # 首行不算：raw_ret[0] 恒为 0，而东财首行涨跌幅是相对发行价（301018 首行 +241.62%）。
                            # 不去掉就是每只白送 1 个「疑似漏计」，闸门退化成统计噪声
    bad_days = int(np.sum(np.abs(raw_ret[plain] - ak_pct[plain]) > 1e-4))
    plain_n = int(plain.sum())
    # 闸门 2：复权因子 tr_index/close 必须单调不减
    factor = tr_index / close.to_numpy()
    factor_back = int((np.diff(factor) < -1e-9).sum())

    # 与东财 hfq 的累计差（只作披露，不作准入）
    hfq = pd.to_numeric(d["close_hfq"], errors="coerce").to_numpy()
    gap_pp = None
    if len(hfq) > 200 and np.isfinite(hfq).all() and hfq[0] > 0 and (hfq > 0).all():
        gap_pp = round(100.0 * ((tr_index[-1] / tr_index[0]) - (hfq[-1] / hfq[0])), 2)

    # 估值列（2021 起才有）：pe/pb 从现役 baostock_daily 侧并入
    if val_path:
        v = pd.read_parquet(val_path, columns=["date", "pe_ttm", "pb_mrq"])
        v["date"] = pd.to_datetime(v["date"])
        out = out.merge(v, on="date", how="left")
    else:
        out["pe_ttm"] = np.nan
        out["pb_mrq"] = np.nan

    os.makedirs(OUT_DIR, exist_ok=True)
    tmp = os.path.join(OUT_DIR, stem + ".parquet.tmp")
    out[COLS].to_parquet(tmp, index=False)
    os.replace(tmp, os.path.join(OUT_DIR, stem + ".parquet"))
    return {"code": stem, "rows": len(out), "events": int(has_ev.sum()),
            "suspect_missing_event_days": bad_days, "plain_days": plain_n,
            "factor_backward_days": factor_back, "hfq_gap_pp": gap_pp}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    prices = sorted(f for f in os.listdir(PRICE_DIR) if f.endswith(".parquet"))
    if args.limit:
        prices = prices[: args.limit]
    reg = pd.read_parquet(REGISTRY)
    reg["status"] = pd.to_numeric(reg["status"], errors="coerce")
    delisted = {c.split(".")[1] + "_" + c.split(".")[0].upper() for c in reg[reg["status"] != 1]["code"]}

    stats, skipped = [], 0
    t0 = __import__("time").time()
    for i, f in enumerate(prices, 1):
        stem = f[: -len(".parquet")]
        ev_path = os.path.join(EVENT_DIR, f)
        val_path = os.path.join(VAL_DIR, f)
        try:
            s = build_one(os.path.join(PRICE_DIR, f),
                          ev_path if os.path.exists(ev_path) else None,
                          val_path if os.path.exists(val_path) else None)
        except Exception as e:
            s = {"code": stem, "skipped": repr(e)[:90]}
        if s.get("skipped"):
            skipped += 1
        else:
            stats.append(s)
        if i % 500 == 0 or i == len(prices):
            print(f"[{i}/{len(prices)}] ok {len(stats)} 跳过 {skipped} 用时 {__import__('time').time()-t0:.0f}s", flush=True)

    df = pd.DataFrame([s for s in stats])
    gates = {
        "stocks_built": int(len(df)),
        "stocks_skipped": skipped,
        "delisted_in_panel": int(len(set(df["code"]) & delisted)),
        "delisted_total_registry": int(len(delisted)),
        "event_rows_total": int(df["events"].sum()),
        "stocks_with_events": int((df["events"] > 0).sum()),
        "suspect_missing_event_days_total": int(df["suspect_missing_event_days"].sum()),
        "plain_days_total": int(df["plain_days"].sum()),
        "suspect_missing_event_share": round(float(df["suspect_missing_event_days"].sum() / max(df["plain_days"].sum(), 1)), 5),
        "stocks_with_suspect_days": int((df["suspect_missing_event_days"] > 0).sum()),
        "factor_backward_days_total": int(df["factor_backward_days"].sum()),
        "hfq_gap_abs_median_pp": None, "hfq_gap_gt15pp_n": None, "hfq_gap_gt15pp_share": None,
    }
    g = pd.to_numeric(df["hfq_gap_pp"], errors="coerce").dropna()
    if len(g):
        gates["hfq_gap_abs_median_pp"] = round(float(np.median(np.abs(g))), 2)
        gates["hfq_gap_p90_abs_pp"] = round(float(np.quantile(np.abs(g), 0.9)), 2)
        gates["hfq_gap_gt15pp_n"] = int((np.abs(g) > 15).sum())
        gates["hfq_gap_gt15pp_share"] = round(float((np.abs(g) > 15).mean()), 4)
        gates["hfq_nonpositive_excluded_n"] = int(len(df) - len(g))
    gates["all_pass"] = bool(
        gates["stocks_built"] >= 5000
        and gates["delisted_in_panel"] >= 250
        and gates["suspect_missing_event_share"] <= 0.01
        and gates["factor_backward_days_total"] == 0
        and gates["stocks_with_events"] >= 2000
    )
    json.dump(gates, open(AUDIT_OUT, "w"), ensure_ascii=False, indent=2)
    print(json.dumps(gates, ensure_ascii=False, indent=2))
    print("闸门：", "全过" if gates["all_pass"] else "有失败")
    return 0 if gates["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
