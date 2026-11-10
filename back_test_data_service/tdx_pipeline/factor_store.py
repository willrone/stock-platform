"""除权除息事件存储 — 基于 mootdx xdxr（加法模型）。

存储: data/tdx_daily/adjust_events/events.parquet
长表: ts_code, ex_date, cash_per_share, rights_per_share, peigu_price, peigu_per_share

**注意**: 早期版本用新浪 hfq.js 乘法因子，已废弃（sina_factors_DEPRECATED.py）。
实测乘法模型在纯现金分红股上引入 0.2~0.3% 误差，因为 A 股标准
（通达信/同花顺/腾讯 qfq）用的是**加法（线性）模型**。

权威性: mootdx xdxr 与东方财富分红数据完全一致（300750 三次除权逐条对拍）。
腾讯 qfq 自身有 0.017% 级别偏差，不作为验证基准。

用法:
    from back_test_data_service.tdx_pipeline.factor_store import refresh_events, get_events

    refresh_events()            # 批量拉取
    evs = get_events("000001.SZ")
"""

from __future__ import annotations

import os
import time
from typing import List, Optional

import pandas as pd

from .adjust import AdjustmentEvent, fetch_events
from .pipeline import FACTOR_DIR, RAW_DIR

EVENT_PATH = os.path.join(FACTOR_DIR, "events.parquet")

COLUMNS = ["ts_code", "ex_date", "cash_per_share", "rights_per_share",
           "peigu_price", "peigu_per_share"]


def load_event_table() -> pd.DataFrame:
    """加载事件长表，不存在返回空表。"""
    if not os.path.exists(EVENT_PATH):
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_parquet(EVENT_PATH)


def save_event_table(df: pd.DataFrame) -> None:
    """保存事件长表（原子写）。"""
    os.makedirs(FACTOR_DIR, exist_ok=True)
    tmp = EVENT_PATH + ".tmp"
    df.to_parquet(tmp, index=False, compression="snappy")
    os.replace(tmp, EVENT_PATH)


def all_ts_codes() -> List[str]:
    """从已拉取的日线分区中收集全部 ts_code。"""
    if not os.path.isdir(RAW_DIR):
        return []
    codes = set()
    for name in os.listdir(RAW_DIR):
        if not name.startswith("date="):
            continue
        part = os.path.join(RAW_DIR, name, "part.parquet")
        if os.path.exists(part):
            df = pd.read_parquet(part, columns=["ts_code"])
            codes.update(df["ts_code"].unique().tolist())
    return sorted(codes)


def refresh_events(
    codes: Optional[List[str]] = None,
    sleep_between: float = 0.02,
    verbose: bool = True,
) -> dict:
    """批量拉取除权除息事件并合并入事件表。

    只拉表中没有的股票（已发生的除权事件不会变，无需重拉）。

    Args:
        codes: ts_code 列表，缺省从已拉取日线分区收集
        sleep_between: 每只间隔秒数（xdxr 0.02s/只，无需过度限速）
        verbose: 打印进度

    Returns:
        {"fetched": N, "failed": N, "no_event": N, "elapsed_s": float}
    """
    if codes is None:
        codes = all_ts_codes()

    existing = load_event_table()
    existing_codes: set = set()
    if len(existing) > 0:
        existing_codes = set(existing["ts_code"].unique())

    rows: List[dict] = []
    stats: dict = {"fetched": 0, "failed": 0, "no_event": 0}
    t0 = time.time()

    for i, code in enumerate(codes):
        if code in existing_codes:
            continue  # 已发生事件不会变，无需重拉
        try:
            events = fetch_events(code)
            for e in events:
                rows.append({
                    "ts_code": code,
                    "ex_date": e.date,
                    "cash_per_share": e.cash_per_share,
                    "rights_per_share": e.rights_per_share,
                    "peigu_price": e.peigu_price,
                    "peigu_per_share": e.peigu_per_share,
                })
            if not events:
                stats["no_event"] += 1
            stats["fetched"] += 1
        except Exception as e:
            stats["failed"] += 1
            if verbose and stats["failed"] <= 10:
                print(f"  事件拉取失败 {code}: {type(e).__name__}: {e}")

        if verbose and (i + 1) % 500 == 0:
            print(f"  事件进度 {i+1}/{len(codes)}  新增 {stats['fetched']}  "
                  f"失败 {stats['failed']}  耗时 {time.time()-t0:.0f}s")
        if sleep_between > 0:
            time.sleep(sleep_between)

    if rows:
        new_df = pd.DataFrame(rows)
        merged = (pd.concat([existing, new_df], ignore_index=True)
                  if len(existing) else new_df)
        merged = merged.sort_values(["ts_code", "ex_date"]).reset_index(drop=True)
        save_event_table(merged)

    stats["elapsed_s"] = round(time.time() - t0, 1)
    if verbose:
        print(f"\n事件拉取完成: 新增 {stats['fetched']} 只"
              f"（其中 {stats['no_event']} 只无除权）, 失败 {stats['failed']} 只, "
              f"耗时 {stats['elapsed_s']}s")
    return stats


def get_events(ts_code: str,
               event_table: Optional[pd.DataFrame] = None) -> List[AdjustmentEvent]:
    """从事件表取单只股票的除权事件（升序）。"""
    if event_table is None:
        event_table = load_event_table()
    tbl: pd.DataFrame = event_table
    sub = tbl[tbl["ts_code"] == ts_code].sort_values("ex_date")
    return [
        AdjustmentEvent(
            date=r["ex_date"],
            cash_per_share=float(r["cash_per_share"]),
            rights_per_share=float(r["rights_per_share"]),
            peigu_price=float(r["peigu_price"]),
            peigu_per_share=float(r["peigu_per_share"]),
        )
        for _, r in sub.iterrows()
    ]
