"""市场附加数据加载（2026-10-25）

`stock_data` 存的是**未复权真实价**（复权因子实测恒为 1.0），
除权日价格会跳空（实测 000001.SZ 有 6 天 |涨跌幅| > 30%），
直接用 `close/close.shift(1)-1` 判定涨跌停会把除权误判成跌停。

baostock 提供了交易所口径的字段，补齐可成交性判断所需的一切：
  pct_chg      涨跌幅（已除权调整）→ 涨跌停判定
  amount       成交额 → 流动性上限
  trade_status 交易状态 → 停牌
  is_st        ST 标记 → 5% 涨跌停
  turn         换手率
  pe_ttm/pb_mrq/ps_ttm  估值

本模块把这些字段按 master 对齐成矩阵，供纸面盘 / 回测复用。
数据缺失的标的用 NaN（调用方需保守处理）。
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = "/Users/rone/Projects/stock-platform"
BAOSTOCK_DIR = os.path.join(ROOT, "data", "parquet", "baostock_daily")

#: 需要加载的字段
EXTRA_FIELDS = ("pct_chg", "amount", "trade_status", "is_st", "turn",
                 "pe_ttm", "pb_mrq", "ps_ttm")


def baostock_available(codes: List[str]) -> bool:
    """是否已有足够的 baostock 数据可用。"""
    if not os.path.isdir(BAOSTOCK_DIR):
        return False
    have = sum(
        1 for c in codes
        if os.path.exists(
            os.path.join(BAOSTOCK_DIR, f"{c.replace('.', '_')}.parquet")
        )
    )
    return have >= max(1, int(len(codes) * 0.8))


def load_market_extras(
    codes: List[str],
    master: pd.DatetimeIndex,
    verbose: bool = True,
) -> Dict[str, pd.DataFrame]:
    """加载附加字段并对齐到 master。

    Args:
        codes: 形如 ``000001.SZ``
        master: 目标日期索引

    Returns:
        {字段名: DataFrame(索引=master, 列=codes)}
        没有数据的字段返回全 NaN 的空 DataFrame（不报错）。
    """
    t0 = time.time()
    n_dates, n_codes = len(master), len(codes)
    out: Dict[str, np.ndarray] = {
        f: np.full((n_dates, n_codes), np.nan, dtype=np.float64)
        for f in EXTRA_FIELDS
    }
    pos_of_date = {ts: i for i, ts in enumerate(master)}
    loaded = 0
    missing = 0

    for j, code in enumerate(codes):
        path = os.path.join(BAOSTOCK_DIR, f"{code.replace('.', '_')}.parquet")
        if not os.path.exists(path):
            missing += 1
            continue
        try:
            df = pd.read_parquet(
                path, columns=["date", *EXTRA_FIELDS]
            )
        except Exception:
            missing += 1
            continue
        if df is None or len(df) == 0:
            missing += 1
            continue
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        # 只保留 master 里的日期
        df = df[df["date"].isin(set(master))]
        if len(df) == 0:
            missing += 1
            continue
        idx = df["date"].map(pos_of_date)
        keep = idx.notna()
        if not keep.any():
            missing += 1
            continue
        idx = idx[keep].astype(int).to_numpy()
        for f in EXTRA_FIELDS:
            vals = pd.to_numeric(df.loc[keep, f], errors="coerce").to_numpy(
                dtype=np.float64
            )
            out[f][idx, j] = vals
        loaded += 1

    if verbose:
        print(
            f"[extras] baostock 附加字段: {loaded}/{n_codes} 只，缺失 {missing}，"
            f"{time.time() - t0:.0f}s",
            flush=True,
        )

    return {
        f: pd.DataFrame(out[f], index=pd.DatetimeIndex(master), columns=codes)
        for f in EXTRA_FIELDS
    }


def get_extra(
    extras: Optional[Dict[str, pd.DataFrame]],
    field: str,
    code: str,
    date,
) -> Optional[float]:
    """从 extras 里取某标的某日的字段值。"""
    if not extras:
        return None
    df = extras.get(field)
    if df is None or code not in df.columns:
        return None
    try:
        v = df.at[date, code]
    except (KeyError, TypeError):
        return None
    if v is None:
        return None
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return None
    if np.isnan(fv):
        return None
    return fv
