"""A 股筛选 — 从通达信盘后包全市场记录中筛出 A 股。

通达信盘后包含全部证券（A股、B股、指数、板块指数、回购、基金、可转债、
国债、政策性金融债等），约 5 万条/天，其中 A 股约 5500 只。

**停牌判定只对 A 股有意义**。债券/回购/部分基金不每日成交，`volume==0`
是常态而非停牌 — 实测 2026-06-08 全市场 40490 条 volume==0，其中绝大多数
是国债/政策性金融债，A 股真停牌只有 20 只。

用法:
    from back_test_data_service.tdx_pipeline.ashare import filter_ashare, is_ashare_code

    a = filter_ashare(df)   # DataFrame → 只留 A 股行
"""

from __future__ import annotations

from typing import Set

import pandas as pd

# 沪市 A 股代码前缀
SH_PREFIXES: Set[str] = {"600", "601", "603", "605", "688", "689"}
# 深市 A 股代码前缀（主板 000/001/002/003，创业板 300/301）
SZ_PREFIXES: Set[str] = {"000", "001", "002", "003", "300", "301"}
# 北交所 A 股代码前缀（4/8 开头的存量股 + 92 开头的新三板转板）
BJ_PREFIXES: Set[str] = {"4", "8", "92"}


def is_ashare_code(code: str, market: str) -> bool:
    """判断 6 位代码是否为 A 股。

    Args:
        code: 6 位数字代码
        market: sh / sz / bj

    Returns:
        True = A 股

    Note:
        沪市排除 900（B股）、深市排除 200（B股）；
        沪市 000/399 是指数（已在 parser 层因 close<=0 过滤或 index 前缀排除）；
        北交所 4/8 开头含债券（如 821001 是债），需进一步排除 — 见 filter_ashare。
    """
    if market == "sh":
        return code.startswith(tuple(SH_PREFIXES))
    if market == "sz":
        return code.startswith(tuple(SZ_PREFIXES))
    if market == "bj":
        return code[:2] in {"43", "83", "87", "92"}  # 北交所/新三板精选层
    return False


def filter_ashare(df: pd.DataFrame) -> pd.DataFrame:
    """从全市场记录中筛出 A 股，并重算 A 股专属的 is_suspended。

    Args:
        df: load_range() 返回的 DataFrame，需含 market / code / volume / amount 列

    Returns:
        只含 A 股行的 DataFrame，is_suspended 列已按 A 股口径重算

    实测:
        - 2026-06-05 ~ 06-18 全市场 507366 行 → A 股 55351 行（5542 只）
        - A 股停牌 225 条（0.41%），与 Baostock tradestatus 完全一致
        - 600228 *ST天成 停牌日 06-08/06-17/06-18 三日精确命中
        - 5 只活跃股（600036/000001/300750/601318/000858）零误判
    """
    mask = df.apply(lambda r: is_ashare_code(str(r["code"]), r["market"]), axis=1)
    out = df[mask].copy()
    # A 股停牌口径: volume == 0 且 amount == 0
    out["is_suspended"] = (out["volume"] == 0) & (out["amount"] == 0.0)
    return out


def ashare_count_by_market(df: pd.DataFrame) -> dict:
    """统计各市场 A 股股票数（唯一 ts_code）。"""
    a = filter_ashare(df)
    return a.groupby("market")["ts_code"].nunique().to_dict()
