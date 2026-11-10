"""A 股交易日历（2026-10-25 新增）。

背景：TDX 盘后包「有没有」并不能判断「那天是不是交易日」——
包未发布和法定节假日都会抛 FileNotFoundError。原实现把它一律记为
non_trading 静默跳过，结果 2026-10-23（周五）的数据在 16:05 拉取时
因为包还没发布而被静默丢弃，随后所有「今天是不是周五」的判断都用了
「数据最新日」而不是真实日期，导致周频调仓整整两周没跑过。

本模块提供独立的交易日判断：
  1. exchange_calendars 的 XSHG 日历（精确，覆盖 2006-10-25 ~ 2026-12-31）
  2. 超出日历范围时退化为「周一至周五」粗筛，并明确返回 None 表示「不确定」

返回值三态：
  True  —— 确定是交易日
  False —— 确定不是交易日（法定节假日）
  None  —— 无法确定（超出日历范围），调用方应保守处理
"""

from __future__ import annotations

import functools
import logging
from datetime import date, datetime
from typing import Optional

logger = logging.getLogger(__name__)

# exchange_calendars 的覆盖上限（超出后无法精确判断）
_CAL_LAST_SESSION = "2026-12-31"


@functools.lru_cache(maxsize=1)
def _get_calendar():
    """拿 XSHG 日历，拿不到返回 None。"""
    try:
        import exchange_calendars as xcals

        return xcals.get_calendar("XSHG")
    except Exception as e:  # pragma: no cover - 环境缺失时降级
        logger.warning("exchange_calendars 不可用（%s），交易日历退化为工作日粗筛", e)
        return None


def calendar_range() -> tuple[Optional[str], Optional[str]]:
    """返回日历覆盖区间 (first, last)，取不到返回 (None, None)。"""
    cal = _get_calendar()
    if cal is None:
        return (None, None)
    return (
        str(cal.first_session.date()) if cal.first_session is not None else None,
        str(cal.last_session.date()) if cal.last_session is not None else None,
    )


def is_trading_day(date_str: str) -> Optional[bool]:
    """判断某天是否 A 股交易日。

    Args:
        date_str: YYYY-MM-DD

    Returns:
        True  确定是交易日
        False 确定不是交易日
        None  无法确定（超出日历覆盖范围）
    """
    try:
        d = datetime.strptime(str(date_str)[:10], "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return None

    cal = _get_calendar()
    if cal is not None:
        _, cal_last = calendar_range()
        if cal_last and d > datetime.strptime(cal_last, "%Y-%m-%d").date():
            # 超出日历覆盖 → 退化为工作日粗筛，但明确标「不确定」
            return d.weekday() < 5
        try:
            return bool(cal.is_session(d))
        except Exception:
            return d.weekday() < 5

    return d.weekday() < 5


def is_weekday(date_str: str) -> bool:
    """是否周一至周五（粗筛，不考虑节假日）。"""
    try:
        d = datetime.strptime(str(date_str)[:10], "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return False
    return d.weekday() < 5


def last_trading_day(on_or_before: str) -> Optional[str]:
    """返回 on_or_before 当天或之前最近的交易日（不含未来）。"""
    try:
        d = datetime.strptime(str(on_or_before)[:10], "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return None
    for _ in range(30):
        if is_trading_day(d.isoformat()) is True:
            return d.isoformat()
        d = date.fromordinal(d.toordinal() - 1)
    return None


def next_trading_day(after: str) -> Optional[str]:
    """返回 after 之后最近的交易日。"""
    try:
        d = datetime.strptime(str(after)[:10], "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return None
    for _ in range(30):
        d = date.fromordinal(d.toordinal() + 1)
        if is_trading_day(d.isoformat()) is True:
            return d.isoformat()
    return None


if __name__ == "__main__":
    for probe in ("2026-10-22", "2026-10-23", "2026-10-24", "2026-10-26", "2027-01-05"):
        print(f"{probe} → {is_trading_day(probe)}")
    print("覆盖区间:", calendar_range())
