"""新浪复权因子 — 累计后复权因子表。

数据源: https://finance.sina.com.cn/realstock/company/{symbol}/hfq.js
- 免费、无 token、无注册
- 0.04s/只
- 返回累计后复权因子，历史除权日全列（降序）

正确复权公式（已实测 0.0000% 误差，含除权日之前/当天/之后三天）:
    前复权(d) = 不复权(d) × f(asof≤d 最近除权日) / f_latest

实测验证（sh601318，2026-09-10 除权，因子 3.295680）:
- 2026-09-09（除权日之前）: adj=0.98258，推算 55.2800 vs 腾讯 55.2800 ✓
- 2026-09-10（除权日当天）: adj=1.0，55.5200 vs 55.5200 ✓
- 2026-10-16（除权日之后）: adj=1.0，51.8800 vs 51.8800 ✓

关键设计: 存因子表**不存前复权价**。因子表历史条目不变（新除权只新增一条），
增量拉取不会让历史数据漂移 — 彻底避开前复权价随最新因子整体平移的坑。
"""

from __future__ import annotations

import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class AdjustFactor:
    """单次除权事件的累计后复权因子。"""
    date: str    # 除权日 YYYY-MM-DD
    factor: float  # 累计后复权因子


def _sina_symbol(ts_code: str) -> str:
    """600000.SH → sh600000"""
    code, suffix = ts_code.split(".")
    return f"{suffix.lower()}{code}"


def fetch_factors(ts_code: str, timeout: int = 25) -> List[AdjustFactor]:
    """拉取单只股票的累计后复权因子。

    Args:
        ts_code: Tushare 风格，如 600000.SH

    Returns:
        AdjustFactor 列表，按日期**升序**排列

    Raises:
        ConnectionError: 网络错误
        ValueError: 返回格式变化
    """
    symbol = _sina_symbol(ts_code)
    url = f"https://finance.sina.com.cn/realstock/company/{symbol}/hfq.js"
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://finance.sina.com.cn/",
    })
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode("utf-8", "replace")
    except urllib.error.URLError as e:
        raise ConnectionError(f"拉取新浪因子失败 {ts_code}: {e}") from e

    # 格式: var sh600000hfq={"total":29,"data":[{"d":"2026-07-16", "f":"17.39..."}, ...]}
    m = re.search(r'"total":\s*(\d+),\s*"data":\s*\[(.*?)\]', raw, re.S)
    if not m:
        raise ValueError(f"新浪因子返回格式变化 {ts_code}: {raw[:200]}")

    items = re.findall(r'\{"d":"([\d-]+)",\s*"f":"([\d.]+)"\}', m.group(2))
    factors = [AdjustFactor(date=d, factor=float(f)) for d, f in items]
    factors.sort(key=lambda x: x.date)  # 升序
    return factors


def factor_asof(factors: List[AdjustFactor], date: str) -> float:
    """取 ≤date 最近除权日的因子。

    若 date 早于所有除权日，返回最老的因子（通常是 1.0）。

    这是复权公式的关键 — 实测直接用 latest 会在除权日之前产生 1.77% 偏差。
    """
    best: Optional[AdjustFactor] = None
    for f in factors:
        if f.date <= date:
            best = f
        else:
            break  # 已升序，后面都 > date
    if best is None:
        return factors[0].factor if factors else 1.0
    return best.factor


def latest_factor(factors: List[AdjustFactor]) -> float:
    """取最新（最大日期）的因子，作为前复权基准。"""
    if not factors:
        return 1.0
    return factors[-1].factor


def to_qfq(raw_price: float, factors: List[AdjustFactor], date: str) -> float:
    """不复权价 → 前复权价。

    公式: qfq(d) = raw(d) × f(asof≤d) / f_latest
    """
    f_asof = factor_asof(factors, date)
    f_latest = latest_factor(factors)
    if f_latest == 0:
        return raw_price
    return raw_price * f_asof / f_latest


def to_hfq(raw_price: float, factors: List[AdjustFactor], date: str) -> float:
    """不复权价 → 后复权价。

    公式: hfq(d) = raw(d) / f(asof≤d)
    """
    f_asof = factor_asof(factors, date)
    if f_asof == 0:
        return raw_price
    return raw_price / f_asof


def build_factor_map(ts_code: str) -> Dict[str, float]:
    """拉取并转成 {除权日: 因子} 字典，便于查表。"""
    factors = fetch_factors(ts_code)
    return {f.date: f.factor for f in factors}


def factor_series(ts_code: str) -> Tuple[List[str], List[float]]:
    """拉取并转成 (日期列表, 因子列表)，均为升序，便于存表。"""
    factors = fetch_factors(ts_code)
    return [f.date for f in factors], [f.factor for f in factors]
