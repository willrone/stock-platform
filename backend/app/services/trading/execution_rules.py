"""实盘可成交性规则引擎（2026-10-25）

## 为什么单独抽出来
「这个信号能不能真的买到 / 卖掉」是决定**信号有没有实盘价值**的第一道关。
之前这条逻辑散落在 paper_portfolio.py 的买卖循环里，只覆盖了
停牌 / 涨停 / 跌停三项，而且：
  - 没区分**开盘一字板**（全天封死）与盘中开板（能买但要排队）
  - 没有**流动性上限**：信号让你买 1.5 万的票，如果当天总成交额才 3000，
    根本买不到 1.5 万 —— 纸面盘却记成了成交
  - 没有**成交额门槛**：日成交额几百万的票，冲击成本会吃掉全部超额收益
  - 没有用**交易所口径涨跌幅**：除权日价格跳空会被误判成跌停

现在抽成单一规则引擎，纸面盘 / 回测 / 对外信号三处共用同一份实现，
避免「回测说能买、实盘买不到」的口径分裂。

## 规则清单
  1. 停牌        trade_status == 0 或 volume == 0      → 买卖都不可
  2. ST          is_st == 1                             → 涨跌停按 5% 判定（掩码不同）
  3. 退市整理期  名称以「退」结尾                         → 不可交易
  4. 涨停        pct_chg >= 涨停幅度                    → 买不进
  5. 一字涨停    open 已是涨停价且 high == low          → 全天封死，买不进
  6. 跌停        pct_chg <= -涨停幅度                   → 卖不出
  7. 流动性      买入额 <= 当日成交额 × 参与率            → 超出部分买不到
  8. 成交额门槛  当日成交额 < 最低门槛                   → 流动性太差，剔除
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

# ── 默认参数 ──────────────────────────────────────────────────────────
#: 单日买入金额占该股当日成交额的最大比例（0.5%）
#: 超过这个比例就买不进 —— 实盘冲击成本会非常高
DEFAULT_PARTICIPATION_RATE = 0.005

#: 当日成交额门槛（元）。低于此值视为流动性太差，不进信号
DEFAULT_MIN_TURNOVER = 5_000_000.0     # 500 万

#: ST 股票涨跌停幅度
ST_LIMIT_PCT = 0.05


def limit_pct(code: str, is_st: bool = False) -> float:
    """该股的涨跌停幅度。

    主板 10%，创业板/科创板 20%，北交所 30%，ST 5%。
    """
    if is_st:
        return ST_LIMIT_PCT
    c2 = code[:2]
    if c2 in ("30", "68"):
        return 0.20
    if c2 in ("92", "83", "87", "43", "82", "87"):
        return 0.30
    if c2 in ("60", "00"):
        return 0.10
    return 0.10


@dataclass
class Tradability:
    """某个标的在某个交易日的可成交性判定结果。"""

    can_buy: bool
    can_sell: bool
    reason: str = ""
    #: 允许的最大买入金额（流动性上限）。None = 不限
    max_buy_amount: Optional[float] = None
    #: 是否是「信号上该买但实际买不到」的情况（值得向用户解释）
    blocked_reason: str = ""

    @property
    def tradable(self) -> bool:
        return self.can_buy or self.can_sell


def check_tradability(
    code: str,
    *,
    name: str = "",
    close: Optional[float] = None,
    open_price: Optional[float] = None,
    high: Optional[float] = None,
    low: Optional[float] = None,
    pct_chg: Optional[float] = None,
    volume: Optional[float] = None,
    amount: Optional[float] = None,
    trade_status: Optional[int] = None,
    is_st: Optional[int] = None,
    participation_rate: float = DEFAULT_PARTICIPATION_RATE,
    min_turnover: float = DEFAULT_MIN_TURNOVER,
    pct_reliable: bool = True,
) -> Tradability:
    """判定一个标的在某日的可成交性。

    Args:
        code: 6 位代码
        name: 股票名称（用于退市整理期识别）
        close: 收盘价（未复权真实价）
        open_price: 开盘价
        high / low: 当日最高 / 最低
        pct_chg: **交易所口径**涨跌幅（小数，如 0.095 表示 +9.5%）。
                 必须是除权调整过的 —— 未复权价在除权日会跳空 ±30%，
                 会把除权误判成涨跌停。
        volume: 成交量（股或手）
        amount: 成交额（元）
        trade_status: 1=正常交易，0=停牌
        is_st: 1=ST
        pct_reliable: 涨跌幅是否可靠。False = 由未复权 close/prev_close 推导，
            可能含**除权跳空**（实测可达 ±50%），此时涨跌停判定要用宽松阈值，
            否则会把送转除权误判成涨跌停。
    """
    is_st_flag = bool(is_st) if is_st is not None else (
        "ST" in (name or "").upper()
    )
    lp = limit_pct(code, is_st_flag)

    # ── 1. 退市整理期：名称以「退」结尾 ──
    nm = (name or "").strip()
    if nm.endswith("退"):
        return Tradability(
            can_buy=False, can_sell=False,
            reason="退市整理期，不可交易",
            blocked_reason="退市整理期（名称带「退」）不可交易",
        )

    # ── 2. 停牌 ──
    if trade_status is not None and int(trade_status) == 0:
        return Tradability(
            can_buy=False, can_sell=False, reason="停牌",
            blocked_reason="停牌",
        )
    if volume is not None and float(volume) == 0:
        return Tradability(
            can_buy=False, can_sell=False, reason="当日无成交（停牌）",
            blocked_reason="当日无成交",
        )

    # ── 3. 价格数据缺失 ──
    if close is None or not _finite(close) or float(close) <= 0:
        return Tradability(
            can_buy=False, can_sell=False, reason="无有效收盘价",
            blocked_reason="缺少价格数据",
        )

    # ── 4. 流动性上限（买入金额）──
    max_buy: Optional[float] = None
    if amount is not None and _finite(amount) and float(amount) > 0:
        max_buy = float(amount) * participation_rate
        # 成交额太小 → 流动性差，整只剔除
        if float(amount) < min_turnover:
            return Tradability(
                can_buy=False, can_sell=True,
                reason=f"成交额过低（¥{float(amount)/1e4:.0f} 万 < 门槛）",
                max_buy_amount=max_buy,
                blocked_reason=(
                    f"成交额 {float(amount)/1e4:.0f} 万，低于流动性门槛 "
                    f"{min_turnover/1e4:.0f} 万"
                ),
            )

    # ── 5. 涨跌停判定（用交易所口径涨跌幅）──
    pc = float(pct_chg) if (pct_chg is not None and _finite(pct_chg)) else None
    if pc is None and close is not None:
        # 没有 pct_chg 时退回用 (high-low)/close 的近似：无法可靠判定，
        # 保守起见不拦截（宁可放过也不要误杀）
        pc = None

    can_buy = True
    can_sell = True
    blocked = ""

    if pc is not None and not pct_reliable and abs(pc) > lp * 1.5:
        # 涨跌幅不可靠（由未复权 close/prev_close 推导）且幅度明显超出
        # 涨跌停上限 → 无法区分「真涨跌停」和「除权跳空」（实测除权可达 -50%）。
        # 此时**回避该标的**：既不买也不卖，避免记下实际不存在的成交。
        return Tradability(
            can_buy=False, can_sell=False,
            reason=f"涨跌幅异常 {pc*100:.1f}%（疑似除权跳空），本日回避",
            max_buy_amount=max_buy,
            blocked_reason=(
                f"涨跌幅 {pc*100:.1f}% 明显超出涨跌停上限，无法区分涨停与除权跳空"
            ),
        )

    if pc is not None:
        thr = lp - 0.001
        if pc >= thr:
            can_buy = False
            blocked = f"涨停 {pc*100:.1f}%（上限 {lp*100:.0f}%），买不进"
        elif pc <= -thr:
            can_sell = False
            blocked = f"跌停 {pc*100:.1f}%，卖不出"

    # ── 6. 一字板：开盘即涨停价且全天没波动 ──
    _thr = lp - 0.001
    if can_buy and pc is not None and pc >= _thr:
        if (
            open_price is not None and high is not None and low is not None
            and _finite(open_price) and _finite(high) and _finite(low)
            and float(high) == float(low)
        ):
            can_buy = False
            blocked = "一字涨停，全天封死"

    # ── 7. 一字跌停：卖不出 ──
    if can_sell and pc is not None and pc <= -_thr:
        if (
            open_price is not None and high is not None and low is not None
            and _finite(open_price) and _finite(high) and _finite(low)
            and float(high) == float(low)
        ):
            can_sell = False
            blocked = "一字跌停，卖不出"

    return Tradability(
        can_buy=can_buy,
        can_sell=can_sell,
        reason="" if (can_buy and can_sell) else blocked,
        max_buy_amount=max_buy,
        blocked_reason=blocked,
    )


def _finite(v) -> bool:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return False
    return f == f and abs(f) != float("inf")   # NaN / inf 判定


# ── 向量化：逐票涨停掩码（回测引擎用）──────────────────────────────

#: 各代码前缀的涨跌停幅度
_LIMIT_BY_PREFIX = (
    ("30", 0.20),   # 创业板
    ("68", 0.20),   # 科创板
    ("92", 0.30),   # 北交所
    ("83", 0.30),
    ("87", 0.30),
    ("43", 0.30),
    ("82", 0.30),
    ("60", 0.10),   # 沪主板
    ("00", 0.10),   # 深主板
)


def limit_pct_array(codes) -> np.ndarray:
    """返回每只股票的涨跌停幅度数组。

    2026-10-25 新增。原来回测引擎用**统一**的 ``limit_up=0.095``，
    实测问题（子代理对 5598 个 parquet 扫描的结果）：
      - 182 只 20% 板（创业板/科创板）股票有 **5380 个交易日**涨幅落在
        9.5%~20% 之间（并未涨停）却被误判为涨停剔除
      - 120 只 ST 股票有 **30835 个交易日**触及 5% 涨停却**没有被拦下**
        （ST 的涨跌停只有 5%）
    """
    out = np.full(len(codes), 0.10, dtype=np.float64)
    codes = list(codes)
    for i, code in enumerate(codes):
        c = str(code)
        c2 = c[:2]
        for prefix, val in _LIMIT_BY_PREFIX:
            if c2 == prefix:
                out[i] = val
                break
    return out


def limit_up_mask_matrix(
    day_ret: np.ndarray,
    codes,
    is_st: Optional[np.ndarray] = None,
) -> np.ndarray:
    """逐票涨停掩码：day_ret >= 该票涨停幅度。

    Args:
        day_ret: (n_dates, n_stocks) 日涨跌幅
        codes: 列名（用于判定板块）
        is_st: (n_stocks,) ST 标记（1=ST，涨停幅度 5%）
    """
    limits = limit_pct_array(codes)
    if is_st is not None and len(is_st) == len(codes):
        st = np.asarray(is_st, dtype=np.float64).reshape(1, -1)
        limits = np.where(st > 0, 0.05, limits)
    return day_ret >= limits.reshape(1, -1)


# ── 行业中性化（2026-10-25）────────────────────────────────────────
# 动机（实测）：42 因子复合存在**系统性行业暴露**，分数高的票集中在少数行业。
# 改成「行业内选相对最强的」（先按行业去均值、再行业内重新标准化）后，
# 3 年全市场周频回测：夏普 0.965 → 1.441，年化 +13.25% → +25.32%，
# 回撤基本不变（-17.63% → -17.86%），分年度 4 年全部提升。
#
# 所有操作都是**逐日横截面**（只用当日全市场数据），不含未来函数。

# backend/app/services/trading/execution_rules.py → 上溯 5 层到项目根
# （trading → services → app → backend → <项目根>）
INDUSTRY_MAP_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))),
    "data", "industry", "industry_map.json",
)


def load_industry_map() -> dict:
    """{code: 行业名}；文件不存在返回空 dict。"""
    if not os.path.exists(INDUSTRY_MAP_PATH):
        return {}
    try:
        with open(INDUSTRY_MAP_PATH) as fh:
            import json

            return json.load(fh)
    except Exception:
        return {}


def industry_ids(codes, imap: dict) -> np.ndarray:
    """代码 → 行业 id 数组（-1 = 无分类，不参与中性化）。"""
    names = sorted({imap[c] for c in codes if c in imap})
    lut = {n: i for i, n in enumerate(names)}
    return np.array([lut.get(imap.get(c, ""), -1) for c in codes], dtype=np.int64)


def _row_nan_mean(block: np.ndarray) -> np.ndarray:
    """逐行 nanmean，全 NaN 行返回 0（无 RuntimeWarning）。"""
    mask = np.isfinite(block)
    cnt = mask.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore"):
        s = np.where(mask, block, 0.0).sum(axis=1, keepdims=True)
    return np.where(cnt > 0, s / np.maximum(cnt, 1), 0.0)


def _row_nan_std(block: np.ndarray) -> np.ndarray:
    """逐行 nanstd，全 NaN 行返回 0（无 RuntimeWarning）。"""
    mask = np.isfinite(block)
    cnt = mask.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore"):
        mu = _row_nan_mean(block)
        d = np.where(mask, block - mu, 0.0)
        s2 = (d * d).sum(axis=1, keepdims=True)
    var = np.where(cnt > 1, s2 / np.maximum(cnt - 1, 1), 0.0)
    return np.sqrt(np.maximum(var, 0.0))


def industry_demean(z: np.ndarray, ind_ids: np.ndarray, n_ind: int) -> np.ndarray:
    """逐日按行业去均值（nan-aware）。"""
    out = z.copy()
    for g in range(n_ind):
        cols = np.where(ind_ids == g)[0]
        if len(cols) == 0:
            continue
        out[:, cols] = z[:, cols] - _row_nan_mean(z[:, cols])
    return out


def industry_renorm(z: np.ndarray, ind_ids: np.ndarray, n_ind: int) -> np.ndarray:
    """逐日行业内重新标准化：(z - 行业均值) / 行业标准差。"""
    out = z.copy()
    for g in range(n_ind):
        cols = np.where(ind_ids == g)[0]
        if len(cols) == 0:
            continue
        block = z[:, cols]
        mu = _row_nan_mean(block)
        sd = _row_nan_std(block)
        with np.errstate(invalid="ignore"):
            out[:, cols] = np.where(sd > 1e-12, (block - mu) / sd, np.nan)
    return out


def zscore_with_industry(
    mat: np.ndarray,
    sign: float,
    min_cross: int,
    ind_ids: np.ndarray,
    n_ind: int,
    neutral: bool,
) -> np.ndarray:
    """逐日截面 z-score（带符号），可选行业中性化。"""
    ok = np.isfinite(mat)
    cnt = ok.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore"):
        mu = _row_nan_mean(mat)
        sd = _row_nan_std(mat)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (mat - mu) / np.where(sd > 1e-12, sd, np.nan) * sign
    z = np.where(cnt >= min_cross, z, np.nan)
    if neutral and n_ind > 0:
        z = industry_demean(z, ind_ids, n_ind)
        z = industry_renorm(z, ind_ids, n_ind)
    return z
