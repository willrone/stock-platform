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

from dataclasses import dataclass
from typing import Optional

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

    if pc is not None:
        if pc >= lp - 0.001:
            can_buy = False
            blocked = f"涨停 {pc*100:.1f}%（上限 {lp*100:.0f}%），买不进"
        elif pc <= -lp + 0.001:
            can_sell = False
            blocked = f"跌停 {pc*100:.1f}%，卖不出"

    # ── 6. 一字板：开盘即涨停价且全天没波动 ──
    if can_buy and pc is not None and pc >= lp - 0.001:
        if (
            open_price is not None and high is not None and low is not None
            and _finite(open_price) and _finite(high) and _finite(low)
            and float(high) == float(low)
        ):
            can_buy = False
            blocked = "一字涨停，全天封死"

    # ── 7. 一字跌停：卖不出 ──
    if can_sell and pc is not None and pc <= -lp + 0.001:
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
