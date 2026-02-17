"""
价值因子策略

2026-02：支持可选的估值阈值筛选（用于提升信号质量/稳定性）。
注意：当前回测数据源未包含真实财务报表 ROE 等字段；
因此这里的 PE/PB 为策略内部估计序列，仅用于"相对过滤/去极值"。
"""

from datetime import datetime
from typing import Any, Dict, List

import pandas as pd
from loguru import logger

from ..base.factor_base import FactorStrategy
from ...models import SignalType, TradingSignal


class ValueFactorStrategy(FactorStrategy):
    """价值因子策略

    2026-02：支持可选的估值阈值筛选（用于提升信号质量/稳定性）。
    注意：当前回测数据源未包含真实财务报表 ROE 等字段；
    因此这里的 PE/PB 为策略内部估计序列，仅用于"相对过滤/去极值"。
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__("ValueFactor", config)
        self.pe_weight = config.get("pe_weight", 0.25)
        self.pb_weight = config.get("pb_weight", 0.25)
        self.ps_weight = config.get("ps_weight", 0.25)
        self.ev_ebitda_weight = config.get("ev_ebitda_weight", 0.25)

        # Optional valuation thresholds (filters)
        # If set, a BUY signal will only be emitted when estimated ratios satisfy the constraints.
        self.pe_max = config.get("pe_max", None)
        self.pb_max = config.get("pb_max", None)
        # placeholder for future fundamental integration
        self.roe_min = config.get("roe_min", None)

    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """计算价值因子指标
        
        由于缺少真实财务数据，使用价格行为构建相对价值指标：
        - 价格相对长期均线的偏离度（替代 PE）
        - 价格波动率的倒数（替代 PB，低波动=稳定=高质量）
        - 短期收益率均值（替代 PS）
        - 长期趋势强度（替代 EV/EBITDA）
        """
        close_prices = data["close"]
        returns = close_prices.pct_change().dropna()

        # 价格相对 252 日均线的偏离度（负值=低估）
        ma252 = close_prices.rolling(window=252, min_periods=126).mean()
        price_deviation = (close_prices - ma252) / (ma252 + 1e-8)

        # 波动率（低波动 = 高质量）
        volatility = returns.rolling(window=63, min_periods=30).std()

        # 短期收益率均值
        short_return = returns.rolling(window=21, min_periods=10).mean()

        # 长期趋势（126日收益率）
        long_return = close_prices / close_prices.shift(126) - 1

        # 价值评分：低偏离 + 低波动 + 正收益趋势
        # 用 rolling z-score 标准化每个因子
        def rolling_zscore(s, window=252):
            m = s.rolling(window=window, min_periods=60).mean()
            sd = s.rolling(window=window, min_periods=60).std() + 1e-8
            return (s - m) / sd

        dev_z = rolling_zscore(price_deviation)
        vol_z = rolling_zscore(volatility)
        ret_z = rolling_zscore(short_return)
        trend_z = rolling_zscore(long_return)

        # 价值评分 = 低偏离(-dev) + 低波动(-vol) + 正收益(+ret) + 正趋势(+trend)
        value_score = (
            -dev_z * self.pe_weight
            + -vol_z * self.pb_weight
            + ret_z * self.ps_weight
            + trend_z * self.ev_ebitda_weight
        )

        return {
            "pe_ratio": price_deviation,
            "pb_ratio": volatility,
            "ps_ratio": short_return,
            "ev_ebitda": long_return,
            "value_score": value_score,
            "price": close_prices,
        }

    def generate_signals(
        self, data: pd.DataFrame, current_date: datetime
    ) -> List[TradingSignal]:
        """价值因子信号生成"""
        signals = []

        try:
            indicators = self.get_cached_indicators(data)

            current_idx = self._get_current_idx(data, current_date)
            if current_idx < 260:
                return signals

            current_price = indicators["price"].iloc[current_idx]
            current_score = indicators["value_score"].iloc[current_idx]
            prev_score = indicators["value_score"].iloc[current_idx - 1]

            stock_code = data.attrs.get("stock_code", "UNKNOWN")

            if prev_score <= 0 and current_score > 0:
                # Optional valuation thresholds
                current_pe = float(indicators["pe_ratio"].iloc[current_idx])
                current_pb = float(indicators["pb_ratio"].iloc[current_idx])

                if self.pe_max is not None and current_pe > float(self.pe_max):
                    return signals
                if self.pb_max is not None and current_pb > float(self.pb_max):
                    return signals

                strength = min(1.0, current_score)
                reason = f"价值因子评分转正: {current_score:.3f}"
                if self.pe_max is not None or self.pb_max is not None:
                    reason += f" (PE≤{self.pe_max}, PB≤{self.pb_max})"

                signal = TradingSignal(
                    timestamp=current_date,
                    stock_code=stock_code,
                    signal_type=SignalType.BUY,
                    strength=strength,
                    price=current_price,
                    reason=reason,
                    metadata={
                        "value_score": current_score,
                        "pe_ratio": current_pe,
                        "pb_ratio": current_pb,
                        "pe_max": self.pe_max,
                        "pb_max": self.pb_max,
                    },
                )
                signals.append(signal)

            elif prev_score >= 0 and current_score < 0:
                strength = min(1.0, abs(current_score))
                signal = TradingSignal(
                    timestamp=current_date,
                    stock_code=stock_code,
                    signal_type=SignalType.SELL,
                    strength=strength,
                    price=current_price,
                    reason=f"价值因子评分转负: {current_score:.3f}",
                    metadata={
                        "value_score": current_score,
                        "pe_ratio": indicators["pe_ratio"].iloc[current_idx],
                        "pb_ratio": indicators["pb_ratio"].iloc[current_idx],
                    },
                )
                signals.append(signal)

            return signals

        except Exception as e:
            logger.error(f"价值因子策略信号生成失败: {e}")
            return []
