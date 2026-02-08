"""
均值回归策略
"""

from datetime import datetime
from typing import Any, Dict, List

import pandas as pd
from loguru import logger

from ..base.statistical_arbitrage_base import StatisticalArbitrageStrategy
from ...models import SignalType, TradingSignal


class MeanReversionStrategy(StatisticalArbitrageStrategy):
    """均值回归策略"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__("MeanReversion", config)
        self.lookback_period = config.get("lookback_period", 20)
        self.zscore_threshold = config.get("zscore_threshold", 2.0)
        self.position_size = config.get("position_size", 0.1)

    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """计算均值回归指标"""
        close_prices = data["close"]

        sma = close_prices.rolling(window=self.lookback_period).mean()
        std = close_prices.rolling(window=self.lookback_period).std()

        zscore = (close_prices - sma) / std

        return {"sma": sma, "std": std, "zscore": zscore, "price": close_prices}

    def generate_signals(
        self, data: pd.DataFrame, current_date: datetime
    ) -> List[TradingSignal]:
        """生成均值回归交易信号"""
        signals = []

        try:
            indicators = self.get_cached_indicators(data)

            current_idx = self._get_current_idx(data, current_date)
            if current_idx < self.lookback_period:
                return signals

            current_price = indicators["price"].iloc[current_idx]
            current_zscore = indicators["zscore"].iloc[current_idx]
            prev_zscore = indicators["zscore"].iloc[current_idx - 1]

            stock_code = data.attrs.get("stock_code", "UNKNOWN")

            if (
                prev_zscore <= -self.zscore_threshold
                and current_zscore > -self.zscore_threshold
            ):
                strength = min(1.0, abs(current_zscore) / self.zscore_threshold)
                signal = TradingSignal(
                    timestamp=current_date,
                    stock_code=stock_code,
                    signal_type=SignalType.BUY,
                    strength=strength,
                    price=current_price,
                    reason=f"价格回归均值，Z-score: {current_zscore:.2f}",
                    metadata={
                        "zscore": current_zscore,
                        "sma": indicators["sma"].iloc[current_idx],
                    },
                )
                signals.append(signal)

            elif (
                prev_zscore >= self.zscore_threshold
                and current_zscore < self.zscore_threshold
            ):
                strength = min(1.0, abs(current_zscore) / self.zscore_threshold)
                signal = TradingSignal(
                    timestamp=current_date,
                    stock_code=stock_code,
                    signal_type=SignalType.SELL,
                    strength=strength,
                    price=current_price,
                    reason=f"价格偏离均值，Z-score: {current_zscore:.2f}",
                    metadata={
                        "zscore": current_zscore,
                        "sma": indicators["sma"].iloc[current_idx],
                    },
                )
                signals.append(signal)

            return signals

        except Exception as e:
            logger.error(f"均值回归策略信号生成失败: {e}")
            return []
