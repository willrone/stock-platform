"""
配对交易策略
"""

from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from ...models import SignalType, TradingSignal
from ..base.statistical_arbitrage_base import StatisticalArbitrageStrategy


class PairsTradingStrategy(StatisticalArbitrageStrategy):
    """配对交易策略"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__("PairsTrading", config)
        self.correlation_threshold = config.get("correlation_threshold", 0.8)
        self.min_data_points = config.get("min_data_points", 60)
        self.lookback_period = config.get("lookback_period", 20)
        self.zscore_threshold = config.get("entry_threshold", 2.0)

    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """计算配对交易指标（单股票模式下使用价格动量）"""
        close_prices = data["close"]

        returns = close_prices.pct_change()
        volatility = returns.rolling(window=self.lookback_period).std()
        zscore = (
            close_prices - close_prices.rolling(window=self.lookback_period).mean()
        ) / (volatility * np.sqrt(self.lookback_period) + 0.001)

        momentum_5d = close_prices / close_prices.shift(5) - 1
        momentum_20d = close_prices / close_prices.shift(20) - 1

        return {
            "price": close_prices,
            "returns": returns,
            "volatility": volatility,
            "zscore": zscore,
            "momentum_5d": momentum_5d,
            "momentum_20d": momentum_20d,
        }

    def find_pairs(self, stock_data: Dict[str, pd.DataFrame]) -> List[Tuple[str, str]]:
        """寻找符合条件的配对"""
        valid_pairs = []
        stock_codes = list(stock_data.keys())

        for i in range(len(stock_codes)):
            for j in range(i + 1, len(stock_codes)):
                code1, code2 = stock_codes[i], stock_codes[j]
                try:
                    corr = self.validate_pair_correlation(
                        stock_data[code1], stock_data[code2]
                    )
                    if abs(corr) >= self.correlation_threshold:
                        valid_pairs.append((code1, code2, corr))
                except Exception as e:
                    logger.warning(f"配对检验失败: {code1}-{code2}, {e}")
                    continue

        valid_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        return valid_pairs[:10]

    def generate_signals(
        self, data: pd.DataFrame, current_date: datetime
    ) -> List[TradingSignal]:
        """生成配对交易信号（单股票模式下使用相对强度）"""
        signals = []

        try:
            indicators = self.get_cached_indicators(data)

            current_idx = self._get_current_idx(data, current_date)
            if current_idx < self.lookback_period + 20:
                return signals

            current_price = indicators["price"].iloc[current_idx]
            current_zscore = indicators["zscore"].iloc[current_idx]
            prev_zscore = indicators["zscore"].iloc[current_idx - 1]
            momentum_5d = indicators["momentum_5d"].iloc[current_idx]
            momentum_20d = indicators["momentum_20d"].iloc[current_idx]

            stock_code = data.attrs.get("stock_code", "UNKNOWN")

            # 买入：Z-score 跌破 -threshold（价格极端偏低），动量确认
            if (
                prev_zscore <= -self.zscore_threshold
                and current_zscore > -self.zscore_threshold
            ):
                strength = min(1.0, abs(current_zscore) / self.zscore_threshold)
                # 动量确认：短期动量开始回升（放宽条件）
                if momentum_5d > momentum_20d or momentum_5d > -0.05:
                    signal = TradingSignal(
                        timestamp=current_date,
                        stock_code=stock_code,
                        signal_type=SignalType.BUY,
                        strength=strength,
                        price=current_price,
                        reason=f"配对交易信号：价差回归，Z-score: {current_zscore:.2f}",
                        metadata={
                            "zscore": current_zscore,
                            "momentum_5d": momentum_5d,
                            "momentum_20d": momentum_20d,
                        },
                    )
                    signals.append(signal)

            # 卖出：Z-score 突破 +threshold（价格极端偏高），动量确认
            elif (
                prev_zscore >= self.zscore_threshold
                and current_zscore < self.zscore_threshold
            ):
                strength = min(1.0, abs(current_zscore) / self.zscore_threshold)
                # 动量确认：短期动量开始回落
                if momentum_5d < momentum_20d or momentum_5d < 0.05:
                    signal = TradingSignal(
                        timestamp=current_date,
                        stock_code=stock_code,
                        signal_type=SignalType.SELL,
                        strength=strength,
                        price=current_price,
                        reason=f"配对交易信号：价差偏离，Z-score: {current_zscore:.2f}",
                        metadata={
                            "zscore": current_zscore,
                            "momentum_5d": momentum_5d,
                            "momentum_20d": momentum_20d,
                        },
                    )
                    signals.append(signal)

            return signals

        except Exception as e:
            logger.error(f"配对交易策略信号生成失败: {e}")
            return []
