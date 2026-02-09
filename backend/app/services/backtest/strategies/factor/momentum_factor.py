"""
动量因子策略
"""

from datetime import datetime
from typing import Any, Dict, List

from app.core.error_handler import ErrorSeverity, TaskError
import pandas as pd
from loguru import logger

from ..base.factor_base import FactorStrategy
from ...models import SignalType, TradingSignal


class MomentumFactorStrategy(FactorStrategy):
    """动量因子策略"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__("MomentumFactor", config)
        self.momentum_periods = config.get("momentum_periods", [21, 63, 126])
        self.momentum_weights = config.get("momentum_weights", [0.5, 0.3, 0.2])
        self.lookback_period = max(
            config.get("lookback_period", 252), max(self.momentum_periods)
        )

    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """计算动量因子指标"""
        close_prices = data["close"]
        momentum_scores = pd.Series(0.0, index=close_prices.index)

        for period, weight in zip(self.momentum_periods, self.momentum_weights):
            if len(close_prices) >= period:
                returns = close_prices / close_prices.shift(period) - 1
                normalized_returns = (returns - returns.mean()) / returns.std()
                momentum_scores += normalized_returns * weight

        return {"momentum": momentum_scores, "price": close_prices}

    def generate_signals(
        self, data: pd.DataFrame, current_date: datetime
    ) -> List[TradingSignal]:
        """动量因子信号生成"""
        signals = []

        try:
            indicators = self.get_cached_indicators(data)

            current_idx = self._get_current_idx(data, current_date)
            if current_idx < self.lookback_period:
                return signals

            current_price = indicators["price"].iloc[current_idx]
            current_momentum = indicators["momentum"].iloc[current_idx]
            prev_momentum = indicators["momentum"].iloc[current_idx - 1]

            stock_code = data.attrs.get("stock_code", "UNKNOWN")

            if prev_momentum <= 0 and current_momentum > 0:
                strength = min(1.0, abs(current_momentum))
                signal = TradingSignal(
                    timestamp=current_date,
                    stock_code=stock_code,
                    signal_type=SignalType.BUY,
                    strength=strength,
                    price=current_price,
                    reason=f"动量转正: {current_momentum:.3f}",
                    metadata={"momentum": current_momentum},
                )
                signals.append(signal)

            elif prev_momentum >= 0 and current_momentum < 0:
                strength = min(1.0, abs(current_momentum))
                signal = TradingSignal(
                    timestamp=current_date,
                    stock_code=stock_code,
                    signal_type=SignalType.SELL,
                    strength=strength,
                    price=current_price,
                    reason=f"动量转负: {current_momentum:.3f}",
                    metadata={"momentum": current_momentum},
                )
                signals.append(signal)

            return signals

        except Exception as e:
            logger.error(f"动量因子策略信号生成失败: {e}")
            return []
