"""
低波动因子策略
"""

from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from loguru import logger

from ..base.factor_base import FactorStrategy
from ...models import SignalType, TradingSignal


class LowVolatilityStrategy(FactorStrategy):
    """低波动因子策略"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__("LowVolatility", config)
        self.volatility_period = config.get("volatility_period", 21)
        self.volatility_window = config.get("volatility_window", 63)

    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """计算低波动因子指标"""
        close_prices = data["close"]

        daily_returns = close_prices.pct_change()
        historical_vol = daily_returns.rolling(
            window=self.volatility_window
        ).std() * np.sqrt(252)

        risk_adjusted_return = daily_returns.rolling(
            window=self.volatility_period
        ).mean() / (daily_returns.rolling(window=self.volatility_period).std() + 1e-8)

        return {
            "volatility": historical_vol,
            "risk_adjusted_return": risk_adjusted_return,
            "price": close_prices,
        }

    def generate_signals(
        self, data: pd.DataFrame, current_date: datetime
    ) -> List[TradingSignal]:
        """低波动因子信号生成"""
        signals = []

        try:
            indicators = self.get_cached_indicators(data)

            current_idx = self._get_current_idx(data, current_date)
            if current_idx < self.volatility_window:
                return signals

            current_price = indicators["price"].iloc[current_idx]
            current_vol = indicators["volatility"].iloc[current_idx]
            current_rar = indicators["risk_adjusted_return"].iloc[current_idx]

            stock_code = data.attrs.get("stock_code", "UNKNOWN")

            if current_rar > 0:
                strength = min(1.0, current_rar / 5)
                signal = TradingSignal(
                    timestamp=current_date,
                    stock_code=stock_code,
                    signal_type=SignalType.BUY,
                    strength=strength,
                    price=current_price,
                    reason=f"低波动高收益，波动率: {current_vol:.2%}",
                    metadata={
                        "volatility": current_vol,
                        "risk_adjusted_return": current_rar,
                    },
                )
                signals.append(signal)

            return signals

        except Exception as e:
            logger.error(f"低波动因子策略信号生成失败: {e}")
            return []
