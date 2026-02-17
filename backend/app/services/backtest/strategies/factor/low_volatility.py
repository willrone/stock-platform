"""
低波动因子策略
"""

from datetime import datetime
from typing import Any, Dict, List

from app.core.error_handler import ErrorSeverity, TaskError
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

            # 买入：低波动且风险调整收益为正，且波动率低于历史中位数
            vol_median = indicators["volatility"].iloc[max(0, current_idx-252):current_idx+1].median()
            if current_rar > 0 and current_vol < vol_median:
                strength = min(1.0, current_rar / 3)
                signal = TradingSignal(
                    timestamp=current_date,
                    stock_code=stock_code,
                    signal_type=SignalType.BUY,
                    strength=strength,
                    price=current_price,
                    reason=f"低波动高收益，波动率: {current_vol:.2%}, RAR: {current_rar:.3f}",
                    metadata={
                        "volatility": current_vol,
                        "risk_adjusted_return": current_rar,
                        "vol_median": vol_median,
                    },
                )
                signals.append(signal)

            # 卖出：风险调整收益转负，或波动率飙升超过历史中位数的1.5倍
            elif current_rar < -0.5 or current_vol > vol_median * 1.5:
                strength = min(1.0, max(abs(current_rar) / 3, current_vol / vol_median - 1) if vol_median > 0 else abs(current_rar) / 3)
                signal = TradingSignal(
                    timestamp=current_date,
                    stock_code=stock_code,
                    signal_type=SignalType.SELL,
                    strength=strength,
                    price=current_price,
                    reason=f"波动率飙升或收益恶化，波动率: {current_vol:.2%}, RAR: {current_rar:.3f}",
                    metadata={
                        "volatility": current_vol,
                        "risk_adjusted_return": current_rar,
                        "vol_median": vol_median,
                    },
                )
                signals.append(signal)

            return signals

        except Exception as e:
            logger.error(f"低波动因子策略信号生成失败: {e}")
            return []
