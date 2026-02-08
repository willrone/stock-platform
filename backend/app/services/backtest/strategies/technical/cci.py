"""
CCI策略 (Commodity Channel Index)
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from ...core.base_strategy import BaseStrategy
from ...models import SignalType, TradingSignal


class CCIStrategy(BaseStrategy):
    """CCI策略 (Commodity Channel Index)"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__("CCI", config)
        self.period = config.get("period", 20)
        self.oversold = config.get("oversold", -100)
        self.overbought = config.get("overbought", 100)

    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """计算CCI指标"""
        high = data["high"]
        low = data["low"]
        close = data["close"]

        typical_price = (high + low + close) / 3
        sma = typical_price.rolling(window=self.period).mean()
        mean_deviation = typical_price.rolling(window=self.period).apply(
            lambda x: np.abs(x - x.mean()).mean(), raw=True
        )

        cci = (typical_price - sma) / (0.015 * mean_deviation)

        return {"cci": cci, "typical_price": typical_price, "price": close}

    def precompute_all_signals(self, data: pd.DataFrame) -> Optional[pd.Series]:
        """[性能优化] 向量化计算全量CCI信号"""
        try:
            indicators = self.get_cached_indicators(data)
            cci = indicators["cci"]
            prev_cci = cci.shift(1)

            # 向量化逻辑判断
            # 买入：CCI 下穿超卖线
            buy_mask = (cci < self.oversold) & (prev_cci >= self.oversold)
            # 卖出：CCI 上穿超买线
            sell_mask = (cci > self.overbought) & (prev_cci <= self.overbought)

            # 构造全量信号 Series
            signals = pd.Series([None] * len(data.index), index=data.index, dtype=object)
            signals[buy_mask.fillna(False)] = SignalType.BUY
            signals[sell_mask.fillna(False)] = SignalType.SELL

            return signals
        except Exception as e:
            logger.error(f"CCI策略向量化计算失败: {e}")
            return None

    def generate_signals(
        self, data: pd.DataFrame, current_date: datetime
    ) -> List[TradingSignal]:
        """生成CCI交易信号"""
        # 性能优化：优先检查是否已有全量预计算信号
        try:
            precomputed = data.attrs.get("_precomputed_signals", {}).get(self.name)
            if precomputed is not None:
                sig_type = precomputed.get(current_date)
                if isinstance(sig_type, SignalType):
                    indicators = self.get_cached_indicators(data)
                    current_idx = self._get_current_idx(data, current_date)
                    stock_code = data.attrs.get("stock_code", "UNKNOWN")
                    current_price = indicators["price"].iloc[current_idx]
                    current_cci = indicators["cci"].iloc[current_idx]
                    return [TradingSignal(
                        timestamp=current_date,
                        stock_code=stock_code,
                        signal_type=sig_type,
                        strength=0.8,
                        price=current_price,
                        reason=f"[向量化] CCI{'超卖' if sig_type == SignalType.BUY else '超买'}: {current_cci:.2f}",
                        metadata={"cci": current_cci},
                    )]
                return []
        except Exception:
            pass

        signals = []

        try:
            indicators = self.get_cached_indicators(data)

            current_idx = self._get_current_idx(data, current_date)
            if current_idx < self.period:
                return signals

            current_price = indicators["price"].iloc[current_idx]
            current_cci = indicators["cci"].iloc[current_idx]
            prev_cci = indicators["cci"].iloc[current_idx - 1]

            stock_code = data.attrs.get("stock_code", "UNKNOWN")

            if current_cci < self.oversold and prev_cci >= self.oversold:
                strength = abs(current_cci) / abs(self.oversold)
                signal = TradingSignal(
                    timestamp=current_date,
                    stock_code=stock_code,
                    signal_type=SignalType.BUY,
                    strength=min(1.0, strength),
                    price=current_price,
                    reason=f"CCI超卖: {current_cci:.2f}",
                    metadata={"cci": current_cci},
                )
                signals.append(signal)

            elif current_cci > self.overbought and prev_cci <= self.overbought:
                strength = current_cci / self.overbought
                signal = TradingSignal(
                    timestamp=current_date,
                    stock_code=stock_code,
                    signal_type=SignalType.SELL,
                    strength=min(1.0, strength),
                    price=current_price,
                    reason=f"CCI超买: {current_cci:.2f}",
                    metadata={"cci": current_cci},
                )
                signals.append(signal)

            return signals

        except Exception as e:
            logger.error(f"CCI策略信号生成失败: {e}")
            return []
