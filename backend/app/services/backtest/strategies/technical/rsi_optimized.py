"""
RSI策略 - 性能优化版本

优化点：
1. 简化指标计算，移除复杂的背离检测和趋势分析
2. 使用向量化计算
3. 减少不必要的循环和条件判断
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from ...core.base_strategy import BaseStrategy
from ...models import SignalType, TradingSignal

# 尝试导入talib
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False


class RSIOptimizedStrategy(BaseStrategy):
    """
    简化的RSI策略 - 专注性能
    
    只保留核心逻辑：
    - RSI < 30: 超卖，买入信号
    - RSI > 70: 超买，卖出信号
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__("RSIOptimized", config)
        self.rsi_period = config.get("rsi_period", 14)
        self.oversold_threshold = config.get("oversold_threshold", 30)
        self.overbought_threshold = config.get("overbought_threshold", 70)

    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """计算RSI指标 - 简化版"""
        close_prices = data["close"]

        # 计算RSI
        if TALIB_AVAILABLE:
            rsi = pd.Series(
                talib.RSI(close_prices.values, timeperiod=self.rsi_period),
                index=close_prices.index,
            )
        else:
            # 使用pandas实现RSI
            delta = close_prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

        return {
            "rsi": rsi,
            "price": close_prices,
        }

    def precompute_all_signals(self, data: pd.DataFrame) -> Optional[pd.Series]:
        """向量化计算全量RSI信号"""
        try:
            indicators = self.get_cached_indicators(data)
            rsi = indicators["rsi"]
            prev_rsi = rsi.shift(1)

            # 简化逻辑：从超卖区回升 -> 买入；从超买区���调 -> 卖出
            buy_mask = (prev_rsi <= self.oversold_threshold) & (rsi > self.oversold_threshold)
            sell_mask = (prev_rsi >= self.overbought_threshold) & (rsi < self.overbought_threshold)

            signals = pd.Series([None] * len(data.index), index=data.index, dtype=object)
            signals[buy_mask.fillna(False)] = SignalType.BUY
            signals[sell_mask.fillna(False)] = SignalType.SELL
            return signals
        except Exception as e:
            logger.error(f"RSI策略向量化计算失败: {e}")
            return None

    def generate_signals(
        self, data: pd.DataFrame, current_date: datetime
    ) -> List[TradingSignal]:
        """生成RSI信号 - 简化版"""
        # 优先使用预计算信号
        try:
            precomputed = data.attrs.get("_precomputed_signals", {}).get(self.name)
            if precomputed is not None:
                sig_type = precomputed.get(current_date)
                if isinstance(sig_type, SignalType):
                    indicators = self.get_cached_indicators(data)
                    current_idx = self._get_current_idx(data, current_date)
                    stock_code = data.attrs.get("stock_code", "UNKNOWN")
                    current_price = indicators["price"].iloc[current_idx]
                    current_rsi = indicators["rsi"].iloc[current_idx]
                    return [TradingSignal(
                        timestamp=current_date,
                        stock_code=stock_code,
                        signal_type=sig_type,
                        strength=0.8,
                        price=current_price,
                        reason=f"RSI信号, RSI: {current_rsi:.2f}",
                        metadata={"rsi": current_rsi},
                    )]
                return []
        except Exception:
            pass

        signals = []

        try:
            indicators = self.get_cached_indicators(data)
            current_idx = self._get_current_idx(data, current_date)
            
            if current_idx < self.rsi_period:
                return signals

            if current_idx < 1:
                return signals

            current_rsi = indicators["rsi"].iloc[current_idx]
            prev_rsi = indicators["rsi"].iloc[current_idx - 1]
            current_price = indicators["price"].iloc[current_idx]
            stock_code = data.attrs.get("stock_code", "UNKNOWN")

            # 买入信号：RSI从超卖区域向上穿越
            if prev_rsi <= self.oversold_threshold and current_rsi > self.oversold_threshold:
                signal = TradingSignal(
                    timestamp=current_date,
                    stock_code=stock_code,
                    signal_type=SignalType.BUY,
                    strength=0.8,
                    price=current_price,
                    reason=f"RSI从超卖区域向上穿越({prev_rsi:.2f}->{current_rsi:.2f})",
                    metadata={"rsi": current_rsi, "prev_rsi": prev_rsi},
                )
                signals.append(signal)

            # 卖出信号：RSI从超买区域向下穿越
            elif prev_rsi >= self.overbought_threshold and current_rsi <= self.overbought_threshold:
                signal = TradingSignal(
                    timestamp=current_date,
                    stock_code=stock_code,
                    signal_type=SignalType.SELL,
                    strength=0.8,
                    price=current_price,
                    reason=f"RSI从超买区域向下穿越({prev_rsi:.2f}->{current_rsi:.2f})",
                    metadata={"rsi": current_rsi, "prev_rsi": prev_rsi},
                )
                signals.append(signal)

            return signals

        except Exception as e:
            logger.error(f"RSI策略信号生成失败: {e}")
            return []
