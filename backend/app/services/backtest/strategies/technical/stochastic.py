"""
随机指标策略
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger

from ...core.base_strategy import BaseStrategy
from ...models import SignalType, TradingSignal


class StochasticStrategy(BaseStrategy):
    """随机指标策略"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__("Stochastic", config)
        self.k_period = config.get("k_period", 14)
        self.d_period = config.get("d_period", 3)
        self.oversold = config.get("oversold", 20)
        self.overbought = config.get("overbought", 80)

    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """计算随机指标"""
        high = data["high"]
        low = data["low"]
        close = data["close"]

        lowest_low = low.rolling(window=self.k_period).min()
        highest_high = high.rolling(window=self.k_period).max()

        k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d_percent = k_percent.rolling(window=self.d_period).mean()

        return {"k_percent": k_percent, "d_percent": d_percent, "price": close}

    def precompute_all_signals(self, data: pd.DataFrame) -> Optional[pd.Series]:
        """[性能优化] 向量化计算全量随��指标信号"""
        try:
            indicators = self.get_cached_indicators(data)
            k = indicators["k_percent"]
            d = indicators["d_percent"]
            prev_k = k.shift(1)
            prev_d = d.shift(1)

            # 买入：深度超卖区金叉 (K < oversold 且 K 上穿 D，且 D 也在超卖区)
            buy_mask = (
                (k < self.oversold)
                & (d < self.oversold + 10)
                & (k > d)
                & (prev_k <= prev_d)
            )
            # 卖出：深度超买区死叉 (K > overbought 且 K 下穿 D，且 D 也在超买区)
            sell_mask = (
                (k > self.overbought)
                & (d > self.overbought - 10)
                & (k < d)
                & (prev_k >= prev_d)
            )

            signals = pd.Series(
                [None] * len(data.index), index=data.index, dtype=object
            )
            signals[buy_mask.fillna(False)] = SignalType.BUY
            signals[sell_mask.fillna(False)] = SignalType.SELL

            return signals
        except Exception as e:
            logger.error(f"Stochastic策略向量化计算失败: {e}")
            return None

    def generate_signals(
        self, data: pd.DataFrame, current_date: datetime
    ) -> List[TradingSignal]:
        """生成随机指标交易信号"""
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
                    current_k = indicators["k_percent"].iloc[current_idx]
                    current_d = indicators["d_percent"].iloc[current_idx]
                    return [
                        TradingSignal(
                            timestamp=current_date,
                            stock_code=stock_code,
                            signal_type=sig_type,
                            strength=0.8,
                            price=current_price,
                            reason=f"[向量化] 随机指标{'超卖金叉' if sig_type == SignalType.BUY else '超买死叉'}，K: {current_k:.2f}, D: {current_d:.2f}",
                            metadata={"k_percent": current_k, "d_percent": current_d},
                        )
                    ]
                return []
        except Exception:
            pass

        signals = []

        try:
            indicators = self.get_cached_indicators(data)

            current_idx = self._get_current_idx(data, current_date)
            if current_idx < self.k_period + self.d_period:
                return signals

            current_price = indicators["price"].iloc[current_idx]
            current_k = indicators["k_percent"].iloc[current_idx]
            current_d = indicators["d_percent"].iloc[current_idx]
            prev_k = indicators["k_percent"].iloc[current_idx - 1]
            prev_d = indicators["d_percent"].iloc[current_idx - 1]

            stock_code = data.attrs.get("stock_code", "UNKNOWN")

            if current_k < self.oversold and prev_k < self.oversold:
                if current_k > current_d and prev_k <= prev_d:
                    strength = (self.oversold - current_k) / self.oversold
                    signal = TradingSignal(
                        timestamp=current_date,
                        stock_code=stock_code,
                        signal_type=SignalType.BUY,
                        strength=min(1.0, strength),
                        price=current_price,
                        reason=f"随机指标超卖金叉，K: {current_k:.2f}, D: {current_d:.2f}",
                        metadata={"k_percent": current_k, "d_percent": current_d},
                    )
                    signals.append(signal)

            elif current_k > self.overbought and prev_k > self.overbought:
                if current_k < current_d and prev_k >= prev_d:
                    strength = (current_k - self.overbought) / (100 - self.overbought)
                    signal = TradingSignal(
                        timestamp=current_date,
                        stock_code=stock_code,
                        signal_type=SignalType.SELL,
                        strength=min(1.0, strength),
                        price=current_price,
                        reason=f"随机指标超买死叉，K: {current_k:.2f}, D: {current_d:.2f}",
                        metadata={"k_percent": current_k, "d_percent": current_d},
                    )
                    signals.append(signal)

            return signals

        except Exception as e:
            logger.error(f"随机指标策略信号生成失败: {e}")
            return []
