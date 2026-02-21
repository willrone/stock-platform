"""
均值回归策略
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from ...models import SignalType, TradingSignal
from ..base.statistical_arbitrage_base import StatisticalArbitrageStrategy


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

    def precompute_all_signals(self, data: pd.DataFrame) -> Optional[pd.Series]:
        """[性能优化] 向量化计算全量均值回归信号

        返回值：正数=买入信号（强度），负数=卖出信号（强度），0=无信号
        强度 = |zscore| / zscore_threshold，clamp 到 [0.3, 1.0]
        """
        try:
            indicators = self.get_cached_indicators(data)
            zscore = indicators["zscore"]
            prev_zscore = zscore.shift(1)

            # 买入：Z-score 从上方穿越 -threshold（价格极端偏低）
            buy_mask = (
                (prev_zscore >= -self.zscore_threshold)
                & (zscore < -self.zscore_threshold)
            )
            # 卖出：Z-score 从下方穿越 +threshold（价格极端偏高）
            sell_mask = (
                (prev_zscore <= self.zscore_threshold)
                & (zscore > self.zscore_threshold)
            )

            # 构造浮点强度信号
            signals = pd.Series(0.0, index=data.index, dtype=np.float64)

            # 买入强度：|zscore| / threshold，clamp 到 [0.3, 1.0]
            buy_strength = (zscore.abs() / self.zscore_threshold).clip(0.3, 1.0)
            signals[buy_mask.fillna(False)] = buy_strength[buy_mask.fillna(False)]

            # 卖出强度（负值）
            sell_strength = -(zscore.abs() / self.zscore_threshold).clip(0.3, 1.0)
            signals[sell_mask.fillna(False)] = sell_strength[sell_mask.fillna(False)]

            return signals
        except Exception as e:
            logger.error(f"MeanReversion策略向量化计算失败: {e}")
            return None

    def compute_score(self, data, current_date) -> float:
        """基于当前 z-score 计算连续评分（-1.0 ~ +1.0）。

        z-score < 0（价格低于均值）→ 正分（看多，均值回归预期上涨）
        z-score > 0（价格高于均值）→ 负分（看空，均值回归预期下跌）

        映射：score = -zscore / zscore_threshold，clamp 到 [-1.0, 1.0]
        """
        try:
            indicators = self.get_cached_indicators(data)
            current_idx = self._get_current_idx(data, current_date)
            if current_idx < self.lookback_period:
                return 0.0

            zscore_val = float(indicators["zscore"].iloc[current_idx])
            if zscore_val != zscore_val:  # NaN check
                return 0.0

            # 均值回归：价格低于均值（z<0）看多，价格高于均值（z>0）看空
            score = -zscore_val / max(self.zscore_threshold, 0.01)
            return max(-1.0, min(1.0, score))
        except Exception:
            return 0.0

    def generate_signals(
        self, data: pd.DataFrame, current_date: datetime
    ) -> List[TradingSignal]:
        """生成均值回归交易信号"""
        # 优先使用预计算信号
        try:
            precomputed = data.attrs.get("_precomputed_signals", {}).get(self.name)
            if precomputed is not None:
                sig_val = precomputed.get(current_date)
                if (
                    isinstance(sig_val, (int, float, np.number))
                    and float(sig_val) != 0.0
                ):
                    fv = float(sig_val)
                    sig_type = SignalType.BUY if fv > 0 else SignalType.SELL
                    indicators = self.get_cached_indicators(data)
                    current_idx = self._get_current_idx(data, current_date)
                    stock_code = data.attrs.get("stock_code", "UNKNOWN")
                    current_price = indicators["price"].iloc[current_idx]
                    current_zscore = indicators["zscore"].iloc[current_idx]
                    return [
                        TradingSignal(
                            timestamp=current_date,
                            stock_code=stock_code,
                            signal_type=sig_type,
                            strength=min(1.0, abs(fv)),
                            price=current_price,
                            reason=f"[向量化] 均值回归信号, Z-score: {current_zscore:.2f}",
                            metadata={
                                "zscore": current_zscore,
                                "sma": indicators["sma"].iloc[current_idx],
                            },
                        )
                    ]
                return []
        except Exception:
            pass

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

            # 买入：Z-score 跌破 -threshold（价格处于极端低位）
            if (
                current_zscore < -self.zscore_threshold
                and prev_zscore >= -self.zscore_threshold
            ):
                strength = min(1.0, abs(current_zscore) / self.zscore_threshold)
                signal = TradingSignal(
                    timestamp=current_date,
                    stock_code=stock_code,
                    signal_type=SignalType.BUY,
                    strength=strength,
                    price=current_price,
                    reason=f"价格极端偏低，Z-score: {current_zscore:.2f}",
                    metadata={
                        "zscore": current_zscore,
                        "sma": indicators["sma"].iloc[current_idx],
                    },
                )
                signals.append(signal)

            # 卖出：Z-score 突破 +threshold（价格处于极端高位）
            elif (
                current_zscore > self.zscore_threshold
                and prev_zscore <= self.zscore_threshold
            ):
                strength = min(1.0, abs(current_zscore) / self.zscore_threshold)
                signal = TradingSignal(
                    timestamp=current_date,
                    stock_code=stock_code,
                    signal_type=SignalType.SELL,
                    strength=strength,
                    price=current_price,
                    reason=f"价格极端偏高，Z-score: {current_zscore:.2f}",
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
