"""
多因子组合策略
"""

from datetime import datetime
from typing import Any, Dict, List
import threading

import numpy as np
import pandas as pd
from loguru import logger

from ..base.factor_base import FactorStrategy
from ...models import SignalType, TradingSignal


class MultiFactorStrategy(FactorStrategy):
    """多因子组合策略"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__("MultiFactor", config)
        self.factors = config.get("factors", ["value", "momentum", "low_volatility"])
        self.factor_weights = config.get("factor_weights", [0.33, 0.33, 0.34])
        self.weighting_method = config.get("weighting_method", "equal")

        # PERF: indicators are expensive (rolling windows). In backtest we call generate_signals
        # every day; without caching this becomes O(T^2). Cache per-stock dataframe.
        self._indicator_cache: Dict[tuple, Dict[str, pd.Series]] = {}
        self._indicator_cache_lock = threading.Lock()

    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """计算多因子综合指标

        注意：该函数包含大量 rolling 计算，回测按天调用时必须做缓存，否则会成为主要性能瓶颈。
        """
        stock_code = data.attrs.get("stock_code", "UNKNOWN")
        cache_key = (
            stock_code,
            id(data),
            tuple(self.factors),
            tuple(float(x) for x in self.factor_weights),
            str(self.weighting_method),
        )
        with self._indicator_cache_lock:
            cached = self._indicator_cache.get(cache_key)
        if cached is not None:
            return cached

        close_prices = data["close"]

        returns = close_prices.pct_change()

        volatility = returns.rolling(window=21).std() * np.sqrt(252)
        volatility_normalized = (volatility - volatility.rolling(window=252).mean()) / (
            volatility.rolling(window=252).std() + 0.001
        )

        momentum_1m = close_prices / close_prices.shift(21) - 1
        momentum_3m = close_prices / close_prices.shift(63) - 1
        momentum_6m = close_prices / close_prices.shift(126) - 1

        momentum_combined = momentum_1m * 0.2 + momentum_3m * 0.3 + momentum_6m * 0.5
        momentum_normalized = (
            momentum_combined - momentum_combined.rolling(window=60).mean()
        ) / (momentum_combined.rolling(window=60).std() + 0.001)

        price_to_ma20 = close_prices / close_prices.rolling(window=20).mean() - 1
        price_to_ma60 = close_prices / close_prices.rolling(window=60).mean() - 1
        value_score = -(price_to_ma20 * 0.6 + price_to_ma60 * 0.4)

        low_vol_normalized = -volatility_normalized

        factor_scores = {
            "value": value_score,
            "momentum": momentum_normalized,
            "low_volatility": low_vol_normalized,
        }

        combined_score = pd.Series(0.0, index=close_prices.index)
        for i, factor_name in enumerate(self.factors):
            if factor_name in factor_scores:
                score = factor_scores[factor_name].fillna(0)
                weight = (
                    self.factor_weights[i]
                    if i < len(self.factor_weights)
                    else 1.0 / len(self.factors)
                )
                combined_score = combined_score + score * weight

        combined_score = combined_score.rolling(window=5).mean()

        out = {
            "combined_score": combined_score,
            "value_score": value_score,
            "momentum_score": momentum_normalized,
            "low_vol_score": low_vol_normalized,
            "price": close_prices,
            "volatility": volatility,
        }

        with self._indicator_cache_lock:
            self._indicator_cache[cache_key] = out

        return out

    def generate_signals(
        self, data: pd.DataFrame, current_date: datetime
    ) -> List[TradingSignal]:
        """多因子策略信号生成"""
        signals = []

        try:
            indicators = self.get_cached_indicators(data)

            current_idx = self._get_current_idx(data, current_date)
            if current_idx < 130:
                return signals

            current_price = float(indicators["price"].iloc[current_idx])
            current_score = float(indicators["combined_score"].iloc[current_idx])
            prev_score = float(indicators["combined_score"].iloc[current_idx - 1])

            if pd.isna(current_score) or pd.isna(prev_score):
                return signals

            stock_code = data.attrs.get("stock_code", "UNKNOWN")

            if prev_score <= 0 and current_score > 0:
                strength = min(1.0, abs(current_score))
                signal = TradingSignal(
                    timestamp=current_date,
                    stock_code=stock_code,
                    signal_type=SignalType.BUY,
                    strength=strength,
                    price=current_price,
                    reason=f"多因子综合评分转正: {current_score:.3f}",
                    metadata={
                        "combined_score": current_score,
                        "value_score": float(
                            indicators["value_score"].iloc[current_idx]
                        )
                        if not pd.isna(indicators["value_score"].iloc[current_idx])
                        else 0,
                        "momentum_score": float(
                            indicators["momentum_score"].iloc[current_idx]
                        )
                        if not pd.isna(indicators["momentum_score"].iloc[current_idx])
                        else 0,
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
                    reason=f"多因子综合评分转负: {current_score:.3f}",
                    metadata={
                        "combined_score": current_score,
                        "value_score": float(
                            indicators["value_score"].iloc[current_idx]
                        )
                        if not pd.isna(indicators["value_score"].iloc[current_idx])
                        else 0,
                        "momentum_score": float(
                            indicators["momentum_score"].iloc[current_idx]
                        )
                        if not pd.isna(indicators["momentum_score"].iloc[current_idx])
                        else 0,
                    },
                )
                signals.append(signal)

            return signals

        except Exception as e:
            logger.error(f"多因子策略信号生成失败: {e}")
            return []
