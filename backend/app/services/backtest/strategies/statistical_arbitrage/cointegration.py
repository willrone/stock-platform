"""
协整策略
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from ...models import SignalType, TradingSignal
from ..base.statistical_arbitrage_base import StatisticalArbitrageStrategy


class CointegrationStrategy(StatisticalArbitrageStrategy):
    """协整策略"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__("Cointegration", config)
        self.lookback_period = config.get("lookback_period", 60)
        self.half_life = config.get("half_life", 20)
        self.entry_threshold = config.get("entry_threshold", 2.0)
        self.exit_threshold = config.get("exit_threshold", 0.5)

    def precompute_all_signals(self, data: pd.DataFrame) -> Optional[pd.Series]:
        """[性能优化] 向量化预计算全量协整信号（按 Z-score 穿越阈值生成买卖信号）"""
        try:
            indicators = self.get_cached_indicators(data)
            zscore = indicators.get("zscore")
            if zscore is None:
                return None

            prev_z = zscore.shift(1)

            # 均值回归强度（当前实现为全局常数 Series）
            mean_rev = indicators.get("mean_reversion_strength")
            if mean_rev is None:
                return None

            # 与日线版本保持一致：
            # prev_z <= -entry 且 z > -entry -> BUY
            # prev_z >=  entry 且 z <  entry -> SELL
            buy_mask = (prev_z <= -self.entry_threshold) & (
                zscore > -self.entry_threshold
            )
            sell_mask = (prev_z >= self.entry_threshold) & (
                zscore < self.entry_threshold
            )

            # 只有在均值回归为负（beta[1]<0）时启用（当前 mean_rev 为负常数/0）
            buy_mask &= mean_rev < 0
            sell_mask &= mean_rev < 0

            # 数据不足：前 lookback_period 天不产生信号
            if len(data.index) > 0:
                idx = np.arange(len(data.index))
                enough_data = idx >= self.lookback_period
                # enough_data 是 ndarray，需要对齐到 index
                enough_data = pd.Series(enough_data, index=data.index)
                buy_mask &= enough_data
                sell_mask &= enough_data

            signals = pd.Series(
                [None] * len(data.index), index=data.index, dtype=object
            )
            signals[buy_mask.fillna(False)] = SignalType.BUY
            signals[sell_mask.fillna(False)] = SignalType.SELL
            return signals
        except Exception as e:
            logger.error(f"Cointegration策略向量化计算失败: {e}")
            return None

    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """计算协整指标（优先复用 DataLoader 预计算列，避免重复 rolling）"""
        close_prices = data["close"]

        returns = close_prices.pct_change().dropna()
        half_life = self._estimate_half_life(returns)

        pre = self._extract_indicators_from_precomputed(
            data,
            {
                "sma": f"MA{self.lookback_period}",
                "std": f"STD{self.lookback_period}",
            },
        )
        if pre is not None and "sma" in pre and "std" in pre:
            sma = pre["sma"]
            std = pre["std"]
        else:
            sma = close_prices.rolling(window=self.lookback_period).mean()
            std = close_prices.rolling(window=self.lookback_period).std()

        zscore = (close_prices - sma) / (std + 0.001)

        mean_reversion_strength = pd.Series(0.0, index=close_prices.index)
        if half_life > 0:
            mean_reversion_strength = pd.Series(
                -np.log(2) / half_life, index=close_prices.index
            )

        return {
            "price": close_prices,
            "returns": returns,
            "half_life": half_life,
            "sma": sma,
            "std": std,
            "zscore": zscore,
            "mean_reversion_strength": mean_reversion_strength,
        }

    def _estimate_half_life(self, returns: pd.Series) -> float:
        """估计半衰期"""
        try:
            lag_returns = returns.shift(1)
            window = min(len(returns) - 1, 252)

            recent_returns = returns.iloc[-window:]
            recent_lag = lag_returns.iloc[-window:]

            valid_idx = ~(recent_returns.isna() | recent_lag.isna())
            if valid_idx.sum() < 10:
                return self.half_life

            X = recent_lag[valid_idx].values.reshape(-1, 1)
            y = recent_returns[valid_idx].values

            X_with_const = np.column_stack([np.ones(len(X)), X])
            beta = np.linalg.lstsq(X_with_const, y, rcond=None)[0]

            if beta[1] >= 0:
                return self.half_life

            half_life = -np.log(2) / beta[1]
            return max(1, min(half_life, 252))

        except Exception as e:
            logger.warning(f"半衰期估计失败: {e}")
            return self.half_life

    def generate_signals(
        self, data: pd.DataFrame, current_date: datetime
    ) -> List[TradingSignal]:
        """生成协整交易信号"""
        # 性能优化：优先使用全量预计算信号
        try:
            precomputed = data.attrs.get("_precomputed_signals", {}).get(self.name)
            if precomputed is not None:
                sig_type = precomputed.get(current_date)
                if isinstance(sig_type, SignalType):
                    indicators = self.get_cached_indicators(data)
                    current_idx = self._get_current_idx(data, current_date)
                    stock_code = data.attrs.get("stock_code", "UNKNOWN")
                    current_price = indicators["price"].iloc[current_idx]
                    current_zscore = indicators["zscore"].iloc[current_idx]
                    half_life = indicators.get("half_life")
                    mean_reversion = indicators["mean_reversion_strength"].iloc[
                        current_idx
                    ]

                    return [
                        TradingSignal(
                            timestamp=current_date,
                            stock_code=stock_code,
                            signal_type=sig_type,
                            strength=min(
                                1.0, abs(current_zscore) / self.entry_threshold
                            )
                            if self.entry_threshold
                            else 0.8,
                            price=current_price,
                            reason=f"[向量化] 协整信号, Z-score: {current_zscore:.2f}, 半衰期: {float(half_life):.1f}",
                            metadata={
                                "zscore": float(current_zscore),
                                "half_life": float(half_life)
                                if half_life is not None
                                else None,
                                "mean_reversion_strength": float(mean_reversion),
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
            half_life = indicators["half_life"]
            mean_reversion = indicators["mean_reversion_strength"].iloc[current_idx]

            stock_code = data.attrs.get("stock_code", "UNKNOWN")

            if (
                prev_zscore <= -self.entry_threshold
                and current_zscore > -self.entry_threshold
            ):
                if mean_reversion < 0:
                    strength = min(1.0, abs(current_zscore) / self.entry_threshold)
                    signal = TradingSignal(
                        timestamp=current_date,
                        stock_code=stock_code,
                        signal_type=SignalType.BUY,
                        strength=strength,
                        price=current_price,
                        reason=f"协整信���：价格回归均衡，Z-score: {current_zscore:.2f}，半衰期: {half_life:.1f}",
                        metadata={
                            "zscore": current_zscore,
                            "half_life": half_life,
                            "mean_reversion_strength": mean_reversion,
                            "sma": indicators["sma"].iloc[current_idx],
                        },
                    )
                    signals.append(signal)

            elif (
                prev_zscore >= self.entry_threshold
                and current_zscore < self.entry_threshold
            ):
                if mean_reversion < 0:
                    strength = min(1.0, abs(current_zscore) / self.entry_threshold)
                    signal = TradingSignal(
                        timestamp=current_date,
                        stock_code=stock_code,
                        signal_type=SignalType.SELL,
                        strength=strength,
                        price=current_price,
                        reason=f"协整信号：价格偏离均衡，Z-score: {current_zscore:.2f}，半衰期: {half_life:.1f}",
                        metadata={
                            "zscore": current_zscore,
                            "half_life": half_life,
                            "mean_reversion_strength": mean_reversion,
                            "sma": indicators["sma"].iloc[current_idx],
                        },
                    )
                    signals.append(signal)

            return signals

        except Exception as e:
            logger.error(f"协整策略信号生成失败: {e}")
            return []
