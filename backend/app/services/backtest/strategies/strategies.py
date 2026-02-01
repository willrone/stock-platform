"""
量化交易策略集合

包含以下策略类型：
1. 技术分析策略：布林带策略、随机指标策略、CCI策略
2. 统计套利策略：配对交易策略、均值回归策略、协整策略
3. 因子投资策略：价值因子策略、动量因子策略、低波动因子策略、多因子组合策略
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats
from scipy.optimize import minimize

from app.core.error_handler import ErrorSeverity, TaskError

from ..core.base_strategy import BaseStrategy
from ..models import BacktestConfig, Position, SignalType, TradingSignal


class BollingerBandStrategy(BaseStrategy):
    """布林带策略"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__("BollingerBands", config)
        self.period = config.get("period", 20)
        self.std_dev = config.get("std_dev", 2)
        self.entry_threshold = config.get("entry_threshold", 0.02)

    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """计算布林带指标（优先复用 DataLoader 预计算列，避免重复 rolling）"""
        close_prices = data["close"]

        # 尝试从预计算数据提取
        pre = self._extract_indicators_from_precomputed(
            data,
            {
                "sma": f"MA{self.period}",
                "std": f"STD{self.period}",
            },
        )
        if pre is not None and "sma" in pre and "std" in pre:
            sma = pre["sma"]
            std = pre["std"]
        else:
            sma = close_prices.rolling(window=self.period).mean()
            std = close_prices.rolling(window=self.period).std()

        upper_band = sma + (std * self.std_dev)
        lower_band = sma - (std * self.std_dev)

        bandwidth = (upper_band - lower_band) / sma
        percent_b = (close_prices - lower_band) / (upper_band - lower_band)

        return {
            "sma": sma,
            "upper_band": upper_band,
            "lower_band": lower_band,
            "bandwidth": bandwidth,
            "percent_b": percent_b,
            "price": close_prices,
        }

    def precompute_all_signals(self, data: pd.DataFrame) -> Optional[pd.Series]:
        """[性能优化] 向量化计算全量布林带信号"""
        try:
            indicators = self.get_cached_indicators(data)
            pb = indicators["percent_b"]
            prev_pb = pb.shift(1)

            # 向量化逻辑：价格突破下轨 (prev <= 0 and current > 0)
            buy_mask = (prev_pb <= 0) & (pb > 0)
            # 价格突破上轨 (prev >= 1 and current < 1)
            sell_mask = (prev_pb >= 1) & (pb < 1)

            # 用 None 初始化，避免未赋值位置默认为 NaN(float)，导致下游误判为 truthy
            signals = pd.Series([None] * len(data.index), index=data.index, dtype=object)
            signals[buy_mask.fillna(False)] = SignalType.BUY
            signals[sell_mask.fillna(False)] = SignalType.SELL
            return signals
        except Exception as e:
            logger.error(f"布林带策略向量化计算失败: {e}")
            return None

    def generate_signals(
        self, data: pd.DataFrame, current_date: datetime
    ) -> List[TradingSignal]:
        """生成布林带交易信号"""
        # 性能优化：优先使用预计算信号序列
        try:
            precomputed = data.attrs.get("_precomputed_signals", {}).get(id(self))
            if precomputed is not None:
                sig_type = precomputed.get(current_date)
                if isinstance(sig_type, SignalType):
                    indicators = self.get_cached_indicators(data)
                    current_idx = self._get_current_idx(data, current_date)
                    current_price = indicators["price"].iloc[current_idx]
                    return [TradingSignal(
                        timestamp=current_date,
                        stock_code=data.attrs.get("stock_code", "UNKNOWN"),
                        signal_type=sig_type,
                        strength=0.8, # 预计算模式下简化强度
                        price=current_price,
                        reason=f"[向量化] 价格突破轨道, %B: {indicators['percent_b'].iloc[current_idx]:.3f}",
                        metadata={
                            "upper_band": indicators["upper_band"].iloc[current_idx],
                            "lower_band": indicators["lower_band"].iloc[current_idx],
                            "percent_b": indicators["percent_b"].iloc[current_idx],
                        },
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
            upper_band = indicators["upper_band"].iloc[current_idx]
            lower_band = indicators["lower_band"].iloc[current_idx]
            percent_b = indicators["percent_b"].iloc[current_idx]
            prev_percent_b = indicators["percent_b"].iloc[current_idx - 1]

            stock_code = data.attrs.get("stock_code", "UNKNOWN")

            if prev_percent_b <= 0 and percent_b > 0:
                strength = min(1.0, percent_b)
                signal = TradingSignal(
                    timestamp=current_date,
                    stock_code=stock_code,
                    signal_type=SignalType.BUY,
                    strength=strength,
                    price=current_price,
                    reason=f"价格突破下轨，PercentB: {percent_b:.3f}",
                    metadata={
                        "upper_band": upper_band,
                        "lower_band": lower_band,
                        "percent_b": percent_b,
                    },
                )
                signals.append(signal)

            elif prev_percent_b >= 1 and percent_b < 1:
                strength = min(1.0, 1 - percent_b)
                signal = TradingSignal(
                    timestamp=current_date,
                    stock_code=stock_code,
                    signal_type=SignalType.SELL,
                    strength=strength,
                    price=current_price,
                    reason=f"价格突破上轨，PercentB: {percent_b:.3f}",
                    metadata={
                        "upper_band": upper_band,
                        "lower_band": lower_band,
                        "percent_b": percent_b,
                    },
                )
                signals.append(signal)

            return signals

        except Exception as e:
            logger.error(f"布林带策略信号生成失败: {e}")
            return []


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

    def generate_signals(
        self, data: pd.DataFrame, current_date: datetime
    ) -> List[TradingSignal]:
        """生成随机指标交易信号"""
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

    def generate_signals(
        self, data: pd.DataFrame, current_date: datetime
    ) -> List[TradingSignal]:
        """生成CCI交易信号"""
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


class StatisticalArbitrageStrategy(BaseStrategy):
    """统计套利策略基类"""

    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.pairs = config.get("pairs", [])
        self.entry_threshold = config.get("entry_threshold", 2.0)
        self.exit_threshold = config.get("exit_threshold", 0.5)
        self.max_holding_period = config.get("max_holding_period", 60)
        self.stop_loss_threshold = config.get("stop_loss_threshold", 3.0)

    def calculate_spread_zscore(self, spread: pd.Series, window: int = 20) -> pd.Series:
        """计算价差的Z-score"""
        mean = spread.rolling(window=window).mean()
        std = spread.rolling(window=window).std()
        zscore = (spread - mean) / std
        return zscore

    def validate_pair_correlation(
        self, data1: pd.DataFrame, data2: pd.DataFrame, min_corr: float = 0.7
    ) -> float:
        """验证配对的相关性"""
        returns1 = data1["close"].pct_change().dropna()
        returns2 = data2["close"].pct_change().dropna()

        min_len = min(len(returns1), len(returns2))
        correlation = returns1.iloc[-min_len:].corr(returns2.iloc[-min_len:])

        return correlation


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

            relative_strength = momentum_20d - momentum_5d

            if (
                prev_zscore <= -self.zscore_threshold
                and current_zscore > -self.zscore_threshold
            ):
                if relative_strength < -0.02:
                    strength = min(1.0, abs(current_zscore) / self.zscore_threshold)
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
                            "relative_strength": relative_strength,
                        },
                    )
                    signals.append(signal)

            elif (
                prev_zscore >= self.zscore_threshold
                and current_zscore < self.zscore_threshold
            ):
                if relative_strength > 0.02:
                    strength = min(1.0, abs(current_zscore) / self.zscore_threshold)
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
                            "relative_strength": relative_strength,
                        },
                    )
                    signals.append(signal)

            return signals

        except Exception as e:
            logger.error(f"配对交易策略信号生成失败: {e}")
            return []


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

    def generate_signals(
        self, data: pd.DataFrame, current_date: datetime
    ) -> List[TradingSignal]:
        """生成均值回归交易信号"""
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

            if (
                prev_zscore <= -self.zscore_threshold
                and current_zscore > -self.zscore_threshold
            ):
                strength = min(1.0, abs(current_zscore) / self.zscore_threshold)
                signal = TradingSignal(
                    timestamp=current_date,
                    stock_code=stock_code,
                    signal_type=SignalType.BUY,
                    strength=strength,
                    price=current_price,
                    reason=f"价格回归均值，Z-score: {current_zscore:.2f}",
                    metadata={
                        "zscore": current_zscore,
                        "sma": indicators["sma"].iloc[current_idx],
                    },
                )
                signals.append(signal)

            elif (
                prev_zscore >= self.zscore_threshold
                and current_zscore < self.zscore_threshold
            ):
                strength = min(1.0, abs(current_zscore) / self.zscore_threshold)
                signal = TradingSignal(
                    timestamp=current_date,
                    stock_code=stock_code,
                    signal_type=SignalType.SELL,
                    strength=strength,
                    price=current_price,
                    reason=f"价格偏离均值，Z-score: {current_zscore:.2f}",
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
            buy_mask = (prev_z <= -self.entry_threshold) & (zscore > -self.entry_threshold)
            sell_mask = (prev_z >= self.entry_threshold) & (zscore < self.entry_threshold)

            # 只有在均值回归为负（beta[1]<0）时启用（当前 mean_rev 为负常数/0）
            buy_mask &= (mean_rev < 0)
            sell_mask &= (mean_rev < 0)

            # 数据不足：前 lookback_period 天不产生信号
            if len(data.index) > 0:
                idx = np.arange(len(data.index))
                enough_data = idx >= self.lookback_period
                # enough_data 是 ndarray，需要对齐到 index
                enough_data = pd.Series(enough_data, index=data.index)
                buy_mask &= enough_data
                sell_mask &= enough_data

            signals = pd.Series([None] * len(data.index), index=data.index, dtype=object)
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
            precomputed = data.attrs.get("_precomputed_signals", {}).get(id(self))
            if precomputed is not None:
                sig_type = precomputed.get(current_date)
                if isinstance(sig_type, SignalType):
                    indicators = self.get_cached_indicators(data)
                    current_idx = self._get_current_idx(data, current_date)
                    stock_code = data.attrs.get("stock_code", "UNKNOWN")
                    current_price = indicators["price"].iloc[current_idx]
                    current_zscore = indicators["zscore"].iloc[current_idx]
                    half_life = indicators.get("half_life")
                    mean_reversion = indicators["mean_reversion_strength"].iloc[current_idx]

                    return [
                        TradingSignal(
                            timestamp=current_date,
                            stock_code=stock_code,
                            signal_type=sig_type,
                            strength=min(1.0, abs(current_zscore) / self.entry_threshold) if self.entry_threshold else 0.8,
                            price=current_price,
                            reason=f"[向量化] 协整信号, Z-score: {current_zscore:.2f}, 半衰期: {float(half_life):.1f}",
                            metadata={
                                "zscore": float(current_zscore),
                                "half_life": float(half_life) if half_life is not None else None,
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
                        reason=f"协整信号：价格回归均衡，Z-score: {current_zscore:.2f}，半衰期: {half_life:.1f}",
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


class FactorStrategy(BaseStrategy):
    """因子投资策略基类"""

    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.lookback_period = config.get("lookback_period", 252)
        self.rebalance_frequency = config.get("rebalance_frequency", "monthly")
        self.market_cap_neutral = config.get("market_cap_neutral", False)
        self.industry_neutral = config.get("industry_neutral", False)
        self.max_position_size = config.get("max_position_size", 0.05)
        self.min_position_size = config.get("min_position_size", 0.01)

    def normalize_factors(self, factors: pd.DataFrame) -> pd.DataFrame:
        """因子标准化"""
        normalized = factors.copy()

        for col in normalized.columns:
            if normalized[col].std() > 0:
                normalized[col] = (
                    normalized[col] - normalized[col].mean()
                ) / normalized[col].std()

        return normalized

    def apply_neutralization(
        self,
        scores: pd.Series,
        market_caps: Optional[pd.Series] = None,
        industries: Optional[pd.Series] = None,
    ) -> pd.Series:
        """应用中性化处理"""
        if market_caps is None and industries is None:
            return scores

        neutralized = scores.copy()

        if market_caps is not None and self.market_cap_neutral:
            log_mcap = np.log(market_caps)
            valid_idx = neutralized.index.intersection(log_mcap.index)
            if len(valid_idx) > 10:
                X = np.column_stack(
                    [np.ones(len(valid_idx)), log_mcap.loc[valid_idx].values]
                )
                y = neutralized.loc[valid_idx].values
                try:
                    beta = np.linalg.lstsq(X, y, rcond=None)[0]
                    neutralized.loc[valid_idx] = y - X @ beta
                except:
                    pass

        if industries is not None and self.industry_neutral:
            valid_idx = neutralized.index.intersection(industries.index)
            if len(valid_idx) > 10:
                unique_industries = industries.loc[valid_idx].unique()
                for ind in unique_industries:
                    ind_mask = industries.loc[valid_idx] == ind
                    if ind_mask.sum() > 5:
                        neutralized.loc[valid_idx[ind_mask]] -= neutralized.loc[
                            valid_idx[ind_mask]
                        ].mean()

        return neutralized


class ValueFactorStrategy(FactorStrategy):
    """价值因子策略"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__("ValueFactor", config)
        self.pe_weight = config.get("pe_weight", 0.25)
        self.pb_weight = config.get("pb_weight", 0.25)
        self.ps_weight = config.get("ps_weight", 0.25)
        self.ev_ebitda_weight = config.get("ev_ebitda_weight", 0.25)

    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """计算价值因子指标"""
        close_prices = data["close"]

        returns = close_prices.pct_change().dropna()
        volatility = returns.rolling(window=21).std()

        pe_estimate = pd.Series(
            1 / (returns.rolling(window=252).mean() + 0.001), index=close_prices.index
        )
        pe_estimate = pe_estimate.clip(-100, 100).fillna(15)

        pb_estimate = pd.Series(
            1 / (volatility + 0.01) * 0.5 + 1, index=close_prices.index
        )
        pb_estimate = pb_estimate.clip(0.1, 10).fillna(2)

        ps_estimate = pd.Series(
            1 / (volatility + 0.01) * 0.3 + 1.5, index=close_prices.index
        )
        ps_estimate = ps_estimate.clip(0.5, 15).fillna(3)

        ev_ebitda_estimate = pd.Series(
            1 / (volatility + 0.01) * 0.4 + 5, index=close_prices.index
        )
        ev_ebitda_estimate = ev_ebitda_estimate.clip(2, 30).fillna(10)

        value_score = pd.Series(0.0, index=close_prices.index)

        pe_normalized = (pe_estimate - pe_estimate.rolling(window=252).mean()) / (
            pe_estimate.rolling(window=252).std() + 0.01
        )
        pb_normalized = (pb_estimate - pb_estimate.rolling(window=252).mean()) / (
            pb_estimate.rolling(window=252).std() + 0.01
        )
        ps_normalized = (ps_estimate - ps_estimate.rolling(window=252).mean()) / (
            ps_estimate.rolling(window=252).std() + 0.01
        )
        ev_normalized = (
            ev_ebitda_estimate - ev_ebitda_estimate.rolling(window=252).mean()
        ) / (ev_ebitda_estimate.rolling(window=252).std() + 0.01)

        value_score = (
            -pe_normalized * self.pe_weight
            + -pb_normalized * self.pb_weight
            + -ps_normalized * self.ps_weight
            + -ev_normalized * self.ev_ebitda_weight
        )

        return {
            "pe_ratio": pe_estimate,
            "pb_ratio": pb_estimate,
            "ps_ratio": ps_estimate,
            "ev_ebitda": ev_ebitda_estimate,
            "value_score": value_score,
            "price": close_prices,
        }

    def generate_signals(
        self, data: pd.DataFrame, current_date: datetime
    ) -> List[TradingSignal]:
        """价值因子信号生成"""
        signals = []

        try:
            indicators = self.get_cached_indicators(data)

            current_idx = self._get_current_idx(data, current_date)
            if current_idx < 260:
                return signals

            current_price = indicators["price"].iloc[current_idx]
            current_score = indicators["value_score"].iloc[current_idx]
            prev_score = indicators["value_score"].iloc[current_idx - 1]

            stock_code = data.attrs.get("stock_code", "UNKNOWN")

            if prev_score <= 0 and current_score > 0:
                strength = min(1.0, current_score)
                signal = TradingSignal(
                    timestamp=current_date,
                    stock_code=stock_code,
                    signal_type=SignalType.BUY,
                    strength=strength,
                    price=current_price,
                    reason=f"价值因子评分转正: {current_score:.3f}",
                    metadata={
                        "value_score": current_score,
                        "pe_ratio": indicators["pe_ratio"].iloc[current_idx],
                        "pb_ratio": indicators["pb_ratio"].iloc[current_idx],
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
                    reason=f"价值因子评分转负: {current_score:.3f}",
                    metadata={
                        "value_score": current_score,
                        "pe_ratio": indicators["pe_ratio"].iloc[current_idx],
                        "pb_ratio": indicators["pb_ratio"].iloc[current_idx],
                    },
                )
                signals.append(signal)

            return signals

        except Exception as e:
            logger.error(f"价值因子策略信号生成失败: {e}")
            return []


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


class MultiFactorStrategy(FactorStrategy):
    """多因子组合策略"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__("MultiFactor", config)
        self.factors = config.get("factors", ["value", "momentum", "low_volatility"])
        self.factor_weights = config.get("factor_weights", [0.33, 0.33, 0.34])
        self.weighting_method = config.get("weighting_method", "equal")

        # PERF: indicators are expensive (rolling windows). In backtest we call generate_signals
        # every day; without caching this becomes O(T^2). Cache per-stock dataframe.
        import threading

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


class AdvancedStrategyFactory:
    """高级策略工厂"""

    _strategies = {
        # 技术分析策略
        "bollinger": BollingerBandStrategy,
        "stochastic": StochasticStrategy,
        "cci": CCIStrategy,
        # 统计套利策略
        "pairs_trading": PairsTradingStrategy,
        "mean_reversion": MeanReversionStrategy,
        "cointegration": CointegrationStrategy,
        # 因子投资策略
        "value_factor": ValueFactorStrategy,
        "momentum_factor": MomentumFactorStrategy,
        "low_volatility": LowVolatilityStrategy,
        "multi_factor": MultiFactorStrategy,
    }

    @classmethod
    def create_strategy(
        cls, strategy_name: str, config: Dict[str, Any]
    ) -> BaseStrategy:
        """创建策略实例"""
        strategy_name = strategy_name.lower()

        if strategy_name not in cls._strategies:
            raise TaskError(
                message=f"未知的策略类型: {strategy_name}，可用策略: {list(cls._strategies.keys())}",
                severity=ErrorSeverity.MEDIUM,
            )

        strategy_class = cls._strategies[strategy_name]
        return strategy_class(config)

    @classmethod
    def get_available_strategies(cls) -> Dict[str, List[str]]:
        """获取可用策略分类列表"""
        categories = {
            "technical": [],
            "statistical_arbitrage": [],
            "factor_investment": [],
        }

        for name in cls._strategies.keys():
            if name in ["bollinger", "stochastic", "cci"]:
                categories["technical"].append(name)
            elif name in ["pairs_trading", "mean_reversion", "cointegration"]:
                categories["statistical_arbitrage"].append(name)
            elif name in [
                "value_factor",
                "momentum_factor",
                "low_volatility",
                "multi_factor",
            ]:
                categories["factor_investment"].append(name)

        return categories

    @classmethod
    def register_strategy(cls, name: str, strategy_class: type, category: str):
        """注册新策略"""
        cls._strategies[name.lower()] = strategy_class
