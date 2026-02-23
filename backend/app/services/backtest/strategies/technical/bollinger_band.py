"""
布林带策略
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger

from ...core.base_strategy import BaseStrategy
from ...models import SignalType, TradingSignal


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
            bw = indicators["bandwidth"]

            # 买入：价格跌破下轨（%B < 0）且前一天还在轨内（prev >= 0），带宽足够（避免窄幅震荡）
            buy_mask = (pb < 0) & (prev_pb >= 0) & (bw > 0.02)
            # 卖出：价格突破上轨（%B > 1）且前一天还在轨内（prev <= 1）
            sell_mask = (pb > 1) & (prev_pb <= 1) & (bw > 0.02)

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
            precomputed = data.attrs.get("_precomputed_signals", {}).get(self.name)
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
            bandwidth = indicators["bandwidth"].iloc[current_idx]

            stock_code = data.attrs.get("stock_code", "UNKNOWN")

            # 带宽过窄时不交易（窄幅震荡，信号不可靠）
            if bandwidth <= 0.02:
                return signals

            # 买入：价格跌破下轨（抄底）
            if percent_b < 0 and prev_percent_b >= 0:
                strength = min(1.0, abs(percent_b))
                signal = TradingSignal(
                    timestamp=current_date,
                    stock_code=stock_code,
                    signal_type=SignalType.BUY,
                    strength=strength,
                    price=current_price,
                    reason=f"价格跌破下轨，PercentB: {percent_b:.3f}",
                    metadata={
                        "upper_band": upper_band,
                        "lower_band": lower_band,
                        "percent_b": percent_b,
                        "bandwidth": bandwidth,
                    },
                )
                signals.append(signal)

            # 卖出：价格突破上轨
            elif percent_b > 1 and prev_percent_b <= 1:
                strength = min(1.0, percent_b - 1)
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
                        "bandwidth": bandwidth,
                    },
                )
                signals.append(signal)

            return signals

        except Exception as e:
            logger.error(f"布林带策略信号生成失败: {e}")
            return []
