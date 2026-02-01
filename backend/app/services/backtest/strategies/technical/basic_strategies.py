"""
基础策略实现

包含移动平均、RSI、MACD等基础技术分析策略
"""

from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from loguru import logger

from ...core.base_strategy import BaseStrategy
from ...models import SignalType, TradingSignal

# 尝试导入talib，如果不存在则使用pandas实现
try:
    import talib

    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    logger.warning("TA-Lib未安装，将使用pandas实现技术指标")


class MovingAverageStrategy(BaseStrategy):
    """移动平均策略"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__("MovingAverage", config)
        self.short_window = config.get("short_window", 5)
        self.long_window = config.get("long_window", 20)
        self.signal_threshold = config.get("signal_threshold", 0.02)

    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """计算移动平均指标"""
        close_prices = data["close"]

        indicators = {
            "sma_short": close_prices.rolling(window=self.short_window).mean(),
            "sma_long": close_prices.rolling(window=self.long_window).mean(),
            "price": close_prices,
        }

        # 计算移动平均差值
        indicators["ma_diff"] = (
            indicators["sma_short"] - indicators["sma_long"]
        ) / indicators["sma_long"]

        return indicators

    def precompute_all_signals(self, data: pd.DataFrame) -> Optional[pd.Series]:
        """[性能优化] 向量化计算全量均线交叉信号"""
        try:
            indicators = self.get_cached_indicators(data)
            ma_diff = indicators["ma_diff"]
            prev_ma_diff = ma_diff.shift(1)

            # 向量化逻辑判断
            buy_mask = (prev_ma_diff <= 0) & (ma_diff > 0) & (abs(ma_diff) > self.signal_threshold)
            sell_mask = (prev_ma_diff >= 0) & (ma_diff < 0) & (abs(ma_diff) > self.signal_threshold)

            # 构造全量信号 Series
            signals = pd.Series(index=data.index, dtype=object)
            signals[buy_mask] = SignalType.BUY
            signals[sell_mask] = SignalType.SELL
            
            return signals
        except Exception as e:
            logger.error(f"MA策略向量化计算失败: {e}")
            return None

    def generate_signals(
        self, data: pd.DataFrame, current_date: datetime
    ) -> List[TradingSignal]:
        """生成移动平均交叉信号"""
        # 性能优化：优先检查是否已有全量预计算信号
        try:
            precomputed = data.attrs.get("_precomputed_signals", {}).get(id(self))
            if precomputed is not None:
                sig_type = precomputed.get(current_date)
                if sig_type:
                    indicators = self.get_cached_indicators(data)
                    current_idx = self._get_current_idx(data, current_date)
                    stock_code = data.attrs.get("stock_code", "UNKNOWN")
                    current_price = indicators["price"].iloc[current_idx]
                    current_ma_diff = indicators["ma_diff"].iloc[current_idx]
                    return [TradingSignal(
                        timestamp=current_date,
                        stock_code=stock_code,
                        signal_type=sig_type,
                        strength=min(1.0, abs(current_ma_diff) * 10),
                        price=current_price,
                        reason=f"[向量化] 均线交叉，差值: {current_ma_diff:.3f}",
                        metadata={
                            "sma_short": indicators["sma_short"].iloc[current_idx],
                            "sma_long": indicators["sma_long"].iloc[current_idx],
                            "ma_diff": current_ma_diff,
                        },
                    )]
                return []
        except Exception:
            pass

        signals = []

        try:
            # 计算指标（按 DataFrame 缓存，避免每个交易日重复计算整段 rolling 指标）
            indicators = self.get_cached_indicators(data)

            # 获取当前数据点（优先使用执行器写入的 attrs 快路径）
            current_idx = self._get_current_idx(data, current_date)
            if current_idx < self.long_window:
                return signals  # 数据不足

            current_price = indicators["price"].iloc[current_idx]
            current_ma_diff = indicators["ma_diff"].iloc[current_idx]
            prev_ma_diff = indicators["ma_diff"].iloc[current_idx - 1]

            stock_code = data.attrs.get("stock_code", "UNKNOWN")

            # 生成买入信号
            if (
                prev_ma_diff <= 0
                and current_ma_diff > 0
                and abs(current_ma_diff) > self.signal_threshold
            ):
                signal = TradingSignal(
                    timestamp=current_date,
                    stock_code=stock_code,
                    signal_type=SignalType.BUY,
                    strength=min(1.0, abs(current_ma_diff) * 10),
                    price=current_price,
                    reason=f"短期均线上穿长期均线，差值: {current_ma_diff:.3f}",
                    metadata={
                        "sma_short": indicators["sma_short"].iloc[current_idx],
                        "sma_long": indicators["sma_long"].iloc[current_idx],
                        "ma_diff": current_ma_diff,
                    },
                )
                signals.append(signal)

            # 生成卖出信号
            elif (
                prev_ma_diff >= 0
                and current_ma_diff < 0
                and abs(current_ma_diff) > self.signal_threshold
            ):
                signal = TradingSignal(
                    timestamp=current_date,
                    stock_code=stock_code,
                    signal_type=SignalType.SELL,
                    strength=min(1.0, abs(current_ma_diff) * 10),
                    price=current_price,
                    reason=f"短期均线下穿长期均线，差值: {current_ma_diff:.3f}",
                    metadata={
                        "sma_short": indicators["sma_short"].iloc[current_idx],
                        "sma_long": indicators["sma_long"].iloc[current_idx],
                        "ma_diff": current_ma_diff,
                    },
                )
                signals.append(signal)

            return signals

        except Exception as e:
            logger.error(f"移动平均策略信号生成失败: {e}")
            return []


class RSIStrategy(BaseStrategy):
    """
    优化的RSI策略 - 基于业界最佳实践

    核心改进：
    1. 趋势对齐：在上升趋势中，只在RSI回调时买入；在下降趋势中，只在RSI反弹时卖出
    2. RSI穿越信号：等待RSI从超买超卖区域穿越回来，而不是仅仅在超买超卖区域就交易
    3. 背离检测：检测价格与RSI的背离作为反转信号
    4. 结合移动平均线判断趋势方向
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__("RSI", config)
        self.rsi_period = config.get("rsi_period", 14)
        self.oversold_threshold = config.get("oversold_threshold", 30)
        self.overbought_threshold = config.get("overbought_threshold", 70)
        # 趋势对齐参数
        self.trend_ma_period = config.get("trend_ma_period", 50)  # 用于判断趋势的均线周期
        self.enable_trend_alignment = config.get("enable_trend_alignment", True)
        self.enable_divergence = config.get("enable_divergence", True)
        self.enable_crossover = config.get("enable_crossover", True)  # 启用RSI穿越信号
        # 趋势对齐的RSI阈值
        self.uptrend_buy_threshold = config.get(
            "uptrend_buy_threshold", 40
        )  # 上升趋势中的买入阈值
        self.downtrend_sell_threshold = config.get(
            "downtrend_sell_threshold", 60
        )  # 下降趋势中的卖出阈值

    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        计算RSI指标及相关指标
        优先从预计算数据提取，如果不可用则现场计算
        """
        # 尝试从预计算数据提取
        indicator_mapping = {
            "rsi": "RSI14",  # 预计算数据中的RSI列名
            "trend_ma": f"MA{self.trend_ma_period}",  # 趋势均线
        }

        precomputed_indicators = self._extract_indicators_from_precomputed(
            data, indicator_mapping
        )
        if precomputed_indicators is not None:
            # 从预计算数据提取成功，补充一些可能需要计算的指标
            close_prices = data["close"]

            # 计算价格变化率（用于背离检测）
            precomputed_indicators["price_change"] = close_prices.pct_change()

            # 如果有成交量数据，加入成交量
            if "volume" in data.columns:
                precomputed_indicators["volume"] = data["volume"]
                # 尝试从预计算数据提取成交量移动平均，如果没有则计算
                if "VOLUME_MA20" in data.columns:
                    precomputed_indicators["volume_ma"] = data["VOLUME_MA20"]
                else:
                    precomputed_indicators["volume_ma"] = (
                        data["volume"].rolling(window=20).mean()
                    )

            return precomputed_indicators

        # Fallback：现场计算
        close_prices = data["close"]

        # 计算RSI
        if TALIB_AVAILABLE:
            rsi = pd.Series(
                talib.RSI(close_prices.values, timeperiod=self.rsi_period),
                index=close_prices.index,
            )
        else:
            # 使用pandas实现RSI（Wilder's smoothing method）
            delta = close_prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

        # 计算趋势均线
        trend_ma = close_prices.rolling(window=self.trend_ma_period).mean()

        # 计算价格变化率（用于背离检测）
        price_change = close_prices.pct_change()

        indicators = {
            "rsi": rsi,
            "price": close_prices,
            "trend_ma": trend_ma,
            "price_change": price_change,
        }

        # 如果有成交量数据，加入成交量
        if "volume" in data.columns:
            indicators["volume"] = data["volume"]
            # 计算成交量移动平均
            indicators["volume_ma"] = data["volume"].rolling(window=20).mean()

        return indicators

    def precompute_all_signals(self, data: pd.DataFrame) -> Optional[pd.Series]:
        """[性能优化] 向量化计算全量RSI信号"""
        try:
            indicators = self.get_cached_indicators(data)
            rsi = indicators["rsi"]
            prev_rsi = rsi.shift(1)

            # 简化版逻辑：从超卖区回升 -> 买入；从超买区回调 -> 卖出
            buy_mask = (prev_rsi <= self.oversold_threshold) & (rsi > self.oversold_threshold)
            sell_mask = (prev_rsi >= self.overbought_threshold) & (rsi < self.overbought_threshold)

            signals = pd.Series(index=data.index, dtype=object)
            signals[buy_mask] = SignalType.BUY
            signals[sell_mask] = SignalType.SELL
            return signals
        except Exception as e:
            logger.error(f"RSI策略向量化计算失败: {e}")
            return None

    def _detect_trend(self, indicators: Dict[str, pd.Series], current_idx: int) -> str:
        """检测当前趋势：'uptrend', 'downtrend', 'sideways'"""
        if current_idx < self.trend_ma_period:
            return "sideways"

        current_price = indicators["price"].iloc[current_idx]
        current_ma = indicators["trend_ma"].iloc[current_idx]

        # 计算最近的价格趋势（使用短期均线）
        if current_idx >= 20:
            short_ma = (
                indicators["price"].iloc[current_idx - 19 : current_idx + 1].mean()
            )
            prev_short_ma = (
                indicators["price"].iloc[current_idx - 20 : current_idx].mean()
            )

            # 价格在趋势均线之上且短期均线上涨 -> 上升趋势
            if current_price > current_ma and short_ma > prev_short_ma:
                return "uptrend"
            # 价格在趋势均线之下且短期均线下跌 -> 下降趋势
            elif current_price < current_ma and short_ma < prev_short_ma:
                return "downtrend"

        # 简单判断：价格相对于趋势均线的位置
        if current_price > current_ma * 1.02:
            return "uptrend"
        elif current_price < current_ma * 0.98:
            return "downtrend"

        return "sideways"

    def _detect_divergence(
        self, indicators: Dict[str, pd.Series], current_idx: int, lookback: int = 20
    ) -> Dict[str, bool]:
        """
        检测背离
        返回: {'bullish_divergence': bool, 'bearish_divergence': bool}
        """
        if current_idx < lookback:
            return {"bullish_divergence": False, "bearish_divergence": False}

        # 获取最近的价格和RSI数据
        price_window = indicators["price"].iloc[
            current_idx - lookback : current_idx + 1
        ]
        rsi_window = indicators["rsi"].iloc[current_idx - lookback : current_idx + 1]

        # 找到价格和RSI的局部极值
        price_lows = []
        price_highs = []
        rsi_lows = []
        rsi_highs = []

        for i in range(2, len(price_window) - 2):
            # 价格低点
            if (
                price_window.iloc[i] < price_window.iloc[i - 1]
                and price_window.iloc[i] < price_window.iloc[i + 1]
                and price_window.iloc[i] < price_window.iloc[i - 2]
                and price_window.iloc[i] < price_window.iloc[i + 2]
            ):
                price_lows.append((i, price_window.iloc[i]))
            # 价格高点
            if (
                price_window.iloc[i] > price_window.iloc[i - 1]
                and price_window.iloc[i] > price_window.iloc[i + 1]
                and price_window.iloc[i] > price_window.iloc[i - 2]
                and price_window.iloc[i] > price_window.iloc[i + 2]
            ):
                price_highs.append((i, price_window.iloc[i]))
            # RSI低点
            if (
                rsi_window.iloc[i] < rsi_window.iloc[i - 1]
                and rsi_window.iloc[i] < rsi_window.iloc[i + 1]
            ):
                rsi_lows.append((i, rsi_window.iloc[i]))
            # RSI高点
            if (
                rsi_window.iloc[i] > rsi_window.iloc[i - 1]
                and rsi_window.iloc[i] > rsi_window.iloc[i + 1]
            ):
                rsi_highs.append((i, rsi_window.iloc[i]))

        # 检测看涨背离：价格创新低，但RSI没有创新低
        bullish_divergence = False
        if len(price_lows) >= 2 and len(rsi_lows) >= 2:
            recent_price_low = price_lows[-1][1]
            prev_price_low = price_lows[-2][1]
            recent_rsi_low = rsi_lows[-1][1]
            prev_rsi_low = rsi_lows[-2][1]

            if recent_price_low < prev_price_low and recent_rsi_low > prev_rsi_low:
                bullish_divergence = True

        # 检测看跌背离：价格创新高，但RSI没有创新高
        bearish_divergence = False
        if len(price_highs) >= 2 and len(rsi_highs) >= 2:
            recent_price_high = price_highs[-1][1]
            prev_price_high = price_highs[-2][1]
            recent_rsi_high = rsi_highs[-1][1]
            prev_rsi_high = rsi_highs[-2][1]

            if recent_price_high > prev_price_high and recent_rsi_high < prev_rsi_high:
                bearish_divergence = True

        return {
            "bullish_divergence": bullish_divergence,
            "bearish_divergence": bearish_divergence,
        }

    def _check_volume_confirmation(
        self, indicators: Dict[str, pd.Series], current_idx: int
    ) -> bool:
        """检查成交量确认"""
        if "volume" not in indicators or "volume_ma" not in indicators:
            return True  # 如果没有成交量数据，默认通过

        if current_idx < 20:
            return True

        current_volume = indicators["volume"].iloc[current_idx]
        volume_ma = indicators["volume_ma"].iloc[current_idx]

        # 成交量高于平均值视为确认
        return current_volume > volume_ma * 0.8

    def generate_signals(
        self, data: pd.DataFrame, current_date: datetime
    ) -> List[TradingSignal]:
        """生成优化的RSI信号"""
        # 性能优化：优先检查是否已有全量预计算信号
        try:
            precomputed = data.attrs.get("_precomputed_signals", {}).get(id(self))
            if precomputed is not None:
                sig_type = precomputed.get(current_date)
                if sig_type:
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
                        reason=f"[向量化] RSI信号, RSI: {current_rsi:.2f}",
                        metadata={"rsi": current_rsi},
                    )]
                return []
        except Exception:
            pass

        signals = []

        try:
            # 计算指标（按 DataFrame 缓存）
            indicators = self.get_cached_indicators(data)

            current_idx = self._get_current_idx(data, current_date)
            if current_idx < max(self.rsi_period, self.trend_ma_period):
                return signals

            # 需要至少2个数据点来判断RSI穿越
            if current_idx < 1:
                return signals

            current_rsi = indicators["rsi"].iloc[current_idx]
            prev_rsi = indicators["rsi"].iloc[current_idx - 1]
            current_price = indicators["price"].iloc[current_idx]
            stock_code = data.attrs.get("stock_code", "UNKNOWN")

            # 检测趋势
            trend = (
                self._detect_trend(indicators, current_idx)
                if self.enable_trend_alignment
                else "sideways"
            )

            # 检测背离
            divergence = (
                self._detect_divergence(indicators, current_idx)
                if self.enable_divergence
                else {"bullish_divergence": False, "bearish_divergence": False}
            )

            # 检查成交量确认
            volume_confirmed = self._check_volume_confirmation(indicators, current_idx)

            # 1. RSI穿越信号（从超卖区域向上穿越）
            if self.enable_crossover:
                # 买入信号：RSI从超卖区域（<30）向上穿越30或40
                if (
                    prev_rsi < self.oversold_threshold
                    and current_rsi >= self.oversold_threshold
                ):
                    # 趋势对齐：在上升趋势中，等待RSI回调到40-50再买入
                    if self.enable_trend_alignment and trend == "uptrend":
                        if (
                            current_rsi >= self.uptrend_buy_threshold
                            and current_rsi <= 55
                        ):
                            strength = min(
                                1.0, (current_rsi - self.oversold_threshold) / 20
                            )
                            if divergence["bullish_divergence"]:
                                strength = min(1.0, strength * 1.2)  # 背离增强信号强度

                            signal = TradingSignal(
                                timestamp=current_date,
                                stock_code=stock_code,
                                signal_type=SignalType.BUY,
                                strength=min(1.0, strength),
                                price=current_price,
                                reason=f"RSI从超卖区域向上穿越({prev_rsi:.2f}->{current_rsi:.2f})，上升趋势回调买入",
                                metadata={
                                    "rsi": current_rsi,
                                    "prev_rsi": prev_rsi,
                                    "trend": trend,
                                    "divergence": divergence["bullish_divergence"],
                                },
                            )
                            signals.append(signal)
                    # 非趋势对齐或横盘：传统超卖反弹
                    elif trend != "downtrend":  # 避免在下降趋势中买入
                        strength = min(
                            1.0, (current_rsi - self.oversold_threshold) / 20
                        )
                        if divergence["bullish_divergence"]:
                            strength = min(1.0, strength * 1.3)

                        signal = TradingSignal(
                            timestamp=current_date,
                            stock_code=stock_code,
                            signal_type=SignalType.BUY,
                            strength=min(1.0, strength),
                            price=current_price,
                            reason=f"RSI从超卖区域向上穿越({prev_rsi:.2f}->{current_rsi:.2f})"
                            + ("，看涨背离" if divergence["bullish_divergence"] else ""),
                            metadata={
                                "rsi": current_rsi,
                                "prev_rsi": prev_rsi,
                                "trend": trend,
                                "divergence": divergence["bullish_divergence"],
                            },
                        )
                        signals.append(signal)

                # 卖出信号：RSI从超买区域（>70）向下穿越70或60
                elif (
                    prev_rsi > self.overbought_threshold
                    and current_rsi <= self.overbought_threshold
                ):
                    # 趋势对齐：在下降趋势中，等待RSI反弹到50-60再卖出
                    if self.enable_trend_alignment and trend == "downtrend":
                        if (
                            current_rsi <= self.downtrend_sell_threshold
                            and current_rsi >= 45
                        ):
                            strength = min(
                                1.0, (self.overbought_threshold - current_rsi) / 20
                            )
                            if divergence["bearish_divergence"]:
                                strength = min(1.0, strength * 1.2)

                            signal = TradingSignal(
                                timestamp=current_date,
                                stock_code=stock_code,
                                signal_type=SignalType.SELL,
                                strength=min(1.0, strength),
                                price=current_price,
                                reason=f"RSI从超买区域向下穿越({prev_rsi:.2f}->{current_rsi:.2f})，下降趋势反弹卖出",
                                metadata={
                                    "rsi": current_rsi,
                                    "prev_rsi": prev_rsi,
                                    "trend": trend,
                                    "divergence": divergence["bearish_divergence"],
                                },
                            )
                            signals.append(signal)
                    # 非趋势对齐或横盘：传统超买回调
                    elif trend != "uptrend":  # 避免在上升趋势中卖出
                        strength = min(
                            1.0, (self.overbought_threshold - current_rsi) / 20
                        )
                        if divergence["bearish_divergence"]:
                            strength = min(1.0, strength * 1.3)

                        signal = TradingSignal(
                            timestamp=current_date,
                            stock_code=stock_code,
                            signal_type=SignalType.SELL,
                            strength=min(1.0, strength),
                            price=current_price,
                            reason=f"RSI从超买区域向下穿越({prev_rsi:.2f}->{current_rsi:.2f})"
                            + ("，看跌背离" if divergence["bearish_divergence"] else ""),
                            metadata={
                                "rsi": current_rsi,
                                "prev_rsi": prev_rsi,
                                "trend": trend,
                                "divergence": divergence["bearish_divergence"],
                            },
                        )
                        signals.append(signal)

            # 2. 背离信号（如果未启用穿越信号，或作为补充）
            if self.enable_divergence:
                # 看涨背离买入信号
                if (
                    divergence["bullish_divergence"]
                    and current_rsi < 50
                    and trend != "downtrend"
                    and volume_confirmed
                ):
                    strength = min(1.0, (50 - current_rsi) / 50 * 1.2)  # 背离信号增强
                    signal = TradingSignal(
                        timestamp=current_date,
                        stock_code=stock_code,
                        signal_type=SignalType.BUY,
                        strength=min(1.0, strength),
                        price=current_price,
                        reason=f"看涨背离，RSI: {current_rsi:.2f}",
                        metadata={
                            "rsi": current_rsi,
                            "trend": trend,
                            "divergence": True,
                        },
                    )
                    signals.append(signal)

                # 看跌背离卖出信号
                elif (
                    divergence["bearish_divergence"]
                    and current_rsi > 50
                    and trend != "uptrend"
                    and volume_confirmed
                ):
                    strength = min(1.0, (current_rsi - 50) / 50 * 1.2)  # 背离信号增强
                    signal = TradingSignal(
                        timestamp=current_date,
                        stock_code=stock_code,
                        signal_type=SignalType.SELL,
                        strength=min(1.0, strength),
                        price=current_price,
                        reason=f"看跌背离，RSI: {current_rsi:.2f}",
                        metadata={
                            "rsi": current_rsi,
                            "trend": trend,
                            "divergence": True,
                        },
                    )
                    signals.append(signal)

            return signals

        except Exception as e:
            logger.error(f"RSI策略信号生成失败: {e}")
            return []


class MACDStrategy(BaseStrategy):
    """MACD策略"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__("MACD", config)
        self.fast_period = config.get("fast_period", 12)
        self.slow_period = config.get("slow_period", 26)
        self.signal_period = config.get("signal_period", 9)

    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """计算MACD指标"""
        close_prices = data["close"]

        # 使用talib或pandas计算MACD
        if TALIB_AVAILABLE:
            macd, macd_signal, macd_hist = talib.MACD(
                close_prices.values,
                fastperiod=self.fast_period,
                slowperiod=self.slow_period,
                signalperiod=self.signal_period,
            )
            macd = pd.Series(macd, index=close_prices.index)
            macd_signal = pd.Series(macd_signal, index=close_prices.index)
            macd_hist = pd.Series(macd_hist, index=close_prices.index)
        else:
            # 使用pandas实现MACD
            ema_fast = close_prices.ewm(span=self.fast_period, adjust=False).mean()
            ema_slow = close_prices.ewm(span=self.slow_period, adjust=False).mean()
            macd = ema_fast - ema_slow
            macd_signal = macd.ewm(span=self.signal_period, adjust=False).mean()
            macd_hist = macd - macd_signal

        return {
            "macd": macd,
            "macd_signal": macd_signal,
            "macd_hist": macd_hist,
            "price": close_prices,
        }

    def generate_signals(
        self, data: pd.DataFrame, current_date: datetime
    ) -> List[TradingSignal]:
        """生成MACD信号"""
        signals = []

        try:
            indicators = self.get_cached_indicators(data)

            current_idx = self._get_current_idx(data, current_date)
            if current_idx < self.slow_period + self.signal_period:
                return signals

            current_hist = indicators["macd_hist"].iloc[current_idx]
            prev_hist = indicators["macd_hist"].iloc[current_idx - 1]
            current_price = indicators["price"].iloc[current_idx]
            stock_code = data.attrs.get("stock_code", "UNKNOWN")

            # MACD金叉信号（买入）
            if prev_hist <= 0 and current_hist > 0:
                signal = TradingSignal(
                    timestamp=current_date,
                    stock_code=stock_code,
                    signal_type=SignalType.BUY,
                    strength=min(1.0, abs(current_hist) * 100),
                    price=current_price,
                    reason=f"MACD金叉，柱状图: {current_hist:.4f}",
                    metadata={
                        "macd": indicators["macd"].iloc[current_idx],
                        "macd_signal": indicators["macd_signal"].iloc[current_idx],
                        "macd_hist": current_hist,
                    },
                )
                signals.append(signal)

            # MACD死叉信号（卖出）
            elif prev_hist >= 0 and current_hist < 0:
                signal = TradingSignal(
                    timestamp=current_date,
                    stock_code=stock_code,
                    signal_type=SignalType.SELL,
                    strength=min(1.0, abs(current_hist) * 100),
                    price=current_price,
                    reason=f"MACD死叉，柱状图: {current_hist:.4f}",
                    metadata={
                        "macd": indicators["macd"].iloc[current_idx],
                        "macd_signal": indicators["macd_signal"].iloc[current_idx],
                        "macd_hist": current_hist,
                    },
                )
                signals.append(signal)

            return signals

        except Exception as e:
            logger.error(f"MACD策略信号生成失败: {e}")
            return []
