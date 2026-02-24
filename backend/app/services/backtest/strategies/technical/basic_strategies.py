"""
基础策略实现

包含移动平均、RSI、MACD等基础技术分析策略
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

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
        # 降低默认阈值：从 0.02 (2%) 降到 0.005 (0.5%)
        # 原因：实际市场中，金叉/死叉时的 ma_diff 通常小于 1%
        self.signal_threshold = config.get("signal_threshold", 0.005)

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
            signals = pd.Series([None] * len(data.index), index=data.index, dtype=object)
            signals[buy_mask.fillna(False)] = SignalType.BUY
            signals[sell_mask.fillna(False)] = SignalType.SELL
            
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
            precomputed = data.attrs.get("_precomputed_signals", {}).get(self.name)
            if precomputed is not None:
                sig_type = precomputed.get(current_date)
                if isinstance(sig_type, SignalType):
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
        logger.info(f"📌 RSI策略实例化: class={self.__class__.__name__}, module={self.__class__.__module__}, rsi_period={config.get('rsi_period', 14)}")
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
        计算RSI指标 - 性能优化版
        优先级：预计算列 > Numba > TA-Lib > pandas rolling
        """
        import time as _time
        _t0 = _time.perf_counter()
        close_prices = data["close"]
        calc_path = "unknown"

        # 1. 优先复用 data_loader 预计算的 RSI 列（零成本）
        precomputed_col = f"RSI{self.rsi_period}"
        if precomputed_col in data.columns:
            rsi = data[precomputed_col]
            calc_path = f"precomputed({precomputed_col})"
        else:
            # 2. Numba 加速
            try:
                from .numba_indicators import NUMBA_AVAILABLE, rsi_wilder

                if NUMBA_AVAILABLE:
                    rsi_values = rsi_wilder(close_prices.values, self.rsi_period)
                    rsi = pd.Series(rsi_values, index=close_prices.index)
                    calc_path = "numba"
                elif TALIB_AVAILABLE:
                    rsi = pd.Series(
                        talib.RSI(close_prices.values, timeperiod=self.rsi_period),
                        index=close_prices.index,
                    )
                    calc_path = "talib"
                else:
                    delta = close_prices.diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
                    rs = gain / loss
                    rsi = 100 - (100 / (1 + rs))
                    calc_path = "pandas_rolling"
            except Exception as e:
                logger.warning(f"Numba RSI 计算失败，回退到 pandas: {e}")
                delta = close_prices.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                calc_path = "pandas_fallback"

        _elapsed_ms = (_time.perf_counter() - _t0) * 1000
        stock_code = data.attrs.get("stock_code", "?")
        if _elapsed_ms > 5:  # 只记录耗时 >5ms 的
            logger.debug(f"📊 RSI.calculate_indicators [{stock_code}]: path={calc_path}, {_elapsed_ms:.1f}ms")

        return {
            "rsi": rsi,
            "price": close_prices,
        }

    def precompute_all_signals(self, data: pd.DataFrame) -> Optional[pd.Series]:
        """[性能优化] 向量化计算全量RSI信号"""
        try:
            indicators = self.get_cached_indicators(data)
            rsi = indicators["rsi"]
            prev_rsi = rsi.shift(1)

            # 简化版逻辑：从超卖区回升 -> 买入；从超买区回调 -> 卖出
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
        """生成RSI信号 - 简化版（移除复杂的趋势和背离检测）"""
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
            if current_idx < self.rsi_period:
                return signals

            # 需要至少2个数据点来判断RSI穿越
            if current_idx < 1:
                return signals

            current_rsi = indicators["rsi"].iloc[current_idx]
            prev_rsi = indicators["rsi"].iloc[current_idx - 1]
            current_price = indicators["price"].iloc[current_idx]
            stock_code = data.attrs.get("stock_code", "UNKNOWN")

            # 简化逻辑：只保留基本的RSI穿越信号
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

    def precompute_all_signals(self, data: pd.DataFrame) -> Optional[pd.Series]:
        """[性能优化] 向量化计算全量MACD信号"""
        try:
            indicators = self.get_cached_indicators(data)
            macd_hist = indicators["macd_hist"]
            prev_hist = macd_hist.shift(1)

            # 向量化逻辑判断：金叉和死叉
            buy_mask = (prev_hist <= 0) & (macd_hist > 0)
            sell_mask = (prev_hist >= 0) & (macd_hist < 0)

            # 构造全量信号 Series
            signals = pd.Series([None] * len(data.index), index=data.index, dtype=object)
            signals[buy_mask.fillna(False)] = SignalType.BUY
            signals[sell_mask.fillna(False)] = SignalType.SELL

            return signals
        except Exception as e:
            logger.error(f"MACD策略向量化计算失败: {e}")
            return None

    def generate_signals(
        self, data: pd.DataFrame, current_date: datetime
    ) -> List[TradingSignal]:
        """生成MACD信号"""
        # 性能优化：优先��查是否已有全量预计算信号
        try:
            precomputed = data.attrs.get("_precomputed_signals", {}).get(self.name)
            if precomputed is not None:
                sig_type = precomputed.get(current_date)
                if isinstance(sig_type, SignalType):
                    indicators = self.get_cached_indicators(data)
                    current_idx = self._get_current_idx(data, current_date)
                    stock_code = data.attrs.get("stock_code", "UNKNOWN")
                    current_price = indicators["price"].iloc[current_idx]
                    current_hist = indicators["macd_hist"].iloc[current_idx]
                    return [TradingSignal(
                        timestamp=current_date,
                        stock_code=stock_code,
                        signal_type=sig_type,
                        strength=min(1.0, abs(current_hist) * 100),
                        price=current_price,
                        reason=f"[向量化] MACD{'金叉' if sig_type == SignalType.BUY else '死叉'}，柱状图: {current_hist:.4f}",
                        metadata={
                            "macd": indicators["macd"].iloc[current_idx],
                            "macd_signal": indicators["macd_signal"].iloc[current_idx],
                            "macd_hist": current_hist,
                        },
                    )]
                return []
        except Exception:
            pass

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
