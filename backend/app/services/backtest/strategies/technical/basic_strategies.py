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
        # 趋势过滤：MA60 方向一致时才触发信号
        self.trend_window = config.get("trend_window", 60)
        self.enable_trend_filter = config.get("enable_trend_filter", True)

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
        """[性能优化] 向量化计算全量均线交叉信号（返回浮点强度，含趋势过滤）

        返回值：正数=买入信号（强度），负数=卖出信号（强度），0=无信号
        强度 = ma_diff * 10，clamp 到 [-1, 1]
        趋势过滤：MA60 向上时才允许金叉买入，MA60 向下时才允许死叉卖出
        """
        try:
            indicators = self.get_cached_indicators(data)
            ma_diff = indicators["ma_diff"]
            prev_ma_diff = ma_diff.shift(1)
            close = indicators["price"]

            # 向量化逻辑判断
            buy_mask = (
                (prev_ma_diff <= 0)
                & (ma_diff > 0)
                & (abs(ma_diff) > self.signal_threshold)
            )
            sell_mask = (
                (prev_ma_diff >= 0)
                & (ma_diff < 0)
                & (abs(ma_diff) > self.signal_threshold)
            )

            # 趋势过滤：MA60 方向一致时才触发信号
            if self.enable_trend_filter:
                trend_ma = close.rolling(window=self.trend_window).mean()
                trend_up = trend_ma > trend_ma.shift(5)  # MA60 向上
                trend_down = trend_ma < trend_ma.shift(5)  # MA60 向下
                buy_mask = buy_mask & trend_up.fillna(False)
                sell_mask = sell_mask & trend_down.fillna(False)

            # 构造浮点强度信号 Series：正=买入，负=卖出，0=无信号
            signals = pd.Series(0.0, index=data.index, dtype=np.float64)
            # 买入强度：ma_diff * 10，clamp 到 (0, 1]
            signals[buy_mask.fillna(False)] = (
                ma_diff[buy_mask.fillna(False)] * 10
            ).clip(0.3, 1.0)
            # 卖出强度：ma_diff * 10（负值），clamp 到 [-1, -0.3)
            signals[sell_mask.fillna(False)] = (
                ma_diff[sell_mask.fillna(False)] * 10
            ).clip(-1.0, -0.3)

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
                sig_val = precomputed.get(current_date)
                # 支持浮点信号（正=买入，负=卖出）和旧枚举格式
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
                    current_ma_diff = indicators["ma_diff"].iloc[current_idx]
                    return [
                        TradingSignal(
                            timestamp=current_date,
                            stock_code=stock_code,
                            signal_type=sig_type,
                            strength=min(1.0, abs(fv)),
                            price=current_price,
                            reason=f"[向量化] 均线交叉，差值: {current_ma_diff:.3f}",
                            metadata={
                                "sma_short": indicators["sma_short"].iloc[current_idx],
                                "sma_long": indicators["sma_long"].iloc[current_idx],
                                "ma_diff": current_ma_diff,
                            },
                        )
                    ]
                elif isinstance(sig_val, SignalType):
                    indicators = self.get_cached_indicators(data)
                    current_idx = self._get_current_idx(data, current_date)
                    stock_code = data.attrs.get("stock_code", "UNKNOWN")
                    current_price = indicators["price"].iloc[current_idx]
                    current_ma_diff = indicators["ma_diff"].iloc[current_idx]
                    return [
                        TradingSignal(
                            timestamp=current_date,
                            stock_code=stock_code,
                            signal_type=sig_val,
                            strength=min(1.0, abs(current_ma_diff) * 10),
                            price=current_price,
                            reason=f"[向量化] 均线交叉，差值: {current_ma_diff:.3f}",
                            metadata={
                                "sma_short": indicators["sma_short"].iloc[current_idx],
                                "sma_long": indicators["sma_long"].iloc[current_idx],
                                "ma_diff": current_ma_diff,
                            },
                        )
                    ]
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
    优化的RSI策略 v2 - P0~P3 全面升级

    核心改进：
    1. P0: generate_signals 与 precompute_all_signals 逻辑统一（共用 _check_filters）
    2. P1: 趋势对齐（MA50）+ 成交量确认
    3. P2: ATR 动态止损止盈 + 动态信号强度
    4. P3: 多周期 RSI 共振（RSI7 + RSI21）
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__("RSI", config)
        self.rsi_period = config.get("rsi_period", 14)
        self.oversold_threshold = config.get("oversold_threshold", 30)
        self.overbought_threshold = config.get("overbought_threshold", 70)
        # P1: 趋势对齐参数
        self.trend_ma_period = config.get("trend_ma_period", 50)
        self.enable_trend_alignment = config.get("enable_trend_alignment", True)
        # P1: 成交量确认参数
        self.enable_volume_confirm = config.get("enable_volume_confirm", False)
        self.volume_threshold = config.get("volume_threshold", 0.8)
        self.volume_ma_period = config.get("volume_ma_period", 20)
        # P2: ATR 参数
        self.atr_period = config.get("atr_period", 14)
        self.atr_stop_multiplier = config.get("atr_stop_multiplier", 2.0)
        self.atr_profit_multiplier = config.get("atr_profit_multiplier", 3.0)
        # P3: 多周期 RSI 参数
        self.enable_multi_rsi = config.get("enable_multi_rsi", False)
        self.rsi_period_short = config.get("rsi_period_short", 7)
        self.rsi_period_long = config.get("rsi_period_long", 21)
        self.long_rsi_oversold = config.get("long_rsi_oversold", 50)
        self.long_rsi_overbought = config.get("long_rsi_overbought", 50)
        # 兼容旧参数（不再���用但避免报错）
        self.enable_divergence = config.get("enable_divergence", False)
        self.enable_crossover = config.get("enable_crossover", True)
        self.uptrend_buy_threshold = config.get("uptrend_buy_threshold", 40)
        self.downtrend_sell_threshold = config.get("downtrend_sell_threshold", 60)

    def _calc_rsi_series(self, close_prices: pd.Series, period: int) -> pd.Series:
        """计算 RSI，支持 Numba/talib/pandas 三级 fallback"""
        try:
            from .numba_indicators import NUMBA_AVAILABLE, rsi_wilder

            if NUMBA_AVAILABLE:
                return pd.Series(rsi_wilder(close_prices.values, period), index=close_prices.index)
        except Exception:
            pass

        if TALIB_AVAILABLE:
            return pd.Series(talib.RSI(close_prices.values, timeperiod=period), index=close_prices.index)

        # pandas fallback
        delta = close_prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """计算全部指标：RSI（多周期）、趋势、成交量、ATR"""
        close_prices = data["close"]

        # 主 RSI
        rsi = self._calc_rsi_series(close_prices, self.rsi_period)

        indicators = {
            "rsi": rsi,
            "price": close_prices,
        }

        # P3: 多周期 RSI
        if self.enable_multi_rsi:
            indicators["rsi_short"] = self._calc_rsi_series(close_prices, self.rsi_period_short)
            indicators["rsi_long"] = self._calc_rsi_series(close_prices, self.rsi_period_long)

        # P1: 趋势判断（用 MA 斜率而非价格位置，避免与超卖/超买矛盾）
        if self.enable_trend_alignment:
            trend_ma = close_prices.rolling(window=self.trend_ma_period).mean()
            indicators["uptrend"] = trend_ma > trend_ma.shift(5)  # MA50 斜率向上
            indicators["downtrend"] = trend_ma < trend_ma.shift(5)  # MA50 斜率向下

        # P1: 成交量确认
        if self.enable_volume_confirm and "volume" in data.columns:
            volume = data["volume"]
            vol_ma = volume.rolling(window=self.volume_ma_period).mean()
            indicators["volume_confirm"] = volume > vol_ma * self.volume_threshold
        else:
            indicators["volume_confirm"] = pd.Series(True, index=close_prices.index)

        # P2: ATR
        if "high" in data.columns and "low" in data.columns:
            high = data["high"]
            low = data["low"]
            prev_close = close_prices.shift(1)
            tr = pd.concat([
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ], axis=1).max(axis=1)
            indicators["atr"] = tr.rolling(window=self.atr_period).mean()
        else:
            indicators["atr"] = pd.Series(0.0, index=close_prices.index)

        return indicators

    def _calc_buy_strength(self, prev_rsi_val: float) -> float:
        """动态买入强度：超卖越深强度越高"""
        strength = (self.oversold_threshold - prev_rsi_val) / self.oversold_threshold
        return float(np.clip(strength, 0.3, 1.0))

    def _calc_sell_strength(self, prev_rsi_val: float) -> float:
        """动态卖出强度：超买越高强度越高"""
        strength = (prev_rsi_val - self.overbought_threshold) / (100 - self.overbought_threshold)
        return float(np.clip(strength, 0.3, 1.0))

    def _check_filters_scalar(self, indicators: Dict, idx: int, signal_type: str) -> bool:
        """P0: 统一过滤逻辑（标量版，供 generate_signals fallback 使用）"""
        # P1: 趋势对齐
        if self.enable_trend_alignment:
            if signal_type == "buy" and "uptrend" in indicators:
                if not indicators["uptrend"].iloc[idx]:
                    return False
            if signal_type == "sell" and "downtrend" in indicators:
                if not indicators["downtrend"].iloc[idx]:
                    return False

        # P1: 成交量确认
        if self.enable_volume_confirm and "volume_confirm" in indicators:
            if not indicators["volume_confirm"].iloc[idx]:
                return False

        # P3: 多周期 RSI 共振
        if self.enable_multi_rsi and "rsi_long" in indicators:
            rsi_long_val = indicators["rsi_long"].iloc[idx]
            if not np.isnan(rsi_long_val):
                if signal_type == "buy" and rsi_long_val >= self.long_rsi_oversold:
                    return False
                if signal_type == "sell" and rsi_long_val <= self.long_rsi_overbought:
                    return False

        return True

    def precompute_all_signals(self, data: pd.DataFrame) -> Optional[pd.Series]:
        """[性能优化] 向量化计算全量RSI信号（P0~P3 全部过滤）

        返回值：正数=买入信号（强度），负数=卖出信号（强度），0=无信号
        """
        try:
            indicators = self.get_cached_indicators(data)
            rsi = indicators["rsi"]
            prev_rsi = rsi.shift(1)

            # 基础穿越信号
            buy_mask = (prev_rsi <= self.oversold_threshold) & (rsi > self.oversold_threshold)
            sell_mask = (prev_rsi >= self.overbought_threshold) & (rsi < self.overbought_threshold)

            # P1: 趋势对齐过滤
            if self.enable_trend_alignment and "uptrend" in indicators:
                buy_mask = buy_mask & indicators["uptrend"].fillna(False)
                sell_mask = sell_mask & indicators["downtrend"].fillna(False)

            # P1: 成交量确认
            if self.enable_volume_confirm and "volume_confirm" in indicators:
                vol_confirm = indicators["volume_confirm"].fillna(False)
                buy_mask = buy_mask & vol_confirm
                sell_mask = sell_mask & vol_confirm

            # P3: 多周期 RSI 共振
            if self.enable_multi_rsi and "rsi_long" in indicators:
                rsi_long = indicators["rsi_long"]
                buy_mask = buy_mask & (rsi_long < self.long_rsi_oversold).fillna(False)
                sell_mask = sell_mask & (rsi_long > self.long_rsi_overbought).fillna(False)

            # 构造浮点强度信号
            signals = pd.Series(0.0, index=data.index, dtype=np.float64)
            # 买入强度：超卖越深强度越高
            buy_strength = (
                (self.oversold_threshold - prev_rsi) / self.oversold_threshold
            ).clip(0.3, 1.0)
            signals[buy_mask.fillna(False)] = buy_strength[buy_mask.fillna(False)]
            # 卖出强度（负值）
            sell_strength = -(
                (prev_rsi - self.overbought_threshold) / (100 - self.overbought_threshold)
            ).clip(0.3, 1.0)
            signals[sell_mask.fillna(False)] = sell_strength[sell_mask.fillna(False)]

            return signals
        except Exception as e:
            logger.error(f"RSI策略向量化计算失败: {e}")
            return None

    def _build_signal_metadata(self, indicators: Dict, idx: int, current_price: float, current_rsi: float, prev_rsi: float) -> Dict:
        """P2: 构建信号 metadata，包含 ATR 动态止损止盈"""
        meta = {"rsi": current_rsi, "prev_rsi": prev_rsi}

        # 多周期 RSI 信息
        if self.enable_multi_rsi and "rsi_long" in indicators:
            rsi_long_val = indicators["rsi_long"].iloc[idx]
            if not np.isnan(rsi_long_val):
                meta["rsi_long"] = float(rsi_long_val)
        if self.enable_multi_rsi and "rsi_short" in indicators:
            rsi_short_val = indicators["rsi_short"].iloc[idx]
            if not np.isnan(rsi_short_val):
                meta["rsi_short"] = float(rsi_short_val)

        # ATR 动态止损止盈
        atr_val = indicators.get("atr")
        if atr_val is not None:
            atr = atr_val.iloc[idx]
            if not np.isnan(atr) and atr > 0:
                meta["atr"] = float(atr)
                meta["suggested_stop_loss"] = float(current_price - self.atr_stop_multiplier * atr)
                meta["suggested_take_profit"] = float(current_price + self.atr_profit_multiplier * atr)

        return meta

    def generate_signals(
        self, data: pd.DataFrame, current_date: datetime
    ) -> List[TradingSignal]:
        """生成RSI信号 - P0: 与 precompute_all_signals 逻辑完全一致"""
        # 优先使用预计算信号（向量化快路径）
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
                    current_rsi = indicators["rsi"].iloc[current_idx]
                    prev_rsi = indicators["rsi"].iloc[current_idx - 1] if current_idx > 0 else current_rsi
                    meta = self._build_signal_metadata(indicators, current_idx, current_price, current_rsi, float(prev_rsi))
                    return [
                        TradingSignal(
                            timestamp=current_date,
                            stock_code=stock_code,
                            signal_type=sig_type,
                            strength=min(1.0, abs(fv)),
                            price=current_price,
                            reason=f"[向量化] RSI信号, RSI: {current_rsi:.2f}",
                            metadata=meta,
                        )
                    ]
                elif isinstance(sig_val, SignalType):
                    indicators = self.get_cached_indicators(data)
                    current_idx = self._get_current_idx(data, current_date)
                    stock_code = data.attrs.get("stock_code", "UNKNOWN")
                    current_price = indicators["price"].iloc[current_idx]
                    current_rsi = indicators["rsi"].iloc[current_idx]
                    prev_rsi = indicators["rsi"].iloc[current_idx - 1] if current_idx > 0 else current_rsi
                    strength = self._calc_buy_strength(float(prev_rsi)) if sig_val == SignalType.BUY else self._calc_sell_strength(float(prev_rsi))
                    meta = self._build_signal_metadata(indicators, current_idx, current_price, current_rsi, float(prev_rsi))
                    return [
                        TradingSignal(
                            timestamp=current_date,
                            stock_code=stock_code,
                            signal_type=sig_val,
                            strength=strength,
                            price=current_price,
                            reason=f"[向量化] RSI信号, RSI: {current_rsi:.2f}",
                            metadata=meta,
                        )
                    ]
                return []
        except Exception:
            pass

        # Fallback: 逐日计算（P0: 与 precompute 过滤逻辑完全一致）
        signals = []

        try:
            indicators = self.get_cached_indicators(data)

            current_idx = self._get_current_idx(data, current_date)
            if current_idx < max(self.rsi_period, 1):
                return signals

            current_rsi = indicators["rsi"].iloc[current_idx]
            prev_rsi = indicators["rsi"].iloc[current_idx - 1]
            current_price = indicators["price"].iloc[current_idx]
            stock_code = data.attrs.get("stock_code", "UNKNOWN")

            # 买入信号：RSI穿越超卖 + 统一过滤
            if (
                prev_rsi <= self.oversold_threshold
                and current_rsi > self.oversold_threshold
                and self._check_filters_scalar(indicators, current_idx, "buy")
            ):
                strength = self._calc_buy_strength(float(prev_rsi))
                meta = self._build_signal_metadata(indicators, current_idx, current_price, float(current_rsi), float(prev_rsi))
                signal = TradingSignal(
                    timestamp=current_date,
                    stock_code=stock_code,
                    signal_type=SignalType.BUY,
                    strength=strength,
                    price=current_price,
                    reason=f"RSI从超卖区域向上穿越({prev_rsi:.2f}->{current_rsi:.2f})",
                    metadata=meta,
                )
                signals.append(signal)

            # 卖出信号：RSI穿越超买 + 统一过滤
            elif (
                prev_rsi >= self.overbought_threshold
                and current_rsi <= self.overbought_threshold
                and self._check_filters_scalar(indicators, current_idx, "sell")
            ):
                strength = self._calc_sell_strength(float(prev_rsi))
                meta = self._build_signal_metadata(indicators, current_idx, current_price, float(current_rsi), float(prev_rsi))
                signal = TradingSignal(
                    timestamp=current_date,
                    stock_code=stock_code,
                    signal_type=SignalType.SELL,
                    strength=strength,
                    price=current_price,
                    reason=f"RSI从超买区域向下穿越({prev_rsi:.2f}->{current_rsi:.2f})",
                    metadata=meta,
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
        """[性能优化] 向量化计算全量MACD信号（返回浮点强度，含零轴过滤）

        返回值：正数=买入信号（强度），负数=卖出信号（强度），0=无信号
        零轴过滤：MACD线在零轴以上的金叉为强信号，零轴以下的金叉强度减半
        """
        try:
            indicators = self.get_cached_indicators(data)
            macd_line = indicators["macd"]
            macd_hist = indicators["macd_hist"]
            prev_hist = macd_hist.shift(1)

            # 向量化逻辑判断：金叉和死叉
            buy_mask = (prev_hist <= 0) & (macd_hist > 0)
            sell_mask = (prev_hist >= 0) & (macd_hist < 0)

            # 构造浮点强度信号 Series
            signals = pd.Series(0.0, index=data.index, dtype=np.float64)

            # 买入强度：柱状图越大强度越高
            buy_strength = (macd_hist * 100).clip(0.3, 1.0)
            # 零轴过滤：MACD线在零轴以下的金叉，强度减半
            below_zero = macd_line < 0
            buy_strength[below_zero] = buy_strength[below_zero] * 0.5
            signals[buy_mask.fillna(False)] = buy_strength[buy_mask.fillna(False)]

            # 卖出强度：柱状图越负强度越高（负值）
            sell_strength = (macd_hist * 100).clip(-1.0, -0.3)
            # 零轴过滤：MACD线在零轴以上的死叉，强度减半
            above_zero = macd_line > 0
            sell_strength[above_zero] = sell_strength[above_zero] * 0.5
            signals[sell_mask.fillna(False)] = sell_strength[sell_mask.fillna(False)]

            return signals
        except Exception as e:
            logger.error(f"MACD策略向量化计算失败: {e}")
            return None

    def generate_signals(
        self, data: pd.DataFrame, current_date: datetime
    ) -> List[TradingSignal]:
        """生成MACD信号"""
        # 性能优化：优先检查是否已有全量预计算信号
        try:
            precomputed = data.attrs.get("_precomputed_signals", {}).get(self.name)
            if precomputed is not None:
                sig_val = precomputed.get(current_date)
                # 支持浮点信号（正=买入，负=卖出）和旧枚举格式
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
                    current_hist = indicators["macd_hist"].iloc[current_idx]
                    return [
                        TradingSignal(
                            timestamp=current_date,
                            stock_code=stock_code,
                            signal_type=sig_type,
                            strength=min(1.0, abs(fv)),
                            price=current_price,
                            reason=f"[向量化] MACD{'金叉' if sig_type == SignalType.BUY else '死叉'}，柱状图: {current_hist:.4f}",
                            metadata={
                                "macd": indicators["macd"].iloc[current_idx],
                                "macd_signal": indicators["macd_signal"].iloc[
                                    current_idx
                                ],
                                "macd_hist": current_hist,
                            },
                        )
                    ]
                elif isinstance(sig_val, SignalType):
                    indicators = self.get_cached_indicators(data)
                    current_idx = self._get_current_idx(data, current_date)
                    stock_code = data.attrs.get("stock_code", "UNKNOWN")
                    current_price = indicators["price"].iloc[current_idx]
                    current_hist = indicators["macd_hist"].iloc[current_idx]
                    return [
                        TradingSignal(
                            timestamp=current_date,
                            stock_code=stock_code,
                            signal_type=sig_val,
                            strength=min(1.0, abs(current_hist) * 100),
                            price=current_price,
                            reason=f"[向量化] MACD{'金叉' if sig_val == SignalType.BUY else '死叉'}，柱状图: {current_hist:.4f}",
                            metadata={
                                "macd": indicators["macd"].iloc[current_idx],
                                "macd_signal": indicators["macd_signal"].iloc[
                                    current_idx
                                ],
                                "macd_hist": current_hist,
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
