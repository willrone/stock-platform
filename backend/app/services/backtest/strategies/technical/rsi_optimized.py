"""
RSI策略 - 性能优化版本

优化点：
1. 简化指标计算，移除复杂的背离检测和趋势分析
2. 使用向量化计算
3. 减少不必要的循环和条件判断
4. 支持预计算列复用、Numba加速、TA-Lib回退
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
    RSI策略 - 性能优化版本
    
    核心逻辑：
    - RSI从超卖区回升: 买入信号
    - RSI从超买区回调: 卖出信号
    
    计算路径优先级：预计算列 > Numba > TA-Lib > pandas rolling
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__("RSI", config)
        logger.info(f"\U0001f4cc RSI策略实例化: class={self.__class__.__name__}, module={self.__class__.__module__}, rsi_period={config.get('rsi_period', 14)}")
        self.rsi_period = config.get("rsi_period", 14)
        self.oversold_threshold = config.get("oversold_threshold", 30)
        self.overbought_threshold = config.get("overbought_threshold", 70)

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
                if TALIB_AVAILABLE:
                    rsi = pd.Series(
                        talib.RSI(close_prices.values, timeperiod=self.rsi_period),
                        index=close_prices.index,
                    )
                    calc_path = "talib_fallback"
                else:
                    delta = close_prices.diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
                    rs = gain / loss
                    rsi = 100 - (100 / (1 + rs))
                    calc_path = "pandas_fallback"

        _elapsed_ms = (_time.perf_counter() - _t0) * 1000
        stock_code = data.attrs.get("stock_code", "?")
        if _elapsed_ms > 5:  # 只记录耗时 >5ms 的
            logger.debug(f"\U0001f4ca RSI.calculate_indicators [{stock_code}]: path={calc_path}, {_elapsed_ms:.1f}ms")

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

            # 简化逻辑：从超卖区回升 -> 买入；从超买区回调 -> 卖出
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
