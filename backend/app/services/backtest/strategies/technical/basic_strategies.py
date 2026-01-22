"""
基础策略实现

包含移动平均、RSI、MACD等基础技术分析策略
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any
from datetime import datetime
from loguru import logger

from ...core.base_strategy import BaseStrategy
from ...models import TradingSignal, SignalType

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
        self.short_window = config.get('short_window', 5)
        self.long_window = config.get('long_window', 20)
        self.signal_threshold = config.get('signal_threshold', 0.02)
    
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """计算移动平均指标"""
        close_prices = data['close']
        
        indicators = {
            'sma_short': close_prices.rolling(window=self.short_window).mean(),
            'sma_long': close_prices.rolling(window=self.long_window).mean(),
            'price': close_prices
        }
        
        # 计算移动平均差值
        indicators['ma_diff'] = (indicators['sma_short'] - indicators['sma_long']) / indicators['sma_long']
        
        return indicators
    
    def generate_signals(self, data: pd.DataFrame, current_date: datetime) -> List[TradingSignal]:
        """生成移动平均交叉信号"""
        signals = []
        
        try:
            # 计算指标
            indicators = self.calculate_indicators(data)
            
            # 获取当前数据点
            current_idx = data.index.get_loc(current_date) if current_date in data.index else -1
            if current_idx < self.long_window:
                return signals  # 数据不足
            
            current_price = indicators['price'].iloc[current_idx]
            current_ma_diff = indicators['ma_diff'].iloc[current_idx]
            prev_ma_diff = indicators['ma_diff'].iloc[current_idx - 1]
            
            stock_code = data.attrs.get('stock_code', 'UNKNOWN')
            
            # 生成买入信号
            if (prev_ma_diff <= 0 and current_ma_diff > 0 and 
                abs(current_ma_diff) > self.signal_threshold):
                
                signal = TradingSignal(
                    timestamp=current_date,
                    stock_code=stock_code,
                    signal_type=SignalType.BUY,
                    strength=min(1.0, abs(current_ma_diff) * 10),
                    price=current_price,
                    reason=f"短期均线上穿长期均线，差值: {current_ma_diff:.3f}",
                    metadata={
                        'sma_short': indicators['sma_short'].iloc[current_idx],
                        'sma_long': indicators['sma_long'].iloc[current_idx],
                        'ma_diff': current_ma_diff
                    }
                )
                signals.append(signal)
            
            # 生成卖出信号
            elif (prev_ma_diff >= 0 and current_ma_diff < 0 and 
                  abs(current_ma_diff) > self.signal_threshold):
                
                signal = TradingSignal(
                    timestamp=current_date,
                    stock_code=stock_code,
                    signal_type=SignalType.SELL,
                    strength=min(1.0, abs(current_ma_diff) * 10),
                    price=current_price,
                    reason=f"短期均线下穿长期均线，差值: {current_ma_diff:.3f}",
                    metadata={
                        'sma_short': indicators['sma_short'].iloc[current_idx],
                        'sma_long': indicators['sma_long'].iloc[current_idx],
                        'ma_diff': current_ma_diff
                    }
                )
                signals.append(signal)
            
            return signals
            
        except Exception as e:
            logger.error(f"移动平均策略信号生成失败: {e}")
            return []


class RSIStrategy(BaseStrategy):
    """RSI策略"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("RSI", config)
        self.rsi_period = config.get('rsi_period', 14)
        self.oversold_threshold = config.get('oversold_threshold', 30)
        self.overbought_threshold = config.get('overbought_threshold', 70)
    
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """计算RSI指标"""
        close_prices = data['close']
        
        # 使用talib或pandas计算RSI
        if TALIB_AVAILABLE:
            rsi = pd.Series(talib.RSI(close_prices.values, timeperiod=self.rsi_period), 
                           index=close_prices.index)
        else:
            # 使用pandas实现RSI
            delta = close_prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
        
        return {
            'rsi': rsi,
            'price': close_prices
        }
    
    def generate_signals(self, data: pd.DataFrame, current_date: datetime) -> List[TradingSignal]:
        """生成RSI信号"""
        signals = []
        
        try:
            indicators = self.calculate_indicators(data)
            
            current_idx = data.index.get_loc(current_date) if current_date in data.index else -1
            if current_idx < self.rsi_period:
                return signals
            
            current_rsi = indicators['rsi'].iloc[current_idx]
            current_price = indicators['price'].iloc[current_idx]
            stock_code = data.attrs.get('stock_code', 'UNKNOWN')
            
            # RSI超卖信号（买入）
            if current_rsi < self.oversold_threshold:
                strength = (self.oversold_threshold - current_rsi) / self.oversold_threshold
                signal = TradingSignal(
                    timestamp=current_date,
                    stock_code=stock_code,
                    signal_type=SignalType.BUY,
                    strength=min(1.0, strength),
                    price=current_price,
                    reason=f"RSI超卖: {current_rsi:.2f}",
                    metadata={'rsi': current_rsi}
                )
                signals.append(signal)
            
            # RSI超买信号（卖出）
            elif current_rsi > self.overbought_threshold:
                strength = (current_rsi - self.overbought_threshold) / (100 - self.overbought_threshold)
                signal = TradingSignal(
                    timestamp=current_date,
                    stock_code=stock_code,
                    signal_type=SignalType.SELL,
                    strength=min(1.0, strength),
                    price=current_price,
                    reason=f"RSI超买: {current_rsi:.2f}",
                    metadata={'rsi': current_rsi}
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
        self.fast_period = config.get('fast_period', 12)
        self.slow_period = config.get('slow_period', 26)
        self.signal_period = config.get('signal_period', 9)
    
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """计算MACD指标"""
        close_prices = data['close']
        
        # 使用talib或pandas计算MACD
        if TALIB_AVAILABLE:
            macd, macd_signal, macd_hist = talib.MACD(
                close_prices.values, 
                fastperiod=self.fast_period,
                slowperiod=self.slow_period, 
                signalperiod=self.signal_period
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
            'macd': macd,
            'macd_signal': macd_signal,
            'macd_hist': macd_hist,
            'price': close_prices
        }
    
    def generate_signals(self, data: pd.DataFrame, current_date: datetime) -> List[TradingSignal]:
        """生成MACD信号"""
        signals = []
        
        try:
            indicators = self.calculate_indicators(data)
            
            current_idx = data.index.get_loc(current_date) if current_date in data.index else -1
            if current_idx < self.slow_period + self.signal_period:
                return signals
            
            current_hist = indicators['macd_hist'].iloc[current_idx]
            prev_hist = indicators['macd_hist'].iloc[current_idx - 1]
            current_price = indicators['price'].iloc[current_idx]
            stock_code = data.attrs.get('stock_code', 'UNKNOWN')
            
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
                        'macd': indicators['macd'].iloc[current_idx],
                        'macd_signal': indicators['macd_signal'].iloc[current_idx],
                        'macd_hist': current_hist
                    }
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
                        'macd': indicators['macd'].iloc[current_idx],
                        'macd_signal': indicators['macd_signal'].iloc[current_idx],
                        'macd_hist': current_hist
                    }
                )
                signals.append(signal)
            
            return signals
            
        except Exception as e:
            logger.error(f"MACD策略信号生成失败: {e}")
            return []
