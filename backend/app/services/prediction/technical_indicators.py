"""
技术指标计算服务
实现各种股票技术指标的计算
"""

import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import math
from dataclasses import dataclass
import pandas as pd

from app.models.stock_simple import StockData


@dataclass
class TechnicalIndicatorResult:
    """技术指标计算结果"""
    stock_code: str
    date: datetime
    indicators: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'stock_code': self.stock_code,
            'date': self.date.isoformat(),
            'indicators': self.indicators
        }


@dataclass
class BatchIndicatorRequest:
    """批量指标计算请求"""
    stock_codes: List[str]
    indicators: List[str]  # 要计算的指标列表
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None


@dataclass
class BatchIndicatorResponse:
    """批量指标计算响应"""
    success: bool
    results: Dict[str, List[TechnicalIndicatorResult]]  # stock_code -> results
    failed_stocks: List[str]
    message: str


class TechnicalIndicatorCalculator:
    """技术指标计算器 - MLOps优化版本"""
    
    def __init__(self):
        # 扩展支持的技术指标
        self.supported_indicators = {
            # 移动平均线
            'MA5', 'MA10', 'MA20', 'MA60', 'SMA', 'EMA', 'WMA',
            # 动量指标
            'RSI', 'STOCH', 'WILLIAMS_R', 'CCI', 'MOMENTUM', 'ROC',
            # 趋势指标
            'MACD', 'BOLLINGER', 'SAR', 'ADX', 'ICHIMOKU',
            # 成交量指标
            'VWAP', 'OBV', 'AD_LINE', 'VOLUME_RSI',
            # 波动率指标
            'ATR', 'VOLATILITY', 'HISTORICAL_VOLATILITY',
            # 复合指标
            'KDJ'
        }
        
        # 指标计算缓存
        self._calculation_cache = {}
        
        # 增量更新支持
        self._incremental_data = {}
    
    def validate_data(self, data: List[StockData]) -> bool:
        """验证输入数据"""
        if not data:
            return False
        
        # 检查数据完整性
        for item in data:
            if not isinstance(item, StockData):
                return False
            if item.close <= 0 or item.high <= 0 or item.low <= 0 or item.open <= 0:
                return False
            if item.high < item.low:
                return False
            if item.volume < 0:
                return False
        
        # 检查数据按日期排序
        dates = [item.date for item in data]
        if dates != sorted(dates):
            return False
        
        return True
    
    def calculate_moving_average(self, data: List[StockData], period: int) -> List[Optional[float]]:
        """计算移动平均线"""
        if len(data) < period:
            return [None] * len(data)
        
        ma_values = []
        
        for i in range(len(data)):
            if i < period - 1:
                ma_values.append(None)
            else:
                # 计算过去period天的平均收盘价
                sum_close = sum(data[j].close for j in range(i - period + 1, i + 1))
                ma_values.append(round(sum_close / period, 4))
        
        return ma_values
    
    def calculate_ema(self, data: List[StockData], period: int) -> List[Optional[float]]:
        """计算指数移动平均线(EMA)"""
        if len(data) < period:
            return [None] * len(data)
        
        ema_values = []
        multiplier = 2 / (period + 1)
        
        for i in range(len(data)):
            if i < period - 1:
                ema_values.append(None)
            elif i == period - 1:
                # 第一个EMA值使用简单平均
                sma = sum(data[j].close for j in range(i - period + 1, i + 1)) / period
                ema_values.append(round(sma, 4))
            else:
                # EMA = (当前价格 * 乘数) + (前一日EMA * (1 - 乘数))
                ema = (data[i].close * multiplier) + (ema_values[i-1] * (1 - multiplier))
                ema_values.append(round(ema, 4))
        
        return ema_values
    
    def calculate_wma(self, data: List[StockData], period: int) -> List[Optional[float]]:
        """计算加权移动平均线(WMA)"""
        if len(data) < period:
            return [None] * len(data)
        
        wma_values = []
        
        for i in range(len(data)):
            if i < period - 1:
                wma_values.append(None)
            else:
                # 计算加权平均
                weighted_sum = 0
                weight_sum = 0
                
                for j in range(period):
                    weight = j + 1  # 权重从1开始递增
                    weighted_sum += data[i - period + 1 + j].close * weight
                    weight_sum += weight
                
                wma = weighted_sum / weight_sum
                wma_values.append(round(wma, 4))
        
        return wma_values
    
    def calculate_stochastic(self, data: List[StockData], k_period: int = 14, d_period: int = 3) -> Dict[str, List[Optional[float]]]:
        """计算随机指标(Stochastic)"""
        if len(data) < k_period:
            return {
                'stoch_k': [None] * len(data),
                'stoch_d': [None] * len(data)
            }
        
        stoch_k = []
        
        for i in range(len(data)):
            if i < k_period - 1:
                stoch_k.append(None)
            else:
                # 获取过去k_period天的最高价和最低价
                high_values = [data[j].high for j in range(i - k_period + 1, i + 1)]
                low_values = [data[j].low for j in range(i - k_period + 1, i + 1)]
                
                highest_high = max(high_values)
                lowest_low = min(low_values)
                current_close = data[i].close
                
                if highest_high == lowest_low:
                    k_value = 50  # 避免除零
                else:
                    k_value = ((current_close - lowest_low) / (highest_high - lowest_low)) * 100
                
                stoch_k.append(round(k_value, 4))
        
        # 计算%D（%K的移动平均）
        stoch_d = []
        for i in range(len(data)):
            if i < k_period - 1 + d_period - 1:
                stoch_d.append(None)
            else:
                # 计算过去d_period天%K的平均值
                k_values = [stoch_k[j] for j in range(i - d_period + 1, i + 1) if stoch_k[j] is not None]
                if k_values:
                    d_value = sum(k_values) / len(k_values)
                    stoch_d.append(round(d_value, 4))
                else:
                    stoch_d.append(None)
        
        return {
            'stoch_k': stoch_k,
            'stoch_d': stoch_d
        }
    
    def calculate_williams_r(self, data: List[StockData], period: int = 14) -> List[Optional[float]]:
        """计算威廉指标(Williams %R)"""
        if len(data) < period:
            return [None] * len(data)
        
        williams_r = []
        
        for i in range(len(data)):
            if i < period - 1:
                williams_r.append(None)
            else:
                # 获取过去period天的最高价和最低价
                high_values = [data[j].high for j in range(i - period + 1, i + 1)]
                low_values = [data[j].low for j in range(i - period + 1, i + 1)]
                
                highest_high = max(high_values)
                lowest_low = min(low_values)
                current_close = data[i].close
                
                if highest_high == lowest_low:
                    wr_value = -50  # 避免除零
                else:
                    wr_value = ((highest_high - current_close) / (highest_high - lowest_low)) * -100
                
                williams_r.append(round(wr_value, 4))
        
        return williams_r
    
    def calculate_cci(self, data: List[StockData], period: int = 20) -> List[Optional[float]]:
        """计算商品通道指数(CCI)"""
        if len(data) < period:
            return [None] * len(data)
        
        cci_values = []
        
        for i in range(len(data)):
            if i < period - 1:
                cci_values.append(None)
            else:
                # 计算典型价格(TP)
                typical_prices = []
                for j in range(i - period + 1, i + 1):
                    tp = (data[j].high + data[j].low + data[j].close) / 3
                    typical_prices.append(tp)
                
                # 计算典型价格的移动平均
                sma_tp = sum(typical_prices) / period
                
                # 计算平均偏差
                mean_deviation = sum(abs(tp - sma_tp) for tp in typical_prices) / period
                
                # 计算CCI
                current_tp = (data[i].high + data[i].low + data[i].close) / 3
                if mean_deviation == 0:
                    cci = 0
                else:
                    cci = (current_tp - sma_tp) / (0.015 * mean_deviation)
                
                cci_values.append(round(cci, 4))
        
        return cci_values
    
    def calculate_atr(self, data: List[StockData], period: int = 14) -> List[Optional[float]]:
        """计算平均真实波幅(ATR)"""
        if len(data) < period + 1:
            return [None] * len(data)
        
        # 计算真实波幅(TR)
        true_ranges = []
        for i in range(1, len(data)):
            high_low = data[i].high - data[i].low
            high_close_prev = abs(data[i].high - data[i-1].close)
            low_close_prev = abs(data[i].low - data[i-1].close)
            
            tr = max(high_low, high_close_prev, low_close_prev)
            true_ranges.append(tr)
        
        # 计算ATR
        atr_values = [None]  # 第一个值为None
        
        for i in range(len(true_ranges)):
            if i < period - 1:
                atr_values.append(None)
            elif i == period - 1:
                # 第一个ATR值使用简单平均
                atr = sum(true_ranges[:period]) / period
                atr_values.append(round(atr, 4))
            else:
                # 后续ATR值使用指数平滑
                prev_atr = atr_values[-1]
                current_tr = true_ranges[i]
                atr = (prev_atr * (period - 1) + current_tr) / period
                atr_values.append(round(atr, 4))
        
        return atr_values
    
    def calculate_vwap(self, data: List[StockData]) -> List[Optional[float]]:
        """计算成交量加权平均价格(VWAP)"""
        vwap_values = []
        cumulative_volume = 0
        cumulative_pv = 0
        
        for item in data:
            typical_price = (item.high + item.low + item.close) / 3
            pv = typical_price * item.volume
            
            cumulative_pv += pv
            cumulative_volume += item.volume
            
            if cumulative_volume > 0:
                vwap = cumulative_pv / cumulative_volume
                vwap_values.append(round(vwap, 4))
            else:
                vwap_values.append(None)
        
        return vwap_values
    
    def calculate_obv(self, data: List[StockData]) -> List[Optional[float]]:
        """计算能量潮(OBV)"""
        if len(data) < 2:
            return [None] * len(data)
        
        obv_values = [0]  # 第一个值设为0
        
        for i in range(1, len(data)):
            prev_obv = obv_values[-1]
            
            if data[i].close > data[i-1].close:
                # 价格上涨，加上成交量
                obv = prev_obv + data[i].volume
            elif data[i].close < data[i-1].close:
                # 价格下跌，减去成交量
                obv = prev_obv - data[i].volume
            else:
                # 价格不变，OBV不变
                obv = prev_obv
            
            obv_values.append(obv)
        
        return obv_values
    
    def calculate_kdj(self, data: List[StockData], k_period: int = 9, d_period: int = 3, j_period: int = 3) -> Dict[str, List[Optional[float]]]:
        """计算KDJ指标"""
        if len(data) < k_period:
            return {
                'kdj_k': [None] * len(data),
                'kdj_d': [None] * len(data),
                'kdj_j': [None] * len(data)
            }
        
        # 计算RSV (Raw Stochastic Value)
        rsv_values = []
        for i in range(len(data)):
            if i < k_period - 1:
                rsv_values.append(None)
            else:
                # 获取过去k_period天的最高价和最低价
                high_values = [data[j].high for j in range(i - k_period + 1, i + 1)]
                low_values = [data[j].low for j in range(i - k_period + 1, i + 1)]
                
                highest_high = max(high_values)
                lowest_low = min(low_values)
                current_close = data[i].close
                
                if highest_high == lowest_low:
                    rsv = 50  # 避免除零
                else:
                    rsv = ((current_close - lowest_low) / (highest_high - lowest_low)) * 100
                
                rsv_values.append(rsv)
        
        # 计算K值
        k_values = []
        for i in range(len(data)):
            if rsv_values[i] is None:
                k_values.append(None)
            elif i == 0 or k_values[i-1] is None:
                k_values.append(rsv_values[i])
            else:
                # K = (2/3) * 前一日K值 + (1/3) * 当日RSV
                k = (2/3) * k_values[i-1] + (1/3) * rsv_values[i]
                k_values.append(round(k, 4))
        
        # 计算D值
        d_values = []
        for i in range(len(data)):
            if k_values[i] is None:
                d_values.append(None)
            elif i == 0 or d_values[i-1] is None:
                d_values.append(k_values[i])
            else:
                # D = (2/3) * 前一日D值 + (1/3) * 当日K值
                d = (2/3) * d_values[i-1] + (1/3) * k_values[i]
                d_values.append(round(d, 4))
        
        # 计算J值
        j_values = []
        for i in range(len(data)):
            if k_values[i] is None or d_values[i] is None:
                j_values.append(None)
            else:
                # J = 3K - 2D
                j = 3 * k_values[i] - 2 * d_values[i]
                j_values.append(round(j, 4))
        
        return {
            'kdj_k': k_values,
            'kdj_d': d_values,
            'kdj_j': j_values
        }
    
    def calculate_incremental_indicators(
        self, 
        new_data: List[StockData], 
        existing_results: Dict[str, List[Optional[float]]],
        indicators: List[str]
    ) -> Dict[str, List[Optional[float]]]:
        """增量计算技术指标"""
        # 这是一个简化的增量更新实现
        # 实际应用中需要根据具体指标的计算特性来优化
        
        # 合并新旧数据
        if not hasattr(self, '_cached_data'):
            self._cached_data = {}
        
        stock_code = new_data[0].stock_code if new_data else None
        if stock_code not in self._cached_data:
            self._cached_data[stock_code] = []
        
        # 添加新数据
        self._cached_data[stock_code].extend(new_data)
        
        # 重新计算指标（简化版本，实际可以优化为只计算新增部分）
        updated_results = self.calculate_indicators(self._cached_data[stock_code], indicators)
        
        # 转换为字典格式
        result_dict = {}
        for indicator in indicators:
            result_dict[indicator] = []
            for result in updated_results:
                if indicator in result.indicators:
                    result_dict[indicator].append(result.indicators[indicator])
                else:
                    result_dict[indicator].append(None)
        
        return result_dict
        """计算移动平均线"""
        if len(data) < period:
            return [None] * len(data)
        
        ma_values = []
        
        for i in range(len(data)):
            if i < period - 1:
                ma_values.append(None)
            else:
                # 计算过去period天的平均收盘价
                sum_close = sum(data[j].close for j in range(i - period + 1, i + 1))
                ma_values.append(round(sum_close / period, 4))
        
        return ma_values
    
    def calculate_rsi(self, data: List[StockData], period: int = 14) -> List[Optional[float]]:
        """计算相对强弱指数(RSI)"""
        if len(data) < period + 1:
            return [None] * len(data)
        
        rsi_values = []
        
        # 计算价格变化
        price_changes = []
        for i in range(1, len(data)):
            change = data[i].close - data[i-1].close
            price_changes.append(change)
        
        for i in range(len(data)):
            if i < period:
                rsi_values.append(None)
            else:
                # 获取过去period天的价格变化
                recent_changes = price_changes[i-period:i]
                
                # 分离上涨和下跌
                gains = [change for change in recent_changes if change > 0]
                losses = [-change for change in recent_changes if change < 0]
                
                # 计算平均收益和平均损失
                avg_gain = sum(gains) / period if gains else 0
                avg_loss = sum(losses) / period if losses else 0
                
                # 计算RSI
                if avg_loss == 0:
                    rsi = 100
                else:
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
                
                rsi_values.append(round(rsi, 4))
        
        return rsi_values
    
    def calculate_macd(self, data: List[StockData], fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Dict[str, List[Optional[float]]]:
        """计算MACD指标"""
        if len(data) < slow_period:
            return {
                'macd': [None] * len(data),
                'signal': [None] * len(data),
                'histogram': [None] * len(data)
            }
        
        # 计算EMA
        def calculate_ema(prices: List[float], period: int) -> List[Optional[float]]:
            if len(prices) < period:
                return [None] * len(prices)
            
            ema_values = []
            multiplier = 2 / (period + 1)
            
            for i in range(len(prices)):
                if i < period - 1:
                    ema_values.append(None)
                elif i == period - 1:
                    # 第一个EMA值使用简单平均
                    sma = sum(prices[:period]) / period
                    ema_values.append(sma)
                else:
                    # EMA = (当前价格 * 乘数) + (前一日EMA * (1 - 乘数))
                    ema = (prices[i] * multiplier) + (ema_values[i-1] * (1 - multiplier))
                    ema_values.append(ema)
            
            return ema_values
        
        # 获取收盘价
        closes = [item.close for item in data]
        
        # 计算快线和慢线EMA
        fast_ema = calculate_ema(closes, fast_period)
        slow_ema = calculate_ema(closes, slow_period)
        
        # 计算MACD线
        macd_line = []
        for i in range(len(data)):
            if fast_ema[i] is None or slow_ema[i] is None:
                macd_line.append(None)
            else:
                macd_line.append(round(fast_ema[i] - slow_ema[i], 4))
        
        # 计算信号线（MACD的EMA）
        macd_values = [v for v in macd_line if v is not None]
        if len(macd_values) >= signal_period:
            signal_ema = calculate_ema(macd_values, signal_period)
            # 调整信号线长度
            signal_line = [None] * (len(macd_line) - len(signal_ema)) + signal_ema
        else:
            signal_line = [None] * len(data)
        
        # 计算柱状图
        histogram = []
        for i in range(len(data)):
            if macd_line[i] is None or signal_line[i] is None:
                histogram.append(None)
            else:
                histogram.append(round(macd_line[i] - signal_line[i], 4))
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    def calculate_bollinger_bands(self, data: List[StockData], period: int = 20, std_dev: float = 2.0) -> Dict[str, List[Optional[float]]]:
        """计算布林带"""
        if len(data) < period:
            return {
                'upper': [None] * len(data),
                'middle': [None] * len(data),
                'lower': [None] * len(data)
            }
        
        upper_band = []
        middle_band = []
        lower_band = []
        
        for i in range(len(data)):
            if i < period - 1:
                upper_band.append(None)
                middle_band.append(None)
                lower_band.append(None)
            else:
                # 获取过去period天的收盘价
                recent_closes = [data[j].close for j in range(i - period + 1, i + 1)]
                
                # 计算移动平均（中轨）
                ma = sum(recent_closes) / period
                
                # 计算标准差
                variance = sum((price - ma) ** 2 for price in recent_closes) / period
                std = math.sqrt(variance)
                
                # 计算上轨和下轨
                upper = ma + (std_dev * std)
                lower = ma - (std_dev * std)
                
                upper_band.append(round(upper, 4))
                middle_band.append(round(ma, 4))
                lower_band.append(round(lower, 4))
        
        return {
            'upper': upper_band,
            'middle': middle_band,
            'lower': lower_band
        }
    
    def calculate_indicators(self, data: List[StockData], indicators: List[str]) -> List[TechnicalIndicatorResult]:
        """计算指定的技术指标 - 增强版本"""
        if not self.validate_data(data):
            raise ValueError("输入数据验证失败")
        
        if not data:
            return []
        
        # 验证指标名称
        invalid_indicators = set(indicators) - self.supported_indicators
        if invalid_indicators:
            raise ValueError(f"不支持的指标: {invalid_indicators}")
        
        results = []
        stock_code = data[0].stock_code
        
        # 计算各种指标
        calculated_indicators = {}
        
        # 移动平均线
        if 'MA5' in indicators:
            calculated_indicators['MA5'] = self.calculate_moving_average(data, 5)
        if 'MA10' in indicators:
            calculated_indicators['MA10'] = self.calculate_moving_average(data, 10)
        if 'MA20' in indicators:
            calculated_indicators['MA20'] = self.calculate_moving_average(data, 20)
        if 'MA60' in indicators:
            calculated_indicators['MA60'] = self.calculate_moving_average(data, 60)
        if 'SMA' in indicators:
            calculated_indicators['SMA'] = self.calculate_moving_average(data, 20)  # 默认20日
        if 'EMA' in indicators:
            calculated_indicators['EMA'] = self.calculate_ema(data, 20)  # 默认20日
        if 'WMA' in indicators:
            calculated_indicators['WMA'] = self.calculate_wma(data, 20)  # 默认20日
        
        # 动量指标
        if 'RSI' in indicators:
            calculated_indicators['RSI'] = self.calculate_rsi(data)
        if 'STOCH' in indicators:
            stoch_result = self.calculate_stochastic(data)
            calculated_indicators['STOCH_K'] = stoch_result['stoch_k']
            calculated_indicators['STOCH_D'] = stoch_result['stoch_d']
        if 'WILLIAMS_R' in indicators:
            calculated_indicators['WILLIAMS_R'] = self.calculate_williams_r(data)
        if 'CCI' in indicators:
            calculated_indicators['CCI'] = self.calculate_cci(data)
        
        # 趋势指标
        if 'MACD' in indicators:
            macd_result = self.calculate_macd(data)
            calculated_indicators['MACD'] = macd_result['macd']
            calculated_indicators['MACD_SIGNAL'] = macd_result['signal']
            calculated_indicators['MACD_HISTOGRAM'] = macd_result['histogram']
        
        # 布林带
        if 'BOLLINGER' in indicators:
            bollinger_result = self.calculate_bollinger_bands(data)
            calculated_indicators['BOLLINGER_UPPER'] = bollinger_result['upper']
            calculated_indicators['BOLLINGER_MIDDLE'] = bollinger_result['middle']
            calculated_indicators['BOLLINGER_LOWER'] = bollinger_result['lower']
        
        # 波动率指标
        if 'ATR' in indicators:
            calculated_indicators['ATR'] = self.calculate_atr(data)
        
        # 成交量指标
        if 'VWAP' in indicators:
            calculated_indicators['VWAP'] = self.calculate_vwap(data)
        if 'OBV' in indicators:
            calculated_indicators['OBV'] = self.calculate_obv(data)
        
        # 复合指标
        if 'KDJ' in indicators:
            kdj_result = self.calculate_kdj(data)
            calculated_indicators['KDJ_K'] = kdj_result['kdj_k']
            calculated_indicators['KDJ_D'] = kdj_result['kdj_d']
            calculated_indicators['KDJ_J'] = kdj_result['kdj_j']
        
        # 组装结果
        for i, stock_data in enumerate(data):
            indicator_values = {}
            
            for indicator_name, values in calculated_indicators.items():
                if i < len(values) and values[i] is not None:
                    indicator_values[indicator_name] = values[i]
            
            if indicator_values:  # 只有当有指标值时才添加结果
                results.append(TechnicalIndicatorResult(
                    stock_code=stock_code,
                    date=stock_data.date,
                    indicators=indicator_values
                ))
        
        return results
    
    async def calculate_all_indicators(self, stock_data: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有支持的技术指标，返回包含指标列的DataFrame
        
        Args:
            stock_data: 包含 open, high, low, close, volume 列的DataFrame，索引为日期
            
        Returns:
            包含所有技术指标的DataFrame，索引为日期
        """
        if stock_data.empty:
            return pd.DataFrame()
        
        # 确保有必要的列
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in stock_data.columns for col in required_cols):
            raise ValueError(f"DataFrame必须包含以下列: {required_cols}")
        
        # 创建结果DataFrame，使用相同的索引
        result_df = pd.DataFrame(index=stock_data.index)
        
        # 转换为StockData列表以便使用现有方法
        stock_data_list = []
        for date, row in stock_data.iterrows():
            stock_data_list.append(StockData(
                stock_code='',  # 不需要stock_code
                date=date if isinstance(date, datetime) else pd.to_datetime(date),
                open=float(row['open']),
                high=float(row['high']),
                low=float(row['low']),
                close=float(row['close']),
                volume=float(row['volume'])
            ))
        
        # 计算所有支持的指标
        all_indicators = list(self.supported_indicators)
        indicator_results = self.calculate_indicators(stock_data_list, all_indicators)
        
        # 将结果转换为DataFrame
        if indicator_results:
            # 按日期组织指标
            indicators_dict = {}
            for result in indicator_results:
                date_key = result.date
                if date_key not in indicators_dict:
                    indicators_dict[date_key] = {}
                indicators_dict[date_key].update(result.indicators)
            
            # 转换为DataFrame
            for date, indicators in indicators_dict.items():
                for indicator_name, value in indicators.items():
                    if indicator_name not in result_df.columns:
                        result_df[indicator_name] = None
                    result_df.loc[date, indicator_name] = value
        
        return result_df
    
    async def calculate_batch_indicators(
        self, 
        request: BatchIndicatorRequest,
        data_service
    ) -> BatchIndicatorResponse:
        """批量计算多只股票的技术指标 - 优化版本"""
        print(f"开始批量计算技术指标: {len(request.stock_codes)} 只股票")
        
        results = {}
        failed_stocks = []
        
        # 并发控制 - 根据系统资源调整
        max_concurrent = min(5, len(request.stock_codes))  # 最多5个并发任务
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def calculate_single_stock(stock_code: str):
            async with semaphore:
                try:
                    # 获取股票数据
                    stock_data = await data_service.get_stock_data(
                        stock_code,
                        request.start_date,
                        request.end_date
                    )
                    
                    if not stock_data:
                        failed_stocks.append(stock_code)
                        return
                    
                    # 计算技术指标
                    indicators_result = self.calculate_indicators(stock_data, request.indicators)
                    results[stock_code] = indicators_result
                    
                    print(f"股票 {stock_code} 指标计算完成: {len(indicators_result)} 条记录")
                
                except Exception as e:
                    print(f"股票 {stock_code} 指标计算失败: {e}")
                    failed_stocks.append(stock_code)
        
        # 并发执行计算任务
        tasks = [calculate_single_stock(code) for code in request.stock_codes]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        success = len(failed_stocks) == 0
        total_results = sum(len(result) for result in results.values())
        message = f"批量计算完成: 成功 {len(results)}, 失败 {len(failed_stocks)}, 总计 {total_results} 条指标记录"
        
        print(message)
        
        return BatchIndicatorResponse(
            success=success,
            results=results,
            failed_stocks=failed_stocks,
            message=message
        )
    
    def get_supported_indicators_info(self) -> Dict[str, Dict[str, str]]:
        """获取支持的技术指标信息"""
        return {
            # 移动平均线
            'MA5': {'category': '趋势指标', 'description': '5日移动平均线'},
            'MA10': {'category': '趋势指标', 'description': '10日移动平均线'},
            'MA20': {'category': '趋势指标', 'description': '20日移动平均线'},
            'MA60': {'category': '趋势指标', 'description': '60日移动平均线'},
            'SMA': {'category': '趋势指标', 'description': '简单移动平均线'},
            'EMA': {'category': '趋势指标', 'description': '指数移动平均线'},
            'WMA': {'category': '趋势指标', 'description': '加权移动平均线'},
            
            # 动量指标
            'RSI': {'category': '动量指标', 'description': '相对强弱指数'},
            'STOCH': {'category': '动量指标', 'description': '随机指标'},
            'WILLIAMS_R': {'category': '动量指标', 'description': '威廉指标'},
            'CCI': {'category': '动量指标', 'description': '商品通道指数'},
            'MOMENTUM': {'category': '动量指标', 'description': '动量指标'},
            'ROC': {'category': '动量指标', 'description': '变化率'},
            
            # 趋势指标
            'MACD': {'category': '趋势指标', 'description': 'MACD指标'},
            'BOLLINGER': {'category': '趋势指标', 'description': '布林带'},
            'SAR': {'category': '趋势指标', 'description': '抛物线SAR'},
            'ADX': {'category': '趋势指标', 'description': '平均趋向指数'},
            
            # 成交量指标
            'VWAP': {'category': '成交量指标', 'description': '成交量加权平均价格'},
            'OBV': {'category': '成交量指标', 'description': '能量潮'},
            'AD_LINE': {'category': '成交量指标', 'description': '累积/派发线'},
            'VOLUME_RSI': {'category': '成交量指标', 'description': '成交量相对强弱指数'},
            
            # 波动率指标
            'ATR': {'category': '波动率指标', 'description': '平均真实波幅'},
            'VOLATILITY': {'category': '波动率指标', 'description': '波动率'},
            'HISTORICAL_VOLATILITY': {'category': '波动率指标', 'description': '历史波动率'},
            
            # 复合指标
            'KDJ': {'category': '复合指标', 'description': 'KDJ指标'},
        }
    
    def validate_indicator_parameters(self, indicator: str, parameters: Dict[str, Any]) -> bool:
        """验证指标参数"""
        # 基本参数验证
        if indicator in ['MA5', 'MA10', 'MA20', 'MA60']:
            return True  # 这些指标有固定参数
        
        if indicator == 'RSI':
            period = parameters.get('period', 14)
            return isinstance(period, int) and 1 <= period <= 100
        
        if indicator == 'MACD':
            fast = parameters.get('fast_period', 12)
            slow = parameters.get('slow_period', 26)
            signal = parameters.get('signal_period', 9)
            return all(isinstance(p, int) and p > 0 for p in [fast, slow, signal]) and fast < slow
        
        if indicator == 'BOLLINGER':
            period = parameters.get('period', 20)
            std_dev = parameters.get('std_dev', 2.0)
            return isinstance(period, int) and period > 0 and isinstance(std_dev, (int, float)) and std_dev > 0
        
        return True  # 默认通过验证