"""
技术指标计算服务
实现各种股票技术指标的计算
"""

import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import math
from dataclasses import dataclass

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
    """技术指标计算器"""
    
    def __init__(self):
        self.supported_indicators = {
            'MA5', 'MA10', 'MA20', 'MA60',  # 移动平均线
            'RSI', 'MACD', 'BOLLINGER'      # 其他技术指标
        }
    
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
        """计算指定的技术指标"""
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
        
        # RSI
        if 'RSI' in indicators:
            calculated_indicators['RSI'] = self.calculate_rsi(data)
        
        # MACD
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
    
    async def calculate_batch_indicators(
        self, 
        request: BatchIndicatorRequest,
        data_service
    ) -> BatchIndicatorResponse:
        """批量计算多只股票的技术指标"""
        print(f"开始批量计算技术指标: {len(request.stock_codes)} 只股票")
        
        results = {}
        failed_stocks = []
        
        # 并发控制
        semaphore = asyncio.Semaphore(3)  # 最多3个并发任务
        
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