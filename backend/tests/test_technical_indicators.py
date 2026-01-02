"""
技术指标计算服务测试
"""

import asyncio
import tempfile
from datetime import datetime, timedelta

import pytest

from app.models.stock_simple import StockData
from app.services.prediction import (
    TechnicalIndicatorCalculator,
    BatchIndicatorRequest,
    TechnicalIndicatorResult
)
from app.services.data.simple_data_service import SimpleDataService


class TestTechnicalIndicatorCalculator:
    """技术指标计算器测试"""
    
    def setup_method(self):
        """设置测试"""
        self.calculator = TechnicalIndicatorCalculator()
        
        # 创建测试数据
        self.test_data = []
        base_date = datetime(2023, 1, 1)
        base_price = 100.0
        
        for i in range(30):  # 30天的数据
            date = base_date + timedelta(days=i)
            # 模拟价格波动
            price_change = (i % 5 - 2) * 2  # -4, -2, 0, 2, 4 的循环
            close_price = base_price + price_change + (i * 0.5)  # 总体上涨趋势
            
            self.test_data.append(StockData(
                stock_code="000001.SZ",
                date=date,
                open=close_price - 1,
                high=close_price + 2,
                low=close_price - 2,
                close=close_price,
                volume=1000000 + i * 10000,
                adj_close=close_price
            ))
    
    def test_validate_data(self):
        """测试数据验证"""
        # 有效数据
        assert self.calculator.validate_data(self.test_data) == True
        
        # 空数据
        assert self.calculator.validate_data([]) == False
        
        # 无效价格数据
        invalid_data = [StockData(
            stock_code="000001.SZ",
            date=datetime(2023, 1, 1),
            open=0,  # 无效的开盘价
            high=100,
            low=90,
            close=95,
            volume=1000000
        )]
        assert self.calculator.validate_data(invalid_data) == False
        
        # 高低价逻辑错误
        invalid_data2 = [StockData(
            stock_code="000001.SZ",
            date=datetime(2023, 1, 1),
            open=100,
            high=90,  # 最高价低于最低价
            low=95,
            close=95,
            volume=1000000
        )]
        assert self.calculator.validate_data(invalid_data2) == False
    
    def test_moving_average_calculation(self):
        """测试移动平均线计算"""
        # 测试MA5
        ma5_values = self.calculator.calculate_moving_average(self.test_data, 5)
        
        # 前4个值应该是None
        assert all(v is None for v in ma5_values[:4])
        
        # 第5个值应该是前5天收盘价的平均
        expected_ma5 = sum(item.close for item in self.test_data[:5]) / 5
        assert abs(ma5_values[4] - expected_ma5) < 0.001
        
        # 测试数据不足的情况
        short_data = self.test_data[:3]
        ma10_values = self.calculator.calculate_moving_average(short_data, 10)
        assert all(v is None for v in ma10_values)
    
    def test_rsi_calculation(self):
        """测试RSI计算"""
        rsi_values = self.calculator.calculate_rsi(self.test_data, 14)
        
        # 前14个值应该是None
        assert all(v is None for v in rsi_values[:14])
        
        # RSI值应该在0-100之间
        valid_rsi_values = [v for v in rsi_values if v is not None]
        assert all(0 <= v <= 100 for v in valid_rsi_values)
        
        # 测试数据不足的情况
        short_data = self.test_data[:10]
        rsi_short = self.calculator.calculate_rsi(short_data, 14)
        assert all(v is None for v in rsi_short)
    
    def test_macd_calculation(self):
        """测试MACD计算"""
        macd_result = self.calculator.calculate_macd(self.test_data)
        
        assert 'macd' in macd_result
        assert 'signal' in macd_result
        assert 'histogram' in macd_result
        
        # 检查数据长度
        assert len(macd_result['macd']) == len(self.test_data)
        assert len(macd_result['signal']) == len(self.test_data)
        assert len(macd_result['histogram']) == len(self.test_data)
        
        # 前面的值应该是None
        assert macd_result['macd'][0] is None
        assert macd_result['signal'][0] is None
        assert macd_result['histogram'][0] is None
    
    def test_bollinger_bands_calculation(self):
        """测试布林带计算"""
        bollinger_result = self.calculator.calculate_bollinger_bands(self.test_data, 20)
        
        assert 'upper' in bollinger_result
        assert 'middle' in bollinger_result
        assert 'lower' in bollinger_result
        
        # 检查数据长度
        assert len(bollinger_result['upper']) == len(self.test_data)
        assert len(bollinger_result['middle']) == len(self.test_data)
        assert len(bollinger_result['lower']) == len(self.test_data)
        
        # 前19个值应该是None
        assert all(v is None for v in bollinger_result['upper'][:19])
        assert all(v is None for v in bollinger_result['middle'][:19])
        assert all(v is None for v in bollinger_result['lower'][:19])
        
        # 有效值应该满足：下轨 < 中轨 < 上轨
        for i in range(19, len(self.test_data)):
            upper = bollinger_result['upper'][i]
            middle = bollinger_result['middle'][i]
            lower = bollinger_result['lower'][i]
            
            if all(v is not None for v in [upper, middle, lower]):
                assert lower < middle < upper
    
    def test_calculate_indicators(self):
        """测试综合指标计算"""
        indicators = ['MA5', 'MA10', 'RSI', 'MACD']
        results = self.calculator.calculate_indicators(self.test_data, indicators)
        
        # 应该有结果
        assert len(results) > 0
        
        # 检查结果结构
        for result in results:
            assert isinstance(result, TechnicalIndicatorResult)
            assert result.stock_code == "000001.SZ"
            assert isinstance(result.date, datetime)
            assert isinstance(result.indicators, dict)
        
        # 检查指标是否存在
        all_indicators = set()
        for result in results:
            all_indicators.update(result.indicators.keys())
        
        # 应该包含请求的指标（或其变体）
        assert 'MA5' in all_indicators or any('MA5' in ind for ind in all_indicators)
        assert 'RSI' in all_indicators or any('RSI' in ind for ind in all_indicators)
    
    def test_unsupported_indicators(self):
        """测试不支持的指标"""
        with pytest.raises(ValueError, match="不支持的指标"):
            self.calculator.calculate_indicators(self.test_data, ['INVALID_INDICATOR'])
    
    def test_invalid_data(self):
        """测试无效数据"""
        invalid_data = [StockData(
            stock_code="000001.SZ",
            date=datetime(2023, 1, 1),
            open=0,  # 无效价格
            high=100,
            low=90,
            close=95,
            volume=1000000
        )]
        
        with pytest.raises(ValueError, match="输入数据验证失败"):
            self.calculator.calculate_indicators(invalid_data, ['MA5'])


def test_batch_indicator_calculation():
    """测试批量指标计算 - 简化版本"""
    async def run_test():
        # 创建技术指标计算器
        calculator = TechnicalIndicatorCalculator()
        
        # 创建模拟数据服务
        class MockDataService:
            async def get_stock_data(self, stock_code, start_date, end_date):
                # 返回模拟数据
                test_data = []
                base_date = start_date
                base_price = 100.0
                
                for i in range(30):  # 30天的数据
                    date = base_date + timedelta(days=i)
                    close_price = base_price + (i * 0.1) + ((i % 5 - 2) * 2)
                    
                    test_data.append(StockData(
                        stock_code=stock_code,
                        date=date,
                        open=close_price - 1,
                        high=close_price + 2,
                        low=close_price - 2,
                        close=close_price,
                        volume=1000000 + i * 10000,
                        adj_close=close_price
                    ))
                
                return test_data
        
        data_service = MockDataService()
        
        # 准备测试数据
        stock_codes = ["000001.SZ", "000002.SZ"]
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 31)
        
        # 创建批量请求
        request = BatchIndicatorRequest(
            stock_codes=stock_codes,
            indicators=['MA5', 'MA10', 'RSI'],
            start_date=start_date,
            end_date=end_date
        )
        
        # 执行批量计算
        response = await calculator.calculate_batch_indicators(request, data_service)
        
        # 验证响应
        assert isinstance(response.success, bool)
        assert isinstance(response.results, dict)
        assert isinstance(response.failed_stocks, list)
        assert isinstance(response.message, str)
        
        # 如果有成功的结果，验证结构
        if response.results:
            for stock_code, results in response.results.items():
                assert stock_code in stock_codes
                assert isinstance(results, list)
                
                if results:  # 如果有结果
                    for result in results:
                        assert isinstance(result, TechnicalIndicatorResult)
                        assert result.stock_code == stock_code
                        assert isinstance(result.indicators, dict)
    
    asyncio.run(run_test())


def test_performance_with_large_dataset():
    """测试大数据集的性能"""
    calculator = TechnicalIndicatorCalculator()
    
    # 创建大数据集（1年的数据）
    large_data = []
    base_date = datetime(2022, 1, 1)
    base_price = 100.0
    
    for i in range(365):
        date = base_date + timedelta(days=i)
        # 跳过周末
        if date.weekday() < 5:
            close_price = base_price + (i * 0.1) + ((i % 10 - 5) * 2)
            large_data.append(StockData(
                stock_code="000001.SZ",
                date=date,
                open=close_price - 1,
                high=close_price + 2,
                low=close_price - 2,
                close=close_price,
                volume=1000000 + i * 1000,
                adj_close=close_price
            ))
    
    # 计算所有指标
    import time
    start_time = time.time()
    
    results = calculator.calculate_indicators(
        large_data, 
        ['MA5', 'MA10', 'MA20', 'MA60', 'RSI', 'MACD', 'BOLLINGER']
    )
    
    end_time = time.time()
    calculation_time = end_time - start_time
    
    print(f"大数据集计算耗时: {calculation_time:.3f}s")
    print(f"数据量: {len(large_data)} 条")
    print(f"结果数量: {len(results)} 条")
    
    # 验证结果
    assert len(results) > 0
    assert calculation_time < 5.0  # 应该在5秒内完成
    
    # 验证结果质量
    for result in results[-10:]:  # 检查最后10条记录
        assert len(result.indicators) > 0
        # 移动平均线应该存在
        assert any('MA' in key for key in result.indicators.keys())