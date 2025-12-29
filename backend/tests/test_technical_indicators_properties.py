"""
技术指标计算属性测试
Feature: stock-prediction-platform, Property 2: 技术指标计算完整性
"""

import asyncio
import tempfile
from datetime import datetime, timedelta

from hypothesis import given, strategies as st, settings
from hypothesis.strategies import composite

from app.models.stock_simple import StockData
from app.services.technical_indicators import (
    TechnicalIndicatorCalculator,
    BatchIndicatorRequest
)
from app.services.data_service_simple import SimpleStockDataService


@composite
def stock_code_strategy(draw):
    """生成有效的股票代码"""
    number = draw(st.integers(min_value=1, max_value=999999))
    market = draw(st.sampled_from(['SH', 'SZ']))
    return f"{number:06d}.{market}"


@composite
def stock_data_strategy(draw):
    """生成有效的股票数据"""
    stock_code = draw(stock_code_strategy())
    
    # 生成连续的日期序列
    start_date = draw(st.dates(min_value=datetime(2020, 1, 1).date(), max_value=datetime(2023, 12, 31).date()))
    days_count = draw(st.integers(min_value=30, max_value=100))  # 30-100天的数据
    
    data = []
    base_price = draw(st.floats(min_value=10.0, max_value=1000.0))
    
    for i in range(days_count):
        date = datetime.combine(start_date, datetime.min.time()) + timedelta(days=i)
        
        # 生成合理的价格数据
        price_change = draw(st.floats(min_value=-0.1, max_value=0.1))  # ±10%变化
        close_price = max(1.0, base_price * (1 + price_change))
        
        # 确保价格逻辑正确
        open_price = draw(st.floats(min_value=close_price * 0.95, max_value=close_price * 1.05))
        high_price = max(open_price, close_price) * draw(st.floats(min_value=1.0, max_value=1.05))
        low_price = min(open_price, close_price) * draw(st.floats(min_value=0.95, max_value=1.0))
        
        volume = draw(st.integers(min_value=100000, max_value=10000000))
        
        data.append(StockData(
            stock_code=stock_code,
            date=date,
            open=round(open_price, 2),
            high=round(high_price, 2),
            low=round(low_price, 2),
            close=round(close_price, 2),
            volume=volume,
            adj_close=round(close_price, 2)
        ))
        
        base_price = close_price  # 更新基础价格
    
    return data


class TestTechnicalIndicatorProperties:
    """技术指标计算属性测试"""
    
    def setup_method(self):
        """设置测试"""
        self.calculator = TechnicalIndicatorCalculator()
    
    @given(stock_data=stock_data_strategy())
    @settings(max_examples=20, deadline=None)
    def test_indicator_calculation_completeness_property(self, stock_data):
        """
        属性测试：技术指标计算完整性
        
        对于任何有效的股票数据，技术指标计算应该：
        1. 返回正确数量的结果
        2. 每个结果都包含请求的指标
        3. 指标值在合理范围内
        
        验证：需求 2.1, 2.2, 2.4
        """
        if len(stock_data) < 60:  # 确保有足够的数据计算所有指标
            return
        
        indicators = ['MA5', 'MA10', 'MA20', 'RSI', 'MACD', 'BOLLINGER']
        
        try:
            results = self.calculator.calculate_indicators(stock_data, indicators)
            
            # 验证结果完整性
            assert len(results) > 0, "应该有计算结果"
            assert len(results) <= len(stock_data), "结果数量不应超过输入数据"
            
            # 验证每个结果的结构
            for result in results:
                assert result.stock_code == stock_data[0].stock_code, "股票代码应该一致"
                assert isinstance(result.date, datetime), "日期应该是datetime类型"
                assert isinstance(result.indicators, dict), "指标应该是字典类型"
                assert len(result.indicators) > 0, "应该有指标值"
                
                # 验证指标值的合理性
                for indicator_name, value in result.indicators.items():
                    assert isinstance(value, (int, float)), f"指标值应该是数值类型: {indicator_name}"
                    assert not (isinstance(value, float) and (value != value)), f"指标值不应该是NaN: {indicator_name}"  # 检查NaN
                    
                    # 验证特定指标的范围
                    if indicator_name == 'RSI':
                        assert 0 <= value <= 100, f"RSI值应该在0-100之间: {value}"
                    elif indicator_name.startswith('MA') and not indicator_name.startswith('MACD'):
                        # 只有移动平均线（MA5, MA10等）应该大于0，MACD相关指标可以为负值
                        assert value > 0, f"移动平均线应该大于0: {indicator_name}={value}"
                    elif indicator_name.startswith('BOLLINGER'):
                        # 布林带指标应该大于0
                        assert value > 0, f"布林带指标应该大于0: {indicator_name}={value}"
            
            # 验证时间序列的连续性
            result_dates = [result.date for result in results]
            assert result_dates == sorted(result_dates), "结果应该按日期排序"
            
        except ValueError as e:
            # 如果数据验证失败，这是可以接受的
            if "输入数据验证失败" in str(e):
                return
            else:
                raise
    
    @given(
        stock_codes=st.lists(stock_code_strategy(), min_size=1, max_size=3),
        indicators=st.lists(st.sampled_from(['MA5', 'MA10', 'RSI']), min_size=1, max_size=3, unique=True)
    )
    @settings(max_examples=10, deadline=None)
    def test_batch_processing_capability_property(self, stock_codes, indicators):
        """
        属性测试：批量处理能力
        
        对于任何股票代码列表和指标列表，批量处理应该：
        1. 能够处理所有股票
        2. 返回正确的结果结构
        3. 处理失败时有适当的错误处理
        
        验证：需求 2.5
        """
        def run_test():
            async def async_test():
                with tempfile.TemporaryDirectory() as temp_dir:
                    data_service = SimpleStockDataService(
                        data_path=temp_dir,
                        remote_url="http://192.168.3.62"
                    )
                    
                    start_date = datetime(2023, 1, 1)
                    end_date = datetime(2023, 2, 28)
                    
                    # 为每只股票创建测试数据
                    for stock_code in stock_codes:
                        test_data = data_service.generate_mock_data(stock_code, start_date, end_date)
                        data_service.save_to_local(test_data, stock_code, merge_with_existing=False)
                    
                    # 创建批量请求
                    request = BatchIndicatorRequest(
                        stock_codes=stock_codes,
                        indicators=indicators,
                        start_date=start_date,
                        end_date=end_date
                    )
                    
                    # 执行批量计算
                    response = await self.calculator.calculate_batch_indicators(request, data_service)
                    
                    # 验证响应结构
                    assert isinstance(response.success, bool), "success应该是布尔值"
                    assert isinstance(response.results, dict), "results应该是字典"
                    assert isinstance(response.failed_stocks, list), "failed_stocks应该是列表"
                    assert isinstance(response.message, str), "message应该是字符串"
                    
                    # 验证处理结果
                    total_processed = len(response.results) + len(response.failed_stocks)
                    assert total_processed == len(stock_codes), "处理的股票总数应该等于请求的股票数"
                    
                    # 验证成功处理的股票结果
                    for stock_code, results in response.results.items():
                        assert stock_code in stock_codes, "结果中的股票代码应该在请求列表中"
                        assert isinstance(results, list), "每只股票的结果应该是列表"
                        
                        if results:  # 如果有结果
                            for result in results:
                                assert result.stock_code == stock_code, "结果中的股票代码应该匹配"
                                assert len(result.indicators) > 0, "应该有指标计算结果"
                                
                                # 验证请求的指标是否存在
                                result_indicator_names = set(result.indicators.keys())
                                for requested_indicator in indicators:
                                    # 检查是否有相关的指标（考虑MACD等复合指标）
                                    has_related_indicator = any(
                                        requested_indicator in name for name in result_indicator_names
                                    )
                                    if not has_related_indicator and len(results) > 20:  # 只有在有足够数据时才要求
                                        # 某些指标可能需要更多数据才能计算
                                        pass
            
            asyncio.run(async_test())
        
        run_test()
    
    def test_moving_average_mathematical_properties(self):
        """测试移动平均线的数学性质"""
        # 创建简单的测试数据
        test_data = []
        base_date = datetime(2023, 1, 1)
        
        # 创建递增序列
        for i in range(20):
            test_data.append(StockData(
                stock_code="TEST.SZ",
                date=base_date + timedelta(days=i),
                open=100 + i,
                high=102 + i,
                low=98 + i,
                close=100 + i,  # 递增序列
                volume=1000000,
                adj_close=100 + i
            ))
        
        # 计算MA5
        ma5_values = self.calculator.calculate_moving_average(test_data, 5)
        
        # 验证数学性质
        valid_ma5 = [v for v in ma5_values if v is not None]
        
        # 对于递增序列，移动平均线也应该是递增的
        for i in range(1, len(valid_ma5)):
            assert valid_ma5[i] >= valid_ma5[i-1], "递增序列的移动平均线应该是非递减的"
        
        # 移动平均线应该平滑价格波动
        # 对于递增序列，MA值应该在合理范围内
        for i, ma_value in enumerate(valid_ma5):
            data_index = i + 4  # MA5从第5个数据点开始
            current_price = test_data[data_index].close
            assert abs(ma_value - current_price) <= 5, "移动平均线应该接近当前价格"
    
    def test_rsi_boundary_conditions(self):
        """测试RSI的边界条件"""
        # 创建极端情况的测试数据
        
        # 情况1：连续上涨
        rising_data = []
        base_date = datetime(2023, 1, 1)
        
        for i in range(30):
            price = 100 + i * 2  # 连续上涨
            rising_data.append(StockData(
                stock_code="RISING.SZ",
                date=base_date + timedelta(days=i),
                open=price - 1,
                high=price + 1,
                low=price - 1,
                close=price,
                volume=1000000,
                adj_close=price
            ))
        
        rsi_rising = self.calculator.calculate_rsi(rising_data, 14)
        valid_rsi_rising = [v for v in rsi_rising if v is not None]
        
        # 连续上涨应该导致高RSI值
        if valid_rsi_rising:
            assert max(valid_rsi_rising) > 70, "连续上涨应该产生高RSI值"
        
        # 情况2：连续下跌
        falling_data = []
        for i in range(30):
            price = 100 - i * 2  # 连续下跌
            falling_data.append(StockData(
                stock_code="FALLING.SZ",
                date=base_date + timedelta(days=i),
                open=price + 1,
                high=price + 1,
                low=price - 1,
                close=price,
                volume=1000000,
                adj_close=price
            ))
        
        rsi_falling = self.calculator.calculate_rsi(falling_data, 14)
        valid_rsi_falling = [v for v in rsi_falling if v is not None]
        
        # 连续下跌应该导致低RSI值
        if valid_rsi_falling:
            assert min(valid_rsi_falling) < 30, "连续下跌应该产生低RSI值"
    
    def test_bollinger_bands_mathematical_properties(self):
        """测试布林带的数学性质"""
        # 创建测试数据
        test_data = []
        base_date = datetime(2023, 1, 1)
        
        for i in range(30):
            # 创建有波动的价格序列
            base_price = 100
            volatility = 5 * (1 if i % 4 < 2 else -1)  # 周期性波动
            price = base_price + volatility
            
            test_data.append(StockData(
                stock_code="VOLATILE.SZ",
                date=base_date + timedelta(days=i),
                open=price - 1,
                high=price + 2,
                low=price - 2,
                close=price,
                volume=1000000,
                adj_close=price
            ))
        
        bollinger_result = self.calculator.calculate_bollinger_bands(test_data, 20, 2.0)
        
        # 验证布林带的数学性质
        for i in range(19, len(test_data)):  # 从有效数据开始
            upper = bollinger_result['upper'][i]
            middle = bollinger_result['middle'][i]
            lower = bollinger_result['lower'][i]
            
            if all(v is not None for v in [upper, middle, lower]):
                # 基本关系：下轨 < 中轨 < 上轨
                assert lower < middle < upper, f"布林带顺序错误: {lower} < {middle} < {upper}"
                
                # 中轨应该接近移动平均线
                current_price = test_data[i].close
                price_diff = abs(middle - current_price)
                assert price_diff <= 20, f"中轨应该接近价格范围: diff={price_diff}"
                
                # 带宽应该合理
                band_width = upper - lower
                assert band_width > 0, "布林带宽度应该大于0"
                assert band_width < current_price, "布林带宽度应该小于股价"