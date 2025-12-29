"""
数据服务属性测试 - 简化版本
"""

import asyncio
import tempfile
from datetime import datetime, timedelta

from hypothesis import given, strategies as st, settings
from hypothesis.strategies import composite

from app.models.stock_simple import StockData
from app.services.data_service_simple import SimpleStockDataService


@composite
def stock_code_strategy(draw):
    """生成有效的股票代码"""
    number = draw(st.integers(min_value=1, max_value=999999))
    market = draw(st.sampled_from(['SH', 'SZ']))
    return f"{number:06d}.{market}"


@composite
def date_range_strategy(draw):
    """生成有效的日期范围"""
    start_year = draw(st.integers(min_value=2020, max_value=2024))
    start_month = draw(st.integers(min_value=1, max_value=12))
    start_day = draw(st.integers(min_value=1, max_value=28))
    
    start_date = datetime(start_year, start_month, start_day)
    days_diff = draw(st.integers(min_value=1, max_value=30))
    end_date = start_date + timedelta(days=days_diff)
    
    return start_date, end_date


def test_data_service_basic():
    """基础数据服务测试"""
    async def run_test():
        with tempfile.TemporaryDirectory() as temp_dir:
            data_service = SimpleStockDataService(
                data_path=temp_dir,
                remote_url="http://192.168.3.62"
            )
            
            # 测试基本功能
            stock_code = "000001.SZ"
            start_date = datetime(2023, 1, 1)
            end_date = datetime(2023, 1, 10)
            
            data = await data_service.get_stock_data(stock_code, start_date, end_date)
            
            if data is not None:
                assert len(data) > 0, "应该返回数据"
                assert all(isinstance(item, StockData) for item in data), "所有项目应该是StockData类型"
    
    asyncio.run(run_test())


@given(
    stock_code=stock_code_strategy(),
    date_range=date_range_strategy()
)
@settings(max_examples=10, deadline=None)
def test_local_first_strategy_property(stock_code, date_range):
    """
    属性测试：数据获取本地优先策略
    """
    start_date, end_date = date_range
    
    def run_test():
        async def async_test():
            with tempfile.TemporaryDirectory() as temp_dir:
                data_service = SimpleStockDataService(
                    data_path=temp_dir,
                    remote_url="http://192.168.3.62"
                )
                
                # 第一次请求：应该从远端获取数据
                data1 = await data_service.get_stock_data(stock_code, start_date, end_date)
                
                if data1 is not None:
                    # 验证数据文件被创建
                    local_file = data_service.get_local_data_path(stock_code)
                    assert local_file.exists(), f"本地文件应该被创建: {local_file}"
                    
                    # 第二次请求：应该从本地获取
                    data2 = await data_service.get_stock_data(stock_code, start_date, end_date)
                    
                    if data2 is not None:
                        # 验证数据长度一致
                        assert len(data1) == len(data2), "数据长度应该一致"
                        
                        # 验证关键字段一致性
                        if len(data1) > 0:
                            assert data1[0].stock_code == data2[0].stock_code, "股票代码应该一致"
                            assert data1[0].date == data2[0].date, "日期应该一致"
                            assert data1[0].close == data2[0].close, "收盘价应该一致"
        
        asyncio.run(async_test())
    
    run_test()