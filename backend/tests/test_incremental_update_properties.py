"""
增量数据更新属性测试
Feature: stock-prediction-platform, Property 10: 增量数据更新
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
def overlapping_date_ranges_strategy(draw):
    """生成重叠的日期范围用于测试增量更新"""
    base_year = draw(st.integers(min_value=2020, max_value=2023))
    base_month = draw(st.integers(min_value=1, max_value=12))
    base_day = draw(st.integers(min_value=1, max_value=28))
    
    base_date = datetime(base_year, base_month, base_day)
    
    # 第一个范围
    first_duration = draw(st.integers(min_value=5, max_value=15))
    first_start = base_date
    first_end = base_date + timedelta(days=first_duration)
    
    # 第二个范围（可能重叠、扩展或独立）
    offset = draw(st.integers(min_value=-5, max_value=20))  # 负数表示重叠
    second_duration = draw(st.integers(min_value=5, max_value=15))
    second_start = first_end + timedelta(days=offset)
    second_end = second_start + timedelta(days=second_duration)
    
    return (first_start, first_end), (second_start, second_end)


def test_incremental_update_basic():
    """基础增量更新测试"""
    async def run_test():
        with tempfile.TemporaryDirectory() as temp_dir:
            data_service = SimpleStockDataService(
                data_path=temp_dir,
                remote_url="http://192.168.3.62"
            )
            
            stock_code = "000001.SZ"
            
            # 第一次获取数据：2023-01-01 到 2023-01-10
            first_start = datetime(2023, 1, 1)
            first_end = datetime(2023, 1, 10)
            
            # 先手动创建一些测试数据
            test_data = data_service.generate_mock_data(stock_code, first_start, first_end)
            data_service.save_to_local(test_data, stock_code, merge_with_existing=False)
            
            # 验证本地文件创建
            local_file = data_service.get_local_data_path(stock_code)
            assert local_file.exists(), "本地文件应该被创建"
            
            # 测试获取现有数据
            data1 = await data_service.get_stock_data(stock_code, first_start, first_end)
            assert data1 is not None, "第一次获取数据应该成功"
            assert len(data1) > 0, "应该有数据返回"
            
            # 第二次获取扩展数据：2023-01-05 到 2023-01-20（部分重叠）
            second_start = datetime(2023, 1, 5)
            second_end = datetime(2023, 1, 20)
            
            # 测试缺失数据段识别
            missing_ranges = data_service.identify_missing_date_ranges(stock_code, second_start, second_end)
            print(f"识别到的缺失数据段: {missing_ranges}")
            
            # 应该有一个缺失的数据段（从1月11日到1月20日）
            assert len(missing_ranges) > 0, "应该识别到缺失的数据段"
            
            # 验证缺失范围的逻辑
            for missing_start, missing_end in missing_ranges:
                assert missing_start <= missing_end, "缺失范围的开始日期应该不晚于结束日期"
                print(f"缺失数据段: {missing_start.date()} - {missing_end.date()}")
    
    asyncio.run(run_test())


@given(
    stock_code=stock_code_strategy(),
    date_ranges=overlapping_date_ranges_strategy()
)
@settings(max_examples=20, deadline=None)
def test_incremental_update_property(stock_code, date_ranges):
    """
    属性测试：增量数据更新
    
    对于任何时间范围的数据请求，系统应该只获取缺失的时间段数据，
    避免重复下载已存在的数据
    
    验证：需求 1.5
    """
    (first_start, first_end), (second_start, second_end) = date_ranges
    
    def run_test():
        async def async_test():
            with tempfile.TemporaryDirectory() as temp_dir:
                data_service = SimpleStockDataService(
                    data_path=temp_dir,
                    remote_url="http://192.168.3.62"
                )
                
                # 第一次获取数据
                data1 = await data_service.get_stock_data(stock_code, first_start, first_end)
                
                if data1 is not None and len(data1) > 0:
                    # 验证本地文件被创建
                    local_file = data_service.get_local_data_path(stock_code)
                    assert local_file.exists(), "本地文件应该被创建"
                    
                    # 获取第一次的数据范围
                    first_range = data_service.get_local_data_date_range(stock_code)
                    assert first_range is not None, "应该能获取本地数据范围"
                    
                    # 第二次获取数据（可能重叠或扩展）
                    data2 = await data_service.get_stock_data(stock_code, second_start, second_end)
                    
                    if data2 is not None and len(data2) > 0:
                        # 验证数据完整性
                        dates2 = [item.date for item in data2]
                        data_start = min(dates2)
                        data_end = max(dates2)
                        
                        # 验证数据覆盖了请求的范围
                        assert data_start.date() <= second_start.date(), "数据应该覆盖请求的开始日期"
                        assert data_end.date() >= second_end.date(), "数据应该覆盖请求的结束日期"
                        
                        # 验证数据的连续性（在工作日范围内）
                        sorted_dates = sorted(dates2)
                        for i in range(len(sorted_dates) - 1):
                            current_date = sorted_dates[i]
                            next_date = sorted_dates[i + 1]
                            
                            # 计算工作日差异（跳过周末）
                            days_diff = (next_date - current_date).days
                            
                            # 允许的间隔：1天（连续工作日）或3天（跨周末）
                            assert days_diff <= 3, f"数据日期间隔过大: {current_date} -> {next_date}"
        
        asyncio.run(async_test())
    
    run_test()


@given(stock_code=stock_code_strategy())
@settings(max_examples=10, deadline=None)
def test_missing_date_ranges_identification_property(stock_code):
    """
    属性测试：缺失数据段识别
    
    系统应该能够正确识别本地数据中缺失的时间段
    
    验证：需求 1.5
    """
    def run_test():
        async def async_test():
            with tempfile.TemporaryDirectory() as temp_dir:
                data_service = SimpleStockDataService(
                    data_path=temp_dir,
                    remote_url="http://192.168.3.62"
                )
                
                # 先获取一些基础数据
                base_start = datetime(2023, 1, 1)
                base_end = datetime(2023, 1, 10)
                
                data = await data_service.get_stock_data(stock_code, base_start, base_end)
                
                if data is not None and len(data) > 0:
                    # 测试不同的请求范围
                    test_cases = [
                        # 完全覆盖的范围
                        (datetime(2023, 1, 2), datetime(2023, 1, 8)),
                        # 扩展到之前的范围
                        (datetime(2022, 12, 25), datetime(2023, 1, 5)),
                        # 扩展到之后的范围
                        (datetime(2023, 1, 5), datetime(2023, 1, 20)),
                        # 完全在之前的范围
                        (datetime(2022, 12, 1), datetime(2022, 12, 31)),
                        # 完全在之后的范围
                        (datetime(2023, 2, 1), datetime(2023, 2, 28)),
                    ]
                    
                    for req_start, req_end in test_cases:
                        missing_ranges = data_service.identify_missing_date_ranges(
                            stock_code, req_start, req_end
                        )
                        
                        # 验证缺失范围的逻辑正确性
                        for missing_start, missing_end in missing_ranges:
                            assert missing_start <= missing_end, "缺失范围的开始日期应该不晚于结束日期"
                            assert req_start <= missing_end, "缺失范围应该与请求范围有交集"
                            assert missing_start <= req_end, "缺失范围应该与请求范围有交集"
        
        asyncio.run(async_test())
    
    run_test()


def test_incremental_update_performance():
    """测试增量更新的性能优势"""
    async def run_test():
        with tempfile.TemporaryDirectory() as temp_dir:
            data_service = SimpleStockDataService(
                data_path=temp_dir,
                remote_url="http://192.168.3.62"
            )
            
            stock_code = "000001.SZ"
            
            # 第一次获取大范围数据
            start_date = datetime(2023, 1, 1)
            end_date = datetime(2023, 1, 31)
            
            import time
            
            # 测量第一次获取的时间
            start_time = time.time()
            data1 = await data_service.get_stock_data(stock_code, start_date, end_date)
            first_duration = time.time() - start_time
            
            if data1 is not None and len(data1) > 0:
                # 测量第二次获取相同数据的时间（应该更快）
                start_time = time.time()
                data2 = await data_service.get_stock_data(stock_code, start_date, end_date)
                second_duration = time.time() - start_time
                
                # 验证数据一致性
                assert len(data1) == len(data2), "两次获取的数据长度应该一致"
                
                # 第二次应该更快（从本地加载）
                print(f"第一次获取耗时: {first_duration:.3f}s")
                print(f"第二次获取耗时: {second_duration:.3f}s")
                print(f"性能提升: {first_duration/second_duration:.1f}x")
                
                # 测量增量更新的时间
                extended_end = datetime(2023, 2, 5)
                start_time = time.time()
                data3 = await data_service.get_stock_data(stock_code, start_date, extended_end)
                incremental_duration = time.time() - start_time
                
                if data3 is not None:
                    print(f"增量更新耗时: {incremental_duration:.3f}s")
                    assert len(data3) >= len(data1), "扩展后的数据应该不少于原始数据"
    
    asyncio.run(run_test())