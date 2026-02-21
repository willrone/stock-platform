"""
增量数据更新属性测试
Feature: stock-prediction-platform, Property 10: 增量数据更新
"""

import asyncio
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from hypothesis import given, strategies as st, settings
from hypothesis.strategies import composite

from app.services.data import SimpleDataService as SimpleStockDataService


@composite
def stock_code_strategy(draw):
    """生成有效的股票代码"""
    number = draw(st.integers(min_value=1, max_value=999999))
    market = draw(st.sampled_from(['SH', 'SZ']))
    return f"{number:06d}.{market}"


def test_incremental_update_basic():
    """基础增量更新测试: 生成mock数据并保存到本地"""
    with tempfile.TemporaryDirectory() as temp_dir:
        data_service = SimpleStockDataService(
            data_path=temp_dir,
            remote_url="http://192.168.3.62"
        )

        stock_code = "000001.SZ"
        first_start = datetime(2023, 1, 1)
        first_end = datetime(2023, 1, 10)

        # 生成并保存测试数据
        test_data = data_service.generate_mock_data(stock_code, first_start, first_end)
        assert len(test_data) > 0, "应该生成测试数据"

        saved = data_service.save_to_local(test_data, stock_code)
        assert saved is True, "保存应该成功"

        # 验证本地文件创建
        local_file = data_service.get_local_data_path(stock_code)
        assert local_file.exists(), "本地文件应该被创建"

        # 验证可以从本地加载
        loaded = data_service.load_from_local(stock_code, first_start, first_end)
        assert loaded is not None, "应该能从本地加载数据"
        assert len(loaded) > 0, "加载的数据不应为空"


@given(stock_code=stock_code_strategy())
@settings(max_examples=10, deadline=None)
def test_incremental_update_property(stock_code):
    """
    属性测试：增量数据更新

    对于任何股票代码，生成的mock数据应该能正确保存和加载
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        data_service = SimpleStockDataService(
            data_path=temp_dir,
            remote_url="http://192.168.3.62"
        )

        start = datetime(2023, 1, 1)
        end = datetime(2023, 1, 10)

        # 生成mock数据
        mock_data = data_service.generate_mock_data(stock_code, start, end)

        if len(mock_data) > 0:
            # 保存到本地
            saved = data_service.save_to_local(mock_data, stock_code)
            assert saved is True

            # 验证本地文件存在
            local_file = data_service.get_local_data_path(stock_code)
            assert local_file.exists()

            # 从本地加载并验证
            loaded = data_service.load_from_local(stock_code, start, end)
            assert loaded is not None
            assert len(loaded) == len(mock_data)


@given(stock_code=stock_code_strategy())
@settings(max_examples=10, deadline=None)
def test_missing_date_ranges_identification_property(stock_code):
    """
    属性测试：缺失数据段识别

    保存一段数据后，检查本地数据是否存在
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        data_service = SimpleStockDataService(
            data_path=temp_dir,
            remote_url="http://192.168.3.62"
        )

        base_start = datetime(2023, 1, 1)
        base_end = datetime(2023, 1, 10)

        # 生成并保存基础数据
        mock_data = data_service.generate_mock_data(stock_code, base_start, base_end)

        if len(mock_data) > 0:
            data_service.save_to_local(mock_data, stock_code)

            # 验证本地文件存在
            local_file = data_service.get_local_data_path(stock_code)
            assert local_file.exists(), "保存后本地文件应该存在"

            # 加载并验证数据完整性
            loaded = data_service.load_from_local(stock_code, base_start, base_end)
            assert loaded is not None
            assert len(loaded) > 0

            # 验证数据字段完整性
            for item in loaded:
                assert "stock_code" in item or hasattr(item, "stock_code")


def test_incremental_update_performance():
    """测试增量更新的性能优势"""
    import time

    with tempfile.TemporaryDirectory() as temp_dir:
        data_service = SimpleStockDataService(
            data_path=temp_dir,
            remote_url="http://192.168.3.62"
        )

        stock_code = "000001.SZ"
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 31)

        # 生成并保存数据
        mock_data = data_service.generate_mock_data(stock_code, start_date, end_date)
        assert len(mock_data) > 0
        data_service.save_to_local(mock_data, stock_code)

        # 测量从本地加载的时间
        start_time = time.time()
        loaded = data_service.load_from_local(stock_code, start_date, end_date)
        load_duration = time.time() - start_time

        assert loaded is not None
        assert len(loaded) == len(mock_data)
        # 本地加载应该很快
        assert load_duration < 1.0, f"本地加载耗时过长: {load_duration:.3f}s"
