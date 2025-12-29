"""
数据服务测试
"""

import asyncio
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from app.models.stock_simple import DataSyncRequest, StockData
from app.services.data_service_simple import SimpleStockDataService


class TestSimpleStockDataService:
    """简化股票数据服务测试"""
    
    @pytest.fixture
    def temp_data_path(self):
        """临时数据目录"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def data_service(self, temp_data_path):
        """数据服务实例"""
        return SimpleStockDataService(
            data_path=temp_data_path,
            remote_url="http://192.168.3.62"
        )
    
    def test_service_initialization(self, data_service, temp_data_path):
        """测试服务初始化"""
        assert data_service.remote_url == "http://192.168.3.62"
        assert data_service.data_path == Path(temp_data_path)
        assert data_service.stocks_path.exists()
    
    def test_get_local_data_path(self, data_service):
        """测试本地数据路径生成"""
        path = data_service.get_local_data_path("000001.SZ")
        assert path.name == "000001.SZ.json"
        assert "stocks" in str(path)
    
    def test_check_local_data_exists_no_file(self, data_service):
        """测试检查本地数据 - 文件不存在"""
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 31)
        
        exists = data_service.check_local_data_exists("000001.SZ", start_date, end_date)
        assert exists is False
    
    def test_generate_mock_data(self, data_service):
        """测试生成模拟数据"""
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 7)  # 一周的数据
        
        data = data_service.generate_mock_data("000001.SZ", start_date, end_date)
        
        # 验证数据结构
        assert len(data) == 5  # 5个工作日
        
        for item in data:
            assert 'stock_code' in item
            assert 'date' in item
            assert 'open' in item
            assert 'high' in item
            assert 'low' in item
            assert 'close' in item
            assert 'volume' in item
            assert 'adj_close' in item
            
            # 验证价格逻辑
            assert item['high'] >= item['open']
            assert item['high'] >= item['close']
            assert item['low'] <= item['open']
            assert item['low'] <= item['close']
            assert item['volume'] > 0
    
    def test_save_and_load_local_data(self, data_service):
        """测试保存和加载本地数据"""
        # 生成测试数据
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 7)
        stock_code = "000001.SZ"
        
        data = data_service.generate_mock_data(stock_code, start_date, end_date)
        
        # 保存数据
        success = data_service.save_to_local(data, stock_code)
        assert success is True
        
        # 验证文件存在
        data_file = data_service.get_local_data_path(stock_code)
        assert data_file.exists()
        
        # 加载数据
        loaded_data = data_service.load_from_local(stock_code, start_date, end_date)
        assert loaded_data is not None
        assert len(loaded_data) == len(data)
        
        # 验证数据内容
        for original, loaded in zip(data, loaded_data):
            assert original['stock_code'] == loaded['stock_code']
            assert original['date'] == loaded['date']
            assert original['open'] == loaded['open']
    
    def test_check_local_data_exists_with_file(self, data_service):
        """测试检查本地数据 - 文件存在"""
        # 先保存一些数据
        start_date = datetime(2023, 1, 2)  # 周一开始
        end_date = datetime(2023, 1, 6)    # 周五结束
        stock_code = "000001.SZ"
        
        data = data_service.generate_mock_data(stock_code, start_date, end_date)
        data_service.save_to_local(data, stock_code)
        
        # 检查数据存在
        exists = data_service.check_local_data_exists(stock_code, start_date, end_date)
        assert exists is True
        
        # 检查超出范围的日期
        future_start = datetime(2023, 2, 1)
        future_end = datetime(2023, 2, 7)
        exists = data_service.check_local_data_exists(stock_code, future_start, future_end)
        assert exists is False
    
    @pytest.mark.asyncio
    async def test_check_remote_service_status(self, data_service):
        """测试检查远端服务状态"""
        status = await data_service.check_remote_service_status()
        
        assert status.service_url == "http://192.168.3.62"
        assert isinstance(status.is_available, bool)
        assert isinstance(status.last_check, datetime)
        assert isinstance(status.response_time_ms, (int, float))
        
        if not status.is_available:
            assert status.error_message is not None
    
    @pytest.mark.asyncio
    async def test_fetch_remote_data(self, data_service):
        """测试从远端获取数据"""
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 7)
        stock_code = "000001.SZ"
        
        data = await data_service.fetch_remote_data(stock_code, start_date, end_date)
        
        # 数据可能为None（如果远端服务不可用）
        if data is not None:
            assert len(data) > 0
            assert all('stock_code' in item for item in data)
            assert all('date' in item for item in data)
    
    @pytest.mark.asyncio
    async def test_get_stock_data_local_priority(self, data_service):
        """测试获取股票数据 - 本地优先策略"""
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 7)
        stock_code = "000001.SZ"
        
        # 第一次获取（应该从远端获取并保存到本地）
        data1 = await data_service.get_stock_data(stock_code, start_date, end_date)
        
        if data1 is not None:
            assert len(data1) > 0
            assert all(isinstance(item, StockData) for item in data1)
            
            # 第二次获取（应该从本地获取）
            data2 = await data_service.get_stock_data(stock_code, start_date, end_date)
            
            assert data2 is not None
            assert len(data2) == len(data1)
            
            # 验证数据一致性
            for item1, item2 in zip(data1, data2):
                assert item1.stock_code == item2.stock_code
                assert item1.date == item2.date
                assert item1.close == item2.close
    
    @pytest.mark.asyncio
    async def test_get_stock_data_force_remote(self, data_service):
        """测试获取股票数据 - 强制远端获取"""
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 7)
        stock_code = "000002.SZ"
        
        # 强制从远端获取
        data = await data_service.get_stock_data(
            stock_code, start_date, end_date, force_remote=True
        )
        
        # 数据可能为None（如果远端服务不可用）
        if data is not None:
            assert len(data) > 0
            assert all(isinstance(item, StockData) for item in data)
    
    @pytest.mark.asyncio
    async def test_sync_multiple_stocks(self, data_service):
        """测试批量同步多只股票"""
        request = DataSyncRequest(
            stock_codes=["000001.SZ", "000002.SZ", "600000.SH"],
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 7),
            force_update=False
        )
        
        response = await data_service.sync_multiple_stocks(request)
        
        assert isinstance(response.success, bool)
        assert isinstance(response.synced_stocks, list)
        assert isinstance(response.failed_stocks, list)
        assert isinstance(response.total_records, int)
        assert isinstance(response.message, str)
        
        # 验证股票代码总数
        total_stocks = len(response.synced_stocks) + len(response.failed_stocks)
        assert total_stocks == len(request.stock_codes)
    
    def test_dict_list_to_stock_data_conversion(self, data_service):
        """测试字典列表到StockData的转换"""
        # 创建测试数据
        dict_data = [
            {
                'stock_code': '000001.SZ',
                'date': '2023-01-01T00:00:00',
                'open': 10.0,
                'high': 11.0,
                'low': 9.5,
                'close': 10.5,
                'volume': 1000000,
                'adj_close': 10.5
            }
        ]
        
        stock_data_list = data_service._dict_list_to_stock_data(dict_data)
        
        assert len(stock_data_list) == 1
        
        stock_data = stock_data_list[0]
        assert isinstance(stock_data, StockData)
        assert stock_data.stock_code == '000001.SZ'
        assert stock_data.open == 10.0
        assert stock_data.high == 11.0
        assert stock_data.low == 9.5
        assert stock_data.close == 10.5
        assert stock_data.volume == 1000000
        assert stock_data.adj_close == 10.5


class TestDataModels:
    """数据模型测试"""
    
    def test_stock_data_creation(self):
        """测试StockData创建"""
        stock_data = StockData(
            stock_code="000001.SZ",
            date=datetime(2023, 1, 1),
            open=10.0,
            high=11.0,
            low=9.5,
            close=10.5,
            volume=1000000,
            adj_close=10.5
        )
        
        assert stock_data.stock_code == "000001.SZ"
        assert stock_data.open == 10.0
        assert stock_data.volume == 1000000
    
    def test_stock_data_to_dict(self):
        """测试StockData转字典"""
        stock_data = StockData(
            stock_code="000001.SZ",
            date=datetime(2023, 1, 1),
            open=10.0,
            high=11.0,
            low=9.5,
            close=10.5,
            volume=1000000
        )
        
        data_dict = stock_data.to_dict()
        
        assert data_dict['stock_code'] == "000001.SZ"
        assert data_dict['open'] == 10.0
        assert data_dict['volume'] == 1000000
        assert 'date' in data_dict
    
    def test_data_sync_request_creation(self):
        """测试DataSyncRequest创建"""
        request = DataSyncRequest(
            stock_codes=["000001.SZ", "000002.SZ"],
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 31),
            force_update=True
        )
        
        assert len(request.stock_codes) == 2
        assert request.force_update is True
        assert request.start_date == datetime(2023, 1, 1)