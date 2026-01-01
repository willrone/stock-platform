"""
API路由属性测试
验证API路由调用真实服务的正确性属性
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from hypothesis import given, strategies as st, settings
from hypothesis.strategies import composite
from fastapi.testclient import TestClient

from app.main import app
from app.services.data import DataService as StockDataService
from app.services.prediction import TechnicalIndicatorCalculator
from app.models.stock import StockData, DataServiceStatus


@composite
def stock_codes(draw):
    """生成股票代码"""
    market = draw(st.sampled_from(['SZ', 'SH']))
    code = draw(st.integers(min_value=1, max_value=999999))
    return f"{code:06d}.{market}"


@composite
def date_ranges(draw):
    """生成日期范围"""
    end_date = draw(st.datetimes(
        min_value=datetime(2020, 1, 1),
        max_value=datetime.now()
    ))
    start_date = draw(st.datetimes(
        min_value=datetime(2020, 1, 1),
        max_value=end_date
    ))
    return start_date, end_date


@composite
def stock_data_list(draw):
    """生成股票数据列表"""
    stock_code = draw(stock_codes())
    size = draw(st.integers(min_value=1, max_value=100))
    
    data_list = []
    base_date = datetime(2023, 1, 1)
    
    for i in range(size):
        data = StockData(
            stock_code=stock_code,
            date=base_date + timedelta(days=i),
            open=draw(st.floats(min_value=1.0, max_value=1000.0)),
            high=draw(st.floats(min_value=1.0, max_value=1000.0)),
            low=draw(st.floats(min_value=1.0, max_value=1000.0)),
            close=draw(st.floats(min_value=1.0, max_value=1000.0)),
            volume=draw(st.integers(min_value=100, max_value=1000000)),
            adj_close=draw(st.floats(min_value=1.0, max_value=1000.0))
        )
        data_list.append(data)
    
    return data_list


class TestAPIRoutesProperties:
    """API路由属性测试类"""
    
    def setup_method(self):
        """测试前设置"""
        # 创建测试应用，禁用限流中间件
        from app.main import create_application
        test_app = create_application()
        
        # 移除限流中间件
        test_app.user_middleware = [
            middleware for middleware in test_app.user_middleware
            if 'RateLimitMiddleware' not in str(middleware.cls)
        ]
        
        self.client = TestClient(test_app)
    
    @pytest.mark.asyncio
    @given(stock_codes(), date_ranges())
    @settings(max_examples=20, deadline=10000)
    async def test_stock_data_api_calls_real_service(self, stock_code, date_range):
        """
        属性 1: API路由真实服务调用
        股票数据API应该调用真实的数据服务而不是返回模拟数据
        **功能: data-management-implementation, 属性 1: API路由真实服务调用**
        **验证: 需求 1.1, 1.2, 1.3**
        """
        start_date, end_date = date_range
        
        # 模拟数据服务返回
        mock_stock_data = [
            StockData(
                stock_code=stock_code,
                date=start_date,
                open=10.0,
                high=11.0,
                low=9.0,
                close=10.5,
                volume=1000000,
                adj_close=10.5
            )
        ]
        
        with patch('app.core.container.get_data_service') as mock_get_service:
            mock_service = AsyncMock(spec=StockDataService)
            mock_service.get_stock_data.return_value = mock_stock_data
            mock_get_service.return_value = mock_service
            
            # 调用API
            response = self.client.get(
                "/api/v1/stocks/data",
                params={
                    "stock_code": stock_code,
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat()
                }
            )
            
            # 验证API调用了真实服务
            assert response.status_code == 200
            mock_service.get_stock_data.assert_called_once_with(
                stock_code, start_date, end_date
            )
            
            # 验证返回的数据来自真实服务
            data = response.json()
            assert data["success"] is True
            assert data["data"]["stock_code"] == stock_code
            assert data["data"]["data_points"] == 1
            assert len(data["data"]["data"]) == 1
    
    @pytest.mark.asyncio
    @given(stock_codes(), date_ranges())
    @settings(max_examples=15, deadline=10000)
    async def test_technical_indicators_api_calls_real_service(self, stock_code, date_range):
        """
        属性: 技术指标API调用真实服务
        技术指标API应该调用真实的计算服务
        **功能: data-management-implementation, 属性 1: API路由真实服务调用**
        """
        start_date, end_date = date_range
        
        # 模拟股票数据
        mock_stock_data = [
            StockData(
                stock_code=stock_code,
                date=start_date + timedelta(days=i),
                open=10.0 + i * 0.1,
                high=11.0 + i * 0.1,
                low=9.0 + i * 0.1,
                close=10.5 + i * 0.1,
                volume=1000000,
                adj_close=10.5 + i * 0.1
            )
            for i in range(20)  # 足够的数据点计算指标
        ]
        
        # 模拟技术指标结果
        from app.services.technical_indicators import TechnicalIndicatorResult
        mock_indicator_results = [
            TechnicalIndicatorResult(
                stock_code=stock_code,
                date=start_date,
                indicators={"MA5": 10.5, "RSI": 65.0}
            )
        ]
        
        with patch('app.core.container.get_data_service') as mock_get_data_service, \
             patch('app.core.container.get_indicators_service') as mock_get_indicators_service:
            
            mock_data_service = AsyncMock(spec=StockDataService)
            mock_data_service.get_stock_data.return_value = mock_stock_data
            mock_get_data_service.return_value = mock_data_service
            
            mock_indicators_service = MagicMock(spec=TechnicalIndicatorCalculator)
            mock_indicators_service.calculate_indicators.return_value = mock_indicator_results
            mock_get_indicators_service.return_value = mock_indicators_service
            
            # 调用API
            response = self.client.get(
                f"/api/v1/stocks/{stock_code}/indicators",
                params={
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "indicators": "MA5,RSI"
                }
            )
            
            # 验证API调用了真实服务
            assert response.status_code == 200
            mock_data_service.get_stock_data.assert_called_once()
            mock_indicators_service.calculate_indicators.assert_called_once()
            
            # 验证返回的数据来自真实计算
            data = response.json()
            assert data["success"] is True
            assert data["data"]["stock_code"] == stock_code
            assert "MA5" in data["data"]["indicators"]
            assert "RSI" in data["data"]["indicators"]
    
    @pytest.mark.asyncio
    @given(st.booleans())
    @settings(max_examples=10, deadline=8000)
    async def test_data_status_api_calls_real_service(self, is_available):
        """
        属性: 数据状态API调用真实服务
        数据状态API应该返回真实的服务状态
        **功能: data-management-implementation, 属性 1: API路由真实服务调用**
        """
        # 模拟服务状态
        mock_status = DataServiceStatus(
            service_url="http://192.168.3.62:8000",
            is_available=is_available,
            last_check=datetime.now(),
            response_time_ms=100.0 if is_available else None,
            error_message=None if is_available else "连接超时"
        )
        
        with patch('app.core.container.get_data_service') as mock_get_service:
            mock_service = AsyncMock(spec=StockDataService)
            mock_service.check_remote_service_status.return_value = mock_status
            mock_service.timeout = 30  # 模拟超时设置
            mock_get_service.return_value = mock_service
            
            # 调用API
            response = self.client.get("/api/v1/data/status")
            
            # 验证API调用了真实服务
            assert response.status_code == 200
            mock_service.check_remote_service_status.assert_called_once()
            
            # 验证返回的数据来自真实状态检查
            data = response.json()
            assert data["success"] == is_available
            assert data["data"]["is_connected"] == is_available
            assert data["data"]["service_url"] == "http://192.168.3.62:8000"
            
            if is_available:
                assert data["data"]["response_time"] == 100.0
                assert data["data"]["error_message"] is None
            else:
                assert data["data"]["error_message"] == "连接超时"
    
    @pytest.mark.asyncio
    @given(stock_data_list())
    @settings(max_examples=10, deadline=10000)
    async def test_api_error_handling_consistency(self, stock_data):
        """
        属性: API错误处理一致性
        当服务抛出异常时，API应该返回一致的错误格式
        **功能: data-management-implementation, 属性 1: API路由真实服务调用**
        """
        stock_code = stock_data[0].stock_code if stock_data else "000001.SZ"
        
        with patch('app.core.container.get_data_service') as mock_get_service:
            mock_service = AsyncMock(spec=StockDataService)
            mock_service.get_stock_data.side_effect = Exception("数据服务异常")
            mock_get_service.return_value = mock_service
            
            # 调用API
            response = self.client.get(
                "/api/v1/stocks/data",
                params={
                    "stock_code": stock_code,
                    "start_date": "2023-01-01T00:00:00",
                    "end_date": "2023-01-31T00:00:00"
                }
            )
            
            # 验证错误处理
            assert response.status_code == 500
            data = response.json()
            assert "数据服务异常" in data["detail"]
    
    @pytest.mark.asyncio
    @given(stock_codes())
    @settings(max_examples=10, deadline=8000)
    async def test_api_handles_empty_data_gracefully(self, stock_code):
        """
        属性: API优雅处理空数据
        当服务返回空数据时，API应该返回适当的响应
        **功能: data-management-implementation, 属性 1: API路由真实服务调用**
        """
        with patch('app.core.container.get_data_service') as mock_get_service:
            mock_service = AsyncMock(spec=StockDataService)
            mock_service.get_stock_data.return_value = None  # 返回空数据
            mock_get_service.return_value = mock_service
            
            # 调用API
            response = self.client.get(
                "/api/v1/stocks/data",
                params={
                    "stock_code": stock_code,
                    "start_date": "2023-01-01T00:00:00",
                    "end_date": "2023-01-31T00:00:00"
                }
            )
            
            # 验证空数据处理
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is False
            assert "未找到" in data["message"]
            assert data["data"] is None


@pytest.fixture(autouse=True)
def cleanup_after_test():
    """测试后自动清理"""
    yield
    # 清理代码可以在这里添加
    # 清理可能的模拟对象
    pass