"""
API路由属性测试
验证API路由调用真实服务的正确性属性
"""

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock
from hypothesis import given, strategies as st, settings
from hypothesis.strategies import composite
from fastapi.testclient import TestClient

from app.main import app
from app.core.container import get_data_service, get_indicators_service
from app.services.data import SimpleDataService as StockDataService
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
        max_value=datetime(2025, 12, 31)
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
    size = draw(st.integers(min_value=1, max_value=10))

    data_list = []
    base_date = datetime(2023, 1, 1)

    for i in range(size):
        base_price = draw(st.floats(min_value=5.0, max_value=500.0, allow_nan=False, allow_infinity=False))
        data = StockData(
            stock_code=stock_code,
            date=base_date + timedelta(days=i),
            open=base_price,
            high=base_price * 1.05,
            low=base_price * 0.95,
            close=base_price * 1.01,
            volume=draw(st.integers(min_value=100, max_value=1000000)),
            adj_close=base_price * 1.01
        )
        data_list.append(data)

    return data_list


class TestAPIRoutesProperties:
    """API路由属性测试类"""

    def setup_method(self):
        """测试前设置"""
        self.client = TestClient(app)
        # 清理之前的 dependency overrides
        app.dependency_overrides.clear()

    def teardown_method(self):
        """测试后清理"""
        app.dependency_overrides.clear()

    @given(stock_codes(), date_ranges())
    @settings(max_examples=5, deadline=10000)
    def test_stock_data_api_calls_real_service(self, stock_code, date_range):
        """
        属性 1: API路由真实服务调用
        股票数据API应该调用真实的数据服务而不是返回模拟数据
        """
        start_date, end_date = date_range

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

        mock_service = AsyncMock(spec=StockDataService)
        mock_service.get_stock_data.return_value = mock_stock_data

        async def override_data_service():
            return mock_service

        app.dependency_overrides[get_data_service] = override_data_service

        response = self.client.get(
            "/api/v1/stocks/data",
            params={
                "stock_code": stock_code,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            }
        )

        assert response.status_code == 200
        mock_service.get_stock_data.assert_called_once()

        data = response.json()
        assert data["success"] is True
        assert data["data"]["stock_code"] == stock_code
        assert data["data"]["data_points"] == 1
        assert len(data["data"]["data"]) == 1

    @given(stock_codes(), date_ranges())
    @settings(max_examples=5, deadline=10000)
    def test_technical_indicators_api_calls_real_service(self, stock_code, date_range):
        """
        属性: 技术指标API调用真实服务
        """
        start_date, end_date = date_range

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
            for i in range(20)
        ]

        from app.services.prediction.technical_indicators import TechnicalIndicatorResult
        mock_indicator_result = MagicMock(spec=TechnicalIndicatorResult)
        mock_indicator_result.indicators = {"MA5": 10.5, "RSI": 65.0}
        mock_indicator_result.to_dict.return_value = {
            "stock_code": stock_code,
            "date": start_date.isoformat(),
            "indicators": {"MA5": 10.5, "RSI": 65.0}
        }
        mock_indicator_results = [mock_indicator_result]

        mock_data_service = AsyncMock(spec=StockDataService)
        mock_data_service.get_stock_data.return_value = mock_stock_data

        mock_indicators_service = MagicMock(spec=TechnicalIndicatorCalculator)
        mock_indicators_service.calculate_indicators.return_value = mock_indicator_results

        async def override_data_service():
            return mock_data_service

        async def override_indicators_service():
            return mock_indicators_service

        app.dependency_overrides[get_data_service] = override_data_service
        app.dependency_overrides[get_indicators_service] = override_indicators_service

        response = self.client.get(
            f"/api/v1/stocks/{stock_code}/indicators",
            params={
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "indicators": "MA5,RSI"
            }
        )

        assert response.status_code == 200
        mock_data_service.get_stock_data.assert_called_once()
        mock_indicators_service.calculate_indicators.assert_called_once()

        data = response.json()
        assert data["success"] is True
        assert data["data"]["stock_code"] == stock_code
        assert "MA5" in data["data"]["indicators"]
        assert "RSI" in data["data"]["indicators"]

    @given(st.booleans())
    @settings(max_examples=5, deadline=8000)
    def test_data_status_api_calls_real_service(self, is_available):
        """
        属性: 数据状态API调用真实服务
        """
        mock_status = DataServiceStatus(
            service_url="http://192.168.3.62:8000",
            is_available=is_available,
            last_check=datetime.now(timezone.utc),
            response_time_ms=100.0 if is_available else None,
            error_message=None if is_available else "连接超时"
        )

        mock_service = AsyncMock(spec=StockDataService)
        mock_service.check_remote_service_status.return_value = mock_status
        mock_service.timeout = 30

        async def override_data_service():
            return mock_service

        app.dependency_overrides[get_data_service] = override_data_service

        response = self.client.get("/api/v1/data/status")

        assert response.status_code == 200
        mock_service.check_remote_service_status.assert_called_once()

        data = response.json()
        assert data["success"] == is_available
        assert data["data"]["is_connected"] == is_available
        assert data["data"]["service_url"] == "http://192.168.3.62:8000"

        if is_available:
            assert data["data"]["response_time"] == 100.0
            assert data["data"]["error_message"] is None
        else:
            assert data["data"]["error_message"] == "连接超时"

    @given(stock_data_list())
    @settings(max_examples=5, deadline=10000)
    def test_api_error_handling_consistency(self, stock_data):
        """
        属性: API错误处理一致性
        """
        stock_code = stock_data[0].stock_code if stock_data else "000001.SZ"

        mock_service = AsyncMock(spec=StockDataService)
        mock_service.get_stock_data.side_effect = Exception("数据服务异常")

        async def override_data_service():
            return mock_service

        app.dependency_overrides[get_data_service] = override_data_service

        response = self.client.get(
            "/api/v1/stocks/data",
            params={
                "stock_code": stock_code,
                "start_date": "2023-01-01T00:00:00",
                "end_date": "2023-01-31T00:00:00"
            }
        )

        assert response.status_code == 500
        data = response.json()
        assert "数据服务异常" in data["message"]

    @given(stock_codes())
    @settings(max_examples=5, deadline=8000)
    def test_api_handles_empty_data_gracefully(self, stock_code):
        """
        属性: API优雅处理空数据
        """
        mock_service = AsyncMock(spec=StockDataService)
        mock_service.get_stock_data.return_value = None

        async def override_data_service():
            return mock_service

        app.dependency_overrides[get_data_service] = override_data_service

        response = self.client.get(
            "/api/v1/stocks/data",
            params={
                "stock_code": stock_code,
                "start_date": "2023-01-01T00:00:00",
                "end_date": "2023-01-31T00:00:00"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert "未找到" in data["message"]


@pytest.fixture(autouse=True)
def cleanup_after_test():
    """测试后自动清理"""
    yield
    app.dependency_overrides.clear()
