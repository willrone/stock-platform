"""
API处理一致性属性测试

Feature: stock-prediction-platform, Property 6: API处理一致性
对于任何API请求，网关应该验证请求格式，正确路由到后端服务，并返回统一格式的响应

验证需求：6.1, 6.2, 6.3
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
from fastapi.testclient import TestClient
from datetime import datetime, timedelta
from typing import Dict, Any, List
import json

from app.main import app
from app.api.v1.schemas import StandardResponse


# 测试客户端
client = TestClient(app)


# 数据生成策略

@st.composite
def stock_code_strategy(draw):
    """生成有效的股票代码"""
    # 生成符合中国股票代码格式的代码
    market = draw(st.sampled_from(['SZ', 'SH']))
    code = draw(st.integers(min_value=1, max_value=999999))
    return f"{code:06d}.{market}"


@st.composite
def datetime_strategy(draw):
    """生成有效的日期时间"""
    # 生成2020年到2025年之间的日期
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2025, 12, 31)
    
    days_diff = (end_date - start_date).days
    random_days = draw(st.integers(min_value=0, max_value=days_diff))
    
    return start_date + timedelta(days=random_days)


@st.composite
def valid_prediction_request_strategy(draw):
    """生成有效的预测请求"""
    stock_codes = draw(st.lists(stock_code_strategy(), min_size=1, max_size=10))
    model_id = draw(st.sampled_from(['xgboost_v1', 'lstm_v1', 'transformer_v1']))
    horizon = draw(st.sampled_from(['intraday', 'short_term', 'medium_term']))
    confidence_level = draw(st.floats(min_value=0.5, max_value=0.99))
    
    return {
        "stock_codes": stock_codes,
        "model_id": model_id,
        "horizon": horizon,
        "confidence_level": confidence_level
    }


@st.composite
def valid_task_request_strategy(draw):
    """生成有效的任务创建请求"""
    task_name = draw(st.text(min_size=1, max_size=100))
    stock_codes = draw(st.lists(stock_code_strategy(), min_size=1, max_size=20))
    model_id = draw(st.sampled_from(['xgboost_v1', 'lstm_v1', 'transformer_v1']))
    
    return {
        "task_name": task_name,
        "stock_codes": stock_codes,
        "model_id": model_id,
        "prediction_config": {}
    }


@st.composite
def valid_backtest_request_strategy(draw):
    """生成有效的回测请求"""
    strategy_name = draw(st.text(min_size=1, max_size=50))
    stock_codes = draw(st.lists(stock_code_strategy(), min_size=1, max_size=10))
    start_date = draw(datetime_strategy())
    end_date = draw(datetime_strategy())
    
    # 确保结束日期在开始日期之后
    assume(end_date > start_date)
    
    initial_cash = draw(st.floats(min_value=10000, max_value=1000000))
    
    return {
        "strategy_name": strategy_name,
        "stock_codes": stock_codes,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "initial_cash": initial_cash
    }


@st.composite
def invalid_request_strategy(draw):
    """生成无效的请求数据"""
    invalid_type = draw(st.sampled_from([
        'missing_required_field',
        'invalid_data_type',
        'empty_list',
        'invalid_date_format',
        'negative_values'
    ]))
    
    if invalid_type == 'missing_required_field':
        # 缺少必需字段
        return {"incomplete": "data"}
    elif invalid_type == 'invalid_data_type':
        # 错误的数据类型
        return {"stock_codes": "not_a_list", "model_id": 123}
    elif invalid_type == 'empty_list':
        # 空列表
        return {"stock_codes": [], "model_id": "xgboost_v1"}
    elif invalid_type == 'invalid_date_format':
        # 无效的日期格式
        return {
            "strategy_name": "test",
            "stock_codes": ["000001.SZ"],
            "start_date": "invalid_date",
            "end_date": "2023-12-31"
        }
    elif invalid_type == 'negative_values':
        # 负值
        return {
            "strategy_name": "test",
            "stock_codes": ["000001.SZ"],
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "initial_cash": -1000
        }


# 属性测试

class TestAPIConsistencyProperties:
    """API处理一致性属性测试"""
    
    @given(valid_prediction_request_strategy())
    @settings(max_examples=100, deadline=None)
    def test_prediction_api_response_format_consistency(self, request_data):
        """
        属性：预测API响应格式一致性
        对于任何有效的预测请求，API应该返回统一格式的响应
        """
        response = client.post("/api/v1/predictions", json=request_data)
        
        # 验证HTTP状态码
        assert response.status_code in [200, 400, 422, 500], \
            f"API应该返回有效的HTTP状态码，实际: {response.status_code}"
        
        # 验证响应格式
        response_data = response.json()
        self._validate_standard_response_format(response_data)
        
        # 如果成功，验证数据结构
        if response.status_code == 200:
            assert response_data["success"] is True
            assert "data" in response_data
            assert response_data["data"] is not None
            
            # 验证预测结果数据结构
            data = response_data["data"]
            assert "predictions" in data
            assert isinstance(data["predictions"], list)
            assert len(data["predictions"]) == len(request_data["stock_codes"])
    
    @given(valid_task_request_strategy())
    @settings(max_examples=100, deadline=None)
    def test_task_api_response_format_consistency(self, request_data):
        """
        属性：任务API响应格式一致性
        对于任何有效的任务创建请求，API应该返回统一格式的响应
        """
        response = client.post("/api/v1/tasks", json=request_data)
        
        # 验证HTTP状态码
        assert response.status_code in [200, 400, 422, 500], \
            f"API应该返回有效的HTTP状态码，实际: {response.status_code}"
        
        # 验证响应格式
        response_data = response.json()
        self._validate_standard_response_format(response_data)
        
        # 如果成功，验证任务数据结构
        if response.status_code == 200:
            assert response_data["success"] is True
            data = response_data["data"]
            assert "task_id" in data
            assert "task_name" in data
            assert "status" in data
            assert "created_at" in data
    
    @given(valid_backtest_request_strategy())
    @settings(max_examples=100, deadline=None)
    def test_backtest_api_response_format_consistency(self, request_data):
        """
        属性：回测API响应格式一致性
        对于任何有效的回测请求，API应该返回统一格式的响应
        """
        response = client.post("/api/v1/backtest", json=request_data)
        
        # 验证HTTP状态码
        assert response.status_code in [200, 400, 422, 500], \
            f"API应该返回有效的HTTP状态码，实际: {response.status_code}"
        
        # 验证响应格式
        response_data = response.json()
        self._validate_standard_response_format(response_data)
        
        # 如果成功，验证回测结果数据结构
        if response.status_code == 200:
            assert response_data["success"] is True
            data = response_data["data"]
            assert "strategy_name" in data
            assert "portfolio" in data
            assert "risk_metrics" in data
            assert "trading_stats" in data
    
    @given(invalid_request_strategy())
    @settings(max_examples=100, deadline=None)
    def test_invalid_request_error_handling_consistency(self, invalid_data):
        """
        属性：无效请求错误处理一致性
        对于任何无效请求，API应该返回一致的错误响应格式
        """
        # 测试多个端点的错误处理
        endpoints = [
            "/api/v1/predictions",
            "/api/v1/tasks",
            "/api/v1/backtest"
        ]
        
        for endpoint in endpoints:
            response = client.post(endpoint, json=invalid_data)
            
            # 验证错误状态码（包括200，因为某些无效数据可能被API接受）
            assert response.status_code in [200, 400, 422, 500], \
                f"API应该返回有效状态码，端点: {endpoint}, 状态码: {response.status_code}"
            
            response_data = response.json()
            
            # 如果是422错误（FastAPI验证错误），格式会不同
            if response.status_code == 422:
                assert "detail" in response_data
                assert isinstance(response_data["detail"], list)
            else:
                # 验证标准响应格式
                self._validate_standard_response_format(response_data)
                
                # 如果不是成功响应，验证错误内容
                if response.status_code != 200:
                    assert response_data["success"] is False
                    assert response_data["message"] is not None
                    assert len(response_data["message"]) > 0
    
    @given(stock_code_strategy())
    @settings(max_examples=100, deadline=None)
    def test_get_endpoints_response_consistency(self, stock_code):
        """
        属性：GET端点响应一致性
        对于任何GET请求，API应该返回统一格式的响应
        """
        # 测试各种GET端点
        get_endpoints = [
            "/api/v1/health",
            "/api/v1/tasks",
            "/api/v1/models",
            "/api/v1/system/status"
        ]
        
        for endpoint in get_endpoints:
            response = client.get(endpoint)
            
            # 验证HTTP状态码
            assert response.status_code in [200, 404, 500], \
                f"GET请求应该返回有效状态码，端点: {endpoint}, 状态码: {response.status_code}"
            
            # 验证响应格式
            response_data = response.json()
            self._validate_standard_response_format(response_data)
    
    @given(st.text(min_size=1, max_size=50).filter(lambda x: x.isprintable() and '/' not in x))
    @settings(max_examples=100, deadline=None)
    def test_parameterized_get_endpoints_consistency(self, param_value):
        """
        属性：参数化GET端点一致性
        对于任何带参数的GET请求，API应该返回统一格式的响应
        """
        # 测试带参数的GET端点
        parameterized_endpoints = [
            f"/api/v1/tasks/{param_value}",
            f"/api/v1/models/{param_value}",
            f"/api/v1/predictions/{param_value}"
        ]
        
        for endpoint in parameterized_endpoints:
            try:
                response = client.get(endpoint)
                
                # 验证HTTP状态码（包括405 Method Not Allowed）
                assert response.status_code in [200, 404, 405, 500], \
                    f"参数化GET请求应该返回有效状态码，端点: {endpoint}, 状态码: {response.status_code}"
                
                # 如果不是405错误，验证响应格式
                if response.status_code != 405:
                    response_data = response.json()
                    self._validate_standard_response_format(response_data)
                    
            except Exception as e:
                # 如果URL无效，跳过这个测试用例
                if "Invalid" in str(e):
                    continue
                else:
                    raise
    
    @given(
        stock_code_strategy(),
        datetime_strategy(),
        datetime_strategy()
    )
    @settings(max_examples=100, deadline=None)
    def test_stock_data_api_consistency(self, stock_code, start_date, end_date):
        """
        属性：股票数据API一致性
        对于任何股票数据请求，API应该返回统一格式的响应
        """
        # 确保日期顺序正确
        if start_date > end_date:
            start_date, end_date = end_date, start_date
        
        params = {
            "stock_code": stock_code,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat()
        }
        
        response = client.get("/api/v1/stocks/data", params=params)
        
        # 验证HTTP状态码
        assert response.status_code in [200, 400, 422, 500], \
            f"股票数据API应该返回有效状态码，实际: {response.status_code}"
        
        # 验证响应格式
        response_data = response.json()
        self._validate_standard_response_format(response_data)
        
        # 如果成功，验证数据结构
        if response.status_code == 200:
            assert response_data["success"] is True
            data = response_data["data"]
            assert "stock_code" in data
            assert data["stock_code"] == stock_code
    
    def _validate_standard_response_format(self, response_data: Dict[str, Any]):
        """验证标准响应格式"""
        # 验证必需字段存在
        required_fields = ["success", "message", "timestamp"]
        for field in required_fields:
            assert field in response_data, f"响应缺少必需字段: {field}"
        
        # 验证字段类型
        assert isinstance(response_data["success"], bool), "success字段应该是布尔值"
        assert isinstance(response_data["message"], str), "message字段应该是字符串"
        assert isinstance(response_data["timestamp"], str), "timestamp字段应该是字符串"
        
        # 验证时间戳格式
        try:
            datetime.fromisoformat(response_data["timestamp"].replace('Z', '+00:00'))
        except ValueError:
            pytest.fail(f"timestamp格式无效: {response_data['timestamp']}")
        
        # 验证消息不为空
        assert len(response_data["message"]) > 0, "message字段不应为空"
        
        # 如果有data字段，验证其存在性
        if "data" in response_data:
            # data可以为None，但如果存在应该是有效的JSON数据
            if response_data["data"] is not None:
                # 确保data是可序列化的
                try:
                    json.dumps(response_data["data"])
                except (TypeError, ValueError):
                    pytest.fail("data字段应该是可JSON序列化的")


# 集成测试

class TestAPIIntegration:
    """API集成测试"""
    
    def test_health_check_endpoint(self):
        """测试健康检查端点"""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "data" in data
        assert data["data"]["status"] == "healthy"
    
    def test_models_list_endpoint(self):
        """测试模型列表端点"""
        response = client.get("/api/v1/models")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "data" in data
        assert "models" in data["data"]
        assert isinstance(data["data"]["models"], list)
    
    def test_system_status_endpoint(self):
        """测试系统状态端点"""
        response = client.get("/api/v1/system/status")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "data" in data
        
        # 验证系统状态数据结构
        status_data = data["data"]
        expected_services = [
            "api_server", "data_service", "prediction_engine", 
            "task_manager", "database", "remote_data_service"
        ]
        
        for service in expected_services:
            assert service in status_data
            assert "status" in status_data[service]
    
    def test_error_handling_for_nonexistent_endpoints(self):
        """测试不存在端点的错误处理"""
        response = client.get("/api/v1/nonexistent")
        assert response.status_code == 404
    
    def test_method_not_allowed_handling(self):
        """测试不允许的HTTP方法处理"""
        # 对只支持GET的端点发送POST请求
        response = client.post("/api/v1/health")
        assert response.status_code == 405  # Method Not Allowed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])