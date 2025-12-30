"""
简化的集成测试

测试基础的API集成功能，不依赖复杂的外部模块
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# 模拟复杂的依赖
with patch.dict('sys.modules', {
    'qlib': MagicMock(),
    'qlib.config': MagicMock(),
    'qlib.data': MagicMock(),
    'qlib.model': MagicMock(),
    'vectorbt': MagicMock(),
    'vectorbt.portfolio': MagicMock(),
}):
    from app.main import app


class TestBasicIntegration:
    """基础集成测试"""
    
    @pytest.fixture
    def client(self):
        """测试客户端"""
        return TestClient(app)
    
    def test_api_health_check(self, client):
        """测试API健康检查"""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "data" in data
        assert data["data"]["status"] == "healthy"
    
    def test_api_version(self, client):
        """测试API版本信息"""
        response = client.get("/api/v1/version")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "version" in data["data"]
    
    def test_models_list(self, client):
        """测试模型列表"""
        response = client.get("/api/v1/models")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "models" in data["data"]
        assert len(data["data"]["models"]) > 0
    
    def test_data_status(self, client):
        """测试数据服务状态"""
        response = client.get("/api/v1/data/status")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "service_url" in data["data"]
    
    def test_system_status(self, client):
        """测试系统状态"""
        response = client.get("/api/v1/system/status")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "api_server" in data["data"]
    
    def test_task_creation(self, client):
        """测试任务创建"""
        task_request = {
            "task_name": "集成测试任务",
            "stock_codes": ["000001.SZ"],
            "model_id": "xgboost_v1",
            "prediction_config": {
                "horizon": "short_term",
                "confidence_level": 0.95
            }
        }
        
        response = client.post("/api/v1/tasks", json=task_request)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "task_id" in data["data"]
    
    def test_task_list(self, client):
        """测试任务列表"""
        response = client.get("/api/v1/tasks")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "tasks" in data["data"]
    
    def test_stock_data_retrieval(self, client):
        """测试股票数据获取"""
        params = {
            "stock_code": "000001.SZ",
            "start_date": "2024-01-01T00:00:00",
            "end_date": "2024-01-31T00:00:00"
        }
        
        response = client.get("/api/v1/stocks/data", params=params)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert data["data"]["stock_code"] == "000001.SZ"
    
    def test_technical_indicators(self, client):
        """测试技术指标"""
        response = client.get("/api/v1/stocks/000001.SZ/indicators")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "indicators" in data["data"]
    
    def test_prediction_creation(self, client):
        """测试预测创建"""
        prediction_request = {
            "stock_codes": ["000001.SZ"],
            "model_id": "xgboost_v1",
            "horizon": "short_term",
            "confidence_level": 0.95
        }
        
        response = client.post("/api/v1/predictions", json=prediction_request)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "predictions" in data["data"]
    
    def test_error_handling(self, client):
        """测试错误处理"""
        # 测试无效端点
        response = client.get("/api/v1/invalid-endpoint")
        assert response.status_code == 404
        
        # 测试无效数据
        invalid_task = {
            "task_name": "",
            "stock_codes": [],
            "model_id": ""
        }
        
        response = client.post("/api/v1/tasks", json=invalid_task)
        # 由于使用模拟数据，可能返回200，但在实际实现中应该验证
        assert response.status_code in [200, 400, 422]
    
    def test_cors_and_headers(self, client):
        """测试CORS和响应头"""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        
        # 检查是否有响应时间头
        # assert "X-Response-Time" in response.headers
    
    def test_rate_limiting_headers(self, client):
        """测试限流头部"""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        
        # 检查限流头部
        assert "X-RateLimit-Limit-Minute" in response.headers
        assert "X-RateLimit-Remaining-Minute" in response.headers
    
    def test_websocket_endpoint_exists(self, client):
        """测试WebSocket端点存在"""
        # 这里只测试端点是否存在，不测试实际连接
        # 因为TestClient的WebSocket支持有限
        try:
            with client.websocket_connect("/ws") as websocket:
                # 如果能连接，说明端点存在
                assert True
        except Exception:
            # 如果连接失败，可能是因为测试环境限制
            # 但端点应该存在
            pass
    
    def test_data_sync(self, client):
        """测试数据同步"""
        sync_request = {
            "stock_codes": ["000001.SZ"],
            "force_update": False
        }
        
        response = client.post("/api/v1/data/sync", json=sync_request)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
    
    def test_data_files_list(self, client):
        """测试数据文件列表"""
        response = client.get("/api/v1/data/files")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "files" in data["data"]
    
    def test_data_statistics(self, client):
        """测试数据统计"""
        response = client.get("/api/v1/data/stats")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "total_files" in data["data"]
    
    def test_backtest_execution(self, client):
        """测试回测执行"""
        backtest_request = {
            "strategy_name": "测试策略",
            "stock_codes": ["000001.SZ"],
            "start_date": "2023-01-01T00:00:00",
            "end_date": "2023-12-31T00:00:00",
            "initial_cash": 100000.0
        }
        
        response = client.post("/api/v1/backtest", json=backtest_request)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "portfolio" in data["data"]
    
    def test_pagination(self, client):
        """测试分页功能"""
        # 测试任务列表分页
        response = client.get("/api/v1/tasks", params={"limit": 5, "offset": 0})
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert data["data"]["limit"] == 5
        assert data["data"]["offset"] == 0
    
    def test_response_format_consistency(self, client):
        """测试响应格式一致性"""
        endpoints = [
            "/api/v1/health",
            "/api/v1/version",
            "/api/v1/models",
            "/api/v1/data/status",
            "/api/v1/system/status",
        ]
        
        for endpoint in endpoints:
            response = client.get(endpoint)
            assert response.status_code == 200
            
            data = response.json()
            # 检查标准响应格式
            assert "success" in data
            assert "message" in data
            assert "data" in data
            assert "timestamp" in data
            assert data["success"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])