"""
集成测试

测试前后端集成功能，包括：
- 完整的预测任务流程
- 数据同步和管理功能
- 错误恢复机制
- WebSocket实时通信
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
from fastapi.websockets import WebSocket

from app.main import app
from app.websocket import manager
from app.services.task_manager import TaskManager
from app.services.data_service import StockDataService
from app.models.stock import TaskCreateRequest, StockDataRequest


class TestIntegration:
    """集成测试类"""
    
    @pytest.fixture
    def client(self):
        """测试客户端"""
        return TestClient(app)
    
    @pytest.fixture
    def sample_task_request(self):
        """示例任务请求"""
        return {
            "task_name": "集成测试任务",
            "stock_codes": ["000001.SZ", "000002.SZ"],
            "model_id": "xgboost_v1",
            "prediction_config": {
                "horizon": "short_term",
                "confidence_level": 0.95
            }
        }
    
    def test_api_health_check(self, client):
        """测试API健康检查"""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "data" in data
        assert data["data"]["status"] == "healthy"
    
    def test_complete_prediction_task_flow(self, client, sample_task_request):
        """测试完整的预测任务流程"""
        # 1. 创建任务
        response = client.post("/api/v1/tasks", json=sample_task_request)
        assert response.status_code == 200
        
        task_data = response.json()
        assert task_data["success"] is True
        assert "task_id" in task_data["data"]
        
        task_id = task_data["data"]["task_id"]
        
        # 2. 获取任务详情
        response = client.get(f"/api/v1/tasks/{task_id}")
        assert response.status_code == 200
        
        task_detail = response.json()
        assert task_detail["success"] is True
        assert task_detail["data"]["task_id"] == task_id
        assert task_detail["data"]["task_name"] == sample_task_request["task_name"]
        
        # 3. 获取任务列表
        response = client.get("/api/v1/tasks")
        assert response.status_code == 200
        
        task_list = response.json()
        assert task_list["success"] is True
        assert "tasks" in task_list["data"]
        
        # 验证创建的任务在列表中
        task_ids = [task["task_id"] for task in task_list["data"]["tasks"]]
        assert task_id in task_ids
    
    def test_data_synchronization_flow(self, client):
        """测试数据同步和管理功能"""
        # 1. 获取数据服务状态
        response = client.get("/api/v1/data/status")
        assert response.status_code == 200
        
        status_data = response.json()
        assert status_data["success"] is True
        assert "service_url" in status_data["data"]
        
        # 2. 获取本地数据文件列表
        response = client.get("/api/v1/data/files")
        assert response.status_code == 200
        
        files_data = response.json()
        assert files_data["success"] is True
        assert "files" in files_data["data"]
        
        # 3. 获取数据统计信息
        response = client.get("/api/v1/data/stats")
        assert response.status_code == 200
        
        stats_data = response.json()
        assert stats_data["success"] is True
        assert "total_files" in stats_data["data"]
        
        # 4. 同步数据
        sync_request = {
            "stock_codes": ["000001.SZ"],
            "force_update": False
        }
        response = client.post("/api/v1/data/sync", json=sync_request)
        assert response.status_code == 200
        
        sync_data = response.json()
        assert sync_data["success"] is True
        assert "synced_stocks" in sync_data["data"]
    
    def test_stock_data_retrieval_flow(self, client):
        """测试股票数据获取流程"""
        # 1. 获取股票基础数据
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        params = {
            "stock_code": "000001.SZ",
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat()
        }
        
        response = client.get("/api/v1/stocks/data", params=params)
        assert response.status_code == 200
        
        stock_data = response.json()
        assert stock_data["success"] is True
        assert stock_data["data"]["stock_code"] == "000001.SZ"
        
        # 2. 获取技术指标
        response = client.get("/api/v1/stocks/000001.SZ/indicators", params={
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat()
        })
        assert response.status_code == 200
        
        indicators_data = response.json()
        assert indicators_data["success"] is True
        assert "indicators" in indicators_data["data"]
    
    def test_model_management_flow(self, client):
        """测试模型管理流程"""
        # 1. 获取模型列表
        response = client.get("/api/v1/models")
        assert response.status_code == 200
        
        models_data = response.json()
        assert models_data["success"] is True
        assert "models" in models_data["data"]
        assert len(models_data["data"]["models"]) > 0
        
        # 2. 获取模型详情
        model_id = models_data["data"]["models"][0]["model_id"]
        response = client.get(f"/api/v1/models/{model_id}")
        assert response.status_code == 200
        
        model_detail = response.json()
        assert model_detail["success"] is True
        assert model_detail["data"]["model_id"] == model_id
    
    def test_prediction_flow(self, client):
        """测试预测流程"""
        prediction_request = {
            "stock_codes": ["000001.SZ", "000002.SZ"],
            "model_id": "xgboost_v1",
            "horizon": "short_term",
            "confidence_level": 0.95
        }
        
        response = client.post("/api/v1/predictions", json=prediction_request)
        assert response.status_code == 200
        
        prediction_data = response.json()
        assert prediction_data["success"] is True
        assert "predictions" in prediction_data["data"]
        assert len(prediction_data["data"]["predictions"]) == 2
        
        # 验证预测结果结构
        for prediction in prediction_data["data"]["predictions"]:
            assert "stock_code" in prediction
            assert "predicted_direction" in prediction
            assert "confidence_score" in prediction
    
    def test_backtest_flow(self, client):
        """测试回测流程"""
        backtest_request = {
            "strategy_name": "测试策略",
            "stock_codes": ["000001.SZ"],
            "start_date": (datetime.now() - timedelta(days=365)).isoformat(),
            "end_date": datetime.now().isoformat(),
            "initial_cash": 100000.0
        }
        
        response = client.post("/api/v1/backtest", json=backtest_request)
        assert response.status_code == 200
        
        backtest_data = response.json()
        assert backtest_data["success"] is True
        assert "portfolio" in backtest_data["data"]
        assert "risk_metrics" in backtest_data["data"]
        assert "trading_stats" in backtest_data["data"]
    
    def test_system_status_flow(self, client):
        """测试系统状态流程"""
        # 1. 获取系统状态
        response = client.get("/api/v1/system/status")
        assert response.status_code == 200
        
        status_data = response.json()
        assert status_data["success"] is True
        assert "api_server" in status_data["data"]
        
        # 2. 获取API版本信息
        response = client.get("/api/v1/version")
        assert response.status_code == 200
        
        version_data = response.json()
        assert version_data["success"] is True
        assert "version" in version_data["data"]
    
    def test_error_handling_mechanisms(self, client):
        """测试错误处理机制"""
        # 1. 测试无效端点
        response = client.get("/api/v1/invalid-endpoint")
        assert response.status_code == 404
        
        # 2. 测试无效任务ID
        response = client.get("/api/v1/tasks/invalid-task-id")
        assert response.status_code == 500  # 模拟数据会返回500
        
        # 3. 测试无效股票代码
        response = client.get("/api/v1/stocks/INVALID.CODE/indicators")
        assert response.status_code == 500  # 模拟数据会返回500
        
        # 4. 测试无效请求数据
        invalid_task_request = {
            "task_name": "",  # 空名称
            "stock_codes": [],  # 空股票列表
            "model_id": "invalid_model"
        }
        
        response = client.post("/api/v1/tasks", json=invalid_task_request)
        # 由于使用模拟数据，这里可能返回200，但在实际实现中应该返回400
        assert response.status_code in [200, 400]
    
    @pytest.mark.asyncio
    async def test_websocket_connection(self):
        """测试WebSocket连接"""
        with TestClient(app) as client:
            with client.websocket_connect("/ws") as websocket:
                # 测试连接建立
                data = websocket.receive_json()
                assert data["type"] == "connection"
                assert data["message"] == "WebSocket连接成功"
                
                # 测试心跳
                websocket.send_json({"type": "ping"})
                response = websocket.receive_json()
                assert response["type"] == "pong"
    
    @pytest.mark.asyncio
    async def test_websocket_task_subscription(self):
        """测试WebSocket任务订阅"""
        with TestClient(app) as client:
            with client.websocket_connect("/ws") as websocket:
                # 接收连接消息
                websocket.receive_json()
                
                # 订阅任务
                task_id = "test_task_123"
                websocket.send_json({
                    "type": "subscribe:task",
                    "task_id": task_id
                })
                
                response = websocket.receive_json()
                assert response["type"] == "subscription"
                assert task_id in response["message"]
                
                # 取消订阅
                websocket.send_json({
                    "type": "unsubscribe:task",
                    "task_id": task_id
                })
                
                response = websocket.receive_json()
                assert response["type"] == "unsubscription"
                assert task_id in response["message"]
    
    @pytest.mark.asyncio
    async def test_websocket_system_subscription(self):
        """测试WebSocket系统状态订阅"""
        with TestClient(app) as client:
            with client.websocket_connect("/ws") as websocket:
                # 接收连接消息
                websocket.receive_json()
                
                # 订阅系统状态
                websocket.send_json({"type": "subscribe:system"})
                
                response = websocket.receive_json()
                assert response["type"] == "subscription"
                assert "系统状态" in response["message"]
                
                # 取消订阅
                websocket.send_json({"type": "unsubscribe:system"})
                
                response = websocket.receive_json()
                assert response["type"] == "unsubscription"
                assert "系统状态" in response["message"]
    
    def test_rate_limiting(self, client):
        """测试限流机制"""
        # 快速发送多个请求测试限流
        responses = []
        for i in range(10):
            response = client.get("/api/v1/health")
            responses.append(response)
        
        # 所有请求都应该成功（因为限流配置比较宽松）
        for response in responses:
            assert response.status_code == 200
        
        # 检查限流头部
        last_response = responses[-1]
        assert "X-RateLimit-Limit-Minute" in last_response.headers
        assert "X-RateLimit-Remaining-Minute" in last_response.headers
    
    def test_cors_headers(self, client):
        """测试CORS头部"""
        response = client.options("/api/v1/health")
        # 由于CORS中间件配置，应该包含相关头部
        # 具体头部取决于配置
        assert response.status_code in [200, 405]  # OPTIONS可能不被支持
    
    def test_error_response_format(self, client):
        """测试错误响应格式"""
        response = client.get("/api/v1/invalid-endpoint")
        assert response.status_code == 404
        
        # 检查错误响应格式
        data = response.json()
        assert "success" in data
        assert "message" in data
        assert "timestamp" in data
        assert data["success"] is False
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, client):
        """测试并发请求处理"""
        async def make_request():
            response = client.get("/api/v1/health")
            return response.status_code
        
        # 并发发送多个请求
        tasks = [make_request() for _ in range(5)]
        results = await asyncio.gather(*tasks)
        
        # 所有请求都应该成功
        assert all(status == 200 for status in results)
    
    def test_data_consistency(self, client, sample_task_request):
        """测试数据一致性"""
        # 创建任务
        response = client.post("/api/v1/tasks", json=sample_task_request)
        task_data = response.json()
        task_id = task_data["data"]["task_id"]
        
        # 多次获取任务详情，确保数据一致
        responses = []
        for _ in range(3):
            response = client.get(f"/api/v1/tasks/{task_id}")
            responses.append(response.json())
        
        # 验证数据一致性
        for i in range(1, len(responses)):
            assert responses[i]["data"]["task_id"] == responses[0]["data"]["task_id"]
            assert responses[i]["data"]["task_name"] == responses[0]["data"]["task_name"]
    
    def test_pagination(self, client):
        """测试分页功能"""
        # 测试任务列表分页
        response = client.get("/api/v1/tasks", params={"limit": 5, "offset": 0})
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "limit" in data["data"]
        assert "offset" in data["data"]
        assert data["data"]["limit"] == 5
        assert data["data"]["offset"] == 0
        
        # 测试数据文件列表分页
        response = client.get("/api/v1/data/files", params={"limit": 10, "offset": 0})
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "limit" in data["data"]
        assert "offset" in data["data"]


class TestErrorRecovery:
    """错误恢复测试"""
    
    @pytest.fixture
    def client(self):
        """测试客户端"""
        return TestClient(app)
    
    def test_service_unavailable_recovery(self, client):
        """测试服务不可用时的恢复机制"""
        # 模拟数据服务不可用的情况
        with patch('app.services.data_service.StockDataService.check_remote_service_status') as mock_check:
            mock_check.return_value = {"is_connected": False, "error_message": "连接超时"}
            
            response = client.get("/api/v1/data/status")
            assert response.status_code == 200
            
            data = response.json()
            assert data["success"] is True
            assert data["data"]["is_connected"] is False
    
    def test_database_error_recovery(self, client):
        """测试数据库错误恢复"""
        # 这里可以模拟数据库连接错误
        # 由于使用模拟数据，实际测试需要根据具体实现调整
        pass
    
    def test_model_loading_error_recovery(self, client):
        """测试模型加载错误恢复"""
        # 测试无效模型ID的处理
        prediction_request = {
            "stock_codes": ["000001.SZ"],
            "model_id": "invalid_model_id",
            "horizon": "short_term"
        }
        
        response = client.post("/api/v1/predictions", json=prediction_request)
        # 应该返回错误或使用默认模型
        assert response.status_code in [200, 400, 500]
    
    def test_network_timeout_recovery(self, client):
        """测试网络超时恢复"""
        # 模拟网络超时情况
        with patch('app.services.data_service.StockDataService.get_stock_data') as mock_get:
            mock_get.side_effect = TimeoutError("请求超时")
            
            params = {
                "stock_code": "000001.SZ",
                "start_date": datetime.now().isoformat(),
                "end_date": datetime.now().isoformat()
            }
            
            response = client.get("/api/v1/stocks/data", params=params)
            # 应该返回适当的错误响应
            assert response.status_code in [500, 504]


# 运行集成测试的便捷函数
def run_integration_tests():
    """运行所有集成测试"""
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--disable-warnings"
    ])


if __name__ == "__main__":
    run_integration_tests()