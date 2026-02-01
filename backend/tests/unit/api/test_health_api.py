"""
健康检查 API 单元测试
测试 /health 端点
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.v1.health import router


@pytest.fixture
def app():
    """创建仅包含健康检查路由的测试应用"""
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def client(app):
    """创建测试客户端"""
    return TestClient(app)


class TestHealthAPI:
    """健康检查 API 测试类"""

    def test_health_check_returns_200(self, client):
        """测试健康检查返回 200"""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_check_response_structure(self, client):
        """测试响应结构符合 StandardResponse"""
        response = client.get("/health")
        data = response.json()
        assert "success" in data
        assert "message" in data
        assert "data" in data
        assert isinstance(data["success"], bool)
        assert isinstance(data["message"], str)
        assert data["success"] is True

    def test_health_check_data_content(self, client):
        """测试 data 包含 status 与 version"""
        response = client.get("/health")
        data = response.json()
        payload = data["data"]
        assert payload["status"] == "healthy"
        assert payload["version"] == "1.0.0"

    def test_health_check_message(self, client):
        """测试成功时的 message"""
        response = client.get("/health")
        data = response.json()
        assert "正常" in data["message"]

    def test_health_check_methods(self, client):
        """测试仅 GET 允许"""
        assert client.get("/health").status_code == 200
        assert client.post("/health").status_code in (404, 405)
        assert client.put("/health").status_code in (404, 405)
        assert client.delete("/health").status_code in (404, 405)
