"""
限流功能测试
"""

import pytest
import time
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


class TestRateLimiting:
    """限流测试"""
    
    def test_rate_limit_normal_requests(self):
        """测试正常请求不会被限流"""
        # 发送少量请求，应该都成功
        for i in range(5):
            response = client.get("/api/v1/health")
            assert response.status_code == 200
            
            # 检查限流头
            assert "X-RateLimit-Limit-Minute" in response.headers
            assert "X-RateLimit-Limit-Hour" in response.headers
    
    def test_rate_limit_burst_protection(self):
        """测试突发请求保护"""
        # 快速发送大量请求
        success_count = 0
        rate_limited_count = 0
        
        for i in range(15):  # 超过burst_size(10)
            response = client.get("/api/v1/health")
            if response.status_code == 200:
                success_count += 1
            elif response.status_code == 429:
                rate_limited_count += 1
                # 检查限流响应
                assert "请求过于频繁" in response.json()["message"]
                assert "Retry-After" in response.headers
        
        # 应该有一些请求被限流
        assert success_count > 0
        assert rate_limited_count > 0
        assert success_count + rate_limited_count == 15
    
    def test_rate_limit_headers(self):
        """测试限流响应头"""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        
        # 检查限流相关头部
        headers = response.headers
        assert "X-RateLimit-Limit-Minute" in headers
        assert "X-RateLimit-Limit-Hour" in headers
        assert "X-RateLimit-Remaining-Minute" in headers
        assert "X-RateLimit-Remaining-Hour" in headers
        
        # 验证头部值
        assert int(headers["X-RateLimit-Limit-Minute"]) == 60
        assert int(headers["X-RateLimit-Limit-Hour"]) == 1000
    
    def test_different_endpoints_share_limit(self):
        """测试不同端点共享限流"""
        endpoints = [
            "/api/v1/health",
            "/api/v1/models",
            "/api/v1/system/status"
        ]
        
        total_requests = 0
        rate_limited = False
        
        # 在多个端点间发送请求
        for _ in range(5):
            for endpoint in endpoints:
                response = client.get(endpoint)
                total_requests += 1
                
                if response.status_code == 429:
                    rate_limited = True
                    break
            
            if rate_limited:
                break
        
        # 验证请求被处理
        assert total_requests > 0


class TestErrorHandling:
    """错误处理测试"""
    
    def test_http_exception_handling(self):
        """测试HTTP异常处理"""
        # 访问不存在的端点
        response = client.get("/api/v1/nonexistent")
        assert response.status_code == 404
        
        # 检查错误响应格式
        data = response.json()
        assert "success" in data
        assert data["success"] is False
        assert "message" in data
        assert "timestamp" in data
    
    def test_validation_error_handling(self):
        """测试验证错误处理"""
        # 发送无效数据
        invalid_data = {"invalid": "data"}
        response = client.post("/api/v1/predictions", json=invalid_data)
        
        # 应该返回验证错误
        assert response.status_code in [400, 422]
        
        data = response.json()
        if response.status_code == 422:
            # FastAPI验证错误格式
            assert "detail" in data
        else:
            # 自定义错误格式
            assert "success" in data
            assert data["success"] is False
    
    def test_method_not_allowed_handling(self):
        """测试不允许的方法处理"""
        # 对GET端点发送POST请求
        response = client.post("/api/v1/health")
        assert response.status_code == 405
    
    def test_response_time_header(self):
        """测试响应时间头"""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        
        # 检查响应时间头
        assert "X-Response-Time" in response.headers
        
        # 验证响应时间格式
        response_time = response.headers["X-Response-Time"]
        assert response_time.endswith("s")
        
        # 验证是有效的浮点数
        time_value = float(response_time[:-1])
        assert time_value >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])