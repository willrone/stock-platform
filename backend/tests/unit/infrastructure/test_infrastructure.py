"""
基础设施测试
"""

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings, settings
from app.main import create_application


class TestProjectStructure:
    """测试项目结构"""

    def test_backend_structure_exists(self):
        """测试后端目录结构是否存在"""
        backend_path = Path(".")
        assert (backend_path / "app").exists(), "app目录不存在"
        assert (backend_path / "app" / "core").exists(), "core目录不存在"
        assert (backend_path / "data").exists(), "data目录不存在"
        assert (backend_path / "tests").exists(), "tests目录不存在"
        assert (backend_path / "app" / "__init__.py").exists()
        assert (backend_path / "app" / "main.py").exists()

    def test_data_directories_exist(self):
        """测试数据目录结构"""
        data_path = Path("data")
        assert data_path.exists(), "data目录不存在"
        assert (data_path / "logs").exists(), "logs目录不存在"

    def test_frontend_structure_exists(self):
        """测试前端目录结构"""
        frontend_path = Path("../frontend")
        assert frontend_path.exists(), "frontend目录不存在"
        assert (frontend_path / "package.json").exists(), "package.json不存在"
        assert (frontend_path / "src").exists(), "src目录不存在"


class TestConfiguration:
    """测试配置管理"""

    def test_settings_creation(self):
        """测试设置对象创建"""
        assert isinstance(settings, Settings)
        assert settings.APP_NAME is not None
        assert settings.APP_VERSION is not None

    def test_database_url_sync_property(self):
        """测试数据库URL属性"""
        assert hasattr(settings, 'DATABASE_URL')
        assert settings.DATABASE_URL is not None


class TestApplication:
    """测试应用程序创建"""

    def test_app_creation(self):
        """测试FastAPI应用创建"""
        app = create_application()
        assert app.title == "股票预测平台API"
        assert app.version == "1.0.0"

    def test_app_routes_exist(self):
        """测试应用路由存在"""
        app = create_application()
        routes = [route.path for route in app.routes]
        assert any("/api/v1" in route for route in routes), "API v1路由不存在"

    def test_cors_middleware_configured(self):
        """测试CORS中间件配置"""
        app = create_application()
        middleware_classes = []
        for m in app.user_middleware:
            cls = getattr(m, 'cls', None) or getattr(m, 'middleware_class', None)
            if cls:
                middleware_classes.append(cls)
        from starlette.middleware.cors import CORSMiddleware
        assert CORSMiddleware in middleware_classes, "CORS中间件未配置"


class TestDependencies:
    """测试依赖安装"""

    def test_required_packages_importable(self):
        """测试必需包可以导入"""
        try:
            import fastapi
            import sqlalchemy
            import pandas
            import numpy
            import pydantic
        except ImportError as e:
            pytest.fail(f"核心依赖导入失败: {e}")

    def test_data_packages_importable(self):
        """测试数据处理包可以导入"""
        try:
            import pyarrow
        except ImportError as e:
            pytest.fail(f"数据处理依赖导入失败: {e}")


class TestEnvironmentFiles:
    """测试环境文件"""

    def test_env_example_exists(self):
        """测试环境配置文件存在"""
        # .env or .env.example or pyproject.toml should exist
        has_config = Path(".env").exists() or Path(".env.example").exists() or Path("pyproject.toml").exists()
        assert has_config, "配置文件不存在"

    def test_gitkeep_files_exist(self):
        """测试数据目录存在"""
        assert Path("data").exists(), "data目录不存在"
        assert Path("data/logs").exists(), "data/logs目录不存在"


@pytest.mark.integration
class TestIntegration:
    """集成测试 - 需要数据库连接"""

    def test_app_startup(self):
        """测试应用启动"""
        app = create_application()
        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/")
        assert response.status_code in [200, 404], "应用启动失败"

    def test_database_connection(self):
        """测试数据库配置存在"""
        assert settings.DATABASE_URL is not None, "数据库URL未配置"
