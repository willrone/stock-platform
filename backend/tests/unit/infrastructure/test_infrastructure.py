"""
基础设施测试

These tests describe the current repository layout and the lightweight FastAPI
application factory.  They deliberately resolve paths from this file instead of
assuming pytest is launched from the repository root.
"""

from pathlib import Path

import pytest
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text

from app.core.config import Settings, settings
from app.main import create_application
from app.middleware.error_handling import (
    ErrorHandlingMiddleware,
    RequestLoggingMiddleware,
)
from app.middleware.rate_limiting import RateLimitMiddleware

BACKEND_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = BACKEND_ROOT.parent
FRONTEND_ROOT = PROJECT_ROOT / "frontend"


class TestProjectStructure:
    """测试项目结构"""

    def test_backend_structure_exists(self) -> None:
        """测试后端目录结构是否存在"""
        assert BACKEND_ROOT.exists(), "backend目录不存在"

        assert (BACKEND_ROOT / "app").exists(), "app目录不存在"
        assert (BACKEND_ROOT / "app" / "core").exists(), "core目录不存在"
        assert (BACKEND_ROOT / "data").exists(), "data目录不存在"
        assert (BACKEND_ROOT / "tests").exists(), "tests目录不存在"

        assert (BACKEND_ROOT / "app" / "__init__.py").exists(), "app/__init__.py不存在"
        assert (BACKEND_ROOT / "app" / "main.py").exists(), "app/main.py不存在"
        assert (BACKEND_ROOT / "requirements.txt").exists(), "requirements.txt不存在"
        assert (BACKEND_ROOT / "pyproject.toml").exists(), "pyproject.toml不存在"

    def test_data_directories_exist(self) -> None:
        """测试数据目录结构"""
        data_path = BACKEND_ROOT / "data"
        assert data_path.exists(), "data目录不存在"

        for directory in ["stocks", "models", "logs"]:
            assert (data_path / directory).exists(), f"{directory}目录不存在"

    def test_frontend_structure_exists(self) -> None:
        """测试前端目录结构"""
        if not FRONTEND_ROOT.exists():
            pytest.skip("frontend目录不存在，跳过前端结构测试")

        assert (FRONTEND_ROOT / "package.json").exists(), "package.json不存在"
        assert (FRONTEND_ROOT / "tsconfig.json").exists(), "tsconfig.json不存在"
        assert (FRONTEND_ROOT / "next.config.js").exists(), "next.config.js不存在"
        assert (FRONTEND_ROOT / "src").exists(), "src目录不存在"
        assert (FRONTEND_ROOT / "src" / "app").exists(), "src/app目录不存在"


class TestConfiguration:
    """测试配置管理"""

    def test_settings_creation(self) -> None:
        """测试设置对象创建"""
        assert isinstance(settings, Settings)
        assert settings.APP_NAME == "Stock Prediction Platform"
        assert settings.APP_VERSION == "0.1.0"

    def test_settings_with_custom_values(self, temp_dir: Path) -> None:
        """测试自定义配置值"""
        custom_settings = Settings(
            DEBUG=True,
            DATABASE_URL=f"sqlite:///{temp_dir}/custom.db",
            DATA_ROOT_PATH=str(temp_dir),
        )

        assert custom_settings.DEBUG is True
        assert str(temp_dir) in custom_settings.DATABASE_URL
        assert custom_settings.DATA_ROOT_PATH == str(temp_dir)

    def test_database_url_sync_property(self, temp_dir: Path) -> None:
        """测试同步数据库URL属性"""
        test_settings = Settings(DATABASE_URL=f"sqlite+aiosqlite:///{temp_dir}/test.db")

        sync_url = test_settings.database_url_sync
        assert "sqlite:///" in sync_url
        assert "sqlite+aiosqlite://" not in sync_url


class TestApplication:
    """测试应用程序创建"""

    def test_app_creation(self) -> None:
        """测试FastAPI应用创建"""
        app = create_application()
        assert app.title == "股票预测平台API"
        assert app.version == "1.0.0"
        assert app.openapi_url == "/api/v1/openapi.json"

    def test_app_routes_exist(self) -> None:
        """测试应用路由存在"""
        app = create_application()
        routes = [route.path for route in app.routes]

        assert any(route.startswith("/api/v1") for route in routes), "API v1路由不存在"
        assert "/metrics" in routes, "metrics路由不存在"

    def test_middleware_configured(self) -> None:
        """测试当前应用中间件配置"""
        app = create_application()
        middleware_classes = [middleware.cls for middleware in app.user_middleware]

        assert CORSMiddleware in middleware_classes, "CORS中间件未配置"
        assert GZipMiddleware in middleware_classes, "GZip中间件未配置"
        assert RateLimitMiddleware in middleware_classes, "RateLimit中间件未配置"
        assert ErrorHandlingMiddleware in middleware_classes, "错误处理中间件未配置"
        assert RequestLoggingMiddleware in middleware_classes, "请求日志中间件未配置"


class TestDependencies:
    """测试依赖安装"""

    def test_required_packages_importable(self) -> None:
        """测试必需包可以导入"""
        import fastapi  # noqa: F401
        import pydantic  # noqa: F401
        import sqlalchemy  # noqa: F401

    def test_ml_packages_importable(self) -> None:
        """测试机器学习包可以导入"""
        try:
            import numpy  # noqa: F401
            import pandas  # noqa: F401
            import sklearn  # noqa: F401
        except ImportError as exc:
            pytest.skip(f"ML依赖可能未安装: {exc}")

    def test_data_packages_importable(self) -> None:
        """测试数据处理包可以导入"""
        import pandas  # noqa: F401
        import pyarrow  # noqa: F401


class TestEnvironmentFiles:
    """测试环境文件"""

    def test_env_example_exists(self) -> None:
        """测试环境配置示例文件存在"""
        env_example = BACKEND_ROOT / ".env.example"
        assert env_example.exists(), ".env.example文件不存在"

        content = env_example.read_text()
        assert "APP_NAME" in content, "APP_NAME配置缺失"
        assert "DATABASE_URL" in content, "DATABASE_URL配置缺失"
        assert "REMOTE_DATA_SERVICE_URL" in content, "REMOTE_DATA_SERVICE_URL配置缺失"

    def test_gitkeep_files_exist(self) -> None:
        """测试.gitkeep文件存在"""
        gitkeep_files = [
            BACKEND_ROOT / "data" / ".gitkeep",
            BACKEND_ROOT / "data" / "stocks" / ".gitkeep",
            BACKEND_ROOT / "data" / "models" / ".gitkeep",
            BACKEND_ROOT / "data" / "logs" / ".gitkeep",
        ]

        for gitkeep_file in gitkeep_files:
            assert gitkeep_file.exists(), f"{gitkeep_file}不存在"


@pytest.mark.integration
class TestIntegration:
    """集成测试"""

    def test_app_startup(self) -> None:
        """测试应用可创建并响应请求"""
        client = TestClient(create_application())
        response = client.get("/")
        assert response.status_code in [200, 404], "应用启动失败"

    def test_database_connection(self) -> None:
        """测试数据库连接"""
        engine = create_engine("sqlite:///:memory:", future=True)
        with engine.connect() as connection:
            assert connection.execute(text("SELECT 1")).scalar_one() == 1
