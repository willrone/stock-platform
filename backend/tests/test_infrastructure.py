"""
基础设施测试
"""

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings, settings
from app.core.database import Base, async_engine
from app.main import create_application


class TestProjectStructure:
    """测试项目结构"""
    
    def test_backend_structure_exists(self):
        """测试后端目录结构是否存在"""
        backend_path = Path("backend")
        assert backend_path.exists(), "backend目录不存在"
        
        # 检查核心目录
        assert (backend_path / "app").exists(), "app目录不存在"
        assert (backend_path / "app" / "core").exists(), "core目录不存在"
        assert (backend_path / "data").exists(), "data目录不存在"
        assert (backend_path / "tests").exists(), "tests目录不存在"
        
        # 检查核心文件
        assert (backend_path / "app" / "__init__.py").exists(), "app/__init__.py不存在"
        assert (backend_path / "app" / "main.py").exists(), "app/main.py不存在"
        assert (backend_path / "requirements.txt").exists(), "requirements.txt不存在"
        assert (backend_path / "pyproject.toml").exists(), "pyproject.toml不存在"
    
    def test_data_directories_exist(self):
        """测试数据目录结构"""
        data_path = Path("backend/data")
        assert data_path.exists(), "data目录不存在"
        
        # 检查子目录
        assert (data_path / "stocks").exists(), "stocks目录不存在"
        assert (data_path / "models").exists(), "models目录不存在"
        assert (data_path / "logs").exists(), "logs目录不存在"
    
    def test_frontend_structure_exists(self):
        """测试前端目录结构"""
        frontend_path = Path("frontend")
        assert frontend_path.exists(), "frontend目录不存在"
        
        # 检查核心文件
        assert (frontend_path / "package.json").exists(), "package.json不存在"
        assert (frontend_path / "tsconfig.json").exists(), "tsconfig.json不存在"
        assert (frontend_path / "next.config.js").exists(), "next.config.js不存在"
        
        # 检查源码目录
        assert (frontend_path / "src").exists(), "src目录不存在"
        assert (frontend_path / "src" / "app").exists(), "src/app目录不存在"


class TestConfiguration:
    """测试配置管理"""
    
    def test_settings_creation(self):
        """测试设置对象创建"""
        assert isinstance(settings, Settings)
        assert settings.APP_NAME == "Stock Prediction Platform"
        assert settings.APP_VERSION == "0.1.0"
    
    def test_settings_with_custom_values(self, temp_dir: Path):
        """测试自定义配置值"""
        custom_settings = Settings(
            DEBUG=True,
            DATABASE_URL=f"sqlite:///{temp_dir}/custom.db",
            DATA_ROOT_PATH=str(temp_dir),
        )
        
        assert custom_settings.DEBUG is True
        assert str(temp_dir) in custom_settings.DATABASE_URL
        assert custom_settings.DATA_ROOT_PATH == str(temp_dir)
    
    def test_database_url_sync_property(self, temp_dir: Path):
        """测试同步数据库URL属性"""
        test_settings = Settings(
            DATABASE_URL=f"sqlite+aiosqlite:///{temp_dir}/test.db"
        )
        
        sync_url = test_settings.database_url_sync
        assert "sqlite:///" in sync_url
        assert "sqlite+aiosqlite://" not in sync_url


class TestApplication:
    """测试应用程序创建"""
    
    def test_app_creation(self):
        """测试FastAPI应用创建"""
        app = create_application()
        assert app.title == "Stock Prediction Platform"
        assert app.version == "0.1.0"
        assert "/api/v1" in app.openapi_url
    
    def test_app_routes_exist(self):
        """测试应用路由存在"""
        app = create_application()
        routes = [route.path for route in app.routes]
        
        # 检查基本路由存在
        assert any("/api/v1" in route for route in routes), "API v1路由不存在"
    
    def test_cors_middleware_configured(self):
        """测试CORS中间件配置"""
        app = create_application()
        
        # 检查中间件是否添加
        middleware_types = [type(middleware.cls) for middleware in app.user_middleware]
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.middleware.gzip import GZipMiddleware
        
        assert CORSMiddleware in middleware_types, "CORS中间件未配置"
        assert GZipMiddleware in middleware_types, "GZip中间件未配置"


class TestDependencies:
    """测试依赖安装"""
    
    def test_required_packages_importable(self):
        """测试必需包可以导入"""
        # 测试核心依赖
        try:
            import fastapi
            import sqlalchemy
            import pandas
            import numpy
            import pydantic
            import uvicorn
        except ImportError as e:
            pytest.fail(f"核心依赖导入失败: {e}")
    
    def test_ml_packages_importable(self):
        """测试机器学习包可以导入"""
        try:
            import torch
            import sklearn
            # 注意：qlib可能需要额外配置，这里先跳过
            # import qlib
        except ImportError as e:
            pytest.skip(f"ML依赖可能未安装: {e}")
    
    def test_data_packages_importable(self):
        """测试数据处理包可以导入"""
        try:
            import pyarrow
            import polars
        except ImportError as e:
            pytest.fail(f"数据处理依赖导入失败: {e}")


class TestEnvironmentFiles:
    """测试环境文件"""
    
    def test_env_example_exists(self):
        """测试环境配置示例文件存在"""
        env_example = Path("backend/.env.example")
        assert env_example.exists(), ".env.example文件不存在"
        
        # 检查关键配置项
        content = env_example.read_text()
        assert "APP_NAME" in content, "APP_NAME配置缺失"
        assert "DATABASE_URL" in content, "DATABASE_URL配置缺失"
        assert "REMOTE_DATA_SERVICE_URL" in content, "REMOTE_DATA_SERVICE_URL配置缺失"
    
    def test_gitkeep_files_exist(self):
        """测试.gitkeep文件存在"""
        gitkeep_files = [
            "backend/data/.gitkeep",
            "backend/data/stocks/.gitkeep",
            "backend/data/models/.gitkeep",
            "backend/data/logs/.gitkeep",
        ]
        
        for gitkeep_file in gitkeep_files:
            assert Path(gitkeep_file).exists(), f"{gitkeep_file}不存在"


@pytest.mark.integration
class TestIntegration:
    """集成测试"""
    
    def test_app_startup(self, test_client: TestClient):
        """测试应用启动"""
        # 测试健康检查端点（如果存在）
        response = test_client.get("/")
        # 由于我们还没有实现根路由，这里可能返回404，这是正常的
        assert response.status_code in [200, 404], "应用启动失败"
    
    def test_database_connection(self, test_db_session):
        """测试数据库连接"""
        # 这个测试通过fixture的成功创建来验证数据库连接
        assert test_db_session is not None, "数据库会话创建失败"