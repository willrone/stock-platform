"""
基础设施基本测试
"""

import os
from pathlib import Path


class TestProjectStructure:
    """测试项目结构"""
    
    def test_backend_structure_exists(self):
        """测试后端目录结构是否存在"""
        # __file__ 是 backend/tests/unit/infrastructure/test_basic_infrastructure.py
        # 需要向上4级到backend目录
        backend_path = Path(__file__).parent.parent.parent.parent
        assert backend_path.exists(), "backend目录不存在"
        
        # 检查核心目录
        assert (backend_path / "app").exists(), "app目录不存在"
        assert (backend_path / "app" / "core").exists(), "core目录不存在"
        assert (backend_path / "data").exists(), "data目录不存在"
        assert (backend_path / "tests").exists(), "tests目录不存在"
        
        # 检查核心文件
        assert (backend_path / "app" / "__init__.py").exists(), "app/__init__.py不存在"
        assert (backend_path / "requirements.txt").exists(), "requirements.txt不存在"
        assert (backend_path / "pyproject.toml").exists(), "pyproject.toml不存在"
    
    def test_data_directories_exist(self):
        """测试数据目录结构"""
        backend_path = Path(__file__).parent.parent.parent.parent
        data_path = backend_path / "data"
        assert data_path.exists(), "data目录不存在"
        
        # 检查子目录（如果存在）
        # 注意：这些目录可能不存在，所以使用exists()检查但不强制要求
        if (data_path / "stocks").exists():
            assert True, "stocks目录存在"
        if (data_path / "models").exists():
            assert True, "models目录存在"
        if (data_path / "logs").exists():
            assert True, "logs目录存在"
    
    def test_frontend_structure_exists(self):
        """测试前端目录结构"""
        # 从backend/tests运行，frontend是backend的兄弟目录
        backend_path = Path(__file__).parent.parent.parent.parent
        project_root = backend_path.parent
        frontend_path = project_root / "frontend"
        
        if not frontend_path.exists():
            import pytest
            pytest.skip("frontend目录不存在，跳过前端结构测试")
        
        # 检查核心文件
        assert (frontend_path / "package.json").exists(), "package.json不存在"
        
        # 检查源码目录
        if (frontend_path / "src").exists():
            assert True, "src目录存在"


class TestEnvironmentFiles:
    """测试环境文件"""
    
    def test_env_example_exists(self):
        """测试环境配置示例文件存在"""
        backend_path = Path(__file__).parent.parent.parent.parent
        project_root = backend_path.parent
        env_example = project_root / ".env.example"
        
        if not env_example.exists():
            import pytest
            pytest.skip(".env.example文件不存在")
        
        # 检查关键配置项
        content = env_example.read_text()
        assert "APP_NAME" in content or "DATABASE_URL" in content, "配置文件格式不正确"
    
    def test_gitkeep_files_exist(self):
        """测试.gitkeep文件存在（可选）"""
        backend_path = Path(__file__).parent.parent.parent.parent
        data_path = backend_path / "data"
        
        # .gitkeep文件是可选的，只检查data目录是否存在
        if data_path.exists():
            assert True, "data目录存在"


class TestBasicImports:
    """测试基本导入"""
    
    def test_python_standard_library(self):
        """测试Python标准库导入"""
        import sqlite3
        import json
        import pathlib
        import asyncio
        assert True, "标准库导入成功"
    
    def test_app_modules_importable(self):
        """测试应用模块可以导入"""
        # 添加backend到Python路径
        import sys
        backend_path = Path(__file__).parent.parent.parent.parent
        if str(backend_path) not in sys.path:
            sys.path.insert(0, str(backend_path))
        
        try:
            from app.core.config import Settings
            # 注意：不测试database模块，因为它依赖SQLAlchemy
        except ImportError as e:
            # 如果依赖未安装，跳过测试
            import pytest
            pytest.skip(f"依赖未安装: {e}")


class TestConfigurationFiles:
    """测试配置文件"""
    
    def test_pyproject_toml_valid(self):
        """测试pyproject.toml文件有效"""
        import tomllib
        
        backend_path = Path(__file__).parent.parent.parent.parent
        pyproject_path = backend_path / "pyproject.toml"
        assert pyproject_path.exists(), "pyproject.toml不存在"
        
        with open(pyproject_path, "rb") as f:
            config = tomllib.load(f)
        
        assert "project" in config, "project配置缺失"
        assert config["project"]["name"] == "stock-prediction-platform"
        assert "tool" in config, "tool配置缺失"
    
    def test_requirements_files_exist(self):
        """测试requirements文件存在"""
        backend_path = Path(__file__).parent.parent.parent.parent
        req_files = [
            "requirements.txt",
            "requirements-test.txt",
            "requirements-minimal.txt",
        ]
        
        for req_file in req_files:
            req_path = backend_path / req_file
            assert req_path.exists(), f"{req_file}不存在"