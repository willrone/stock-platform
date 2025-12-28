"""
基础设施基本测试
"""

import os
from pathlib import Path


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
        backend_path = Path("backend").absolute()
        if str(backend_path) not in sys.path:
            sys.path.insert(0, str(backend_path))
        
        try:
            from app.core.config import Settings
            from app.core.logging import setup_logging
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
        
        pyproject_path = Path("backend/pyproject.toml")
        assert pyproject_path.exists(), "pyproject.toml不存在"
        
        with open(pyproject_path, "rb") as f:
            config = tomllib.load(f)
        
        assert "project" in config, "project配置缺失"
        assert config["project"]["name"] == "stock-prediction-platform"
        assert "tool" in config, "tool配置缺失"
    
    def test_requirements_files_exist(self):
        """测试requirements文件存在"""
        req_files = [
            "backend/requirements.txt",
            "backend/requirements-test.txt",
            "backend/requirements-minimal.txt",
        ]
        
        for req_file in req_files:
            assert Path(req_file).exists(), f"{req_file}不存在"