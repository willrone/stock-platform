"""
Pytest 配置和共享fixtures
"""

import asyncio
import tempfile
from pathlib import Path
from typing import AsyncGenerator, Generator

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import Settings
from app.core.database import Base, get_async_session
from app.main import create_application


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """创建事件循环用于测试"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """创建临时目录"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def test_settings(temp_dir: Path) -> Settings:
    """测试环境配置"""
    return Settings(
        DEBUG=True,
        DATABASE_URL=f"sqlite+aiosqlite:///{temp_dir}/test.db",
        DATA_ROOT_PATH=str(temp_dir),
        PARQUET_DATA_PATH=str(temp_dir / "stocks"),
        MODEL_STORAGE_PATH=str(temp_dir / "models"),
        QLIB_DATA_PATH=str(temp_dir / "qlib_data"),
        QLIB_CACHE_PATH=str(temp_dir / "qlib_cache"),
    )


@pytest_asyncio.fixture
async def test_db_engine(test_settings: Settings):
    """测试数据库引擎"""
    engine = create_async_engine(
        test_settings.DATABASE_URL,
        echo=False,
        future=True,
    )
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield engine
    
    await engine.dispose()


@pytest_asyncio.fixture
async def test_db_session(test_db_engine) -> AsyncGenerator[AsyncSession, None]:
    """测试数据库会话"""
    async_session = sessionmaker(
        test_db_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    
    async with async_session() as session:
        yield session


@pytest.fixture
def test_client(test_settings: Settings, test_db_session: AsyncSession) -> TestClient:
    """测试客户端"""
    app = create_application()
    
    # 覆盖依赖
    async def override_get_db():
        yield test_db_session
    
    app.dependency_overrides[get_async_session] = override_get_db
    
    return TestClient(app)