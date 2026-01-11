"""
数据库配置和连接管理
"""

from pathlib import Path
from typing import AsyncGenerator

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.core.config import settings


class Base(DeclarativeBase):
    """SQLAlchemy 基础模型类"""
    pass


# 异步数据库引擎
async_engine = create_async_engine(
    settings.DATABASE_URL.replace("sqlite://", "sqlite+aiosqlite://"),
    echo=settings.DEBUG,
    future=True,
)

# 同步数据库引擎（用于 Alembic 迁移和多进程任务执行）
# 优化连接池配置以支持多进程访问
sync_engine = create_engine(
    settings.database_url_sync,
    echo=settings.DEBUG,
    future=True,
    # 连接池配置（支持多进程）
    pool_size=10,  # 连接池大小
    max_overflow=20,  # 允许溢出的连接数
    pool_timeout=30,  # 获取连接的超时时间（秒）
    pool_pre_ping=True,  # 连接前ping检查，确保连接有效
    pool_recycle=3600,  # 连接回收时间（秒），1小时
)

# 会话工厂
AsyncSessionLocal = async_sessionmaker(
    async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

SessionLocal = sessionmaker(
    sync_engine,
    expire_on_commit=False,
)


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """获取异步数据库会话"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_db() -> None:
    """初始化数据库"""
    # 确保数据目录存在
    db_path = Path(settings.DATABASE_URL.replace("sqlite:///", ""))
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 导入所有模型以确保它们被注册到Base.metadata
    from app.models import task_models  # noqa: F401
    from app.models import backtest_detailed_models  # noqa: F401
    
    # 创建所有表
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)