"""
数据库配置和连接管理
"""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, TypeVar

from loguru import logger
from sqlalchemy import create_engine, event, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.core.config import settings


class Base(DeclarativeBase):
    """SQLAlchemy 基础模型类"""


# SQLite 连接配置，启用 WAL 模式以提高并发性能
def _configure_sqlite_connection(dbapi_conn, connection_record):
    """配置 SQLite 连接，启用 WAL 模式"""
    if "sqlite" in settings.DATABASE_URL.lower():
        # 启用 WAL 模式（Write-Ahead Logging），提高并发性能
        dbapi_conn.execute("PRAGMA journal_mode=WAL")
        # 设置超时时间（毫秒），默认 30 秒
        dbapi_conn.execute("PRAGMA busy_timeout=30000")
        # 启用外键约束
        dbapi_conn.execute("PRAGMA foreign_keys=ON")
        # 优化同步设置（NORMAL 模式在 WAL 模式下性能更好）
        dbapi_conn.execute("PRAGMA synchronous=NORMAL")


# 异步数据库引擎
# 对于异步引擎，需要在连接时配置 SQLite
async def _configure_async_sqlite_connection(dbapi_conn, connection_record):
    """配置异步 SQLite 连接"""
    _configure_sqlite_connection(dbapi_conn, connection_record)


async_engine = create_async_engine(
    settings.DATABASE_URL.replace("sqlite://", "sqlite+aiosqlite://"),
    echo=settings.DEBUG,
    future=True,
    # SQLite 特定配置
    connect_args={
        "check_same_thread": False,  # 允许多线程访问
        "timeout": 30.0,  # 连接超时时间（秒）
    }
    if "sqlite" in settings.DATABASE_URL.lower()
    else {},
    pool_pre_ping=True,  # 连接前ping检查
)

# 为异步引擎注册 SQLite 连接配置
# 注意：对于 aiosqlite，需要在 sync_engine 上注册事件
if "sqlite" in settings.DATABASE_URL.lower():

    @event.listens_for(async_engine.sync_engine, "connect")
    def set_sqlite_pragma_async(dbapi_conn, connection_record):
        _configure_sqlite_connection(dbapi_conn, connection_record)


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
    # SQLite 特定配置
    connect_args={
        "check_same_thread": False,  # 允许多线程访问
        "timeout": 30.0,  # 连接超时时间（秒）
    }
    if "sqlite" in settings.database_url_sync.lower()
    else {},
)

# 为同步引擎注册 SQLite 连接配置
if "sqlite" in settings.database_url_sync.lower():

    @event.listens_for(sync_engine, "connect")
    def set_sqlite_pragma_sync(dbapi_conn, connection_record):
        _configure_sqlite_connection(dbapi_conn, connection_record)


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
    """获取异步数据库会话（异步生成器，用于依赖注入）"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


@asynccontextmanager
async def get_async_session_context() -> AsyncGenerator[AsyncSession, None]:
    """获取异步数据库会话（异步上下文管理器，用于直接使用）"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


# 重试机制相关函数
T = TypeVar("T")


async def retry_db_operation(
    operation: Callable[[], Any],
    max_retries: int = 5,
    retry_delay: float = 0.5,
    backoff_factor: float = 2.0,
    operation_name: str = "数据库操作",
) -> Any:
    """
    重试数据库操作，处理 database is locked 错误

    Args:
        operation: 要执行的异步操作函数
        max_retries: 最大重试次数
        retry_delay: 初始重试延迟（秒）
        backoff_factor: 退避因子，每次重试延迟时间乘以该因子
        operation_name: 操作名称，用于日志记录

    Returns:
        操作的结果

    Raises:
        最后一次尝试的异常
    """
    last_exception = None
    current_delay = retry_delay

    for attempt in range(max_retries + 1):
        try:
            if asyncio.iscoroutinefunction(operation):
                result = await operation()
            else:
                result = operation()
            return result
        except OperationalError as e:
            error_msg = str(e).lower()
            if "database is locked" in error_msg or "database locked" in error_msg:
                last_exception = e
                if attempt < max_retries:
                    logger.warning(
                        f"{operation_name} 遇到数据库锁定，"
                        f"第 {attempt + 1}/{max_retries} 次重试，"
                        f"等待 {current_delay:.2f} 秒后重试"
                    )
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff_factor
                else:
                    logger.error(f"{operation_name} 重试 {max_retries} 次后仍然失败: {e}")
            else:
                # 非锁定错误，直接抛出
                raise
        except Exception:
            # 其他异常，直接抛出
            raise

    # 如果所有重试都失败，抛出最后一次的异常
    if last_exception:
        raise last_exception


def retry_db_operation_sync(
    operation: Callable[[], Any],
    max_retries: int = 5,
    retry_delay: float = 0.5,
    backoff_factor: float = 2.0,
    operation_name: str = "数据库操作",
    session: Any = None,
) -> Any:
    """
    重试同步数据库操作，处理 database is locked 错误

    Args:
        operation: 要执行的操作函数
        max_retries: 最大重试次数
        retry_delay: 初始重试延迟（秒）
        backoff_factor: 退避因子
        operation_name: 操作名称，用于日志记录
        session: SQLAlchemy session，用于在重试前 rollback

    Returns:
        操作的结果

    Raises:
        最后一次尝试的异常
    """
    import time

    last_exception = None
    current_delay = retry_delay

    for attempt in range(max_retries + 1):
        try:
            result = operation()
            return result
        except OperationalError as e:
            error_msg = str(e).lower()
            if "database is locked" in error_msg or "database locked" in error_msg:
                last_exception = e
                # rollback session 以清除 PendingRollbackError 状态
                if session is not None:
                    try:
                        session.rollback()
                    except Exception:
                        pass
                if attempt < max_retries:
                    logger.warning(
                        f"{operation_name} 遇到数据库锁定，"
                        f"第 {attempt + 1}/{max_retries} 次重试，"
                        f"等待 {current_delay:.2f} 秒后重试"
                    )
                    time.sleep(current_delay)
                    current_delay *= backoff_factor
                else:
                    logger.error(f"{operation_name} 重试 {max_retries} 次后仍然失败: {e}")
            else:
                raise
        except Exception:
            raise

    if last_exception:
        raise last_exception


async def init_db() -> None:
    """初始化数据库"""
    # 确保数据目录存在
    db_path = Path(
        settings.DATABASE_URL.replace("sqlite:///", "").replace(
            "sqlite+aiosqlite:///", ""
        )
    )
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # 导入所有模型以确保它们被注册到Base.metadata
    from app.models import backtest_detailed_models  # noqa: F401
    from app.models import strategy_config_models  # noqa: F401
    from app.models import task_models  # noqa: F401

    # 创建所有表
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

        # 如果是 SQLite，启用 WAL 模式
        # 注意：WAL 模式已经在连接时通过事件监听器配置，这里只是确保设置正确
        if "sqlite" in settings.DATABASE_URL.lower():
            # 使用 text() 包装 SQL 语句以便在异步引擎中执行
            await conn.execute(text("PRAGMA journal_mode=WAL"))
            await conn.execute(text("PRAGMA busy_timeout=30000"))
            await conn.execute(text("PRAGMA foreign_keys=ON"))
            await conn.execute(text("PRAGMA synchronous=NORMAL"))
