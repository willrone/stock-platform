"""
数据库配置和连接管理（PostgreSQL）
"""

import asyncio
import json
import math
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Callable, TypeVar

from loguru import logger
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.core.config import settings


def _pg_json_serializer(obj: Any) -> str:
    """自定义 JSON 序列化器，将 NaN/Infinity 替换为 0（PostgreSQL 不接受非标准 JSON token）"""

    class _SafeEncoder(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, float):
                if math.isnan(o) or math.isinf(o):
                    return 0.0
            return super().default(o)

        def encode(self, o):
            return super().encode(self._sanitize(o))

        def _sanitize(self, o):
            if isinstance(o, float):
                if math.isnan(o) or math.isinf(o):
                    return 0.0
                return o
            if isinstance(o, dict):
                return {k: self._sanitize(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)):
                return [self._sanitize(v) for v in o]
            return o

    return json.dumps(obj, cls=_SafeEncoder, default=str)


class Base(DeclarativeBase):
    """SQLAlchemy 基础模型类"""


# ──────────────────────────── 引擎 ────────────────────────────

# 异步引擎（postgresql+asyncpg）
async_engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    future=True,
    pool_size=20,
    max_overflow=10,
    pool_timeout=30,
    pool_pre_ping=True,
    pool_recycle=3600,
    json_serializer=_pg_json_serializer,
)

# 同步引擎（postgresql://，用于 Alembic 迁移和多进程任务执行）
sync_engine = create_engine(
    settings.database_url_sync,
    echo=settings.DEBUG,
    future=True,
    pool_size=20,
    max_overflow=10,
    pool_timeout=30,
    pool_pre_ping=True,
    pool_recycle=3600,
    json_serializer=_pg_json_serializer,
)

# ──────────────────────────── 会话工厂 ────────────────────────────

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


# ──────────────────────────── 重试机制 ────────────────────────────

T = TypeVar("T")


def _is_retryable_pg_error(e: OperationalError) -> bool:
    """判断是否为 PostgreSQL 可重试错误（死锁 / 序列化失败）"""
    msg = str(e).lower()
    return (
        "deadlock detected" in msg
        or "could not serialize access" in msg
        or "serialization failure" in msg
    )


async def retry_db_operation(
    operation: Callable[[], Any],
    max_retries: int = 5,
    retry_delay: float = 0.5,
    backoff_factor: float = 2.0,
    operation_name: str = "数据库操作",
) -> Any:
    """
    重试数据库操作，处理 PostgreSQL 死锁 / 序列化错误

    Args:
        operation: 要执行的异步操作函数
        max_retries: 最大重试次数
        retry_delay: 初始重试延迟（秒）
        backoff_factor: 退避因子
        operation_name: 操作名称，用于日志记录

    Returns:
        操作的结果
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
            if _is_retryable_pg_error(e):
                last_exception = e
                if attempt < max_retries:
                    logger.warning(
                        f"{operation_name} 遇到可重试错误，"
                        f"第 {attempt + 1}/{max_retries} 次重试，"
                        f"等待 {current_delay:.2f} 秒后重试"
                    )
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff_factor
                else:
                    logger.error(
                        f"{operation_name} 重试 {max_retries} 次后仍然失败: {e}"
                    )
            else:
                raise
        except Exception:
            raise

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
    重试同步数据库操作，处理 PostgreSQL 死锁 / 序列化错误

    Args:
        operation: 要执行的操作函数
        max_retries: 最大重试次数
        retry_delay: 初始重试延迟（秒）
        backoff_factor: 退避因子
        operation_name: 操作名称，用于日志记录
        session: SQLAlchemy session，用于在重试前 rollback
    """
    import time

    last_exception = None
    current_delay = retry_delay

    for attempt in range(max_retries + 1):
        try:
            result = operation()
            return result
        except OperationalError as e:
            if _is_retryable_pg_error(e):
                last_exception = e
                if session is not None:
                    try:
                        session.rollback()
                    except Exception:
                        pass
                if attempt < max_retries:
                    logger.warning(
                        f"{operation_name} 遇到可重试错误，"
                        f"第 {attempt + 1}/{max_retries} 次重试，"
                        f"等待 {current_delay:.2f} 秒后重试"
                    )
                    time.sleep(current_delay)
                    current_delay *= backoff_factor
                else:
                    logger.error(
                        f"{operation_name} 重试 {max_retries} 次后仍然失败: {e}"
                    )
            else:
                raise
        except Exception:
            raise

    if last_exception:
        raise last_exception


# ──────────────────────────── 初始化 ────────────────────────────


async def init_db() -> None:
    """初始化数据库（创建所有表）"""
    # 导入所有模型以确保它们被注册到 Base.metadata
    from app.models import backtest_detailed_models  # noqa: F401
    from app.models import strategy_config_models  # noqa: F401
    from app.models import task_models  # noqa: F401

    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
