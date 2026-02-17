"""
连接池管理服务
实现HTTP客户端连接池和数据库连接池管理
"""

import asyncio
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

import httpx
import psutil
from loguru import logger
from sqlalchemy import create_engine, pool
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker


class ConnectionStatus(Enum):
    """连接状态"""

    IDLE = "idle"
    ACTIVE = "active"
    CLOSED = "closed"
    ERROR = "error"


@dataclass
class ConnectionStats:
    """连接统计信息"""

    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    failed_connections: int = 0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time_ms: float = 0.0
    peak_connections: int = 0
    pool_utilization: float = 0.0


@dataclass
class PoolConfig:
    """连接池配置"""

    max_connections: int = 100
    min_connections: int = 5
    max_keepalive_connections: int = 20
    keepalive_expiry: float = 30.0
    timeout: float = 30.0
    retry_attempts: int = 3
    retry_delay: float = 1.0
    health_check_interval: float = 60.0


class HTTPConnectionPool:
    """HTTP连接池管理器"""

    def __init__(self, config: PoolConfig):
        self.config = config
        self.stats = ConnectionStats()
        self._clients: Dict[str, httpx.AsyncClient] = {}
        self._client_stats: Dict[str, ConnectionStats] = {}
        self._lock = asyncio.Lock()

        # 健康检查任务
        self._health_check_task = None
        self._start_health_check()

    def _start_health_check(self):
        """启动健康检查任务"""

        async def health_check_loop():
            while True:
                try:
                    await self._perform_health_check()
                    await asyncio.sleep(self.config.health_check_interval)
                except Exception as e:
                    logger.error(f"HTTP连接池健康检查失败: {e}")
                    await asyncio.sleep(10)

        self._health_check_task = asyncio.create_task(health_check_loop())

    async def _perform_health_check(self):
        """执行健康检查"""
        async with self._lock:
            for pool_name, client in self._clients.items():
                try:
                    # 检查客户端是否仍然有效
                    if client.is_closed:
                        logger.warning(f"HTTP客户端已关闭，重新创建: {pool_name}")
                        await self._recreate_client(pool_name)

                except Exception as e:
                    logger.error(f"HTTP客户端健康检查失败 {pool_name}: {e}")

    async def _recreate_client(self, pool_name: str):
        """重新创建HTTP客户端"""
        if pool_name in self._clients:
            old_client = self._clients[pool_name]
            if not old_client.is_closed:
                await old_client.aclose()

        # 创建新客户端
        limits = httpx.Limits(
            max_connections=self.config.max_connections,
            max_keepalive_connections=self.config.max_keepalive_connections,
            keepalive_expiry=self.config.keepalive_expiry,
        )

        timeout = httpx.Timeout(self.config.timeout)

        self._clients[pool_name] = httpx.AsyncClient(
            limits=limits, timeout=timeout, follow_redirects=True
        )

        logger.info(f"重新创建HTTP客户端: {pool_name}")

    async def get_client(self, pool_name: str = "default") -> httpx.AsyncClient:
        """获取HTTP客户端"""
        async with self._lock:
            if pool_name not in self._clients:
                await self._create_client(pool_name)

            client = self._clients[pool_name]

            # 检查客户端状态
            if client.is_closed:
                await self._recreate_client(pool_name)
                client = self._clients[pool_name]

            return client

    async def _create_client(self, pool_name: str):
        """创建HTTP客户端"""
        limits = httpx.Limits(
            max_connections=self.config.max_connections,
            max_keepalive_connections=self.config.max_keepalive_connections,
            keepalive_expiry=self.config.keepalive_expiry,
        )

        timeout = httpx.Timeout(self.config.timeout)

        self._clients[pool_name] = httpx.AsyncClient(
            limits=limits, timeout=timeout, follow_redirects=True
        )

        self._client_stats[pool_name] = ConnectionStats()

        logger.info(f"创建HTTP客户端: {pool_name}")

    @asynccontextmanager
    async def request(
        self, method: str, url: str, pool_name: str = "default", **kwargs
    ):
        """执行HTTP请求"""
        client = await self.get_client(pool_name)
        stats = self._client_stats.get(pool_name, ConnectionStats())

        start_time = time.time()

        try:
            stats.total_requests += 1
            self.stats.total_requests += 1

            # 执行请求
            response = await client.request(method, url, **kwargs)

            # 更新成功统计
            stats.successful_requests += 1
            self.stats.successful_requests += 1

            # 计算响应时间
            response_time = (time.time() - start_time) * 1000
            stats.average_response_time_ms = (
                stats.average_response_time_ms * (stats.successful_requests - 1)
                + response_time
            ) / stats.successful_requests

            yield response

        except Exception as e:
            # 更新失败统计
            stats.failed_requests += 1
            self.stats.failed_requests += 1

            logger.error(f"HTTP请求失败 {method} {url}: {e}")
            raise

    async def get_pool_stats(self, pool_name: str = "default") -> ConnectionStats:
        """获取连接池统计信息"""
        if pool_name not in self._client_stats:
            return ConnectionStats()

        stats = self._client_stats[pool_name]

        # 更新连接池利用率
        if pool_name in self._clients:
            self._clients[pool_name]
            # 这里需要根据httpx的实际API来获取连接信息
            # 由于httpx没有直接暴露连接池信息，我们使用估算值
            stats.pool_utilization = min(
                1.0, stats.active_connections / self.config.max_connections
            )

        return stats

    async def close_all(self):
        """关闭所有连接"""
        async with self._lock:
            for pool_name, client in self._clients.items():
                try:
                    await client.aclose()
                    logger.info(f"关闭HTTP客户端: {pool_name}")
                except Exception as e:
                    logger.error(f"关闭HTTP客户端失败 {pool_name}: {e}")

            self._clients.clear()
            self._client_stats.clear()

        # 取消健康检查任务
        if self._health_check_task:
            self._health_check_task.cancel()


class DatabaseConnectionPool:
    """数据库连接池管理器"""

    def __init__(self, database_url: str, config: PoolConfig):
        self.database_url = database_url
        self.config = config
        self.stats = ConnectionStats()

        # 创建同步引擎
        self.sync_engine = create_engine(
            database_url,
            poolclass=pool.QueuePool,
            pool_size=config.min_connections,
            max_overflow=config.max_connections - config.min_connections,
            pool_timeout=config.timeout,
            pool_recycle=3600,  # 1小时回收连接
            pool_pre_ping=True,  # 连接前ping检查
            echo=False,
        )

        # 创建异步引擎（如果支持）
        self.async_engine = None
        if database_url.startswith(
            ("postgresql+asyncpg", "mysql+aiomysql", "sqlite+aiosqlite")
        ):
            self.async_engine = create_async_engine(
                database_url,
                pool_size=config.min_connections,
                max_overflow=config.max_connections - config.min_connections,
                pool_timeout=config.timeout,
                pool_recycle=3600,
                pool_pre_ping=True,
                echo=False,
            )

        # 创建会话工厂
        self.SessionLocal = sessionmaker(
            bind=self.sync_engine, autocommit=False, autoflush=False
        )

        if self.async_engine:
            self.AsyncSessionLocal = sessionmaker(
                bind=self.async_engine,
                class_=AsyncSession,
                autocommit=False,
                autoflush=False,
            )

        # 启动监控
        self._start_monitoring()

    def _start_monitoring(self):
        """启动连接池监控"""

        def monitor_loop():
            while True:
                try:
                    self._update_stats()
                    time.sleep(30)  # 每30秒更新一次统计
                except Exception as e:
                    logger.error(f"数据库连接池监控失败: {e}")
                    time.sleep(10)

        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()

    def _update_stats(self):
        """更新连接池统计信息"""
        try:
            pool_status = self.sync_engine.pool.status()

            self.stats.total_connections = pool_status.pool_size + pool_status.overflow
            self.stats.active_connections = pool_status.checked_out
            self.stats.idle_connections = (
                pool_status.pool_size - pool_status.checked_out
            )

            # 更新峰值连接数
            self.stats.peak_connections = max(
                self.stats.peak_connections, self.stats.active_connections
            )

            # 计算池利用率
            if self.config.max_connections > 0:
                self.stats.pool_utilization = (
                    self.stats.active_connections / self.config.max_connections
                )

        except Exception as e:
            logger.error(f"更新数据库连接池统计失败: {e}")

    @asynccontextmanager
    async def get_session(self):
        """获取数据库会话"""
        if self.async_engine:
            # 使用异步会话
            async with self.AsyncSessionLocal() as session:
                try:
                    yield session
                    await session.commit()
                except Exception:
                    await session.rollback()
                    raise
        else:
            # 使用同步会话（在线程池中执行）
            session = self.SessionLocal()
            try:
                yield session
                session.commit()
            except Exception:
                session.rollback()
                raise
            finally:
                session.close()

    def get_sync_session(self):
        """获取同步数据库会话"""
        return self.SessionLocal()

    def get_pool_info(self) -> Dict[str, Any]:
        """获取连接池信息"""
        try:
            pool_status = self.sync_engine.pool.status()

            return {
                "pool_size": pool_status.pool_size,
                "checked_out": pool_status.checked_out,
                "overflow": pool_status.overflow,
                "checked_in": pool_status.checked_in,
                "total_connections": pool_status.pool_size + pool_status.overflow,
                "utilization": pool_status.checked_out / self.config.max_connections
                if self.config.max_connections > 0
                else 0,
                "stats": self.stats,
            }

        except Exception as e:
            logger.error(f"获取数据库连接池信息失败: {e}")
            return {"error": str(e)}

    async def close(self):
        """关闭连接池"""
        try:
            if self.async_engine:
                await self.async_engine.dispose()

            self.sync_engine.dispose()
            logger.info("数据库连接池已关闭")

        except Exception as e:
            logger.error(f"关闭数据库连接池失败: {e}")


class ConnectionPoolManager:
    """连接池管理器"""

    def __init__(self):
        self.http_pools: Dict[str, HTTPConnectionPool] = {}
        self.db_pools: Dict[str, DatabaseConnectionPool] = {}
        self._lock = asyncio.Lock()

    async def create_http_pool(
        self, name: str, config: Optional[PoolConfig] = None
    ) -> HTTPConnectionPool:
        """创建HTTP连接池"""
        async with self._lock:
            if name in self.http_pools:
                return self.http_pools[name]

            config = config or PoolConfig()
            pool = HTTPConnectionPool(config)
            self.http_pools[name] = pool

            logger.info(f"创建HTTP连接池: {name}")
            return pool

    def create_db_pool(
        self, name: str, database_url: str, config: Optional[PoolConfig] = None
    ) -> DatabaseConnectionPool:
        """创建数据库连接池"""
        if name in self.db_pools:
            return self.db_pools[name]

        config = config or PoolConfig()
        pool = DatabaseConnectionPool(database_url, config)
        self.db_pools[name] = pool

        logger.info(f"创建数据库连接池: {name}")
        return pool

    async def get_http_pool(self, name: str) -> Optional[HTTPConnectionPool]:
        """获取HTTP连接池"""
        return self.http_pools.get(name)

    def get_db_pool(self, name: str) -> Optional[DatabaseConnectionPool]:
        """获取数据库连接池"""
        return self.db_pools.get(name)

    async def get_all_stats(self) -> Dict[str, Any]:
        """获取所有连接池统计信息"""
        stats = {
            "http_pools": {},
            "db_pools": {},
            "system_info": {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "network_connections": len(psutil.net_connections()),
            },
        }

        # HTTP连接池统计
        for name, pool in self.http_pools.items():
            stats["http_pools"][name] = await pool.get_pool_stats()

        # 数据库连接池统计
        for name, pool in self.db_pools.items():
            stats["db_pools"][name] = pool.get_pool_info()

        return stats

    async def close_all(self):
        """关闭所有连接池"""
        # 关闭HTTP连接池
        for name, pool in self.http_pools.items():
            try:
                await pool.close_all()
                logger.info(f"关闭HTTP连接池: {name}")
            except Exception as e:
                logger.error(f"关闭HTTP连接池失败 {name}: {e}")

        # 关闭数据库连接池
        for name, pool in self.db_pools.items():
            try:
                await pool.close()
                logger.info(f"关闭数据库连接池: {name}")
            except Exception as e:
                logger.error(f"关闭数据库连接池失败 {name}: {e}")

        self.http_pools.clear()
        self.db_pools.clear()


# 全局连接池管理器实例
connection_pool_manager = ConnectionPoolManager()
