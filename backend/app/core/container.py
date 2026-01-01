"""
依赖注入容器
管理所有服务组件的生命周期和依赖关系
"""

from typing import Optional
import asyncio
from contextlib import asynccontextmanager

from app.services.data import SimpleDataService
from app.services.prediction import TechnicalIndicatorCalculator


class ServiceContainer:
    """服务容器，管理所有服务组件"""
    
    def __init__(self):
        self._data_service: Optional[SimpleDataService] = None
        self._indicators_service: Optional[TechnicalIndicatorCalculator] = None
        self._initialized = False
    
    async def initialize(self):
        """初始化所有服务"""
        if self._initialized:
            return
        
        # 初始化基础服务 - 使用简化的数据服务
        self._data_service = SimpleDataService()
        self._indicators_service = TechnicalIndicatorCalculator()
        
        self._initialized = True
    
    async def cleanup(self):
        """清理所有服务"""
        if self._data_service:
            await self._data_service.__aexit__(None, None, None)
        
        self._initialized = False
    
    @property
    def data_service(self) -> SimpleDataService:
        """获取数据服务"""
        if not self._initialized:
            raise RuntimeError("服务容器未初始化")
        return self._data_service
    
    @property
    def indicators_service(self) -> TechnicalIndicatorCalculator:
        """获取技术指标服务"""
        if not self._initialized:
            raise RuntimeError("服务容器未初始化")
        return self._indicators_service


# 全局服务容器实例
_container: Optional[ServiceContainer] = None


async def get_container() -> ServiceContainer:
    """获取服务容器实例"""
    global _container
    if _container is None:
        _container = ServiceContainer()
        await _container.initialize()
    return _container


async def cleanup_container():
    """清理服务容器"""
    global _container
    if _container:
        await _container.cleanup()
        _container = None


@asynccontextmanager
async def container_lifespan():
    """服务容器生命周期管理器"""
    try:
        container = await get_container()
        yield container
    finally:
        await cleanup_container()


# 依赖注入函数，用于FastAPI的Depends
async def get_data_service() -> SimpleDataService:
    """获取数据服务依赖"""
    container = await get_container()
    return container.data_service


async def get_indicators_service() -> TechnicalIndicatorCalculator:
    """获取技术指标服务依赖"""
    container = await get_container()
    return container.indicators_service