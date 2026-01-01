"""
依赖注入容器
管理所有服务组件的生命周期和依赖关系
"""

from typing import Optional
import asyncio
from contextlib import asynccontextmanager

from app.services.data import DataService as StockDataService
from app.services.prediction import TechnicalIndicatorCalculator
from app.services.data import ParquetManager
from app.core.config import settings


class ServiceContainer:
    """服务容器，管理所有服务组件"""
    
    def __init__(self):
        self._data_service: Optional[StockDataService] = None
        self._indicators_service: Optional[TechnicalIndicatorCalculator] = None
        self._parquet_manager: Optional[ParquetManager] = None
        self._data_sync_engine: Optional['DataSyncEngine'] = None
        self._monitoring_service: Optional['DataMonitoringService'] = None
        self._initialized = False
    
    async def initialize(self):
        """初始化所有服务"""
        if self._initialized:
            return
        
        # 初始化基础服务
        self._data_service = StockDataService()
        self._indicators_service = TechnicalIndicatorCalculator()
        self._parquet_manager = ParquetManager(settings.PARQUET_DATA_PATH)
        
        # 延迟导入避免循环依赖
        from app.services.data import DataSyncEngine
        from app.services.infrastructure import DataMonitoringService
        
        # 初始化复合服务
        self._data_sync_engine = DataSyncEngine(
            data_service=self._data_service,
            parquet_manager=self._parquet_manager
        )
        
        self._monitoring_service = DataMonitoringService(
            data_service=self._data_service,
            indicators_service=self._indicators_service,
            parquet_manager=self._parquet_manager,
            sync_engine=self._data_sync_engine
        )
        
        self._initialized = True
    
    async def cleanup(self):
        """清理所有服务"""
        if self._data_service:
            await self._data_service.__aexit__(None, None, None)
        
        if self._data_sync_engine:
            await self._data_sync_engine.cleanup()
        
        self._initialized = False
    
    @property
    def data_service(self) -> StockDataService:
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
    
    @property
    def parquet_manager(self) -> ParquetManager:
        """获取Parquet管理器"""
        if not self._initialized:
            raise RuntimeError("服务容器未初始化")
        return self._parquet_manager
    
    @property
    def data_sync_engine(self) -> 'DataSyncEngine':
        """获取数据同步引擎"""
        if not self._initialized:
            raise RuntimeError("服务容器未初始化")
        return self._data_sync_engine
    
    @property
    def monitoring_service(self) -> 'DataMonitoringService':
        """获取监控服务"""
        if not self._initialized:
            raise RuntimeError("服务容器未初始化")
        return self._monitoring_service


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
async def get_data_service() -> StockDataService:
    """获取数据服务依赖"""
    container = await get_container()
    return container.data_service


async def get_indicators_service() -> TechnicalIndicatorCalculator:
    """获取技术指标服务依赖"""
    container = await get_container()
    return container.indicators_service


async def get_parquet_manager() -> ParquetManager:
    """获取Parquet管理器依赖"""
    container = await get_container()
    return container.parquet_manager


async def get_data_sync_engine() -> 'DataSyncEngine':
    """获取数据同步引擎依赖"""
    container = await get_container()
    return container.data_sync_engine


async def get_monitoring_service() -> 'DataMonitoringService':
    """获取监控服务依赖"""
    container = await get_container()
    return container.monitoring_service