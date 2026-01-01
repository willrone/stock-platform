"""
数据管理模块

该模块包含所有与数据相关的服务，包括数据获取、存储、同步、验证和生命周期管理。

组件:
- StockDataService: 主要数据服务接口
- SimpleStockDataService: 简化版数据服务
- DataSyncEngine: 数据同步引擎
- DataValidator: 数据验证服务
- DataLifecycleManager: 数据生命周期管理
- ParquetManager: Parquet 文件管理
- StreamProcessor: 流数据处理
"""

from .data_service import StockDataService
from .data_service_simple import SimpleStockDataService
from .data_sync_engine import DataSyncEngine
from .data_validator import DataValidator
from .data_lifecycle import DataLifecycleManager
from .parquet_manager import ParquetManager
from .stream_processor import StreamProcessor

# 为了向后兼容，提供别名
DataService = StockDataService

__all__ = [
    'StockDataService',
    'DataService',  # 别名
    'SimpleStockDataService', 
    'DataSyncEngine',
    'DataValidator',
    'DataLifecycleManager',
    'ParquetManager',
    'StreamProcessor'
]