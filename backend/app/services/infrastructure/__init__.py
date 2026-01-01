"""
基础设施模块

该模块包含所有系统基础功能服务，包括：
- 缓存服务和缓存管理
- 连接池管理（HTTP和数据库）
- 监控服务和性能指标
- 增强日志记录和日志轮转
- 指标收集和统计
- WebSocket 连接管理

主要组件：
- CacheManager: 缓存管理器
- ConnectionPoolManager: 连接池管理器
- DataMonitoringService: 数据监控服务
- EnhancedLogger: 增强日志记录器
- WebSocketManager: WebSocket 管理器
"""

# 缓存服务
from .cache_service import (
    CacheManager,
    LRUCache,
    CachePolicy,
    CacheEntry,
    CacheStats,
    cached  # 装饰器函数
)

# 连接池
from .connection_pool import (
    ConnectionPoolManager,
    HTTPConnectionPool,
    DatabaseConnectionPool,
    ConnectionStatus,
    ConnectionStats,
    PoolConfig
)

# 监控服务
from .monitoring_service import (
    DataMonitoringService,
    ServiceHealthStatus,
    PerformanceMetrics,
    ErrorStatistics
)

# 增强日志
from .enhanced_logger import (
    EnhancedLogger,
    LogRotationManager,
    StructuredFormatter,
    LogLevel,
    LogCategory,
    StructuredLogEntry,
    get_logger,
    get_api_logger,
    get_data_logger,
    get_monitoring_logger
)

# WebSocket 管理
from .websocket_manager import (
    WebSocketManager,
    WebSocketMessage,
    ClientConnection
)

# 注意：metrics_collector.py 当前为空文件，暂不导入

__all__ = [
    # 缓存服务
    'CacheManager',
    'LRUCache',
    'CachePolicy',
    'CacheEntry',
    'CacheStats',
    'cached',
    
    # 连接池
    'ConnectionPoolManager',
    'HTTPConnectionPool',
    'DatabaseConnectionPool',
    'ConnectionStatus',
    'ConnectionStats',
    'PoolConfig',
    
    # 监控服务
    'DataMonitoringService',
    'ServiceHealthStatus',
    'PerformanceMetrics',
    'ErrorStatistics',
    
    # 增强日志
    'EnhancedLogger',
    'LogRotationManager',
    'StructuredFormatter',
    'LogLevel',
    'LogCategory',
    'StructuredLogEntry',
    'get_logger',
    'get_api_logger',
    'get_data_logger',
    'get_monitoring_logger',
    
    # WebSocket 管理
    'WebSocketManager',
    'WebSocketMessage',
    'ClientConnection'
]