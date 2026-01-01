# 基础设施模块

该模块包含所有系统基础功能服务，为其他业务模块提供底层支持，包括缓存、连接池、监控、日志等核心功能。

## 主要组件

### 缓存服务
- **CacheManager**: 缓存管理器，统一管理多种缓存策略
- **LRUCache**: LRU（最近最少使用）缓存实现
- **CachePolicy**: 缓存策略枚举
- **CacheEntry**: 缓存条目
- **CacheStats**: 缓存统计信息
- **cached**: 缓存装饰器函数

### 连接池管理
- **ConnectionPoolManager**: 连接池管理器，统一管理各种连接池
- **HTTPConnectionPool**: HTTP 连接池管理器
- **DatabaseConnectionPool**: 数据库连接池管理器
- **ConnectionStatus**: 连接状态枚举
- **ConnectionStats**: 连接统计信息
- **PoolConfig**: 连接池配置

### 监控服务
- **DataMonitoringService**: 数据监控服务
- **ServiceHealthStatus**: 服务健康状态
- **PerformanceMetrics**: 性能指标
- **ErrorStatistics**: 错误统计

### 增强日志
- **EnhancedLogger**: 增强的日志记录器
- **LogRotationManager**: 日志轮转管理器
- **StructuredFormatter**: 结构化日志格式化器
- **LogLevel**: 日志级别枚举
- **LogCategory**: 日志分类枚举
- **StructuredLogEntry**: 结构化日志条目

### WebSocket 管理
- **WebSocketManager**: WebSocket 连接管理器
- **WebSocketMessage**: WebSocket 消息格式
- **ClientConnection**: 客户端连接信息

## 使用示例

### 缓存服务使用

```python
# 导入缓存服务
from app.services.infrastructure import CacheManager, cached

# 使用缓存管理器
cache_manager = CacheManager()
await cache_manager.set("key", "value", ttl=3600)
value = await cache_manager.get("key")

# 使用缓存装饰器
@cached(cache_name="api_cache", ttl=300)
async def expensive_api_call(param):
    # 耗时的 API 调用
    return result
```

### 连接池使用

```python
# 导入连接池
from app.services.infrastructure import ConnectionPoolManager

# 创建连接池管理器
pool_manager = ConnectionPoolManager()

# 获取 HTTP 连接
async with pool_manager.get_http_connection() as conn:
    response = await conn.get("https://api.example.com/data")

# 获取数据库连接
async with pool_manager.get_db_connection() as conn:
    result = await conn.execute("SELECT * FROM stocks")
```

### 增强日志使用

```python
# 导入日志服务
from app.services.infrastructure import get_logger, LogLevel, LogCategory

# 获取日志记录器
logger = get_logger("my_service")

# 记录结构化日志
await logger.log_structured(
    level=LogLevel.INFO,
    category=LogCategory.BUSINESS,
    message="用户操作",
    extra_data={"user_id": 123, "action": "login"}
)
```

### WebSocket 管理使用

```python
# 导入 WebSocket 管理器
from app.services.infrastructure import WebSocketManager, WebSocketMessage

# 创建 WebSocket 管理器
ws_manager = WebSocketManager()

# 发送消息给所有客户端
message = WebSocketMessage(
    type="notification",
    data={"title": "系统通知", "content": "任务已完成"}
)
await ws_manager.broadcast(message)
```

## 缓存策略

支持多种缓存策略：

- **LRU**: 最近最少使用，适用于内存有限的场景
- **TTL**: 基于时间的过期策略
- **LFU**: 最少使用频率（计划支持）
- **FIFO**: 先进先出（计划支持）

## 连接池配置

连接池支持以下配置选项：

```python
pool_config = PoolConfig(
    max_connections=100,        # 最大连接数
    min_connections=10,         # 最小连接数
    connection_timeout=30,      # 连接超时时间
    idle_timeout=300,          # 空闲超时时间
    max_retries=3              # 最大重试次数
)
```

## 监控指标

监控服务提供以下指标：

- **性能指标**: 响应时间、吞吐量、CPU/内存使用率
- **错误统计**: 错误率、错误类型分布
- **健康状态**: 服务可用性、依赖服务状态
- **业务指标**: 自定义业务相关指标

## 日志功能

增强日志提供以下功能：

- **结构化日志**: JSON 格式的结构化日志输出
- **日志轮转**: 自动日志文件轮转和压缩
- **多级别日志**: DEBUG、INFO、WARNING、ERROR、CRITICAL
- **日志分类**: 系统、业务、安全、性能等分类
- **异步日志**: 高性能异步日志写入

## 配置选项

基础设施模块支持以下配置：

- **缓存配置**: 缓存大小、过期策略、清理频率
- **连接池配置**: 连接数限制、超时设置、重试策略
- **监控配置**: 监控频率、告警阈值、指标收集
- **日志配置**: 日志级别、输出格式、轮转策略

## 性能优化

基础设施模块包含多项性能优化：

- **连接复用**: 减少连接建立开销
- **缓存预热**: 提前加载热点数据
- **异步处理**: 非阻塞 I/O 操作
- **批量操作**: 减少网络往返次数
- **资源池化**: 复用昂贵的资源对象

## 依赖关系

该模块是其他模块的基础依赖：

- 为数据模块提供缓存和连接池
- 为任务模块提供 WebSocket 通信
- 为所有模块提供日志和监控服务

## 注意事项

1. 缓存大小应根据可用内存合理设置
2. 连接池大小应根据并发需求调整
3. 日志级别在生产环境中建议设置为 INFO 或以上
4. 监控指标应定期清理以避免内存泄漏
5. WebSocket 连接应及时清理断开的连接