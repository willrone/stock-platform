# 数据管理模块

该模块负责处理所有与数据相关的操作，包括数据获取、存储、同步、验证和生命周期管理。

## 主要组件

### 数据服务
- **DataService**: 主要数据服务接口，提供股票数据的获取和管理功能
- **SimpleDataService**: 简化版数据服务，适用于轻量级数据操作

### 数据同步
- **DataSyncEngine**: 数据同步引擎，负责数据的同步和更新
- **SyncProgressTracker**: 同步进度跟踪器

### 数据验证
- **DataValidator**: 数据验证服务，确保数据质量和完整性
- **ValidationLevel**: 验证级别枚举
- **ValidationRule**: 验证规则定义

### 数据生命周期管理
- **DataLifecycleManager**: 数据生命周期管理器
- **RetentionPolicy**: 数据保留策略

### 文件管理
- **ParquetManager**: Parquet 文件管理器，处理 Parquet 格式的数据文件
- **StreamProcessor**: 流数据处理器，处理实时数据流

## 使用示例

```python
# 导入数据服务
from app.services.data import DataService, ParquetManager

# 创建数据服务实例
data_service = DataService()

# 获取股票数据
stock_data = await data_service.get_stock_data("000001.SZ", "2024-01-01", "2024-12-31")

# 使用 Parquet 管理器
parquet_manager = ParquetManager()
parquet_manager.save_data("000001.SZ", stock_data)
```

## 配置

数据模块支持以下配置选项：

- 数据源配置
- 缓存策略
- 同步频率
- 验证规则

## 依赖关系

该模块依赖于：
- 基础设施模块（缓存、连接池）
- 核心配置模块
- 数据库模型

## 注意事项

1. 数据服务支持多种数据源
2. 建议使用连接池来优化数据库连接
3. 数据验证是可选的，但建议在生产环境中启用
4. Parquet 文件管理器支持分区存储