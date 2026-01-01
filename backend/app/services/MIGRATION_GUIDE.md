# 服务重构迁移指南

本指南帮助开发者从旧的服务导入方式迁移到新的模块化导入方式。

## 概述

服务重构将原来的 28 个服务文件按功能重新组织到 6 个模块中：

- **data**: 数据管理模块
- **models**: 模型管理模块  
- **prediction**: 预测引擎模块
- **backtest**: 回测引擎模块
- **tasks**: 任务管理模块
- **infrastructure**: 基础设施模块

## 向后兼容性

**重要**: 现有的导入语句仍然有效，但会显示弃用警告。建议尽快迁移到新的导入方式。

```python
# 旧的导入方式（仍然有效，但会显示警告）
from app.services import DataService

# 新的推荐导入方式
from app.services.data import DataService
```

## 迁移映射表

### 数据管理模块

| 旧导入 | 新导入 |
|--------|--------|
| `from app.services.data_service import StockDataService` | `from app.services.data import DataService` |
| `from app.services.data_service_simple import SimpleStockDataService` | `from app.services.data import SimpleDataService` |
| `from app.services.data_sync_engine import DataSyncEngine` | `from app.services.data import DataSyncEngine` |
| `from app.services.data_validator import DataValidator` | `from app.services.data import DataValidator` |
| `from app.services.data_lifecycle import DataLifecycleManager` | `from app.services.data import DataLifecycleManager` |
| `from app.services.parquet_manager import ParquetManager` | `from app.services.data import ParquetManager` |
| `from app.services.stream_processor import StreamProcessor` | `from app.services.data import StreamProcessor` |

### 模型管理模块

| 旧导入 | 新导入 |
|--------|--------|
| `from app.services.model_training_service import ModelTrainingService` | `from app.services.models import ModelTrainingService` |
| `from app.services.model_training import ModelTrainingService` | `from app.services.models import DeepModelTrainingService` |
| `from app.services.model_storage import ModelStorage` | `from app.services.models import ModelStorage` |
| `from app.services.model_deployment_service import ModelDeploymentService` | `from app.services.models import ModelDeploymentService` |
| `from app.services.model_evaluation import ModelEvaluator` | `from app.services.models import ModelEvaluator` |
| `from app.services.advanced_training import AdvancedTrainingService` | `from app.services.models import AdvancedTrainingService` |
| `from app.services.modern_models import TimesNet` | `from app.services.models import TimesNet` |

### 预测引擎模块

| 旧导入 | 新导入 |
|--------|--------|
| `from app.services.prediction_engine import PredictionEngine` | `from app.services.prediction import PredictionEngine` |
| `from app.services.prediction_fallback import PredictionFallbackEngine` | `from app.services.prediction import PredictionFallbackEngine` |
| `from app.services.risk_assessment import RiskAssessmentService` | `from app.services.prediction import RiskAssessmentService` |
| `from app.services.feature_extractor import FeatureExtractor` | `from app.services.prediction import FeatureExtractor` |
| `from app.services.technical_indicators import TechnicalIndicatorCalculator` | `from app.services.prediction import TechnicalIndicatorCalculator` |

### 回测引擎模块

| 旧导入 | 新导入 |
|--------|--------|
| `from app.services.backtest_engine import BacktestEngine` | `from app.services.backtest import BacktestEngine` |
| `from app.services.backtest_executor import BacktestExecutor` | `from app.services.backtest import BacktestExecutor` |

### 任务管理模块

| 旧导入 | 新导入 |
|--------|--------|
| `from app.services.task_manager import TaskManager` | `from app.services.tasks import TaskManager` |
| `from app.services.task_queue import TaskQueueManager` | `from app.services.tasks import TaskQueueManager` |
| `from app.services.task_execution_engine import TaskExecutionEngine` | `from app.services.tasks import TaskExecutionEngine` |
| `from app.services.task_notification_service import TaskNotificationService` | `from app.services.tasks import TaskNotificationService` |

### 基础设施模块

| 旧导入 | 新导入 |
|--------|--------|
| `from app.services.cache_service import CacheManager` | `from app.services.infrastructure import CacheManager` |
| `from app.services.connection_pool import ConnectionPoolManager` | `from app.services.infrastructure import ConnectionPoolManager` |
| `from app.services.monitoring_service import DataMonitoringService` | `from app.services.infrastructure import DataMonitoringService` |
| `from app.services.enhanced_logger import EnhancedLogger` | `from app.services.infrastructure import EnhancedLogger` |
| `from app.services.metrics_collector import MetricsCollector` | `from app.services.infrastructure import MetricsCollector` |
| `from app.services.websocket_manager import WebSocketManager` | `from app.services.infrastructure import WebSocketManager` |

## 批量迁移脚本

为了帮助快速迁移，我们提供了一个 Python 脚本来自动更新导入语句：

```python
#!/usr/bin/env python3
"""
服务导入迁移脚本
自动将旧的服务导入更新为新的模块化导入
"""

import os
import re
from pathlib import Path

# 迁移映射
MIGRATION_MAP = {
    # 数据管理模块
    r'from app\.services\.data_service import (.+)': r'from app.services.data import \1',
    r'from app\.services\.data_service_simple import (.+)': r'from app.services.data import \1',
    r'from app\.services\.data_sync_engine import (.+)': r'from app.services.data import \1',
    r'from app\.services\.data_validator import (.+)': r'from app.services.data import \1',
    r'from app\.services\.data_lifecycle import (.+)': r'from app.services.data import \1',
    r'from app\.services\.parquet_manager import (.+)': r'from app.services.data import \1',
    r'from app\.services\.stream_processor import (.+)': r'from app.services.data import \1',
    
    # 模型管理模块
    r'from app\.services\.model_training_service import (.+)': r'from app.services.models import \1',
    r'from app\.services\.model_training import (.+)': r'from app.services.models import \1',
    r'from app\.services\.model_storage import (.+)': r'from app.services.models import \1',
    r'from app\.services\.model_deployment_service import (.+)': r'from app.services.models import \1',
    r'from app\.services\.model_evaluation import (.+)': r'from app.services.models import \1',
    r'from app\.services\.advanced_training import (.+)': r'from app.services.models import \1',
    r'from app\.services\.modern_models import (.+)': r'from app.services.models import \1',
    
    # 预测引擎模块
    r'from app\.services\.prediction_engine import (.+)': r'from app.services.prediction import \1',
    r'from app\.services\.prediction_fallback import (.+)': r'from app.services.prediction import \1',
    r'from app\.services\.risk_assessment import (.+)': r'from app.services.prediction import \1',
    r'from app\.services\.feature_extractor import (.+)': r'from app.services.prediction import \1',
    r'from app\.services\.technical_indicators import (.+)': r'from app.services.prediction import \1',
    
    # 回测引擎模块
    r'from app\.services\.backtest_engine import (.+)': r'from app.services.backtest import \1',
    r'from app\.services\.backtest_executor import (.+)': r'from app.services.backtest import \1',
    
    # 任务管理模块
    r'from app\.services\.task_manager import (.+)': r'from app.services.tasks import \1',
    r'from app\.services\.task_queue import (.+)': r'from app.services.tasks import \1',
    r'from app\.services\.task_execution_engine import (.+)': r'from app.services.tasks import \1',
    r'from app\.services\.task_notification_service import (.+)': r'from app.services.tasks import \1',
    
    # 基础设施模块
    r'from app\.services\.cache_service import (.+)': r'from app.services.infrastructure import \1',
    r'from app\.services\.connection_pool import (.+)': r'from app.services.infrastructure import \1',
    r'from app\.services\.monitoring_service import (.+)': r'from app.services.infrastructure import \1',
    r'from app\.services\.enhanced_logger import (.+)': r'from app.services.infrastructure import \1',
    r'from app\.services\.metrics_collector import (.+)': r'from app.services.infrastructure import \1',
    r'from app\.services\.websocket_manager import (.+)': r'from app.services.infrastructure import \1',
}

def migrate_file(file_path):
    """迁移单个文件的导入语句"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    for old_pattern, new_pattern in MIGRATION_MAP.items():
        content = re.sub(old_pattern, new_pattern, content)
    
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"已更新: {file_path}")
        return True
    
    return False

def migrate_directory(directory):
    """迁移目录下所有 Python 文件"""
    updated_files = 0
    
    for file_path in Path(directory).rglob("*.py"):
        if migrate_file(file_path):
            updated_files += 1
    
    print(f"总共更新了 {updated_files} 个文件")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("使用方法: python migrate_imports.py <目录路径>")
        sys.exit(1)
    
    directory = sys.argv[1]
    if not os.path.exists(directory):
        print(f"目录不存在: {directory}")
        sys.exit(1)
    
    migrate_directory(directory)
```

## 使用迁移脚本

1. 将上述脚本保存为 `migrate_imports.py`
2. 运行脚本：
   ```bash
   python migrate_imports.py /path/to/your/project
   ```

## 逐步迁移建议

### 阶段 1: 了解新结构
1. 阅读各模块的 README.md 文档
2. 了解新的模块职责划分
3. 熟悉新的导入路径

### 阶段 2: 更新测试文件
1. 首先更新测试文件的导入语句
2. 运行测试确保功能正常
3. 修复任何导入相关的问题

### 阶段 3: 更新应用代码
1. 逐个模块更新应用代码
2. 优先更新使用频率高的模块
3. 保持向后兼容性直到迁移完成

### 阶段 4: 清理和优化
1. 移除所有弃用警告
2. 优化导入语句
3. 更新文档和注释

## 常见问题

### Q: 迁移后会影响现有功能吗？
A: 不会。所有服务的功能保持不变，只是导入路径发生了变化。

### Q: 可以混合使用新旧导入方式吗？
A: 可以，但不建议。建议统一使用新的导入方式。

### Q: 如何处理循环导入问题？
A: 新的模块结构已经考虑了依赖关系，避免了循环导入。如果遇到问题，请检查导入路径是否正确。

### Q: 性能会受到影响吗？
A: 不会。新的模块结构可能会略微提升导入性能，因为减少了不必要的模块加载。

## 获取帮助

如果在迁移过程中遇到问题，请：

1. 查看相关模块的 README.md 文档
2. 检查迁移映射表
3. 运行测试验证功能
4. 联系开发团队获取支持

## 迁移检查清单

- [ ] 阅读迁移指南
- [ ] 备份现有代码
- [ ] 更新测试文件导入
- [ ] 运行测试验证功能
- [ ] 更新应用代码导入
- [ ] 验证所有功能正常
- [ ] 移除弃用警告
- [ ] 更新相关文档