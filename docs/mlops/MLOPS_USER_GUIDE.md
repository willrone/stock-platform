# MLOps功能使用指南

## 概述

本指南介绍如何使用股票预测平台的MLOps功能，包括特征工程、模型训练、监控告警、A/B测试等核心功能。

## 目录

1. [快速开始](#快速开始)
2. [特征工程](#特征工程)
3. [模型训练与管理](#模型训练与管理)
4. [监控与告警](#监控与告警)
5. [A/B测试](#ab测试)
6. [数据版本控制](#数据版本控制)
7. [模型解释性](#模型解释性)
8. [系统管理](#系统管理)
9. [故障排除](#故障排除)

## 快速开始

### 1. 系统部署

使用部署脚本快速部署MLOps系统：

```bash
# 完整部署
./scripts/deploy_mlops.sh

# 跳过测试的快速部署
./scripts/deploy_mlops.sh --skip-tests

# 创建系统服务
./scripts/deploy_mlops.sh --create-service
```

### 2. 检查系统状态

```bash
# 快速状态检查
./scripts/status_mlops.sh quick

# 完整状态检查
./scripts/status_mlops.sh full
```

### 3. 访问系统

- **后端API**: http://localhost:8000
- **API文档**: http://localhost:8000/docs
- **前端界面**: http://localhost:3000 (如果已部署)

## 特征工程

### 技术指标计算

系统支持多种技术指标的自动计算：

#### 支持的指标

- **RSI (相对强弱指数)**: 衡量价格变动的速度和幅度
- **MACD (移动平均收敛散度)**: 趋势跟踪指标
- **布林带**: 价格波动范围指标
- **移动平均线 (SMA/EMA)**: 趋势平滑指标

#### API使用示例

```python
# 计算技术指标
import requests

# 配置技术指标
indicator_config = {
    "indicators": [
        {
            "name": "RSI",
            "period": 14,
            "enabled": True
        },
        {
            "name": "MACD",
            "fast_period": 12,
            "slow_period": 26,
            "signal_period": 9,
            "enabled": True
        }
    ],
    "stock_codes": ["000001.SZ", "000002.SZ"],
    "start_date": "2023-01-01",
    "end_date": "2023-12-31"
}

response = requests.post(
    "http://localhost:8000/api/v1/features/compute",
    json=indicator_config
)

if response.status_code == 200:
    features = response.json()["data"]
    print(f"计算完成，生成 {len(features)} 个特征")
```

### 特征存储管理

#### 查询特征

```python
# 查询已计算的特征
response = requests.get(
    "http://localhost:8000/api/v1/features/list",
    params={
        "stock_code": "000001.SZ",
        "feature_type": "technical_indicator",
        "limit": 100
    }
)

features = response.json()["data"]["features"]
```

#### 特征缓存

系统自动缓存计算结果，提升查询性能：

- **缓存时间**: 1小时 (可配置)
- **缓存策略**: LRU (最近最少使用)
- **缓存清理**: 自动清理过期数据

## 模型训练与管理

### 创建训练任务

#### 支持的模型类型

**传统机器学习模型**:
- LightGBM
- XGBoost
- 线性回归
- 随机森林

**深度学习模型**:
- MLP (多层感知机)
- LSTM (长短期记忆网络)
- Transformer
- Informer
- TimesNet
- PatchTST

#### 训练示例

```python
# 创建训练任务
training_config = {
    "model_name": "股票预测模型_v1",
    "model_type": "lightgbm",
    "stock_codes": ["000001.SZ", "000002.SZ"],
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "hyperparameters": {
        "learning_rate": 0.1,
        "max_depth": 6,
        "num_leaves": 31,
        "validation_split": 0.2
    }
}

response = requests.post(
    "http://localhost:8000/api/v1/models/train",
    json=training_config
)

if response.status_code == 200:
    model_info = response.json()["data"]
    model_id = model_info["model_id"]
    print(f"训练任务已创建: {model_id}")
```

### 训练进度监控

#### WebSocket实时监控

```javascript
// 前端WebSocket连接示例
const ws = new WebSocket(`ws://localhost:8000/api/v1/training/ws/${modelId}`);

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    
    switch(data.type) {
        case 'progress_update':
            updateProgressBar(data.progress_percentage);
            updateMetrics(data);
            break;
        case 'training_finished':
            showTrainingComplete(data);
            break;
        case 'error':
            showError(data.message);
            break;
    }
};
```

#### REST API查询

```python
# 查询训练进度
response = requests.get(f"http://localhost:8000/api/v1/training/tasks/{model_id}/progress")
progress = response.json()["data"]

print(f"训练进度: {progress['progress_percentage']}%")
print(f"当前阶段: {progress['training_stage']}")
print(f"当前轮次: {progress['current_epoch']}/{progress['total_epochs']}")
```

### 模型生命周期管理

#### 生命周期阶段

1. **development**: 开发阶段
2. **testing**: 测试阶段
3. **staging**: 预发布阶段
4. **production**: 生产阶段
5. **deprecated**: 已弃用
6. **archived**: 已归档

#### 阶段转换

```python
# 将模型转换到生产阶段
response = requests.post(
    f"http://localhost:8000/api/v1/models/{model_id}/lifecycle/transition",
    json={
        "new_stage": "production",
        "notes": "模型验证通过，部署到生产环境"
    }
)
```

### 模型血缘追踪

```python
# 查询模型血缘信息
response = requests.get(f"http://localhost:8000/api/v1/models/{model_id}/lineage")
lineage = response.json()["data"]

print("数据依赖:", lineage["data_dependencies"])
print("特征依赖:", lineage["feature_dependencies"])
print("配置依赖:", lineage["config_dependencies"])
```

## 监控与告警

### 性能监控

#### 查询监控指标

```python
# 获取模型性能指标
response = requests.get(
    f"http://localhost:8000/api/v1/monitoring/metrics/{model_id}",
    params={
        "time_range": "7d",
        "include_predictions": True
    }
)

metrics = response.json()["data"]
print(f"平均准确率: {metrics['performance_summary']['average_accuracy']}")
print(f"平均延迟: {metrics['performance_summary']['average_latency']}ms")
```

#### 监控仪表板

```python
# 获取监控仪表板数据
response = requests.get("http://localhost:8000/api/v1/monitoring/dashboard")
dashboard = response.json()["data"]

print(f"活跃模型数: {dashboard['performance_overview']['active_models']}")
print(f"活跃告警数: {dashboard['active_alerts']['count']}")
```

### 告警配置

#### 创建告警规则

```python
# 创建性能告警
alert_config = {
    "alert_type": "performance",
    "metric_name": "accuracy",
    "threshold": 0.8,
    "comparison": "lt",  # 小于阈值时告警
    "enabled": True,
    "notification_channels": ["email", "websocket"],
    "description": "模型准确率低于80%时告警"
}

response = requests.post(
    "http://localhost:8000/api/v1/monitoring/alerts",
    json=alert_config
)
```

#### 查询告警历史

```python
# 查询告警历史
response = requests.get(
    "http://localhost:8000/api/v1/monitoring/alerts/history",
    params={
        "time_range": "7d",
        "severity": "high"
    }
)

alerts = response.json()["data"]["alert_history"]
```

### 数据漂移检测

系统自动检测输入数据的分布变化：

- **检测方法**: KS检验、卡方检验
- **检测频率**: 每小时 (可配置)
- **漂移阈值**: 0.1 (可配置)

## A/B测试

### 创建A/B测试

```python
# 创建A/B测试
ab_test_config = {
    "test_name": "模型A vs 模型B",
    "model_a_id": "model_a_id",
    "model_b_id": "model_b_id",
    "traffic_split": 0.5,  # 50/50分流
    "duration_days": 7,
    "success_metrics": ["accuracy", "precision", "recall"]
}

response = requests.post(
    "http://localhost:8000/api/v1/ab-testing/tests",
    json=ab_test_config
)
```

### 查询测试结果

```python
# 查询A/B测试结果
response = requests.get(f"http://localhost:8000/api/v1/ab-testing/tests/{test_id}/results")
results = response.json()["data"]

print(f"统计显著性: {results['statistical_significance']}")
print(f"置信度: {results['confidence_level']}")
```

## 数据版本控制

### 数据版本管理

```python
# 创建数据版本
version_config = {
    "dataset_name": "stock_data_2024",
    "data_path": "data/stocks/2024",
    "description": "2024年股票数据",
    "tags": ["stocks", "2024", "training"]
}

response = requests.post(
    "http://localhost:8000/api/v1/data-versioning/versions",
    json=version_config
)
```

### 数据血缘追踪

```python
# 查询数据血缘
response = requests.get(f"http://localhost:8000/api/v1/data-versioning/lineage/{version_id}")
lineage = response.json()["data"]

print("上游数据源:", lineage["upstream_sources"])
print("下游使用者:", lineage["downstream_consumers"])
```

## 模型解释性

### SHAP解释

```python
# 获取模型SHAP解释
response = requests.post(
    f"http://localhost:8000/api/v1/explainability/shap/{model_id}",
    json={
        "sample_data": sample_features,
        "explanation_type": "local"  # local 或 global
    }
)

explanation = response.json()["data"]
print("特征重要性:", explanation["feature_importance"])
```

### 技术指标影响分析

```python
# 分析技术指标影响
response = requests.get(
    f"http://localhost:8000/api/v1/explainability/technical-analysis/{model_id}",
    params={"stock_code": "000001.SZ"}
)

analysis = response.json()["data"]
print("指标重要性排序:", analysis["indicator_importance"])
```

## 系统管理

### 性能优化

#### 查看性能报告

```python
# 获取性能报告
response = requests.get("http://localhost:8000/api/v1/system/performance/report")
report = response.json()["data"]

print("CPU使用率:", report["system_resources"]["cpu"]["percent"])
print("内存使用率:", report["system_resources"]["memory"]["percent"])
print("优化建议:", report["optimization_suggestions"])
```

#### 内存清理

```python
# 执行内存清理
response = requests.post("http://localhost:8000/api/v1/system/performance/cleanup")
cleanup_info = response.json()["data"]

print(f"释放内存: {cleanup_info['memory_freed']} 字节")
```

### 错误处理

#### 查看错误统计

```python
# 获取错误统计
response = requests.get("http://localhost:8000/api/v1/system/errors/statistics")
stats = response.json()["data"]

print("错误总数:", stats["total_errors"])
print("按类别统计:", stats["category_statistics"])
print("熔断器状态:", stats["circuit_breaker_status"])
```

## 故障排除

### 常见问题

#### 1. 服务无法启动

**问题**: 后端服务启动失败

**解决方案**:
```bash
# 检查端口占用
netstat -tuln | grep 8000

# 检查日志
tail -f logs/backend.log

# 重启服务
pkill -f uvicorn
./scripts/deploy_mlops.sh
```

#### 2. 模型训练失败

**问题**: 模型训练过程中出错

**解决方案**:
```bash
# 检查训练日志
tail -f backend/logs/mlops.log

# 检查数据完整性
python -c "
from app.services.data.data_manager import data_manager
data_manager.validate_data('000001.SZ', '2023-01-01', '2023-12-31')
"

# 重新训练
curl -X POST http://localhost:8000/api/v1/models/train \
  -H "Content-Type: application/json" \
  -d '{"model_name": "test", "model_type": "lightgbm", ...}'
```

#### 3. 特征计算异常

**问题**: 技术指标计算失败

**解决方案**:
```python
# 检查特征管道状态
from app.services.features.feature_pipeline import feature_pipeline
status = feature_pipeline.get_status()
print("管道状态:", status)

# 重新计算特征
feature_pipeline.compute_features(
    stock_codes=["000001.SZ"],
    indicators=["RSI", "MACD"],
    start_date="2023-01-01",
    end_date="2023-12-31"
)
```

#### 4. 监控告警异常

**问题**: 告警系统不工作

**解决方案**:
```python
# 测试告警配置
import requests

response = requests.post(
    "http://localhost:8000/api/v1/monitoring/test-alert",
    params={
        "alert_type": "performance",
        "metric_name": "accuracy",
        "test_value": 0.5
    }
)

print("告警测试结果:", response.json())
```

### 日志分析

#### 重要日志文件

- **后端服务日志**: `logs/backend.log`
- **MLOps系统日志**: `backend/logs/mlops.log`
- **前端服务日志**: `logs/frontend.log`

#### 日志级别

- **INFO**: 一般信息
- **WARNING**: 警告信息
- **ERROR**: 错误信息
- **CRITICAL**: 严重错误

#### 日志查看命令

```bash
# 实时查看日志
tail -f backend/logs/mlops.log

# 查看错误日志
grep "ERROR" backend/logs/mlops.log

# 查看最近100行日志
tail -n 100 backend/logs/mlops.log
```

### 性能调优

#### 数据库优化

```sql
-- 创建索引提升查询性能
CREATE INDEX idx_model_info_status ON model_info(status);
CREATE INDEX idx_model_info_created_at ON model_info(created_at);
```

#### 缓存优化

```python
# 调整缓存配置
cache_config = {
    "feature_cache_ttl": 3600,  # 1小时
    "model_cache_size": 100,    # 缓存100个模型
    "query_cache_enabled": True
}
```

## 配置参考

### 环境变量

```bash
# MLOps功能开关
MLOPS_ENABLED=true
QLIB_CACHE_DIR=backend/data/qlib_cache
FEATURE_CACHE_ENABLED=true
MONITORING_ENABLED=true
AB_TESTING_ENABLED=true

# 性能配置
PARALLEL_WORKERS=4
BATCH_SIZE=1000
CACHE_SIZE=10000

# 监控配置
MONITORING_INTERVAL=60
ALERT_THRESHOLD_ACCURACY=0.8
DRIFT_DETECTION_THRESHOLD=0.1
```

### 配置文件

主要配置文件位于 `backend/config/mlops_config.yaml`，包含：

- 特征工程配置
- Qlib集成配置
- 训练配置
- 监控配置
- A/B测试配置
- 系统配置

## 最佳实践

### 1. 模型开发流程

1. **数据准备**: 确保数据质量和完整性
2. **特征工程**: 计算和验证技术指标
3. **模型训练**: 使用合适的模型类型和参数
4. **模型验证**: 在测试集上验证模型性能
5. **生产部署**: 通过生命周期管理部署到生产环境
6. **监控维护**: 持续监控模型性能和数据漂移

### 2. 监控策略

1. **设置合理的告警阈值**: 避免过多误报
2. **多维度监控**: 准确率、延迟、错误率等
3. **定期检查**: 每日查看监控仪表板
4. **及时响应**: 收到告警后及时处理

### 3. 性能优化

1. **合理使用缓存**: 平衡内存使用和查询性能
2. **批量处理**: 大量数据处理时使用批量操作
3. **异步处理**: 长时间任务使用异步处理
4. **资源监控**: 定期检查系统资源使用情况

## 支持与反馈

如果在使用过程中遇到问题，请：

1. 查看本指南的故障排除部分
2. 检查系统日志文件
3. 使用状态检查脚本诊断问题
4. 联系技术支持团队

---

*本指南会持续更新，请关注最新版本。*