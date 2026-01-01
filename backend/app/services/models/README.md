# 模型管理模块

该模块包含所有与机器学习模型相关的服务，处理模型的完整生命周期，包括训练、存储、部署和评估。

## 主要组件

### 模型训练
- **ModelTrainingService**: 模型训练服务接口，支持传统机器学习模型
- **DeepModelTrainingService**: 深度学习模型训练服务
- **ModelTrainer**: 模型训练器基类
- **RandomForestTrainer**: 随机森林训练器
- **LinearRegressionTrainer**: 线性回归训练器
- **TrainerFactory**: 训练器工厂

### 模型存储
- **ModelStorage**: 模型存储管理器，处理模型的持久化
- **ModelVersionManager**: 模型版本管理器
- **ModelMetadata**: 模型元数据定义

### 模型部署
- **ModelDeploymentService**: 模型部署服务
- **ModelPerformanceMonitor**: 模型性能监控器
- **DeploymentEvaluator**: 部署评估器

### 模型评估
- **ModelEvaluator**: 模型评估器
- **TimeSeriesValidator**: 时间序列交叉验证器
- **FinancialMetricsCalculator**: 金融指标计算器

### 高级训练
- **AdvancedTrainingService**: 高级训练服务
- **EnsembleModelManager**: 集成模型管理器
- **OnlineLearningManager**: 在线学习管理器

### 现代深度学习模型
- **TimesNet**: TimesNet 时间序列模型
- **PatchTST**: PatchTST 模型
- **Informer**: Informer 模型
- **LSTMModel**: LSTM 基线模型
- **TransformerModel**: Transformer 模型

## 使用示例

```python
# 导入模型服务
from app.services.models import ModelTrainingService, ModelStorage

# 创建训练服务
training_service = ModelTrainingService()

# 训练模型
config = TrainingConfig(
    model_type=ModelType.RANDOM_FOREST,
    hyperparameters={"n_estimators": 100}
)
result = await training_service.train_model(config, training_data)

# 存储模型
storage = ModelStorage()
model_id = await storage.save_model(result.model, result.metadata)
```

## 支持的模型类型

- **传统机器学习**: 随机森林、线性回归、XGBoost
- **深度学习**: LSTM、Transformer、TimesNet、PatchTST、Informer
- **集成学习**: 投票集成、堆叠集成、Bagging
- **在线学习**: 增量学习、自适应学习

## 配置

模型模块支持以下配置：

- 训练参数配置
- 模型存储路径
- 部署环境配置
- 评估指标设置

## 依赖关系

该模块依赖于：
- 数据模块（训练数据）
- 基础设施模块（缓存、监控）
- 任务模块（异步训练）

## 注意事项

1. 深度学习模型需要 GPU 支持
2. 模型版本管理支持自动版本控制
3. 集成学习可以显著提升模型性能
4. 在线学习适用于数据流场景