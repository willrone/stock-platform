"""
模型管理模块

该模块包含所有与机器学习模型相关的服务，包括：
- 模型训练和训练服务
- 模型存储和版本管理
- 模型部署和性能监控
- 模型评估和指标计算
- 高级训练功能（集成学习、在线学习）
- 现代深度学习模型实现

主要组件：
- ModelTrainingService: 模型训练服务接口
- ModelTrainer: 模型训练器基类和具体实现
- ModelStorage: 模型存储管理器
- ModelDeploymentService: 模型部署服务
- ModelEvaluator: 模型评估器
- AdvancedTrainingService: 高级训练服务
- 现代模型: TimesNet, PatchTST, Informer等深度学习模型
"""

# 模型训练服务
from .model_training_service import (
    ModelTrainingService,
    ModelTrainer,
    RandomForestTrainer,
    LinearRegressionTrainer,
    TrainerFactory,
    TrainingConfig,
    TrainingResult
)

# 模型训练（深度学习）
from .model_training import (
    ModelTrainingService as DeepModelTrainingService,
    QlibDataProvider,
    LSTMModel,
    TransformerModel,
    PositionalEncoding,
    ModelType,
    TrainingConfig as DeepTrainingConfig,
    ModelMetrics
)

# 模型存储
from .model_storage import (
    ModelStorage,
    ModelVersionManager,
    ModelMetadata,
    ModelStatus,
    ModelType as StorageModelType
)

# 模型部署服务
from .model_deployment_service import (
    ModelDeploymentService,
    ModelEvaluator as DeploymentEvaluator,
    ModelPerformanceMonitor,
    DeploymentStatus,
    EvaluationMetric,
    ModelEvaluation,
    DeploymentConfig,
    DeploymentRecord
)

# 模型评估
from .model_evaluation import (
    ModelEvaluator,
    ModelVersionManager as EvaluationVersionManager,
    TimeSeriesValidator,
    FinancialMetricsCalculator,
    BacktestMetrics,
    ModelVersion,
    ModelStatus as EvaluationModelStatus
)

# 高级训练
from .advanced_training import (
    AdvancedTrainingService,
    EnsembleModelManager,
    OnlineLearningManager,
    EnsembleMethod,
    EnsembleConfig,
    OnlineLearningConfig,
    ModelType as AdvancedModelType
)

# 现代模型
from .modern_models import (
    TimesNet,
    TimesBlock,
    Inception_Block_V1,
    PatchTST,
    Informer,
    InformerEncoderLayer,
    ProbAttention,
    PositionalEncoding as ModernPositionalEncoding
)

__all__ = [
    # 模型训练服务
    'ModelTrainingService',
    'ModelTrainer',
    'RandomForestTrainer',
    'LinearRegressionTrainer',
    'TrainerFactory',
    'TrainingConfig',
    'TrainingResult',
    
    # 深度学习训练
    'DeepModelTrainingService',
    'QlibDataProvider',
    'LSTMModel',
    'TransformerModel',
    'PositionalEncoding',
    'ModelType',
    'DeepTrainingConfig',
    'ModelMetrics',
    
    # 模型存储
    'ModelStorage',
    'ModelVersionManager',
    'ModelMetadata',
    'ModelStatus',
    'StorageModelType',
    
    # 模型部署
    'ModelDeploymentService',
    'DeploymentEvaluator',
    'ModelPerformanceMonitor',
    'DeploymentStatus',
    'EvaluationMetric',
    'ModelEvaluation',
    'DeploymentConfig',
    'DeploymentRecord',
    
    # 模型评估
    'ModelEvaluator',
    'EvaluationVersionManager',
    'TimeSeriesValidator',
    'FinancialMetricsCalculator',
    'BacktestMetrics',
    'ModelVersion',
    'EvaluationModelStatus',
    
    # 高级训练
    'AdvancedTrainingService',
    'EnsembleModelManager',
    'OnlineLearningManager',
    'EnsembleMethod',
    'EnsembleConfig',
    'OnlineLearningConfig',
    'AdvancedModelType',
    
    # 现代模型
    'TimesNet',
    'TimesBlock',
    'Inception_Block_V1',
    'PatchTST',
    'Informer',
    'InformerEncoderLayer',
    'ProbAttention',
    'ModernPositionalEncoding'
]