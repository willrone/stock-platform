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
    LinearRegressionTrainer,
    ModelTrainer,
    ModelTrainingService,
    RandomForestTrainer,
    TrainerFactory,
    TrainingConfig,
    TrainingResult,
)

# 模型训练（深度学习）- 可选导入
try:
    from .model_training import LSTMModel, ModelMetrics
    from .model_training import ModelTrainingService as DeepModelTrainingService
    from .model_training import ModelType, PositionalEncoding, QlibDataProvider
    from .model_training import TrainingConfig as DeepTrainingConfig
    from .model_training import TransformerModel

    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    DEEP_LEARNING_AVAILABLE = False
    DeepModelTrainingService = None
    QlibDataProvider = None
    LSTMModel = None
    TransformerModel = None
    PositionalEncoding = None
    ModelType = None
    DeepTrainingConfig = None
    ModelMetrics = None

# 模型部署服务
from .model_deployment_service import (
    DeploymentConfig,
    DeploymentRecord,
    DeploymentStatus,
    EvaluationMetric,
    ModelDeploymentService,
    ModelEvaluation,
)
from .model_deployment_service import ModelEvaluator as DeploymentEvaluator
from .model_deployment_service import ModelPerformanceMonitor

# 模型评估和版本管理
from .model_evaluation import ModelStatus, ModelVersionManager

# 模型存储
from .model_storage import ModelStorage

# 从shared_types导入ModelMetadata和ModelType
from .shared_types import ModelMetadata
from .shared_types import ModelType as StorageModelType

# 模型评估 - 可选导入
try:
    from .model_evaluation import (
        BacktestMetrics,
        FinancialMetricsCalculator,
        ModelEvaluator,
    )
    from .model_evaluation import ModelStatus as EvaluationModelStatus
    from .model_evaluation import ModelVersion
    from .model_evaluation import ModelVersionManager as EvaluationVersionManager
    from .model_evaluation import TimeSeriesValidator

    MODEL_EVALUATION_AVAILABLE = True
except ImportError:
    MODEL_EVALUATION_AVAILABLE = False
    ModelEvaluator = None
    EvaluationVersionManager = None
    TimeSeriesValidator = None
    FinancialMetricsCalculator = None
    BacktestMetrics = None
    ModelVersion = None
    EvaluationModelStatus = None

# 高级训练 - 可选导入
try:
    from .advanced_training import (
        AdvancedTrainingService,
        EnsembleConfig,
        EnsembleMethod,
        EnsembleModelManager,
    )
    from .advanced_training import ModelType as AdvancedModelType
    from .advanced_training import OnlineLearningConfig, OnlineLearningManager

    ADVANCED_TRAINING_AVAILABLE = True
except ImportError:
    ADVANCED_TRAINING_AVAILABLE = False
    AdvancedTrainingService = None
    EnsembleModelManager = None
    OnlineLearningManager = None
    EnsembleMethod = None
    EnsembleConfig = None
    OnlineLearningConfig = None
    AdvancedModelType = None

# 现代模型 - 可选导入
try:
    from .modern_models import (
        Inception_Block_V1,
        Informer,
        InformerEncoderLayer,
        PatchTST,
    )
    from .modern_models import PositionalEncoding as ModernPositionalEncoding
    from .modern_models import ProbAttention, TimesBlock, TimesNet

    MODERN_MODELS_AVAILABLE = True
except ImportError:
    MODERN_MODELS_AVAILABLE = False
    TimesNet = None
    TimesBlock = None
    Inception_Block_V1 = None
    PatchTST = None
    Informer = None
    InformerEncoderLayer = None
    ProbAttention = None
    ModernPositionalEncoding = None

__all__ = [
    # 模型训练服务
    "ModelTrainingService",
    "ModelTrainer",
    "RandomForestTrainer",
    "LinearRegressionTrainer",
    "TrainerFactory",
    "TrainingConfig",
    "TrainingResult",
    # 深度学习训练
    "DeepModelTrainingService",
    "QlibDataProvider",
    "LSTMModel",
    "TransformerModel",
    "PositionalEncoding",
    "ModelType",
    "DeepTrainingConfig",
    "ModelMetrics",
    # 模型存储
    "ModelStorage",
    "ModelVersionManager",
    "ModelMetadata",
    "ModelStatus",
    "StorageModelType",
    # 模型部署
    "ModelDeploymentService",
    "DeploymentEvaluator",
    "ModelPerformanceMonitor",
    "DeploymentStatus",
    "EvaluationMetric",
    "ModelEvaluation",
    "DeploymentConfig",
    "DeploymentRecord",
    # 模型评估
    "ModelEvaluator",
    "EvaluationVersionManager",
    "TimeSeriesValidator",
    "FinancialMetricsCalculator",
    "BacktestMetrics",
    "ModelVersion",
    "EvaluationModelStatus",
    # 高级训练
    "AdvancedTrainingService",
    "EnsembleModelManager",
    "OnlineLearningManager",
    "EnsembleMethod",
    "EnsembleConfig",
    "OnlineLearningConfig",
    "AdvancedModelType",
    # 现代模型
    "TimesNet",
    "TimesBlock",
    "Inception_Block_V1",
    "PatchTST",
    "Informer",
    "InformerEncoderLayer",
    "ProbAttention",
    "ModernPositionalEncoding",
]
