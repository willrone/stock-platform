"""
Qlib 集成服务模块

提供 Qlib 框架的集成功能，包括：
- 增强的数据提供器（EnhancedQlibDataProvider）
- Alpha158 因子计算器（Alpha158Calculator）
- 因子缓存管理（FactorCache）
- 数据格式转换（QlibFormatConverter）
- 数据质量验证���DataQualityValidator）
- 数据处理工具（DataTypeOptimizer, MissingValueHandler）
- 模型配置构建（QlibModelConfigBuilder）
- 统一训练引擎（UnifiedQlibTrainingEngine）
- 模型配置管理器（QlibModelManager）
- 自定义模型实现（CustomModels）
"""

# 主入口
from .enhanced_qlib_provider import EnhancedQlibDataProvider

# Alpha158 因子计算
from .alpha158 import Alpha158Calculator

# 缓存管理
from .cache import FactorCache

# 数据转换
from .converters import ColumnStandardizer, QlibFormatConverter

# 数据处理
from .data_processing import (
    DataTypeOptimizer,
    FundamentalFeatureCalculator,
    MissingValueHandler,
)

# 模型配置和预测
from .model import QlibModelConfigBuilder, QlibModelPredictor

# 数据验证
from .validators import DataQualityValidator, ValidationReport
from .qlib_model_manager import (
    HyperparameterSpec,
    ModelCategory,
    ModelComplexity,
    ModelMetadata,
    QlibModelManager,
)
from .unified_qlib_training_engine import (
    QlibModelType,
    QlibTrainingConfig,
    QlibTrainingResult,
    UnifiedQlibTrainingEngine,
)

# 尝试导入自定义模型（可能因为依赖问题失败）
try:
    from .custom_models import (
        CustomInformerModel,
        CustomPatchTSTModel,
        CustomTimesNetModel,
        CustomTransformerModel,
    )

    CUSTOM_MODELS_AVAILABLE = True
except ImportError:
    CUSTOM_MODELS_AVAILABLE = False

__all__ = [
    # 数据提供器
    "EnhancedQlibDataProvider",
    "Alpha158Calculator",
    "FactorCache",
    # 数据转换
    "QlibFormatConverter",
    "ColumnStandardizer",
    # 数据处理
    "DataTypeOptimizer",
    "MissingValueHandler",
    "FundamentalFeatureCalculator",
    # 数据验证
    "DataQualityValidator",
    "ValidationReport",
    # 模型配置和预测
    "QlibModelConfigBuilder",
    "QlibModelPredictor",
    # 训练引擎
    "UnifiedQlibTrainingEngine",
    "QlibTrainingConfig",
    "QlibTrainingResult",
    "QlibModelType",
    # 模型管理器
    "QlibModelManager",
    "ModelMetadata",
    "HyperparameterSpec",
    "ModelCategory",
    "ModelComplexity",
    # 可用性标志
    "CUSTOM_MODELS_AVAILABLE",
]

# 如果自定义模型可用，添加到导出列表
if CUSTOM_MODELS_AVAILABLE:
    __all__.extend(
        [
            "CustomTransformerModel",
            "CustomInformerModel",
            "CustomTimesNetModel",
            "CustomPatchTSTModel",
        ]
    )
