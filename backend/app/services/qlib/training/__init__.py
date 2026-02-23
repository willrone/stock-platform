"""
Qlib训练引擎模���化包

将统一Qlib训练引擎拆分为多个模块化文件。
支持多种特征集和标签类型的统一训练 Pipeline。
"""

from .config import (
    DataSplitMethod,
    FeatureSetChoice,
    LabelType,
    QlibModelType,
    QlibTrainingConfig,
    QlibTrainingResult,
)
from .engine import UnifiedQlibTrainingEngine
from .feature_sets import FeatureSetType

__all__ = [
    "DataSplitMethod",
    "FeatureSetChoice",
    "FeatureSetType",
    "LabelType",
    "QlibModelType",
    "QlibTrainingConfig",
    "QlibTrainingResult",
    "UnifiedQlibTrainingEngine",
]
