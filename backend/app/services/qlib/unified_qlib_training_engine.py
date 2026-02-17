"""
统一Qlib训练引擎 - 向后兼容包装器

这个文件保持原有的导入接口，实际实现已拆分到 training/ 模块中
"""

# 导出所有公共接口
from .training.config import QlibModelType, QlibTrainingConfig, QlibTrainingResult
from .training.data_preprocessing import OutlierHandler, RobustFeatureScaler
from .training.engine import UnifiedQlibTrainingEngine
from .training.qlib_check import QLIB_AVAILABLE

# 保持向后兼容的导出
__all__ = [
    "QlibModelType",
    "QlibTrainingConfig",
    "QlibTrainingResult",
    "OutlierHandler",
    "RobustFeatureScaler",
    "UnifiedQlibTrainingEngine",
    "QLIB_AVAILABLE",
]
