"""
业务服务层

注意：此文件提供向后兼容性支持。建议使用新的模块化导入方式：
- from app.services.data import DataService
- from app.services.models import ModelTrainingService
- from app.services.prediction import PredictionEngine
- from app.services.backtest import BacktestEngine
- from app.services.tasks import TaskManager
- from app.services.infrastructure import CacheManager
"""

import warnings
from importlib import import_module
from typing import Any

_DEPRECATED_SERVICE_IMPORTS = {
    # 数据管理
    "SimpleDataService": "data",
    # 模型管理
    "ModelTrainingService": "models",
    "ModelStorage": "models",
    "ModelDeploymentService": "models",
    "ModelEvaluator": "models",
    "AdvancedTrainingService": "models",
    "TimesNet": "models",
    "PatchTST": "models",
    "Informer": "models",
    # 预测引擎
    "PredictionEngine": "prediction",
    "PredictionFallbackEngine": "prediction",
    "RiskAssessmentService": "prediction",
    "FeatureExtractor": "prediction",
    "TechnicalIndicatorCalculator": "prediction",
    # 回测引擎
    "BacktestEngine": "backtest",
    "BacktestExecutor": "backtest",
    # 任务管理
    "TaskManager": "tasks",
    "TaskQueueManager": "tasks",
    "TaskExecutionEngine": "tasks",
    "TaskNotificationService": "tasks",
    # 基础设施
    "CacheManager": "infrastructure",
    "ConnectionPoolManager": "infrastructure",
    "DataMonitoringService": "infrastructure",
    "EnhancedLogger": "infrastructure",
    "WebSocketManager": "infrastructure",
}


# 向后兼容性警告
def _deprecated_import_warning(old_import: str, new_import: str) -> Any:
    """发出弃用警告"""
    warnings.warn(
        f"从 'app.services.{old_import}' 导入已弃用。"
        f"请使用 'app.services.{new_import}' 代替。",
        DeprecationWarning,
        stacklevel=3,
    )


def _load_deprecated_service(name: str, module_name: str) -> Any:
    """按需加载兼容导出，并保留弃用提示。"""
    _deprecated_import_warning(name, f"{module_name}.{name}")
    module = import_module(f".{module_name}", __name__)
    return getattr(module, name)


# 数据管理模块的向后兼容导入
def __getattr__(name: str) -> Any:
    """动态导入以支持向后兼容性"""
    module_name = _DEPRECATED_SERVICE_IMPORTS.get(name)
    if module_name is not None:
        return _load_deprecated_service(name, module_name)

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# 延迟导入函数（保持向后兼容）
def get_model_training_service() -> Any:
    """延迟导入模型训练服务（已弃用）"""
    _deprecated_import_warning(
        "get_model_training_service()", "models.ModelTrainingService"
    )
    from .models import (
        DeepModelTrainingService,
        DeepTrainingConfig,
        ModelMetrics,
        ModelType,
    )

    return DeepModelTrainingService, DeepTrainingConfig, ModelType, ModelMetrics


def get_modern_models() -> Any:
    """延迟导入现代模型（已弃用）"""
    _deprecated_import_warning(
        "get_modern_models()", "models.TimesNet, models.PatchTST, models.Informer"
    )
    from .models import Informer, PatchTST, TimesNet

    return TimesNet, PatchTST, Informer


# 为了支持 from app.services import * 的用法，定义 __all__
__all__ = [
    *_DEPRECATED_SERVICE_IMPORTS,
    # 延迟导入函数
    "get_model_training_service",
    "get_modern_models",
]
