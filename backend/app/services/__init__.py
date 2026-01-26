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
from typing import Any


# 向后兼容性警告
def _deprecated_import_warning(old_import: str, new_import: str):
    """发出弃用警告"""
    warnings.warn(
        f"从 'app.services.{old_import}' 导入已弃用。" f"请使用 'app.services.{new_import}' 代替。",
        DeprecationWarning,
        stacklevel=3,
    )


# 数据管理模块的向后兼容导入
def __getattr__(name: str) -> Any:
    """动态导入以支持向后兼容性"""

    # 数据管理服务
    if name == "SimpleDataService":
        _deprecated_import_warning("SimpleDataService", "data.SimpleDataService")
        from .data import SimpleDataService

        return SimpleDataService

    # 模型管理服务
    elif name == "ModelTrainingService":
        _deprecated_import_warning(
            "ModelTrainingService", "models.ModelTrainingService"
        )
        from .models import ModelTrainingService

        return ModelTrainingService
    elif name == "ModelStorage":
        _deprecated_import_warning("ModelStorage", "models.ModelStorage")
        from .models import ModelStorage

        return ModelStorage
    elif name == "ModelDeploymentService":
        _deprecated_import_warning(
            "ModelDeploymentService", "models.ModelDeploymentService"
        )
        from .models import ModelDeploymentService

        return ModelDeploymentService
    elif name == "ModelEvaluator":
        _deprecated_import_warning("ModelEvaluator", "models.ModelEvaluator")
        from .models import ModelEvaluator

        return ModelEvaluator
    elif name == "AdvancedTrainingService":
        _deprecated_import_warning(
            "AdvancedTrainingService", "models.AdvancedTrainingService"
        )
        from .models import AdvancedTrainingService

        return AdvancedTrainingService
    elif name == "TimesNet":
        _deprecated_import_warning("TimesNet", "models.TimesNet")
        from .models import TimesNet

        return TimesNet
    elif name == "PatchTST":
        _deprecated_import_warning("PatchTST", "models.PatchTST")
        from .models import PatchTST

        return PatchTST
    elif name == "Informer":
        _deprecated_import_warning("Informer", "models.Informer")
        from .models import Informer

        return Informer

    # 预测引擎服务
    elif name == "PredictionEngine":
        _deprecated_import_warning("PredictionEngine", "prediction.PredictionEngine")
        from .prediction import PredictionEngine

        return PredictionEngine
    elif name == "PredictionFallbackEngine":
        _deprecated_import_warning(
            "PredictionFallbackEngine", "prediction.PredictionFallbackEngine"
        )
        from .prediction import PredictionFallbackEngine

        return PredictionFallbackEngine
    elif name == "RiskAssessmentService":
        _deprecated_import_warning(
            "RiskAssessmentService", "prediction.RiskAssessmentService"
        )
        from .prediction import RiskAssessmentService

        return RiskAssessmentService
    elif name == "FeatureExtractor":
        _deprecated_import_warning("FeatureExtractor", "prediction.FeatureExtractor")
        from .prediction import FeatureExtractor

        return FeatureExtractor
    elif name == "TechnicalIndicatorCalculator":
        _deprecated_import_warning(
            "TechnicalIndicatorCalculator", "prediction.TechnicalIndicatorCalculator"
        )
        from .prediction import TechnicalIndicatorCalculator

        return TechnicalIndicatorCalculator

    # 回测引擎服务
    elif name == "BacktestEngine":
        _deprecated_import_warning("BacktestEngine", "backtest.BacktestEngine")
        from .backtest import BacktestEngine

        return BacktestEngine
    elif name == "BacktestExecutor":
        _deprecated_import_warning("BacktestExecutor", "backtest.BacktestExecutor")
        from .backtest import BacktestExecutor

        return BacktestExecutor

    # 任务管理服务
    elif name == "TaskManager":
        _deprecated_import_warning("TaskManager", "tasks.TaskManager")
        from .tasks import TaskManager

        return TaskManager
    elif name == "TaskQueueManager":
        _deprecated_import_warning("TaskQueueManager", "tasks.TaskQueueManager")
        from .tasks import TaskQueueManager

        return TaskQueueManager
    elif name == "TaskExecutionEngine":
        _deprecated_import_warning("TaskExecutionEngine", "tasks.TaskExecutionEngine")
        from .tasks import TaskExecutionEngine

        return TaskExecutionEngine
    elif name == "TaskNotificationService":
        _deprecated_import_warning(
            "TaskNotificationService", "tasks.TaskNotificationService"
        )
        from .tasks import TaskNotificationService

        return TaskNotificationService

    # 基础设施服务
    elif name == "CacheManager":
        _deprecated_import_warning("CacheManager", "infrastructure.CacheManager")
        from .infrastructure import CacheManager

        return CacheManager
    elif name == "ConnectionPoolManager":
        _deprecated_import_warning(
            "ConnectionPoolManager", "infrastructure.ConnectionPoolManager"
        )
        from .infrastructure import ConnectionPoolManager

        return ConnectionPoolManager
    elif name == "DataMonitoringService":
        _deprecated_import_warning(
            "DataMonitoringService", "infrastructure.DataMonitoringService"
        )
        from .infrastructure import DataMonitoringService

        return DataMonitoringService
    elif name == "EnhancedLogger":
        _deprecated_import_warning("EnhancedLogger", "infrastructure.EnhancedLogger")
        from .infrastructure import EnhancedLogger

        return EnhancedLogger
    elif name == "WebSocketManager":
        _deprecated_import_warning(
            "WebSocketManager", "infrastructure.WebSocketManager"
        )
        from .infrastructure import WebSocketManager

        return WebSocketManager

    # 如果没有找到匹配的属性，抛出 AttributeError
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# 延迟导入函数（保持向后兼容）
def get_model_training_service():
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


def get_modern_models():
    """延迟导入现代模型（已弃用）"""
    _deprecated_import_warning(
        "get_modern_models()", "models.TimesNet, models.PatchTST, models.Informer"
    )
    from .models import Informer, PatchTST, TimesNet

    return TimesNet, PatchTST, Informer


# 为了支持 from app.services import * 的用法，定义 __all__
__all__ = [
    # 数据管理
    "SimpleDataService",
    # 模型管理
    "ModelTrainingService",
    "ModelStorage",
    "ModelDeploymentService",
    "ModelEvaluator",
    "AdvancedTrainingService",
    "TimesNet",
    "PatchTST",
    "Informer",
    # 预测引擎
    "PredictionEngine",
    "PredictionFallbackEngine",
    "RiskAssessmentService",
    "FeatureExtractor",
    "TechnicalIndicatorCalculator",
    # 回测引擎
    "BacktestEngine",
    "BacktestExecutor",
    # 任务管理
    "TaskManager",
    "TaskQueueManager",
    "TaskExecutionEngine",
    "TaskNotificationService",
    # 基础设施
    "CacheManager",
    "ConnectionPoolManager",
    "DataMonitoringService",
    "EnhancedLogger",
    "WebSocketManager",
    # 延迟导入函数
    "get_model_training_service",
    "get_modern_models",
]
