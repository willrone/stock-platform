"""
Qlib训练配置类

包含模型类型、训练配置和训练结果的数据类定义
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class QlibModelType(Enum):
    """支持的Qlib模型类型"""

    # 传统机器学习模型
    LIGHTGBM = "lightgbm"
    XGBOOST = "xgboost"
    LINEAR = "linear"

    # 深度学习模型
    MLP = "mlp"
    TRANSFORMER = "transformer"
    INFORMER = "informer"
    TIMESNET = "timesnet"
    PATCHTST = "patchtst"


@dataclass
class QlibTrainingConfig:
    """Qlib训练配置"""

    model_type: QlibModelType
    hyperparameters: Dict[str, Any]
    sequence_length: int = 60
    prediction_horizon: int = 5
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    use_alpha_factors: bool = True
    cache_features: bool = True
    # 特征选择配置
    selected_features: Optional[List[str]] = None  # 用户选择的特征列表，None表示使用所有特征
    # 早停策略配置
    enable_early_stopping: bool = True
    early_stopping_monitor: str = "val_loss"
    early_stopping_min_delta: float = 0.001
    enable_overfitting_detection: bool = True
    enable_adaptive_patience: bool = True
    # Embargo 期配置（防止信息泄漏）
    embargo_days: int = 20
    # 市场中性化配置
    enable_neutralization: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "model_type": self.model_type.value,
            "hyperparameters": self.hyperparameters,
            "sequence_length": self.sequence_length,
            "prediction_horizon": self.prediction_horizon,
            "validation_split": self.validation_split,
            "early_stopping_patience": self.early_stopping_patience,
            "use_alpha_factors": self.use_alpha_factors,
            "cache_features": self.cache_features,
            "selected_features": self.selected_features,
            "enable_early_stopping": self.enable_early_stopping,
            "early_stopping_monitor": self.early_stopping_monitor,
            "early_stopping_min_delta": self.early_stopping_min_delta,
            "enable_overfitting_detection": self.enable_overfitting_detection,
            "enable_adaptive_patience": self.enable_adaptive_patience,
            "embargo_days": self.embargo_days,
            "enable_neutralization": self.enable_neutralization,
        }


@dataclass
class QlibTrainingResult:
    """Qlib训练结果"""

    model_path: str
    model_config: Dict[str, Any]
    training_metrics: Dict[str, float]
    validation_metrics: Dict[str, float]
    feature_importance: Optional[Dict[str, float]]
    training_history: List[Dict[str, Any]]
    training_duration: float
    # 样本数信息
    train_samples: int = 0
    validation_samples: int = 0
    test_samples: int = 0
    # 早停相关信息
    early_stopped: bool = False
    stopped_epoch: int = 0
    best_epoch: int = 0
    early_stopping_reason: Optional[str] = None
    feature_correlation: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "model_path": self.model_path,
            "model_config": self.model_config,
            "training_metrics": self.training_metrics,
            "validation_metrics": self.validation_metrics,
            "feature_importance": self.feature_importance,
            "training_history": self.training_history,
            "training_duration": self.training_duration,
            "train_samples": self.train_samples,
            "validation_samples": self.validation_samples,
            "test_samples": self.test_samples,
            "early_stopped": self.early_stopped,
            "stopped_epoch": self.stopped_epoch,
            "best_epoch": self.best_epoch,
            "early_stopping_reason": self.early_stopping_reason,
            "feature_correlation": self.feature_correlation,
        }
