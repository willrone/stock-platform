"""
Qlib训练配置类

包含模型类型、训练配置和训练结果的数据类定义。
支持多种特征集（Alpha158 / 手工62特征）和标签类型（回归 / 二分类）。
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


class LabelType(Enum):
    """标签类型"""

    REGRESSION = "regression"  # 回归：预测未来N天收益率
    BINARY = "binary"  # 二分类：涨>阈值=1


class FeatureSetChoice(Enum):
    """特征集选择"""

    ALPHA158 = "alpha158"  # Qlib Alpha158因子（158个）
    TECHNICAL_62 = "technical_62"  # 手工技术指标（62个）
    CUSTOM = "custom"  # 自定义特征列表


class DataSplitMethod(Enum):
    """数据分割方式"""

    RATIO = "ratio"  # 按比例分割（默认80/20）
    HARDCUT = "hardcut"  # 按固定日期硬切
    PURGED_CV = "purged_cv"  # Purged K-Fold（防信息泄漏，推荐）


@dataclass
class QlibTrainingConfig:
    """Qlib训练配置"""

    model_type: QlibModelType
    hyperparameters: Dict[str, Any]
    sequence_length: int = 60
    prediction_horizon: int = 5
    validation_split: float = 0.2
    early_stopping_patience: int = 50
    use_alpha_factors: bool = True
    cache_features: bool = True
    # 特征选择配置
    selected_features: Optional[List[str]] = None
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
    # === 统一训练体系新增配置 ===
    # 特征集选择（默认 alpha158 保持向后兼容）
    feature_set: str = "alpha158"
    # 标签类型（默认 regression 保持向后兼容）
    label_type: str = "regression"
    # 二分类阈值（仅 label_type=binary 时生效）
    binary_threshold: float = 0.003
    # 数据分割方式（默认 purged_cv，防信息泄漏）
    split_method: str = "purged_cv"
    # 硬切日期（仅 split_method=hardcut 时生效）
    train_end_date: Optional[str] = None
    val_end_date: Optional[str] = None
    # Purged K-Fold 配置（仅 split_method=purged_cv 时生效）
    purged_cv_splits: int = 5
    purged_cv_purge_days: int = 20
    # 标签 CSRankNorm 配置（截面排名标准化）
    enable_cs_rank_norm: bool = False
    # Stacking 集成配置
    enable_stacking: bool = False
    stacking_ridge_alpha: float = 1.0
    # === 滚动训练配置（P2） ===
    enable_rolling: bool = False
    rolling_window_type: str = "sliding"
    rolling_step: int = 60
    rolling_train_window: int = 480
    rolling_valid_window: int = 60
    enable_sample_decay: bool = True
    sample_decay_rate: float = 0.999

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
            "feature_set": self.feature_set,
            "label_type": self.label_type,
            "binary_threshold": self.binary_threshold,
            "split_method": self.split_method,
            "train_end_date": self.train_end_date,
            "val_end_date": self.val_end_date,
            "purged_cv_splits": self.purged_cv_splits,
            "purged_cv_purge_days": self.purged_cv_purge_days,
            "enable_cs_rank_norm": self.enable_cs_rank_norm,
            "enable_stacking": self.enable_stacking,
            "stacking_ridge_alpha": self.stacking_ridge_alpha,
            "enable_rolling": self.enable_rolling,
            "rolling_window_type": self.rolling_window_type,
            "rolling_step": self.rolling_step,
            "rolling_train_window": self.rolling_train_window,
            "rolling_valid_window": self.rolling_valid_window,
            "enable_sample_decay": self.enable_sample_decay,
            "sample_decay_rate": self.sample_decay_rate,
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
