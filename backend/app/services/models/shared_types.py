"""
共享类型定义

存放模型管理和训练模块共享的枚举类型、数据类和常量。
"""

from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime


class ModelType(Enum):
    """支持的模型类型"""
    TRANSFORMER = "transformer"
    TIMESNET = "timesnet"
    PATCHTST = "patchtst"
    INFORMER = "informer"
    LSTM = "lstm"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    RANDOM_FOREST = "random_forest"
    LINEAR_REGRESSION = "linear_regression"
    NEURAL_NETWORK = "neural_network"
    ENSEMBLE = "ensemble"


class ModelStatus(Enum):
    """模型状态"""
    TRAINING = "training"
    TRAINED = "trained"
    VALIDATING = "validating"
    READY = "ready"
    DEPLOYED = "deployed"
    DEPRECATED = "deprecated"
    FAILED = "failed"
    COMPLETED = "completed"
    ARCHIVED = "archived"


class EnsembleMethod(Enum):
    """集成方法类型"""
    VOTING = "voting"           # 投票集成
    WEIGHTED = "weighted"       # 加权集成
    STACKING = "stacking"       # 堆叠集成
    BAGGING = "bagging"         # 装袋集成


@dataclass
class TrainingConfig:
    """模型训练配置"""
    model_type: ModelType
    sequence_length: int = 60  # 输入序列长度
    prediction_horizon: int = 5  # 预测天数
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    feature_columns: List[str] = None
    target_column: str = "close"
    
    def __post_init__(self):
        if self.feature_columns is None:
            self.feature_columns = [
                "open", "high", "low", "close", "volume",
                "ma_5", "ma_10", "ma_20", "ma_60",
                "rsi", "macd", "macd_signal", "bb_upper", "bb_lower"
            ]


@dataclass
class ModelMetrics:
    """模型评估指标"""
    accuracy: float
    precision: float
    recall: float
    sharpe_ratio: float
    max_drawdown: float
    total_return: float
    win_rate: float
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "total_return": self.total_return,
            "win_rate": self.win_rate
        }


@dataclass
class BacktestMetrics:
    """回测评估指标"""
    # 基础分类指标
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    
    # 金融指标
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    
    # 风险指标
    volatility: float
    var_95: float  # 95% VaR
    calmar_ratio: float
    
    # 交易指标
    total_trades: int
    avg_trade_return: float
    max_consecutive_losses: int
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "total_return": self.total_return,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "volatility": self.volatility,
            "var_95": self.var_95,
            "calmar_ratio": self.calmar_ratio,
            "total_trades": self.total_trades,
            "avg_trade_return": self.avg_trade_return,
            "max_consecutive_losses": self.max_consecutive_losses
        }


@dataclass
class EnsembleConfig:
    """集成模型配置"""
    method: EnsembleMethod
    base_models: List[str]      # 基础模型ID列表
    weights: Optional[List[float]] = None  # 权重（用于加权集成）
    meta_model_type: Optional[ModelType] = None  # 元模型类型（用于堆叠集成）
    voting_strategy: str = "soft"  # 投票策略：hard或soft


@dataclass
class OnlineLearningConfig:
    """在线学习配置"""
    update_frequency: int = 5   # 更新频率（天）
    learning_rate_decay: float = 0.95  # 学习率衰减
    memory_size: int = 1000     # 记忆缓冲区大小
    adaptation_threshold: float = 0.1  # 性能下降阈值


@dataclass
class ModelVersion:
    """模型版本信息"""
    model_id: str
    version: str
    model_type: str
    parameters: Dict[str, Any]
    metrics: BacktestMetrics
    file_path: str
    created_at: datetime
    status: ModelStatus
    training_data_hash: str
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "model_id": self.model_id,
            "version": self.version,
            "model_type": self.model_type,
            "parameters": self.parameters,
            "metrics": self.metrics.to_dict(),
            "file_path": self.file_path,
            "created_at": self.created_at.isoformat(),
            "status": self.status.value,
            "training_data_hash": self.training_data_hash
        }
        return result


@dataclass
class ModelMetadata:
    """模型元数据"""
    model_id: str
    model_name: str
    model_type: ModelType
    version: str
    description: str
    created_by: str
    created_at: datetime
    updated_at: datetime
    status: ModelStatus
    
    # 训练信息
    training_data_info: Dict[str, Any]
    hyperparameters: Dict[str, Any]
    training_config: Dict[str, Any]
    
    # 性能指标
    performance_metrics: Dict[str, float]
    validation_metrics: Dict[str, float]
    
    # 部署信息
    deployment_info: Optional[Dict[str, Any]] = None
    
    # 文件信息
    model_file_path: Optional[str] = None
    model_file_size: Optional[int] = None
    model_file_hash: Optional[str] = None
    
    # 依赖信息
    dependencies: Optional[Dict[str, str]] = None
    feature_columns: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "model_type": self.model_type.value,
            "version": self.version,
            "description": self.description,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "status": self.status.value,
            "training_data_info": self.training_data_info,
            "hyperparameters": self.hyperparameters,
            "training_config": self.training_config,
            "performance_metrics": self.performance_metrics,
            "validation_metrics": self.validation_metrics,
            "deployment_info": self.deployment_info,
            "model_file_path": self.model_file_path,
            "model_file_size": self.model_file_size,
            "model_file_hash": self.model_file_hash,
            "dependencies": self.dependencies,
            "feature_columns": self.feature_columns
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        """从字典创建"""
        data = data.copy()
        data['model_type'] = ModelType(data['model_type'])
        data['status'] = ModelStatus(data['status'])
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        return cls(**data)
