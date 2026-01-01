"""
任务管理相关数据模型
"""

from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from sqlalchemy import Column, Integer, String, DateTime, Float, Text, JSON, Boolean
from sqlalchemy.dialects.postgresql import UUID
import uuid

from app.core.database import Base


class TaskType(Enum):
    """任务类型"""
    PREDICTION = "prediction"
    BACKTEST = "backtest"
    TRAINING = "training"
    DATA_SYNC = "data_sync"


class TaskStatus(Enum):
    """任务状态"""
    CREATED = "created"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class Task(Base):
    """任务表"""
    __tablename__ = "tasks"
    
    task_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    task_name = Column(String(255), nullable=False)
    task_type = Column(String(50), nullable=False)
    status = Column(String(50), nullable=False, default=TaskStatus.CREATED.value)
    user_id = Column(String(255), nullable=True)
    config = Column(JSON, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    progress = Column(Float, nullable=False, default=0.0)
    result = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)
    estimated_duration = Column(Integer, nullable=True)  # 预估时长（秒）
    
    def to_dict(self):
        return {
            "task_id": self.task_id,
            "task_name": self.task_name,
            "task_type": self.task_type,
            "status": self.status,
            "user_id": self.user_id,
            "config": self.config,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "progress": self.progress,
            "result": self.result,
            "error_message": self.error_message,
            "estimated_duration": self.estimated_duration
        }


class PredictionResult(Base):
    """预测结果表"""
    __tablename__ = "prediction_results"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(String, nullable=False)
    stock_code = Column(String(20), nullable=False)
    prediction_date = Column(DateTime, nullable=False)
    predicted_price = Column(Float, nullable=False)
    predicted_direction = Column(Integer, nullable=False)  # 1: 上涨, -1: 下跌, 0: 持平
    confidence_score = Column(Float, nullable=False)
    confidence_interval_lower = Column(Float, nullable=True)
    confidence_interval_upper = Column(Float, nullable=True)
    model_id = Column(String(255), nullable=False)
    features_used = Column(JSON, nullable=True)
    risk_metrics = Column(JSON, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            "id": self.id,
            "task_id": self.task_id,
            "stock_code": self.stock_code,
            "prediction_date": self.prediction_date.isoformat() if self.prediction_date else None,
            "predicted_price": self.predicted_price,
            "predicted_direction": self.predicted_direction,
            "confidence_score": self.confidence_score,
            "confidence_interval": {
                "lower": self.confidence_interval_lower,
                "upper": self.confidence_interval_upper
            },
            "model_id": self.model_id,
            "features_used": self.features_used,
            "risk_metrics": self.risk_metrics,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class BacktestResult(Base):
    """回测结果表"""
    __tablename__ = "backtest_results"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(String, nullable=False)
    backtest_id = Column(String, nullable=False, unique=True)
    strategy_name = Column(String(255), nullable=False)
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    initial_cash = Column(Float, nullable=False)
    final_value = Column(Float, nullable=False)
    total_return = Column(Float, nullable=False)
    annualized_return = Column(Float, nullable=False)
    volatility = Column(Float, nullable=False)
    sharpe_ratio = Column(Float, nullable=False)
    max_drawdown = Column(Float, nullable=False)
    win_rate = Column(Float, nullable=False)
    profit_factor = Column(Float, nullable=False)
    total_trades = Column(Integer, nullable=False)
    trade_history = Column(JSON, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            "id": self.id,
            "task_id": self.task_id,
            "backtest_id": self.backtest_id,
            "strategy_name": self.strategy_name,
            "period": {
                "start_date": self.start_date.isoformat() if self.start_date else None,
                "end_date": self.end_date.isoformat() if self.end_date else None
            },
            "portfolio": {
                "initial_cash": self.initial_cash,
                "final_value": self.final_value,
                "total_return": self.total_return,
                "annualized_return": self.annualized_return
            },
            "risk_metrics": {
                "volatility": self.volatility,
                "sharpe_ratio": self.sharpe_ratio,
                "max_drawdown": self.max_drawdown
            },
            "trading_stats": {
                "total_trades": self.total_trades,
                "win_rate": self.win_rate,
                "profit_factor": self.profit_factor
            },
            "trade_history": self.trade_history,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class ModelInfo(Base):
    """模型信息表"""
    __tablename__ = "model_info"
    
    model_id = Column(String, primary_key=True)
    model_name = Column(String(255), nullable=False)
    model_type = Column(String(100), nullable=False)
    version = Column(String(50), nullable=False)
    parent_model_id = Column(String, nullable=True)  # 父模型ID，用于版本管理
    file_path = Column(String(500), nullable=False)
    training_data_start = Column(DateTime, nullable=True)
    training_data_end = Column(DateTime, nullable=True)
    performance_metrics = Column(JSON, nullable=True)
    hyperparameters = Column(JSON, nullable=True)
    status = Column(String(50), nullable=False, default="training")
    training_progress = Column(Float, nullable=True, default=0.0)  # 训练进度 0-100
    training_stage = Column(String(100), nullable=True)  # 当前训练阶段
    evaluation_report = Column(JSON, nullable=True)  # 评估报告
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    deployed_at = Column(DateTime, nullable=True)
    
    def to_dict(self):
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "model_type": self.model_type,
            "version": self.version,
            "parent_model_id": self.parent_model_id,
            "file_path": self.file_path,
            "training_data_period": {
                "start": self.training_data_start.isoformat() if self.training_data_start else None,
                "end": self.training_data_end.isoformat() if self.training_data_end else None
            },
            "performance_metrics": self.performance_metrics,
            "hyperparameters": self.hyperparameters,
            "status": self.status,
            "training_progress": self.training_progress,
            "training_stage": self.training_stage,
            "evaluation_report": self.evaluation_report,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "deployed_at": self.deployed_at.isoformat() if self.deployed_at else None
        }


# 数据传输对象 (DTOs)
@dataclass
class PredictionTaskConfig:
    """预测任务配置"""
    stock_codes: List[str]
    model_id: str
    horizon: str = "short_term"
    confidence_level: float = 0.95
    features: Optional[List[str]] = None


@dataclass
class BacktestTaskConfig:
    """回测任务配置"""
    strategy_name: str
    stock_codes: List[str]
    start_date: datetime
    end_date: datetime
    initial_cash: float = 100000.0
    strategy_config: Optional[Dict[str, Any]] = None


@dataclass
class TrainingTaskConfig:
    """训练任务配置"""
    model_name: str
    model_type: str
    stock_codes: List[str]
    start_date: datetime
    end_date: datetime
    hyperparameters: Optional[Dict[str, Any]] = None
    validation_split: float = 0.2


@dataclass
class RiskMetrics:
    """风险指标"""
    value_at_risk: float
    expected_shortfall: float
    volatility: float
    max_drawdown: float
    sharpe_ratio: float
    
    def to_dict(self):
        return {
            "value_at_risk": self.value_at_risk,
            "expected_shortfall": self.expected_shortfall,
            "volatility": self.volatility,
            "max_drawdown": self.max_drawdown,
            "sharpe_ratio": self.sharpe_ratio
        }