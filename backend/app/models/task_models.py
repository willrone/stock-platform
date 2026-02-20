"""
任务管理相关数据模型（PostgreSQL）
"""

import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.sql import func

from app.core.database import Base


# ──────────────────────────── 枚举（保持不变） ────────────────────────────


class TaskType(Enum):
    """任务类型"""

    PREDICTION = "prediction"
    BACKTEST = "backtest"
    TRAINING = "training"
    DATA_SYNC = "data_sync"
    HYPERPARAMETER_OPTIMIZATION = "hyperparameter_optimization"
    QLIB_PRECOMPUTE = "qlib_precompute"


class TaskStatus(Enum):
    """任务状态"""

    CREATED = "created"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


# ──────────────────────────── ORM 模型 ────────────────────────────


class Task(Base):
    """任务表"""

    __tablename__ = "tasks"

    task_id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    task_name = Column(String(255), nullable=False)
    task_type = Column(
        String(50),
        nullable=False,
        comment="任务类型",
    )
    status = Column(
        String(50),
        nullable=False,
        server_default=TaskStatus.CREATED.value,
        comment="任务状态",
    )
    user_id = Column(String(255), nullable=True)
    config = Column(JSONB, nullable=True)
    created_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    progress = Column(Float, nullable=False, server_default=text("0"))
    result = Column(JSONB, nullable=True)
    error_message = Column(Text, nullable=True)
    estimated_duration = Column(Integer, nullable=True, comment="预估时长（秒）")

    __table_args__ = (
        Index("ix_tasks_status", "status"),
        Index("ix_tasks_task_type", "task_type"),
        Index("ix_tasks_user_id", "user_id"),
        Index("ix_tasks_created_at", "created_at"),
    )

    def to_dict(self):
        return {
            "task_id": str(self.task_id),
            "task_name": self.task_name,
            "task_type": self.task_type,
            "status": self.status,
            "user_id": self.user_id,
            "config": self.config,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat()
            if self.completed_at
            else None,
            "progress": self.progress,
            "result": self.result,
            "error_message": self.error_message,
            "estimated_duration": self.estimated_duration,
        }


class PredictionResult(Base):
    """预测结果表"""

    __tablename__ = "prediction_results"

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    task_id = Column(
        UUID(as_uuid=True),
        ForeignKey("tasks.task_id", ondelete="CASCADE"),
        nullable=False,
    )
    stock_code = Column(String(20), nullable=False)
    prediction_date = Column(DateTime(timezone=True), nullable=False)
    predicted_price = Column(Numeric(15, 2), nullable=False)
    predicted_direction = Column(
        Integer, nullable=False, comment="1: 上涨, -1: 下跌, 0: 持平"
    )
    confidence_score = Column(Float, nullable=False)
    confidence_interval_lower = Column(Numeric(15, 2), nullable=True)
    confidence_interval_upper = Column(Numeric(15, 2), nullable=True)
    model_id = Column(
        UUID(as_uuid=True),
        ForeignKey("model_info.model_id", ondelete="SET NULL"),
        nullable=True,
    )
    features_used = Column(JSONB, nullable=True)
    risk_metrics = Column(JSONB, nullable=True)
    created_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        Index("ix_prediction_task_id", "task_id"),
        Index("ix_prediction_stock_date", "stock_code", "prediction_date"),
    )

    def to_dict(self):
        return {
            "id": str(self.id),
            "task_id": str(self.task_id),
            "stock_code": self.stock_code,
            "prediction_date": self.prediction_date.isoformat()
            if self.prediction_date
            else None,
            "predicted_price": float(self.predicted_price)
            if self.predicted_price
            else None,
            "predicted_direction": self.predicted_direction,
            "confidence_score": self.confidence_score,
            "confidence_interval": {
                "lower": float(self.confidence_interval_lower)
                if self.confidence_interval_lower
                else None,
                "upper": float(self.confidence_interval_upper)
                if self.confidence_interval_upper
                else None,
            },
            "model_id": str(self.model_id) if self.model_id else None,
            "features_used": self.features_used,
            "risk_metrics": self.risk_metrics,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class BacktestResult(Base):
    """回测结果表（合并了原 BacktestDetailedResult 的扩展字段）"""

    __tablename__ = "backtest_results"

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    task_id = Column(
        UUID(as_uuid=True),
        ForeignKey("tasks.task_id", ondelete="CASCADE"),
        nullable=False,
    )
    backtest_id = Column(
        UUID(as_uuid=True),
        nullable=False,
        unique=True,
        server_default=text("gen_random_uuid()"),
    )
    strategy_name = Column(String(255), nullable=False)
    start_date = Column(DateTime(timezone=True), nullable=False)
    end_date = Column(DateTime(timezone=True), nullable=False)
    initial_cash = Column(Numeric(15, 2), nullable=False)
    final_value = Column(Numeric(15, 2), nullable=False)
    total_return = Column(Float, nullable=False)
    annualized_return = Column(Float, nullable=False)
    volatility = Column(Float, nullable=False)
    sharpe_ratio = Column(Float, nullable=False)
    max_drawdown = Column(Float, nullable=False)
    win_rate = Column(Float, nullable=False)
    profit_factor = Column(Float, nullable=False)
    total_trades = Column(Integer, nullable=False)
    # trade_history ��删除（与 trade_records 表冗余）

    # ── 从 BacktestDetailedResult 合并的扩展风险指标 ──
    sortino_ratio = Column(Float, nullable=True, comment="索提诺比率")
    calmar_ratio = Column(Float, nullable=True, comment="卡玛比率")
    max_drawdown_duration = Column(Integer, nullable=True, comment="最大回撤持续天数")
    var_95 = Column(Float, nullable=True, comment="95% VaR")
    downside_deviation = Column(Float, nullable=True, comment="下行偏差")

    # ── 从 BacktestDetailedResult 合并的 JSONB 分析数据 ──
    drawdown_analysis = Column(JSONB, nullable=True, comment="回撤详细分析数据")
    monthly_returns = Column(JSONB, nullable=True, comment="月度收益分析数据")
    position_analysis = Column(JSONB, nullable=True, comment="持仓分析数据")
    rolling_metrics = Column(JSONB, nullable=True, comment="滚动指标数据")
    benchmark_comparison = Column(JSONB, nullable=True, comment="基准对比数据")

    created_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    __table_args__ = (
        Index("ix_backtest_task_id", "task_id"),
        Index("ix_backtest_strategy", "strategy_name"),
    )

    def to_dict(self):
        # 从 benchmark_comparison JSONB 中提取捕获率等指标
        bm = self.benchmark_comparison or {}
        return {
            "id": str(self.id),
            "task_id": str(self.task_id),
            "backtest_id": str(self.backtest_id),
            "strategy_name": self.strategy_name,
            "period": {
                "start_date": self.start_date.isoformat()
                if self.start_date
                else None,
                "end_date": self.end_date.isoformat() if self.end_date else None,
            },
            "portfolio": {
                "initial_cash": float(self.initial_cash)
                if self.initial_cash
                else None,
                "final_value": float(self.final_value) if self.final_value else None,
                "total_return": self.total_return,
                "annualized_return": self.annualized_return,
            },
            "risk_metrics": {
                "volatility": self.volatility,
                "sharpe_ratio": self.sharpe_ratio,
                "max_drawdown": self.max_drawdown,
                "sortino_ratio": self.sortino_ratio,
                "calmar_ratio": self.calmar_ratio,
                "max_drawdown_duration": self.max_drawdown_duration,
                "var_95": self.var_95,
                "downside_deviation": self.downside_deviation,
            },
            # 前端 BacktestDetailedResult 接口期望的字段名
            "extended_risk_metrics": {
                "sortino_ratio": self.sortino_ratio,
                "calmar_ratio": self.calmar_ratio,
                "max_drawdown_duration": self.max_drawdown_duration,
                "var_95": self.var_95,
                "var_99": bm.get("var_99", 0),
                "cvar_95": bm.get("cvar_95", 0),
                "cvar_99": bm.get("cvar_99", 0),
                "downside_deviation": self.downside_deviation,
            },
            "trading_stats": {
                "total_trades": self.total_trades,
                "win_rate": self.win_rate,
                "profit_factor": self.profit_factor,
            },
            "drawdown_analysis": self.drawdown_analysis,
            "monthly_returns": self.monthly_returns,
            "position_analysis": self.position_analysis,
            "rolling_metrics": self.rolling_metrics,
            "benchmark_comparison": self.benchmark_comparison,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class ModelInfo(Base):
    """模型信息表"""

    __tablename__ = "model_info"

    model_id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    model_name = Column(String(255), nullable=False)
    model_type = Column(String(100), nullable=False)
    version = Column(String(50), nullable=False)
    parent_model_id = Column(
        UUID(as_uuid=True),
        ForeignKey("model_info.model_id", ondelete="SET NULL"),
        nullable=True,
        comment="父模型ID，用于版本管理",
    )
    file_path = Column(
        String(500),
        nullable=False,
        comment="模型文件相对路径（相对于 MODEL_STORAGE_PATH）",
    )
    training_data_start = Column(DateTime(timezone=True), nullable=True)
    training_data_end = Column(DateTime(timezone=True), nullable=True)
    performance_metrics = Column(JSONB, nullable=True)
    hyperparameters = Column(JSONB, nullable=True)
    status = Column(String(50), nullable=False, server_default=text("'training'"))
    training_progress = Column(Float, nullable=True, server_default=text("0"))
    training_stage = Column(String(100), nullable=True, comment="当前训练阶段")
    evaluation_report = Column(JSONB, nullable=True, comment="评估报告")
    created_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
    deployed_at = Column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("ix_model_info_model_type", "model_type"),
        Index("ix_model_info_status", "status"),
    )

    def to_dict(self):
        return {
            "model_id": str(self.model_id),
            "model_name": self.model_name,
            "model_type": self.model_type,
            "version": self.version,
            "parent_model_id": str(self.parent_model_id)
            if self.parent_model_id
            else None,
            "file_path": self.file_path,
            "training_data_period": {
                "start": self.training_data_start.isoformat()
                if self.training_data_start
                else None,
                "end": self.training_data_end.isoformat()
                if self.training_data_end
                else None,
            },
            "performance_metrics": self.performance_metrics,
            "hyperparameters": self.hyperparameters,
            "status": self.status,
            "training_progress": self.training_progress,
            "training_stage": self.training_stage,
            "evaluation_report": self.evaluation_report,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "deployed_at": self.deployed_at.isoformat() if self.deployed_at else None,
        }


class ModelLifecycleEvent(Base):
    """模型生命周期事件表"""

    __tablename__ = "model_lifecycle_events"

    event_id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    model_id = Column(
        UUID(as_uuid=True),
        ForeignKey("model_info.model_id", ondelete="CASCADE"),
        nullable=False,
    )
    from_status = Column(String(50), nullable=False)
    to_status = Column(String(50), nullable=False)
    reason = Column(Text, nullable=True)
    event_metadata = Column(JSONB, nullable=True, comment="附加元数据")
    created_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (Index("ix_lifecycle_model_id", "model_id"),)

    def to_dict(self):
        return {
            "event_id": str(self.event_id),
            "model_id": str(self.model_id),
            "from_status": self.from_status,
            "to_status": self.to_status,
            "reason": self.reason,
            "event_metadata": self.event_metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# ──────────────────────────── DTOs（保持不变） ────────────────────────────


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
            "sharpe_ratio": self.sharpe_ratio,
        }
