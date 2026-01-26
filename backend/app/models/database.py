"""
数据库模型定义
定义SQLite数据库的表结构
"""

import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


class TaskStatus(Enum):
    """任务状态枚举"""

    PENDING = "pending"  # 待执行
    RUNNING = "running"  # 进行中
    COMPLETED = "completed"  # 已完成
    FAILED = "failed"  # 失败
    CANCELLED = "cancelled"  # 已取消


@dataclass
class Task:
    """任务模型"""

    id: Optional[int] = None
    name: str = ""
    description: str = ""
    stock_codes: str = ""  # JSON字符串存储股票代码列表
    indicators: str = ""  # JSON字符串存储指标列表
    models: str = ""  # JSON字符串存储模型列表
    parameters: str = ""  # JSON字符串存储参数
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "stock_codes": json.loads(self.stock_codes) if self.stock_codes else [],
            "indicators": json.loads(self.indicators) if self.indicators else [],
            "models": json.loads(self.models) if self.models else [],
            "parameters": json.loads(self.parameters) if self.parameters else {},
            "status": self.status.value,
            "progress": self.progress,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat()
            if self.completed_at
            else None,
            "error_message": self.error_message,
        }


@dataclass
class TaskResult:
    """任务结果模型"""

    id: Optional[int] = None
    task_id: int = 0
    stock_code: str = ""
    prediction_date: Optional[datetime] = None
    prediction_value: float = 0.0
    confidence: float = 0.0
    model_name: str = ""
    indicators_used: str = ""  # JSON字符串存储使用的指标
    backtest_metrics: str = ""  # JSON字符串存储回测指标
    created_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "task_id": self.task_id,
            "stock_code": self.stock_code,
            "prediction_date": self.prediction_date.isoformat()
            if self.prediction_date
            else None,
            "prediction_value": self.prediction_value,
            "confidence": self.confidence,
            "model_name": self.model_name,
            "indicators_used": json.loads(self.indicators_used)
            if self.indicators_used
            else [],
            "backtest_metrics": json.loads(self.backtest_metrics)
            if self.backtest_metrics
            else {},
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


@dataclass
class ModelMetadata:
    """模型元数据模型"""

    id: Optional[int] = None
    name: str = ""
    model_type: str = ""  # transformer, timesnet, patchtst, informer, lstm, xgboost
    version: str = ""
    parameters: str = ""  # JSON字符串存储模型参数
    training_data_info: str = ""  # JSON字符串存储训练数据信息
    performance_metrics: str = ""  # JSON字符串存储性能指标
    file_path: str = ""  # 模型文件路径
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    is_active: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "model_type": self.model_type,
            "version": self.version,
            "parameters": json.loads(self.parameters) if self.parameters else {},
            "training_data_info": json.loads(self.training_data_info)
            if self.training_data_info
            else {},
            "performance_metrics": json.loads(self.performance_metrics)
            if self.performance_metrics
            else {},
            "file_path": self.file_path,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "is_active": self.is_active,
        }


@dataclass
class SystemConfig:
    """系统配置模型"""

    id: Optional[int] = None
    key: str = ""
    value: str = ""
    description: str = ""
    updated_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "key": self.key,
            "value": self.value,
            "description": self.description,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class DatabaseManager:
    """数据库管理器"""

    def __init__(self, db_path: str = "data/stock_prediction.db"):
        self.db_path = db_path
        self.ensure_db_directory()
        self.init_database()

    def ensure_db_directory(self):
        """确保数据库目录存在"""
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)

    def get_connection(self) -> sqlite3.Connection:
        """获取数据库连接"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # 使结果可以按列名访问
        return conn

    def init_database(self):
        """初始化数据库表结构"""
        with self.get_connection() as conn:
            # 创建任务表
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    description TEXT,
                    stock_codes TEXT NOT NULL,
                    indicators TEXT NOT NULL,
                    models TEXT NOT NULL,
                    parameters TEXT,
                    status TEXT NOT NULL DEFAULT 'pending',
                    progress REAL DEFAULT 0.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    error_message TEXT
                )
            """
            )

            # 创建任务结果表
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS task_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id INTEGER NOT NULL,
                    stock_code TEXT NOT NULL,
                    prediction_date TIMESTAMP,
                    prediction_value REAL NOT NULL,
                    confidence REAL DEFAULT 0.0,
                    model_name TEXT NOT NULL,
                    indicators_used TEXT,
                    backtest_metrics TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (task_id) REFERENCES tasks (id)
                )
            """
            )

            # 创建模型元数据表
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS model_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    model_type TEXT NOT NULL,
                    version TEXT NOT NULL,
                    parameters TEXT,
                    training_data_info TEXT,
                    performance_metrics TEXT,
                    file_path TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1
                )
            """
            )

            # 创建系统配置表
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS system_config (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key TEXT NOT NULL UNIQUE,
                    value TEXT NOT NULL,
                    description TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # 创建索引
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks (status)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_tasks_created_at ON tasks (created_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_task_results_task_id ON task_results (task_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_task_results_stock_code ON task_results (stock_code)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_model_metadata_type ON model_metadata (model_type)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_model_metadata_active ON model_metadata (is_active)"
            )

            conn.commit()

    def execute_query(self, query: str, params: tuple = ()) -> sqlite3.Cursor:
        """执行查询"""
        conn = self.get_connection()
        try:
            cursor = conn.execute(query, params)
            conn.commit()
            return cursor
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def fetch_one(self, query: str, params: tuple = ()) -> Optional[sqlite3.Row]:
        """获取单条记录"""
        with self.get_connection() as conn:
            cursor = conn.execute(query, params)
            return cursor.fetchone()

    def fetch_all(self, query: str, params: tuple = ()) -> list:
        """获取多条记录"""
        with self.get_connection() as conn:
            cursor = conn.execute(query, params)
            return cursor.fetchall()
