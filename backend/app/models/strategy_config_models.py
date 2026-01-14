"""
策略配置数据模型
"""

from datetime import datetime
from typing import Optional, Dict, Any
from sqlalchemy import Column, String, DateTime, Text, JSON, Boolean
from sqlalchemy.dialects.postgresql import UUID
import uuid

from app.core.database import Base


class StrategyConfig(Base):
    """策略配置表"""
    __tablename__ = "strategy_configs"
    
    config_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    config_name = Column(String(255), nullable=False)
    strategy_name = Column(String(100), nullable=False)
    parameters = Column(JSON, nullable=False)  # 策略参数配置
    description = Column(Text, nullable=True)
    user_id = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "config_id": self.config_id,
            "config_name": self.config_name,
            "strategy_name": self.strategy_name,
            "parameters": self.parameters,
            "description": self.description,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

