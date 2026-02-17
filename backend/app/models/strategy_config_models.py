"""
策略配置数据模型
"""

from datetime import datetime
from typing import Any, Dict

from sqlalchemy import JSON, Column, DateTime, String, Text

from app.core.database import Base


class StrategyConfig(Base):
    """策略配置表"""

    __tablename__ = "strategy_configs"

    config_id = Column(String, primary_key=True)  # 由API层手动生成
    config_name = Column(String(255), nullable=False)
    strategy_name = Column(String(100), nullable=False)
    parameters = Column(JSON, nullable=False)  # 策略参数配置
    description = Column(Text, nullable=True)
    user_id = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

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
