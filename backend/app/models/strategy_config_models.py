"""
策略配置数据模型（PostgreSQL）
"""

from typing import Any, Dict

from sqlalchemy import Column, DateTime, Index, String, Text, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.sql import func

from app.core.database import Base


class StrategyConfig(Base):
    """策略配置表"""

    __tablename__ = "strategy_configs"

    config_id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    config_name = Column(String(255), nullable=False)
    strategy_name = Column(String(100), nullable=False)
    parameters = Column(JSONB, nullable=False, comment="策略参数配置")
    description = Column(Text, nullable=True)
    user_id = Column(String(255), nullable=True)
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
        Index("ix_strategy_config_strategy_name", "strategy_name"),
        Index("ix_strategy_config_user_id", "user_id"),
    )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "config_id": str(self.config_id),
            "config_name": self.config_name,
            "strategy_name": self.strategy_name,
            "parameters": self.parameters,
            "description": self.description,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
