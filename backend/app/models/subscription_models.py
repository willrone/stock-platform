"""
订阅、套餐和用量相关数据模型。

Stripe 只负责支付和订阅状态，本地表保存平台实际使用的套餐配置以及
用户用量，便于在 Stripe 不可用时仍能读取订阅信息。
"""

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
)

from app.core.database import Base


def generate_uuid() -> str:
    """生成模型主键。"""
    return str(uuid.uuid4())


def utcnow() -> datetime:
    """返回无时区 UTC 时间，和项目现有 SQLite DateTime 字段保持一致。"""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class SubscriptionPlan(Base):
    """套餐配置表。"""

    __tablename__ = "subscription_plans"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    name = Column(
        String(100), nullable=False, unique=True, index=True
    )  # free / pro / enterprise
    display_name = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    monthly_price_cents = Column(Integer, default=0, nullable=False)  # 月付价格（分）
    yearly_price_cents = Column(Integer, default=0, nullable=False)  # 年付价格（分）

    # 功能限额，-1 表示不限量
    monthly_backtest_limit = Column(Integer, default=50, nullable=False)
    max_strategies = Column(Integer, default=20, nullable=False)
    max_concurrent_tasks = Column(Integer, default=3, nullable=False)
    max_data_points = Column(Integer, default=100000, nullable=False)

    # Stripe 产品/价格 ID
    stripe_product_id = Column(String(255), nullable=True)
    stripe_monthly_price_id = Column(String(255), nullable=True)
    stripe_yearly_price_id = Column(String(255), nullable=True)

    # 元数据
    features = Column(JSON, nullable=True)  # 特性列表 ["实时数据", "API 访问", ...]
    is_active = Column(Boolean, default=True, nullable=False)
    sort_order = Column(Integer, default=0, nullable=False)
    created_at = Column(DateTime, default=utcnow, nullable=False)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow, nullable=False)

    def to_dict(self) -> Dict[str, Any]:
        """转换为 API 可直接返回的字典。"""
        return {
            "id": self.id,
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "monthly_price_cents": self.monthly_price_cents,
            "yearly_price_cents": self.yearly_price_cents,
            "monthly_backtest_limit": self.monthly_backtest_limit,
            "max_strategies": self.max_strategies,
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "max_data_points": self.max_data_points,
            "features": self.features or [],
            "is_active": self.is_active,
            "sort_order": self.sort_order,
        }


class UserSubscription(Base):
    """用户订阅记录。"""

    __tablename__ = "user_subscriptions"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    plan_id = Column(
        String(36), ForeignKey("subscription_plans.id"), nullable=False, index=True
    )
    stripe_subscription_id = Column(String(255), nullable=True, index=True)
    stripe_customer_id = Column(String(255), nullable=True, index=True)
    status = Column(
        String(50), default="active", nullable=False
    )  # active / canceled / past_due / trialing
    billing_interval = Column(
        String(20), default="monthly", nullable=False
    )  # monthly / yearly
    current_period_start = Column(DateTime, nullable=True)
    current_period_end = Column(DateTime, nullable=True)
    cancel_at = Column(DateTime, nullable=True)
    canceled_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=utcnow, nullable=False)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow, nullable=False)

    def to_dict(self) -> Dict[str, Any]:
        """转换为 API 可直接返回的字典。"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "plan_id": self.plan_id,
            "stripe_subscription_id": self.stripe_subscription_id,
            "stripe_customer_id": self.stripe_customer_id,
            "status": self.status,
            "billing_interval": self.billing_interval,
            "current_period_start": (
                self.current_period_start.isoformat()
                if self.current_period_start
                else None
            ),
            "current_period_end": (
                self.current_period_end.isoformat() if self.current_period_end else None
            ),
            "cancel_at": self.cancel_at.isoformat() if self.cancel_at else None,
            "canceled_at": self.canceled_at.isoformat() if self.canceled_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class UsageRecord(Base):
    """用量记录。"""

    __tablename__ = "usage_records"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    usage_type = Column(String(50), nullable=False)  # backtest / api_call / data_fetch
    quantity = Column(Integer, default=1, nullable=False)
    # metadata 是 SQLAlchemy Declarative API 的保留属性，因此使用同名数据库列
    # 和 metadata_json Python 映射属性，兼容 SQLite 表结构与 ORM。
    metadata_json = Column("metadata", JSON, nullable=True)
    created_at = Column(DateTime, default=utcnow, nullable=False, index=True)

    def to_dict(self) -> Dict[str, Any]:
        """转换为 API 可直接返回的字典。"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "usage_type": self.usage_type,
            "quantity": self.quantity,
            "metadata": self.metadata_json,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# 允许实例层继续使用需求中约定的 ``record.metadata``，同时不破坏
# Declarative API 在类定义阶段对 Base.metadata 的处理。
UsageRecord.metadata = property(  # type: ignore[attr-defined]
    lambda self: self.metadata_json,
    lambda self, value: setattr(self, "metadata_json", value),
)


DEFAULT_PLANS: List[Dict[str, Any]] = [
    {
        "name": "free",
        "display_name": "免费版",
        "description": "适合体验股票预测和基础回测功能",
        "monthly_price_cents": 0,
        "yearly_price_cents": 0,
        "monthly_backtest_limit": 50,
        "max_strategies": 20,
        "max_concurrent_tasks": 3,
        "max_data_points": 100000,
        "features": ["基础行情数据", "基础回测"],
        "sort_order": 0,
    },
    {
        "name": "pro",
        "display_name": "专业版",
        "description": "面向活跃量化研究者的完整功能套餐",
        "monthly_price_cents": 9900,
        "yearly_price_cents": 99000,
        "monthly_backtest_limit": 500,
        "max_strategies": 100,
        "max_concurrent_tasks": 10,
        "max_data_points": 1000000,
        "stripe_product_id": "prod_pro_placeholder",
        "stripe_monthly_price_id": "price_pro_monthly_placeholder",
        "stripe_yearly_price_id": "price_pro_yearly_placeholder",
        "features": ["实时数据", "高级回测", "API 访问"],
        "sort_order": 1,
    },
    {
        "name": "enterprise",
        "display_name": "企业版",
        "description": "适合团队和企业级量化研究场景",
        "monthly_price_cents": 49900,
        "yearly_price_cents": 499000,
        "monthly_backtest_limit": -1,
        "max_strategies": -1,
        "max_concurrent_tasks": 50,
        "max_data_points": -1,
        "stripe_product_id": "prod_enterprise_placeholder",
        "stripe_monthly_price_id": "price_enterprise_monthly_placeholder",
        "stripe_yearly_price_id": "price_enterprise_yearly_placeholder",
        "features": ["实时数据", "高级回测", "API 访问", "团队协作", "专属支持"],
        "sort_order": 2,
    },
]
