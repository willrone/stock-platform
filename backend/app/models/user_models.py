"""
用户模型
"""

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, List, Optional

from sqlalchemy import Column, String, DateTime, Boolean, Integer
from sqlalchemy.orm import relationship

from app.core.database import Base

if TYPE_CHECKING:  # 仅类型检查用；运行时由 SQLAlchemy 注册表按字符串解析
    from app.models.commerce_models import BillingRecord, UsageEvent, UserBalance


def generate_uuid() -> str:
    return str(uuid.uuid4())


class User(Base):
    __tablename__ = "users"

    # 类使用 SQLAlchemy 1.x 风格注解（Optional/List 而非 Mapped[]），
    # 声明这些注解不参与映射解析
    __allow_unmapped__ = True

    id = Column(String(36), primary_key=True, default=generate_uuid)
    email = Column(String(255), unique=True, nullable=False, index=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    is_admin = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    reset_token = Column(String(255), nullable=True)
    reset_token_expires_at = Column(DateTime, nullable=True)
    email_verified = Column(Boolean, default=False, nullable=False)
    avatar_url = Column(String(500), nullable=True)
    last_login_at = Column(DateTime, nullable=True)

    # 用量限制
    monthly_backtest_limit = Column(Integer, default=50)
    max_strategies = Column(Integer, default=20)
    max_concurrent_tasks = Column(Integer, default=3)

    # Stripe/Customer ID（将来支付用）
    stripe_customer_id = Column(String(255), nullable=True)
    subscription_tier = Column(String(50), default="free")  # free | pro | enterprise

    # 计费相关关系（与 commerce_models 双向绑定，缺一侧会导致
    # SQLAlchemy mapper 配置失败：'User' has no property 'usage_events'）
    balance: Optional["UserBalance"] = relationship(
        "UserBalance", back_populates="user", uselist=False, lazy="select"
    )
    usage_events: List["UsageEvent"] = relationship(
        "UsageEvent", back_populates="user", lazy="select"
    )
    billing_records: List["BillingRecord"] = relationship(
        "BillingRecord", back_populates="user", lazy="select"
    )

    def __repr__(self):
        return f"<User {self.email}>"
