"""
按使用量计费模型

支持按使用量计费的核心数据结构：
- UsageEvent: 每次操作的计费事件记录
- UserBalance: 用户钱包余额
- BillingRecord: 扣费账单记录
- CommercePricingRule: 计费规则配置
"""

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    BOOLEAN,
    BigInteger,
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
)
from sqlalchemy.orm import relationship

from app.core.database import Base


def generate_uuid() -> str:
    """生成模型主键。"""
    return str(uuid.uuid4())


def utcnow() -> datetime:
    """返回无时区 UTC 时间。"""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class EventType(str, Enum):
    """计费事件类型枚举。"""
    
    BACKTEST_BASIC = "backtest_basic"        # 基础回测
    BACKTEST_REALTIME = "backtest_realtime"  # 实时回测
    BACKTEST_ADVANCED = "backtest_advanced"  # 高级策略回测
    API_CALL = "api_call"                    # API 调用
    DATA_DOWNLOAD = "data_download"          # 数据下载
    REALTIME_QUOTE = "realtime_quote"        # 实时行情
    MODEL_TRAINING = "model_training"        # 模型训练
    OPTIMIZATION = "optimization"            # 超参数优化


class ChargeStatus(str, Enum):
    """扣费状态。"""
    
    PENDING = "pending"      # 待扣费
    CHARGED = "charged"      # 已扣费
    FAILED = "failed"        # 扣费失败
    REFUNDED = "refunded"    # 已退款
    CREDITED = "credited"    # 已抵扣（赠额）


class BillingRecord(Base):
    """扣费账单记录表。"""
    
    __tablename__ = "billing_records"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    
    # 关联的用量事件
    usage_event_id = Column(
        String(36), ForeignKey("usage_events.id", ondelete="CASCADE"), nullable=False
    )
    
    # 扣费信息
    amount_cents = Column(Integer, nullable=False)  # 扣费金额（分）
    charge_status = Column(
        String(20), default="pending", nullable=False
    )  # pending/charged/failed/refunded
    
    # 支付方式（future）
    payment_method = Column(String(50), nullable=True)  # credit_card, alipay, wechat
    
    # 余额快照（扣费前的余额）
    balance_before_cents = Column(Integer, nullable=True)
    balance_after_cents = Column(Integer, nullable=True)
    
    # 备注
    remark = Column(Text, nullable=True)
    
    # 时间戳
    charged_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=utcnow, nullable=False, index=True)
    
    # 关联关系：与 User 一对多；与 UsageEvent 为单向关系（不配对 back_populates ——
    # 两表互为外键时配对关系会触发 SQLAlchemy "same direction" 冲突；
    # service 层全部走 ID 查询，无需 ORM 双向导航）
    user = relationship("User", back_populates="billing_records")
    usage_event = relationship("UsageEvent", foreign_keys=[usage_event_id])
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为 API 可直接返回的字典。"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "usage_event_id": self.usage_event_id,
            "amount_cents": self.amount_cents,
            "amount_yuan": self.amount_cents / 100.0,
            "charge_status": self.charge_status,
            "payment_method": self.payment_method,
            "balance_before_cents": self.balance_before_cents,
            "balance_after_cents": self.balance_after_cents,
            "remark": self.remark,
            "charged_at": self.charged_at.isoformat() if self.charged_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }



class UsageEvent(Base):
    """用量事件记录表 - 每次触发计费的原始记录。"""
    
    __tablename__ = "usage_events"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    
    # 事件类型
    event_type = Column(String(50), nullable=False, index=True)  # backtest_basic, api_call 等
    
    # 费用信息
    unit_cost = Column(Numeric(10, 2), nullable=False)  # 单价（元）
    quantity = Column(Integer, nullable=False, default=1)  # 消耗次数/数量
    total_cost = Column(Numeric(10, 2), nullable=False)  # 总费用（元）
    
    # 事件详情（JSON 格式，存储具体业务数据）
    # 例如：backtest 记录任务 ID，API 记录接口路径
    metadata_json = Column("metadata", Text, nullable=True)
    
    # 关联的计费记录
    billing_record_id = Column(
        String(36), ForeignKey("billing_records.id", ondelete="SET NULL"), nullable=True
    )
    
    # 时间戳
    created_at = Column(DateTime, default=utcnow, nullable=False, index=True)
    
    # 关联关系：与 User 一对多；与 BillingRecord 为单向关系（不配对 back_populates ——
    # 两表互为外键时配对关系会触发 SQLAlchemy "same direction" 冲突，
    # 且 service 层全部走 ID 查询，无需 ORM 双向导航）
    user = relationship("User", back_populates="usage_events")
    billing_record = relationship("BillingRecord", foreign_keys=[billing_record_id])
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为 API 可直接返回的字典。"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "event_type": self.event_type,
            "unit_cost": float(self.unit_cost),
            "quantity": self.quantity,
            "total_cost": float(self.total_cost),
            "metadata": self.metadata_json,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class UserBalance(Base):
    """用户钱包余额表。"""
    
    __tablename__ = "user_balances"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, unique=True
    )
    
    # 余额（分），避免浮点数精度问题
    balance_cents = Column(Integer, default=0, nullable=False)
    credited_cents = Column(Integer, default=0, nullable=False)  # 赠额（不可提现）
    
    # 累计消费统计
    total_spent_cents = Column(Integer, default=0, nullable=False)
    
    # 更新时间
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow, nullable=False)
    
    # 关联关系
    user = relationship("User", back_populates="balance")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为 API 可直接返回的字典。"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "balance_cents": self.balance_cents,
            "balance_yuan": self.balance_cents / 100.0,
            "credited_cents": self.credited_cents,
            "total_spent_cents": self.total_spent_cents,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class CommercePricingRule(Base):
    """计费规则配置表。"""
    
    __tablename__ = "commerce_pricing_rules"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    
    # 事件类型
    event_type = Column(String(50), nullable=False, unique=True, index=True)
    
    # 定价
    base_price_cents = Column(Integer, nullable=False)  # 基础价格（分）
    discounted_price_cents = Column(Integer, nullable=True)  # 折扣价（分）
    
    # 计费参数
    unit_name = Column(String(50), nullable=True)  # 单位名：次、条、分钟
    description = Column(Text, nullable=True)
    
    # 生效控制
    is_active = Column(Boolean, default=True, nullable=False)
    effective_from = Column(DateTime, nullable=True)
    effective_until = Column(DateTime, nullable=True)
    
    # 套餐优惠（JSON 格式）
    # {"pro": 0.7, "enterprise": 0.5} 表示 Pro 7 折，Enterprise 5 折
    tier_discounts_json = Column("tier_discounts", Text, nullable=True)
    
    # 创建/更新时间
    created_at = Column(DateTime, default=utcnow, nullable=False)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow, nullable=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为 API 可直接返回的字典。"""
        import json
        
        return {
            "id": self.id,
            "event_type": self.event_type,
            "base_price_cents": self.base_price_cents,
            "base_price_yuan": self.base_price_cents / 100.0,
            "discounted_price_cents": self.discounted_price_cents,
            "discounted_price_yuan": self.discounted_price_cents / 100.0
            if self.discounted_price_cents
            else None,
            "unit_name": self.unit_name,
            "description": self.description,
            "is_active": self.is_active,
            "effective_from": self.effective_from.isoformat() if self.effective_from else None,
            "effective_until": self.effective_until.isoformat()
            if self.effective_until
            else None,
            "tier_discounts": json.loads(self.tier_discounts_json)
            if self.tier_discounts_json
            else {},
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


# ──────────────────────────────────────────────────────────────
# 默认计费规则（MVP 版本）
# ──────────────────────────────────────────────────────────────

DEFAULT_PRICING_RULES: List[Dict[str, Any]] = [
    {
        "event_type": EventType.BACKTEST_BASIC.value,
        "base_price_cents": 50,  # ¥0.5/次
        "unit_name": "次",
        "description": "使用历史数据进行基础回测",
        "is_active": True,
        "tier_discounts": {"pro": "0.6", "enterprise": "0.3"},  # Pro 6 折，Enterprise 3 折
    },
    {
        "event_type": EventType.BACKTEST_REALTIME.value,
        "base_price_cents": 200,  # ¥2/次
        "unit_name": "次",
        "description": "结合实时行情数据进行回测",
        "is_active": True,
        "tier_discounts": {"pro": "0.6", "enterprise": "0.3"},
    },
    {
        "event_type": EventType.BACKTEST_ADVANCED.value,
        "base_price_cents": 100,  # ¥1/次
        "unit_name": "次",
        "description": "多因子/深度学习等高级策略回测",
        "is_active": True,
        "tier_discounts": {"pro": "0.7", "enterprise": "0.5"},
    },
    {
        "event_type": EventType.API_CALL.value,
        "base_price_cents": 1,  # ¥0.01/次
        "unit_name": "次",
        "description": "公共 API 调用",
        "is_active": True,
        "tier_discounts": {"pro": "0.5", "enterprise": "0.1"},
    },
    {
        "event_type": EventType.DATA_DOWNLOAD.value,
        "base_price_cents": 5,  # ¥0.05/次
        "unit_name": "次",
        "description": "数据文件下载（CSV/Excel）",
        "is_active": True,
        "tier_discounts": {"pro": "0.6", "enterprise": "0.3"},
    },
    {
        "event_type": EventType.REALTIME_QUOTE.value,
        "base_price_cents": 5,  # ¥0.05/次
        "unit_name": "次",
        "description": "实时行情推送",
        "is_active": True,
        "tier_discounts": {"pro": "0.6", "enterprise": "0.3"},
    },
]
