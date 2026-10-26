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
from enum import Enum
from typing import Any, Dict, List

from sqlalchemy import (
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

    BACKTEST_BASIC = "backtest_basic"  # 基础回测
    BACKTEST_REALTIME = "backtest_realtime"  # 实时回测
    BACKTEST_ADVANCED = "backtest_advanced"  # 高级策略回测
    API_CALL = "api_call"  # API 调用
    DATA_DOWNLOAD = "data_download"  # 数据下载
    REALTIME_QUOTE = "realtime_quote"  # 实时行情
    MODEL_TRAINING = "model_training"  # 模型训练
    OPTIMIZATION = "optimization"  # 超参数优化


class ChargeStatus(str, Enum):
    """扣费状态。"""

    PENDING = "pending"  # 待扣费
    CHARGED = "charged"  # 已扣费
    FAILED = "failed"  # 扣费失败
    REFUNDED = "refunded"  # 已退款
    CREDITED = "credited"  # 已抵扣（赠额）


class BillingRecord(Base):
    """扣费账单记录表。"""

    __tablename__ = "billing_records"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # 关联的用量事件
    # ⚠️ 2026-10-27 改为**可空**：本表兼作资金流水，充值到账记录
    # （charge_status='credited'）没有对应的用量事件。
    # 之前是 NOT NULL → 管理员确认到账必然 IntegrityError 500
    # （迁移：commerce_004_billing_usage_event_nullable）
    usage_event_id = Column(
        String(36), ForeignKey("usage_events.id", ondelete="CASCADE"), nullable=True
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
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # 事件类型
    event_type = Column(
        String(50), nullable=False, index=True
    )  # backtest_basic, api_call 等

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
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
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
            "discounted_price_yuan": (
                self.discounted_price_cents / 100.0
                if self.discounted_price_cents
                else None
            ),
            "unit_name": self.unit_name,
            "description": self.description,
            "is_active": self.is_active,
            "effective_from": (
                self.effective_from.isoformat() if self.effective_from else None
            ),
            "effective_until": (
                self.effective_until.isoformat() if self.effective_until else None
            ),
            "tier_discounts": (
                json.loads(self.tier_discounts_json) if self.tier_discounts_json else {}
            ),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class PaymentOrderStatus(str, Enum):
    """充值订单状态。"""

    PENDING = "pending"        # 待支付（已下单，未确认到账）
    PAID = "paid"              # 已到账（管理员确认或支付回调）
    CANCELLED = "cancelled"    # 已取消
    REFUNDED = "refunded"      # 已退款


class PaymentOrder(Base):
    """充值订单表。

    2026-10-25 新增。背景：原 ``POST /api/v1/commerce/deposit`` 直接给调用方
    加余额、无任何支付验证 —— 任何注册用户都能给自己充无限余额。
    对外收费前这是致命的收入漏洞。

    改为**两段式订单制**：
      1. 用户下单 → 订单 ``pending``，返回订单号 + 收款指引
      2. 管理员确认到账（或接了支付渠道后由回调确认）→ 订单 ``paid``，
         此时才写入 ``user_balances.balance_cents`` 并生成 ``billing_records``

    没有自动支付渠道时（Stripe key 为空），走人工对账：
    用户转账后把订单号发给管理员，管理员在后台点"确认到账"即可。
    """

    __tablename__ = "payment_orders"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    order_no = Column(String(32), nullable=False, unique=True, index=True)

    user_id = Column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # 充值金额（分）
    amount_cents = Column(Integer, nullable=False)

    # 订单状态：pending / paid / cancelled / refunded
    status = Column(
        String(20), default=PaymentOrderStatus.PENDING.value, nullable=False
    )

    # 支付渠道：manual_wechat / manual_alipay / stripe / ...
    payment_method = Column(String(50), default="manual", nullable=False)

    # 2026-10-25 新增：本次充值对应的套餐（free / pro / enterprise）
    # 管理员确认到账时据此自动升级 user.subscription_tier —— 否则出现
    # 「收了钱但套餐没变」：用户付费了，付费墙照样把他挡在门外。
    plan_id = Column(String(50), nullable=True)

    # 用户备注（如"已微信转账"）
    user_remark = Column(Text, nullable=True)

    # 管理员备注 / 到账凭证说明
    admin_remark = Column(Text, nullable=True)

    # 管理员操作信息
    confirmed_at = Column(DateTime, nullable=True)
    confirmed_by = Column(String(64), nullable=True)

    # 关联的充值账单（到账后生成的 billing_records.id）
    billing_record_id = Column(String(36), nullable=True)

    created_at = Column(DateTime, default=utcnow, nullable=False)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow, nullable=False)

    user = relationship("User")

    def to_dict(self) -> Dict[str, Any]:
        """转换为 API 可直接返回的字典。"""
        return {
            "id": self.id,
            "order_no": self.order_no,
            "user_id": self.user_id,
            "amount_cents": self.amount_cents,
            "amount_yuan": self.amount_cents / 100.0,
            "status": self.status,
            "payment_method": self.payment_method,
            "plan_id": self.plan_id,
            "user_remark": self.user_remark,
            "admin_remark": self.admin_remark,
            "billing_record_id": self.billing_record_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "confirmed_at": (
                self.confirmed_at.isoformat() if self.confirmed_at else None
            ),
        }


class PaymentConfig(Base):
    """收款配置（单行表，id 固定为 'default'）。

    2026-10-25 新增。没有自动支付渠道时，管理员在这里填自己的收款码 /
    收款账号，前端充值页展示给用户，用户转账后填订单号，管理员确认到账。
    """

    __tablename__ = "payment_configs"

    id = Column(String(36), primary_key=True, default="default")

    # 收款方式名称，如"微信转账"
    method_name = Column(String(50), default="", nullable=False)

    # 收款账号（微信号 / 支付宝账号）
    payee_account = Column(String(100), default="", nullable=False)

    # 收款二维码图片 URL（前端展示，用户扫码）
    qr_image_url = Column(String(500), default="", nullable=False)

    # 支付说明
    instructions = Column(Text, nullable=True)

    # 是否开启在线充值（关闭时前端只显示"请联系客服"）
    is_enabled = Column(Boolean, default=True, nullable=False)

    created_at = Column(DateTime, default=utcnow, nullable=False)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow, nullable=False)

    def to_dict(self) -> Dict[str, Any]:
        """转换为 API 可直接返回的字典。"""
        return {
            "method_name": self.method_name,
            "payee_account": self.payee_account,
            "qr_image_url": self.qr_image_url,
            "instructions": self.instructions,
            "is_enabled": self.is_enabled,
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
        "tier_discounts": {
            "pro": "0.6",
            "enterprise": "0.3",
        },  # Pro 6 折，Enterprise 3 折
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
