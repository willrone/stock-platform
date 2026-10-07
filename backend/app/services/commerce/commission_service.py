"""
计费与扣费服务

按使用量计费的核心业务逻辑：
- 创建用量事件
- 查询/扣减用户余额
- 生成账单记录
- 计费规则配置管理
"""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from app.core.config import settings
from app.core.database import SessionLocal
from app.models.commerce_models import (
    BillingRecord,
    CommercePricingRule,
    DEFAULT_PRICING_RULES,
    EventType,
    UserBalance,
    UsageEvent,
)
from app.models.user_models import User


class CommerceService:
    """按使用量计费服务。"""
    
    def __init__(self, db: Optional[AsyncSession] = None) -> None:
        self.db = db
    
    # ──────────────────────────────────────────────────────
    # 用量事件
    # ──────────────────────────────────────────────────────
    
    async def record_usage(
        self,
        user_id: str,
        event_type: str,
        quantity: int = 1,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[UsageEvent, Optional[BillingRecord]]:
        """记录一次用量并尝试扣费。
        
        Args:
            user_id: 用户 ID
            event_type: 事件类型（backtest_basic, api_call 等）
            quantity: 消耗数量
            metadata: 事件详情 JSON
            
        Returns:
            (用量事件, 扣费记录)
            - 用量事件必定创建
            - 扣费记录只在余额充足时生成，余额不足则返回 None
        """
        # 1. 获取计费规则
        pricing = await self._get_pricing_rule(event_type)
        if pricing is None:
            # 未知事件类型，不计费
            logger.warning(f"未知的事件类型: {event_type}, 跳过计费")
            event = self._create_event_record(
                user_id, event_type, 0, 0, quantity, metadata
            )
            return event, None
        
        # 2. 计算费用
        unit_cost = pricing.base_price_cents  # 分
        total_cost = unit_cost * quantity
        
        # 3. 创建用量事件
        event = self._create_event_record(
            user_id, event_type, unit_cost, total_cost, quantity, metadata
        )
        
        # 4. 尝试扣费
        billing_record = await self._charge_user(user_id, event, total_cost)
        
        return event, billing_record
    
    def _create_event_record(
        self,
        user_id: str,
        event_type: str,
        unit_cost: int,
        total_cost: int,
        quantity: int,
        metadata: Optional[Dict[str, Any]],
    ) -> UsageEvent:
        """创建用量事件记录（同步，用于内存操作）。"""
        import json
        
        return UsageEvent(
            user_id=user_id,
            event_type=event_type,
            unit_cost=unit_cost,
            quantity=quantity,
            total_cost=total_cost,
            metadata_json=json.dumps(metadata, ensure_ascii=False) if metadata else None,
        )
    
    async def _charge_user(
        self, user_id: str, event: UsageEvent, amount_cents: int
    ) -> Optional[BillingRecord]:
        """尝试从用户余额扣费。
        
        逻辑：
        1. 查询用户余额
        2. 余额充足 → 直接扣减，生成账单（charged）
        3. 余额不足 → 标记为 pending（待缴），允许继续使用
        4. 创建账单记录
        
        Args:
            user_id: 用户 ID
            event: 用量事件
            amount_cents: 扣费金额（分）
            
        Returns:
            账单记录，或 None（创建失败）
        """
        if amount_cents <= 0:
            return None
        
        db = self.db or SessionLocal()
        try:
            # 查询用户余额
            balance = await self._get_or_create_balance(db, user_id)
            
            # 创建账单记录
            balance_before = balance.balance_cents
            
            if balance.balance_cents >= amount_cents:
                # 余额充足，直接扣减
                balance.balance_cents -= amount_cents
                balance.total_spent_cents += amount_cents
                charge_status = "charged"
                balance_after = balance.balance_cents
                charged_at = datetime.now(timezone.utc).replace(tzinfo=None)
                remark = "余额充足，自动扣费成功"
            else:
                # 余额不足，标记为待缴
                charge_status = "pending"
                balance_after = balance.balance_cents
                charged_at = None
                remark = f"余额不足（{balance.balance_cents/100:.2f}元），待充值"
                logger.warning(
                    f"用户 {user_id} 余额不足，"
                    f"本次费用 {amount_cents/100:.2f} 元，"
                    f"当前余额 {balance.balance_cents/100:.2f} 元"
                )
            
            billing_record = BillingRecord(
                user_id=user_id,
                usage_event_id=event.id,
                amount_cents=amount_cents,
                charge_status=charge_status,
                balance_before_cents=balance_before,
                balance_after_cents=balance_after,
                charged_at=charged_at,
                remark=remark,
            )
            
            db.add(billing_record)
            await db.flush()
            
            # 更新关联的用量事件
            event.billing_record_id = billing_record.id
            await db.flush()
            
            # 余额充足时才提交扣减
            if charge_status == "charged":
                await db.flush()
            
            logger.info(
                f"计费记录创建: user_id={user_id}, "
                f"event_type={event.event_type}, "
                f"amount={amount_cents/100:.2f}元, "
                f"status={charge_status}"
            )
            
            return billing_record
            
        except Exception as exc:
            logger.opt(exception=True).error(
                f"扣费失败: user_id={user_id}, error={exc}"
            )
            return None
        finally:
            if not self.db:
                await db.close()
    
    async def _get_or_create_balance(
        self, db: AsyncSession, user_id: str
    ) -> UserBalance:
        """获取用户余额，如果不存在则创建（初始赠送 ¥10）。"""
        result = await db.execute(
            select(UserBalance).where(UserBalance.user_id == user_id)
        )
        balance = result.scalar_one_or_none()
        
        if balance is None:
            balance = UserBalance(
                user_id=user_id,
                balance_cents=1000,  # 赠送 ¥10 初始额度
                credited_cents=1000,  # 标记为赠额
            )
            db.add(balance)
            await db.flush()
            logger.info(f"用户 {user_id} 余额已创建，赠送 ¥10")
        
        return balance
    
    # ──────────────────────────────────────────────────────
    # 计费规则
    # ──────────────────────────────────────────────────────
    
    async def _get_pricing_rule(self, event_type: str) -> Optional[CommercePricingRule]:
        """获取指定事件类型的计费规则。"""
        db = self.db or SessionLocal()
        try:
            result = await db.execute(
                select(CommercePricingRule)
                .where(CommercePricingRule.event_type == event_type)
                .where(CommercePricingRule.is_active.is_(True))
            )
            rule = result.scalar_one_or_none()
            return rule
        finally:
            if not self.db:
                await db.close()
    
    async def get_pricing_rules(
        self, event_type: Optional[str] = None
    ) -> List[CommercePricingRule]:
        """查询计费规则列表。"""
        db = self.db or SessionLocal()
        try:
            query = select(CommercePricingRule).where(CommercePricingRule.is_active.is_(True))
            if event_type:
                query = query.where(CommercePricingRule.event_type == event_type)
            result = await db.execute(query)
            return list(result.scalars().all())
        finally:
            if not self.db:
                await db.close()
    
    # ──────────────────────────────────────────────────────
    # 余额管理
    # ──────────────────────────────────────────────────────
    
    async def get_balance(self, user_id: str) -> Optional[UserBalance]:
        """查询用户余额。"""
        db = self.db or SessionLocal()
        try:
            result = await db.execute(
                select(UserBalance).where(UserBalance.user_id == user_id)
            )
            return result.scalar_one_or_none()
        finally:
            if not self.db:
                await db.close()
    
    async def deposit(
        self, user_id: str, amount_cents: int, remark: str = ""
    ) -> Tuple[UserBalance, BillingRecord]:
        """用户充值。
        
        Args:
            user_id: 用户 ID
            amount_cents: 充值金额（分）
            remark: 充值备注
            
        Returns:
            (更新后的余额, 充值账单记录)
        """
        db = self.db or SessionLocal()
        try:
            balance = await self._get_or_create_balance(db, user_id)
            
            # 创建充值账单记录
            record = BillingRecord(
                user_id=user_id,
                usage_event_id=None,
                amount_cents=amount_cents,
                charge_status="credited",  # 充值状态
                balance_before_cents=balance.balance_cents,
                balance_after_cents=balance.balance_cents + amount_cents,
                charged_at=datetime.now(timezone.utc).replace(tzinfo=None),
                remark=f"充值: {remark}" if remark else "用户充值",
            )
            
            balance.balance_cents += amount_cents
            db.add(record)
            await db.flush()
            
            logger.info(
                f"用户 {user_id} 充值成功: +{amount_cents/100:.2f}元"
            )
            
            return balance, record
        finally:
            if not self.db:
                await db.close()
    
    async def refund(
        self, user_id: str, billing_record_id: str, amount_cents: int
    ) -> Optional[BillingRecord]:
        """退款操作。
        
        Args:
            user_id: 用户 ID
            billing_record_id: 原始账单 ID
            amount_cents: 退款金额（分）
            
        Returns:
            退款账单记录
        """
        db = self.db or SessionLocal()
        try:
            balance = await self._get_or_create_balance(db, user_id)
            
            refund_record = BillingRecord(
                user_id=user_id,
                usage_event_id=None,
                amount_cents=amount_cents,
                charge_status="refunded",
                balance_before_cents=balance.balance_cents,
                balance_after_cents=balance.balance_cents + amount_cents,
                charged_at=datetime.now(timezone.utc).replace(tzinfo=None),
                remark=f"退款: 原始账单 {billing_record_id}",
            )
            
            balance.balance_cents += amount_cents
            db.add(refund_record)
            await db.flush()
            
            logger.info(f"用户 {user_id} 退款成功: +{amount_cents/100:.2f}元")
            
            return refund_record
        finally:
            if not self.db:
                await db.close()
    
    # ──────────────────────────────────────────────────────
    # 用量查询
    # ──────────────────────────────────────────────────────
    
    async def get_usage_summary(
        self, user_id: str, month: Optional[int] = None, year: Optional[int] = None
    ) -> Dict[str, Any]:
        """查询用户当月用量汇总。
        
        Returns:
            {
                "month": 9,
                "year": 2026,
                "total_spent_cents": 1250,
                "total_spent_yuan": 12.50,
                "events": {
                    "backtest_basic": {"count": 5, "total_cost": 250},
                    "api_call": {"count": 100, "total_cost": 100},
                }
            }
        """
        db = self.db or SessionLocal()
        try:
            now = datetime.now(timezone.utc)
            target_month = month or now.month
            target_year = year or now.year
            
            # 当月开始和结束时间
            from datetime import timedelta
            
            month_start = datetime(target_year, target_month, 1, tzinfo=timezone.utc).replace(tzinfo=None)
            
            # 下个月第一天
            next_month = target_month + 1 if target_month < 12 else 1
            next_year = target_year if target_month < 12 else target_year + 1
            month_end = datetime(next_year, next_month, 1, tzinfo=timezone.utc).replace(tzinfo=None)
            
            # 查询当月所有用量事件
            result = await db.execute(
                select(UsageEvent)
                .where(UsageEvent.user_id == user_id)
                .where(UsageEvent.created_at >= month_start)
                .where(UsageEvent.created_at < month_end)
            )
            events = result.scalars().all()
            
            # 按事件类型汇总
            events_summary: Dict[str, Dict[str, Any]] = {}
            total_spent = 0
            
            for event in events:
                event_type = event.event_type
                if event_type not in events_summary:
                    events_summary[event_type] = {
                        "count": 0,
                        "total_cost_cents": 0,
                        "total_cost_yuan": 0,
                    }
                
                events_summary[event_type]["count"] += 1
                events_summary[event_type]["total_cost_cents"] += int(event.total_cost)
                events_summary[event_type]["total_cost_yuan"] = (
                    events_summary[event_type]["total_cost_cents"] / 100.0
                )
                total_spent += int(event.total_cost)
            
            return {
                "month": target_month,
                "year": target_year,
                "total_spent_cents": total_spent,
                "total_spent_yuan": total_spent / 100.0,
                "events": events_summary,
            }
        finally:
            if not self.db:
                await db.close()
    
    async def get_billing_history(
        self, user_id: str, limit: int = 50, offset: int = 0
    ) -> List[BillingRecord]:
        """查询用户账单历史。"""
        db = self.db or SessionLocal()
        try:
            result = await db.execute(
                select(BillingRecord)
                .where(BillingRecord.user_id == user_id)
                .order_by(BillingRecord.created_at.desc())
                .offset(offset)
                .limit(limit)
            )
            return list(result.scalars().all())
        finally:
            if not self.db:
                await db.close()
    
    async def can_proceed(self, user_id: str, event_type: str) -> Tuple[bool, str]:
        """检查用户是否可以执行指定操作（余额检查）。
        
        Returns:
            (是否允许，提示信息)
        """
        balance = await self.get_balance(user_id)
        
        if balance is None or balance.balance_cents <= 0:
            return False, "余额不足，请充值后再试"
        
        # 获取计费规则
        pricing = await self._get_pricing_rule(event_type)
        if pricing is None:
            return True, "允许执行（不计费事件）"
        
        cost_yuan = pricing.base_price_cents / 100.0
        if balance.balance_cents < pricing.base_price_cents:
            return False, f"余额不足，本次操作需要 {cost_yuan} 元，当前余额 {balance.balance_cents/100:.2f} 元"
        
        return True, f"允许执行，本次费用 {cost_yuan} 元"


# ──────────────────────────────────────────────────────────
# 辅助函数（供外部调用）
# ──────────────────────────────────────────────────────────


async def check_can_proceed(user_id: str, event_type: str) -> Tuple[bool, str]:
    """检查用户是否可以执行指定操作（余额检查）。
    
    Returns:
        (是否允许, 提示信息)
    """
    service = CommerceService()
    balance = await service.get_balance(user_id)
    
    if balance is None or balance.balance_cents <= 0:
        return False, "余额不足，请充值后再试"
    
    # 获取计费规则
    pricing = await service._get_pricing_rule(event_type)
    if pricing is None:
        return True, "允许执行（不计费事件）"
    
    cost_yuan = pricing.base_price_cents / 100.0
    if balance.balance_cents < pricing.base_price_cents:
        return False, f"余额不足，本次操作需要 {cost_yuan} 元，当前余额 {balance.balance_cents/100:.2f} 元"
    
    return True, f"允许执行，本次费用 {cost_yuan} 元"


async def seed_default_pricing_rules(db) -> List[CommercePricingRule]:
    """初始化默认计费规则（MVP 版本）。
    
    只在首次运行时创建已有规则。
    """
    rules = []
    for rule_data in DEFAULT_PRICING_RULES:
        event_type = rule_data["event_type"]
        
        # 检查是否已存在
        result = await db.execute(
            select(CommercePricingRule).where(
                CommercePricingRule.event_type == event_type
            )
        )
        existing = result.scalar_one_or_none()
        
        if existing is None:
            rule = CommercePricingRule(
                event_type=event_type,
                base_price_cents=rule_data["base_price_cents"],
                unit_name=rule_data["unit_name"],
                description=rule_data["description"],
                is_active=rule_data["is_active"],
                tier_discounts_json=rule_data.get("tier_discounts"),
            )
            db.add(rule)
            rules.append(rule)
    
    if rules:
        await db.flush()
        logger.info(f"初始化 {len(rules)} 条默认计费规则")
    
    return rules