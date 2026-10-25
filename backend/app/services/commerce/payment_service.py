"""
充值订单服务（两段式订单制）

2026-10-25 新增。解决的问题：
  原 ``CommerceService.deposit()`` 直接给调用方加余额、**无任何支付验证** ——
  任何注册用户都能给自己充无限余额。对外收费前这是致命的收入漏洞。

改为两段式：
  1. ``create_order()``   用户下单 → 订单 pending，返回订单号 + 收款指引
  2. ``confirm_order()``  管理员确认到账 → 订单 paid，余额入账 + 生成账单

没有自动支付渠道时（STRIPE_SECRET_KEY 为空）走人工对账：
  用户扫码转账 → 把订单号告诉管理员 → 管理员在后台点"确认到账"。

同时提供 ``PaymentConfig`` 的读写，用于在设置里维护收款码/账号。
"""

from __future__ import annotations

import random
import string
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import AsyncSessionLocal
from app.models.commerce_models import (
    BillingRecord,
    PaymentConfig,
    PaymentOrder,
    PaymentOrderStatus,
    UserBalance,
)

# 单笔充值上下限（分）
MIN_DEPOSIT_CENTS = 1000        # ¥10
MAX_DEPOSIT_CENTS = 500_000      # ¥5000


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _gen_order_no() -> str:
    """生成人类可读、可口述的订单号（如 PO-8F3K2M9Q）。"""
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # 去掉易混字符
    body = "".join(random.choices(alphabet, k=8))
    return f"PO-{body}"


class PaymentService:
    """充值订单 + 收款配置服务。"""

    def __init__(self, db: Optional[AsyncSession] = None) -> None:
        self.db = db

    async def _session(self) -> AsyncSession:
        return self.db or AsyncSessionLocal()

    # ──────────────────────────────────────────────────────
    # 收款配置
    # ──────────────────────────────────────────────────────

    async def get_config(self) -> Dict[str, Any]:
        """读取收款配置（不存在则返回默认空配置）。"""
        db = await self._session()
        try:
            row = await db.get(PaymentConfig, "default")
            if row is None:
                return {
                    "method_name": "",
                    "payee_account": "",
                    "qr_image_url": "",
                    "instructions": "",
                    "is_enabled": False,
                }
            return row.to_dict()
        finally:
            if not self.db:
                await db.close()

    async def update_config(
        self,
        method_name: str,
        payee_account: str,
        qr_image_url: str = "",
        instructions: str = "",
        is_enabled: bool = True,
    ) -> Dict[str, Any]:
        """更新收款配置（管理员操作）。"""
        db = await self._session()
        try:
            row = await db.get(PaymentConfig, "default")
            if row is None:
                row = PaymentConfig(id="default")
                db.add(row)
            row.method_name = method_name
            row.payee_account = payee_account
            row.qr_image_url = qr_image_url
            row.instructions = instructions
            row.is_enabled = is_enabled
            await db.commit()
            await db.refresh(row)
            return row.to_dict()
        finally:
            if not self.db:
                await db.close()

    # ──────────────────────────────────────────────────────
    # 订单
    # ──────────────────────────────────────────────────────

    async def create_order(
        self,
        user_id: str,
        amount_cents: int,
        remark: str = "",
        payment_method: str = "manual",
        plan_id: Optional[str] = None,
    ) -> PaymentOrder:
        """创建充值订单（pending）。**不加分**。

        Args:
            plan_id: 本次充值对应的套餐（free / pro / enterprise）。
                管理员确认到账时会据此自动升级用户套餐。
        """
        if amount_cents < MIN_DEPOSIT_CENTS:
            raise ValueError(f"充值金额不得低于 ¥{MIN_DEPOSIT_CENTS/100:.0f}")
        if amount_cents > MAX_DEPOSIT_CENTS:
            raise ValueError(f"单笔充值不得高于 ¥{MAX_DEPOSIT_CENTS/100:.0f}")

        db = await self._session()
        try:
            # 订单号唯一性重试
            order_no = _gen_order_no()
            for _ in range(5):
                exists = await db.scalar(
                    select(PaymentOrder.id).where(PaymentOrder.order_no == order_no)
                )
                if not exists:
                    break
                order_no = _gen_order_no()

            order = PaymentOrder(
                order_no=order_no,
                user_id=user_id,
                amount_cents=amount_cents,
                status=PaymentOrderStatus.PENDING.value,
                payment_method=payment_method,
                plan_id=plan_id or None,
                user_remark=remark or None,
            )
            db.add(order)
            await db.commit()
            await db.refresh(order)
            logger.info(
                f"创建充值订单 {order.order_no}: user={user_id} "
                f"金额 ¥{order.amount_cents/100:.2f} plan={plan_id or '-'}"
            )
            return order
        finally:
            if not self.db:
                await db.close()

    async def get_order(self, order_id: str) -> Optional[PaymentOrder]:
        """按订单 ID 或订单号查订单。"""
        db = await self._session()
        try:
            row = await db.get(PaymentOrder, order_id)
            if row is not None:
                return row
            return await db.scalar(
                select(PaymentOrder).where(PaymentOrder.order_no == order_id)
            )
        finally:
            if not self.db:
                await db.close()

    async def list_orders(
        self,
        user_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """列订单（user_id=None 时管理员看全部）。"""
        db = await self._session()
        try:
            stmt = select(PaymentOrder).order_by(PaymentOrder.created_at.desc())
            if user_id:
                stmt = stmt.where(PaymentOrder.user_id == user_id)
            if status:
                stmt = stmt.where(PaymentOrder.status == status)
            rows = (await db.execute(stmt.limit(limit))).scalars().all()
            return [r.to_dict() for r in rows]
        finally:
            if not self.db:
                await db.close()

    async def confirm_order(
        self,
        order_id: str,
        admin_id: str = "",
        admin_remark: str = "",
    ) -> Tuple[Optional[PaymentOrder], Optional[str]]:
        """管理员确认到账 → 订单 paid，余额入账。

        Returns:
            (订单, 错误信息) —— 出错时订单为 None
        """
        db = await self._session()
        try:
            order = await db.get(PaymentOrder, order_id)
            if order is None:
                order = await db.scalar(
                    select(PaymentOrder).where(PaymentOrder.order_no == order_id)
                )
            if order is None:
                return None, "订单不存在"
            if order.status == PaymentOrderStatus.PAID.value:
                return None, "订单已确认到账，请勿重复操作"
            if order.status == PaymentOrderStatus.CANCELLED.value:
                return None, "订单已取消，无法确认"

            # 幂等保护：并发确认时靠行级状态判断
            if order.status != PaymentOrderStatus.PENDING.value:
                return None, f"订单状态异常（{order.status}）"

            balance = await db.scalar(
                select(UserBalance).where(UserBalance.user_id == order.user_id)
            )
            if balance is None:
                balance = UserBalance(user_id=order.user_id, balance_cents=0)
                db.add(balance)
                await db.flush()

            before = balance.balance_cents
            balance.balance_cents += order.amount_cents
            balance.updated_at = _utcnow()

            record = BillingRecord(
                user_id=order.user_id,
                usage_event_id=None,
                amount_cents=order.amount_cents,
                charge_status="credited",
                balance_before_cents=before,
                balance_after_cents=balance.balance_cents,
                charged_at=_utcnow(),
                remark=f"充值订单 {order.order_no}",
            )
            db.add(record)
            await db.flush()

            order.status = PaymentOrderStatus.PAID.value
            order.confirmed_at = _utcnow()
            order.confirmed_by = admin_id or "admin"
            order.admin_remark = admin_remark or None
            order.billing_record_id = record.id

            # 2026-10-25 新增：确认到账时自动升级套餐
            # 背景：原来确认到账只加余额，不改 subscription_tier → 出现
            # 「收了钱但套餐没变」，用户付费了付费墙照样把他挡住。
            upgraded_to = None
            # 用 getattr 拿实例值（Pyright 把 order.plan_id 当类属性 Column 看待）
            _plan_id = getattr(order, "plan_id", None)
            if _plan_id:
                from app.models.user_models import User

                user = await db.get(User, order.user_id)
                if user is not None:
                    current = (user.subscription_tier or "free").lower()
                    target = str(_plan_id).lower()
                    # 只升不降（避免误操作把付费用户降级）
                    _rank = {"free": 0, "pro": 1, "enterprise": 2}
                    if _rank.get(target, 0) > _rank.get(current, 0):
                        # setattr 绕过 Pyright 的 Column[str] 类型误报
                        setattr(user, "subscription_tier", target)
                        upgraded_to = target
                        logger.info(
                            f"用户 {user.id} 套餐升级: {current} → {target} "
                            f"（订单 {order.order_no}）"
                        )
                    else:
                        upgraded_to = current

            await db.commit()
            await db.refresh(order)
            logger.info(
                f"充值订单 {order.order_no} 已确认: +¥{order.amount_cents/100:.2f} "
                f"→ user={order.user_id} 余额 ¥{balance.balance_cents/100:.2f}"
                + (f" 套餐→{upgraded_to}" if upgraded_to else "")
            )
            return order, None
        finally:
            if not self.db:
                await db.close()

    async def cancel_order(self, order_id: str, admin_remark: str = "") -> Tuple[Optional[PaymentOrder], Optional[str]]:
        """取消订单（管理员操作）。"""
        db = await self._session()
        try:
            order = await db.get(PaymentOrder, order_id)
            if order is None:
                order = await db.scalar(
                    select(PaymentOrder).where(PaymentOrder.order_no == order_id)
                )
            if order is None:
                return None, "订单不存在"
            if order.status == PaymentOrderStatus.PAID.value:
                return None, "订单已到账，不能取消（如需退款请用退款接口）"
            order.status = PaymentOrderStatus.CANCELLED.value
            order.admin_remark = admin_remark or order.admin_remark
            await db.commit()
            await db.refresh(order)
            return order, None
        finally:
            if not self.db:
                await db.close()
