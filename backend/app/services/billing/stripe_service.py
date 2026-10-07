"""Stripe 计费服务。

Stripe SDK 是可选依赖：未安装时模块仍可被导入，只有真正调用 Stripe
API 时才会返回明确错误，方便本地开发和无支付环境运行。
"""

from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Iterator, Optional

from loguru import logger
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import SessionLocal
from app.models.subscription_models import SubscriptionPlan, UserSubscription
from app.models.user_models import User

try:  # Stripe 在部分开发/测试环境中不是必装依赖
    import stripe as stripe_sdk
except ImportError:  # pragma: no cover - 由运行环境决定
    stripe_sdk = None

# 保留一个易于测试替换的模块级名称。
stripe = stripe_sdk


class StripeService:
    """封装客户、Checkout、Portal 和 webhook 同步逻辑。"""

    def __init__(self, db: Optional[Session] = None) -> None:
        self.db = db

    @contextmanager
    def _session_scope(self) -> Iterator[Session]:
        """使用调用方会话，或为独立服务调用创建短生命周期会话。"""
        if self.db is not None:
            yield self.db
            return

        session = SessionLocal()
        try:
            yield session
        finally:
            session.close()

    @staticmethod
    def _value(obj: Any, key: str, default: Any = None) -> Any:
        """同时兼容 dict、StripeObject 和普通对象。"""
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(key, default)
        try:
            return getattr(obj, key)
        except AttributeError:
            return default

    @staticmethod
    def _as_datetime(timestamp: Any) -> Optional[datetime]:
        """把 Stripe Unix 时间戳转换为项目使用的无时区 UTC 时间。"""
        if timestamp is None:
            return None
        if isinstance(timestamp, datetime):
            return timestamp.replace(tzinfo=None)
        try:
            return datetime.utcfromtimestamp(float(timestamp))
        except (TypeError, ValueError, OverflowError):
            return None

    @staticmethod
    def _require_stripe() -> Any:
        """检查 SDK 和密钥，避免导入阶段因可选依赖失败。"""
        if stripe is None:
            raise RuntimeError("Stripe SDK 未安装，请安装 stripe 包后再启用支付功能")
        if not settings.STRIPE_SECRET_KEY:
            raise RuntimeError("未配置 STRIPE_SECRET_KEY")
        stripe.api_key = settings.STRIPE_SECRET_KEY
        return stripe

    @staticmethod
    def _get_plan(session: Session, plan_id: str) -> SubscriptionPlan:
        plan = (
            session.query(SubscriptionPlan)
            .filter(
                SubscriptionPlan.id == plan_id,
                SubscriptionPlan.is_active.is_(True),
            )
            .first()
        )
        if plan is None:
            raise ValueError("套餐不存在或已停用")
        return plan

    @staticmethod
    def _get_user(session: Session, user_id: str) -> User:
        user = session.get(User, user_id)
        if user is None:
            raise ValueError("用户不存在")
        return user

    @staticmethod
    def _apply_plan_to_user(user: User, plan: SubscriptionPlan) -> None:
        """同步现有 User 限额字段，保持历史代码读取方式不变。"""
        user.subscription_tier = plan.name
        user.monthly_backtest_limit = plan.monthly_backtest_limit
        user.max_strategies = plan.max_strategies
        user.max_concurrent_tasks = plan.max_concurrent_tasks

    def create_customer(self, user: User) -> str:
        """创建 Stripe Customer，并回写 User.stripe_customer_id。"""
        with self._session_scope() as session:
            managed_user = self._get_user(session, user.id)
            if managed_user.stripe_customer_id:
                return managed_user.stripe_customer_id

            stripe_api = self._require_stripe()
            customer = stripe_api.Customer.create(
                email=managed_user.email,
                name=managed_user.username,
                metadata={"user_id": managed_user.id},
            )
            customer_id = self._value(customer, "id")
            if not customer_id:
                raise RuntimeError("Stripe 创建客户未返回 customer ID")
            managed_user.stripe_customer_id = str(customer_id)
            session.commit()
            logger.info(
                "Stripe 客户创建成功: user_id={}, customer_id={}",
                managed_user.id,
                customer_id,
            )
            return str(customer_id)

    def create_checkout_session(
        self, user: User, plan_id: str, interval: str = "monthly"
    ) -> Dict[str, Any]:
        """创建订阅 Checkout 会话。"""
        if interval not in {"monthly", "yearly"}:
            raise ValueError("billing interval 必须是 monthly 或 yearly")

        with self._session_scope() as session:
            managed_user = self._get_user(session, user.id)
            plan = self._get_plan(session, plan_id)
            price_id = (
                plan.stripe_monthly_price_id
                if interval == "monthly"
                else plan.stripe_yearly_price_id
            )
            if not price_id or "placeholder" in price_id:
                raise ValueError("该套餐尚未配置有效的 Stripe Price ID")

            customer_id = self.create_customer(managed_user)
            stripe_api = self._require_stripe()
            checkout = stripe_api.checkout.Session.create(
                customer=customer_id,
                mode="subscription",
                line_items=[{"price": price_id, "quantity": 1}],
                success_url="http://localhost:3000/billing/success?session_id={CHECKOUT_SESSION_ID}",
                cancel_url="http://localhost:3000/billing/cancel",
                metadata={
                    "user_id": managed_user.id,
                    "plan_id": plan.id,
                    "billing_interval": interval,
                },
                subscription_data={
                    "metadata": {
                        "user_id": managed_user.id,
                        "plan_id": plan.id,
                    }
                },
            )
            session_id = self._value(checkout, "id")
            url = self._value(checkout, "url")
            if not url:
                raise RuntimeError("Stripe Checkout 未返回会话 URL")
            logger.info(
                "Stripe Checkout 创建成功: user_id={}, session_id={}",
                managed_user.id,
                session_id,
            )
            return {"id": session_id, "url": url}

    def create_portal_session(self, user: User) -> Dict[str, Any]:
        """创建 Stripe 客户门户会话。"""
        customer_id = self.create_customer(user)
        stripe_api = self._require_stripe()
        portal = stripe_api.billing_portal.Session.create(
            customer=customer_id,
            return_url="http://localhost:3000/settings/billing",
        )
        url = self._value(portal, "url")
        if not url:
            raise RuntimeError("Stripe Customer Portal 未返回会话 URL")
        return {"id": self._value(portal, "id"), "url": url}

    def handle_webhook(self, payload: bytes, sig_header: str) -> Dict[str, Any]:
        """验证并处理 Stripe webhook 事件。"""
        stripe_api = self._require_stripe()
        if not settings.STRIPE_WEBHOOK_SECRET:
            raise ValueError("未配置 STRIPE_WEBHOOK_SECRET")
        try:
            event = stripe_api.Webhook.construct_event(
                payload, sig_header, settings.STRIPE_WEBHOOK_SECRET
            )
        except Exception as exc:
            logger.warning("Stripe webhook 签名验证失败: {}", exc)
            raise ValueError("无效的 Stripe webhook 签名") from exc

        event_type = self._value(event, "type")
        event_data = self._value(event, "data", {})
        event_object = self._value(event_data, "object", {})

        with self._session_scope() as session:
            if event_type == "checkout.session.completed":
                self._handle_checkout_completed(session, event_object)
            elif event_type == "customer.subscription.updated":
                self._handle_subscription_updated(session, event_object)
            elif event_type == "customer.subscription.deleted":
                self._handle_subscription_deleted(session, event_object)
            elif event_type == "invoice.payment_failed":
                self._handle_payment_failed(session, event_object)
            else:
                logger.debug("忽略未处理的 Stripe webhook 事件: {}", event_type)
                return {"status": "ignored", "event_type": event_type}
            session.commit()

        logger.info("Stripe webhook 处理成功: {}", event_type)
        return {"status": "processed", "event_type": event_type}

    def _find_plan_by_price(
        self, session: Session, price_id: Optional[str]
    ) -> Optional[SubscriptionPlan]:
        if not price_id:
            return None
        return (
            session.query(SubscriptionPlan)
            .filter(
                (SubscriptionPlan.stripe_monthly_price_id == price_id)
                | (SubscriptionPlan.stripe_yearly_price_id == price_id)
            )
            .first()
        )

    def _find_user_by_customer(
        self, session: Session, customer_id: Optional[str]
    ) -> Optional[User]:
        if not customer_id:
            return None
        return (
            session.query(User).filter(User.stripe_customer_id == customer_id).first()
        )

    @staticmethod
    def _normalize_interval(interval: Any) -> str:
        """统一 Stripe 的 month/year 与本地 monthly/yearly 表示。"""
        return {"month": "monthly", "year": "yearly"}.get(str(interval), str(interval))

    def _subscription_details(self, subscription: Any) -> Dict[str, Any]:
        items = self._value(subscription, "items", {})
        data = self._value(items, "data", []) or []
        item = data[0] if data else {}
        price = self._value(item, "price", {}) or self._value(subscription, "plan", {})
        recurring = self._value(price, "recurring", {})
        return {
            "price_id": self._value(price, "id"),
            "interval": self._normalize_interval(
                self._value(recurring, "interval")
                or self._value(price, "interval")
                or "monthly"
            ),
        }

    def _handle_checkout_completed(self, session: Session, checkout: Any) -> None:
        metadata = self._value(checkout, "metadata", {}) or {}
        user_id = metadata.get("user_id") if isinstance(metadata, dict) else None
        customer_id = self._value(checkout, "customer")
        user = (
            session.get(User, user_id)
            if user_id
            else self._find_user_by_customer(session, customer_id)
        )
        if user is None:
            logger.warning(
                "Checkout 完成但找不到本地用户: user_id={}, customer_id={}",
                user_id,
                customer_id,
            )
            return

        plan_id = metadata.get("plan_id") if isinstance(metadata, dict) else None
        plan = session.get(SubscriptionPlan, plan_id) if plan_id else None
        subscription_id = self._value(checkout, "subscription")
        if plan is None and subscription_id:
            existing = (
                session.query(UserSubscription)
                .filter(UserSubscription.stripe_subscription_id == subscription_id)
                .first()
            )
            plan = session.get(SubscriptionPlan, existing.plan_id) if existing else None
        if plan is None:
            logger.warning("Checkout 完成但找不到本地套餐: plan_id={}", plan_id)
            return

        if customer_id:
            user.stripe_customer_id = str(customer_id)
        self._apply_plan_to_user(user, plan)
        subscription = (
            session.query(UserSubscription)
            .filter(UserSubscription.stripe_subscription_id == subscription_id)
            .first()
            if subscription_id
            else None
        )
        if subscription is None:
            subscription = UserSubscription(
                user_id=user.id,
                plan_id=plan.id,
                stripe_subscription_id=subscription_id,
                stripe_customer_id=customer_id,
                status="active",
                billing_interval=(
                    metadata.get("billing_interval", "monthly")
                    if isinstance(metadata, dict)
                    else "monthly"
                ),
            )
            session.add(subscription)
        else:
            subscription.plan_id = plan.id
            subscription.status = "active"
            subscription.stripe_customer_id = (
                customer_id or subscription.stripe_customer_id
            )

    def _handle_subscription_updated(
        self, session: Session, subscription_data: Any
    ) -> None:
        subscription_id = self._value(subscription_data, "id")
        if not subscription_id:
            return
        customer_id = self._value(subscription_data, "customer")
        row = (
            session.query(UserSubscription)
            .filter(UserSubscription.stripe_subscription_id == subscription_id)
            .first()
        )
        user = self._find_user_by_customer(session, customer_id)
        details = self._subscription_details(subscription_data)
        plan = self._find_plan_by_price(session, details["price_id"])
        if row is None:
            if user is None or plan is None:
                logger.warning(
                    "订阅更新但找不到本地记录: subscription_id={}", subscription_id
                )
                return
            row = UserSubscription(
                user_id=user.id,
                plan_id=plan.id,
                stripe_subscription_id=str(subscription_id),
            )
            session.add(row)
        if user is None:
            user = session.get(User, row.user_id)
        if user is None:
            return

        if plan is not None:
            row.plan_id = plan.id
            self._apply_plan_to_user(user, plan)
        row.stripe_customer_id = customer_id or row.stripe_customer_id
        row.status = self._value(subscription_data, "status", "active")
        row.billing_interval = details["interval"]
        row.current_period_start = self._as_datetime(
            self._value(subscription_data, "current_period_start")
        )
        row.current_period_end = self._as_datetime(
            self._value(subscription_data, "current_period_end")
        )
        row.cancel_at = self._as_datetime(self._value(subscription_data, "cancel_at"))
        row.canceled_at = self._as_datetime(
            self._value(subscription_data, "canceled_at")
        )

    def _handle_subscription_deleted(
        self, session: Session, subscription_data: Any
    ) -> None:
        subscription_id = self._value(subscription_data, "id")
        if not subscription_id:
            return
        row = (
            session.query(UserSubscription)
            .filter(UserSubscription.stripe_subscription_id == subscription_id)
            .first()
        )
        if row is None:
            return
        row.status = "canceled"
        row.canceled_at = (
            self._as_datetime(self._value(subscription_data, "canceled_at"))
            or datetime.utcnow()
        )
        user = session.get(User, row.user_id)
        if user is not None:
            free_plan = (
                session.query(SubscriptionPlan)
                .filter(SubscriptionPlan.name == "free")
                .first()
            )
            if free_plan is not None:
                self._apply_plan_to_user(user, free_plan)

    def _handle_payment_failed(self, session: Session, invoice: Any) -> None:
        subscription_id = self._value(invoice, "subscription")
        customer_id = self._value(invoice, "customer")
        query = session.query(UserSubscription)
        row = (
            query.filter(
                UserSubscription.stripe_subscription_id == subscription_id
            ).first()
            if subscription_id
            else query.filter(
                UserSubscription.stripe_customer_id == customer_id
            ).first()
        )
        if row is not None:
            row.status = "past_due"
            if customer_id:
                row.stripe_customer_id = customer_id


__all__ = ["StripeService"]
