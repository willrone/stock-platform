"""计费 API 路由。

套餐列表和 Stripe webhook 分别暴露在公开路由；订阅、Checkout、Portal 和
用量接口挂在受保护路由下，由现有 JWT/X-User-ID 依赖完成认证。
"""

from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from loguru import logger
from pydantic import BaseModel, Field
from sqlalchemy import func

from app.api.v1.dependencies import get_current_user
from app.core.database import SessionLocal
from app.models.subscription_models import (
    SubscriptionPlan,
    UsageRecord,
    UserSubscription,
)
from app.models.user_models import User
from app.services.billing.stripe_service import StripeService

public_router = APIRouter(prefix="/billing", tags=["计费"])
router = APIRouter(prefix="/billing", tags=["计费"])
webhook_router = APIRouter(prefix="/billing", tags=["Stripe Webhook"])


class CheckoutRequest(BaseModel):
    """Checkout 参数。"""

    plan_id: str = Field(..., min_length=1)
    interval: str = Field(default="monthly", pattern="^(monthly|yearly)$")


class UsageRecordRequest(BaseModel):
    """内部用量记录参数。"""

    usage_type: str = Field(..., min_length=1, max_length=50)
    quantity: int = Field(default=1, ge=1)
    metadata: Optional[Dict[str, Any]] = None


def _get_user(session, user_id: str) -> User:
    user = session.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="用户不存在")
    return user


@public_router.get("/plans", summary="获取可用套餐")
def get_plans() -> list[Dict[str, Any]]:
    """公开返回当前启用的套餐，不暴露 Stripe 私密配置。"""
    session = SessionLocal()
    try:
        plans = (
            session.query(SubscriptionPlan)
            .filter(SubscriptionPlan.is_active.is_(True))
            .order_by(SubscriptionPlan.sort_order, SubscriptionPlan.created_at)
            .all()
        )
        return [plan.to_dict() for plan in plans]
    finally:
        session.close()


@router.get("/subscription", summary="获取当前用户订阅")
def get_subscription(user_id: str = Depends(get_current_user)) -> Dict[str, Any]:
    session = SessionLocal()
    try:
        user = _get_user(session, user_id)
        subscription = (
            session.query(UserSubscription)
            .filter(UserSubscription.user_id == user.id)
            .order_by(UserSubscription.created_at.desc())
            .first()
        )
        plan = (
            session.get(SubscriptionPlan, subscription.plan_id)
            if subscription
            else None
        )
        return {
            "subscription": subscription.to_dict() if subscription else None,
            "plan": plan.to_dict() if plan else None,
            "subscription_tier": user.subscription_tier,
        }
    finally:
        session.close()


@router.post("/checkout", summary="创建 Stripe Checkout 会话")
def create_checkout(
    request_data: CheckoutRequest,
    user_id: str = Depends(get_current_user),
) -> Dict[str, str]:
    session = SessionLocal()
    try:
        user = _get_user(session, user_id)
        result = StripeService(session).create_checkout_session(
            user, request_data.plan_id, request_data.interval
        )
        return {"url": result["url"]}
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.warning("创建 Checkout 失败: user_id={}, error={}", user_id, exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.opt(exception=True).error(
            "创建 Checkout 未预期失败: user_id={}", user_id
        )
        raise HTTPException(status_code=502, detail="Stripe Checkout 创建失败") from exc
    finally:
        session.close()


@router.post("/portal", summary="创建 Stripe 客户门户会话")
def create_portal(user_id: str = Depends(get_current_user)) -> Dict[str, str]:
    session = SessionLocal()
    try:
        user = _get_user(session, user_id)
        result = StripeService(session).create_portal_session(user)
        return {"url": result["url"]}
    except HTTPException:
        raise
    except RuntimeError as exc:
        logger.warning("创建客户门户失败: user_id={}, error={}", user_id, exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.opt(exception=True).error("创建客户门户未预期失败: user_id={}", user_id)
        raise HTTPException(status_code=502, detail="Stripe 客户门户创建失败") from exc
    finally:
        session.close()


@webhook_router.post(
    "/webhook", status_code=status.HTTP_200_OK, summary="Stripe webhook"
)
async def stripe_webhook(
    request: Request,
    stripe_signature: Optional[str] = Header(default=None, alias="Stripe-Signature"),
) -> Dict[str, Any]:
    """Stripe webhook 端点，不使用用户认证，仅依赖 Stripe 签名。"""
    if not stripe_signature:
        raise HTTPException(status_code=400, detail="缺少 Stripe-Signature 请求头")
    payload = await request.body()
    try:
        return StripeService().handle_webhook(payload, stripe_signature)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.opt(exception=True).error("处理 Stripe webhook 失败")
        raise HTTPException(status_code=500, detail="Stripe webhook 处理失败") from exc


@router.get("/usage", summary="获取本月用量")
def get_usage(user_id: str = Depends(get_current_user)) -> Dict[str, Any]:
    session = SessionLocal()
    try:
        _get_user(session, user_id)
        now = datetime.utcnow()
        month_start = datetime(now.year, now.month, 1)
        rows = (
            session.query(
                UsageRecord.usage_type,
                func.coalesce(func.sum(UsageRecord.quantity), 0).label("quantity"),
            )
            .filter(
                UsageRecord.user_id == user_id,
                UsageRecord.created_at >= month_start,
            )
            .group_by(UsageRecord.usage_type)
            .all()
        )
        usage = {usage_type: int(quantity) for usage_type, quantity in rows}
        return {
            "year": now.year,
            "month": now.month,
            "usage": usage,
            "total": sum(usage.values()),
        }
    finally:
        session.close()


@router.post("/usage/record", summary="记录用量")
def record_usage(
    request_data: UsageRecordRequest,
    user_id: str = Depends(get_current_user),
) -> Dict[str, Any]:
    session = SessionLocal()
    try:
        _get_user(session, user_id)
        record = UsageRecord(
            user_id=user_id,
            usage_type=request_data.usage_type,
            quantity=request_data.quantity,
            metadata_json=request_data.metadata,
        )
        session.add(record)
        session.commit()
        session.refresh(record)
        return record.to_dict()
    except HTTPException:
        raise
    except Exception as exc:
        session.rollback()
        logger.opt(exception=True).error(
            "记录用量失败: user_id={}, usage_type={}", user_id, request_data.usage_type
        )
        raise HTTPException(status_code=500, detail="记录用量失败") from exc
    finally:
        session.close()


__all__ = ["public_router", "router", "webhook_router"]
