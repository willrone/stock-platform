"""
按使用量计费 API 路由。

提供：
- 用量记录（回测/API/数据操作计费事件）
- 余额查询/充值/退款
- 计费规则查询
- 用量统计/账单历史
"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app.api.v1.dependencies import get_current_user
from app.core.database import AsyncSessionLocal
from app.services.commerce.commission_service import (
    CommerceService,
    check_can_proceed,
    seed_default_pricing_rules,
)

router = APIRouter(prefix="/commerce", tags=["按量计费"])


# ──────────────────────────────────────────────────────────
# 请求/响应模型
# ──────────────────────────────────────────────────────────


class UsageRecordRequest(BaseModel):
    """用量记录请求。"""

    event_type: str = Field(..., description="事件类型：backtest_basic, api_call 等")
    quantity: int = Field(default=1, ge=1, description="消耗数量")
    metadata: Optional[Dict[str, Any]] = Field(None, description="事件详情")


class UsageRecordResponse(BaseModel):
    """用量记录响应。"""

    event_id: str
    event_type: str
    unit_cost: float
    quantity: int
    total_cost: float
    billing_status: str
    message: str


class DepositRequest(BaseModel):
    """充值请求。"""

    amount_cents: int = Field(..., ge=1, description="充值金额（分）")
    remark: Optional[str] = Field(None, description="充值备注")


class DepositResponse(BaseModel):
    """充值响应。"""

    balance_cents: int
    balance_yuan: float
    credited_cents: int
    billing_record_id: str


class RefundRequest(BaseModel):
    """退款请求。"""

    billing_record_id: str = Field(..., description="原始账单 ID")
    amount_cents: int = Field(..., ge=1, description="退款金额（分）")


class PricingRuleResponse(BaseModel):
    """计费规则响应。"""

    event_type: str
    base_price_cents: int
    base_price_yuan: float
    discounted_price_cents: Optional[int]
    discounted_price_yuan: Optional[float]
    unit_name: Optional[str]
    description: Optional[str]
    is_active: bool


class BillingRecordResponse(BaseModel):
    """账单记录响应。"""

    id: str
    user_id: str
    usage_event_id: str
    amount_cents: int
    amount_yuan: float
    charge_status: str
    payment_method: Optional[str] = None
    balance_before_cents: Optional[int] = None
    balance_after_cents: Optional[int] = None
    remark: Optional[str] = None
    charged_at: Optional[str] = None
    created_at: Optional[str] = None


class UsageSummaryResponse(BaseModel):
    """用量汇总响应。"""

    month: int
    year: int
    total_spent_cents: int
    total_spent_yuan: float
    events: Dict[str, Dict[str, Any]]


class BalanceResponse(BaseModel):
    """余额响应。"""

    user_id: str
    balance_cents: int
    balance_yuan: float
    credited_cents: int
    total_spent_cents: int
    updated_at: Optional[str]


# ──────────────────────────────────────────────────────────
# 路由
# ──────────────────────────────────────────────────────────


@router.post("/usage", response_model=UsageRecordResponse, summary="记录用量并扣费")
async def record_usage(
    request: UsageRecordRequest,
    user_id: str = Depends(get_current_user),
) -> UsageRecordResponse:
    """记录一次用量事件并从用户余额扣费。

    - 回测：每次 ¥0.5 ~ ¥2
    - API 调用：每次 ¥0.01
    - 数据下载：每次 ¥0.05
    - 实时行情：每次 ¥0.05
    """
    service = CommerceService()
    event, billing = await service.record_usage(
        user_id=user_id,
        event_type=request.event_type,
        quantity=request.quantity,
        metadata=request.metadata,
    )

    billing_status = billing.charge_status if billing else "skipped"
    message = (
        "扣费成功"
        if billing and billing.charge_status == "charged"
        else "余额不足，待充值"
    )

    return UsageRecordResponse(
        event_id=event.id,
        event_type=event.event_type,
        unit_cost=float(event.unit_cost),
        quantity=event.quantity,
        total_cost=float(event.total_cost),
        billing_status=billing_status,
        message=message,
    )


@router.get("/balance", response_model=BalanceResponse, summary="查询余额")
async def get_balance(
    user_id: str = Depends(get_current_user),
) -> BalanceResponse:
    """查询用户当前余额。"""
    service = CommerceService()
    balance = await service.get_balance(user_id)

    if balance is None:
        balance = await service._get_or_create_balance(
            await AsyncSessionLocal().__aenter__(), user_id
        )

    return BalanceResponse(
        user_id=balance.user_id,
        balance_cents=balance.balance_cents,
        balance_yuan=balance.balance_cents / 100.0,
        credited_cents=balance.credited_cents,
        total_spent_cents=balance.total_spent_cents,
        updated_at=balance.updated_at.isoformat() if balance.updated_at else None,
    )


@router.post("/deposit", response_model=DepositResponse, summary="用户充值")
async def deposit(
    request: DepositRequest,
    user_id: str = Depends(get_current_user),
) -> DepositResponse:
    """用户充值到余额。"""
    service = CommerceService()
    balance, record = await service.deposit(
        user_id=user_id,
        amount_cents=request.amount_cents,
        remark=request.remark,
    )

    return DepositResponse(
        balance_cents=balance.balance_cents,
        balance_yuan=balance.balance_cents / 100.0,
        credited_cents=balance.credited_cents,
        billing_record_id=record.id,
    )


@router.post("/refund", summary="退款")
async def refund(
    request: RefundRequest,
    user_id: str = Depends(get_current_user),
) -> Dict[str, Any]:
    """退款操作。"""
    service = CommerceService()
    record = await service.refund(
        user_id=user_id,
        billing_record_id=request.billing_record_id,
        amount_cents=request.amount_cents,
    )

    if record is None:
        raise HTTPException(status_code=404, detail="退款记录创建失败")

    return {
        "refund_record_id": record.id,
        "amount_yuan": record.amount_cents / 100.0,
        "status": record.charge_status,
    }


@router.get("/usage/summary", response_model=UsageSummaryResponse, summary="用量汇总")
async def usage_summary(
    user_id: str = Depends(get_current_user),
    month: Optional[int] = Query(None, description="月份（1-12），默认当月"),
    year: Optional[int] = Query(None, description="年份，默认当年"),
) -> UsageSummaryResponse:
    """查询用户当月用量汇总。"""
    service = CommerceService()
    summary = await service.get_usage_summary(user_id, month, year)
    return UsageSummaryResponse(**summary)


@router.get(
    "/billing/history",
    response_model=List[BillingRecordResponse],
    summary="账单历史",
)
async def billing_history(
    user_id: str = Depends(get_current_user),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> List[Dict[str, Any]]:
    """查询用户账单历史。"""
    service = CommerceService()
    records = await service.get_billing_history(user_id, limit, offset)
    return [r.to_dict() for r in records]


@router.get(
    "/pricing/rules", response_model=List[PricingRuleResponse], summary="计费规则列表"
)
async def get_pricing_rules(
    event_type: Optional[str] = Query(None, description="事件类型筛选"),
) -> List[PricingRuleResponse]:
    """查询计费规则列表（公开）。"""
    service = CommerceService()
    rules = await service.get_pricing_rules(event_type)
    return [PricingRuleResponse(**r.to_dict()) for r in rules]


@router.post("/pricing/seed", summary="初始化默认计费规则")
async def seed_pricing(
    user_id: str = Depends(get_current_user),
) -> Dict[str, Any]:
    """初始化默认计费规则（仅管理员）。"""
    # 注意：需要管理员权限检查
    db = AsyncSessionLocal()
    try:
        rules = await seed_default_pricing_rules(db)
        return {
            "seeded": len(rules),
            "rules": [r.event_type for r in rules],
        }
    finally:
        await db.close()


@router.get("/check/{event_type}", summary="检查是否可以执行操作")
async def check_can_execute(
    event_type: str,
    user_id: str = Depends(get_current_user),
) -> Dict[str, Any]:
    """检查用户余额是否足够执行指定操作。"""
    can_proceed, message = await check_can_proceed(user_id, event_type)

    service = CommerceService()
    pricing = await service._get_pricing_rule(event_type)
    cost_yuan = pricing.base_price_cents / 100.0 if pricing else 0

    return {
        "can_proceed": can_proceed,
        "message": message,
        "event_type": event_type,
        "estimated_cost_yuan": cost_yuan,
    }
