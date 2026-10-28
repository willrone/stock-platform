"""
按使用量计费 API 路由。

提供：
- 用量记录（回测/API/数据操作计费事件）
- 余额查询/充值/退款
- 计费规则查询
- 用量统计/账单历史
"""

from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app.api.v1.dependencies import get_current_user, require_admin_user
from app.api.v1.schemas import StandardResponse
from app.core.database import AsyncSessionLocal
from app.services.commerce.commission_service import (
    CommerceService,
    check_can_proceed,
    seed_default_pricing_rules,
)
from app.services.commerce.payment_service import (
    MAX_DEPOSIT_CENTS,
    MIN_DEPOSIT_CENTS,
    PaymentService,
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
    # 2026-10-25 新增：本次充值对应的套餐（free / pro / enterprise）。
    # 管理员确认到账时据此自动升级用户套餐 —— 否则「收了钱但套餐没变」。
    plan_id: Optional[str] = Field(
        None, description="对应套餐：free / pro / enterprise"
    )


class DepositResponse(BaseModel):
    """充值响应（2026-10-25 起为订单制，不立即加分）。"""

    balance_cents: int
    balance_yuan: float
    credited_cents: int
    billing_record_id: Optional[str] = Field(
        None, description="到账后才有的账单 ID（订单制下为 None）"
    )
    order_no: Optional[str] = Field(None, description="充值订单号")
    order_status: Optional[str] = Field(None, description="订单状态：pending/paid")
    amount_cents: Optional[int] = Field(None, description="本次充值金额（分）")
    payment_method: Optional[str] = Field(None, description="支付方式")
    payee_account: Optional[str] = Field(None, description="收款账号")
    qr_image_url: Optional[str] = Field(None, description="收款码图片 URL")
    instructions: Optional[str] = Field(None, description="支付说明")


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


@router.post(
    "/usage", response_model=StandardResponse, summary="记录用量并扣费"
)
async def record_usage(
    request: UsageRecordRequest,
    user_id: str = Depends(get_current_user),
) -> StandardResponse:
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
    charged = bool(billing and billing.charge_status == "charged")
    message = "用量记录并扣费成功" if charged else "余额不足，待充值"

    return StandardResponse(
        success=True,
        message=message,
        data=UsageRecordResponse(
            event_id=event.id,
            event_type=event.event_type,
            unit_cost=float(event.unit_cost),
            quantity=event.quantity,
            total_cost=float(event.total_cost),
            billing_status=billing_status,
            message="扣费成功" if charged else "余额不足，待充值",
        ),
    )


@router.get("/balance", response_model=StandardResponse, summary="查询余额")
async def get_balance(
    user_id: str = Depends(get_current_user),
) -> StandardResponse:
    """查询用户当前余额。"""
    service = CommerceService()
    # get_or_create_balance 自己管 session 生命周期
    # （旧写法 AsyncSessionLocal().__aenter__() 从不关闭 → session 泄漏）
    balance = await service.get_or_create_balance(user_id)

    return StandardResponse(
        success=True,
        message="余额查询成功",
        data=BalanceResponse(
            user_id=balance.user_id,
            balance_cents=balance.balance_cents,
            balance_yuan=balance.balance_cents / 100.0,
            credited_cents=balance.credited_cents,
            total_spent_cents=balance.total_spent_cents,
            updated_at=balance.updated_at.isoformat() if balance.updated_at else None,
        ),
    )


@router.post(
    "/deposit",
    response_model=StandardResponse,
    summary="创建充值订单（两段式）",
)
async def deposit(
    request: DepositRequest,
    user_id: str = Depends(get_current_user),
) -> StandardResponse:
    """创建充值订单。

    ⚠️ 2026-10-25 语义变更：此接口**不再直接给余额加分**。
    原实现无任何支付验证 → 任何注册用户可自助充值无限余额（收入漏洞）。
    现改为两段式订单制：
      1. 本接口创建订单（pending），返回订单号 + 收款指引
      2. 管理员确认到账后余额才增加（见 /payment/orders/{order_no}/confirm）
    没有自动支付渠道时走人工对账：用户转账 → 报订单号 → 管理员确认。
    """
    service = PaymentService()
    try:
        order = await service.create_order(
            user_id=user_id,
            amount_cents=request.amount_cents,
            remark=request.remark or "",
            plan_id=request.plan_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    config = await service.get_config()
    commerce = CommerceService()
    balance = await commerce.get_or_create_balance(user_id)
    balance_cents = int(getattr(balance, "balance_cents", 0) or 0)
    credited_cents = int(getattr(balance, "credited_cents", 0) or 0)
    return StandardResponse(
        success=True,
        message="充值下单成功",
        data=DepositResponse(
            balance_cents=balance_cents,
            balance_yuan=balance_cents / 100.0,
            credited_cents=credited_cents,
            billing_record_id=None,
            order_no=order.order_no,
            order_status=order.status,
            amount_cents=order.amount_cents,
            payment_method=order.payment_method,
            payee_account=config.get("payee_account", ""),
            qr_image_url=config.get("qr_image_url", ""),
            instructions=config.get("instructions", ""),
        ),
    )


# ──────────────────────────────────────────────────────────
# 充值订单（用户侧 / 管理侧）
# ──────────────────────────────────────────────────────────


@router.get("/payment/orders", response_model=StandardResponse, summary="我的充值订单")
async def list_my_orders(
    user_id: str = Depends(get_current_user),
    limit: int = Query(50, ge=1, le=200),
) -> StandardResponse:
    """列出当前用户的充值订单。"""
    orders = await PaymentService().list_orders(user_id=user_id, limit=limit)
    return StandardResponse(
        success=True,
        message="我的充值订单获取成功",
        data={"orders": orders, "count": len(orders)},
    )


@router.get(
    "/payment/config", response_model=StandardResponse, summary="收款方式配置（公开）"
)
async def get_payment_config() -> StandardResponse:
    """获取收款指引（用户充值页要显示的收款码/账号）。"""
    config = await PaymentService().get_config()
    config["min_deposit_cents"] = MIN_DEPOSIT_CENTS
    config["max_deposit_cents"] = MAX_DEPOSIT_CENTS
    return StandardResponse(
        success=True,
        message="收款方式配置获取成功",
        data=config,
    )


@router.get(
    "/admin/payment/orders",
    response_model=StandardResponse,
    summary="全部充值订单（管理员）",
)
async def admin_list_orders(
    admin: Any = Depends(require_admin_user),
    status: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=500),
) -> StandardResponse:
    """管理员查看全部充值订单。"""
    orders = await PaymentService().list_orders(user_id=None, status=status, limit=limit)
    return StandardResponse(
        success=True,
        message="全部充值订单获取成功",
        data={"orders": orders, "count": len(orders)},
    )


@router.post(
    "/admin/payment/orders/{order_no}/confirm",
    response_model=StandardResponse,
    summary="确认订单到账（管理员）",
)
async def admin_confirm_order(
    order_no: str,
    admin: Any = Depends(require_admin_user),
    remark: str = Query("", description="管理员备注"),
) -> StandardResponse:
    """确认充值订单已到账 → 余额入账（+ 自动升级套餐）。"""
    order, err = await PaymentService().confirm_order(
        order_no, admin_id=str(getattr(admin, "id", "")), admin_remark=remark
    )
    if err or order is None:
        raise HTTPException(status_code=400, detail=err or "订单确认失败")

    # 返回用户当前套餐，管理员能直接看到「已升级到 pro」
    tier = None
    try:
        from app.models.user_models import User
        from sqlalchemy import select

        db = AsyncSessionLocal()
        try:
            u = await db.scalar(select(User).where(User.id == order.user_id))
            tier = getattr(u, "subscription_tier", None)
        finally:
            await db.close()
    except Exception:  # noqa: BLE001
        tier = None

    return StandardResponse(
        success=True,
        message="管理员确认到账成功",
        data={
            "order": order.to_dict(),
            "message": "已确认到账，余额已入账",
            "subscription_tier": tier,
        },
    )


@router.post(
    "/admin/payment/orders/{order_no}/cancel",
    response_model=StandardResponse,
    summary="取消充值订单（管理员）",
)
async def admin_cancel_order(
    order_no: str,
    admin: Any = Depends(require_admin_user),
    remark: str = Query("", description="管理员备注"),
) -> StandardResponse:
    """取消充值订单。"""
    order, err = await PaymentService().cancel_order(order_no, admin_remark=remark)
    if err:
        raise HTTPException(status_code=400, detail=err)
    return StandardResponse(
        success=True,
        message="充值订单取消成功",
        data={"order": order.to_dict(), "message": "订单已取消"},
    )


class PaymentConfigRequest(BaseModel):
    """收款配置更新请求（JSON body，避免中文 query 编码问题）。"""

    method_name: str = Field(..., description="收款方式名称，如 微信转账")
    payee_account: str = Field(..., description="收款账号")
    qr_image_url: str = Field("", description="收款码图片 URL")
    instructions: str = Field("", description="支付说明")
    is_enabled: bool = Field(True, description="是否开启在线充值")


@router.post(
    "/admin/payment/config", response_model=StandardResponse, summary="设置收款方式（管理员）"
)
async def admin_update_payment_config(
    request: PaymentConfigRequest,
    admin: Any = Depends(require_admin_user),
) -> StandardResponse:
    """设置收款方式（人工收款场景下替代 Stripe）。"""
    config = await PaymentService().update_config(
        method_name=request.method_name,
        payee_account=request.payee_account,
        qr_image_url=request.qr_image_url,
        instructions=request.instructions,
        is_enabled=request.is_enabled,
    )
    return StandardResponse(
        success=True,
        message="收款方式配置更新成功",
        data={"config": config, "message": "收款配置已更新"},
    )


@router.post("/refund", response_model=StandardResponse, summary="退款")
async def refund(
    request: RefundRequest,
    user_id: str = Depends(get_current_user),
) -> StandardResponse:
    """退款操作。"""
    service = CommerceService()
    record = await service.refund(
        user_id=user_id,
        billing_record_id=request.billing_record_id,
        amount_cents=request.amount_cents,
    )

    if record is None:
        raise HTTPException(status_code=404, detail="退款记录创建失败")

    return StandardResponse(
        success=True,
        message="退款成功",
        data={
            "refund_record_id": record.id,
            "amount_yuan": record.amount_cents / 100.0,
            "status": record.charge_status,
        },
    )


@router.get(
    "/usage/summary", response_model=StandardResponse, summary="用量汇总"
)
async def usage_summary(
    user_id: str = Depends(get_current_user),
    month: Optional[int] = Query(None, description="月份（1-12），默认当月"),
    year: Optional[int] = Query(None, description="年份，默认当年"),
) -> StandardResponse:
    """查询用户当月用量汇总。"""
    service = CommerceService()
    summary = await service.get_usage_summary(user_id, month, year)
    return StandardResponse(
        success=True,
        message="用量汇总查询成功",
        data=UsageSummaryResponse(**summary),
    )


@router.get(
    "/billing/history",
    response_model=StandardResponse,
    summary="账单历史",
)
async def billing_history(
    user_id: str = Depends(get_current_user),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> StandardResponse:
    """查询用户账单历史。"""
    service = CommerceService()
    records = await service.get_billing_history(user_id, limit, offset)
    return StandardResponse(
        success=True,
        message="账单历史查询成功",
        data=[r.to_dict() for r in records],
    )


@router.get(
    "/pricing/rules",
    response_model=StandardResponse,
    summary="计费规则列表",
)
async def get_pricing_rules(
    event_type: Optional[str] = Query(None, description="事件类型筛选"),
) -> StandardResponse:
    """查询计费规则列表（公开）。"""
    service = CommerceService()
    rules = await service.get_pricing_rules(event_type)
    return StandardResponse(
        success=True,
        message="计费规则列表获取成功",
        data=[PricingRuleResponse(**r.to_dict()) for r in rules],
    )


@router.post("/pricing/seed", response_model=StandardResponse, summary="初始化默认计费规则")
async def seed_pricing(
    user_id: str = Depends(get_current_user),
) -> StandardResponse:
    """初始化默认计费规则（仅管理员）。"""
    # 注意：需要管理员权限检查
    db = AsyncSessionLocal()
    try:
        rules = await seed_default_pricing_rules(db)
        return StandardResponse(
            success=True,
            message="默认计费规则初始化成功",
            data={
                "seeded": len(rules),
                "rules": [r.event_type for r in rules],
            },
        )
    finally:
        await db.close()


@router.get("/check/{event_type}", response_model=StandardResponse, summary="检查是否可以执行操作")
async def check_can_execute(
    event_type: str,
    user_id: str = Depends(get_current_user),
) -> StandardResponse:
    """检查用户余额是否足够执行指定操作。"""
    can_proceed, message = await check_can_proceed(user_id, event_type)

    service = CommerceService()
    pricing = await service._get_pricing_rule(event_type)
    cost_yuan = pricing.base_price_cents / 100.0 if pricing else 0

    return StandardResponse(
        success=True,
        message=message,
        data={
            "can_proceed": can_proceed,
            "message": message,
            "event_type": event_type,
            "estimated_cost_yuan": cost_yuan,
        },
    )
