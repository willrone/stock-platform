"""
Laya 决策模型 API 路由

提供交易决策辅助的 RESTful API。
"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.api.v1.dependencies import get_current_user
from app.api.v1.schemas import StandardResponse
from app.services.laya import (
    TRADING_QUESTION_TEMPLATES,
    LayaService,
    TradingState,
)

router = APIRouter(prefix="/laya", tags=["Laya 决策模型"])


class DecisionRequest(BaseModel):
    """决策请求体。"""

    state: Dict[str, Any] = Field(..., description="交易状态数据")
    question_types: List[str] = Field(
        default=["action", "confidence"], description="问题类型列表"
    )
    template_name: Optional[str] = Field(
        default=None, description="预定义模板名称（如 full_analysis）"
    )
    user_id: Optional[str] = Field(default=None, description="用户ID（用于计费）")


class DecisionResponse(BaseModel):
    """决策响应体。"""

    decision: Optional[Dict[str, Any]] = None
    health_status: Optional[Dict[str, Any]] = None
    fallback_reason: Optional[str] = None


@router.post("/decision", response_model=StandardResponse)
async def make_decision(request: DecisionRequest):
    """基于 Laya 模型做出交易决策。

    接收交易状态，返回决策建议和置信度。
    """
    laya_service = LayaService()

    if not laya_service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Laya 模型服务未就绪",
        )

    # 构建交易状态
    state = TradingState(**request.state)

    # 确定问题类型
    question_types = request.question_types
    if request.template_name and request.template_name in TRADING_QUESTION_TEMPLATES:
        question_types = list(
            TRADING_QUESTION_TEMPLATES[request.template_name]["questions"].keys()
        )

    # 执行分析
    decision = await laya_service.analyze(
        state=state,
        question_types=question_types,
        user_id=request.user_id,
        metadata={"template": request.template_name},
    )

    if decision is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Laya 决策分析失败",
        )

    return StandardResponse(
        success=True,
        message="决策分析完成",
        data=DecisionResponse(
            decision=decision.to_dict(), health_status=laya_service.get_health_status()
        ),
    )


@router.post("/enhance", response_model=StandardResponse)
async def enhance_signal(request: DecisionRequest):
    """增强交易信号。

    用 Laya 对基础策略信号做二次验证。
    """
    laya_service = LayaService()

    if not laya_service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Laya 模型服务未就绪",
        )

    # 构建交易状态
    state = TradingState(**request.state)

    # 基础信号（从请求中获取）
    base_signal = request.state.get("base_signal", {"action": "hold"})

    # 增强信号
    result = await laya_service.enhance_signal(
        base_signal=base_signal, state=state, user_id=request.user_id
    )

    return StandardResponse(success=True, message="信号增强完成", data=result)


@router.post("/signal/verify", response_model=StandardResponse)
async def verify_signal(
    signal: Dict[str, Any], user_id: str = Depends(get_current_user)
):
    """验证交易信号。

    简化版：只返回信号验证结果。
    """
    laya_service = LayaService()

    if not laya_service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Laya 模型服务未就绪",
        )

    # 提取信号中的价格数据
    state_dict = {
        "current_price": signal.get("current_price", 0),
        "rsi": signal.get("rsi", 50),
        "macd_signal": signal.get("macd_signal", "neutral"),
        "position_size": signal.get("position_size", 0),
        "market_regime": signal.get("market_regime", "trending"),
        "symbol": signal.get("symbol", ""),
    }

    state = TradingState(**state_dict)

    decision = await laya_service.analyze(
        state=state,
        question_types=["action", "confidence"],
        user_id=user_id,
        metadata={"original_signal": signal},
    )

    if decision is None:
        return StandardResponse(
            success=False,
            message="Laya 分析失败，返回原始信号",
            data={"original_signal": signal, "enhanced": False},
        )

    return StandardResponse(
        success=True,
        message="信号验证完成",
        data={
            "original_signal": signal,
            "laya_decision": decision.to_dict(),
            "enhanced": True,
            "final_action": (
                decision.action
                if decision.confidence >= 0.4
                else signal.get("action", "hold")
            ),
        },
    )


@router.get("/health", response_model=StandardResponse)
async def health_check():
    """Laya 服务健康检查。"""
    laya_service = LayaService()

    return StandardResponse(
        success=True, message="服务状态正常", data=laya_service.get_health_status()
    )


@router.post("/initialize", response_model=StandardResponse)
async def initialize_laya():
    """手动初始化 Laya 模型（可选自动加载）。"""
    laya_service = LayaService()

    success = await laya_service.initialize()

    if success:
        return StandardResponse(
            success=True,
            message="Laya 模型初始化成功",
            data=laya_service.get_health_status(),
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Laya 模型初始化失败",
        )
