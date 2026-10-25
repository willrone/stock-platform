"""
可执行交易信号 API（2026-10-25）

对外提供「今天该买什么、买多少、为什么」：
  GET /api/v1/trading/strategies                可用信号策略列表
  GET /api/v1/trading/signal?strategy=xxx      今日可执行信号（含股数/金额/成本）
  GET /api/v1/trading/performance?strategy=xxx 纸面业绩（真实 track record）

这些是付费用户真正买的东西 —— 不是回测报告，是可以照着下单的清单。
"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query

from app.services.trading.signal_service import (
    SIGNAL_STRATEGIES,
    TradingSignalService,
)

router = APIRouter(prefix="/trading", tags=["可执行信号"])


@router.get("/strategies", summary="可用信号策略列表")
async def list_signal_strategies() -> Dict[str, Any]:
    """列出所有可产出信号的策略及其纸面盘状态。"""
    svc = TradingSignalService()
    return {
        "strategies": svc.list_strategies(),
        "count": len(SIGNAL_STRATEGIES),
    }


@router.get("/signal", summary="今日可执行信号")
async def get_daily_signal(
    strategy: str = Query(
        "honest_42f", description="策略 key，见 /trading/strategies"
    ),
) -> Dict[str, Any]:
    """返回某策略今天的买卖清单。

    字段说明：
      buys   买入清单（代码/名称/价格/股数/金额/成本）—— 可直接照着下单
      sells  卖出清单（同上）
      holds  当前持仓（代码/名称/股数/成本/买入日）
    """
    svc = TradingSignalService()
    try:
        return svc.daily_signal(strategy)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@router.get("/performance", summary="纸面业绩（真实 track record）")
async def get_performance(
    strategy: str = Query("honest_42f", description="策略 key"),
) -> Dict[str, Any]:
    """纸面组合的业绩、持仓与交易记录 —— 对外展示的可验证记录。"""
    svc = TradingSignalService()
    try:
        return svc.performance(strategy)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
