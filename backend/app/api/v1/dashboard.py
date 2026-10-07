"""
用户用量统计 API

提供当前用户的：
- 资源使用量（backtest、策略、并发任务）
- 配额上限对比
- 近 30 天每日调用次数（来源：Task 表）
- 订阅等级

对外路径：/api/v1/dashboard/*
"""

from datetime import datetime, timedelta
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger
from sqlalchemy import and_, func, select

from app.api.v1.dependencies import get_current_user
from app.api.v1.schemas import StandardResponse
from app.core.database import AsyncSessionLocal
from app.models.strategy_config_models import StrategyConfig
from app.models.user_models import User
from app.models.task_models import Task, TaskStatus, TaskType
from app.repositories.task_repository import TaskRepository

router = APIRouter(prefix="/dashboard", tags=["用量统计"])


@router.get("/stats", response_model=StandardResponse, summary="当前用户用量汇总")
async def get_dashboard_stats(user_id: str = Depends(get_current_user)) -> Any:
    """
    用途：用量汇总页一次拉全。

    返回:
      - user: { id, email, username, subscription_tier, is_admin }
      - usage: {
          backtests_this_month, monthly_backtest_limit, backtests_remaining,
          strategy_count, max_strategies,
          active_tasks, max_concurrent_tasks,
        }
      - limits: { 各维度用量占比 % }
    """
    try:
        async with AsyncSessionLocal() as session:
            # 1. 用户基本信息
            user_row = await session.get(User, user_id)
            if user_row is None:
                raise HTTPException(status_code=404, detail=f"用户不存在: {user_id}")

            # 2. 本月回测次数（自然月）
            now = datetime.utcnow()
            month_start = datetime(now.year, now.month, 1)
            backtest_count_stmt = (
                select(func.count(Task.task_id))
                .where(
                    and_(
                        Task.user_id == user_id,
                        Task.task_type == TaskType.BACKTEST.value,
                        Task.created_at >= month_start,
                    )
                )
            )
            backtest_count = (await session.execute(backtest_count_stmt)).scalar() or 0

            # 3. 策略配置数
            strategy_count_stmt = (
                select(func.count(StrategyConfig.config_id))
                .where(StrategyConfig.user_id == user_id)
            )
            strategy_count = (await session.execute(strategy_count_stmt)).scalar() or 0

            # 4. 当前活跃任务数（running + pending）
            task_repo = TaskRepository(session)
            active_running = task_repo.count_tasks_by_user(
                user_id=user_id,
                status_filter=TaskStatus.RUNNING,
            )
            active_pending = task_repo.count_tasks_by_user(
                user_id=user_id,
                status_filter=TaskStatus.PENDING,
            )
            active_total = active_running + active_pending

            usage = {
                "backtests_this_month": int(backtest_count),
                "monthly_backtest_limit": user_row.monthly_backtest_limit,
                "backtests_remaining": max(
                    0, user_row.monthly_backtest_limit - int(backtest_count)
                ),
                "strategy_count": int(strategy_count),
                "max_strategies": user_row.max_strategies,
                "active_tasks": int(active_total),
                "max_concurrent_tasks": user_row.max_concurrent_tasks,
            }

            limits = {
                "backtest_usage_pct": round(
                    100.0 * backtest_count / max(user_row.monthly_backtest_limit, 1), 1
                ),
                "strategy_usage_pct": round(
                    100.0 * strategy_count / max(user_row.max_strategies, 1), 1
                ),
                "concurrent_usage_pct": round(
                    100.0 * active_total / max(user_row.max_concurrent_tasks, 1), 1
                ),
            }

            return StandardResponse(
                success=True,
                message="用量统计获取成功",
                data={
                    "user": {
                        "id": user_row.id,
                        "email": user_row.email,
                        "username": user_row.username,
                        "subscription_tier": user_row.subscription_tier,
                        "is_admin": user_row.is_admin,
                    },
                    "usage": usage,
                    "limits": limits,
                    "timestamp": now.isoformat(),
                },
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"用量统计获取失败: user_id={user_id}, error={e}")
        raise HTTPException(status_code=500, detail=f"用量统计获取失败: {str(e)}")


@router.get(
    "/requests/today",
    response_model=StandardResponse,
    summary="近 30 天 API 调用次数（按天聚合）",
)
async def get_dashboard_requests_today(
    user_id: str = Depends(get_current_user),
) -> Any:
    """
    用途：用量趋势图。

    数据来源：Task 表 created_at 聚合（用户级）。RequestLoggingMiddleware
    是进程内日志，未暴露历史 buffer，避免外部依赖。
    """
    try:
        now = datetime.utcnow()
        since = now - timedelta(days=30)

        async with AsyncSessionLocal() as session:
            rows = await session.execute(
                select(
                    func.date(Task.created_at).label("day"),
                    func.count(Task.task_id).label("count"),
                )
                .where(and_(Task.user_id == user_id, Task.created_at >= since))
                .group_by(func.date(Task.created_at))
                .order_by(func.date(Task.created_at))
            )
            series = [
                {"date": str(row.day), "count": int(row.count)} for row in rows
            ]

        return StandardResponse(
            success=True,
            message="近 30 天用量趋势获取成功",
            data={
                "series": series,
                "window_days": 30,
                "source": "task_table",
                "timestamp": now.isoformat(),
            },
        )

    except Exception as e:
        logger.error(f"用量趋势获取失败: user_id={user_id}, error={e}")
        raise HTTPException(status_code=500, detail=f"用量趋势获取失败: {str(e)}")
