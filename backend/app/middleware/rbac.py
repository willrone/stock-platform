"""用户角色、套餐和资源配额依赖。"""

from datetime import datetime
from enum import Enum
from typing import Any, Awaitable, Callable

from fastapi import Depends, HTTPException, status
from loguru import logger
from sqlalchemy import func, select

from app.api.v1.dependencies import require_current_user
from app.core.database import AsyncSessionLocal
from app.models.strategy_config_models import StrategyConfig
from app.models.task_models import Task, TaskStatus, TaskType
from app.models.user_models import User


class Role(str, Enum):
    """平台角色/订阅等级。"""

    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"
    ADMIN = "admin"


TIER_CONFIG: dict[Role, dict[str, int]] = {
    Role.FREE: {
        "monthly_backtest_limit": 50,
        "max_strategies": 20,
        "max_concurrent_tasks": 3,
    },
    Role.PRO: {
        "monthly_backtest_limit": 500,
        "max_strategies": 100,
        "max_concurrent_tasks": 10,
    },
    Role.ENTERPRISE: {
        "monthly_backtest_limit": -1,
        "max_strategies": -1,
        "max_concurrent_tasks": 50,
    },
    Role.ADMIN: {
        "monthly_backtest_limit": -1,
        "max_strategies": -1,
        "max_concurrent_tasks": -1,
    },
}

_ROLE_ORDER = {
    Role.FREE: 0,
    Role.PRO: 1,
    Role.ENTERPRISE: 2,
    Role.ADMIN: 3,
}


def _user_role(user: User) -> Role:
    if user.is_admin:
        return Role.ADMIN
    try:
        return Role(str(user.subscription_tier or Role.FREE.value).lower())
    except ValueError:
        return Role.FREE


def _quota_limit(user: User, quota_type: str) -> int:
    configured = getattr(user, quota_type, None)
    if configured is not None:
        return int(configured)
    return TIER_CONFIG[_user_role(user)].get(quota_type, -1)


async def _quota_usage(user: User, quota_type: str) -> int:
    now = datetime.utcnow()
    month_start = datetime(now.year, now.month, 1)
    async with AsyncSessionLocal() as session:
        if quota_type == "monthly_backtest_limit":
            stmt = select(func.count(Task.task_id)).where(
                Task.user_id == user.id,
                Task.task_type == TaskType.BACKTEST.value,
                Task.created_at >= month_start,
            )
        elif quota_type == "max_strategies":
            stmt = select(func.count(StrategyConfig.config_id)).where(
                StrategyConfig.user_id == user.id
            )
        elif quota_type == "max_concurrent_tasks":
            stmt = select(func.count(Task.task_id)).where(
                Task.user_id == user.id,
                Task.status.in_(
                    [
                        TaskStatus.CREATED.value,
                        TaskStatus.QUEUED.value,
                        TaskStatus.RUNNING.value,
                    ]
                ),
            )
        else:
            raise ValueError(f"未知配额类型: {quota_type}")
        return int((await session.execute(stmt)).scalar_one() or 0)


async def enforce_quota(user: User, quota_type: str) -> User:
    """检查用户配额，耗尽时返回 429。"""
    limit = _quota_limit(user, quota_type)
    if limit < 0:
        return user
    usage = await _quota_usage(user, quota_type)
    if usage >= limit:
        logger.warning(
            "用户配额已用尽: user_id={}, quota={}, usage={}, limit={}",
            user.id,
            quota_type,
            usage,
            limit,
        )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"配额已用尽: {quota_type}",
        )
    return user


def require_tier(min_tier: Role | str) -> Callable[..., Awaitable[User]]:
    """创建最低套餐等级依赖。"""
    required = min_tier if isinstance(min_tier, Role) else Role(str(min_tier).lower())

    async def dependency(user: User = Depends(require_current_user)) -> User:
        if _ROLE_ORDER[_user_role(user)] < _ROLE_ORDER[required]:
            raise HTTPException(status_code=403, detail="当前套餐不支持该功能")
        return user

    return dependency


def check_quota(quota_type: str) -> Callable[..., Awaitable[User]]:
    """创建运行时配额检查依赖。"""

    async def dependency(user: User = Depends(require_current_user)) -> User:
        return await enforce_quota(user, quota_type)

    return dependency


__all__ = ["Role", "TIER_CONFIG", "require_tier", "check_quota", "enforce_quota"]
