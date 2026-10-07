"""
按使用量计费系统 - 启用/禁用开关

提供快速启用或禁用所有计费功能的接口。
当功能被禁用时：
- 所有计费事件将被忽略
- 余额查询将返回默认值
- 充值和退款操作将被拒绝
- 用量记录将被静默存储
"""

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(prefix="/commerce", tags=["计费控制"])

# 全局开关状态
_COMMERCE_ENABLED: bool = True
_COMMERCE_ENABLE_TIME: Optional[datetime] = None
_ENABLE_REASON: str = "系统启动时启用"


def is_commerce_enabled() -> bool:
    """检查计费功能是否启用"""
    return _COMMERCE_ENABLED


async def set_commerce_enabled(
    enabled: bool,
    reason: str = "系统管理员操作",
    background_tasks: Optional[BackgroundTasks] = None,
    db_session: Optional[AsyncSession] = None,
) -> Dict[str, Any]:
    """启用或禁用计费功能

    Args:
        enabled: 是否启用功能
        reason: 启用原因/理由
        background_tasks: 后台任务
        db_session: 数据库会话

    Returns:
        操作结果
    """
    global _COMMERCE_ENABLED, _COMMERCE_ENABLE_TIME, _ENABLE_REASON

    old_status = _COMMERCE_ENABLED
    _COMMERCE_ENABLED = enabled
    _COMMERCE_ENABLE_TIME = datetime.now(timezone.utc).replace(tzinfo=None)
    _ENABLE_REASON = reason

    result: Dict[str, Any] = {
        "success": True,
        "old_status": old_status,
        "new_status": enabled,
        "reason": reason,
        "timestamp": (
            _COMMERCE_ENABLE_TIME.isoformat() if _COMMERCE_ENABLE_TIME else None
        ),
    }

    if enabled:
        # 启用计费功能，重新加载默认规则
        result["message"] = "计费功能已启用"
    else:
        # 禁用计费功能
        result["message"] = "计费功能已禁用，所有计费操作将被忽略"

    return result


@router.get("/status", summary="获取计费功能状态")
async def get_commerce_status() -> Dict[str, Any]:
    """获取当前计费功能状态"""
    return {
        "enabled": is_commerce_enabled(),
        "enable_time": (
            _COMMERCE_ENABLE_TIME.isoformat() if _COMMERCE_ENABLE_TIME else None
        ),
        "reason": _ENABLE_REASON,
    }


@router.post("/toggle", summary="启用或禁用计费功能")
async def toggle_commerce(
    request: Dict[str, Any],
) -> Dict[str, Any]:
    """启用或禁用计费功能

    请求体示例:
    {
        "enabled": true,      # true=启用，false=禁用
        "reason": "系统维护"   # 操作原因
    }
    """
    enabled = request.get("enabled", False)
    reason = request.get("reason", "系统管理员操作")

    if not isinstance(enabled, bool):
        raise HTTPException(status_code=400, detail="enabled 必须是布尔值")

    return await set_commerce_enabled(enabled, reason)


@router.post("/enable", summary="启用计费功能")
async def enable_commerce(
    request: Dict[str, Any],
) -> Dict[str, Any]:
    """启用计费功能"""
    reason = request.get("reason", "系统管理员操作")
    return await set_commerce_enabled(True, reason)


@router.post("/disable", summary="禁用计费功能")
async def disable_commerce(
    request: Dict[str, Any],
) -> Dict[str, Any]:
    """禁用计费功能"""
    reason = request.get("reason", "系统管理员操作")
    return await set_commerce_enabled(False, reason)


@router.post("/reset", summary="重置计费功能（重启）")
async def reset_commerce(
    request: Dict[str, Any],
) -> Dict[str, Any]:
    """重置计费功能到默认启用状态"""
    reason = request.get("reason", "系统重启，恢复默认设置")

    global _COMMERCE_ENABLED, _COMMERCE_ENABLE_TIME, _ENABLE_REASON

    old_status = _COMMERCE_ENABLED
    _COMMERCE_ENABLED = True
    _COMMERCE_ENABLE_TIME = datetime.now(timezone.utc).replace(tzinfo=None)
    _ENABLE_REASON = reason

    return {
        "success": True,
        "old_status": old_status,
        "new_status": True,
        "reason": reason,
        "message": "计费功能已重置为启用状态",
    }
