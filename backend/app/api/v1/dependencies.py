"""
API依赖注入和共享函数

注意：任务执行函数（execute_prediction_task_simple, execute_backtest_task_simple）
会在独立进程中执行，不能依赖全局变量或单例。每个进程必须独立创建所需资源。
"""

from typing import Any

from fastapi import Depends, Header, HTTPException, status
from loguru import logger
from sqlalchemy.orm import Session

from app.core.database import SessionLocal
from app.repositories.task_repository import (
    ModelInfoRepository,
    PredictionResultRepository,
    TaskRepository,
)
from app.services.tasks import TaskQueueManager
from app.services.tasks.task_executors import (
    execute_backtest_task_simple,
    execute_prediction_task_simple,
    execute_qlib_precompute_task_simple,
)


# 用户认证依赖
async def get_current_user(
    authorization: str | None = Header(None, alias="Authorization"),
) -> str:
    """从 Bearer token 获取当前用户 ID，不再接受 X-User-ID 后门。"""
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="缺少 Authorization 请求头",
            headers={"WWW-Authenticate": "Bearer"},
        )
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization 格式应为 Bearer <token>",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = token.strip()
    # 解析 JWT token 获取用户 ID
    try:
        from app.core.security import decode_access_token
        payload = decode_access_token(token)
        user_id = payload.get("sub")
        if user_id:
            logger.debug("使用 Bearer token 认证: {}", user_id)
            return str(user_id)
    except (PermissionError, Exception) as exc:
        logger.warning("Bearer token 认证失败: {}", exc)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Token 认证失败: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    # 没有任何有效认证信息
    logger.warning("未提供有效的认证信息")
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="未提供有效的认证信息，请提供 Authorization: Bearer <token>",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def require_current_user(
    user_id: str = Depends(get_current_user),
) -> Any:
    """解析并返回当前 User ORM 对象。"""
    from app.models.user_models import User

    session: Session = SessionLocal()
    try:
        user = session.query(User).filter(User.id == user_id).first()
        if user is None:
            raise HTTPException(status_code=401, detail="用户不存在")
        if not user.is_active:
            raise HTTPException(status_code=403, detail="账户已被禁用")
        return user
    finally:
        session.close()


async def require_admin_user(
    user: Any = Depends(require_current_user),
) -> Any:
    """要求当前用户具备管理员权限。"""
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="需要管理员权限")
    return user

task_queue_manager = TaskQueueManager()

# 启动任务调度器（在模块加载时启动）
try:
    task_queue_manager.start_all_schedulers()
    logger.info("任务队列管理器已启动")
except Exception as e:
    logger.warning(f"任务队列管理器启动失败: {e}")


def get_task_repository() -> Any:
    """获取任务仓库（使用同步会话）"""
    session = SessionLocal()
    try:
        return TaskRepository(session), session
    except Exception:
        session.close()
        raise


def get_prediction_result_repository() -> Any:
    """获取预测结果仓库（使用同步会话）"""
    session = SessionLocal()
    try:
        return PredictionResultRepository(session), session
    except Exception:
        session.close()
        raise


def get_model_info_repository() -> Any:
    """获取模型信息仓库（使用同步会话）"""
    session = SessionLocal()
    try:
        return ModelInfoRepository(session), session
    except Exception:
        session.close()
        raise
