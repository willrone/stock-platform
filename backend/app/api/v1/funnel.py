"""漏斗埋点 API。

    POST /api/v1/funnel/track   —— 公开（可选带 token；带了就做身份归并）
    GET  /api/v1/funnel/report  —— 仅管理员

设计取舍：
    - track 必须**永不抛错影响业务**：埋点失败对用户是不可见的，
      所以异常一律吞掉并返回 204/200。
    - track 走公开路由：注册前的事件也要能落库（此时没有 token）。
    - report 只给管理员：里面含用户数等经营数据。
"""

from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, Header, Query, status
from loguru import logger
from pydantic import BaseModel, Field

from app.api.v1.dependencies import require_admin_user
from app.core.database import SessionLocal
from app.services.funnel import funnel_service

public_router = APIRouter(prefix="/funnel", tags=["funnel"])
router = APIRouter(prefix="/funnel", tags=["funnel"])


class TrackRequest(BaseModel):
    """埋点请求体。event 取值见 app.services.funnel.funnel_service.ALLOWED_EVENTS。"""

    event: str = Field(..., max_length=50, examples=["view_signal"])
    props: Optional[Dict[str, Any]] = None
    anon_id: Optional[str] = Field(None, max_length=64)
    session_id: Optional[str] = Field(None, max_length=64)


def _optional_user_id(authorization: Optional[str]) -> Optional[str]:
    """从 Bearer token 解析 user_id；解析失败返回 None（不报错）。"""
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return None
    try:
        from app.core.security import decode_access_token

        payload = decode_access_token(token.strip())
        sub = payload.get("sub")
        return str(sub) if sub else None
    except Exception as exc:  # noqa: BLE001 —— 埋点不该因 token 问题失败
        logger.debug(f"漏斗埋点 token 解析失败（忽略）: {exc}")
        return None


@public_router.post("/track", status_code=status.HTTP_202_ACCEPTED)
def track(
    req: TrackRequest, authorization: Optional[str] = Header(None)
) -> Dict[str, Any]:
    """写入一条埋点事件。始终返回 202；未知事件名返回 ok=false。"""
    if req.event not in funnel_service.ALLOWED_EVENTS:
        return {"ok": False, "reason": "unknown_event"}

    user_id = _optional_user_id(authorization)
    session = SessionLocal()
    try:
        funnel_service.record_event(
            session,
            req.event,
            user_id=user_id,
            anon_id=req.anon_id,
            props=req.props,
            session_id=req.session_id,
            source="frontend",
        )
        return {"ok": True}
    except Exception as exc:  # noqa: BLE001
        session.rollback()
        logger.warning(f"漏斗埋点写入失败（忽略）: {exc}")
        return {"ok": False, "reason": "write_failed"}
    finally:
        session.close()


@router.get("/report")
def report(
    days: int = Query(30, ge=1, le=365, description="统计窗口天数"),
    _admin: Any = Depends(require_admin_user),
) -> Dict[str, Any]:
    """返回漏斗报告（仅管理员）。"""
    session = SessionLocal()
    try:
        return funnel_service.build_report(session, days=days)
    finally:
        session.close()
