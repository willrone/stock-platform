"""用户认证 API 路由。"""

import secrets
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, status
from loguru import logger
from pydantic import BaseModel, Field
from sqlalchemy.exc import IntegrityError, OperationalError, SQLAlchemyError

from app.api.v1.dependencies import require_current_user
from app.api.v1.schemas import StandardResponse
from app.core.config import settings
from app.core.database import SessionLocal
from app.core.security import (
    create_access_token,
    create_refresh_token,
    decode_access_token,
    decode_refresh_token,
    hash_password,
    verify_password,
)
from app.models.user_models import User
from app.services.billing.email_service import send_password_reset_email
from app.services.funnel import funnel_service

router = APIRouter(prefix="/auth", tags=["auth"])


class RegisterRequest(BaseModel):
    email: str = Field(..., examples=["user@example.com"])
    username: str = Field(..., min_length=2, max_length=50, examples=["quant_wang"])
    password: str = Field(..., min_length=6, max_length=128)
    # 2026-11-01 漏斗埋点：前端生成的匿名设备标识。带上它才能把
    # "注册前看了价格页" 这类匿名行为归并到注册后的 user_id 上。
    anon_id: Optional[str] = Field(None, max_length=64)


class LoginRequest(BaseModel):
    email: str = Field(..., examples=["user@example.com"])
    password: str = Field(..., min_length=1)


class RefreshTokenRequest(BaseModel):
    refresh_token: str = Field(..., min_length=1)


class ForgotPasswordRequest(BaseModel):
    email: str = Field(..., examples=["user@example.com"])


class ResetPasswordRequest(BaseModel):
    token: str = Field(..., min_length=1)
    password: str = Field(..., min_length=6, max_length=128)


class ProfileUpdateRequest(BaseModel):
    username: str = Field(..., min_length=2, max_length=50)


class AuthResponse(StandardResponse):
    """认证响应（信封 + 顶层冗余字段，双形态）。

    2026-10-27：登录/注册原先只返回裸 dict（access_token/refresh_token/
    token_type/user），与项目其它 40+ 端点的 StandardResponse 信封不一致；
    前端 api.ts 的响应拦截器要求 success 字段，任何走信封路径读 token 的
    逻辑都会拿到 undefined。

    这里同时给出两种形态，兼容过渡期的新旧前端：
      - 信封形态：success / message / data / timestamp，data 内含四个 token 字段；
      - 顶层形态：access_token / refresh_token / token_type / user 直接在响应体顶层。

    顶层字段是**过渡期兼容冗余**（frontend/src/app/login/page.tsx:46 与
    register/page.tsx:51 写的是 localStorage.setItem('access_token', data.access_token)，
    其中 data 是 axios response.data 即响应体顶层）。后端全量统一、
    前端全部改为读 data.access_token 之后，可以删除这里的顶层冗余字段。

    注意：字段顺序即 JSON 输出顺序 —— 基类字段在前（success/message/data/
    timestamp），子类字段在后（access_token/...）。
    """

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user: dict


class UserInfo(BaseModel):
    id: str
    email: str
    username: str
    is_active: bool
    subscription_tier: str
    created_at: datetime
    email_verified: bool = False
    avatar_url: Optional[str] = None
    last_login_at: Optional[datetime] = None
    # 2026-10-25：前端需要按角色显示管理端入口（如「充值订单」确认到账）
    is_admin: bool = False


class ForgotPasswordResponse(BaseModel):
    message: str
    # 仅开发/测试环境返回，生产环境通过邮件发送且不在 API 中暴露。
    reset_token: Optional[str] = None


class MessageResponse(BaseModel):
    message: str


def _track_funnel(session, event: str, **kwargs) -> None:
    """记录漏斗事件。**埋点失败绝不影响主流程**（注册/登录必须成功）。"""
    try:
        funnel_service.record_event(session, event, **kwargs)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"漏斗埋点失败（忽略）: {exc}")
        try:
            session.rollback()
        except Exception:  # noqa: BLE001
            pass


def _get_user_by_email(session, email: str):
    return session.query(User).filter(User.email == email).first()


def _get_user_by_id(session, user_id: str):
    return session.query(User).filter(User.id == user_id).first()


def _user_payload(user: User) -> dict:
    return {
        "id": user.id,
        "email": user.email,
        "username": user.username,
        "subscription_tier": user.subscription_tier,
        # 2026-10-25：前端按角色显示管理端入口（充值订单确认到账）
        "is_admin": bool(getattr(user, "is_admin", False)),
    }


def _tokens_for_user(user: User) -> tuple[str, str]:
    return (
        create_access_token(user_id=user.id, email=user.email),
        create_refresh_token(user_id=user.id, email=user.email),
    )


def _auth_response(user: User, message: str = "认证成功") -> AuthResponse:
    """构造认证响应（信封 + 顶层冗余字段，双形态）。

    - data 里放 access_token/refresh_token/token_type/user（信封路径消费）
    - 同样的四个字段同时冗余在响应体顶层（旧前端路径消费）
    """
    access_token, refresh_token = _tokens_for_user(user)
    user_payload = _user_payload(user)
    token_payload = {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "user": user_payload,
    }
    return AuthResponse(
        success=True,
        message=message,
        data=token_payload,
        # 顶层冗余：过渡期兼容旧前端（详见 AuthResponse 文档字符串）
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        user=user_payload,
    )


@router.post("/register", response_model=AuthResponse, status_code=201)
def register(req: RegisterRequest):
    session = SessionLocal()
    try:
        if len(req.password) < 6:
            raise HTTPException(status_code=400, detail="密码至少 6 位")
        if _get_user_by_email(session, req.email):
            raise HTTPException(status_code=409, detail="该邮箱已注册")
        if session.query(User).filter(User.username == req.username).first():
            raise HTTPException(status_code=409, detail="该用户名已被使用")

        user = User(
            email=req.email,
            username=req.username,
            hashed_password=hash_password(req.password),
        )
        session.add(user)
        try:
            session.commit()
        except IntegrityError as integrity_err:
            session.rollback()
            logger.warning("注册唯一约束冲突: {}", integrity_err)
            raise HTTPException(status_code=409, detail="邮箱或用户名已被使用")
        session.refresh(user)
        # 漏斗：注册成功（服务端事实，前端不重复上报）
        _track_funnel(
            session,
            funnel_service.EVENT_SIGNUP,
            user_id=user.id,
            anon_id=req.anon_id,
            props={"username": user.username},
            source="backend",
        )
        return _auth_response(user, message="注册成功")
    except HTTPException:
        raise
    except OperationalError as op_err:
        session.rollback()
        logger.opt(exception=True).error("注册时数据库不可用: {}", op_err)
        raise HTTPException(status_code=503, detail="数据库暂时不可用，请稍后重试")
    except SQLAlchemyError as db_err:
        session.rollback()
        logger.opt(exception=True).error("注册时数据库错误: {}", db_err)
        raise HTTPException(status_code=500, detail="注册失败（数据库错误）")
    except Exception as exc:
        session.rollback()
        logger.opt(exception=True).error("注册未预期异常: {}", exc)
        raise HTTPException(status_code=500, detail=f"注册失败: {exc}")
    finally:
        session.close()


@router.post("/login", response_model=AuthResponse)
def login(req: LoginRequest):
    session = SessionLocal()
    try:
        user = _get_user_by_email(session, req.email)
        if not user or not verify_password(req.password, user.hashed_password):
            raise HTTPException(status_code=401, detail="邮箱或密码错误")
        if not user.is_active:
            raise HTTPException(status_code=403, detail="账户已被禁用")

        user.last_login_at = datetime.utcnow()
        session.commit()
        session.refresh(user)
        # 漏斗：登录成功（服务端事实）
        _track_funnel(
            session,
            funnel_service.EVENT_LOGIN,
            user_id=user.id,
            source="backend",
        )
        return _auth_response(user, message="登录成功")
    except HTTPException:
        raise
    except OperationalError as op_err:
        session.rollback()
        logger.opt(exception=True).error("登录时数据库不可用: {}", op_err)
        raise HTTPException(status_code=503, detail="数据库暂时不可用，请稍后重试")
    except SQLAlchemyError as db_err:
        session.rollback()
        logger.opt(exception=True).error("登录时数据库错误: {}", db_err)
        raise HTTPException(status_code=500, detail="登录失败（数据库错误）")
    except Exception as exc:
        session.rollback()
        logger.opt(exception=True).error("登录未预期异常: {}", exc)
        raise HTTPException(status_code=500, detail=f"登录失败: {exc}")
    finally:
        session.close()


@router.post("/refresh", response_model=AuthResponse)
def refresh_token(req: RefreshTokenRequest):
    """使用 refresh token 换发新的 access/refresh token。"""
    try:
        payload = decode_refresh_token(req.refresh_token)
    except PermissionError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    session = SessionLocal()
    try:
        user = _get_user_by_id(session, str(payload.get("sub", "")))
        if user is None:
            raise HTTPException(status_code=401, detail="用户不存在")
        if not user.is_active:
            raise HTTPException(status_code=403, detail="账户已被禁用")
        return _auth_response(user, message="Token 刷新成功")
    finally:
        session.close()


@router.post("/forgot-password", response_model=ForgotPasswordResponse)
def forgot_password(req: ForgotPasswordRequest):
    """生成短期密码重置 token；生产环境不泄露账户是否存在。"""
    session = SessionLocal()
    try:
        user = _get_user_by_email(session, req.email)
        reset_token = None
        if user:
            reset_token = secrets.token_urlsafe(32)
            user.reset_token = reset_token
            user.reset_token_expires_at = datetime.utcnow() + timedelta(hours=1)
            session.commit()
            logger.info("已生成密码重置 token: user_id={}", user.id)

        expose_token = settings.ENVIRONMENT in ("development", "test")
        # 生产环境发送邮件；开发/测试环境直接返回 token
        if not expose_token and reset_token:
            send_password_reset_email(req.email, reset_token)
        return ForgotPasswordResponse(
            message="如果该邮箱存在，密码重置链接将发送到邮箱",
            reset_token=reset_token if expose_token else None,
        )
    except SQLAlchemyError as exc:
        session.rollback()
        logger.opt(exception=True).error("生成密码重置 token 失败: {}", exc)
        raise HTTPException(status_code=500, detail="生成密码重置 token 失败")
    finally:
        session.close()


@router.post("/reset-password", response_model=MessageResponse)
def reset_password(req: ResetPasswordRequest):
    session = SessionLocal()
    try:
        user = (
            session.query(User)
            .filter(
                User.reset_token == req.token,
                User.reset_token_expires_at > datetime.utcnow(),
            )
            .first()
        )
        if user is None:
            raise HTTPException(status_code=400, detail="重置 token 无效或已过期")

        user.hashed_password = hash_password(req.password)
        user.reset_token = None
        user.reset_token_expires_at = None
        session.commit()
        return MessageResponse(message="密码重置成功")
    except HTTPException:
        raise
    except SQLAlchemyError as exc:
        session.rollback()
        logger.opt(exception=True).error("重置密码失败: {}", exc)
        raise HTTPException(status_code=500, detail="重置密码失败")
    finally:
        session.close()


@router.put("/profile", response_model=UserInfo)
def update_profile(
    req: ProfileUpdateRequest,
    current_user: User = Depends(require_current_user),
):
    session = SessionLocal()
    try:
        user = _get_user_by_id(session, current_user.id)
        if user is None:
            raise HTTPException(status_code=404, detail="用户不存在")
        existing = (
            session.query(User)
            .filter(User.username == req.username, User.id != user.id)
            .first()
        )
        if existing:
            raise HTTPException(status_code=409, detail="该用户名已被使用")
        user.username = req.username
        session.commit()
        session.refresh(user)
        return UserInfo(
            id=user.id,
            email=user.email,
            username=user.username,
            is_active=user.is_active,
            subscription_tier=user.subscription_tier,
            created_at=user.created_at,
            email_verified=user.email_verified,
            avatar_url=user.avatar_url,
            last_login_at=user.last_login_at,
            is_admin=bool(getattr(user, "is_admin", False)),
        )
    except HTTPException:
        raise
    except IntegrityError as exc:
        session.rollback()
        logger.warning("更新用户资料唯一约束冲突: {}", exc)
        raise HTTPException(status_code=409, detail="该用户名已被使用")
    finally:
        session.close()


async def get_current_user_from_auth(
    authorization: str | None = Header(default=None, alias="Authorization"),
) -> str:
    """从 Authorization: Bearer <token> 中解析当前用户 ID。"""
    if not authorization:
        raise HTTPException(status_code=401, detail="缺少 Authorization 请求头")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise HTTPException(
            status_code=401, detail="Authorization 格式应为 Bearer <token>"
        )
    try:
        payload = decode_access_token(token.strip())
        return str(payload["sub"])
    except (PermissionError, KeyError) as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


@router.get("/me", response_model=UserInfo)
def get_profile(user_id: str = Depends(get_current_user_from_auth)):
    session = SessionLocal()
    try:
        user = _get_user_by_id(session, user_id)
        if not user:
            raise HTTPException(status_code=404, detail="用户不存在")
        return UserInfo(
            id=user.id,
            email=user.email,
            username=user.username,
            is_active=user.is_active,
            subscription_tier=user.subscription_tier,
            created_at=user.created_at,
            email_verified=user.email_verified,
            avatar_url=user.avatar_url,
            last_login_at=user.last_login_at,
            is_admin=bool(getattr(user, "is_admin", False)),
        )
    except HTTPException:
        raise
    except OperationalError as op_err:
        session.rollback()
        logger.opt(exception=True).error("/auth/me 数据库不可用: {}", op_err)
        raise HTTPException(status_code=503, detail="数据库暂时不可用，请稍后重试")
    except SQLAlchemyError as db_err:
        session.rollback()
        logger.opt(exception=True).error("/auth/me 数据库错误: {}", db_err)
        raise HTTPException(status_code=500, detail="获取用户信息失败（数据库错误）")
    finally:
        session.close()
