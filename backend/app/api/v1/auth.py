"""用户认证 API 路由。"""

import secrets
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, status
from loguru import logger
from pydantic import BaseModel, Field
from sqlalchemy.exc import IntegrityError, OperationalError, SQLAlchemyError

from app.api.v1.dependencies import require_current_user
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

router = APIRouter(prefix="/auth", tags=["auth"])


class RegisterRequest(BaseModel):
    email: str = Field(..., examples=["user@example.com"])
    username: str = Field(..., min_length=2, max_length=50, examples=["quant_wang"])
    password: str = Field(..., min_length=6, max_length=128)


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


class AuthResponse(BaseModel):
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


class ForgotPasswordResponse(BaseModel):
    message: str
    # 仅开发/测试环境返回，生产环境通过邮件发送且不在 API 中暴露。
    reset_token: Optional[str] = None


class MessageResponse(BaseModel):
    message: str


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
    }


def _tokens_for_user(user: User) -> tuple[str, str]:
    return (
        create_access_token(user_id=user.id, email=user.email),
        create_refresh_token(user_id=user.id, email=user.email),
    )


def _auth_response(user: User) -> AuthResponse:
    access_token, refresh_token = _tokens_for_user(user)
    return AuthResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        user=_user_payload(user),
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
        return _auth_response(user)
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
        return _auth_response(user)
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
        return _auth_response(user)
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
