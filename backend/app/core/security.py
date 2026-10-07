"""
密码哈希和 JWT 工具
"""

import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
import jwt

from app.core.config import settings

# === 密码哈希 ===
# 说明：原实现依赖 passlib 1.7.4 + bcrypt 5.x，存在兼容性问题（短密码也会抛
#       "password cannot be longer than 72 bytes"），这里改为直接调用 bcrypt。
MAX_BCRYPT_PASSWORD_BYTES = 72  # bcrypt 硬限制
_BCRYPT_ROUNDS = 12


def _truncate_password(password: str) -> bytes:
    """统一把密码截断到 bcrypt 兼容的最大长度（72 字节，UTF-8 编码）。"""
    if password is None:
        return b""
    if isinstance(password, str):
        encoded = password.encode("utf-8")
    else:
        encoded = bytes(password)
    if len(encoded) > MAX_BCRYPT_PASSWORD_BYTES:
        encoded = encoded[:MAX_BCRYPT_PASSWORD_BYTES]
    return encoded


def hash_password(password: str) -> str:
    """生成 bcrypt 哈希。"""
    secret = _truncate_password(password)
    salt = bcrypt.gensalt(rounds=_BCRYPT_ROUNDS)
    return bcrypt.hashpw(secret, salt).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """验证 bcrypt 哈希。"""
    if not plain_password or not hashed_password:
        return False
    secret = _truncate_password(plain_password)
    try:
        return bcrypt.checkpw(secret, hashed_password.encode("utf-8"))
    except (ValueError, TypeError):
        return False


# === JWT ===
def create_access_token(
    user_id: str,
    email: str,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """创建 JWT access token"""
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "email": email,
        "token_type": "access",
        "iat": now,
        "exp": now
        + (
            expires_delta or timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
        ),
        "jti": str(uuid.uuid4()),
    }
    secret = settings.JWT_SECRET
    return jwt.encode(payload, secret, algorithm="HS256")


def decode_access_token(token: str) -> dict:
    """解码并验证 JWT token，返回 payload"""
    secret = settings.JWT_SECRET
    try:
        payload = jwt.decode(token, secret, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        raise PermissionError("Token 已过期")
    except jwt.InvalidTokenError:
        raise PermissionError("无效的 Token")

    if payload.get("token_type", "access") != "access":
        raise PermissionError("Token 类型错误")
    return payload


def create_refresh_token(
    user_id: str,
    email: str,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """创建 JWT refresh token。"""
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "email": email,
        "token_type": "refresh",
        "iat": now,
        "exp": now
        + (expires_delta or timedelta(days=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS)),
        "jti": str(uuid.uuid4()),
    }
    return jwt.encode(payload, settings.JWT_SECRET, algorithm="HS256")


def decode_refresh_token(token: str) -> dict:
    """解码 refresh token，并拒绝 access token。"""
    try:
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        raise PermissionError("Refresh token 已过期")
    except jwt.InvalidTokenError:
        raise PermissionError("无效的 Refresh token")

    if payload.get("token_type") != "refresh":
        raise PermissionError("Token 类型错误")
    return payload
