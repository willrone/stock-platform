"""真实用户漏斗埋点模型。

为什么需要这张表：
    平台 23 个用户全是测试号、真实用户 0。没有漏斗数据时，"用户卡在哪一环"
    只能靠猜；等真用户进来再补埋点，就永远答不出"流失在哪一步"。

设计要点：
    1. user_id 可空 —— 注册前的行为（看价格页、点升级）也要留痕，用 anon_id 认。
    2. anon_id → user_id 的归并（merge）发生在首次带身份的事件到达时，
       这样"注册前看了价格页 → 注册"能连成一条路径。
    3. created_at 一律存 naive UTC（与项目其余表一致，datetime.utcnow）。
    4. 用 SQLAlchemy 2.0 的 Mapped[] 标注（不是仓库里旧的 Column(...) 写法），
       这样属性在类型检查器里就是 str / datetime，而不是 Column[str]。
"""

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, ForeignKey, Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


def generate_uuid() -> str:
    return str(uuid.uuid4())


class FunnelEvent(Base):
    """漏斗事件流水（只追加，不更新、不删除）。"""

    __tablename__ = "funnel_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)

    # 已识别用户；注册前事件为 NULL
    user_id: Mapped[Optional[str]] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    # 匿名设备标识（前端 localStorage 生成），归并到 user_id 后仍保留以便追溯
    anon_id: Mapped[Optional[str]] = mapped_column(
        String(64), nullable=True, index=True
    )

    # 事件名，取值见 app.services.funnel.funnel_service.ALLOWED_EVENTS
    event: Mapped[str] = mapped_column(String(50), nullable=False, index=True)

    # 附加属性（JSON 字符串，≤2KB）
    props: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # 前端会话标识
    session_id: Mapped[Optional[str]] = mapped_column(
        String(64), nullable=True, index=True
    )

    # 事件来源：backend / frontend
    source: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False, index=True
    )

    __table_args__ = (
        Index("ix_funnel_events_event_created", "event", "created_at"),
        Index("ix_funnel_events_user_created", "user_id", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<FunnelEvent {self.event} user={self.user_id} anon={self.anon_id}>"
