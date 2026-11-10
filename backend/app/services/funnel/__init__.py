"""漏斗服务。

事件注册表在 funnel_service 里（唯一事实源），前端对应
frontend/src/services/funnel.ts 的 FunnelEventName。
"""

from .funnel_service import (
    ALLOWED_EVENTS,
    BACKEND_EVENTS,
    EVENT_CLICK_UPGRADE,
    EVENT_FIRST_SCREEN,
    EVENT_LOGIN,
    EVENT_QUOTA_EXHAUSTED,
    EVENT_SIGNUP,
    EVENT_VIEW_PRICING,
    EVENT_VIEW_SIGNAL,
    EVENT_WALLET_OPEN,
    FRONTEND_EVENTS,
    FUNNEL_STEPS,
    build_report,
    merge_anonymous_events,
    record_event,
)

__all__ = [
    "ALLOWED_EVENTS",
    "BACKEND_EVENTS",
    "FRONTEND_EVENTS",
    "FUNNEL_STEPS",
    "EVENT_SIGNUP",
    "EVENT_LOGIN",
    "EVENT_FIRST_SCREEN",
    "EVENT_VIEW_SIGNAL",
    "EVENT_VIEW_PRICING",
    "EVENT_CLICK_UPGRADE",
    "EVENT_WALLET_OPEN",
    "EVENT_QUOTA_EXHAUSTED",
    "record_event",
    "merge_anonymous_events",
    "build_report",
]
