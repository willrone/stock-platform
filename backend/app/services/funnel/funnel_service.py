"""漏斗服务层：事件写入、匿名归并、漏斗报告。

事件注册表（唯一事实源）
----------------------
前后端各埋一半，**同一个事件名只能有一个来源**，否则报告会重复计数：

    后端记录：signup / login / quota_exhausted
        —— 这三个是服务端才知道的事实（注册成功、登录成功、配额打满），
           放后端才不会被前端报错、网络失败吃掉。
    前端记录：first_screen / view_signal / view_pricing / click_upgrade / wallet_open
        —— UI 触点，只有前端知道。

新增事件名时必须同时更新：
    - 本文件 ALLOWED_EVENTS
    - frontend/src/services/funnel.ts 的 FunnelEventName
    - docs/backtest/ 下的漏斗口径文档（若有）
"""

import json
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from sqlalchemy.orm import Session

from app.models.funnel_models import FunnelEvent

# ── 事件名常量 ────────────────────────────────────────────────────────────
EVENT_SIGNUP = "signup"
EVENT_LOGIN = "login"
EVENT_FIRST_SCREEN = "first_screen"
EVENT_VIEW_SIGNAL = "view_signal"
EVENT_VIEW_PRICING = "view_pricing"
EVENT_CLICK_UPGRADE = "click_upgrade"
EVENT_WALLET_OPEN = "wallet_open"
EVENT_QUOTA_EXHAUSTED = "quota_exhausted"

BACKEND_EVENTS: frozenset = frozenset(
    {EVENT_SIGNUP, EVENT_LOGIN, EVENT_QUOTA_EXHAUSTED}
)
FRONTEND_EVENTS: frozenset = frozenset(
    {
        EVENT_FIRST_SCREEN,
        EVENT_VIEW_SIGNAL,
        EVENT_VIEW_PRICING,
        EVENT_CLICK_UPGRADE,
        EVENT_WALLET_OPEN,
    }
)
ALLOWED_EVENTS: frozenset = BACKEND_EVENTS | FRONTEND_EVENTS

# 漏斗主链路（顺序即步骤顺序）
FUNNEL_STEPS: Sequence = (
    (EVENT_SIGNUP, "注册"),
    (EVENT_FIRST_SCREEN, "首屏"),
    (EVENT_VIEW_SIGNAL, "首条信号"),
    (EVENT_VIEW_PRICING, "付费墙触点"),
    (EVENT_CLICK_UPGRADE, "升级点击"),
)

MAX_PROPS_BYTES = 2048
MAX_ANON_ID_LEN = 64
MAX_SESSION_ID_LEN = 64


def _sanitize_id(value: Optional[str], max_len: int) -> Optional[str]:
    """标识类字段清洗：只保留安全字符，避免注入/超长。"""
    if value is None:
        return None
    value = str(value).strip()
    if not value:
        return None
    cleaned = "".join(ch for ch in value if ch.isalnum() or ch in "-_")
    return cleaned[:max_len] or None


def _dump_props(props: Optional[Dict[str, Any]]) -> Optional[str]:
    """props → JSON 字符串；超限直接丢弃（不抛错，埋点不能拖垮业务）。"""
    if not props:
        return None
    try:
        payload = json.dumps(props, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return None
    if len(payload.encode("utf-8")) > MAX_PROPS_BYTES:
        return json.dumps(
            {"_truncated": True, "_bytes": len(payload.encode("utf-8"))},
            ensure_ascii=False,
        )
    return payload


def merge_anonymous_events(session: Session, anon_id: str, user_id: str) -> int:
    """把某 anon_id 下尚未归并的事件归到 user_id，返回归并条数。

    触发时机：首次带身份的埋点请求到达（此时同时拿到了 anon_id 和 token）。
    """
    if not anon_id or not user_id:
        return 0
    rows: Iterable[FunnelEvent] = (
        session.query(FunnelEvent)
        .filter(FunnelEvent.anon_id == anon_id, FunnelEvent.user_id.is_(None))
        .all()
    )
    count = 0
    for row in rows:
        row.user_id = user_id
        count += 1
    if count:
        session.commit()
    return count


def record_event(
    session: Session,
    event: str,
    *,
    user_id: Optional[str] = None,
    anon_id: Optional[str] = None,
    props: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
    source: str = "backend",
) -> Optional[FunnelEvent]:
    """写入一条漏斗事件。未知事件名返回 None（不抛错）。"""
    if event not in ALLOWED_EVENTS:
        return None

    anon_id = _sanitize_id(anon_id, MAX_ANON_ID_LEN)
    session_id = _sanitize_id(session_id, MAX_SESSION_ID_LEN)

    # 身份已知时，顺带把此前匿名的行为归并过来（注册前看了价格页也算同一人）
    if user_id and anon_id:
        merge_anonymous_events(session, anon_id, user_id)

    row = FunnelEvent(
        user_id=user_id,
        anon_id=anon_id,
        event=event,
        props=_dump_props(props),
        session_id=session_id,
        source=source,
        created_at=datetime.utcnow(),
    )
    session.add(row)
    session.commit()
    return row


def _user_key(row: FunnelEvent) -> str:
    """事件归属的人：已识别用 user_id，否则用 anon_id 兜底。"""
    if row.user_id:
        return row.user_id
    return f"anon:{row.anon_id or row.id}"


def build_report(
    session: Session,
    days: int = 30,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """生成漏斗报告。

    返回结构：
        totals    —— 窗口内事件总量、涉及人数、已识别/匿名拆分
        funnel    —— 每步：触达人数 + 顺序漏斗转化人数 + 转化率
        paywall   —— 付费墙相关指标（配额打满是服务端事实，另算）
        retention —— 注册 cohort 的 D1/D7 留存
        daily     —— 按天的事件计数
    """
    now = now or datetime.utcnow()
    since = now - timedelta(days=days)

    rows: List[FunnelEvent] = (
        session.query(FunnelEvent)
        .filter(FunnelEvent.created_at >= since)
        .order_by(FunnelEvent.created_at)
        .all()
    )

    # 每个事件 → 触达用户集合
    users_by_event: Dict[str, Set[str]] = {}
    for row in rows:
        users_by_event.setdefault(row.event, set()).add(_user_key(row))

    # 顺序漏斗：converted[i] = users(step_i) ∩ converted[i-1]
    funnel: List[Dict[str, Any]] = []
    converted: Optional[Set[str]] = None
    start_users = len(users_by_event.get(EVENT_SIGNUP, set()))
    for event, label in FUNNEL_STEPS:
        touched = users_by_event.get(event, set())
        if converted is None:
            converted = set(touched)
        else:
            converted = converted & touched
        funnel.append(
            {
                "step": event,
                "label": label,
                "users": len(touched),
                "converted": len(converted),
                "conv_from_prev": (
                    None
                    if not funnel
                    else (
                        len(converted) / funnel[-1]["converted"]
                        if funnel[-1]["converted"]
                        else None
                    )
                ),
                "conv_from_start": (
                    len(converted) / start_users if start_users else None
                ),
            }
        )

    # 留存：注册 cohort（必须有真实 user_id）
    signup_rows = [r for r in rows if r.event == EVENT_SIGNUP and r.user_id]
    first_seen: Dict[str, datetime] = {}
    for row in signup_rows:
        prev = first_seen.get(row.user_id)
        if prev is None or row.created_at < prev:
            first_seen[row.user_id] = row.created_at

    by_user_day: Dict[str, Set[Any]] = {}
    for row in rows:
        if not row.user_id:
            continue
        by_user_day.setdefault(row.user_id, set()).add(row.created_at.date())

    d1_hits = 0
    d7_hits = 0
    for user_id, first_dt in first_seen.items():
        days_seen = by_user_day.get(user_id, set())
        if (first_dt.date() + timedelta(days=1)) in days_seen:
            d1_hits += 1
        window = {
            first_dt.date() + timedelta(days=i) for i in range(1, 8)
        }
        if days_seen & window:
            d7_hits += 1

    cohort = len(first_seen)
    retention = {
        "cohort_users": cohort,
        "d1": (d1_hits / cohort) if cohort else None,
        "d7": (d7_hits / cohort) if cohort else None,
        "d1_users": d1_hits,
        "d7_users": d7_hits,
    }

    # 按天计数
    daily_map: Dict[Any, Dict[str, int]] = {}
    for row in rows:
        day = row.created_at.date()
        bucket = daily_map.setdefault(day, {})
        bucket[row.event] = bucket.get(row.event, 0) + 1
    daily = [
        {"date": str(day), **{k: v for k, v in sorted(counter.items())}}
        for day, counter in sorted(daily_map.items())
    ]

    identified = {r.user_id for r in rows if r.user_id}
    anonymous = {_user_key(r) for r in rows if not r.user_id}

    return {
        "generated_at": now.isoformat(),
        "window_days": days,
        "totals": {
            "events": len(rows),
            "users": len(identified | anonymous),
            "identified_users": len(identified),
            "anonymous_users": len(anonymous),
        },
        "funnel": funnel,
        "paywall": {
            "view_pricing_users": len(users_by_event.get(EVENT_VIEW_PRICING, set())),
            "click_upgrade_users": len(
                users_by_event.get(EVENT_CLICK_UPGRADE, set())
            ),
            "wallet_open_users": len(users_by_event.get(EVENT_WALLET_OPEN, set())),
            "quota_exhausted_users": len(
                users_by_event.get(EVENT_QUOTA_EXHAUSTED, set())
            ),
        },
        "retention": retention,
        "daily": daily,
        "events_breakdown": {
            event: len(users_by_event.get(event, set()))
            for event in sorted(ALLOWED_EVENTS)
            if users_by_event.get(event)
        },
    }
