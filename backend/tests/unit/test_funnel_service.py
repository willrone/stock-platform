"""漏斗服务单元测试。

覆盖：
    1. 事件写入 / 未知事件名拒收
    2. 匿名 → 已识别用户的归并（注册前看了价格页要能连到注册后的人）
    3. 顺序漏斗转化（后一步必须完成前一步才算转化）
    4. D1 / D7 留存（按注册 cohort）
    5. props 超限截断、标识字段清洗
"""

import json
import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from app.core.database import Base  # noqa: E402
from app.models.funnel_models import FunnelEvent  # noqa: E402
from app.models.user_models import User  # noqa: E402
from app.services.funnel import funnel_service  # noqa: E402


def _session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(
        bind=engine, tables=[User.__table__, FunnelEvent.__table__], checkfirst=True
    )
    return sessionmaker(bind=engine)()


def _mk_user(session, uid="u1", email="a@example.com"):
    user = User(id=uid, email=email, username=uid, hashed_password="x")
    session.add(user)
    session.commit()
    return user


# ── 1. 写入与校验 ─────────────────────────────────────────────────────────
def test_record_event_and_reject_unknown():
    session = _session()
    user = _mk_user(session)

    row = funnel_service.record_event(
        session,
        funnel_service.EVENT_VIEW_SIGNAL,
        user_id=str(user.id),
        props={"source": "today-signal"},
        source="frontend",
    )
    assert row is not None
    assert row.user_id == user.id
    assert json.loads(row.props)["source"] == "today-signal"

    # 未知事件名不写库、不抛错
    assert funnel_service.record_event(session, "not_a_real_event", user_id=user.id) is None
    assert session.query(FunnelEvent).filter(FunnelEvent.event == "not_a_real_event").count() == 0


# ── 2. 匿名归并 ──────────────────────────────────────────────────────────
def test_anonymous_merge_to_user():
    session = _session()
    user = _mk_user(session)

    # 注册前匿名看了价格页
    funnel_service.record_event(
        session, funnel_service.EVENT_VIEW_PRICING, anon_id="anon-abc", source="frontend"
    )
    assert session.query(FunnelEvent).filter(FunnelEvent.user_id.is_(None)).count() == 1

    # 注册（后端记录，带 anon_id）→ 匿名行为应被归并
    funnel_service.record_event(
        session,
        funnel_service.EVENT_SIGNUP,
        user_id=str(user.id),
        anon_id="anon-abc",
        source="backend",
    )
    assert session.query(FunnelEvent).filter(FunnelEvent.user_id.is_(None)).count() == 0
    assert session.query(FunnelEvent).filter(FunnelEvent.user_id == user.id).count() == 2


def test_merge_only_touches_same_anon():
    session = _session()
    user = _mk_user(session)
    funnel_service.record_event(session, funnel_service.EVENT_VIEW_SIGNAL, anon_id="other")
    funnel_service.record_event(
        session, funnel_service.EVENT_SIGNUP, user_id=user.id, anon_id="mine"
    )
    # 别人的匿名事件不能被归并
    assert session.query(FunnelEvent).filter(FunnelEvent.user_id.is_(None)).count() == 1


# ── 3. 顺序漏斗 ──────────────────────────────────────────────────────────
def test_sequential_funnel():
    session = _session()
    now = datetime.utcnow()

    # u1 走完 5 步；u2 只到第 3 步；u3 只注册
    for step, _label in funnel_service.FUNNEL_STEPS:
        funnel_service.record_event(
            session, step, user_id="u1", source="backend"
        )
    funnel_service.record_event(session, funnel_service.EVENT_SIGNUP, user_id="u2")
    funnel_service.record_event(session, funnel_service.EVENT_FIRST_SCREEN, user_id="u2")
    funnel_service.record_event(session, funnel_service.EVENT_VIEW_SIGNAL, user_id="u2")
    funnel_service.record_event(session, funnel_service.EVENT_SIGNUP, user_id="u3")

    report = funnel_service.build_report(session, days=30, now=now)
    funnel = {step["step"]: step for step in report["funnel"]}

    # 触达人数
    assert funnel[funnel_service.EVENT_SIGNUP]["users"] == 3
    assert funnel[funnel_service.EVENT_VIEW_SIGNAL]["users"] == 2
    # 顺序转化：u2 卡在第 3 步，第 4 步顺序转化只剩 u1
    assert funnel[funnel_service.EVENT_VIEW_PRICING]["converted"] == 1
    assert funnel[funnel_service.EVENT_CLICK_UPGRADE]["converted"] == 1
    assert funnel[funnel_service.EVENT_SIGNUP]["converted"] == 3
    assert funnel[funnel_service.EVENT_CLICK_UPGRADE]["conv_from_start"] == 1 / 3


# ── 4. 留存 ──────────────────────────────────────────────────────────────
def test_retention_d1_d7():
    session = _session()
    now = datetime.utcnow()

    # u1 注册于 3 天前 → D1 命中，D7 也算命中（3 天内活跃）
    def _at(days_ago, hour=10):
        return now - timedelta(days=days_ago, hours=now.hour - hour)

    funnel_service.record_event(session, funnel_service.EVENT_SIGNUP, user_id="u1")
    session.query(FunnelEvent).update(
        {"created_at": now - timedelta(days=3)}, synchronize_session=False
    )
    session.commit()
    funnel_service.record_event(session, funnel_service.EVENT_LOGIN, user_id="u1")
    session.query(FunnelEvent).filter(FunnelEvent.event == funnel_service.EVENT_LOGIN).update(
        {"created_at": now - timedelta(days=2)}, synchronize_session=False
    )
    session.commit()

    # u2 注册但之后没有任何行为 → 无留存
    funnel_service.record_event(session, funnel_service.EVENT_SIGNUP, user_id="u2")
    session.query(FunnelEvent).filter(FunnelEvent.user_id == "u2").update(
        {"created_at": now - timedelta(days=5)}, synchronize_session=False
    )
    session.commit()

    report = funnel_service.build_report(session, days=30, now=now)
    ret = report["retention"]
    assert ret["cohort_users"] == 2
    assert ret["d1_users"] == 1  # u1 在注册后第 2 天活跃 → D1 未命中？

    # u1 注册在 3 天前，活跃在 2 天前 → 注册次日(=2天前)命中 D1 ✓
    assert ret["d1_users"] == 1
    assert ret["d7_users"] == 1
    assert abs(ret["d1"] - 0.5) < 1e-9


# ── 5. 清洗与截断 ────────────────────────────────────────────────────────
def test_props_truncated_and_ids_sanitized():
    session = _session()
    big = {f"k{i}": "x" * 100 for i in range(40)}
    row = funnel_service.record_event(
        session, funnel_service.EVENT_VIEW_SIGNAL, props=big, anon_id="a b!c<script>"
    )
    assert row is not None
    assert json.loads(row.props)["_truncated"] is True
    assert row.anon_id == "abcscript"  # 非法字符被剔除
    assert row.anon_id is not None and len(row.anon_id) <= 64
