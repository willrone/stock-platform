#!/usr/bin/env python3
"""漏斗报告（CLI）—— 直接读库，不依赖 API 是否在跑。

用法：
    cd backend && env -u PYTHONPATH .venv-py313/bin/python3 scripts/funnel_report.py
    cd backend && env -u PYTHONPATH .venv-py313/bin/python3 scripts/funnel_report.py --days 7 --json

为什么要有 CLI：管理端 API 需要管理员 token，而"我现在想看一眼漏斗"
这种高频小需求不该每次都去拿 token。CLI 与 API 共用同一个
`app.services.funnel.funnel_service.build_report`，口径不会漂移。
"""

import argparse
import json
import sys
from pathlib import Path

from typing import cast

from sqlalchemy import Table

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.database import Base, SessionLocal, sync_engine  # noqa: E402
from app.models.funnel_models import FunnelEvent  # noqa: E402
from app.services.funnel import funnel_service  # noqa: E402


def _pct(value):
    if value is None:
        return "  —  "
    return f"{value * 100:5.1f}%"


def render(report: dict) -> str:
    out = []
    out.append("=" * 74)
    out.append(
        f"漏斗报告（窗口 {report['window_days']} 天，截至 {report['generated_at'][:16]}）"
    )
    out.append("=" * 74)
    totals = report["totals"]
    out.append(
        f"总量：事件 {totals['events']} 条 · 涉及 {totals['users']} 人"
        f"（已识别 {totals['identified_users']} / 匿名 {totals['anonymous_users']}）"
    )
    out.append("")
    out.append("漏斗（顺序转化：后一步必须完成前一步）")
    out.append(f"  {'步骤':<12}{'触达':>6}{'顺序转化':>10}{'上一步':>9}{'起点':>9}")
    for step in report["funnel"]:
        out.append(
            f"  {step['label']:<12}{step['users']:>6}{step['converted']:>10}"
            f"{_pct(step['conv_from_prev']):>9}{_pct(step['conv_from_start']):>9}"
        )
    out.append("")
    paywall = report["paywall"]
    out.append("付费墙")
    out.append(
        f"  看价格页 {paywall['view_pricing_users']} 人 · "
        f"点升级 {paywall['click_upgrade_users']} 人 · "
        f"开钱包 {paywall['wallet_open_users']} 人 · "
        f"配额打满 {paywall['quota_exhausted_users']} 人"
    )
    ret = report["retention"]
    out.append("")
    out.append(f"留存（注册 cohort {ret['cohort_users']} 人）")
    out.append(f"  D1 {_pct(ret['d1'])}（{ret['d1_users']} 人）  D7 {_pct(ret['d7'])}（{ret['d7_users']} 人）")
    if report["daily"]:
        out.append("")
        out.append("按天")
        for day in report["daily"]:
            detail = " · ".join(
                f"{k}={v}" for k, v in day.items() if k != "date" and v
            )
            out.append(f"  {day['date']}  {detail}")
    if report["events_breakdown"]:
        out.append("")
        out.append("事件明细（触达人数）")
        for event, users in sorted(report["events_breakdown"].items()):
            out.append(f"  {event:<20}{users}")
    out.append("=" * 74)
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description="漏斗报告")
    parser.add_argument("--days", type=int, default=30, help="统计窗口天数（默认 30）")
    parser.add_argument("--json", action="store_true", help="输出原始 JSON")
    args = parser.parse_args()

    # 表可能还不存在（后端首次启动前）—— 只建漏斗这一张，不动其它表
    Base.metadata.create_all(
        bind=sync_engine,
        tables=[cast(Table, FunnelEvent.__table__)],
        checkfirst=True,
    )

    session = SessionLocal()
    try:
        report = funnel_service.build_report(session, days=args.days)
    finally:
        session.close()

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(render(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
