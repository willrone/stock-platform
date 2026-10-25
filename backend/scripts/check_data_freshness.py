#!/usr/bin/env python3
"""数据新鲜度校验（2026-10-25 新增）

背景：2026-10-23（周五）16:05 拉 TDX 盘后包时，当天的包还没发布
（TDX 通常 17:00 后才出当日日线），被 pipeline 静默记为「非交易日」。
随后所有判断「今天是不是周五」的代码都用「数据最新日」而不是真实日期，
于是周频调仓整整两周没跑过 —— 信号日期停在 10-16，数据却到 10-22。

本脚本在拉取后做校验：
  - 今天是交易日 且 已过发布时刻(默认 17:30) → 数据最新日必须 >= 今天，否则算陈旧
  - 输出人可读的结论；--fail-if-stale 时陈旧则退出码 1（让 cron 报警到微信）
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import date, datetime, time as dtime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "back_test_data_service", "tdx_pipeline"))

from trading_calendar import is_trading_day  # noqa: E402

RAW_DIR = os.path.join(ROOT, "data", "tdx_daily", "raw")
# TDX 当日日线包的稳定发布时刻（早于此点当天包多半还没出）
PUBLISH_TIME = dtime(17, 30)


def latest_partition_date() -> str | None:
    if not os.path.isdir(RAW_DIR):
        return None
    dates = sorted(
        d.split("=", 1)[1]
        for d in os.listdir(RAW_DIR)
        if d.startswith("date=")
    )
    return dates[-1] if dates else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--fail-if-stale",
        action="store_true",
        help="数据陈旧时以退出码 1 结束（用于 cron 报警）",
    )
    ap.add_argument(
        "--publish-time",
        default="17:30",
        help="当日盘后包的发布时刻（默认 17:30）",
    )
    ap.add_argument(
        "--now",
        default="",
        help="覆盖当前时间（测试用，格式 YYYY-MM-DD HH:MM）",
    )
    ap.add_argument(
        "--latest-override",
        default="",
        help="覆盖数据最新日（测试用，格式 YYYY-MM-DD）",
    )
    args = ap.parse_args()

    hh, mm = (int(x) for x in args.publish_time.split(":"))
    publish_time = dtime(hh, mm)

    if args.now:
        now = datetime.strptime(args.now, "%Y-%m-%d %H:%M")
    else:
        now = datetime.now()

    today = now.date()
    today_str = today.isoformat()
    latest = args.latest_override or latest_partition_date()
    td = is_trading_day(today_str)

    print("=== 数据新鲜度校验 ===")
    print(f"当前时间   : {now:%Y-%m-%d %H:%M:%S} ({today.strftime('%A')})")
    print(f"数据最新日 : {latest or '（无数据）'}")
    print(f"发布时刻   : {args.publish_time}")
    print(f"今日是交易日: {td}")

    if latest is None:
        print("❌ 没有任何数据分区")
        return 1 if args.fail_if_stale else 0

    latest_d = datetime.strptime(latest, "%Y-%m-%d").date()

    # 非交易日 → 数据停在上一个交易日即正常
    if td is False:
        print(f"✅ 今天不是交易日，数据停在 {latest} 属正常")
        return 0

    # 交易日
    if latest_d >= today:
        print(f"✅ 数据已是最新（{latest}）")
        return 0

    # 交易日但数据落后
    gap = (today - latest_d).days
    if now.time() < publish_time:
        print(
            f"⏳ 今天 {today_str} 是交易日，但还没到发布时刻 {args.publish_time}，"
            f"数据停在 {latest}（差 {gap} 天）—— 属正常，等待后续重试"
        )
        return 0

    print(
        f"❌ 今天是交易日且已过发布时刻 {args.publish_time}，"
        f"但数据仍停在 {latest}（差 {gap} 天）—— 数据陈旧！"
    )
    print("   下游信号/回测会用到过期数据，请检查 TDX 拉取是否失败。")
    return 1 if args.fail_if_stale else 0


if __name__ == "__main__":
    sys.exit(main())
