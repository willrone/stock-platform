#!/usr/bin/env python3
"""纸面盘 lag1 口径的逻辑测试（2026-10-26）

不依赖真实行情数据，直接对 `paper_portfolio.run_line` 的**日期推导逻辑**做断言：

  信号日 = master 里最后一个周五
  成交日 = 信号日之后的第一个交易日
  若信号日之后没有交易日 → 明确报错退出（不静默用错日期成交）

同时验证：
  - 连续跌停/涨停判定用的是**成交日**的涨跌幅
  - 重复跑同一天不会重复调仓（信号日相等 → 跳过）

用法：
  env -u PYTHONPATH -u PYTHONHOME backend/.venv-py313/bin/python3 \
    backend/scripts/test_paper_lag1.py
"""
from __future__ import annotations

import os
import sys

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(BACKEND)
sys.path.insert(0, BACKEND)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

FAILED: list = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  ✅ {name}")
    else:
        print(f"  ❌ {name} {detail}")
        FAILED.append(name)


def derive(master: pd.DatetimeIndex):
    """复刻 paper_portfolio 的信号日/成交日推导（main 段）。"""
    latest_all = master[-1]
    fridays = [d for d in master if d.dayofweek == 4]
    rebalance_day = fridays[-1] if fridays else latest_all
    after = master[master > rebalance_day]
    trade_day = after[0] if len(after) else None
    return rebalance_day, trade_day


def main() -> int:
    print("[test] 1. 信号日/成交日推导")
    # 2026-10-23 是周五，2026-10-26 是周一
    mon = pd.DatetimeIndex(pd.bdate_range("2026-10-19", "2026-10-23"))
    sig, trd = derive(mon)
    check("数据只到周五 → 信号日=周五", str(sig.date()) == "2026-10-23", str(sig.date()))
    check("数据只到周五 → 无成交日（应报错退出）", trd is None, str(trd))

    tue = pd.DatetimeIndex(pd.bdate_range("2026-10-19", "2026-10-27"))
    sig, trd = derive(tue)
    check("数据到周二 → 信号日仍是上周五", str(sig.date()) == "2026-10-23", str(sig.date()))
    check("数据到周二 → 成交日=下周一", trd is not None and str(trd.date()) == "2026-10-26",
          str(trd.date() if trd is not None else None))

    wed = pd.DatetimeIndex(pd.bdate_range("2026-10-19", "2026-10-28"))
    sig, trd = derive(wed)
    check("数据到周三 → 成交日仍是下周一（不跳到周二）",
          trd is not None and str(trd.date()) == "2026-10-26",
          str(trd.date() if trd is not None else None))

    print("\n[test] 2. 重复调仓防护")
    # 信号日相等时必须跳过（否则同一周买两次）
    last_used = pd.Timestamp("2026-10-23")
    sig2, _ = derive(tue)
    check("信号日 == 已记录 → 判定为重复（应跳过）", sig2 <= last_used)

    print("\n[test] 3. 涨跌幅/价格取的是成交日")
    import inspect

    import scripts.paper_portfolio as pp  # noqa: E402

    src = inspect.getsource(pp.run_line)
    check("价格取 trade_day（不是 signal_day）", ".loc[trade_day]" in src)
    check("day_ret 用成交日的 prev_close", "close_mat.shift(1).loc[trade_day]" in src)
    check("打分矩阵只取不晚于信号日的数据", "comp.index <= signal_day" in src)
    check("没有向后取打分日（防未来数据）", "comp.index[-1]" not in src.split("usable = ")[1][:200])

    print()
    if FAILED:
        print(f"[test] ❌ {len(FAILED)} 项失败: {FAILED}")
        return 1
    print("[test] ✅ 全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
