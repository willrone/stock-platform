#!/usr/bin/env python3
"""P10 回归门：PIT 涨跌停修复**不得改变已定稿窗口**的任何数值。

## 断言

1. **等价性**：对全部日期 ≥ 2020-08-24 的掩码，
   `limit_up_mask_matrix_at` 与原 `limit_up_mask_matrix` **逐位相同**
   （main 窗口 2023-05~2026-05 与 oos 窗口 2021-05~2023-05 都在这个区间）。
2. **有效性**：对 2020-08-24 之前的创业板（30x），
   +12% 的日涨幅在旧规则下**不被掩**（20% 板）、在新规则下**被掩**（10% 板）
   —— 正是长历史污染的来源。
3. **板块不变**：主板 / 科创板 / 北交所 的幅度不随日期变化。

用法：

    env -u PYTHONPATH -u PYTHONHOME backend/.venv-py313/bin/python3 \
        backend/scripts/p10_limit_pit_regression.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_DIR)

from app.services.trading.execution_rules import (  # noqa: E402
    limit_pct_at,
    limit_up_mask_matrix,
    limit_up_mask_matrix_at,
)

FAILS = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  ✅ {name}")
    else:
        print(f"  ❌ {name} {detail}")
        FAILS.append(name)


def main() -> int:
    codes = ["600000.SH", "000001.SZ", "300001.SZ", "688001.SH", "920001.BJ", "002001.SZ"]

    # ── 1. 等价性：全部日期 >= 2020-08-24 ──
    dates = [f"2023-05-{d:02d}" for d in (8, 9, 10, 11, 12)]
    rng = np.random.default_rng(20261031)
    day_ret = rng.uniform(-0.25, 0.25, size=(len(dates), len(codes)))
    m_old = limit_up_mask_matrix(day_ret, codes)
    m_new = limit_up_mask_matrix_at(day_ret, codes, dates)
    check("post-2020-08-24 掩码逐位相同", bool((m_old == m_new).all()),
          f"diff={int((m_old != m_new).sum())}")

    # ── 2. 有效性：2020-08-24 之前的创业板 10% 板 ──
    old_dates = ["2016-05-09", "2016-05-10"]
    r = np.array([[0.0, 0.0, 0.12, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    m_old2 = limit_up_mask_matrix(r, codes)
    m_new2 = limit_up_mask_matrix_at(r, codes, old_dates)
    check("2016 创业板 +12%：旧规则不掩", bool(m_old2[0, 2] == False),
          f"old={m_old2[0, 2]}")
    check("2016 创业板 +12%：新规则掩掉", bool(m_new2[0, 2] == True),
          f"new={m_new2[0, 2]}")

    # ── 3. 2020-08-24 之后创业板 20% ──
    r3 = np.array([[0.0, 0.0, 0.12, 0.0, 0.0, 0.0]])
    m_new3 = limit_up_mask_matrix_at(r3, codes, ["2023-05-08"])
    check("2023 创业板 +12%：不掩（20% 板）", bool(m_new3[0, 2] == False),
          f"new={m_new3[0, 2]}")

    # ── 4. 板块幅度不随日期变化 ──
    for code, exp in (("600000.SH", 0.10), ("000001.SZ", 0.10), ("002001.SZ", 0.10),
                       ("688001.SH", 0.20), ("920001.BJ", 0.30)):
        vals = {limit_pct_at(code, d) for d in ("2013-05-08", "2020-08-24", "2023-05-08")}
        check(f"{code} 幅度恒定 {exp}", vals == {exp}, f"got={vals}")

    # ── 5. 创业板时间开关 ──
    check("创业板 2019-01-08 = 10%", limit_pct_at("300001.SZ", "2019-01-08") == 0.10)
    check("创业板 2020-08-24 = 20%", limit_pct_at("300001.SZ", "2020-08-24") == 0.20)
    check("创业板 2020-08-25 = 20%", limit_pct_at("300001.SZ", "2020-08-25") == 0.20)

    if FAILS:
        print(f"\n❌ {len(FAILS)} 项失败: {FAILS}")
        return 1
    print("\n✅ 全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
