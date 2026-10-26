"""baostock 字段单位回归测试 —— 2026-10-25

## 为什么要有这个测试
baostock 的 `pct_chg` 单位是**百分数**（0.5155 表示 +0.5155%），
而我们的规则引擎按**小数**用（0.5155 = +51.55%）。

踩过的坑：东莞控股 000828.SZ 在 2026-10-23 被报成
「涨停 51.5%（上限 10%），买不进」—— 实际它当天只涨了 0.515%。
放大 100 倍的后果是**所有股票都会被误判成涨跌停**，信号全被拦死。

这个测试锁住单位换算，防止回归。
"""
from __future__ import annotations

import os
import sys

import pandas as pd

BACKEND = "/Users/rone/Projects/stock-platform/backend"
sys.path.insert(0, BACKEND)
sys.path.insert(0, os.path.join(BACKEND, "scripts"))

from scripts.market_data import load_market_extras  # noqa: E402

BS_DIR = os.path.join(BACKEND, "..", "data", "parquet", "baostock_daily")
FAILURES: list[str] = []


def check(name: str, cond: bool, extra: str = "") -> None:
    print(f"  {'✅' if cond else '❌'} {name} {extra}")
    if not cond:
        FAILURES.append(name)


def main() -> int:
    print("=== baostock 字段单位 ===")
    test_code = "000828.SZ"
    path = os.path.join(BS_DIR, "000828_SZ.parquet")
    if not os.path.exists(path):
        print("⏭  跳过（baostock 数据尚未下载）")
        return 0

    raw = pd.read_parquet(path, columns=["date", "close", "pct_chg"])
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw.sort_values("date").tail(3)
    last = raw.iloc[-1]

    master = pd.DatetimeIndex([last["date"]])
    extras = load_market_extras([test_code], master, verbose=False)
    pc = extras["pct_chg"].loc[master[0], test_code]

    check(
        "pct_chg 已转成小数（不是百分数）",
        abs(pc - float(last["pct_chg"]) / 100.0) < 1e-9,
        f"got={pc}",
    )
    check(
        "数值量级合理（|pc| < 0.35，排除放大 100 倍的误判）",
        abs(pc) < 0.35,
        f"got={pc}",
    )
    # 与真实 close 收益率交叉验证
    prev = raw.iloc[-2]
    real_ret = float(last["close"]) / float(prev["close"]) - 1.0
    check(
        "与未复权 close 收益率一致（无除权日时）",
        abs(pc - real_ret) < 0.005,
        f"pc={pc:.6f} real={real_ret:.6f}",
    )

    print()
    if FAILURES:
        print(f"❌ {len(FAILURES)} 项失败: {FAILURES}")
        return 1
    print("✅ 全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
