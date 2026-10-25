"""实盘可成交性规则引擎单元测试 —— 2026-10-25

不依赖网络/大文件，纯函数级验证。运行：
    env -u PYTHONPATH -u PYTHONHOME \
      backend/.venv-py313/bin/python3 backend/tests/unit/trading/test_execution_rules.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, "/Users/rone/Projects/stock-platform/backend")

from app.services.trading.execution_rules import (  # noqa: E402
    DEFAULT_MIN_TURNOVER,
    ST_LIMIT_PCT,
    check_tradability,
    limit_pct,
)

FAILURES: list[str] = []


def check(name: str, cond: bool, extra: str = "") -> None:
    if cond:
        print(f"  ✅ {name}")
    else:
        print(f"  ❌ {name} {extra}")
        FAILURES.append(name)


def base(**kw: object) -> dict:
    """一个「正常可买可卖」的默认输入。"""
    args: dict = dict(
        name="测试股份",
        close=10.0,
        open_price=9.9,
        high=10.1,
        low=9.8,
        pct_chg=0.01,
        volume=1_000_000,
        amount=100_000_000.0,
        trade_status=1,
        is_st=0,
    )
    args.update(kw)
    return args


def main() -> int:
    print("=== 涨跌停幅度 ===")
    check("主板 600000 → 10%", limit_pct("600000.SH") == 0.10)
    check("主板 000001 → 10%", limit_pct("000001.SZ") == 0.10)
    check("创业板 300750 → 20%", limit_pct("300750.SZ") == 0.20)
    check("科创板 688001 → 20%", limit_pct("688001.SH") == 0.20)
    check("北交所 920001 → 30%", limit_pct("920001.BJ") == 0.30)
    check("ST 主板 → 5%", limit_pct("000001.SZ", is_st=True) == ST_LIMIT_PCT)

    print("\n=== 正常情况 ===")
    t = check_tradability("600000.SH", **base())
    check("正常票可买可卖", t.can_buy and t.can_sell)
    check("正常票有流动性上限", t.max_buy_amount is not None
          and abs(t.max_buy_amount - 100_000_000 * 0.005) < 1)

    print("\n=== 停牌 ===")
    t = check_tradability("600000.SH", **base(trade_status=0))
    check("trade_status=0 → 买卖都不可", not t.can_buy and not t.can_sell)
    t = check_tradability("600000.SH", **base(volume=0))
    check("volume=0 → 买卖都不可", not t.can_buy and not t.can_sell)

    print("\n=== 涨跌停 ===")
    t = check_tradability("600000.SH", **base(pct_chg=0.0999))
    check("主板 +9.99% → 买不进", not t.can_buy and t.can_sell)
    t = check_tradability("600000.SH", **base(pct_chg=-0.0999))
    check("主板 -9.99% → 卖不出", t.can_buy and not t.can_sell)
    t = check_tradability("300750.SZ", **base(pct_chg=0.199))
    check("创业板 +19.9% → 买不进", not t.can_buy and t.can_sell)
    t = check_tradability("300750.SZ", **base(pct_chg=0.11))
    check("创业板 +11% → 正常（20% 限制）", t.can_buy and t.can_sell)
    t = check_tradability("920001.BJ", **base(pct_chg=0.2999))
    check("北交所 +29.99% → 买不进", not t.can_buy and t.can_sell)
    t = check_tradability("920001.BJ", **base(pct_chg=0.29))
    check("北交所 +29% → 正常（30% 限制）", t.can_buy and t.can_sell)

    print("\n=== ST（5% 涨跌停，掩码不同）===")
    t = check_tradability("000001.SZ", **base(name="ST测试", is_st=1, pct_chg=0.049))
    check("ST +4.9% → 买不进", not t.can_buy and t.can_sell)
    t = check_tradability("000001.SZ", **base(name="ST测试", is_st=1, pct_chg=0.04))
    check("ST +4% → 正常", t.can_buy and t.can_sell)

    print("\n=== 一字板 ===")
    t = check_tradability(
        "600000.SH", **base(pct_chg=0.0999, open_price=11.0, high=11.0, low=11.0)
    )
    check("一字涨停 → 买不进", not t.can_buy and t.can_sell)
    t = check_tradability(
        "600000.SH", **base(pct_chg=-0.0999, open_price=9.0, high=9.0, low=9.0)
    )
    check("一字跌停 → 卖不出", t.can_buy and not t.can_sell)

    print("\n=== 退市整理期 ===")
    t = check_tradability("002731.SZ", **base(name="萃华退"))
    check("名称带「退」→ 买卖都不可", not t.can_buy and not t.can_sell)
    t = check_tradability("600000.SH", **base(name="退市整理"))
    check("「退市整理」不算退市股（不以「退」结尾）", t.can_buy and t.can_sell)

    print("\n=== 流动性 ===")
    t = check_tradability(
        "600000.SH", **base(amount=1_000_000.0)   # 100 万 < 500 万门槛
    )
    check("成交额低于门槛 → 买不进（但可卖）", not t.can_buy and t.can_sell)
    t = check_tradability("600000.SH", **base(amount=1_000_000_000.0))
    check("成交额 10 亿 → 上限 500 万", abs(t.max_buy_amount - 5_000_000) < 1)
    t = check_tradability("600000.SH", **base(amount=300_000.0))
    check("极低成交额 → 拦截", not t.can_buy)

    print("\n=== 数据缺失 ===")
    t = check_tradability("600000.SH", **base(close=None))
    check("无收盘价 → 拦截", not t.can_buy and not t.can_sell)
    t = check_tradability("600000.SH", **base(pct_chg=None))
    check("无 pct_chg → 保守放行（不误杀）", t.can_buy and t.can_sell)

    print("\n=== 除权跳空不误判 ===")
    # 10 送 10 → 未复权价腰斩 -50%，但交易所口径涨跌幅是 0
    t = check_tradability(
        "600000.SH", **base(close=5.0, open_price=5.0, high=5.05, low=4.98,
                            pct_chg=0.0)
    )
    check("除权日(pct_chg=0) → 正常可交易", t.can_buy and t.can_sell)

    print()
    if FAILURES:
        print(f"❌ {len(FAILURES)} 项失败: {FAILURES}")
        return 1
    print("✅ 全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
