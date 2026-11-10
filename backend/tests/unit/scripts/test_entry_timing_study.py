"""entry_timing_study 的契约测试（§12.80）。

锁四件事，因为它们决定结论方向：
1. 三种节奏必须在**同一段日历**上观测（否则分批白多活几十天）；
2. 记账必须用份额（漏乘 LOT 会把净值算成千把块，实测踩过）；
3. 现金不允许为负，买不满的零头必须留在现金里；
4. 佣金必须走"万3 且单笔最低 5 元"。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.entry_timing_study import (
    CAPITAL,
    LOT,
    MIN_FEE,
    buy_batch,
    load_closes,
    simulate,
    summarize,
)


@pytest.fixture(scope="module")
def px() -> pd.DataFrame:
    return load_closes()


def test_buy_batch_uses_integer_lots_and_keeps_remainder():
    lots, amount, fee = buy_batch(50_000.0, 1.484, 45_000.0)
    assert lots == 303                       # 45000 / 148.4 = 303.2 → 只能 303 手
    assert amount == pytest.approx(303 * 1.484 * LOT, rel=1e-9)
    assert amount <= 45_000.0
    assert fee == pytest.approx(max(5.0, 3e-4 * amount), rel=1e-9)


def test_buy_batch_hits_the_5_yuan_floor_on_small_orders():
    _, amount, fee = buy_batch(1_000.0, 1.484, 1_000.0)   # 6 手 ≈ 890 元
    assert amount < 1_666.7          # 低于"5 元 = 万3"的临界成交额
    assert fee == MIN_FEE


def test_buy_batch_returns_zero_when_budget_cannot_fill_one_lot():
    lots, amount, fee = buy_batch(100.0, 10.45, 100.0)    # 一手要 1045 元
    assert (lots, amount, fee) == (0, 0.0, 0.0)


def test_simulate_accounts_shares_not_lots(px):
    r = simulate(px, 0, 1, 252)
    assert r, "起始索引应能跑出一个结果"
    invested = CAPITAL - r["cash_left"] - r["fees"]
    assert invested > 0.97 * CAPITAL, "整数手摩擦不应吃掉超过 3% 的建仓金额"
    assert r["invested_pct"] > 97.0
    assert r["cash_left"] >= -0.01, "现金不允许为负"
    # 净值量级必须是 9 万级别（踩过：记手数不记份额 → 净值 ~1 千）
    assert abs(r["ret_pct"]) < 60.0


def test_all_modes_share_the_same_observation_end(px):
    start = 40
    lump = simulate(px, start, 1, 252)
    drip = simulate(px, start, 6, 252)
    assert lump["start"] == drip["start"]
    assert lump["end"] == drip["end"], "分批不能靠多观测几十天来赢"


def test_drip_leaves_more_unfilled_remainder_than_lump(px):
    start = 40
    lump = simulate(px, start, 1, 252)
    drip = simulate(px, start, 6, 252)
    # 每批预算更小 → 整数手摩擦更大 → 剩余现金更多
    assert drip["cash_left"] >= lump["cash_left"] - 1e-6
    assert drip["fees"] >= lump["fees"] - 1e-6, "批数越多，最低 5 元佣金触发得越多"


def test_simulate_refuses_to_run_off_the_right_edge(px):
    assert simulate(px, len(px.index) - 10, 1, 252) == {}
    assert simulate(px, 0, 6, 252) == simulate(px, 0, 6, 252)  # 确定性，无随机


def test_summarize_reports_tail_not_only_mean():
    rows = [
        {"ret_pct": 10.0, "max_dd_pct": -5.0, "days_below_90k": 10},
        {"ret_pct": -8.0, "max_dd_pct": -12.0, "days_below_90k": 200},
        {"ret_pct": 20.0, "max_dd_pct": -3.0, "days_below_90k": 5},
    ]
    s = summarize(rows)
    assert s["n_entry_dates"] == 3
    assert s["ret_mean_pp"] == pytest.approx(7.33, abs=0.01)   # (10 − 8 + 20)/3
    assert s["ret_worst_pp"] == pytest.approx(-8.0)
    assert s["dd_worst_pp"] == pytest.approx(-12.0)
    assert s["share_of_entries_negative"] == pytest.approx(33.3, abs=0.1)
    assert s["max_days_below_90k"] == 200


def test_lump_beats_drip_on_mean_but_not_on_worst_dd(px):
    """§12.80 的结论方向：分批买的不是"少亏"，而是"少赚"。"""
    starts = [i for i in range(0, len(px.index) - 6 * 21 - 252, 21)]
    lump = summarize([r for i in starts if (r := simulate(px, i, 1, 252))])
    drip = summarize([r for i in starts if (r := simulate(px, i, 6, 252))])
    assert lump["ret_mean_pp"] > drip["ret_mean_pp"]
    assert lump["dd_worst_pp"] <= drip["dd_worst_pp"] + 1.5, (
        "一次性建仓的最差回撤不应比分批差太多，否则这条主仓的结论要改写"
    )
