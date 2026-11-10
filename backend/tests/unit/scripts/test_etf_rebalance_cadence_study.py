"""etf_rebalance_cadence_study 的契约测试（合成价，不读真实库、不联网）。

锁四件事：
1. band 是「检查但不交易」——阈值没过就不许成交、不许花钱；
2. hold 永不再平衡 → 笔数/成交额必须恒 0；
3. 成本对节奏单调：daily > weekly > monthly > annual（这条若反了就是撮合写错）；
4. 整数手与现金纪律：任何时点 lot 是整数、现金不得为负、建仓投入率 ≤ 100%。
"""
from __future__ import annotations

import pandas as pd
import pytest

from scripts.etf_rebalance_cadence_study import (
    CAPITAL,
    CODES,
    LOT,
    _cadence_dates,
    simulate,
)


def _panel(dates: list[str], a: list[float], b: list[float]) -> pd.DataFrame:
    return pd.DataFrame({CODES[0]: a, CODES[1]: b}, index=pd.to_datetime(dates))


def _days(start: str, n: int) -> list[str]:
    return [d.strftime("%Y-%m-%d") for d in pd.bdate_range(start=start, periods=n)]


def test_band_gates_trading_not_just_checking():
    """漂移上限 ~5.06pp：band=6pp 必须一毛不拔；band=5pp 必须正好擦一次（阈值确实在切）。"""
    n = 200
    dates = _days("2023-01-02", n)
    px = _panel(dates, [1.0 * 1.0005 ** i for i in range(n)], [7.0 * 0.9995 ** i for i in range(n)])
    w = simulate(px, "weekly")
    loose = simulate(px, "weekly", band=0.06)
    tight = simulate(px, "weekly", band=0.05)
    assert w["checks"] == loose["checks"] == tight["checks"] > 0, "band 不减少盯市，只减少成交"
    assert loose["n_trades"] == 0 and loose["acted"] == 0, "阈值没过就不许成交"
    assert loose["fees_total"] < w["fees_total"], "不动手就不许花钱"
    assert tight["acted"] == 1 and tight["n_trades"] == 2, f"5pp 阈值应当正好触发一次，实测 {tight['acted']}"
    assert w["n_trades"] > 20, f"无阈值时应该一直在擦回 50/50，实测 {w['n_trades']} 笔"


def test_cash_cushion_makes_the_two_legs_drift_asymmetrically():
    """留了现金 → 两腿偏离不相等（本例腿1 最高 5.06pp，腿0 最高只有 4.87pp）。
    所以 band=4.9pp 时，成交**只能**是被腿1 触发的：若代码只盯第一条腿，这里会 0 次成交，必红。"""
    n = 200
    dates = _days("2023-01-02", n)
    px = _panel(dates, [1.0 * 1.0005 ** i for i in range(n)], [7.0 * 0.9995 ** i for i in range(n)])
    only_leg1_can_trigger = simulate(px, "weekly", band=0.049)
    assert only_leg1_can_trigger["acted"] == 1, "判定必须看两腿里偏离更大的那条"


def test_hold_never_trades_after_open():
    n = 400
    dates = _days("2023-01-02", n)
    px = _panel(dates, [1.0 * (1.01 if i % 2 == 0 else 0.99) for i in range(n)],
                [7.0 * (0.99 if i % 2 == 0 else 1.01) for i in range(n)])
    h = simulate(px, "hold")
    assert h["checks"] == 0 and h["n_trades"] == 0 and h["acted"] == 0
    assert h["traded_value"] == 0.0
    assert h["initial_invested_pct"] <= 1.0


def test_cost_is_monotone_in_cadence():
    """抖动行情下：换手与佣金必须随节奏变密而单调上升。反了 = 撮合或计数写错。"""
    n = 600
    dates = _days("2023-01-02", n)
    a, b = [], []
    for i in range(n):
        a.append(1.0 * (1.02 if i % 3 == 0 else 0.985))
        b.append(7.0 * (0.98 if i % 3 == 0 else 1.015))
    px = _panel(dates, a, b)
    fees = {c: simulate(px, c)["fees_total"] for c in ("annual", "monthly", "weekly", "daily")}
    assert fees["daily"] > fees["weekly"] > fees["monthly"] > fees["annual"], fees
    tr = {c: simulate(px, c)["n_trades"] for c in ("annual", "monthly", "weekly", "daily")}
    assert tr["daily"] > tr["weekly"] > tr["monthly"] > tr["annual"], tr
    # 建仓两笔各 45,000 元 → 万3 = 13.5 元/笔（未被 5 元下限抬起）
    assert fees["annual"] >= 27.0, fees["annual"]


def test_flat_panel_moves_only_by_fee_drag():
    """全程同价 → 不该有任何「赚到钱」，净值只会被佣金拖低；整数手会让首次检查擦 1 手。"""
    n = 300
    dates = _days("2023-01-02", n)
    px = _panel(dates, [1.0] * n, [7.0] * n)
    r = simulate(px, "annual")
    assert r["n_trades"] <= 2, "同价面板最多只因整数手取整擦 1~2 手，不该反复换手"
    assert -0.06 < r["total_pct"] < 0.0, f"同价面板的收益必须只来自佣金拖累，实测 {r['total_pct']}"
    assert r["max_dd_pct"] < 0.0 and r["max_dd_pct"] > -0.06


def test_integer_lots_and_cash_never_negative():
    n = 300
    dates = _days("2023-01-02", n)
    px = _panel(dates, [1.0 * 1.003 ** i for i in range(n)], [7.0] * n)
    r = simulate(px, "weekly")
    assert 0.95 < r["initial_invested_pct"] <= 1.0
    # 独立复算一遍：整数手 + 万3最低5元 → 现金任何时点都不该透支
    lots = {c: int((CAPITAL * 0.5) // (float(px.iloc[0][c]) * LOT)) for c in CODES}
    cash = CAPITAL - sum(lots[c] * LOT * float(px.iloc[0][c]) for c in CODES)
    cash -= sum(max(lots[c] * LOT * float(px.iloc[0][c]) * 0.0003, 5.0) for c in CODES)
    assert cash >= 0
    for d in px.index:
        prices = {c: float(px.loc[d][c]) for c in CODES}
        nav = cash + sum(lots[c] * LOT * prices[c] for c in CODES)
        assert nav > 0 and all(float(l).is_integer() for l in lots.values())
        if d not in _cadence_dates(px.index, "weekly"):
            continue
        for c in CODES:
            want = int((nav * 0.5) // (prices[c] * LOT))
            delta = want - lots[c]
            if delta:
                cash -= delta * LOT * prices[c]
                cash -= max(abs(delta) * LOT * prices[c] * 0.0003, 5.0)
                lots[c] = want
        assert cash >= -1e-9, "整数手向下取整只会留现金，不该透支"
    assert abs(sum(lots[c] * LOT * float(px.iloc[-1][c]) for c in CODES) + cash - r["final_nav"]) < 1.0, \
        "独立复算的终值必须与脚本一致（含每笔佣金）"
