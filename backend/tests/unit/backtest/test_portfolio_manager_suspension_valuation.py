"""持仓停牌/缺数据日估值回退语义测试（引擎串行路径停牌估值缺陷 2026-10-13 修复）。

锁定语义：
- 持仓股当日无有效价（停牌/数据缺行/NaN）→ 按 last_prices 最近已知有效价计价，而非 0
- 当日有效价出现时正常计价，且快照后缓存刷新
- 含成本/无成本两条路径一致
- portfolio_history positions payload 停牌日显示最近已知价
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

import pytest

from app.services.backtest.core.portfolio_manager import PortfolioManager
from app.services.backtest.core.portfolio_manager_array import PortfolioManagerArray
from app.services.backtest.models import BacktestConfig, SignalType, TradingSignal

ManagerKind = Literal["classic", "array"]


def _build_manager(kind: ManagerKind, codes: list[str]):
    config = BacktestConfig(
        initial_cash=100_000.0,
        commission_rate=0.0,
        slippage_rate=0.0,
        max_position_size=0.5,
        cash_reserve_ratio=0.0,
        board_lot_size=100,
    )
    if kind == "classic":
        return PortfolioManager(config)
    return PortfolioManagerArray(config, codes)


def _buy(manager, code: str, price: float, ts: datetime) -> None:
    signal = TradingSignal(
        timestamp=ts,
        stock_code=code,
        signal_type=SignalType.BUY,
        strength=0.9,
        price=price,
        reason="suspension valuation test",
    )
    trade, reason = manager.execute_signal(signal, {code: price})
    assert trade is not None, f"buy failed: {reason}"


@pytest.mark.parametrize("kind", ["classic", "array"])
def test_suspended_holding_valued_at_last_known_price(kind: ManagerKind) -> None:
    code = "000001.SZ"
    manager = _build_manager(kind, [code])
    day1 = datetime(2024, 1, 2)
    day2 = datetime(2024, 1, 3)

    _buy(manager, code, 10.0, day1)
    manager.record_portfolio_snapshot(day1, {code: 10.0})

    # day2 停牌：current_prices 完全缺失该股
    value = manager.get_portfolio_value({})
    held_qty = manager.positions[code].quantity
    assert held_qty > 0
    assert value == pytest.approx(manager.cash + held_qty * 10.0, rel=1e-12), (
        f"{kind}: 停牌持仓被按 0 计价（修复前行为），value={value}"
    )


@pytest.mark.parametrize("kind", ["classic", "array"])
def test_nan_and_nonpositive_prices_fall_back(kind: ManagerKind) -> None:
    code = "000002.SZ"
    manager = _build_manager(kind, [code])
    day1 = datetime(2024, 1, 2)
    day2 = datetime(2024, 1, 3)

    _buy(manager, code, 20.0, day1)
    manager.record_portfolio_snapshot(day1, {code: 20.0})
    held_qty = manager.positions[code].quantity

    # NaN / 0 / 负价都视为无效 → 回退
    for bad in (float("nan"), 0.0, -5.0):
        value = manager.get_portfolio_value({code: bad})
        assert value == pytest.approx(manager.cash + held_qty * 20.0, rel=1e-12), (
            f"{kind}: 无效价 {bad} 未回退到最近已知价, value={value}"
        )


@pytest.mark.parametrize("kind", ["classic", "array"])
def test_resume_day_uses_fresh_price_and_cache_updates(kind: ManagerKind) -> None:
    code = "000003.SZ"
    manager = _build_manager(kind, [code])
    day1 = datetime(2024, 1, 2)
    day2 = datetime(2024, 1, 3)
    day3 = datetime(2024, 1, 4)

    _buy(manager, code, 10.0, day1)
    manager.record_portfolio_snapshot(day1, {code: 10.0})
    held_qty = manager.positions[code].quantity

    # day2 复牌：现价 12
    manager.record_portfolio_snapshot(day2, {code: 12.0})
    value_d2 = manager.get_portfolio_value({code: 12.0})
    assert value_d2 == pytest.approx(manager.cash + held_qty * 12.0, rel=1e-12)

    # day3 再停牌 → 回退价应为 12（复牌日已刷新缓存），不是 10
    value_d3 = manager.get_portfolio_value({})
    assert value_d3 == pytest.approx(manager.cash + held_qty * 12.0, rel=1e-12), (
        f"{kind}: 复牌后缓存未刷新, value={value_d3}"
    )


@pytest.mark.parametrize("kind", ["classic", "array"])
def test_without_cost_path_matches(kind: ManagerKind) -> None:
    code = "000004.SZ"
    manager = _build_manager(kind, [code])
    day1 = datetime(2024, 1, 2)

    _buy(manager, code, 8.0, day1)
    manager.record_portfolio_snapshot(day1, {code: 8.0})

    value_nc = manager.get_portfolio_value_without_cost({})
    # 无佣金滑点 → 无成本路径与含成本一致
    value_cost = manager.get_portfolio_value({})
    assert value_nc == pytest.approx(value_cost, rel=1e-12), (
        f"{kind}: 无成本路径停牌回退不一致"
    )


@pytest.mark.parametrize("kind", ["classic", "array"])
def test_snapshot_payload_records_last_known_price(kind: ManagerKind) -> None:
    code = "000005.SZ"
    manager = _build_manager(kind, [code])
    day1 = datetime(2024, 1, 2)
    day2 = datetime(2024, 1, 3)

    _buy(manager, code, 15.0, day1)
    manager.record_portfolio_snapshot(day1, {code: 15.0})
    manager.record_portfolio_snapshot(day2, {})  # 停牌日

    snap = manager.portfolio_history[-1]
    payload = snap["positions"][code]
    held_qty = manager.positions[code].quantity
    assert payload["current_price"] == pytest.approx(15.0, rel=1e-12), (
        f"{kind}: 停牌日 positions payload current_price 应为最近已知价"
    )
    assert payload["market_value"] == pytest.approx(held_qty * 15.0, rel=1e-12)
    # equity 曲线停牌日不假崩
    assert snap["portfolio_value"] == pytest.approx(
        manager.cash + held_qty * 15.0, rel=1e-12
    )


@pytest.mark.parametrize("kind", ["classic", "array"])
def test_unheld_code_never_invented(kind: ManagerKind) -> None:
    """未持仓的股票即使出现在 last_prices 也不应计入总值。"""
    code = "000006.SZ"
    manager = _build_manager(kind, [code])
    manager.last_prices["GHOST.SZ"] = 999.0
    value = manager.get_portfolio_value({})
    assert value == pytest.approx(manager.config.initial_cash, rel=1e-12)
