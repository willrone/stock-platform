"""
LowVolatilityStrategy unit tests.

Created: 2026-07-25
Purpose: Verify low-volatility factor signal generation — BUY when
         risk-adjusted return (mean/std) is positive.
"""

import numpy as np
import pandas as pd
import pytest

from app.services.backtest.models import SignalType
from app.services.backtest.strategies.strategies import LowVolatilityStrategy


def _make_data(
    close: list[float], n_stocks: int = 3
) -> dict[str, pd.DataFrame]:
    dates = pd.bdate_range("2024-01-01", periods=len(close))
    result = {}
    for i in range(1, n_stocks + 1):
        code = f"00000{i}.SZ"
        df = pd.DataFrame(
            {
                "open": [c * 0.99 for c in close],
                "high": [c * 1.02 for c in close],
                "low": [c * 0.98 for c in close],
                "close": close,
                "volume": [1_000_000] * len(close),
            },
            index=dates,
        )
        df.attrs["stock_code"] = code
        result[code] = df
    return result


def test_low_volatility_indicators_shape() -> None:
    """calculate_indicators returns volatility, risk_adjusted_return, price."""
    n = 100
    close = [50.0] * n
    data_dict = _make_data(close, n_stocks=1)
    df = data_dict["000001.SZ"]

    strategy = LowVolatilityStrategy(
        {"volatility_period": 21, "volatility_window": 63}
    )
    ind = strategy.calculate_indicators(df)

    assert "volatility" in ind
    assert "risk_adjusted_return" in ind
    assert "price" in ind
    assert len(ind["volatility"]) == n


def test_low_volatility_generate_signals_returns_list() -> None:
    """generate_signals always returns a list."""
    n = 100
    close = [50.0] * n
    data_dict = _make_data(close, n_stocks=1)
    df = data_dict["000001.SZ"]

    strategy = LowVolatilityStrategy(
        {"volatility_period": 21, "volatility_window": 63}
    )
    strategy.get_cached_indicators(df)
    result = strategy.generate_signals(df, df.index[-1])
    assert isinstance(result, list)


def test_low_volatility_short_data_returns_empty() -> None:
    """With fewer bars than volatility_window, generate_signals returns []."""
    n = 30
    close = [50.0] * n
    data_dict = _make_data(close, n_stocks=1)
    df = data_dict["000001.SZ"]

    strategy = LowVolatilityStrategy(
        {"volatility_period": 21, "volatility_window": 63}
    )
    strategy.get_cached_indicators(df)
    result = strategy.generate_signals(df, df.index[-1])
    assert result == []


def test_low_volatility_buy_when_rar_positive() -> None:
    """With a steady uptrend, risk_adjusted_return > 0 so BUY is emitted."""
    # Steady uptrend: low volatility, positive return
    n = 100
    close = list(range(50, 50 + n))
    data_dict = _make_data(close, n_stocks=1)
    df = data_dict["000001.SZ"]

    strategy = LowVolatilityStrategy(
        {"volatility_period": 10, "volatility_window": 20}
    )
    strategy.get_cached_indicators(df)

    buy_signals = []
    for dt in df.index[20:]:
        sigs = strategy.generate_signals(df, dt)
        for s in sigs:
            if s.signal_type == SignalType.BUY:
                buy_signals.append(s)

    assert len(buy_signals) >= 1
