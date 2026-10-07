"""
BollingerBandStrategy unit tests.

Created: 2026-07-25
Purpose: Verify Bollinger band signal generation logic with
         synthetic OHLCV data — BUY on lower-band breakout,
         SELL on upper-band breakout.
"""

import numpy as np
import pandas as pd
import pytest

from app.services.backtest.models import SignalType
from app.services.backtest.strategies.strategies import BollingerBandStrategy


def _make_data(
    close: list[float],
    period: int = 20,
    n_stocks: int = 3,
) -> dict[str, pd.DataFrame]:
    """Build a multi-stock dict of OHLCV DataFrames with the given close series."""
    dates = pd.bdate_range("2024-01-01", periods=len(close))
    result = {}
    for i in range(1, n_stocks + 1):
        code = f"00000{i}.SZ"
        df = pd.DataFrame(
            {
                "open": close,
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


def test_bollinger_precompute_buy_lower_band_breakout() -> None:
    """Precompute signals: BUY when %B crosses upward through 0 (price below
    lower band then rises into the band)."""
    period = 20
    # Flat for 20 bars then a sharp dip and recovery
    close = [50.0] * period + [45.0, 44.0, 46.0, 48.0, 50.0]
    data_dict = _make_data(close, period=period, n_stocks=1)
    stock_code = "000001.SZ"
    df = data_dict[stock_code]

    strategy = BollingerBandStrategy(
        {"period": period, "std_dev": 2, "entry_threshold": 0.02}
    )

    signals = strategy.precompute_all_signals(df)

    assert signals is not None
    # The day after the low (44 -> 46) should trigger a buy
    # when percent_b crosses from <=0 to >0
    buy_idx = signals[signals == SignalType.BUY]
    assert len(buy_idx) >= 1, "Expected at least one BUY signal after lower-band breakout"

    for dt in buy_idx.index:
        i = df.index.get_loc(dt)
        assert i > period, "Signals before warmup period are invalid"


def test_bollinger_precompute_sell_upper_band_breakout() -> None:
    """Precompute signals: SELL when %B crosses downward through 1 (price above
    upper band then falls into the band)."""
    period = 20
    # Flat then spike up then drop back
    close = [50.0] * period + [55.0, 56.0, 54.0, 52.0, 50.0]
    data_dict = _make_data(close, period=period, n_stocks=1)
    df = data_dict["000001.SZ"]

    strategy = BollingerBandStrategy(
        {"period": period, "std_dev": 2, "entry_threshold": 0.02}
    )

    signals = strategy.precompute_all_signals(df)

    assert signals is not None
    sell_idx = signals[signals == SignalType.SELL]
    assert len(sell_idx) >= 1, "Expected at least one SELL signal after upper-band breakout"


def test_bollinger_indicators_shape() -> None:
    """calculate_indicators returns expected keys with correct length."""
    period = 20
    close = [50.0] * period + [45.0, 44.0, 46.0, 48.0, 50.0]
    data_dict = _make_data(close, period=period, n_stocks=1)
    df = data_dict["000001.SZ"]

    strategy = BollingerBandStrategy(
        {"period": period, "std_dev": 2, "entry_threshold": 0.02}
    )
    ind = strategy.calculate_indicators(df)

    expected_keys = {"sma", "upper_band", "lower_band", "bandwidth", "percent_b", "price"}
    assert expected_keys.issubset(ind.keys())
    for v in ind.values():
        assert len(v) == len(df)


def test_bollinger_generate_signals_returns_list() -> None:
    """generate_signals returns a list (possibly empty) for any valid date."""
    period = 20
    close = [50.0] * (period + 5)
    data_dict = _make_data(close, period=period, n_stocks=1)
    df = data_dict["000001.SZ"]

    strategy = BollingerBandStrategy(
        {"period": period, "std_dev": 2, "entry_threshold": 0.02}
    )

    # Force caching of indicators
    _ = strategy.get_cached_indicators(df)

    signals = strategy.generate_signals(df, df.index[-1])
    assert isinstance(signals, list)
    for s in signals:
        assert s.signal_type in (SignalType.BUY, SignalType.SELL)
