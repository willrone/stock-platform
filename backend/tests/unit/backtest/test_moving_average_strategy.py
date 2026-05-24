"""MovingAverageStrategy contract tests."""

import pandas as pd

from app.services.backtest.models import SignalType
from app.services.backtest.strategies.technical.basic_strategies import (
    MovingAverageStrategy,
)


def test_moving_average_precompute_uses_close_not_qlib_ma_feature_columns() -> None:
    """Qlib MA* factor columns must not override plain close rolling means."""

    index = pd.bdate_range("2024-01-01", periods=12)
    close = pd.Series([10, 9, 8, 7, 8, 9, 10, 11, 12, 13, 14, 15], index=index)
    data = pd.DataFrame(
        {
            "close": close,
            # Deliberately misleading Qlib-style factor columns.  If the strategy
            # reuses these columns, no close-price golden cross will be produced.
            "MA3": [1.0] * len(index),
            "MA5": [1.0] * len(index),
        },
        index=index,
    )

    strategy = MovingAverageStrategy(
        {"short_window": 3, "long_window": 5, "signal_threshold": 0.0}
    )

    signals = strategy.precompute_all_signals(data)

    assert signals is not None
    assert signals.loc[pd.Timestamp("2024-01-09")] == SignalType.BUY
    assert signals.notna().sum() == 1


def test_moving_average_indicators_ignore_qlib_ma_feature_columns() -> None:
    index = pd.bdate_range("2024-01-01", periods=6)
    data = pd.DataFrame(
        {
            "close": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "MA2": [999.0] * 6,
            "MA3": [999.0] * 6,
        },
        index=index,
    )
    strategy = MovingAverageStrategy(
        {"short_window": 2, "long_window": 3, "signal_threshold": 0.0}
    )

    indicators = strategy.calculate_indicators(data)

    assert indicators["sma_short"].iloc[-1] == 5.5
    assert indicators["sma_long"].iloc[-1] == 5.0
