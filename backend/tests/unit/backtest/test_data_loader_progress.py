"""DataLoader progress reporting contract tests."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd

from app.services.backtest.execution.data_loader import DataLoader


def _frame() -> pd.DataFrame:
    idx = pd.bdate_range("2024-01-01", "2024-01-31")
    return pd.DataFrame(
        {
            "open": [1.0] * len(idx),
            "high": [1.0] * len(idx),
            "low": [1.0] * len(idx),
            "close": [1.0] * len(idx),
            "volume": [1000] * len(idx),
        },
        index=idx,
    )


def test_load_multiple_stocks_reports_progress_for_parallel_loads() -> None:
    loader = DataLoader(data_dir="/tmp", max_workers=2)
    progress = MagicMock()

    with patch.object(loader, "load_stock_data", return_value=_frame()):
        result = loader.load_multiple_stocks(
            ["000001.SZ", "000002.SZ"],
            datetime(2024, 1, 1),
            datetime(2024, 1, 31),
            progress_callback=progress,
        )

    assert set(result) == {"000001.SZ", "000002.SZ"}
    assert progress.call_count == 2
    assert progress.call_args.args[:2] == (2, 2)
