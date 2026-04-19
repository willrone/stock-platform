from pathlib import Path

import pandas as pd

from app.services.data.official_qlib_data_builder import OfficialQlibDataBuilder


class _FakeLoader:
    def __init__(self):
        self.calls = []

    def _load_base_data(self, stock_code: str):
        self.calls.append(stock_code)
        index = pd.to_datetime(["2024-01-02", "2024-01-03"])
        return pd.DataFrame(
            {
                "open": [10.0, 10.5],
                "high": [10.2, 10.8],
                "low": [9.9, 10.1],
                "close": [10.1, 10.6],
                "volume": [1000, 1200],
            },
            index=index,
        )


class _FakeConverter:
    def __init__(self):
        self.calls = []

    def convert_parquet_to_bin(self, parquet_data, stock_code, qlib_data_path):
        self.calls.append((stock_code, qlib_data_path, list(parquet_data.columns)))
        return Path(qlib_data_path) / "features" / stock_code.replace('.', '_').lower() / "close.day.bin"


def test_prepare_stocks_builds_clean_official_qlib_bins(tmp_path) -> None:
    loader = _FakeLoader()
    converter = _FakeConverter()
    builder = OfficialQlibDataBuilder(
        official_qlib_data_path=tmp_path / "qlib_official_data",
        data_loader=loader,
        bin_converter=converter,
    )

    result = builder.prepare_stocks(["600036.SH", "601288.SH"])

    assert result["success"] == ["600036.SH", "601288.SH"]
    assert result["failed"] == []
    assert loader.calls == ["600036.SH", "601288.SH"]
    assert [call[0] for call in converter.calls] == ["600036.SH", "601288.SH"]
    assert all(call[2] == ["open", "high", "low", "close", "volume"] for call in converter.calls)
    assert builder.official_qlib_data_path == tmp_path / "qlib_official_data"
