"""
回归测试：Parquet 加载优化（pyarrow filters + column selection）

验证 pyarrow filters 过滤 vs 全量加载再过滤，输出 DataFrame 完全一致。
"""

import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_parquet(tmp_path: Path) -> Path:
    """创建测试用 parquet 文件，模拟 Qlib 预计算格式。"""
    np.random.seed(42)
    n_rows = 500
    dates = pd.bdate_range("2020-01-01", periods=n_rows)
    stock_code = "000001_SZ"

    data = {
        "$open": np.random.uniform(10, 20, n_rows).astype(np.float32),
        "$high": np.random.uniform(10, 20, n_rows).astype(np.float32),
        "$low": np.random.uniform(10, 20, n_rows).astype(np.float32),
        "$close": np.random.uniform(10, 20, n_rows).astype(np.float32),
        "$volume": np.random.randint(1000, 100000, n_rows),
        "RSI14": np.random.uniform(20, 80, n_rows),
        "MA20": np.random.uniform(10, 20, n_rows),
        "MACD": np.random.uniform(-1, 1, n_rows),
    }
    df = pd.DataFrame(data)
    df["stock_code"] = stock_code
    df["date"] = dates
    df = df.set_index(["stock_code", "date"])

    out = tmp_path / f"{stock_code}.parquet"
    df.to_parquet(out, engine="pyarrow", compression="snappy")
    return out


class TestParquetFilterEquivalence:
    """pyarrow filters 过滤 vs 全量加载再过滤，输出完全一致。"""

    def test_date_filter_matches_full_load(self, sample_parquet: Path):
        """日期范围过滤：pyarrow filters 结果 == 全量加载后手动过滤。"""
        start = pd.Timestamp("2020-06-01")
        end = pd.Timestamp("2021-06-01")

        # 全量加载 + 手动过滤
        df_full = pd.read_parquet(sample_parquet)
        dates = df_full.index.get_level_values("date")
        df_expected = df_full[(dates >= start) & (dates <= end)]

        # pyarrow filters
        df_filtered = pd.read_parquet(
            sample_parquet,
            filters=[("date", ">=", start), ("date", "<=", end)],
        )

        pd.testing.assert_frame_equal(df_filtered, df_expected)

    def test_column_selection_matches_full_load(self, sample_parquet: Path):
        """列选择：只读取指定列，结果 == 全量加载后选列。"""
        cols = ["$open", "$close", "RSI14"]

        df_full = pd.read_parquet(sample_parquet)
        df_expected = df_full[cols]

        df_filtered = pd.read_parquet(sample_parquet, columns=cols)

        pd.testing.assert_frame_equal(df_filtered, df_expected)

    def test_column_and_date_filter_combined(self, sample_parquet: Path):
        """列选择 + 日期过滤组合。"""
        cols = ["$open", "$close", "$volume"]
        start = pd.Timestamp("2020-03-01")
        end = pd.Timestamp("2020-12-31")

        df_full = pd.read_parquet(sample_parquet)
        dates = df_full.index.get_level_values("date")
        df_expected = df_full.loc[(dates >= start) & (dates <= end), cols]

        df_filtered = pd.read_parquet(
            sample_parquet,
            columns=cols,
            filters=[("date", ">=", start), ("date", "<=", end)],
        )

        pd.testing.assert_frame_equal(df_filtered, df_expected)

    def test_multiindex_preserved(self, sample_parquet: Path):
        """过滤后 MultiIndex (stock_code, date) 保持不变。"""
        df = pd.read_parquet(
            sample_parquet,
            columns=["$close"],
            filters=[("date", ">=", pd.Timestamp("2020-06-01"))],
        )
        assert isinstance(df.index, pd.MultiIndex)
        assert df.index.names == ["stock_code", "date"]

    def test_empty_date_range_returns_empty(self, sample_parquet: Path):
        """日期范围无交集时返回空 DataFrame。"""
        df = pd.read_parquet(
            sample_parquet,
            filters=[
                ("date", ">=", pd.Timestamp("2030-01-01")),
                ("date", "<=", pd.Timestamp("2030-12-31")),
            ],
        )
        assert len(df) == 0

    def test_nonexistent_column_raises(self, sample_parquet: Path):
        """请求不存在的列应抛出异常。"""
        with pytest.raises(Exception):
            pd.read_parquet(sample_parquet, columns=["nonexistent_col"])

    def test_all_columns_no_filter_unchanged(self, sample_parquet: Path):
        """不传 columns/filters 时，结果与原始完全一致。"""
        df_full = pd.read_parquet(sample_parquet)
        df_again = pd.read_parquet(sample_parquet)
        pd.testing.assert_frame_equal(df_full, df_again)


class TestQlibFormatConverterLoadOptimized:
    """测试 QlibFormatConverter.load_qlib_data 优化后的行为。"""

    def test_load_with_date_range_filters(self, sample_parquet: Path):
        """load_qlib_data 使用日期范围时应只返回范围内数据。"""
        from app.services.data.qlib_format_converter import QlibFormatConverter

        converter = QlibFormatConverter()
        start = datetime(2020, 6, 1)
        end = datetime(2021, 1, 1)

        df = converter.load_qlib_data(
            sample_parquet,
            stock_code="000001_SZ",
            start_date=start,
            end_date=end,
        )

        if not df.empty:
            dates = df.index.get_level_values("date") if isinstance(df.index, pd.MultiIndex) else df.index
            assert dates.min() >= pd.Timestamp(start)
            assert dates.max() <= pd.Timestamp(end)

    def test_load_without_date_range_returns_all(self, sample_parquet: Path):
        """不传日期范围时返回全部数据。"""
        from app.services.data.qlib_format_converter import QlibFormatConverter

        converter = QlibFormatConverter()
        df = converter.load_qlib_data(sample_parquet, stock_code="000001_SZ")

        df_full = pd.read_parquet(sample_parquet)
        assert len(df) == len(df_full)
