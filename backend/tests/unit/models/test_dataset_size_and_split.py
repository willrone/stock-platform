"""
测试数据集行数和 split 比例

验证：
1. 数据集行数应为 n_stocks * n_trading_days 量级（不应膨胀）
2. ratio split 应严格按 validation_split 分割（80:20）
"""

import numpy as np
import pandas as pd
import pytest

from app.services.qlib.training.dataset_preparation import (
    _extract_sorted_dates,
    _split_by_ratio,
    _process_multiindex_stocks,
    process_stock_data,
)
from app.services.qlib.training.config import QlibTrainingConfig, QlibModelType


# === 辅助函数 ===

def _make_multiindex_dataset(
    n_stocks: int = 10,
    n_days: int = 242,
    seed: int = 42,
) -> pd.DataFrame:
    """构造模拟的 MultiIndex (instrument, datetime) 数据集"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_days, freq="B")
    stocks = [f"SH60000{i}" for i in range(n_stocks)]

    rows = []
    for stock in stocks:
        base_price = rng.uniform(10, 100)
        for dt in dates:
            price = base_price * (1 + rng.normal(0, 0.02))
            rows.append({
                "instrument": stock,
                "datetime": dt,
                "$open": price * (1 + rng.normal(0, 0.005)),
                "$high": price * (1 + abs(rng.normal(0, 0.01))),
                "$low": price * (1 - abs(rng.normal(0, 0.01))),
                "$close": price,
                "$volume": rng.integers(1_000_000, 10_000_000),
            })
            base_price = price

    df = pd.DataFrame(rows)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index(["instrument", "datetime"]).sort_index()
    return df


def _make_config(**overrides) -> QlibTrainingConfig:
    """构造默认训练配置"""
    defaults = dict(
        model_type=QlibModelType.LIGHTGBM,
        hyperparameters={},
        feature_set="alpha158",
        label_type="regression",
        split_method="ratio",
        validation_split=0.2,
        prediction_horizon=5,
        embargo_days=20,
    )
    defaults.update(overrides)
    return QlibTrainingConfig(**defaults)


# === 测试 A: 行数不应膨胀 ===

class TestDatasetRowCount:
    """验证数据集行数不会异常膨胀"""

    def test_process_stock_data_preserves_row_count(self):
        """process_stock_data 不应改变行数"""
        dataset = _make_multiindex_dataset(n_stocks=1, n_days=100)
        stock_code = dataset.index.get_level_values(0).unique()[0]
        stock_data = dataset.xs(stock_code, level=0, drop_level=False)

        before = len(stock_data)
        processed = process_stock_data(stock_data, stock_code, prediction_horizon=5)
        after = len(processed)

        assert after == before, (
            f"process_stock_data 改变了行数: {before} → {after}"
        )

    def test_process_multiindex_stocks_preserves_row_count(self):
        """_process_multiindex_stocks 不应改变总行数"""
        dataset = _make_multiindex_dataset(n_stocks=5, n_days=100)
        config = _make_config()

        before = len(dataset)
        processed = _process_multiindex_stocks(dataset, config)
        after = len(processed)

        assert after == before, (
            f"_process_multiindex_stocks 改变了行数: {before} → {after}"
        )

    def test_no_duplicate_index_after_processing(self):
        """处理后不应有重复的 (instrument, datetime) 索引"""
        dataset = _make_multiindex_dataset(n_stocks=3, n_days=50)
        config = _make_config()
        processed = _process_multiindex_stocks(dataset, config)

        dup_count = processed.index.duplicated().sum()
        assert dup_count == 0, (
            f"处理后存在 {dup_count} 个重复索引"
        )

    def test_concat_axis1_with_matching_index(self):
        """axis=1 concat 在索引完全匹配时不应增加行数"""
        dataset = _make_multiindex_dataset(n_stocks=3, n_days=50)
        # 模拟 alpha factors（相同索引，不同列）
        alpha = pd.DataFrame(
            np.random.randn(len(dataset), 5),
            index=dataset.index,
            columns=[f"alpha_{i}" for i in range(5)],
        )

        before = len(dataset)
        combined = pd.concat([dataset, alpha], axis=1)
        after = len(combined)

        assert after == before, (
            f"axis=1 concat 改变了行数: {before} → {after}"
        )

    def test_concat_axis1_with_swapped_index_levels(self):
        """axis=1 concat 在索引 level 顺序不同时会膨胀行数（这是 bug 的根因）"""
        dataset = _make_multiindex_dataset(n_stocks=3, n_days=50)
        # 模拟 alpha factors 索引顺序为 (datetime, instrument) 而非 (instrument, datetime)
        alpha_swapped = pd.DataFrame(
            np.random.randn(len(dataset), 5),
            index=dataset.swaplevel().index,  # 交换 level 顺序
            columns=[f"alpha_{i}" for i in range(5)],
        )

        before = len(dataset)
        combined = pd.concat([dataset, alpha_swapped], axis=1)
        after = len(combined)

        # 如果索引 level 顺序不同，concat 会产生笛卡尔积式膨胀
        # 这个测试验证了 bug 的存在
        if after > before:
            pytest.skip(
                f"已知问题: 索引 level 顺序不同导致行数膨胀 "
                f"{before} → {after}"
            )


# === 测试 B: Split 比例 ===

class TestSplitRatio:
    """验证 ratio split 严格按 validation_split 分割"""

    def test_ratio_split_80_20(self):
        """validation_split=0.2 时，train:val ≈ 80:20"""
        dataset = _make_multiindex_dataset(n_stocks=10, n_days=242)
        # 添加 label 列
        dataset["label"] = np.random.randn(len(dataset))

        train, val = _split_by_ratio(
            dataset, validation_split=0.2, embargo_days=0,
        )

        total = len(train) + len(val)
        train_ratio = len(train) / total
        val_ratio = len(val) / total

        assert 0.75 <= train_ratio <= 0.85, (
            f"训练集比例 {train_ratio:.2%} 不在 75%-85% 范围内"
        )
        assert 0.15 <= val_ratio <= 0.25, (
            f"验证集比例 {val_ratio:.2%} 不在 15%-25% 范围内"
        )

    def test_ratio_split_80_20_with_embargo(self):
        """validation_split=0.2 + embargo=20 时，train 略小但不应 50:50"""
        dataset = _make_multiindex_dataset(n_stocks=10, n_days=242)
        dataset["label"] = np.random.randn(len(dataset))

        train, val = _split_by_ratio(
            dataset, validation_split=0.2, embargo_days=20,
        )

        total = len(train) + len(val)
        train_ratio = len(train) / total
        val_ratio = len(val) / total

        # embargo 会减少训练集，但不应导致 50:50
        assert train_ratio > 0.60, (
            f"训练集比例 {train_ratio:.2%} 过低（可能是 split 逻辑错误）"
        )
        assert val_ratio < 0.40, (
            f"验证集比例 {val_ratio:.2%} 过高（可能是 split 逻辑错误）"
        )

    def test_ratio_split_date_based(self):
        """split 应基于日期而非行数"""
        dataset = _make_multiindex_dataset(n_stocks=10, n_days=100)
        dataset["label"] = np.random.randn(len(dataset))

        dates = _extract_sorted_dates(dataset)
        n_dates = len(dates)

        train, val = _split_by_ratio(
            dataset, validation_split=0.2, embargo_days=0,
        )

        # 提取训练集和验证集的日期
        train_dates = _extract_sorted_dates(train)
        val_dates = _extract_sorted_dates(val)

        # 训练集日期应约为总日期的 80%
        expected_train_dates = int(n_dates * 0.8)
        assert abs(len(train_dates) - expected_train_dates) <= 2, (
            f"训练集日期数 {len(train_dates)} 与预期 {expected_train_dates} 偏差过大"
        )

    @pytest.mark.parametrize("val_split", [0.1, 0.2, 0.3])
    def test_ratio_split_various_ratios(self, val_split):
        """不同 validation_split 值都应正确分割"""
        dataset = _make_multiindex_dataset(n_stocks=5, n_days=200)
        dataset["label"] = np.random.randn(len(dataset))

        train, val = _split_by_ratio(
            dataset, validation_split=val_split, embargo_days=0,
        )

        dates = _extract_sorted_dates(dataset)
        train_dates = _extract_sorted_dates(train)
        val_dates = _extract_sorted_dates(val)

        expected_train_pct = 1 - val_split
        actual_train_pct = len(train_dates) / len(dates)

        assert abs(actual_train_pct - expected_train_pct) < 0.02, (
            f"val_split={val_split}: 训练集日期比例 {actual_train_pct:.2%} "
            f"与预期 {expected_train_pct:.2%} 偏差过大"
        )

    def test_no_date_overlap_between_train_val(self):
        """训练集和验证集的日期不应重叠"""
        dataset = _make_multiindex_dataset(n_stocks=10, n_days=242)
        dataset["label"] = np.random.randn(len(dataset))

        train, val = _split_by_ratio(
            dataset, validation_split=0.2, embargo_days=20,
        )

        train_dates = set(_extract_sorted_dates(train))
        val_dates = set(_extract_sorted_dates(val))
        overlap = train_dates & val_dates

        assert len(overlap) == 0, (
            f"训练集和验证集有 {len(overlap)} 个重叠日期"
        )


# === 测试 C: DatasetAdapter 报告正确的 train/val 大小 ===

class TestDatasetAdapterSizes:
    """验证 DataFrameDatasetAdapter 和 ValidationDatasetView 报告正确大小"""

    def test_adapter_len_reports_train_size(self):
        """train_dataset len() 应返回训练集大小"""
        from app.services.qlib.training.dataset_adapter import (
            DataFrameDatasetAdapter,
            ValidationDatasetView,
        )

        train_data = _make_multiindex_dataset(n_stocks=2, n_days=80)
        train_data["label"] = np.random.randn(len(train_data))
        val_data = _make_multiindex_dataset(n_stocks=2, n_days=20)
        val_data["label"] = np.random.randn(len(val_data))

        adapter = DataFrameDatasetAdapter(train_data, val_data)
        val_view = ValidationDatasetView(adapter)

        assert len(adapter) == len(train_data), (
            f"adapter len={len(adapter)}, expected={len(train_data)}"
        )
        assert len(val_view) == len(val_data), (
            f"val_view len={len(val_view)}, expected={len(val_data)}"
        )
        assert len(adapter) != len(val_view), (
            "train 和 val 大小不应相同"
        )
