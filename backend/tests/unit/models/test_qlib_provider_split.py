"""Qlib provider 第一刀拆分回归测试。"""

from pathlib import Path

import pandas as pd
import pytest

from app.services.qlib import QlibDataAdapter
from app.services.qlib.enhanced_qlib_provider import (
    EnhancedQlibDataProvider,
    FactorCache,
)


class TestFactorCacheModule:
    """验证 FactorCache 已稳定抽离到独立模块。"""

    def test_save_and_load_factors_round_trip(self, tmp_path: Path) -> None:
        """应能在独立模块中完成缓存读写。"""
        cache = FactorCache(cache_dir=str(tmp_path))
        cache_key = cache.get_cache_key(
            ["000001.SZ", "000002.SZ"],
            (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-31")),
        )
        factors = pd.DataFrame({"RET1": [0.1, 0.2], "MA5": [1.0, 2.0]})

        cache.save_factors(cache_key, factors)
        cached_factors = cache.get_cached_factors(cache_key)

        assert cached_factors is not None
        pd.testing.assert_frame_equal(cached_factors, factors)


class TestQlibDataAdapter:
    """验证 adapter 承接 DataFrame↔Qlib 转换职责。"""

    def test_convert_to_qlib_format_creates_multiindex(self) -> None:
        """adapter 应创建标准 MultiIndex 并映射基础列名。"""
        adapter = QlibDataAdapter()
        raw_data = pd.DataFrame(
            {
                "stock_code": ["000001.SZ", "000001.SZ"],
                "date": ["2024-01-02", "2024-01-03"],
                "open": [10.0, 10.5],
                "high": [10.3, 10.8],
                "low": [9.8, 10.1],
                "close": [10.1, 10.6],
                "volume": [1000, 1200],
            }
        )

        converted = adapter._convert_to_qlib_format(raw_data)

        assert isinstance(converted.index, pd.MultiIndex)
        assert converted.index.names == ["instrument", "datetime"]
        assert {"$open", "$high", "$low", "$close", "$volume"}.issubset(
            converted.columns
        )

    @pytest.mark.asyncio
    async def test_validate_and_fix_qlib_format_repairs_missing_columns(self) -> None:
        """adapter 应补齐缺失的 Qlib 必要列。"""
        adapter = QlibDataAdapter()
        raw_data = pd.DataFrame(
            {
                "close": [10.0, 10.2],
                "high": [10.5, 10.6],
                "low": [9.9, 10.0],
                "volume": [100, 120],
            },
            index=pd.MultiIndex.from_tuples(
                [
                    ("000001.SZ", pd.Timestamp("2024-01-02")),
                    ("000001.SZ", pd.Timestamp("2024-01-03")),
                ],
                names=["instrument", "datetime"],
            ),
        )

        is_valid, fixed = await adapter.validate_and_fix_qlib_format(raw_data)

        assert is_valid is True
        assert {"$open", "$high", "$low", "$close", "$volume"}.issubset(fixed.columns)


class TestEnhancedProviderDelegation:
    """验证 provider 已退化为薄协调层。"""

    def test_provider_delegates_missing_value_handling(self, monkeypatch) -> None:
        """provider 的同步 helper 应委派给 adapter。"""
        provider = EnhancedQlibDataProvider()
        sample = pd.DataFrame({"$close": [1.0, None]})
        expected = pd.DataFrame({"$close": [1.0, 1.0]})

        def fake_handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
            pd.testing.assert_frame_equal(df, sample)
            return expected

        monkeypatch.setattr(
            provider.data_adapter,
            "_handle_missing_values",
            fake_handle_missing_values,
        )

        result = provider._handle_missing_values(sample)

        pd.testing.assert_frame_equal(result, expected)

    @pytest.mark.asyncio
    async def test_provider_convert_dataframe_to_qlib_uses_adapter(self) -> None:
        """provider 对外接口仍应保持可用。"""
        provider = EnhancedQlibDataProvider()
        raw_data = pd.DataFrame(
            {
                "stock_code": ["000001.SZ", "000001.SZ"],
                "date": ["2024-01-02", "2024-01-03"],
                "open": [10.0, 10.5],
                "high": [10.3, 10.8],
                "low": [9.8, 10.1],
                "close": [10.1, 10.6],
                "volume": [1000, 1200],
            }
        )

        is_valid, converted, conversion_info = await provider.convert_dataframe_to_qlib(
            raw_data
        )

        assert is_valid is True
        assert isinstance(converted.index, pd.MultiIndex)
        assert conversion_info["final_shape"] == converted.shape
