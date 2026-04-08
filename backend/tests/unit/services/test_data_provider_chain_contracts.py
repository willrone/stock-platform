"""数据提供链 contract tests。"""

import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from app.models.stock_simple import StockData
from app.services.data.parquet_manager import ParquetManager
from app.services.infrastructure.cache_service import LRUCache
from app.services.qlib.enhanced_qlib_provider import EnhancedQlibDataProvider
from app.services.qlib.qlib_data_adapter import QlibDataAdapter


def _build_stock_data(
    stock_code: str,
    trade_date: datetime,
    close_price: float,
) -> StockData:
    """构造最小可用的股票数据样本。"""
    return StockData(
        stock_code=stock_code,
        date=trade_date,
        open=close_price - 0.2,
        high=close_price + 0.3,
        low=close_price - 0.4,
        close=close_price,
        volume=1_000,
        adj_close=close_price,
    )


class TestParquetManagerContracts:
    """验证 Parquet 数据落盘与读取合同。"""

    def test_save_and_load_keeps_unique_rows_sorted(self, tmp_path: Path) -> None:
        """重复保存同月数据时，应保持去重和按日期排序。"""
        manager = ParquetManager(str(tmp_path))
        stock_code = "000001.SZ"
        jan_02 = datetime(2024, 1, 2)
        jan_03 = datetime(2024, 1, 3)
        jan_04 = datetime(2024, 1, 4)

        first_batch = [
            _build_stock_data(stock_code, jan_02, 10.1),
            _build_stock_data(stock_code, jan_03, 10.2),
        ]
        second_batch = [
            _build_stock_data(stock_code, jan_03, 10.2),
            _build_stock_data(stock_code, jan_04, 10.3),
        ]

        assert manager.save_stock_data(first_batch) is True
        assert manager.save_stock_data(second_batch) is True

        loaded = manager.load_stock_data(stock_code, jan_02, jan_04)

        assert [item.date for item in loaded] == [jan_02, jan_03, jan_04]
        assert [item.close for item in loaded] == [10.1, 10.2, 10.3]


class TestCacheContracts:
    """验证内存缓存的 hit/miss/eviction 合同。"""

    def test_lru_cache_tracks_hits_misses_and_evictions(self) -> None:
        """访问热点键后，新写入应淘汰最久未使用的键。"""
        cache = LRUCache(max_size=2, default_ttl=60, memory_limit_mb=10)

        assert cache.put("alpha", "A") is True
        assert cache.put("beta", "B") is True
        assert cache.get("alpha") == "A"
        assert cache.put("gamma", "C") is True

        assert cache.get("beta") is None
        assert cache.get("gamma") == "C"

        stats = cache.get_stats()
        assert stats.hits == 2
        assert stats.misses == 1
        assert stats.evictions == 1
        assert cache.contains("alpha") is True
        assert cache.contains("beta") is False

    def test_lru_cache_respects_ttl_expiration(self) -> None:
        """TTL 到期后，缓存读取应按 miss 处理。"""
        cache = LRUCache(max_size=1, default_ttl=None, memory_limit_mb=10)

        assert cache.put("short-lived", "value", ttl=0.01) is True
        time.sleep(0.03)

        assert cache.get("short-lived") is None
        assert cache.get_stats().misses == 1


class TestQlibAdapterContracts:
    """验证 Qlib adapter 对核心 OHLCV 合同的修复能力。"""

    @pytest.mark.asyncio
    async def test_validate_and_fix_repairs_core_ohlcv_contract(self) -> None:
        """adapter 应修复缺列、high/low 反转和负成交量。"""
        adapter = QlibDataAdapter()
        raw_data = pd.DataFrame(
            {
                "high": [9.0, 11.0],
                "low": [10.0, 10.5],
                "close": [9.8, 10.8],
                "volume": [-100, 200],
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
        assert fixed.index.names == ["instrument", "datetime"]
        assert {"$open", "$high", "$low", "$close", "$volume"}.issubset(
            fixed.columns
        )
        assert (fixed["$high"] >= fixed["$low"]).all()
        assert (fixed["$volume"] >= 0).all()
        assert pd.api.types.is_integer_dtype(fixed["$volume"])


class TestEnhancedProviderContracts:
    """验证 provider 暴露的缓存边界合同。"""

    @pytest.mark.asyncio
    async def test_provider_cache_stats_and_clear_follow_factor_cache_dir(
        self,
        tmp_path: Path,
    ) -> None:
        """provider 应按 factor cache 目录统计并清理 parquet 缓存。"""
        provider = EnhancedQlibDataProvider()
        cache_dir = tmp_path / "factor-cache"
        cache_dir.mkdir()
        provider.alpha_calculator.factor_cache.cache_dir = cache_dir

        (cache_dir / "alpha.parquet").write_text("alpha", encoding="utf-8")
        (cache_dir / "beta.parquet").write_text("beta", encoding="utf-8")
        (cache_dir / "note.txt").write_text("keep", encoding="utf-8")

        stats = await provider.get_cache_stats()

        assert stats["cache_files"] == 2
        assert stats["cache_dir"] == str(cache_dir)
        assert stats["qlib_initialized"] is False

        await provider.clear_cache()

        assert not (cache_dir / "alpha.parquet").exists()
        assert not (cache_dir / "beta.parquet").exists()
        assert (cache_dir / "note.txt").exists()
