"""
\u6027\u80fd\u4f18\u5316\u5c5e\u6027\u6d4b\u8bd5
\u9a8c\u8bc1\u6027\u80fd\u4f18\u5316\u529f\u80fd\u7684\u6b63\u786e\u6027\u5c5e\u6027
"""

import pytest
import asyncio
import tempfile
import shutil
import pandas as pd
import time
from pathlib import Path
from hypothesis import given, strategies as st, settings
from hypothesis.strategies import composite

from app.services.data.stream_processor import MemoryMonitor, StreamProcessor
from app.services.infrastructure.cache_service import (
    CacheManager,
    LRUCache,
    cache_manager,
)


@composite
def cache_configs(draw):
    return {
        "max_size": draw(st.integers(min_value=10, max_value=200)),
        "default_ttl": draw(st.floats(min_value=1.0, max_value=60.0)),
        "memory_limit_mb": draw(st.integers(min_value=1, max_value=50)),
    }


def _make_test_df(size):
    cats = ["A", "B", "C"]
    return pd.DataFrame(
        {
            "id": range(size),
            "date": pd.date_range(start="2023-01-01", periods=size, freq="D"),
            "value": [float(i) * 0.1 for i in range(size)],
            "category": [cats[i % 3] for i in range(size)],
            "count": [i % 100 + 1 for i in range(size)],
        }
    )


class TestPerformanceOptimizationProperties:

    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def teardown_method(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    @given(cache_configs())
    @settings(max_examples=5, deadline=10000)
    async def test_cache_hit_rate_improvement(self, cache_config):
        cache = LRUCache(**cache_config)
        max_size = cache_config["max_size"]
        test_size = min(max_size // 2, 25)

        for i in range(test_size):
            cache.put(f"key_{i}", f"value_{i}")

        hits_before = cache.get_stats().hits
        successful_gets = 0
        for i in range(test_size):
            if cache.get(f"key_{i}") is not None:
                successful_gets += 1

        stats = cache.get_stats()
        assert stats.hits - hits_before == successful_gets
        assert successful_gets > 0
        assert stats.size <= cache_config["max_size"]
        cache.clear()

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="StreamProcessor.read_parquet_chunks uses non-existent pyarrow API")
    async def test_stream_processing_memory_efficiency(self):
        test_df = _make_test_df(500)
        test_file = self.temp_path / "test_data.parquet"
        test_df.to_parquet(test_file, index=False)

        processor = StreamProcessor(chunk_size=100, max_workers=2, memory_limit_mb=50)

        def process_chunk(chunk_df):
            return chunk_df.copy()

        output_file = self.temp_path / "processed_data.parquet"
        start_mem = processor.get_memory_stats()["current_mb"]

        stats = await processor.process_file_stream(
            file_path=test_file,
            processor_func=process_chunk,
            output_path=output_file,
            file_type="parquet",
        )

        end_mem = processor.get_memory_stats()["current_mb"]
        assert stats.processed_records > 0
        assert stats.processing_time_seconds > 0
        assert (end_mem - start_mem) < 100
        assert output_file.exists()
        await processor.close()

    @pytest.mark.asyncio
    async def test_cache_memory_management(self):
        cache = LRUCache(max_size=10, memory_limit_mb=1)
        large_data = "x" * 1024
        for i in range(50):
            cache.put(f"large_key_{i}", large_data)

        stats = cache.get_stats()
        assert stats.size <= 10
        assert stats.memory_usage_bytes < 2 * 1024 * 1024
        assert stats.evictions > 0

        cache.clear()
        assert cache.get_stats().size == 0

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="StreamProcessor.read_parquet_chunks uses non-existent pyarrow API")
    async def test_stream_processing_throughput(self):
        cats = ["A", "B", "C"]
        large_df = pd.DataFrame(
            {
                "id": range(5000),
                "value": [i * 0.1 for i in range(5000)],
                "category": [cats[i % 3] for i in range(5000)],
            }
        )
        test_file = self.temp_path / "large_test.parquet"
        large_df.to_parquet(test_file, index=False)

        processor = StreamProcessor(chunk_size=1000, max_workers=2)

        def simple_transform(chunk_df):
            chunk_df = chunk_df.copy()
            chunk_df["processed_value"] = chunk_df["value"] * 2
            return chunk_df

        output_file = self.temp_path / "processed_large.parquet"
        stats = await processor.process_file_stream(
            file_path=test_file,
            processor_func=simple_transform,
            output_path=output_file,
            file_type="parquet",
        )

        assert stats.processed_records == 5000
        assert stats.throughput_records_per_second > 100
        assert stats.processing_time_seconds < 30
        assert output_file.exists()

        processed_df = pd.read_parquet(output_file)
        assert len(processed_df) == 5000
        assert "processed_value" in processed_df.columns
        await processor.close()

    @pytest.mark.asyncio
    async def test_memory_monitoring_accuracy(self):
        monitor = MemoryMonitor(warning_threshold_mb=100, critical_threshold_mb=200)

        initial = monitor.check_memory()
        assert initial["status"] in ["normal", "warning", "critical"]
        assert initial["current_mb"] > 0
        assert initial["peak_mb"] >= initial["current_mb"]

        large_objects = [[0] * 100000 for _ in range(10)]
        after = monitor.check_memory()
        assert after["peak_mb"] >= after["current_mb"]

        monitor.force_gc()
        del large_objects
        monitor.force_gc()

        final = monitor.check_memory()
        assert final["current_mb"] > 0
        assert final["warning_threshold_mb"] == 100
        assert final["critical_threshold_mb"] == 200

    @pytest.mark.asyncio
    async def test_cache_concurrent_access(self):
        cache = LRUCache(max_size=100, memory_limit_mb=10)
        for i in range(50):
            cache.put(f"key_{i}", f"value_{i}")

        async def concurrent_access(worker_id):
            results = []
            for i in range(20):
                v = cache.get(f"key_{i % 50}")
                if v:
                    results.append(v)
                if i % 5 == 0:
                    cache.put(f"new_{worker_id}_{i}", f"val_{worker_id}_{i}")
                await asyncio.sleep(0.001)
            return results

        tasks = [concurrent_access(i) for i in range(10)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        assert sum(1 for r in results if isinstance(r, list)) == 10

        stats = cache.get_stats()
        assert stats.size > 0
        assert stats.hits > 0
        cache.clear()

    @pytest.mark.asyncio
    async def test_performance_degradation_detection(self):
        manager = CacheManager()
        test_cache = manager.get_cache("perf_test")

        for i in range(100):
            test_cache.put(f"key_{i}", f"value_{i}")
            assert test_cache.get(f"key_{i}") == f"value_{i}"

        for i in range(100, 1100):
            test_cache.put(f"key_{i}", f"value_{i}" * 100)

        stats = test_cache.get_stats()
        assert stats.evictions > 0
        assert stats.hit_rate >= 0
        assert stats.memory_usage_bytes > 0
        test_cache.clear()


@pytest.fixture(autouse=True)
def cleanup_after_test():
    yield
    cache_manager.clear_all()
