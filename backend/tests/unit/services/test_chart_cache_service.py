"""
图表缓存服务测试
测试 ChartCacheService 类的各项功能
"""

from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import AsyncGenerator, Dict

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from app.core import database as database_module
from app.core.database import Base
from app.models import backtest_detailed_models  # noqa: F401 - 注册表
from app.models.backtest_detailed_models import BacktestChartCache
from app.services.backtest.utils.chart_cache_service import ChartCacheService


@pytest_asyncio.fixture
async def db_session(tmp_path, monkeypatch) -> AsyncGenerator[AsyncSession, None]:
    """为图表缓存服务提供隔离的 SQLite 会话。"""
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'chart-cache.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    @asynccontextmanager
    async def _session_context():
        async with async_session() as session:
            try:
                yield session
            finally:
                await session.close()

    monkeypatch.setattr(database_module, "get_async_session_context", _session_context)

    # ChartCacheService 在模块导入时绑定了函数名，也需要同步 patch。
    import app.services.backtest.utils.chart_cache_service as chart_cache_module

    monkeypatch.setattr(
        chart_cache_module, "get_async_session_context", _session_context
    )

    async with async_session() as session:
        yield session

    await engine.dispose()


@pytest.fixture
def cache_service() -> ChartCacheService:
    """创建图表缓存服务实例。"""
    return ChartCacheService()


@pytest.fixture
def sample_chart_data() -> Dict:
    """示例图表数据。"""
    return {
        "dates": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "values": [100, 105, 110],
        "metadata": {"type": "equity_curve", "version": "1.0"},
    }


@pytest.fixture
def sample_task_id() -> str:
    """示例任务ID。"""
    return "test-task-123"


class TestChartCacheService:
    """图表缓存服务测试类。"""

    def test_service_initialization(self, cache_service):
        """测试服务初始化。"""
        assert cache_service is not None
        assert hasattr(cache_service, "logger")
        assert cache_service.DEFAULT_CACHE_EXPIRY_HOURS == 24
        assert isinstance(cache_service.SUPPORTED_CHART_TYPES, list)
        assert len(cache_service.SUPPORTED_CHART_TYPES) > 0

    def test_supported_chart_types(self, cache_service):
        """测试支持的图表类型。"""
        supported_types = cache_service.SUPPORTED_CHART_TYPES

        expected_types = [
            "equity_curve",
            "drawdown_curve",
            "monthly_heatmap",
            "trade_distribution",
        ]

        for chart_type in expected_types:
            assert chart_type in supported_types, f"缺少图表类型: {chart_type}"

    @pytest.mark.asyncio
    async def test_get_cached_chart_data_nonexistent(
        self, cache_service, sample_task_id, db_session
    ):
        """测试获取不存在的缓存数据。"""
        result = await cache_service.get_cached_chart_data(
            sample_task_id, "equity_curve"
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_get_cached_chart_data_unsupported_type(
        self, cache_service, sample_task_id, db_session
    ):
        """测试获取不支持的图表类型。"""
        result = await cache_service.get_cached_chart_data(
            sample_task_id, "unsupported_type"
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_cache_chart_data_basic(
        self, cache_service, sample_task_id, sample_chart_data, db_session
    ):
        """测试基本的缓存图表数据。"""
        result = await cache_service.cache_chart_data(
            sample_task_id, "equity_curve", sample_chart_data
        )

        assert result is True
        cached = await cache_service.get_cached_chart_data(
            sample_task_id, "equity_curve"
        )
        assert cached == sample_chart_data

    @pytest.mark.asyncio
    async def test_cache_chart_data_unsupported_type(
        self, cache_service, sample_task_id, sample_chart_data, db_session
    ):
        """测试缓存不支持的图表类型。"""
        result = await cache_service.cache_chart_data(
            sample_task_id, "unsupported_type", sample_chart_data
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_cache_round_trip_replaces_existing_record(
        self, cache_service, sample_task_id, sample_chart_data, db_session
    ):
        """测试同一任务同一图表类型会更新已有缓存。"""
        assert await cache_service.cache_chart_data(
            sample_task_id, "equity_curve", sample_chart_data
        )

        updated_data = {**sample_chart_data, "values": [200, 205, 210]}
        assert await cache_service.cache_chart_data(
            sample_task_id, "equity_curve", updated_data
        )

        cached_data = await cache_service.get_cached_chart_data(
            sample_task_id, "equity_curve"
        )
        assert cached_data == updated_data

    @pytest.mark.asyncio
    async def test_invalidate_specific_chart_cache(
        self, cache_service, sample_task_id, sample_chart_data, db_session
    ):
        """测试删除特定图表缓存。"""
        await cache_service.cache_chart_data(
            sample_task_id, "equity_curve", sample_chart_data
        )
        await cache_service.cache_chart_data(
            sample_task_id, "drawdown_curve", sample_chart_data
        )

        result = await cache_service.invalidate_cache(sample_task_id, "equity_curve")

        assert result is True
        assert (
            await cache_service.get_cached_chart_data(sample_task_id, "equity_curve")
            is None
        )
        assert (
            await cache_service.get_cached_chart_data(sample_task_id, "drawdown_curve")
            == sample_chart_data
        )

    @pytest.mark.asyncio
    async def test_invalidate_all_chart_cache(
        self, cache_service, sample_task_id, sample_chart_data, db_session
    ):
        """测试删除任务下所有图表缓存。"""
        await cache_service.cache_chart_data(
            sample_task_id, "equity_curve", sample_chart_data
        )
        await cache_service.cache_chart_data(
            sample_task_id, "drawdown_curve", sample_chart_data
        )

        result = await cache_service.invalidate_cache(sample_task_id)

        assert result is True
        assert (
            await cache_service.get_cached_chart_data(sample_task_id, "equity_curve")
            is None
        )
        assert (
            await cache_service.get_cached_chart_data(sample_task_id, "drawdown_curve")
            is None
        )

    @pytest.mark.asyncio
    async def test_cleanup_expired_cache(
        self, cache_service, sample_task_id, sample_chart_data, db_session
    ):
        """测试清理过期缓存。"""
        await cache_service.cache_chart_data(
            sample_task_id, "equity_curve", sample_chart_data, expiry_hours=-1
        )
        await cache_service.cache_chart_data(
            sample_task_id, "drawdown_curve", sample_chart_data, expiry_hours=1
        )

        result = await cache_service.cleanup_expired_cache()

        assert result == 1
        assert (
            await cache_service.get_cached_chart_data(sample_task_id, "equity_curve")
            is None
        )
        assert (
            await cache_service.get_cached_chart_data(sample_task_id, "drawdown_curve")
            == sample_chart_data
        )

    @pytest.mark.asyncio
    async def test_get_cache_statistics(
        self, cache_service, sample_task_id, sample_chart_data, db_session
    ):
        """测试获取缓存统计信息。"""
        await cache_service.cache_chart_data(
            sample_task_id, "equity_curve", sample_chart_data
        )
        db_session.add(
            BacktestChartCache(
                task_id=sample_task_id,
                chart_type="drawdown_curve",
                chart_data=sample_chart_data,
                data_hash="expired",
                expires_at=datetime.now(timezone.utc).replace(tzinfo=None)
                - timedelta(hours=1),
            )
        )
        await db_session.commit()

        stats = await cache_service.get_cache_statistics()

        assert stats["total_cache_records"] == 2
        assert stats["expired_records"] == 1
        assert stats["active_records"] == 1
        assert stats["cache_by_type"]["equity_curve"] == 1
        assert stats["cache_by_type"]["drawdown_curve"] == 1

    def test_default_cache_expiry(self, cache_service):
        """测试默认缓存过期时间。"""
        assert cache_service.DEFAULT_CACHE_EXPIRY_HOURS == 24
        assert isinstance(cache_service.DEFAULT_CACHE_EXPIRY_HOURS, int)
        assert cache_service.DEFAULT_CACHE_EXPIRY_HOURS > 0

    @pytest.mark.asyncio
    async def test_batch_cache_charts(
        self, cache_service, sample_task_id, sample_chart_data, db_session
    ):
        """测试批量缓存多种图表类型。"""
        charts_data = {
            "equity_curve": sample_chart_data,
            "drawdown_curve": sample_chart_data,
            "monthly_heatmap": sample_chart_data,
        }

        results = await cache_service.batch_cache_charts(sample_task_id, charts_data)

        assert results == {
            "equity_curve": True,
            "drawdown_curve": True,
            "monthly_heatmap": True,
        }
        for chart_type in charts_data:
            assert (
                await cache_service.get_cached_chart_data(sample_task_id, chart_type)
                == sample_chart_data
            )

    @pytest.mark.asyncio
    async def test_cache_with_different_task_ids(
        self, cache_service, sample_chart_data, db_session
    ):
        """测试不同任务ID的缓存隔离。"""
        task_id_1 = "task-1"
        task_id_2 = "task-2"
        data_1 = {**sample_chart_data, "values": [1]}
        data_2 = {**sample_chart_data, "values": [2]}

        await cache_service.cache_chart_data(task_id_1, "equity_curve", data_1)
        await cache_service.cache_chart_data(task_id_2, "equity_curve", data_2)

        assert (
            await cache_service.get_cached_chart_data(task_id_1, "equity_curve")
            == data_1
        )
        assert (
            await cache_service.get_cached_chart_data(task_id_2, "equity_curve")
            == data_2
        )

    @pytest.mark.asyncio
    async def test_cache_data_serialization(
        self, cache_service, sample_task_id, db_session
    ):
        """测试缓存数据的序列化。"""
        complex_data = {
            "dates": ["2024-01-01", "2024-01-02"],
            "values": [100, 105],
            "nested": {"key": "value", "number": 42},
            "list": [1, 2, 3],
        }

        result = await cache_service.cache_chart_data(
            sample_task_id, "equity_curve", complex_data
        )

        assert result is True
        assert (
            await cache_service.get_cached_chart_data(sample_task_id, "equity_curve")
            == complex_data
        )

    @pytest.mark.asyncio
    async def test_is_cache_valid(
        self, cache_service, sample_task_id, sample_chart_data, db_session
    ):
        """测试缓存有效性检查。"""
        assert (
            await cache_service.is_cache_valid(sample_task_id, "equity_curve") is False
        )

        await cache_service.cache_chart_data(
            sample_task_id, "equity_curve", sample_chart_data
        )
        data_hash = cache_service._calculate_data_hash(sample_chart_data)

        assert (
            await cache_service.is_cache_valid(sample_task_id, "equity_curve") is True
        )
        assert (
            await cache_service.is_cache_valid(
                sample_task_id, "equity_curve", data_hash
            )
            is True
        )
        assert (
            await cache_service.is_cache_valid(
                sample_task_id, "equity_curve", "different"
            )
            is False
        )

    def test_chart_type_validation(self, cache_service):
        """测试图表类型验证。"""
        for chart_type in cache_service.SUPPORTED_CHART_TYPES:
            assert chart_type in cache_service.SUPPORTED_CHART_TYPES

        unsupported_types = ["invalid_type", "unknown_chart", ""]
        for chart_type in unsupported_types:
            assert chart_type not in cache_service.SUPPORTED_CHART_TYPES
