"""
回测详细仓库单元测试
使用内存 SQLite 测试 BacktestDetailedRepository CRUD
"""

import asyncio
from datetime import datetime
from typing import AsyncGenerator

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from app.core.database import Base
from app.models import backtest_detailed_models  # noqa: F401 - 注册表
from app.repositories.backtest_detailed_repository import BacktestDetailedRepository

# 尝试导入 pytest_asyncio，如果不可用则使用替代方案
try:
    import pytest_asyncio

    @pytest_asyncio.fixture
    async def db_engine(tmp_path):
        """内存 SQLite 引擎"""
        url = f"sqlite+aiosqlite:///{tmp_path / 'test.db'}"
        engine = create_async_engine(url, echo=False)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        yield engine
        await engine.dispose()

    @pytest_asyncio.fixture
    async def db_session(db_engine) -> AsyncGenerator[AsyncSession, None]:
        """异步数据库会话"""
        async_session = sessionmaker(
            db_engine, class_=AsyncSession, expire_on_commit=False
        )
        async with async_session() as session:
            yield session

except ImportError:
    # 如果没有 pytest_asyncio，使用同步 fixture + asyncio.run
    @pytest.fixture
    def db_engine(tmp_path):
        """创建异步数据库引擎"""
        url = f"sqlite+aiosqlite:///{tmp_path / 'test.db'}"
        engine = create_async_engine(url, echo=False)

        async def _create_tables():
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)

        asyncio.run(_create_tables())
        yield engine

        async def _dispose():
            await engine.dispose()

        asyncio.run(_dispose())

    @pytest.fixture
    def db_session_factory(db_engine):
        """返回创建 session 的工厂函数"""
        async_session = sessionmaker(
            db_engine, class_=AsyncSession, expire_on_commit=False
        )

        async def _get_session():
            async with async_session() as session:
                yield session

        return _get_session


class TestBacktestDetailedRepository:
    """回测详细仓库测试类"""

    @pytest.mark.asyncio
    async def test_create_and_get_detailed_result(self, db_session):
        """创建并获取回测详细结果"""
        repo = BacktestDetailedRepository(db_session)
        task_id = "task-repo-1"
        backtest_id = "bt-repo-1"
        extended = {"sortino_ratio": 1.2, "calmar_ratio": 0.8}
        analysis = {"position_analysis": {"stock_performance": []}}

        created = await repo.create_detailed_result(
            task_id=task_id,
            backtest_id=backtest_id,
            extended_metrics=extended,
            analysis_data=analysis,
        )
        assert created is not None
        await db_session.commit()

        got = await repo.get_detailed_result_by_task_id(task_id)
        assert got is not None
        assert got.task_id == task_id
        assert got.backtest_id == backtest_id
        assert got.sortino_ratio == 1.2
        d = got.to_dict()
        assert d["task_id"] == task_id
        assert "position_analysis" in d

    @pytest.mark.asyncio
    async def test_get_detailed_result_not_found(self, db_session):
        """获取不存在的任务返回 None"""
        repo = BacktestDetailedRepository(db_session)
        got = await repo.get_detailed_result_by_task_id("nonexistent")
        assert got is None

    @pytest.mark.asyncio
    async def test_batch_create_and_get_portfolio_snapshots(self, db_session):
        """批量创建并获取组合快照"""
        repo = BacktestDetailedRepository(db_session)
        task_id = "task-snap-1"
        backtest_id = "bt-snap-1"
        snapshots_data = [
            {
                "date": "2024-01-01",
                "portfolio_value": 100_000.0,
                "cash": 50_000.0,
                "positions_count": 2,
                "total_return": 0.0,
                "drawdown": 0.0,
            },
            {
                "date": "2024-01-02",
                "portfolio_value": 101_000.0,
                "cash": 49_000.0,
                "positions_count": 2,
                "total_return": 0.01,
                "drawdown": 0.0,
            },
        ]

        ok = await repo.batch_create_portfolio_snapshots(
            task_id=task_id,
            backtest_id=backtest_id,
            snapshots_data=snapshots_data,
        )
        assert ok is True
        await db_session.commit()

        snapshots = await repo.get_portfolio_snapshots(task_id=task_id)
        assert len(snapshots) == 2
        assert snapshots[0].portfolio_value == 100_000.0
        assert snapshots[1].portfolio_value == 101_000.0

    @pytest.mark.asyncio
    async def test_batch_create_and_get_trade_records(self, db_session):
        """批量创建并获取交易记录"""
        repo = BacktestDetailedRepository(db_session)
        task_id = "task-trade-1"
        backtest_id = "bt-trade-1"
        trades_data = [
            {
                "trade_id": "t1",
                "stock_code": "000001.SZ",
                "stock_name": "平安银行",
                "action": "BUY",
                "quantity": 1000,
                "price": 10.0,
                "timestamp": "2024-01-01T10:00:00",
                "commission": 5.0,
                "pnl": None,
            },
            {
                "trade_id": "t2",
                "stock_code": "000001.SZ",
                "stock_name": "平安银行",
                "action": "SELL",
                "quantity": 1000,
                "price": 11.0,
                "timestamp": "2024-01-05T10:00:00",
                "commission": 5.0,
                "pnl": 990.0,
            },
        ]

        ok = await repo.batch_create_trade_records(
            task_id=task_id,
            backtest_id=backtest_id,
            trades_data=trades_data,
        )
        assert ok is True
        await db_session.commit()

        trades = await repo.get_trade_records(
            task_id=task_id, order_by="timestamp", order_desc=False
        )
        assert len(trades) == 2
        # 按时间戳升序：BUY (01-01) 在前，SELL (01-05) 在后
        assert trades[0].action == "BUY"
        assert trades[1].action == "SELL"
        assert trades[1].pnl == 990.0

    @pytest.mark.asyncio
    async def test_get_trade_statistics(self, db_session):
        """获取交易统计"""
        repo = BacktestDetailedRepository(db_session)
        task_id = "task-stats-1"
        backtest_id = "bt-stats-1"
        trades_data = [
            {
                "trade_id": "s1",
                "stock_code": "000001.SZ",
                "stock_name": "平安",
                "action": "BUY",
                "quantity": 100,
                "price": 10.0,
                "timestamp": "2024-01-01T09:00:00",
                "commission": 0.0,
                "pnl": None,
            },
            {
                "trade_id": "s2",
                "stock_code": "000001.SZ",
                "stock_name": "平安",
                "action": "SELL",
                "quantity": 100,
                "price": 12.0,
                "timestamp": "2024-01-10T09:00:00",
                "commission": 0.0,
                "pnl": 200.0,
            },
        ]
        await repo.batch_create_trade_records(
            task_id=task_id,
            backtest_id=backtest_id,
            trades_data=trades_data,
        )
        await db_session.commit()

        stats = await repo.get_trade_statistics(task_id)
        assert isinstance(stats, dict)
        assert "total_trades" in stats
        assert stats["total_trades"] == 2
        assert stats["buy_trades"] == 1
        assert stats["sell_trades"] == 1

    @pytest.mark.asyncio
    async def test_update_detailed_result(self, db_session):
        """更新回测详细结果"""
        repo = BacktestDetailedRepository(db_session)
        task_id = "task-update-1"
        backtest_id = "bt-update-1"

        await repo.create_detailed_result(
            task_id=task_id,
            backtest_id=backtest_id,
            extended_metrics={"sortino_ratio": 1.0},
            analysis_data={},
        )
        await db_session.commit()

        ok = await repo.update_detailed_result(
            task_id=task_id,
            extended_metrics={"sortino_ratio": 2.0, "calmar_ratio": 1.0},
            analysis_data={"position_analysis": {"updated": True}},
        )
        assert ok is True
        await db_session.commit()

        got = await repo.get_detailed_result_by_task_id(task_id)
        assert got is not None
        assert got.sortino_ratio == 2.0
        assert got.calmar_ratio == 1.0
        assert got.position_analysis == {"updated": True}
