"""
ST1：数据库集成测试 — 持久化层完整读写链路

使用 SQLite in-memory 数据库，测试 BacktestDetailedRepository 的完整 CRUD 操作。
覆盖：回测记录、快照、交易记录、信号记录、统计数据、事务回滚、幂等性。

注意：项目的 app.core.database 会在 import 时创建 PostgreSQL 引擎，
因此本测试在 import 之前先 patch 环境变量和 settings，使其指向 SQLite。
"""

import os
import uuid
from datetime import datetime, timezone

# ── 在导入任何 app 模块之前，先 patch DATABASE_URL 为 SQLite ──
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"

import pytest
import pytest_asyncio
import sqlalchemy as sa
import sqlalchemy.ext.asyncio as sa_async
from sqlalchemy import JSON, String, event, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

# ── patch 引擎创建，避免 app.core.database 连接 PostgreSQL ──
_orig_create_async_engine = sa_async.create_async_engine
_orig_create_engine = sa.create_engine
_dummy_async = _orig_create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
_dummy_sync = _orig_create_engine("sqlite:///:memory:", echo=False)

_PG_ONLY_KW = {"pool_size", "max_overflow", "pool_timeout", "pool_pre_ping",
                "pool_recycle", "json_serializer"}

def _fake_async_engine(url, **kw):
    if "postgresql" in str(url) or "asyncpg" in str(url):
        return _dummy_async
    return _orig_create_async_engine(url, **{k: v for k, v in kw.items() if k not in _PG_ONLY_KW})

def _fake_sync_engine(url, **kw):
    if "postgresql" in str(url):
        return _dummy_sync
    return _orig_create_engine(url, **{k: v for k, v in kw.items() if k not in _PG_ONLY_KW})

sa_async.create_async_engine = _fake_async_engine
sa.create_engine = _fake_sync_engine

# 现在可以安全 import app 模块了
from app.core.database import Base
from app.models.backtest_detailed_models import (
    BacktestStatistics,
    PortfolioSnapshot,
    SignalRecord,
    TradeRecord,
)
from app.models.task_models import BacktestResult, Task
from app.repositories.backtest_detailed_repository import BacktestDetailedRepository


# ────────────────────────── Fixtures ──────────────────────────


def _remap_pg_types_for_sqlite(metadata):
    """将 PostgreSQL 特有类型（JSONB, UUID, Enum）映射为 SQLite 兼容类型"""
    from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
    from sqlalchemy import Enum as SA_Enum

    for table in metadata.tables.values():
        for col in table.columns:
            if isinstance(col.type, JSONB):
                col.type = JSON()
            elif isinstance(col.type, PG_UUID):
                col.type = String(36)
            elif isinstance(col.type, SA_Enum):
                col.type = String(50)
            # 清除 PostgreSQL 特有的 server_default（如 gen_random_uuid()）
            if col.server_default is not None:
                sd_text = str(col.server_default.arg) if hasattr(col.server_default, 'arg') else str(col.server_default)
                if "gen_random_uuid" in sd_text or "uuid" in sd_text.lower():
                    col.server_default = None
                    # 为 UUID 主键列添加 Python 侧默认值
                    if col.primary_key:
                        col.default = sa.ColumnDefault(lambda: str(uuid.uuid4()))


@pytest_asyncio.fixture
async def engine():
    """创建 SQLite in-memory 异步引擎，自动将 PG 类型映射为 SQLite 兼容类型"""
    eng = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)

    @event.listens_for(eng.sync_engine, "connect")
    def _set_sqlite_pragma(dbapi_conn, connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    _remap_pg_types_for_sqlite(Base.metadata)

    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture
async def session(engine):
    """创建异步 session"""
    _Session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with _Session() as sess:
        yield sess


@pytest_asyncio.fixture
async def repo(session):
    """创建 BacktestDetailedRepository 实例"""
    return BacktestDetailedRepository(session)


def _make_task_id() -> str:
    return str(uuid.uuid4())


def _make_backtest_id() -> str:
    return str(uuid.uuid4())


async def _insert_task(session: AsyncSession, task_id: str) -> None:
    """插入一条 tasks 占位行（满足外键约束）"""
    task = Task(
        task_id=task_id,
        task_name="test_task",
        task_type="backtest",
        status="running",
    )
    session.add(task)
    await session.flush()


async def _insert_backtest_result(
    session: AsyncSession, task_id: str, backtest_id: str
) -> None:
    """插入一条 backtest_results 占位行（满足子表外键约束）"""
    br = BacktestResult(
        task_id=task_id,
        backtest_id=backtest_id,
        strategy_name="rsi",
        start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end_date=datetime(2024, 6, 30, tzinfo=timezone.utc),
        initial_cash=100000,
        final_value=110000,
        total_return=0.1,
        annualized_return=0.2,
        volatility=0.15,
        sharpe_ratio=1.5,
        max_drawdown=0.05,
        win_rate=0.6,
        profit_factor=1.8,
        total_trades=20,
    )
    session.add(br)
    await session.flush()


# ────────────────────────── 测试用例 ──────────────────────────


class TestBacktestResultCRUD:
    """测试回测记录的创建与读取"""

    @pytest.mark.asyncio
    async def test_create_and_read_backtest_result(self, session, repo):
        """1. 创建回测记录 → 写入快照数据 → 读取验证完整性"""
        task_id = _make_task_id()
        backtest_id = _make_backtest_id()
        await _insert_task(session, task_id)

        report = {
            "strategy_name": "rsi",
            "start_date": datetime(2024, 1, 1, tzinfo=timezone.utc),
            "end_date": datetime(2024, 6, 30, tzinfo=timezone.utc),
            "initial_cash": 100000.0,
            "final_value": 115000.0,
            "total_return": 0.15,
            "annualized_return": 0.30,
            "volatility": 0.18,
            "sharpe_ratio": 1.67,
            "max_drawdown": 0.08,
            "win_rate": 0.65,
            "profit_factor": 2.0,
            "total_trades": 30,
        }
        extended_metrics = {
            "sortino_ratio": 2.1,
            "calmar_ratio": 3.5,
            "max_drawdown_duration": 15,
            "var_95": 0.02,
            "downside_deviation": 0.01,
        }
        analysis_data = {
            "drawdown_analysis": {"max_drawdown": 0.08},
            "monthly_returns": [{"month": "2024-01", "return": 0.03}],
        }

        result = await repo.create_detailed_result(
            task_id=task_id,
            backtest_id=backtest_id,
            extended_metrics=extended_metrics,
            analysis_data=analysis_data,
            backtest_report=report,
        )
        await session.commit()

        assert result is not None
        assert str(result.backtest_id) == backtest_id

        # 读取验证
        fetched = await repo.get_detailed_result_by_task_id(task_id)
        assert fetched is not None
        assert fetched.strategy_name == "rsi"
        assert float(fetched.total_return) == pytest.approx(0.15, abs=1e-6)
        assert fetched.sortino_ratio == pytest.approx(2.1, abs=1e-6)
        assert fetched.drawdown_analysis == {"max_drawdown": 0.08}

    @pytest.mark.asyncio
    async def test_update_existing_backtest_result(self, session, repo):
        """占位行 → UPDATE 填充完整数据"""
        task_id = _make_task_id()
        backtest_id = _make_backtest_id()
        await _insert_task(session, task_id)
        await _insert_backtest_result(session, task_id, backtest_id)

        updated = await repo.create_detailed_result(
            task_id=task_id,
            backtest_id=backtest_id,
            extended_metrics={"sortino_ratio": 3.0},
            analysis_data={"drawdown_analysis": {"max_drawdown": 0.12}},
            backtest_report={
                "strategy_name": "rsi",
                "total_return": 0.25,
                "total_trades": 50,
                "start_date": datetime(2024, 1, 1, tzinfo=timezone.utc),
                "end_date": datetime(2024, 6, 30, tzinfo=timezone.utc),
                "initial_cash": 100000,
                "final_value": 125000,
                "annualized_return": 0.5,
                "volatility": 0.2,
                "sharpe_ratio": 2.5,
                "max_drawdown": 0.12,
                "win_rate": 0.7,
                "profit_factor": 2.5,
            },
        )
        await session.commit()

        fetched = await repo.get_detailed_result_by_task_id(task_id)
        assert float(fetched.total_return) == pytest.approx(0.25, abs=1e-6)
        assert fetched.total_trades == 50


class TestPortfolioSnapshots:
    """测试组合快照的批量写入与读取"""

    @pytest.mark.asyncio
    async def test_write_and_read_snapshots(self, session, repo):
        """1. 写入快照数据 → 读取验证完整性"""
        task_id = _make_task_id()
        backtest_id = _make_backtest_id()
        await _insert_task(session, task_id)
        await _insert_backtest_result(session, task_id, backtest_id)

        snapshots_data = [
            {
                "date": datetime(2024, 1, 2, tzinfo=timezone.utc),
                "portfolio_value": 100500.0,
                "cash": 50000.0,
                "positions_count": 3,
                "total_return": 0.005,
                "drawdown": 0.0,
                "positions": {"000001.SZ": {"quantity": 100}},
            },
            {
                "date": datetime(2024, 1, 3, tzinfo=timezone.utc),
                "portfolio_value": 101000.0,
                "cash": 48000.0,
                "positions_count": 4,
                "total_return": 0.01,
                "drawdown": 0.0,
                "positions": {"000001.SZ": {"quantity": 100}, "000002.SZ": {"quantity": 200}},
            },
        ]

        ok = await repo.batch_create_portfolio_snapshots(backtest_id, snapshots_data)
        await session.commit()
        assert ok is True

        results = await repo.get_portfolio_snapshots(backtest_id)
        assert len(results) == 2
        assert float(results[0].portfolio_value) == pytest.approx(100500.0, abs=0.01)
        assert results[1].positions_count == 4


class TestTradeRecords:
    """测试交易记录的批量写入与读取"""

    @pytest.mark.asyncio
    async def test_write_and_read_trades(self, session, repo):
        """2. 写入交易记录 → 读取验证字段正确"""
        task_id = _make_task_id()
        backtest_id = _make_backtest_id()
        await _insert_task(session, task_id)
        await _insert_backtest_result(session, task_id, backtest_id)

        trades_data = [
            {
                "trade_id": "T001",
                "stock_code": "000001.SZ",
                "stock_name": "平安银行",
                "action": "BUY",
                "quantity": 100,
                "price": 15.50,
                "timestamp": datetime(2024, 1, 5, 9, 30, tzinfo=timezone.utc),
                "commission": 15.50,
                "pnl": None,
                "holding_days": None,
                "technical_indicators": {"rsi": 28.5},
            },
            {
                "trade_id": "T002",
                "stock_code": "000001.SZ",
                "stock_name": "平安银行",
                "action": "SELL",
                "quantity": 100,
                "price": 16.80,
                "timestamp": datetime(2024, 2, 10, 14, 0, tzinfo=timezone.utc),
                "commission": 16.80,
                "pnl": 113.70,
                "holding_days": 36,
                "technical_indicators": {"rsi": 72.3},
            },
        ]

        ok = await repo.batch_create_trade_records(backtest_id, trades_data)
        await session.commit()
        assert ok is True

        records = await repo.get_trade_records(backtest_id, limit=50)
        assert len(records) == 2

        buy_record = [r for r in records if (r.action.value if hasattr(r.action, 'value') else r.action) == "BUY"][0]
        assert buy_record.stock_code == "000001.SZ"
        assert buy_record.quantity == 100
        assert float(buy_record.price) == pytest.approx(15.50, abs=0.01)

        sell_record = [r for r in records if (r.action.value if hasattr(r.action, 'value') else r.action) == "SELL"][0]
        assert float(sell_record.pnl) == pytest.approx(113.70, abs=0.01)
        assert sell_record.holding_days == 36

    @pytest.mark.asyncio
    async def test_trade_records_count(self, session, repo):
        """验证交易记录计数"""
        task_id = _make_task_id()
        backtest_id = _make_backtest_id()
        await _insert_task(session, task_id)
        await _insert_backtest_result(session, task_id, backtest_id)

        trades_data = [
            {
                "trade_id": f"T{i:03d}",
                "stock_code": "000001.SZ",
                "action": "BUY" if i % 2 == 0 else "SELL",
                "quantity": 100,
                "price": 15.0 + i * 0.1,
                "timestamp": datetime(2024, 1, i + 1, tzinfo=timezone.utc),
            }
            for i in range(10)
        ]

        await repo.batch_create_trade_records(backtest_id, trades_data)
        await session.commit()

        total = await repo.get_trade_records_count(backtest_id)
        assert total == 10

        buy_count = await repo.get_trade_records_count(backtest_id, action="BUY")
        assert buy_count == 5


class TestSignalRecords:
    """测试信号���录的写入与读取"""

    @pytest.mark.asyncio
    async def test_write_and_read_signals(self, session, repo):
        """3. 写入信号记录 → 读取验证"""
        task_id = _make_task_id()
        backtest_id = _make_backtest_id()
        await _insert_task(session, task_id)
        await _insert_backtest_result(session, task_id, backtest_id)

        signals_data = [
            {
                "signal_id": "S001",
                "stock_code": "000001.SZ",
                "stock_name": "平安银行",
                "signal_type": "BUY",
                "timestamp": datetime(2024, 1, 5, 9, 30, tzinfo=timezone.utc),
                "price": 15.50,
                "strength": 0.85,
                "reason": "RSI 超卖反弹",
                "metadata": {"rsi": 28.5},
                "executed": True,
            },
            {
                "signal_id": "S002",
                "stock_code": "000002.SZ",
                "stock_name": "万科A",
                "signal_type": "SELL",
                "timestamp": datetime(2024, 1, 6, 10, 0, tzinfo=timezone.utc),
                "price": 12.30,
                "strength": 0.72,
                "reason": "RSI 超买",
                "metadata": {"rsi": 75.2},
                "executed": False,
                "execution_reason": "持仓不足",
            },
        ]

        ok = await repo.batch_save_signal_records(backtest_id, signals_data)
        await session.commit()
        assert ok is True

        records = await repo.get_signal_records(backtest_id, limit=50)
        assert len(records) == 2

        buy_sig = [r for r in records if (r.signal_type.value if hasattr(r.signal_type, 'value') else r.signal_type) == "BUY"][0]
        assert buy_sig.stock_code == "000001.SZ"
        assert buy_sig.strength == pytest.approx(0.85, abs=0.01)
        assert buy_sig.executed is True

        sell_sig = [r for r in records if (r.signal_type.value if hasattr(r.signal_type, 'value') else r.signal_type) == "SELL"][0]
        assert sell_sig.executed is False

    @pytest.mark.asyncio
    async def test_signal_records_count_and_filter(self, session, repo):
        """验证信号记录计数和过滤"""
        task_id = _make_task_id()
        backtest_id = _make_backtest_id()
        await _insert_task(session, task_id)
        await _insert_backtest_result(session, task_id, backtest_id)

        signals_data = [
            {
                "signal_id": f"S{i:03d}",
                "stock_code": f"00000{i % 3 + 1}.SZ",
                "signal_type": "BUY" if i % 2 == 0 else "SELL",
                "timestamp": datetime(2024, 1, i + 1, 9, 30, tzinfo=timezone.utc),
                "price": 10.0 + i,
                "strength": 0.5 + i * 0.05,
                "executed": i % 3 == 0,
            }
            for i in range(6)
        ]

        await repo.batch_save_signal_records(backtest_id, signals_data)
        await session.commit()

        total = await repo.get_signal_records_count(backtest_id)
        assert total == 6

        buy_count = await repo.get_signal_records_count(backtest_id, signal_type="BUY")
        assert buy_count == 3

        executed_count = await repo.get_signal_records_count(backtest_id, executed=True)
        assert executed_count == 2  # i=0, i=3


class TestStatistics:
    """测试统计数据的写入与读取"""

    @pytest.mark.asyncio
    async def test_write_and_read_statistics(self, session, repo):
        """4. 写入统计数据 → 读取验证"""
        task_id = _make_task_id()
        backtest_id = _make_backtest_id()
        await _insert_task(session, task_id)
        await _insert_backtest_result(session, task_id, backtest_id)

        stats = BacktestStatistics(
            backtest_id=backtest_id,
            total_signals=100,
            buy_signals=55,
            sell_signals=45,
            executed_signals=80,
            unexecuted_signals=20,
            execution_rate=0.8,
            avg_signal_strength=0.65,
            total_trades=60,
            buy_trades=30,
            sell_trades=30,
            winning_trades=20,
            losing_trades=10,
            win_rate=0.667,
            avg_profit=500.0,
            avg_loss=-300.0,
            profit_factor=1.67,
            total_commission=180.0,
            total_pnl=7000.0,
            avg_holding_days=12.5,
        )
        session.add(stats)
        await session.commit()

        # 通过 repo 的 get_trade_statistics 读取
        trade_stats = await repo.get_trade_statistics(backtest_id)
        assert trade_stats["total_trades"] == 60
        assert trade_stats["win_rate"] == pytest.approx(0.667, abs=0.001)
        assert float(trade_stats["total_pnl"]) == pytest.approx(7000.0, abs=0.01)

        # 通过 repo 的 get_signal_statistics 读取
        signal_stats = await repo.get_signal_statistics(backtest_id)
        assert signal_stats["total_signals"] == 100
        assert signal_stats["execution_rate"] == pytest.approx(0.8, abs=0.01)


class TestTransactionRollback:
    """测试事务回滚：模拟写入中途失败，验证数据一致性"""

    @pytest.mark.asyncio
    async def test_rollback_on_partial_write(self, session, repo):
        """5. 事务回滚：模拟写入中途失败，验证不会出现半写入状态"""
        task_id = _make_task_id()
        backtest_id = _make_backtest_id()
        await _insert_task(session, task_id)
        await _insert_backtest_result(session, task_id, backtest_id)
        await session.commit()

        # 在新事务中：先写入快照，再故意制造错误
        snapshots_data = [
            {
                "date": datetime(2024, 1, 2, tzinfo=timezone.utc),
                "portfolio_value": 100500.0,
                "cash": 50000.0,
            },
        ]
        await repo.batch_create_portfolio_snapshots(backtest_id, snapshots_data)
        await session.flush()

        # 模拟写入交易记录时失败 → 回滚整个事务
        try:
            bad_trade = TradeRecord(
                backtest_id=backtest_id,
                trade_id="BAD",
                stock_code="000001.SZ",
                action="INVALID_ACTION",  # 非法枚举值
                quantity=100,
                price=10.0,
                timestamp=datetime(2024, 1, 5, tzinfo=timezone.utc),
            )
            session.add(bad_trade)
            await session.flush()
            # 如果 SQLite 没有严格枚举检查，手动抛异常模拟失败
            raise ValueError("模拟写入失败")
        except Exception:
            await session.rollback()

        # 回滚后，之前 flush 的快照也应该被撤销
        snapshots = await repo.get_portfolio_snapshots(backtest_id)
        assert len(snapshots) == 0, "事务回滚后不应有残留快照数据"


class TestIdempotency:
    """测试幂等性：同一 task_id 重复写入不应产生重复数据"""

    @pytest.mark.asyncio
    async def test_duplicate_backtest_result_is_idempotent(self, session, repo):
        """6. 幂等性：同一 backtest_id 重复调用 create_detailed_result 不产生重复"""
        task_id = _make_task_id()
        backtest_id = _make_backtest_id()
        await _insert_task(session, task_id)

        report = {
            "strategy_name": "rsi",
            "total_return": 0.10,
            "total_trades": 20,
            "start_date": datetime(2024, 1, 1, tzinfo=timezone.utc),
            "end_date": datetime(2024, 6, 30, tzinfo=timezone.utc),
            "initial_cash": 100000,
            "final_value": 110000,
            "annualized_return": 0.2,
            "volatility": 0.15,
            "sharpe_ratio": 1.5,
            "max_drawdown": 0.05,
            "win_rate": 0.6,
            "profit_factor": 1.8,
        }

        # 第一次写入
        await repo.create_detailed_result(
            task_id=task_id,
            backtest_id=backtest_id,
            extended_metrics={},
            analysis_data={},
            backtest_report=report,
        )
        await session.commit()

        # 第二次写入（相同 backtest_id，更新数据）
        report["total_return"] = 0.20
        await repo.create_detailed_result(
            task_id=task_id,
            backtest_id=backtest_id,
            extended_metrics={},
            analysis_data={},
            backtest_report=report,
        )
        await session.commit()

        # 应该只有一条记录，且是更新后的值
        fetched = await repo.get_detailed_result_by_task_id(task_id)
        assert fetched is not None
        assert float(fetched.total_return) == pytest.approx(0.20, abs=1e-6)

        # 验证数据库中只有一条 backtest_results 记录
        from sqlalchemy import select as sa_select, func as sa_func
        count_stmt = sa_select(sa_func.count()).select_from(BacktestResult).where(
            BacktestResult.task_id == task_id
        )
        count_result = await session.execute(count_stmt)
        assert count_result.scalar() == 1


class TestDeleteTaskData:
    """测试数据删除"""

    @pytest.mark.asyncio
    async def test_delete_cascades_to_subtables(self, session, repo):
        """删除任务数据时，子表数据也应被清理"""
        task_id = _make_task_id()
        backtest_id = _make_backtest_id()
        await _insert_task(session, task_id)
        await _insert_backtest_result(session, task_id, backtest_id)

        # 写入子表数据
        await repo.batch_create_portfolio_snapshots(backtest_id, [
            {"date": datetime(2024, 1, 2, tzinfo=timezone.utc), "portfolio_value": 100000, "cash": 50000},
        ])
        await repo.batch_create_trade_records(backtest_id, [
            {"trade_id": "T001", "stock_code": "000001.SZ", "action": "BUY",
             "quantity": 100, "price": 10.0, "timestamp": datetime(2024, 1, 5, tzinfo=timezone.utc)},
        ])
        await repo.batch_save_signal_records(backtest_id, [
            {"signal_id": "S001", "stock_code": "000001.SZ", "signal_type": "BUY",
             "timestamp": datetime(2024, 1, 5, tzinfo=timezone.utc), "price": 10.0, "strength": 0.8},
        ])
        await session.commit()

        # 删除
        ok = await repo.delete_task_data(task_id)
        await session.commit()
        assert ok is True

        # 验证子表数据已清空
        assert len(await repo.get_portfolio_snapshots(backtest_id)) == 0
        assert len(await repo.get_trade_records(backtest_id)) == 0
        assert len(await repo.get_signal_records(backtest_id)) == 0
