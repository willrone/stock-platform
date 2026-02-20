"""
回测数据持久化统一服务 — BacktestPersistenceService

职责：
1. 管理 backtest_id 生命周期（生成 → 占位 → 填充 → 清理）
2. 统一所有数据库写入操作（消除分散写入）
3. 统一所有数据库读取操作（前端 API 统一数据源）
4. 保证事务一致性（子表写入在同一事务中）
5. 精简 tasks.result（只存标量摘要，不存时序数据）
"""

import math
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger

from .data_contracts import BacktestSummary
from .signal_writer import StreamSignalWriter


def _safe_float(value, default=0.0) -> float:
    """安全转换为 float，处理 numpy/pandas 类型和 NaN/Inf"""
    if value is None:
        return default
    try:
        if isinstance(value, (np.integer, np.floating)):
            value = value.item()
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except (TypeError, ValueError):
        return default


def _safe_int(value, default=0) -> int:
    """安全转换为 int"""
    if value is None:
        return default
    try:
        if isinstance(value, (np.integer, np.floating)):
            value = value.item()
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_python_type(value):
    """将 numpy/pandas 类型递归转换为 Python 原生类型"""
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    elif isinstance(value, np.ndarray):
        return value.tolist()
    elif hasattr(value, "to_pydatetime"):
        return value.to_pydatetime().isoformat()
    elif isinstance(value, datetime):
        return value.isoformat()
    elif isinstance(value, dict):
        return {_to_python_type(k): _to_python_type(v) for k, v in value.items()}
    elif isinstance(value, (list, tuple)):
        return [_to_python_type(v) for v in value]
    return value


def _ensure_datetime(value) -> Optional[datetime]:
    """确保值为 datetime 类型"""
    if value is None:
        return None
    try:
        import pandas as pd
        if isinstance(value, pd.Timestamp):
            return value.to_pydatetime()
    except ImportError:
        pass
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            try:
                return datetime.fromisoformat(f"{value}T00:00:00")
            except ValueError:
                return None
    return None


class BacktestPersistenceService:
    """
    回测数据持久化统一服务

    写入接口使用 psycopg2 直连（子进程环境）或 SQLAlchemy async session。
    读取接口使用 BacktestDetailedRepository 作为底层 DAO。
    """

    def __init__(self, db_url: Optional[str] = None):
        """
        初始化持久化服务

        Args:
            db_url: 同步数据库连接 URL（postgresql://...），
                    不传则从 settings 获取
        """
        if db_url is None:
            from app.core.config import settings
            self._db_url = settings.database_url_sync
        else:
            self._db_url = db_url

    # ==================== 生命周期管理（写入） ====================

    def create_backtest_session(
        self,
        task_id: str,
        strategy_name: str = "unknown",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> str:
        """
        创建回测会话，生成唯一 backtest_id 并写入占位行。

        调用时机：BacktestExecutor.run_backtest() 开始时
        替代：backtest_executor._create_placeholder_backtest_result()

        使用 psycopg2 直连（子进程环境安全）。

        Returns:
            backtest_id (str)
        """
        import psycopg2

        backtest_id = str(uuid.uuid4())

        sql = """
            INSERT INTO backtest_results
                (task_id, backtest_id, strategy_name, start_date, end_date,
                 initial_cash, final_value, total_return, annualized_return,
                 volatility, sharpe_ratio, max_drawdown, win_rate,
                 profit_factor, total_trades)
            VALUES (%s, %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s)
            ON CONFLICT (backtest_id) DO NOTHING
        """
        try:
            conn = psycopg2.connect(self._db_url)
            try:
                cur = conn.cursor()
                cur.execute(sql, (
                    task_id, backtest_id, strategy_name,
                    (start_date.isoformat() if start_date else datetime.now(timezone.utc).isoformat()),
                    (end_date.isoformat() if end_date else datetime.now(timezone.utc).isoformat()),
                    0, 0, 0, 0,   # initial_cash, final_value, total_return, annualized_return
                    0, 0, 0, 0,   # volatility, sharpe_ratio, max_drawdown, win_rate
                    0, 0,         # profit_factor, total_trades
                ))
                conn.commit()
                cur.close()
                logger.info(f"占位 backtest_results 行已创建: backtest_id={backtest_id}")
            finally:
                conn.close()
        except Exception as e:
            logger.warning(f"创建占位 backtest_results 行失败（信号写入可能受影响）: {e}")

        return backtest_id

    def create_signal_writer(self, backtest_id: str) -> StreamSignalWriter:
        """创建流式信号写入器，供回测循环使用"""
        return StreamSignalWriter(
            backtest_id=backtest_id,
            db_url=self._db_url,
        )

    async def save_backtest_results(
        self,
        task_id: str,
        backtest_id: str,
        backtest_report: dict,
    ) -> bool:
        """
        回测完成后，一次性保存所有结构化数据。

        替代：dependencies.py 中 run_backtest_and_save() 的 ~200 行逻辑

        在同一个事务中完成：
        1. UPDATE backtest_results 占位行（填充完整指标+分析数据）
        2. INSERT portfolio_snapshots（含真实 drawdown 和 daily_return）
        3. INSERT trade_records
        4. UPDATE tasks.result = 精简摘要（BacktestSummary）
        5. UPDATE tasks.status = COMPLETED, progress = 100

        单独事务（允许失败）：
        6. INSERT/UPDATE backtest_statistics（预计算统计）
        """
        from sqlalchemy.ext.asyncio import AsyncSession as _AsyncSession
        from sqlalchemy.ext.asyncio import async_sessionmaker as _async_sessionmaker
        from sqlalchemy.ext.asyncio import create_async_engine as _create_async_engine

        from app.core.config import settings as _settings
        from app.core.database import _pg_json_serializer, retry_db_operation
        from app.repositories.backtest_detailed_repository import BacktestDetailedRepository
        from app.services.backtest.models import EnhancedPositionAnalysis
        from app.services.backtest.statistics import StatisticsCalculator
        from app.services.backtest.utils import BacktestDataAdapter

        # 子进程专属 async_engine（不复用主进程连接池）
        subprocess_engine = _create_async_engine(
            _settings.DATABASE_URL,
            echo=False,
            future=True,
            pool_size=5,
            max_overflow=3,
            pool_timeout=30,
            pool_pre_ping=True,
            pool_recycle=3600,
            json_serializer=_pg_json_serializer,
        )

        try:
            # 数据适配：将原始报告转换为增强格式
            adapter = BacktestDataAdapter()
            enhanced_result = await adapter.adapt_backtest_result(backtest_report)

            # 准备扩展风险指标
            extended_metrics = {}
            if enhanced_result.extended_risk_metrics:
                erm = enhanced_result.extended_risk_metrics
                extended_metrics = {
                    "sortino_ratio": _to_python_type(erm.sortino_ratio),
                    "calmar_ratio": _to_python_type(erm.calmar_ratio),
                    "max_drawdown_duration": _to_python_type(erm.max_drawdown_duration),
                    "var_95": _to_python_type(erm.var_95),
                    "downside_deviation": _to_python_type(erm.downside_deviation),
                    "var_99": _to_python_type(getattr(erm, "var_99", 0)),
                    "cvar_95": _to_python_type(getattr(erm, "cvar_95", 0)),
                    "cvar_99": _to_python_type(getattr(erm, "cvar_99", 0)),
                }

            # 准备持仓分析数据
            position_analysis_data = None
            if enhanced_result.position_analysis:
                if isinstance(enhanced_result.position_analysis, EnhancedPositionAnalysis):
                    position_analysis_data = enhanced_result.position_analysis.to_dict()
                elif isinstance(enhanced_result.position_analysis, list):
                    position_analysis_data = [pa.to_dict() for pa in enhanced_result.position_analysis]
                else:
                    position_analysis_data = enhanced_result.position_analysis

            # 准备分析数据 JSONB
            analysis_data = {
                "drawdown_analysis": _to_python_type(
                    enhanced_result.drawdown_analysis.to_dict()
                ) if enhanced_result.drawdown_analysis else {},
                "monthly_returns": _to_python_type(
                    [mr.to_dict() for mr in enhanced_result.monthly_returns]
                ) if enhanced_result.monthly_returns else [],
                "position_analysis": _to_python_type(position_analysis_data) if position_analysis_data else None,
                "benchmark_comparison": {
                    **(_to_python_type(enhanced_result.benchmark_data) if enhanced_result.benchmark_data else {}),
                    "var_99": extended_metrics.get("var_99", 0),
                    "cvar_95": extended_metrics.get("cvar_95", 0),
                    "cvar_99": extended_metrics.get("cvar_99", 0),
                },
                "rolling_metrics": _to_python_type(
                    enhanced_result.rolling_metrics
                ) if enhanced_result.rolling_metrics else {},
            }

            _SessionLocal = _async_sessionmaker(
                subprocess_engine,
                class_=_AsyncSession,
                expire_on_commit=False,
            )

            async with _SessionLocal() as session:
                try:
                    async def _save_all():
                        repository = BacktestDetailedRepository(session)

                        # ── 幂等清理：删除子表旧数据（信号不删，已在循环中写入） ──
                        from sqlalchemy import delete as _sa_delete
                        from sqlalchemy.future import select as _sa_select

                        from app.models.backtest_detailed_models import (
                            BacktestStatistics as _BacktestStatistics,
                            PortfolioSnapshot as _PortfolioSnapshot,
                            TradeRecord as _TradeRecord,
                        )
                        from app.models.task_models import BacktestResult as _BacktestResult

                        _bt_stmt = _sa_select(_BacktestResult.backtest_id).where(
                            _BacktestResult.task_id == task_id
                        )
                        _bt_result = await session.execute(_bt_stmt)
                        _existing_ids = [row[0] for row in _bt_result.fetchall()]

                        if _existing_ids:
                            for _bt_id in _existing_ids:
                                for _model in (_PortfolioSnapshot, _TradeRecord, _BacktestStatistics):
                                    await session.execute(
                                        _sa_delete(_model).where(_model.backtest_id == _bt_id)
                                    )

                        # ── 1. UPDATE backtest_results 占位行 ──
                        await repository.create_detailed_result(
                            task_id=task_id,
                            backtest_id=backtest_id,
                            extended_metrics=extended_metrics,
                            analysis_data=analysis_data,
                            backtest_report=backtest_report,
                        )

                        # ── 2. INSERT portfolio_snapshots（含真实 drawdown 和 daily_return） ──
                        portfolio_history = enhanced_result.portfolio_history or []
                        if portfolio_history:
                            snapshots_data = self._build_snapshots_data(
                                portfolio_history,
                                backtest_report.get("initial_cash", 100000.0),
                            )
                            if snapshots_data:
                                await repository.batch_create_portfolio_snapshots(
                                    backtest_id=backtest_id,
                                    snapshots_data=snapshots_data,
                                )
                                logger.info(f"保存 {len(snapshots_data)} 个组合快照: task_id={task_id}")

                        # ── 3. INSERT trade_records ──
                        trade_history = enhanced_result.trade_history or []
                        if trade_history:
                            trades_data = self._build_trades_data(trade_history)
                            if trades_data:
                                await repository.batch_create_trade_records(
                                    backtest_id=backtest_id,
                                    trades_data=trades_data,
                                )
                                logger.info(f"保存 {len(trades_data)} 条交易记录: task_id={task_id}")

                        # ── 4. UPDATE tasks.result = 精简摘要 + status = COMPLETED ──
                        summary = self._build_summary(backtest_id, backtest_report, extended_metrics)
                        from sqlalchemy import update as _sa_update
                        from app.models.task_models import Task as _Task

                        await session.execute(
                            _sa_update(_Task).where(_Task.task_id == task_id).values(
                                result=summary,
                                status="completed",
                                progress=100.0,
                                completed_at=datetime.now(timezone.utc),
                            )
                        )

                        # 提交主事务
                        await session.commit()
                        logger.info(f"回测详细数据保存成功: task_id={task_id}")

                        # ── 5. 统计预计算（独立事务，允许失败） ──
                        try:
                            calculator = StatisticsCalculator(session)
                            await calculator.calculate_all_statistics(task_id, backtest_id)
                            await session.flush()
                            await session.commit()
                            logger.info(f"统计信息计算成功: task_id={task_id}")
                        except Exception as stats_err:
                            await session.rollback()
                            logger.warning(f"统计计算失败（不影响主流程）: {stats_err}")

                    await retry_db_operation(
                        _save_all,
                        max_retries=3,
                        retry_delay=0.2,
                        operation_name=f"保存回测详细数据 (task_id={task_id})",
                    )
                    return True

                except Exception as e:
                    await session.rollback()
                    import traceback
                    logger.error(f"保存回测详细数据失败: {task_id}, 错误: {e}\n{traceback.format_exc()}")
                    return False

        except Exception as e:
            import traceback
            logger.error(f"save_backtest_results 外层异常: {task_id}, 错误: {e}\n{traceback.format_exc()}")
            return False
        finally:
            try:
                await subprocess_engine.dispose()
            except Exception:
                pass

    async def save_task_failure(self, task_id: str, error_message: str) -> bool:
        """回测失败时更新任务状态"""
        from sqlalchemy.ext.asyncio import AsyncSession as _AsyncSession
        from sqlalchemy.ext.asyncio import async_sessionmaker as _async_sessionmaker
        from sqlalchemy.ext.asyncio import create_async_engine as _create_async_engine

        from app.core.config import settings as _settings
        from app.core.database import _pg_json_serializer

        subprocess_engine = _create_async_engine(
            _settings.DATABASE_URL,
            echo=False, future=True, pool_size=2, max_overflow=1,
            pool_pre_ping=True, json_serializer=_pg_json_serializer,
        )
        try:
            _SessionLocal = _async_sessionmaker(
                subprocess_engine, class_=_AsyncSession, expire_on_commit=False,
            )
            async with _SessionLocal() as session:
                try:
                    from sqlalchemy import update as _sa_update
                    from app.models.task_models import Task as _Task

                    await session.execute(
                        _sa_update(_Task).where(_Task.task_id == task_id).values(
                            status="failed",
                            error_message=error_message,
                        )
                    )
                    await session.commit()
                    return True
                except Exception as e:
                    await session.rollback()
                    logger.error(f"save_task_failure 失败: {e}")
                    return False
        finally:
            try:
                await subprocess_engine.dispose()
            except Exception:
                pass

    # ==================== 读取接口 ====================

    async def get_backtest_summary(self, task_id: str, session) -> Optional[dict]:
        """获取回测概览数据"""
        from app.repositories.backtest_detailed_repository import BacktestDetailedRepository

        repo = BacktestDetailedRepository(session)
        result = await repo.get_detailed_result_by_task_id(task_id)
        if not result:
            return None
        d = result.to_dict()
        # 返回精简摘要
        return {
            "backtest_id": d.get("backtest_id"),
            "strategy_name": d.get("strategy_name"),
            "period": d.get("period"),
            "portfolio": d.get("portfolio"),
            "risk_metrics": d.get("risk_metrics"),
            "trading_stats": d.get("trading_stats"),
        }

    async def get_backtest_detail(self, task_id: str, session) -> Optional[dict]:
        """获取回测详细结果（完整 to_dict）"""
        from app.repositories.backtest_detailed_repository import BacktestDetailedRepository

        repo = BacktestDetailedRepository(session)
        result = await repo.get_detailed_result_by_task_id(task_id)
        if not result:
            return None
        return result.to_dict()

    async def get_portfolio_snapshots(
        self,
        task_id: str,
        session,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[dict]:
        """获取组合快照"""
        from app.repositories.backtest_detailed_repository import BacktestDetailedRepository

        repo = BacktestDetailedRepository(session)
        backtest_id = await repo.get_backtest_id_by_task_id(task_id)
        if not backtest_id:
            return []
        snapshots = await repo.get_portfolio_snapshots(
            backtest_id=backtest_id,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
        )
        return [s.to_dict() for s in snapshots]

    async def get_trade_records(
        self,
        task_id: str,
        session,
        stock_code: Optional[str] = None,
        action: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        offset: int = 0,
        limit: int = 50,
        order_by: str = "timestamp",
        order_desc: bool = True,
    ) -> dict:
        """获取交易记录（分页）"""
        from app.repositories.backtest_detailed_repository import BacktestDetailedRepository

        repo = BacktestDetailedRepository(session)
        backtest_id = await repo.get_backtest_id_by_task_id(task_id)
        if not backtest_id:
            return {"items": [], "total": 0}

        trades = await repo.get_trade_records(
            backtest_id=backtest_id,
            stock_code=stock_code,
            action=action,
            start_date=start_date,
            end_date=end_date,
            offset=offset,
            limit=limit,
            order_by=order_by,
            order_desc=order_desc,
        )
        total = await repo.get_trade_records_count(
            backtest_id=backtest_id,
            stock_code=stock_code,
            action=action,
            start_date=start_date,
            end_date=end_date,
        )
        return {"items": [t.to_dict() for t in trades], "total": total}

    async def get_signal_records(
        self,
        task_id: str,
        session,
        stock_code: Optional[str] = None,
        signal_type: Optional[str] = None,
        executed: Optional[bool] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        offset: int = 0,
        limit: int = 50,
        order_by: str = "timestamp",
        order_desc: bool = True,
    ) -> dict:
        """获取信号记录（分页）"""
        from app.repositories.backtest_detailed_repository import BacktestDetailedRepository

        repo = BacktestDetailedRepository(session)
        backtest_id = await repo.get_backtest_id_by_task_id(task_id)
        if not backtest_id:
            return {"items": [], "total": 0}

        signals = await repo.get_signal_records(
            backtest_id=backtest_id,
            stock_code=stock_code,
            signal_type=signal_type,
            start_date=start_date,
            end_date=end_date,
            executed=executed,
            offset=offset,
            limit=limit,
            order_by=order_by,
            order_desc=order_desc,
        )
        total = await repo.get_signal_records_count(
            backtest_id=backtest_id,
            stock_code=stock_code,
            signal_type=signal_type,
            start_date=start_date,
            end_date=end_date,
            executed=executed,
        )

        # 安全转换
        items = []
        for sig in signals:
            try:
                d = sig.to_dict()
                if "execution_reason" not in d:
                    d["execution_reason"] = None
                items.append(d)
            except Exception:
                pass
        return {"items": items, "total": total}

    async def get_statistics(self, task_id: str, session) -> dict:
        """获取统计信息"""
        from app.repositories.backtest_detailed_repository import BacktestDetailedRepository

        repo = BacktestDetailedRepository(session)
        backtest_id = await repo.get_backtest_id_by_task_id(task_id)
        if not backtest_id:
            return {}

        trade_stats = await repo.get_trade_statistics(backtest_id)
        signal_stats = await repo.get_signal_statistics(backtest_id)
        return {"trade_statistics": trade_stats, "signal_statistics": signal_stats}

    async def get_benchmark_data(
        self, task_id: str, session, benchmark_symbol: str = "000300.SH"
    ) -> Optional[dict]:
        """获取基准对比数据"""
        from app.repositories.backtest_detailed_repository import BacktestDetailedRepository

        repo = BacktestDetailedRepository(session)
        backtest_id = await repo.get_backtest_id_by_task_id(task_id)
        if not backtest_id:
            return None
        benchmark = await repo.get_benchmark_data(backtest_id, benchmark_symbol)
        if not benchmark:
            return None
        return benchmark.to_dict()

    async def delete_backtest_data(self, task_id: str, session) -> bool:
        """删除回测相关所有数据"""
        from app.repositories.backtest_detailed_repository import BacktestDetailedRepository

        repo = BacktestDetailedRepository(session)
        return await repo.delete_task_data(task_id)

    # ==================== 内部辅助方法 ====================

    def _build_snapshots_data(
        self,
        portfolio_history: List[dict],
        initial_cash: float,
    ) -> List[dict]:
        """构建快照数据，计算真实 drawdown 和 daily_return"""
        snapshots = []
        peak = initial_cash
        prev_value = initial_cash

        for snapshot in portfolio_history:
            date_value = snapshot.get("date")
            date_value = _ensure_datetime(date_value)
            if date_value is None:
                continue

            pv = _safe_float(snapshot.get("portfolio_value", 0))
            cash = _safe_float(snapshot.get("cash", 0))

            # 真实 drawdown 计算
            peak = max(peak, pv)
            drawdown = (peak - pv) / peak if peak > 0 else 0.0

            # daily_return 计算
            daily_return = (pv - prev_value) / prev_value if prev_value > 0 else 0.0
            prev_value = pv

            snapshots.append({
                "date": date_value,
                "portfolio_value": _to_python_type(pv),
                "cash": _to_python_type(cash),
                "positions_count": _safe_int(snapshot.get("positions_count", 0)),
                "total_return": _to_python_type(_safe_float(snapshot.get("total_return", 0))),
                "drawdown": _to_python_type(drawdown),
                "positions": _to_python_type(snapshot.get("positions", {})),
            })

        return snapshots

    def _build_trades_data(self, trade_history: List[dict]) -> List[dict]:
        """构建交易记录数据"""
        trades = []
        for trade in trade_history:
            ts = _ensure_datetime(trade.get("timestamp"))
            if ts is None:
                continue
            trades.append({
                "trade_id": trade.get("trade_id", ""),
                "stock_code": trade.get("stock_code", ""),
                "stock_name": trade.get("stock_code", ""),
                "action": trade.get("action", ""),
                "quantity": _safe_int(trade.get("quantity", 0)),
                "price": _to_python_type(_safe_float(trade.get("price", 0))),
                "timestamp": ts,
                "commission": _to_python_type(_safe_float(trade.get("commission", 0))),
                "pnl": _to_python_type(_safe_float(trade.get("pnl", 0))),
                "holding_days": _to_python_type(_safe_int(trade.get("holding_days", 0))),
                "technical_indicators": {},
            })
        return trades

    def _build_summary(
        self,
        backtest_id: str,
        report: dict,
        extended_metrics: dict,
    ) -> dict:
        """构建精简摘要（存入 tasks.result）"""

        def _safe_cost_dict(d):
            """安全转换成本分析 dict 中的数值"""
            if not isinstance(d, dict):
                return None
            return {k: _safe_float(v) for k, v in d.items()}

        summary = BacktestSummary(
            backtest_id=backtest_id,
            strategy_name=report.get("strategy_name", ""),
            stock_count=len(report.get("stock_codes", [])),
            start_date=str(report.get("start_date", "")),
            end_date=str(report.get("end_date", "")),
            initial_cash=_safe_float(report.get("initial_cash", 100000)),
            final_value=_safe_float(report.get("final_value", 0)),
            total_return=_safe_float(report.get("total_return", 0)),
            annualized_return=_safe_float(report.get("annualized_return", 0)),
            volatility=_safe_float(report.get("volatility", 0)),
            sharpe_ratio=_safe_float(report.get("sharpe_ratio", 0)),
            max_drawdown=_safe_float(report.get("max_drawdown", 0)),
            win_rate=_safe_float(report.get("win_rate", 0)),
            profit_factor=_safe_float(report.get("profit_factor", 0)),
            total_trades=_safe_int(report.get("total_trades", 0)),
            total_signals=_safe_int(report.get("total_signals", 0)),
            sortino_ratio=_safe_float(extended_metrics.get("sortino_ratio", 0)),
            calmar_ratio=_safe_float(extended_metrics.get("calmar_ratio", 0)),
            cost_statistics=_safe_cost_dict(report.get("cost_statistics")),
            excess_return_with_cost=_safe_cost_dict(report.get("excess_return_with_cost")),
            excess_return_without_cost=_safe_cost_dict(report.get("excess_return_without_cost")),
        )
        return summary.model_dump()
