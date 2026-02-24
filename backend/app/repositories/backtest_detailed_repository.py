"""
回测详细数据仓库
用于管理回测详细结果、图表缓存、组合快照等数据的CRUD操作
"""

import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from loguru import logger
from sqlalchemy import and_, asc, delete, desc, func, or_, select, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.database import retry_db_operation
from app.models.backtest_detailed_models import (
    BacktestBenchmark,
    BacktestChartCache,
    BacktestDetailedResult,
    BacktestStatistics,
    PortfolioSnapshot,
    SignalRecord,
    TradeRecord,
)


class BacktestDetailedRepository:
    """回测详细数据仓库"""

    def __init__(self, session: AsyncSession):
        self.session = session
        self.logger = logger.bind(repository="backtest_detailed")

    def _ensure_datetime(self, value: Optional[datetime]) -> Optional[datetime]:
        """Ensure datetime input for SQLite DateTime columns."""
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                # Fallback for date-only values.
                return datetime.fromisoformat(f"{value}T00:00:00")
        return value

    # ==================== BacktestDetailedResult 相关操作 ====================

    async def create_detailed_result(
        self,
        task_id: str,
        backtest_id: str,
        extended_metrics: Dict[str, float],
        analysis_data: Dict[str, Any],
    ) -> Optional[BacktestDetailedResult]:
        """创建回测详细结果记录"""
        try:
            detailed_result = BacktestDetailedResult(
                task_id=task_id,
                backtest_id=backtest_id,
                sortino_ratio=extended_metrics.get("sortino_ratio", 0.0),
                calmar_ratio=extended_metrics.get("calmar_ratio", 0.0),
                max_drawdown_duration=extended_metrics.get("max_drawdown_duration", 0),
                var_95=extended_metrics.get("var_95", 0.0),
                downside_deviation=extended_metrics.get("downside_deviation", 0.0),
                drawdown_analysis=analysis_data.get("drawdown_analysis"),
                monthly_returns=analysis_data.get("monthly_returns"),
                position_analysis=analysis_data.get("position_analysis"),
                benchmark_comparison=analysis_data.get("benchmark_comparison"),
                rolling_metrics=analysis_data.get("rolling_metrics"),
            )

            self.session.add(detailed_result)
            await self.session.flush()

            self.logger.info(f"创建回测详细结果: task_id={task_id}, backtest_id={backtest_id}")
            return detailed_result

        except Exception as e:
            self.logger.error("创建回测详细结果失败: {}", e, exc_info=True)
            return None

    async def get_detailed_result_by_task_id(
        self, task_id: str
    ) -> Optional[BacktestDetailedResult]:
        """根据任务ID获取回测详细结果"""
        try:

            async def _get_result():
                stmt = select(BacktestDetailedResult).where(
                    BacktestDetailedResult.task_id == task_id
                )
                result = await self.session.execute(stmt)
                return result.scalar_one_or_none()

            return await retry_db_operation(
                _get_result,
                max_retries=3,
                retry_delay=0.1,
                operation_name=f"获取回测详细结果 (task_id={task_id})",
            )

        except Exception as e:
            self.logger.error("获取回测详细结果失败: {}", e, exc_info=True)
            return None

    async def update_detailed_result(
        self,
        task_id: str,
        extended_metrics: Optional[Dict[str, float]] = None,
        analysis_data: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """更新回测详细结果"""
        try:
            stmt = select(BacktestDetailedResult).where(
                BacktestDetailedResult.task_id == task_id
            )
            result = await self.session.execute(stmt)
            detailed_result = result.scalar_one_or_none()

            if not detailed_result:
                self.logger.warning(f"未找到回测详细结果: task_id={task_id}")
                return False

            # 更新扩展指标
            if extended_metrics:
                detailed_result.sortino_ratio = extended_metrics.get(
                    "sortino_ratio", detailed_result.sortino_ratio
                )
                detailed_result.calmar_ratio = extended_metrics.get(
                    "calmar_ratio", detailed_result.calmar_ratio
                )
                detailed_result.max_drawdown_duration = extended_metrics.get(
                    "max_drawdown_duration", detailed_result.max_drawdown_duration
                )
                detailed_result.var_95 = extended_metrics.get(
                    "var_95", detailed_result.var_95
                )
                detailed_result.downside_deviation = extended_metrics.get(
                    "downside_deviation", detailed_result.downside_deviation
                )

            # 更新分析数据
            if analysis_data:
                if "drawdown_analysis" in analysis_data:
                    detailed_result.drawdown_analysis = analysis_data[
                        "drawdown_analysis"
                    ]
                if "monthly_returns" in analysis_data:
                    detailed_result.monthly_returns = analysis_data["monthly_returns"]
                if "position_analysis" in analysis_data:
                    detailed_result.position_analysis = analysis_data[
                        "position_analysis"
                    ]
                if "benchmark_comparison" in analysis_data:
                    detailed_result.benchmark_comparison = analysis_data[
                        "benchmark_comparison"
                    ]
                if "rolling_metrics" in analysis_data:
                    detailed_result.rolling_metrics = analysis_data["rolling_metrics"]

            detailed_result.updated_at = datetime.utcnow()
            await self.session.flush()

            self.logger.info(f"更新回测详细结果: task_id={task_id}")
            return True

        except Exception as e:
            self.logger.error("更新回测详细结果失败: {}", e, exc_info=True)
            return False

    # ==================== PortfolioSnapshot 相关操作 ====================

    async def batch_create_portfolio_snapshots(
        self, task_id: str, backtest_id: str, snapshots_data: List[Dict[str, Any]]
    ) -> bool:
        """批量创建组合快照"""
        try:
            snapshots = []
            for snapshot_data in snapshots_data:
                snapshot = PortfolioSnapshot(
                    task_id=task_id,
                    backtest_id=backtest_id,
                    snapshot_date=self._ensure_datetime(snapshot_data["date"]),
                    portfolio_value=snapshot_data["portfolio_value"],
                    cash=snapshot_data["cash"],
                    positions_count=snapshot_data.get("positions_count", 0),
                    total_return=snapshot_data.get("total_return", 0.0),
                    drawdown=snapshot_data.get("drawdown", 0.0),
                    positions=snapshot_data.get("positions"),
                )
                snapshots.append(snapshot)

            self.session.add_all(snapshots)
            await self.session.flush()

            self.logger.info(f"批量创建组合快照: task_id={task_id}, count={len(snapshots)}")
            return True

        except Exception as e:
            self.logger.error("批量创建组合快照失败: {}", e, exc_info=True)
            return False

    async def get_portfolio_snapshots(
        self,
        task_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[PortfolioSnapshot]:
        """获取组合快照列表"""
        try:
            stmt = select(PortfolioSnapshot).where(PortfolioSnapshot.task_id == task_id)

            if start_date:
                stmt = stmt.where(PortfolioSnapshot.snapshot_date >= start_date)
            if end_date:
                stmt = stmt.where(PortfolioSnapshot.snapshot_date <= end_date)

            stmt = stmt.order_by(PortfolioSnapshot.snapshot_date)

            if limit:
                stmt = stmt.limit(limit)

            result = await self.session.execute(stmt)
            return result.scalars().all()

        except Exception as e:
            self.logger.error("获取组合快照失败: {}", e, exc_info=True)
            return []

    # ==================== TradeRecord 相关操作 ====================

    async def batch_create_trade_records(
        self, task_id: str, backtest_id: str, trades_data: List[Dict[str, Any]]
    ) -> bool:
        """批量创建交易记录"""
        try:
            trades = []
            for trade_data in trades_data:
                trade = TradeRecord(
                    task_id=task_id,
                    backtest_id=backtest_id,
                    trade_id=trade_data["trade_id"],
                    stock_code=trade_data["stock_code"],
                    stock_name=trade_data.get("stock_name"),
                    action=trade_data["action"],
                    quantity=trade_data["quantity"],
                    price=trade_data["price"],
                    timestamp=self._ensure_datetime(trade_data["timestamp"]),
                    commission=trade_data.get("commission", 0.0),
                    pnl=trade_data.get("pnl"),
                    holding_days=trade_data.get("holding_days"),
                    technical_indicators=trade_data.get("technical_indicators"),
                )
                trades.append(trade)

            self.session.add_all(trades)
            await self.session.flush()

            self.logger.info(f"批量创建交易记录: task_id={task_id}, count={len(trades)}")
            return True

        except Exception as e:
            self.logger.error("批量创建交易记录失败: {}", e, exc_info=True)
            return False

    async def get_trade_records(
        self,
        task_id: str,
        stock_code: Optional[str] = None,
        action: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        offset: int = 0,
        limit: int = 50,
        order_by: str = "timestamp",
        order_desc: bool = True,
    ) -> List[TradeRecord]:
        """获取交易记录列表"""
        try:
            stmt = select(TradeRecord).where(TradeRecord.task_id == task_id)

            if stock_code:
                stmt = stmt.where(TradeRecord.stock_code == stock_code)
            if action:
                stmt = stmt.where(TradeRecord.action == action)
            if start_date:
                stmt = stmt.where(TradeRecord.timestamp >= start_date)
            if end_date:
                stmt = stmt.where(TradeRecord.timestamp <= end_date)

            # 排序
            if order_by == "timestamp":
                order_col = TradeRecord.timestamp
            elif order_by == "pnl":
                order_col = TradeRecord.pnl
            elif order_by == "price":
                order_col = TradeRecord.price
            else:
                order_col = TradeRecord.timestamp

            if order_desc:
                stmt = stmt.order_by(desc(order_col))
            else:
                stmt = stmt.order_by(asc(order_col))

            stmt = stmt.offset(offset).limit(limit)

            result = await self.session.execute(stmt)
            return result.scalars().all()

        except Exception as e:
            self.logger.error("获取交易记录失败: {}", e, exc_info=True)
            return []

    async def get_trade_records_count(
        self,
        task_id: str,
        stock_code: Optional[str] = None,
        action: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> int:
        """获取交易记录总数"""
        try:
            stmt = select(func.count(TradeRecord.id)).where(
                TradeRecord.task_id == task_id
            )

            if stock_code:
                stmt = stmt.where(TradeRecord.stock_code == stock_code)
            if action:
                stmt = stmt.where(TradeRecord.action == action)
            if start_date:
                stmt = stmt.where(TradeRecord.timestamp >= start_date)
            if end_date:
                stmt = stmt.where(TradeRecord.timestamp <= end_date)

            result = await self.session.execute(stmt)
            return result.scalar() or 0

        except Exception as e:
            self.logger.error("获取交易记录总数失败: {}", e, exc_info=True)
            return 0

    async def get_trade_statistics(self, task_id: str) -> Dict[str, Any]:
        """获取交易统计信息（优化：优先从统计表读取，不存在则实时计算）"""
        try:
            # 优先从统计表读取
            try:
                stats_stmt = select(BacktestStatistics).where(
                    BacktestStatistics.task_id == task_id
                )
                stats_result = await self.session.execute(stats_stmt)
                stats = stats_result.scalar_one_or_none()

                if stats:
                    # 从统计表返回
                    return {
                        "total_trades": stats.total_trades,
                        "buy_trades": stats.buy_trades,
                        "sell_trades": stats.sell_trades,
                        "winning_trades": stats.winning_trades,
                        "losing_trades": stats.losing_trades,
                        "win_rate": stats.win_rate,
                        "avg_profit": stats.avg_profit,
                        "avg_loss": stats.avg_loss,
                        "profit_factor": stats.profit_factor,
                        "total_commission": stats.total_commission,
                        "total_pnl": stats.total_pnl,
                        "avg_holding_days": stats.avg_holding_days,
                    }
            except Exception as stats_error:
                # 如果统计表不存在或其他错误，回退到实时计算
                error_str = str(stats_error).lower()
                if "no such table" in error_str or (
                    "table" in error_str and "does not exist" in error_str
                ):
                    self.logger.debug(f"统计表不存在，回退到实时计算: task_id={task_id}")
                else:
                    self.logger.warning(
                        f"查询统计表失败，回退到实时计算: task_id={task_id}, error={stats_error}"
                    )

            # 向后兼容：如果统计表不存在或没有数据，实时计算
            try:
                total_stmt = select(func.count(TradeRecord.id)).where(
                    TradeRecord.task_id == task_id
                )
                total_result = await self.session.execute(total_stmt)
                total_trades = total_result.scalar() or 0

                buy_stmt = select(func.count(TradeRecord.id)).where(
                    and_(TradeRecord.task_id == task_id, TradeRecord.action == "BUY")
                )
                buy_result = await self.session.execute(buy_stmt)
                buy_trades = buy_result.scalar() or 0

                sell_stmt = select(func.count(TradeRecord.id)).where(
                    and_(TradeRecord.task_id == task_id, TradeRecord.action == "SELL")
                )
                sell_result = await self.session.execute(sell_stmt)
                sell_trades = sell_result.scalar() or 0

                profit_stmt = select(func.count(TradeRecord.id)).where(
                    and_(TradeRecord.task_id == task_id, TradeRecord.pnl > 0)
                )
                profit_result = await self.session.execute(profit_stmt)
                profit_trades = profit_result.scalar() or 0

                loss_stmt = select(func.count(TradeRecord.id)).where(
                    and_(TradeRecord.task_id == task_id, TradeRecord.pnl < 0)
                )
                loss_result = await self.session.execute(loss_stmt)
                loss_trades = loss_result.scalar() or 0

                pnl_stmt = select(func.sum(TradeRecord.pnl)).where(
                    TradeRecord.task_id == task_id
                )
                pnl_result = await self.session.execute(pnl_stmt)
                total_pnl = pnl_result.scalar() or 0.0

                avg_profit_stmt = select(func.avg(TradeRecord.pnl)).where(
                    and_(TradeRecord.task_id == task_id, TradeRecord.pnl > 0)
                )
                avg_profit_result = await self.session.execute(avg_profit_stmt)
                avg_profit = avg_profit_result.scalar() or 0.0

                avg_loss_stmt = select(func.avg(TradeRecord.pnl)).where(
                    and_(TradeRecord.task_id == task_id, TradeRecord.pnl < 0)
                )
                avg_loss_result = await self.session.execute(avg_loss_stmt)
                avg_loss = avg_loss_result.scalar() or 0.0

                commission_stmt = select(func.sum(TradeRecord.commission)).where(
                    TradeRecord.task_id == task_id
                )
                commission_result = await self.session.execute(commission_stmt)
                total_commission = commission_result.scalar() or 0.0

                holding_stmt = select(func.avg(TradeRecord.holding_days)).where(
                    and_(
                        TradeRecord.task_id == task_id,
                        TradeRecord.holding_days.isnot(None),
                    )
                )
                holding_result = await self.session.execute(holding_stmt)
                avg_holding_days = holding_result.scalar() or 0.0

                profit_factor = 0.0
                if avg_loss != 0:
                    profit_factor = abs(avg_profit / avg_loss)

                return {
                    "total_trades": total_trades,
                    "buy_trades": buy_trades,
                    "sell_trades": sell_trades,
                    "winning_trades": profit_trades,
                    "losing_trades": loss_trades,
                    "win_rate": profit_trades / total_trades
                    if total_trades > 0
                    else 0.0,
                    "avg_profit": float(avg_profit),
                    "avg_loss": float(avg_loss),
                    "profit_factor": float(profit_factor),
                    "total_commission": float(total_commission),
                    "total_pnl": float(total_pnl),
                    "avg_holding_days": float(avg_holding_days),
                }
            except Exception as calc_error:
                # 如果交易记录表不存在，返回空统计
                error_str = str(calc_error).lower()
                if "no such table" in error_str or (
                    "table" in error_str and "does not exist" in error_str
                ):
                    self.logger.info(f"交易记录表不存在，返回空统计: task_id={task_id}")
                else:
                    self.logger.warning(
                        f"计算交易统计失败: task_id={task_id}, error={calc_error}"
                    )
                # 返回空统计
                return {
                    "total_trades": 0,
                    "buy_trades": 0,
                    "sell_trades": 0,
                    "winning_trades": 0,
                    "losing_trades": 0,
                    "win_rate": 0.0,
                    "avg_profit": 0.0,
                    "avg_loss": 0.0,
                    "profit_factor": 0.0,
                    "total_commission": 0.0,
                    "total_pnl": 0.0,
                    "avg_holding_days": 0.0,
                }

        except Exception as e:
            import traceback

            error_detail = traceback.format_exc()
            self.logger.error("获取交易统计失败: {}\n{}", e, error_detail, exc_info=True)
            # 返回空统计而不是抛出异常，避免前端报错
            return {
                "total_trades": 0,
                "buy_trades": 0,
                "sell_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "avg_profit": 0.0,
                "avg_loss": 0.0,
                "profit_factor": 0.0,
                "total_commission": 0.0,
                "total_pnl": 0.0,
                "avg_holding_days": 0.0,
            }

    # ==================== SignalRecord 相关操作 ====================

    async def save_signal_record(
        self,
        task_id: str,
        backtest_id: str,
        signal_id: str,
        stock_code: str,
        stock_name: Optional[str],
        signal_type: str,
        timestamp: datetime,
        price: float,
        strength: float,
        reason: Optional[str] = None,
        signal_metadata: Optional[Dict[str, Any]] = None,
        executed: bool = False,
    ) -> Optional[SignalRecord]:
        """保存信号记录"""
        try:
            signal_record = SignalRecord(
                task_id=task_id,
                backtest_id=backtest_id,
                signal_id=signal_id,
                stock_code=stock_code,
                stock_name=stock_name,
                signal_type=signal_type,
                timestamp=self._ensure_datetime(timestamp),
                price=price,
                strength=strength,
                reason=reason,
                signal_metadata=signal_metadata,
                executed=executed,
            )

            self.session.add(signal_record)
            await self.session.flush()

            self.logger.debug(
                f"保存信号记录: task_id={task_id}, signal_id={signal_id}, type={signal_type}"
            )
            return signal_record

        except Exception as e:
            self.logger.error("保存信号记录失败: {}", e, exc_info=True)
            return None

    async def batch_save_signal_records(
        self, task_id: str, backtest_id: str, signals_data: List[Dict[str, Any]]
    ) -> bool:
        """批量保存信号记录（原生SQL批量插入，绕过ORM开销）"""
        if not signals_data:
            return True

        try:
            now = datetime.utcnow().isoformat()
            rows = []
            for sd in signals_data:
                ts = self._ensure_datetime(sd["timestamp"])
                metadata = sd.get("metadata")
                if metadata is not None and not isinstance(metadata, str):
                    metadata = json.dumps(metadata, ensure_ascii=False)
                rows.append({
                    "task_id": task_id,
                    "backtest_id": backtest_id,
                    "signal_id": sd["signal_id"],
                    "stock_code": sd["stock_code"],
                    "stock_name": sd.get("stock_name"),
                    "signal_type": sd["signal_type"],
                    "timestamp": ts.isoformat() if ts else None,
                    "price": sd["price"],
                    "strength": sd.get("strength", 0.0),
                    "reason": sd.get("reason"),
                    "signal_metadata": metadata,
                    "executed": 1 if sd.get("executed", False) else 0,
                    "execution_reason": sd.get("execution_reason"),
                    "created_at": now,
                })

            # SQLite 单条 INSERT 最多 999 个参数，14 列 → 每批最多 71 行，取 60 安全值
            CHUNK = 60
            for start in range(0, len(rows), CHUNK):
                chunk = rows[start:start + CHUNK]
                placeholders = []
                params = {}
                for i, r in enumerate(chunk):
                    ph = ", ".join(f":v{i}_{k}" for k in r)
                    placeholders.append(f"({ph})")
                    for k, v in r.items():
                        params[f"v{i}_{k}"] = v
                values_sql = ", ".join(placeholders)
                sql = text(f"""
                    INSERT INTO signal_records
                        (task_id, backtest_id, signal_id, stock_code, stock_name,
                         signal_type, timestamp, price, strength, reason,
                         signal_metadata, executed, execution_reason, created_at)
                    VALUES {values_sql}
                """)
                await self.session.execute(sql, params)

            await self.session.flush()
            self.logger.info(f"批量保存信号记录(原生SQL): task_id={task_id}, count={len(rows)}")
            return True

        except Exception as e:
            self.logger.error("批量保存信号记录失败: {}", e, exc_info=True)
            return False

    async def get_signal_records(
        self,
        task_id: str,
        stock_code: Optional[str] = None,
        signal_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        executed: Optional[bool] = None,
        offset: int = 0,
        limit: int = 50,
        order_by: str = "timestamp",
        order_desc: bool = True,
    ) -> List[SignalRecord]:
        """获取信号记录列表"""
        try:
            stmt = select(SignalRecord).where(SignalRecord.task_id == task_id)

            if stock_code:
                stmt = stmt.where(SignalRecord.stock_code == stock_code)
            if signal_type:
                stmt = stmt.where(SignalRecord.signal_type == signal_type)
            if start_date:
                stmt = stmt.where(SignalRecord.timestamp >= start_date)
            if end_date:
                stmt = stmt.where(SignalRecord.timestamp <= end_date)
            if executed is not None:
                stmt = stmt.where(SignalRecord.executed == executed)

            # 排序
            if order_by == "timestamp":
                order_col = SignalRecord.timestamp
            elif order_by == "price":
                order_col = SignalRecord.price
            elif order_by == "strength":
                order_col = SignalRecord.strength
            else:
                order_col = SignalRecord.timestamp

            if order_desc:
                stmt = stmt.order_by(desc(order_col))
            else:
                stmt = stmt.order_by(asc(order_col))

            stmt = stmt.offset(offset).limit(limit)

            result = await self.session.execute(stmt)
            return result.scalars().all()

        except Exception as e:
            self.logger.error("获取信号记录失败: {}", e, exc_info=True)
            return []

    async def get_signal_records_count(
        self,
        task_id: str,
        stock_code: Optional[str] = None,
        signal_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        executed: Optional[bool] = None,
    ) -> int:
        """获取信号记录总数"""
        try:

            async def _get_count():
                stmt = select(func.count(SignalRecord.id)).where(
                    SignalRecord.task_id == task_id
                )

                if stock_code:
                    stmt = stmt.where(SignalRecord.stock_code == stock_code)
                if signal_type:
                    stmt = stmt.where(SignalRecord.signal_type == signal_type)
                if start_date:
                    stmt = stmt.where(SignalRecord.timestamp >= start_date)
                if end_date:
                    stmt = stmt.where(SignalRecord.timestamp <= end_date)
                if executed is not None:
                    stmt = stmt.where(SignalRecord.executed == executed)

                result = await self.session.execute(stmt)
                return result.scalar() or 0

            return await retry_db_operation(
                _get_count,
                max_retries=3,
                retry_delay=0.1,
                operation_name=f"获取信号记录总数 (task_id={task_id})",
            )

        except Exception as e:
            self.logger.error("获取信号记录总数失败: {}", e, exc_info=True)
            return 0

    async def get_signal_statistics(self, task_id: str) -> Dict[str, Any]:
        """获取信号统计信息（优化：优先从统计表读取，不存在则实时计算）"""
        try:
            # 优先从统计表读取
            try:
                stats_stmt = select(BacktestStatistics).where(
                    BacktestStatistics.task_id == task_id
                )
                stats_result = await self.session.execute(stats_stmt)
                stats = stats_result.scalar_one_or_none()

                if stats:
                    # 从统计表返回
                    return {
                        "total_signals": stats.total_signals,
                        "buy_signals": stats.buy_signals,
                        "sell_signals": stats.sell_signals,
                        "executed_signals": stats.executed_signals,
                        "unexecuted_signals": stats.unexecuted_signals,
                        "execution_rate": stats.execution_rate,
                        "avg_strength": stats.avg_signal_strength,
                    }
            except Exception as stats_error:
                # 如果统计表不存在或其他错误，回退到实时计算
                error_str = str(stats_error).lower()
                if "no such table" in error_str or (
                    "table" in error_str and "does not exist" in error_str
                ):
                    self.logger.debug(f"统计表不存在，回退到实时计算: task_id={task_id}")
                else:
                    self.logger.warning(
                        f"查询统计表失败，回退到实时计算: task_id={task_id}, error={stats_error}"
                    )

            # 向后兼容：如果统计表不存在或没有数据，实时计算
            try:
                import time

                start_time = time.time()

                base_where = SignalRecord.task_id == task_id

                total_stmt = select(func.count(SignalRecord.id)).where(base_where)
                buy_stmt = select(func.count(SignalRecord.id)).where(
                    and_(base_where, SignalRecord.signal_type == "BUY")
                )
                sell_stmt = select(func.count(SignalRecord.id)).where(
                    and_(base_where, SignalRecord.signal_type == "SELL")
                )
                executed_stmt = select(func.count(SignalRecord.id)).where(
                    and_(base_where, SignalRecord.executed == True)
                )
                avg_strength_stmt = select(func.avg(SignalRecord.strength)).where(
                    base_where
                )

                total_result = await self.session.execute(total_stmt)
                buy_result = await self.session.execute(buy_stmt)
                sell_result = await self.session.execute(sell_stmt)
                executed_result = await self.session.execute(executed_stmt)
                avg_strength_result = await self.session.execute(avg_strength_stmt)

                total_signals = total_result.scalar() or 0
                buy_signals = buy_result.scalar() or 0
                sell_signals = sell_result.scalar() or 0
                executed_signals = executed_result.scalar() or 0
                avg_strength = avg_strength_result.scalar() or 0.0

                unexecuted_signals = total_signals - executed_signals
                execution_rate = (
                    executed_signals / total_signals if total_signals > 0 else 0.0
                )

                elapsed_time = time.time() - start_time
                if elapsed_time > 1.0:
                    self.logger.warning(
                        f"信号统计查询耗时较长: {elapsed_time:.2f}秒, task_id={task_id}, total_signals={total_signals}"
                    )

                return {
                    "total_signals": total_signals,
                    "buy_signals": buy_signals,
                    "sell_signals": sell_signals,
                    "executed_signals": executed_signals,
                    "unexecuted_signals": unexecuted_signals,
                    "execution_rate": execution_rate,
                    "avg_strength": float(avg_strength) if avg_strength else 0.0,
                }
            except Exception as calc_error:
                # 如果信号记录表不存在，返回空统计
                error_str = str(calc_error).lower()
                if "no such table" in error_str or (
                    "table" in error_str and "does not exist" in error_str
                ):
                    self.logger.info(f"信号记录表不存在，返回空统计: task_id={task_id}")
                else:
                    self.logger.warning(
                        f"计算信号统计失败: task_id={task_id}, error={calc_error}"
                    )
                # 返回空统计
                return {
                    "total_signals": 0,
                    "buy_signals": 0,
                    "sell_signals": 0,
                    "executed_signals": 0,
                    "unexecuted_signals": 0,
                    "execution_rate": 0.0,
                    "avg_strength": 0.0,
                }

        except Exception as e:
            import traceback

            error_detail = traceback.format_exc()
            self.logger.error("获取信号统计失败: {}\n{}", e, error_detail, exc_info=True)
            # 返回空统计而不是抛出异常，避免前端报错
            return {
                "total_signals": 0,
                "buy_signals": 0,
                "sell_signals": 0,
                "executed_signals": 0,
                "unexecuted_signals": 0,
                "execution_rate": 0.0,
                "avg_strength": 0.0,
            }

    async def mark_signal_as_executed(
        self, task_id: str, stock_code: str, timestamp: datetime, signal_type: str
    ) -> bool:
        """标记信号为已执行（基于股票代码、时间和类型匹配）"""
        try:
            # 查找匹配的信号（在相同日期和时间窗口内）
            timestamp_start = timestamp.replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            timestamp_end = timestamp_start + timedelta(days=1)

            stmt = (
                select(SignalRecord)
                .where(
                    and_(
                        SignalRecord.task_id == task_id,
                        SignalRecord.stock_code == stock_code,
                        SignalRecord.signal_type == signal_type,
                        SignalRecord.timestamp >= timestamp_start,
                        SignalRecord.timestamp < timestamp_end,
                        SignalRecord.executed == False,
                    )
                )
                .order_by(SignalRecord.timestamp)
            )

            result = await self.session.execute(stmt)
            signals = result.scalars().all()

            if signals:
                # 标记第一个匹配的信号为已执行
                signal = signals[0]
                signal.executed = True
                signal.execution_reason = None  # 已执行时清空未执行原因
                await self.session.flush()
                self.logger.debug(
                    f"标记信号为已执行: task_id={task_id}, signal_id={signal.signal_id}"
                )
                return True

            return False

        except Exception as e:
            self.logger.error("标记信号为已执行失败: {}", e, exc_info=True)
            return False

    async def update_signal_execution_reason(
        self,
        task_id: str,
        stock_code: str,
        timestamp: datetime,
        signal_type: str,
        execution_reason: str,
    ) -> bool:
        """更新信号的未执行原因（基于股票代码、时间和类型匹配）"""
        try:
            # 查找匹配的信号（在相同日期和时间窗口内）
            timestamp_start = timestamp.replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            timestamp_end = timestamp_start + timedelta(days=1)

            stmt = (
                select(SignalRecord)
                .where(
                    and_(
                        SignalRecord.task_id == task_id,
                        SignalRecord.stock_code == stock_code,
                        SignalRecord.signal_type == signal_type,
                        SignalRecord.timestamp >= timestamp_start,
                        SignalRecord.timestamp < timestamp_end,
                        SignalRecord.executed == False,  # 只更新未执行的信号
                    )
                )
                .order_by(SignalRecord.timestamp)
            )

            result = await self.session.execute(stmt)
            signals = result.scalars().all()

            if signals:
                # 更新第一个匹配的信号的未执行原因
                signal = signals[0]
                signal.execution_reason = execution_reason
                await self.session.flush()
                self.logger.debug(
                    f"更新信号未执行原因: task_id={task_id}, signal_id={signal.signal_id}, reason={execution_reason}"
                )
                return True

            return False

        except Exception as e:
            self.logger.error("更新信号未执行原因失败: {}", e, exc_info=True)
            return False

    async def batch_mark_signals_as_executed(
        self, task_id: str, signal_keys: List[tuple]
    ) -> int:
        """
        批量标记信号为已执行（分块 IN 查询，避免参数爆炸）

        Args:
            task_id: 任务ID
            signal_keys: 信号键列表，每个元素为 (stock_code, timestamp, signal_type)

        Returns:
            成功更新的记录数
        """
        if not signal_keys:
            return 0

        try:
            total_updated = 0
            # 每批处理 200 条，避免 SQLite 参数上限
            CHUNK = 200
            for start in range(0, len(signal_keys), CHUNK):
                chunk = signal_keys[start:start + CHUNK]
                conditions = []
                params = {"task_id": task_id}

                for i, (stock_code, timestamp, signal_type) in enumerate(chunk):
                    ts = self._ensure_datetime(timestamp)
                    date_str = ts.strftime("%Y-%m-%d") if ts else ""
                    params[f"sc_{i}"] = stock_code
                    params[f"st_{i}"] = signal_type
                    params[f"ds_{i}"] = date_str
                    conditions.append(
                        f"(stock_code = :sc_{i} AND signal_type = :st_{i} "
                        f"AND date(timestamp) = :ds_{i})"
                    )

                where_clause = " OR ".join(conditions)
                sql = text(f"""
                    UPDATE signal_records
                    SET executed = 1, execution_reason = NULL
                    WHERE task_id = :task_id
                    AND executed = 0
                    AND ({where_clause})
                """)
                result = await self.session.execute(sql, params)
                total_updated += result.rowcount

            await self.session.flush()
            self.logger.info(
                f"批量标记信号为已执行: task_id={task_id}, "
                f"请求数={len(signal_keys)}, 更新数={total_updated}"
            )
            return total_updated

        except Exception as e:
            self.logger.error("批量标记信号为已执行失败: {}", e, exc_info=True)
            return 0

    async def batch_update_signal_execution_reasons(
        self, task_id: str, signal_reasons: List[tuple]
    ) -> int:
        """
        批量更新信号的未执行原因（分块 + 精简 CASE WHEN，避免参数爆炸）

        Args:
            task_id: 任务ID
            signal_reasons: 信号原因列表，每个元素为 (stock_code, timestamp, signal_type, execution_reason)

        Returns:
            成功更新的记录数
        """
        if not signal_reasons:
            return 0

        try:
            total_updated = 0
            # 每批处理 150 条（CASE WHEN 比纯 IN 多参数，取保守值）
            CHUNK = 150
            for start in range(0, len(signal_reasons), CHUNK):
                chunk = signal_reasons[start:start + CHUNK]
                case_when_parts = []
                where_conditions = []
                params = {"task_id": task_id}

                for i, (stock_code, timestamp, signal_type, execution_reason) in enumerate(chunk):
                    ts = self._ensure_datetime(timestamp)
                    date_str = ts.strftime("%Y-%m-%d") if ts else ""
                    params[f"sc_{i}"] = stock_code
                    params[f"st_{i}"] = signal_type
                    params[f"ds_{i}"] = date_str
                    params[f"r_{i}"] = execution_reason

                    condition = (
                        f"(stock_code = :sc_{i} AND signal_type = :st_{i} "
                        f"AND date(timestamp) = :ds_{i})"
                    )
                    where_conditions.append(condition)
                    case_when_parts.append(f"WHEN {condition} THEN :r_{i}")

                case_when_clause = " ".join(case_when_parts)
                where_clause = " OR ".join(where_conditions)
                sql = text(f"""
                    UPDATE signal_records
                    SET execution_reason = CASE {case_when_clause} END
                    WHERE task_id = :task_id
                    AND executed = 0
                    AND ({where_clause})
                """)
                result = await self.session.execute(sql, params)
                total_updated += result.rowcount

            await self.session.flush()
            self.logger.info(
                f"批量更新信号未执行原因: task_id={task_id}, "
                f"请求数={len(signal_reasons)}, 更新数={total_updated}"
            )
            return total_updated

        except Exception as e:
            self.logger.error("批量更新信号未执行原因失败: {}", e, exc_info=True)
            return 0

    # ==================== BacktestBenchmark 相关操作 ====================

    async def create_benchmark_data(
        self,
        task_id: str,
        backtest_id: str,
        benchmark_symbol: str,
        benchmark_name: str,
        benchmark_data: List[Dict[str, Any]],
        comparison_metrics: Dict[str, float],
    ) -> Optional[BacktestBenchmark]:
        """创建基准数据"""
        try:
            benchmark = BacktestBenchmark(
                task_id=task_id,
                backtest_id=backtest_id,
                benchmark_symbol=benchmark_symbol,
                benchmark_name=benchmark_name,
                benchmark_data=benchmark_data,
                correlation=comparison_metrics.get("correlation"),
                beta=comparison_metrics.get("beta"),
                alpha=comparison_metrics.get("alpha"),
                tracking_error=comparison_metrics.get("tracking_error"),
                information_ratio=comparison_metrics.get("information_ratio"),
                excess_return=comparison_metrics.get("excess_return"),
            )

            self.session.add(benchmark)
            await self.session.flush()

            self.logger.info(f"创建基准数据: task_id={task_id}, benchmark={benchmark_symbol}")
            return benchmark

        except Exception as e:
            self.logger.error(f"创建基准数据失败: {e}", exc_info=True)
            return None

    async def get_benchmark_data(
        self, task_id: str, benchmark_symbol: str
    ) -> Optional[BacktestBenchmark]:
        """获取基准数据"""
        try:
            stmt = select(BacktestBenchmark).where(
                and_(
                    BacktestBenchmark.task_id == task_id,
                    BacktestBenchmark.benchmark_symbol == benchmark_symbol,
                )
            )
            result = await self.session.execute(stmt)
            return result.scalar_one_or_none()

        except Exception as e:
            self.logger.error(f"获取基准数据失败: {e}", exc_info=True)
            return None

    # ==================== 数据清理相关操作 ====================

    async def delete_task_data(self, task_id: str) -> bool:
        """删除任务相关的所有详细数据"""
        try:
            # 删除各个表中的数据
            tables_and_models = [
                (BacktestDetailedResult, "回测详细结果"),
                (PortfolioSnapshot, "组合快照"),
                (TradeRecord, "交易记录"),
                (SignalRecord, "信号记录"),
                (BacktestBenchmark, "基准数据"),
            ]

            deleted_counts = {}

            for model_class, table_name in tables_and_models:
                stmt = delete(model_class).where(model_class.task_id == task_id)
                result = await self.session.execute(stmt)
                deleted_count = result.rowcount
                deleted_counts[table_name] = deleted_count

                if deleted_count > 0:
                    self.logger.info(f"删除{table_name}: {deleted_count}条记录")

            await self.session.flush()

            total_deleted = sum(deleted_counts.values())
            self.logger.info(f"删除任务数据完成: task_id={task_id}, 总计删除{total_deleted}条记录")

            return True

        except Exception as e:
            self.logger.error(f"删除任务数据失败: {e}", exc_info=True)
            return False

    async def cleanup_old_data(self, days_to_keep: int = 30) -> Dict[str, int]:
        """清理旧数据"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)

            # 清理各个表中的旧数据
            tables_and_models = [
                (BacktestDetailedResult, "回测详细结果"),
                (BacktestChartCache, "图表缓存"),
                (PortfolioSnapshot, "组合快照"),
                (TradeRecord, "交易记录"),
                (BacktestBenchmark, "基准数据"),
            ]

            cleanup_results = {}

            for model_class, table_name in tables_and_models:
                stmt = delete(model_class).where(model_class.created_at < cutoff_date)
                result = await self.session.execute(stmt)
                deleted_count = result.rowcount
                cleanup_results[table_name] = deleted_count

                if deleted_count > 0:
                    self.logger.info(f"清理{table_name}旧数据: {deleted_count}条记录")

            await self.session.flush()

            total_cleaned = sum(cleanup_results.values())
            self.logger.info(f"数据清理完成: 清理了{total_cleaned}条记录")

            return cleanup_results

        except Exception as e:
            self.logger.error(f"清理旧数据失败: {e}", exc_info=True)
            return {}
