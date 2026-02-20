"""
回测详细数据仓库
用于管理回测详细结果、图表缓存、组合快照等数据的CRUD操作
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from loguru import logger
from sqlalchemy import and_, asc, delete, desc, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

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
        # Handle pandas Timestamp (must check before datetime since Timestamp
        # may or may not be a datetime subclass depending on pandas version)
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
        backtest_report: Optional[Dict[str, Any]] = None,
    ) -> Optional[BacktestDetailedResult]:
        """创建或更新回测详细结果记录（即 BacktestResult）

        如果 backtest_executor 已预创建占位行，则 UPDATE；否则 INSERT。

        Args:
            task_id: 任务ID（外键关联 tasks 表）
            backtest_id: 回测ID（UUID）
            extended_metrics: 扩展风险指标
            analysis_data: 分析数据（drawdown, monthly_returns 等）
            backtest_report: 回测报告数据（包含 strategy_name, total_return 等必填字段）
        """
        try:
            report = backtest_report or {}

            field_values = dict(
                task_id=task_id,
                strategy_name=report.get("strategy_name", "unknown"),
                start_date=self._ensure_datetime(report.get("start_date")) or datetime.now(timezone.utc),
                end_date=self._ensure_datetime(report.get("end_date")) or datetime.now(timezone.utc),
                initial_cash=report.get("initial_cash", 100000.0),
                final_value=report.get("final_value", 0.0),
                total_return=report.get("total_return", 0.0),
                annualized_return=report.get("annualized_return", 0.0),
                volatility=report.get("volatility", 0.0),
                sharpe_ratio=report.get("sharpe_ratio", 0.0),
                max_drawdown=report.get("max_drawdown", 0.0),
                win_rate=report.get("win_rate", 0.0),
                profit_factor=report.get("profit_factor", 0.0),
                total_trades=report.get("total_trades", 0),
                # 扩展风险指标
                sortino_ratio=extended_metrics.get("sortino_ratio", 0.0),
                calmar_ratio=extended_metrics.get("calmar_ratio", 0.0),
                max_drawdown_duration=extended_metrics.get("max_drawdown_duration", 0),
                var_95=extended_metrics.get("var_95", 0.0),
                downside_deviation=extended_metrics.get("downside_deviation", 0.0),
                # JSONB 分析数据
                drawdown_analysis=analysis_data.get("drawdown_analysis"),
                monthly_returns=analysis_data.get("monthly_returns"),
                position_analysis=analysis_data.get("position_analysis"),
                rolling_metrics=analysis_data.get("rolling_metrics"),
                benchmark_comparison=analysis_data.get("benchmark_comparison"),
            )

            # 尝试查找已有的占位行（由 backtest_executor 预创建）
            stmt = select(BacktestDetailedResult).where(
                BacktestDetailedResult.backtest_id == backtest_id
            )
            result = await self.session.execute(stmt)
            existing = result.scalar_one_or_none()

            if existing:
                # UPDATE 占位行
                for key, value in field_values.items():
                    setattr(existing, key, value)
                await self.session.flush()
                self.logger.info(f"更新回测详细结果（占位行）: task_id={task_id}, backtest_id={backtest_id}")
                return existing
            else:
                # INSERT 新行
                detailed_result = BacktestDetailedResult(
                    backtest_id=backtest_id,
                    **field_values,
                )
                self.session.add(detailed_result)
                await self.session.flush()
                self.logger.info(f"创建回测详细结果: task_id={task_id}, backtest_id={backtest_id}")
                return detailed_result

        except Exception as e:
            self.logger.error("创建回测详细结果失败: {}", e, exc_info=True)
            return None

    async def get_backtest_id_by_task_id(self, task_id: str) -> Optional[str]:
        """根据 task_id 查找对应的 backtest_id"""
        try:
            stmt = select(BacktestDetailedResult.backtest_id).where(
                BacktestDetailedResult.task_id == task_id
            )
            result = await self.session.execute(stmt)
            row = result.scalar_one_or_none()
            if row is not None:
                return str(row)
            return None
        except Exception as e:
            self.logger.error("根据task_id查找backtest_id失败: {}", e, exc_info=True)
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

            detailed_result.updated_at = datetime.now(timezone.utc)
            await self.session.flush()

            self.logger.info(f"更新回测详细结果: task_id={task_id}")
            return True

        except Exception as e:
            self.logger.error("更新回测详细结果失败: {}", e, exc_info=True)
            return False

    # ==================== PortfolioSnapshot 相关操作 ====================

    async def batch_create_portfolio_snapshots(
        self, backtest_id: str, snapshots_data: List[Dict[str, Any]]
    ) -> bool:
        """批量创建组合快照"""
        try:
            snapshots = []
            for snapshot_data in snapshots_data:
                snapshot = PortfolioSnapshot(
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

            self.logger.info(f"批量创建组合快照: backtest_id={backtest_id}, count={len(snapshots)}")
            return True

        except Exception as e:
            self.logger.error("批量创建组合快照失败: {}", e, exc_info=True)
            return False

    async def get_portfolio_snapshots(
        self,
        backtest_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[PortfolioSnapshot]:
        """获取组合快照列表"""
        try:
            stmt = select(PortfolioSnapshot).where(PortfolioSnapshot.backtest_id == backtest_id)

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
        self, backtest_id: str, trades_data: List[Dict[str, Any]]
    ) -> bool:
        """批量创建交易记录"""
        try:
            trades = []
            for trade_data in trades_data:
                trade = TradeRecord(
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

            self.logger.info(f"批量创建交易记录: backtest_id={backtest_id}, count={len(trades)}")
            return True

        except Exception as e:
            self.logger.error("批量创建交易记录失败: {}", e, exc_info=True)
            return False

    async def get_trade_records(
        self,
        backtest_id: str,
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
            stmt = select(TradeRecord).where(TradeRecord.backtest_id == backtest_id)

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
        backtest_id: str,
        stock_code: Optional[str] = None,
        action: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> int:
        """获取交易记录总数"""
        try:
            stmt = select(func.count(TradeRecord.id)).where(
                TradeRecord.backtest_id == backtest_id
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

    async def get_trade_statistics(self, backtest_id: str) -> Dict[str, Any]:
        """获取交易统计信息（优化：优先从统计表读取，不存在则实时计算）"""
        try:
            # 优先从统计表读取
            try:
                stats_stmt = select(BacktestStatistics).where(
                    BacktestStatistics.backtest_id == backtest_id
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
                    self.logger.debug(f"统计表不存在，回退到实时计算: backtest_id={backtest_id}")
                else:
                    self.logger.warning(
                        f"查询统计表失败，回退到实时计算: backtest_id={backtest_id}, error={stats_error}"
                    )

            # 向后兼容：如果统计表不存在或没有数据，实时计算
            try:
                total_stmt = select(func.count(TradeRecord.id)).where(
                    TradeRecord.backtest_id == backtest_id
                )
                total_result = await self.session.execute(total_stmt)
                total_trades = total_result.scalar() or 0

                buy_stmt = select(func.count(TradeRecord.id)).where(
                    and_(TradeRecord.backtest_id == backtest_id, TradeRecord.action == "BUY")
                )
                buy_result = await self.session.execute(buy_stmt)
                buy_trades = buy_result.scalar() or 0

                sell_stmt = select(func.count(TradeRecord.id)).where(
                    and_(TradeRecord.backtest_id == backtest_id, TradeRecord.action == "SELL")
                )
                sell_result = await self.session.execute(sell_stmt)
                sell_trades = sell_result.scalar() or 0

                profit_stmt = select(func.count(TradeRecord.id)).where(
                    and_(TradeRecord.backtest_id == backtest_id, TradeRecord.pnl > 0)
                )
                profit_result = await self.session.execute(profit_stmt)
                profit_trades = profit_result.scalar() or 0

                loss_stmt = select(func.count(TradeRecord.id)).where(
                    and_(TradeRecord.backtest_id == backtest_id, TradeRecord.pnl < 0)
                )
                loss_result = await self.session.execute(loss_stmt)
                loss_trades = loss_result.scalar() or 0

                pnl_stmt = select(func.sum(TradeRecord.pnl)).where(
                    TradeRecord.backtest_id == backtest_id
                )
                pnl_result = await self.session.execute(pnl_stmt)
                total_pnl = pnl_result.scalar() or 0.0

                avg_profit_stmt = select(func.avg(TradeRecord.pnl)).where(
                    and_(TradeRecord.backtest_id == backtest_id, TradeRecord.pnl > 0)
                )
                avg_profit_result = await self.session.execute(avg_profit_stmt)
                avg_profit = avg_profit_result.scalar() or 0.0

                avg_loss_stmt = select(func.avg(TradeRecord.pnl)).where(
                    and_(TradeRecord.backtest_id == backtest_id, TradeRecord.pnl < 0)
                )
                avg_loss_result = await self.session.execute(avg_loss_stmt)
                avg_loss = avg_loss_result.scalar() or 0.0

                commission_stmt = select(func.sum(TradeRecord.commission)).where(
                    TradeRecord.backtest_id == backtest_id
                )
                commission_result = await self.session.execute(commission_stmt)
                total_commission = commission_result.scalar() or 0.0

                holding_stmt = select(func.avg(TradeRecord.holding_days)).where(
                    and_(
                        TradeRecord.backtest_id == backtest_id,
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
                    self.logger.info(f"交易记录表不存在，返回空���计: backtest_id={backtest_id}")
                else:
                    self.logger.warning(
                        f"计算交易统计失败: backtest_id={backtest_id}, error={calc_error}"
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
                f"保存信号记录: backtest_id={backtest_id}, signal_id={signal_id}, type={signal_type}"
            )
            return signal_record

        except Exception as e:
            self.logger.error("保存信号记录失败: {}", e, exc_info=True)
            return None

    async def batch_save_signal_records(
        self, backtest_id: str, signals_data: List[Dict[str, Any]]
    ) -> bool:
        """批量保存信号记录"""
        try:
            signals = []
            for signal_data in signals_data:
                signal = SignalRecord(
                    backtest_id=backtest_id,
                    signal_id=signal_data["signal_id"],
                    stock_code=signal_data["stock_code"],
                    stock_name=signal_data.get("stock_name"),
                    signal_type=signal_data["signal_type"],
                    timestamp=self._ensure_datetime(signal_data["timestamp"]),
                    price=signal_data["price"],
                    strength=signal_data.get("strength", 0.0),
                    reason=signal_data.get("reason"),
                    signal_metadata=signal_data.get(
                        "metadata"
                    ),  # 从metadata字段读取，但存储为signal_metadata
                    executed=signal_data.get("executed", False),
                    execution_reason=signal_data.get(
                        "execution_reason"
                    ),  # 添加 execution_reason 字段
                )
                signals.append(signal)

            self.session.add_all(signals)
            await self.session.flush()

            self.logger.info(f"批量保存信号记录: backtest_id={backtest_id}, count={len(signals)}")
            return True

        except Exception as e:
            self.logger.error("批量保存信号记录失败: {}", e, exc_info=True)
            return False

    async def get_signal_records(
        self,
        backtest_id: str,
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
            stmt = select(SignalRecord).where(SignalRecord.backtest_id == backtest_id)

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
        backtest_id: str,
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
                    SignalRecord.backtest_id == backtest_id
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
                operation_name=f"获取信号记录总数 (backtest_id={backtest_id})",
            )

        except Exception as e:
            self.logger.error("获取信号记录总数失败: {}", e, exc_info=True)
            return 0

    async def get_signal_statistics(self, backtest_id: str) -> Dict[str, Any]:
        """获取信号统计信息（优化：优先从统计表读取，不存在则实时计算）"""
        try:
            # 优先从统计表读取
            try:
                stats_stmt = select(BacktestStatistics).where(
                    BacktestStatistics.backtest_id == backtest_id
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
                    self.logger.debug(f"统计表不存在，回退到实时计算: backtest_id={backtest_id}")
                else:
                    self.logger.warning(
                        f"查询统计表失败，回退到实时计算: backtest_id={backtest_id}, error={stats_error}"
                    )

            # 向后兼容：如果统计表不存在或没有数据，实时计算
            try:
                import time

                start_time = time.time()

                base_where = SignalRecord.backtest_id == backtest_id

                total_stmt = select(func.count(SignalRecord.id)).where(base_where)
                buy_stmt = select(func.count(SignalRecord.id)).where(
                    and_(base_where, SignalRecord.signal_type == "BUY")
                )
                sell_stmt = select(func.count(SignalRecord.id)).where(
                    and_(base_where, SignalRecord.signal_type == "SELL")
                )
                executed_stmt = select(func.count(SignalRecord.id)).where(
                    and_(base_where, SignalRecord.executed == True)  # noqa: E712
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
                        f"信号统计查询耗时较长: {elapsed_time:.2f}秒, backtest_id={backtest_id}, total_signals={total_signals}"
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
                    self.logger.info(f"信号记录表不存在，返回空统计: backtest_id={backtest_id}")
                else:
                    self.logger.warning(
                        f"计算信号统计失败: backtest_id={backtest_id}, error={calc_error}"
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
        self, backtest_id: str, stock_code: str, timestamp: datetime, signal_type: str
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
                        SignalRecord.backtest_id == backtest_id,
                        SignalRecord.stock_code == stock_code,
                        SignalRecord.signal_type == signal_type,
                        SignalRecord.timestamp >= timestamp_start,
                        SignalRecord.timestamp < timestamp_end,
                        SignalRecord.executed == False,  # noqa: E712
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
                    f"标记信号为已执行: backtest_id={backtest_id}, signal_id={signal.signal_id}"
                )
                return True

            return False

        except Exception as e:
            self.logger.error("标记信号为已执行失败: {}", e, exc_info=True)
            return False

    async def update_signal_execution_reason(
        self,
        backtest_id: str,
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
                        SignalRecord.backtest_id == backtest_id,
                        SignalRecord.stock_code == stock_code,
                        SignalRecord.signal_type == signal_type,
                        SignalRecord.timestamp >= timestamp_start,
                        SignalRecord.timestamp < timestamp_end,
                        SignalRecord.executed == False,  # noqa: E712  只更新未执行的信号
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
                    f"更新信号未执行原因: backtest_id={backtest_id}, signal_id={signal.signal_id}, reason={execution_reason}"
                )
                return True

            return False

        except Exception as e:
            self.logger.error("更新信号未执行原因失败: {}", e, exc_info=True)
            return False

    async def batch_mark_signals_as_executed(
        self, backtest_id: str, signal_keys: List[tuple]
    ) -> int:
        """
        批量标记信号为已执行

        Args:
            backtest_id: 回测ID
            signal_keys: 信号键列表，每个元素为 (stock_code, timestamp, signal_type)

        Returns:
            成功更新的记录数
        """
        if not signal_keys:
            return 0

        try:
            # 构建 CASE WHEN 语句
            case_conditions = []
            params = {"backtest_id": backtest_id}

            for i, (stock_code, timestamp, signal_type) in enumerate(signal_keys):
                # 将时间戳转换为日期范围
                ts = self._ensure_datetime(timestamp)
                timestamp_start = ts.replace(hour=0, minute=0, second=0, microsecond=0)
                timestamp_end = timestamp_start + timedelta(days=1)

                params[f"stock_code_{i}"] = stock_code
                params[f"signal_type_{i}"] = signal_type
                params[f"ts_start_{i}"] = str(timestamp_start)
                params[f"ts_end_{i}"] = str(timestamp_end)

                case_conditions.append(
                    f"(stock_code = :stock_code_{i} AND signal_type = :signal_type_{i} "
                    f"AND timestamp >= :ts_start_{i} AND timestamp < :ts_end_{i})"
                )

            # 构建完整的 UPDATE 语句
            where_clause = " OR ".join(case_conditions)
            sql = text(
                f"""
                UPDATE signal_records
                SET executed = true, execution_reason = NULL
                WHERE backtest_id = :backtest_id
                AND executed = false
                AND ({where_clause})
            """
            )

            result = await self.session.execute(sql, params)
            updated_count = result.rowcount
            await self.session.flush()

            self.logger.info(
                f"批量标记信号为已执行: backtest_id={backtest_id}, "
                f"请求数={len(signal_keys)}, 更新数={updated_count}"
            )
            return updated_count

        except Exception as e:
            self.logger.error("批量标记信号为已执行失败: {}", e, exc_info=True)
            return 0

    async def batch_update_signal_execution_reasons(
        self, backtest_id: str, signal_reasons: List[tuple]
    ) -> int:
        """
        批量更新信号的未执行原因

        Args:
            backtest_id: 回测ID
            signal_reasons: 信号原因列表，每个元素为 (stock_code, timestamp, signal_type, execution_reason)

        Returns:
            成功更新的记录数
        """
        if not signal_reasons:
            return 0

        try:
            # 构建 CASE WHEN 语句用于 execution_reason
            case_when_parts = []
            where_conditions = []
            params = {"backtest_id": backtest_id}

            for i, (stock_code, timestamp, signal_type, execution_reason) in enumerate(
                signal_reasons
            ):
                # 将时间戳转换为日期范围
                ts = self._ensure_datetime(timestamp)
                timestamp_start = ts.replace(hour=0, minute=0, second=0, microsecond=0)
                timestamp_end = timestamp_start + timedelta(days=1)

                params[f"stock_code_{i}"] = stock_code
                params[f"signal_type_{i}"] = signal_type
                params[f"ts_start_{i}"] = str(timestamp_start)
                params[f"ts_end_{i}"] = str(timestamp_end)
                params[f"reason_{i}"] = execution_reason

                condition = (
                    f"(stock_code = :stock_code_{i} AND signal_type = :signal_type_{i} "
                    f"AND timestamp >= :ts_start_{i} AND timestamp < :ts_end_{i})"
                )
                where_conditions.append(condition)
                case_when_parts.append(f"WHEN {condition} THEN :reason_{i}")

            # 构建完整的 UPDATE 语句
            case_when_clause = " ".join(case_when_parts)
            where_clause = " OR ".join(where_conditions)
            sql = text(
                f"""
                UPDATE signal_records
                SET execution_reason = CASE {case_when_clause} END
                WHERE backtest_id = :backtest_id
                AND executed = false
                AND ({where_clause})
            """
            )

            result = await self.session.execute(sql, params)
            updated_count = result.rowcount
            await self.session.flush()

            self.logger.info(
                f"批量更新信号未执行原因: backtest_id={backtest_id}, "
                f"请求数={len(signal_reasons)}, 更新数={updated_count}"
            )
            return updated_count

        except Exception as e:
            self.logger.error("批量更新信号未执行原因失败: {}", e, exc_info=True)
            return 0

    # ==================== BacktestBenchmark 相关操作 ====================

    async def create_benchmark_data(
        self,
        backtest_id: str,
        benchmark_symbol: str,
        benchmark_name: str,
        benchmark_data: List[Dict[str, Any]],
        comparison_metrics: Dict[str, float],
    ) -> Optional[BacktestBenchmark]:
        """创建基准数据"""
        try:
            benchmark = BacktestBenchmark(
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

            self.logger.info(f"创建基准数据: backtest_id={backtest_id}, benchmark={benchmark_symbol}")
            return benchmark

        except Exception as e:
            self.logger.error(f"创建基准数据失败: {e}", exc_info=True)
            return None

    async def get_benchmark_data(
        self, backtest_id: str, benchmark_symbol: str
    ) -> Optional[BacktestBenchmark]:
        """获取基准数据"""
        try:
            stmt = select(BacktestBenchmark).where(
                and_(
                    BacktestBenchmark.backtest_id == backtest_id,
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
            # 先查出该 task 下所有 backtest_id
            bt_stmt = select(BacktestDetailedResult.backtest_id).where(
                BacktestDetailedResult.task_id == task_id
            )
            bt_result = await self.session.execute(bt_stmt)
            backtest_ids = [row[0] for row in bt_result.all()]

            deleted_counts = {}

            if backtest_ids:
                # 子表按 backtest_id 删除
                sub_tables = [
                    (PortfolioSnapshot, "组合快照"),
                    (TradeRecord, "交易记录"),
                    (SignalRecord, "信号记录"),
                    (BacktestBenchmark, "基准数据"),
                    (BacktestStatistics, "统计信息"),
                    (BacktestChartCache, "图表缓存"),
                ]

                for model_class, table_name in sub_tables:
                    stmt = delete(model_class).where(
                        model_class.backtest_id.in_(backtest_ids)
                    )
                    result = await self.session.execute(stmt)
                    deleted_count = result.rowcount
                    deleted_counts[table_name] = deleted_count

                    if deleted_count > 0:
                        self.logger.info(f"删除{table_name}: {deleted_count}条记录")

            # 主表按 task_id 删除
            main_stmt = delete(BacktestDetailedResult).where(
                BacktestDetailedResult.task_id == task_id
            )
            main_result = await self.session.execute(main_stmt)
            deleted_counts["回测详细结果"] = main_result.rowcount

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
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_to_keep)

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
