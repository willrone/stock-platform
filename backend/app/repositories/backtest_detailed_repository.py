"""
回测详细数据仓库
用于管理回测详细结果、图表缓存、组合快照等数据的CRUD操作
"""

from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete, and_, or_, desc, asc, func
from sqlalchemy.orm import selectinload
from loguru import logger

from app.models.backtest_detailed_models import (
    BacktestDetailedResult,
    BacktestChartCache,
    PortfolioSnapshot,
    TradeRecord,
    BacktestBenchmark
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
        analysis_data: Dict[str, Any]
    ) -> Optional[BacktestDetailedResult]:
        """创建回测详细结果记录"""
        try:
            detailed_result = BacktestDetailedResult(
                task_id=task_id,
                backtest_id=backtest_id,
                sortino_ratio=extended_metrics.get('sortino_ratio', 0.0),
                calmar_ratio=extended_metrics.get('calmar_ratio', 0.0),
                max_drawdown_duration=extended_metrics.get('max_drawdown_duration', 0),
                var_95=extended_metrics.get('var_95', 0.0),
                downside_deviation=extended_metrics.get('downside_deviation', 0.0),
                drawdown_analysis=analysis_data.get('drawdown_analysis'),
                monthly_returns=analysis_data.get('monthly_returns'),
                position_analysis=analysis_data.get('position_analysis'),
                benchmark_comparison=analysis_data.get('benchmark_comparison'),
                rolling_metrics=analysis_data.get('rolling_metrics')
            )
            
            self.session.add(detailed_result)
            await self.session.flush()
            
            self.logger.info(f"创建回测详细结果: task_id={task_id}, backtest_id={backtest_id}")
            return detailed_result
            
        except Exception as e:
            self.logger.error("创建回测详细结果失败: {}", e, exc_info=True)
            return None
    
    async def get_detailed_result_by_task_id(self, task_id: str) -> Optional[BacktestDetailedResult]:
        """根据任务ID获取回测详细结果"""
        try:
            stmt = select(BacktestDetailedResult).where(
                BacktestDetailedResult.task_id == task_id
            )
            result = await self.session.execute(stmt)
            return result.scalar_one_or_none()
            
        except Exception as e:
            self.logger.error("获取回测详细结果失败: {}", e, exc_info=True)
            return None
    
    async def update_detailed_result(
        self,
        task_id: str,
        extended_metrics: Optional[Dict[str, float]] = None,
        analysis_data: Optional[Dict[str, Any]] = None
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
                detailed_result.sortino_ratio = extended_metrics.get('sortino_ratio', detailed_result.sortino_ratio)
                detailed_result.calmar_ratio = extended_metrics.get('calmar_ratio', detailed_result.calmar_ratio)
                detailed_result.max_drawdown_duration = extended_metrics.get('max_drawdown_duration', detailed_result.max_drawdown_duration)
                detailed_result.var_95 = extended_metrics.get('var_95', detailed_result.var_95)
                detailed_result.downside_deviation = extended_metrics.get('downside_deviation', detailed_result.downside_deviation)
            
            # 更新分析数据
            if analysis_data:
                if 'drawdown_analysis' in analysis_data:
                    detailed_result.drawdown_analysis = analysis_data['drawdown_analysis']
                if 'monthly_returns' in analysis_data:
                    detailed_result.monthly_returns = analysis_data['monthly_returns']
                if 'position_analysis' in analysis_data:
                    detailed_result.position_analysis = analysis_data['position_analysis']
                if 'benchmark_comparison' in analysis_data:
                    detailed_result.benchmark_comparison = analysis_data['benchmark_comparison']
                if 'rolling_metrics' in analysis_data:
                    detailed_result.rolling_metrics = analysis_data['rolling_metrics']
            
            detailed_result.updated_at = datetime.utcnow()
            await self.session.flush()
            
            self.logger.info(f"更新回测详细结果: task_id={task_id}")
            return True
            
        except Exception as e:
            self.logger.error("更新回测详细结果失败: {}", e, exc_info=True)
            return False
    
    # ==================== PortfolioSnapshot 相关操作 ====================
    
    async def batch_create_portfolio_snapshots(
        self,
        task_id: str,
        backtest_id: str,
        snapshots_data: List[Dict[str, Any]]
    ) -> bool:
        """批量创建组合快照"""
        try:
            snapshots = []
            for snapshot_data in snapshots_data:
                snapshot = PortfolioSnapshot(
                    task_id=task_id,
                    backtest_id=backtest_id,
                    snapshot_date=self._ensure_datetime(snapshot_data['date']),
                    portfolio_value=snapshot_data['portfolio_value'],
                    cash=snapshot_data['cash'],
                    positions_count=snapshot_data.get('positions_count', 0),
                    total_return=snapshot_data.get('total_return', 0.0),
                    drawdown=snapshot_data.get('drawdown', 0.0),
                    positions=snapshot_data.get('positions')
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
        limit: Optional[int] = None
    ) -> List[PortfolioSnapshot]:
        """获取组合快照列表"""
        try:
            stmt = select(PortfolioSnapshot).where(
                PortfolioSnapshot.task_id == task_id
            )
            
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
        self,
        task_id: str,
        backtest_id: str,
        trades_data: List[Dict[str, Any]]
    ) -> bool:
        """批量创建交易记录"""
        try:
            trades = []
            for trade_data in trades_data:
                trade = TradeRecord(
                    task_id=task_id,
                    backtest_id=backtest_id,
                    trade_id=trade_data['trade_id'],
                    stock_code=trade_data['stock_code'],
                    stock_name=trade_data.get('stock_name'),
                    action=trade_data['action'],
                    quantity=trade_data['quantity'],
                    price=trade_data['price'],
                    timestamp=self._ensure_datetime(trade_data['timestamp']),
                    commission=trade_data.get('commission', 0.0),
                    pnl=trade_data.get('pnl'),
                    holding_days=trade_data.get('holding_days'),
                    technical_indicators=trade_data.get('technical_indicators')
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
        order_desc: bool = True
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
    
    async def get_trade_statistics(self, task_id: str) -> Dict[str, Any]:
        """获取交易统计信息"""
        try:
            # 总交易数
            total_stmt = select(func.count(TradeRecord.id)).where(TradeRecord.task_id == task_id)
            total_result = await self.session.execute(total_stmt)
            total_trades = total_result.scalar() or 0

            # 买入/卖出交易数
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
            
            # 盈利交易数
            profit_stmt = select(func.count(TradeRecord.id)).where(
                and_(TradeRecord.task_id == task_id, TradeRecord.pnl > 0)
            )
            profit_result = await self.session.execute(profit_stmt)
            profit_trades = profit_result.scalar() or 0
            
            # 亏损交易数
            loss_stmt = select(func.count(TradeRecord.id)).where(
                and_(TradeRecord.task_id == task_id, TradeRecord.pnl < 0)
            )
            loss_result = await self.session.execute(loss_stmt)
            loss_trades = loss_result.scalar() or 0
            
            # 总盈亏
            pnl_stmt = select(func.sum(TradeRecord.pnl)).where(TradeRecord.task_id == task_id)
            pnl_result = await self.session.execute(pnl_stmt)
            total_pnl = pnl_result.scalar() or 0.0

            # 平均盈利/亏损
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

            # 总手续费
            commission_stmt = select(func.sum(TradeRecord.commission)).where(TradeRecord.task_id == task_id)
            commission_result = await self.session.execute(commission_stmt)
            total_commission = commission_result.scalar() or 0.0
            
            # 平均持仓天数
            holding_stmt = select(func.avg(TradeRecord.holding_days)).where(
                and_(TradeRecord.task_id == task_id, TradeRecord.holding_days.isnot(None))
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
                "win_rate": profit_trades / total_trades if total_trades > 0 else 0.0,
                "avg_profit": float(avg_profit),
                "avg_loss": float(avg_loss),
                "profit_factor": float(profit_factor),
                "total_commission": float(total_commission),
                "total_pnl": float(total_pnl),
                "avg_holding_days": float(avg_holding_days)
            }
            
        except Exception as e:
            self.logger.error("获取交易统计失败: {}", e, exc_info=True)
            return {}
    
    # ==================== BacktestBenchmark 相关操作 ====================
    
    async def create_benchmark_data(
        self,
        task_id: str,
        backtest_id: str,
        benchmark_symbol: str,
        benchmark_name: str,
        benchmark_data: List[Dict[str, Any]],
        comparison_metrics: Dict[str, float]
    ) -> Optional[BacktestBenchmark]:
        """创建基准数据"""
        try:
            benchmark = BacktestBenchmark(
                task_id=task_id,
                backtest_id=backtest_id,
                benchmark_symbol=benchmark_symbol,
                benchmark_name=benchmark_name,
                benchmark_data=benchmark_data,
                correlation=comparison_metrics.get('correlation'),
                beta=comparison_metrics.get('beta'),
                alpha=comparison_metrics.get('alpha'),
                tracking_error=comparison_metrics.get('tracking_error'),
                information_ratio=comparison_metrics.get('information_ratio'),
                excess_return=comparison_metrics.get('excess_return')
            )
            
            self.session.add(benchmark)
            await self.session.flush()
            
            self.logger.info(f"创建基准数据: task_id={task_id}, benchmark={benchmark_symbol}")
            return benchmark
            
        except Exception as e:
            self.logger.error(f"创建基准数据失败: {e}", exc_info=True)
            return None
    
    async def get_benchmark_data(self, task_id: str, benchmark_symbol: str) -> Optional[BacktestBenchmark]:
        """获取基准数据"""
        try:
            stmt = select(BacktestBenchmark).where(
                and_(
                    BacktestBenchmark.task_id == task_id,
                    BacktestBenchmark.benchmark_symbol == benchmark_symbol
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
                (BacktestBenchmark, "基准数据")
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
                (BacktestBenchmark, "基准数据")
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
