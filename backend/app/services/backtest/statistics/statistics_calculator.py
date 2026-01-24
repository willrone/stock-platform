"""
回测统计计算服务
用于计算回测任务的各项统计指标
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, distinct, case
from sqlalchemy.sql import func as sql_func
from collections import defaultdict

from app.models.backtest_detailed_models import SignalRecord, TradeRecord, BacktestStatistics

logger = logging.getLogger(__name__)


class StatisticsCalculator:
    """回测统计计算器"""
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self.logger = logger
    
    async def calculate_all_statistics(self, task_id: str, backtest_id: str) -> BacktestStatistics:
        """
        计算所有统计信息并创建或更新统计记录
        
        Args:
            task_id: 任务ID
            backtest_id: 回测ID
            
        Returns:
            BacktestStatistics: 统计记录对象
        """
        try:
            self.logger.info(f"开始计算统计信息: task_id={task_id}")
            
            # 检查是否已存在统计记录
            stmt = select(BacktestStatistics).where(BacktestStatistics.task_id == task_id)
            result = await self.session.execute(stmt)
            existing_stats = result.scalar_one_or_none()
            
            # 计算各项统计
            signal_stats = await self._calculate_signal_statistics(task_id)
            trade_stats = await self._calculate_trade_statistics(task_id)
            position_stats = await self._calculate_position_statistics(task_id)
            time_range_stats = await self._calculate_time_range_statistics(task_id)
            stock_distribution_stats = await self._calculate_stock_distribution_statistics(task_id)
            performance_stats = await self._calculate_performance_statistics(task_id)
            
            # 合并所有统计
            if existing_stats:
                # 更新现有记录
                self._update_statistics_object(
                    existing_stats,
                    task_id,
                    backtest_id,
                    signal_stats,
                    trade_stats,
                    position_stats,
                    time_range_stats,
                    stock_distribution_stats,
                    performance_stats
                )
                await self.session.flush()
                self.logger.info(f"更新统计信息成功: task_id={task_id}")
                return existing_stats
            else:
                # 创建新记录
                new_stats = BacktestStatistics(
                    task_id=task_id,
                    backtest_id=backtest_id
                )
                self._update_statistics_object(
                    new_stats,
                    task_id,
                    backtest_id,
                    signal_stats,
                    trade_stats,
                    position_stats,
                    time_range_stats,
                    stock_distribution_stats,
                    performance_stats
                )
                self.session.add(new_stats)
                await self.session.flush()
                self.logger.info(f"创建统计信息成功: task_id={task_id}")
                return new_stats
                
        except Exception as e:
            self.logger.error(f"计算统计信息失败: task_id={task_id}, error={e}", exc_info=True)
            raise
    
    def _update_statistics_object(
        self,
        stats: BacktestStatistics,
        task_id: str,
        backtest_id: str,
        signal_stats: Dict[str, Any],
        trade_stats: Dict[str, Any],
        position_stats: Dict[str, Any],
        time_range_stats: Dict[str, Any],
        stock_distribution_stats: Dict[str, Any],
        performance_stats: Dict[str, Any]
    ):
        """更新统计对象"""
        stats.task_id = task_id
        stats.backtest_id = backtest_id
        
        # 信号统计
        stats.total_signals = signal_stats.get('total_signals', 0)
        stats.buy_signals = signal_stats.get('buy_signals', 0)
        stats.sell_signals = signal_stats.get('sell_signals', 0)
        stats.executed_signals = signal_stats.get('executed_signals', 0)
        stats.unexecuted_signals = signal_stats.get('unexecuted_signals', 0)
        stats.execution_rate = signal_stats.get('execution_rate', 0.0)
        stats.avg_signal_strength = signal_stats.get('avg_strength', 0.0)
        
        # 交易统计
        stats.total_trades = trade_stats.get('total_trades', 0)
        stats.buy_trades = trade_stats.get('buy_trades', 0)
        stats.sell_trades = trade_stats.get('sell_trades', 0)
        stats.winning_trades = trade_stats.get('winning_trades', 0)
        stats.losing_trades = trade_stats.get('losing_trades', 0)
        stats.win_rate = trade_stats.get('win_rate', 0.0)
        stats.avg_profit = trade_stats.get('avg_profit', 0.0)
        stats.avg_loss = trade_stats.get('avg_loss', 0.0)
        stats.profit_factor = trade_stats.get('profit_factor', 0.0)
        stats.total_commission = trade_stats.get('total_commission', 0.0)
        stats.total_pnl = trade_stats.get('total_pnl', 0.0)
        stats.avg_holding_days = trade_stats.get('avg_holding_days', 0.0)
        
        # 持仓统计
        stats.total_stocks = position_stats.get('total_stocks', 0)
        stats.profitable_stocks = position_stats.get('profitable_stocks', 0)
        stats.avg_stock_return = position_stats.get('avg_stock_return', 0.0)
        stats.max_stock_return = position_stats.get('max_stock_return')
        stats.min_stock_return = position_stats.get('min_stock_return')
        
        # 时间范围统计
        stats.first_signal_date = time_range_stats.get('first_signal_date')
        stats.last_signal_date = time_range_stats.get('last_signal_date')
        stats.first_trade_date = time_range_stats.get('first_trade_date')
        stats.last_trade_date = time_range_stats.get('last_trade_date')
        stats.trading_days = time_range_stats.get('trading_days', 0)
        
        # 股票分布统计
        stats.unique_stocks_signaled = stock_distribution_stats.get('unique_stocks_signaled', 0)
        stats.unique_stocks_traded = stock_distribution_stats.get('unique_stocks_traded', 0)
        stats.most_signaled_stock = stock_distribution_stats.get('most_signaled_stock')
        stats.most_traded_stock = stock_distribution_stats.get('most_traded_stock')
        
        # 性能指标统计
        stats.max_single_profit = performance_stats.get('max_single_profit')
        stats.max_single_loss = performance_stats.get('max_single_loss')
        stats.max_consecutive_wins = performance_stats.get('max_consecutive_wins', 0)
        stats.max_consecutive_losses = performance_stats.get('max_consecutive_losses', 0)
        stats.largest_position_size = performance_stats.get('largest_position_size')
    
    async def _calculate_signal_statistics(self, task_id: str) -> Dict[str, Any]:
        """计算信号统计信息"""
        try:
            base_where = SignalRecord.task_id == task_id
            
            # 并行执行多个查询
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
            avg_strength_stmt = select(func.avg(SignalRecord.strength)).where(base_where)
            
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
            
            return {
                "total_signals": total_signals,
                "buy_signals": buy_signals,
                "sell_signals": sell_signals,
                "executed_signals": executed_signals,
                "unexecuted_signals": total_signals - executed_signals,
                "execution_rate": executed_signals / total_signals if total_signals > 0 else 0.0,
                "avg_strength": float(avg_strength) if avg_strength else 0.0
            }
        except Exception as e:
            self.logger.error(f"计算信号统计失败: {e}", exc_info=True)
            return {}
    
    async def _calculate_trade_statistics(self, task_id: str) -> Dict[str, Any]:
        """计算交易统计信息"""
        try:
            base_where = TradeRecord.task_id == task_id
            
            # 总交易数
            total_stmt = select(func.count(TradeRecord.id)).where(base_where)
            total_result = await self.session.execute(total_stmt)
            total_trades = total_result.scalar() or 0
            
            # 买入/卖出交易数
            buy_stmt = select(func.count(TradeRecord.id)).where(
                and_(base_where, TradeRecord.action == "BUY")
            )
            sell_stmt = select(func.count(TradeRecord.id)).where(
                and_(base_where, TradeRecord.action == "SELL")
            )
            buy_result = await self.session.execute(buy_stmt)
            sell_result = await self.session.execute(sell_stmt)
            buy_trades = buy_result.scalar() or 0
            sell_trades = sell_result.scalar() or 0
            
            # 盈利/亏损交易数
            profit_stmt = select(func.count(TradeRecord.id)).where(
                and_(base_where, TradeRecord.pnl > 0)
            )
            loss_stmt = select(func.count(TradeRecord.id)).where(
                and_(base_where, TradeRecord.pnl < 0)
            )
            profit_result = await self.session.execute(profit_stmt)
            loss_result = await self.session.execute(loss_stmt)
            winning_trades = profit_result.scalar() or 0
            losing_trades = loss_result.scalar() or 0
            
            # 总盈亏
            pnl_stmt = select(func.sum(TradeRecord.pnl)).where(base_where)
            pnl_result = await self.session.execute(pnl_stmt)
            total_pnl = pnl_result.scalar() or 0.0
            
            # 平均盈利/亏损
            avg_profit_stmt = select(func.avg(TradeRecord.pnl)).where(
                and_(base_where, TradeRecord.pnl > 0)
            )
            avg_loss_stmt = select(func.avg(TradeRecord.pnl)).where(
                and_(base_where, TradeRecord.pnl < 0)
            )
            avg_profit_result = await self.session.execute(avg_profit_stmt)
            avg_loss_result = await self.session.execute(avg_loss_stmt)
            avg_profit = avg_profit_result.scalar() or 0.0
            avg_loss = avg_loss_result.scalar() or 0.0
            
            # 总手续费
            commission_stmt = select(func.sum(TradeRecord.commission)).where(base_where)
            commission_result = await self.session.execute(commission_stmt)
            total_commission = commission_result.scalar() or 0.0
            
            # 平均持仓天数
            holding_stmt = select(func.avg(TradeRecord.holding_days)).where(
                and_(base_where, TradeRecord.holding_days.isnot(None))
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
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "win_rate": winning_trades / total_trades if total_trades > 0 else 0.0,
                "avg_profit": float(avg_profit),
                "avg_loss": float(avg_loss),
                "profit_factor": float(profit_factor),
                "total_commission": float(total_commission),
                "total_pnl": float(total_pnl),
                "avg_holding_days": float(avg_holding_days)
            }
        except Exception as e:
            self.logger.error(f"计算交易统计失败: {e}", exc_info=True)
            return {}
    
    async def _calculate_position_statistics(self, task_id: str) -> Dict[str, Any]:
        """计算持仓统计信息"""
        try:
            # 获取所有交易记录
            stmt = select(TradeRecord).where(TradeRecord.task_id == task_id)
            result = await self.session.execute(stmt)
            trades = result.scalars().all()
            
            if not trades:
                return {
                    "total_stocks": 0,
                    "profitable_stocks": 0,
                    "avg_stock_return": 0.0,
                    "max_stock_return": None,
                    "min_stock_return": None
                }
            
            # 按股票分组计算收益
            stock_returns = defaultdict(float)
            for trade in trades:
                if trade.pnl is not None:
                    stock_returns[trade.stock_code] += trade.pnl
            
            if not stock_returns:
                return {
                    "total_stocks": 0,
                    "profitable_stocks": 0,
                    "avg_stock_return": 0.0,
                    "max_stock_return": None,
                    "min_stock_return": None
                }
            
            returns_list = list(stock_returns.values())
            profitable_count = sum(1 for r in returns_list if r > 0)
            
            return {
                "total_stocks": len(stock_returns),
                "profitable_stocks": profitable_count,
                "avg_stock_return": sum(returns_list) / len(returns_list) if returns_list else 0.0,
                "max_stock_return": max(returns_list) if returns_list else None,
                "min_stock_return": min(returns_list) if returns_list else None
            }
        except Exception as e:
            self.logger.error(f"计算持仓统计失败: {e}", exc_info=True)
            return {}
    
    async def _calculate_time_range_statistics(self, task_id: str) -> Dict[str, Any]:
        """计算时间范围统计信息"""
        try:
            # 信号时间范围
            signal_min_stmt = select(func.min(SignalRecord.timestamp)).where(
                SignalRecord.task_id == task_id
            )
            signal_max_stmt = select(func.max(SignalRecord.timestamp)).where(
                SignalRecord.task_id == task_id
            )
            signal_min_result = await self.session.execute(signal_min_stmt)
            signal_max_result = await self.session.execute(signal_max_stmt)
            first_signal_date = signal_min_result.scalar()
            last_signal_date = signal_max_result.scalar()
            
            # 交易时间范围
            trade_min_stmt = select(func.min(TradeRecord.timestamp)).where(
                TradeRecord.task_id == task_id
            )
            trade_max_stmt = select(func.max(TradeRecord.timestamp)).where(
                TradeRecord.task_id == task_id
            )
            trade_min_result = await self.session.execute(trade_min_stmt)
            trade_max_result = await self.session.execute(trade_max_stmt)
            first_trade_date = trade_min_result.scalar()
            last_trade_date = trade_max_result.scalar()
            
            # 计算交易天数（去重后的日期数）
            trading_days = 0
            if first_trade_date and last_trade_date:
                # 获取所有交易日期（去重）
                dates_stmt = select(
                    func.date(TradeRecord.timestamp).label('trade_date')
                ).where(
                    TradeRecord.task_id == task_id
                ).distinct()
                dates_result = await self.session.execute(dates_stmt)
                trading_days = len(set(row.trade_date for row in dates_result.all()))
            
            return {
                "first_signal_date": first_signal_date,
                "last_signal_date": last_signal_date,
                "first_trade_date": first_trade_date,
                "last_trade_date": last_trade_date,
                "trading_days": trading_days
            }
        except Exception as e:
            self.logger.error(f"计算时间范围统计失败: {e}", exc_info=True)
            return {}
    
    async def _calculate_stock_distribution_statistics(self, task_id: str) -> Dict[str, Any]:
        """计算股票分布统计信息"""
        try:
            # 信号股票分布
            signal_stocks_stmt = select(
                func.count(distinct(SignalRecord.stock_code))
            ).where(SignalRecord.task_id == task_id)
            signal_stocks_result = await self.session.execute(signal_stocks_stmt)
            unique_stocks_signaled = signal_stocks_result.scalar() or 0
            
            # 交易股票分布
            trade_stocks_stmt = select(
                func.count(distinct(TradeRecord.stock_code))
            ).where(TradeRecord.task_id == task_id)
            trade_stocks_result = await self.session.execute(trade_stocks_stmt)
            unique_stocks_traded = trade_stocks_result.scalar() or 0
            
            # 信号最多的股票
            most_signaled_stmt = select(
                SignalRecord.stock_code,
                func.count(SignalRecord.id).label('count')
            ).where(
                SignalRecord.task_id == task_id
            ).group_by(
                SignalRecord.stock_code
            ).order_by(
                func.count(SignalRecord.id).desc()
            ).limit(1)
            most_signaled_result = await self.session.execute(most_signaled_stmt)
            most_signaled_row = most_signaled_result.first()
            most_signaled_stock = most_signaled_row[0] if most_signaled_row else None
            
            # 交易最多的股票
            most_traded_stmt = select(
                TradeRecord.stock_code,
                func.count(TradeRecord.id).label('count')
            ).where(
                TradeRecord.task_id == task_id
            ).group_by(
                TradeRecord.stock_code
            ).order_by(
                func.count(TradeRecord.id).desc()
            ).limit(1)
            most_traded_result = await self.session.execute(most_traded_stmt)
            most_traded_row = most_traded_result.first()
            most_traded_stock = most_traded_row[0] if most_traded_row else None
            
            return {
                "unique_stocks_signaled": unique_stocks_signaled,
                "unique_stocks_traded": unique_stocks_traded,
                "most_signaled_stock": most_signaled_stock,
                "most_traded_stock": most_traded_stock
            }
        except Exception as e:
            self.logger.error(f"计算股票分布统计失败: {e}", exc_info=True)
            return {}
    
    async def _calculate_performance_statistics(self, task_id: str) -> Dict[str, Any]:
        """计算性能指标统计信息"""
        try:
            # 获取所有交易记录
            stmt = select(TradeRecord).where(
                TradeRecord.task_id == task_id
            ).order_by(TradeRecord.timestamp)
            result = await self.session.execute(stmt)
            trades = result.scalars().all()
            
            if not trades:
                return {
                    "max_single_profit": None,
                    "max_single_loss": None,
                    "max_consecutive_wins": 0,
                    "max_consecutive_losses": 0,
                    "largest_position_size": None
                }
            
            # 单笔最大盈利/亏损
            max_profit = None
            max_loss = None
            largest_position = None
            
            consecutive_wins = 0
            consecutive_losses = 0
            max_consecutive_wins = 0
            max_consecutive_losses = 0
            
            for trade in trades:
                # 最大盈利/亏损
                if trade.pnl is not None:
                    if trade.pnl > 0:
                        if max_profit is None or trade.pnl > max_profit:
                            max_profit = trade.pnl
                        consecutive_wins += 1
                        consecutive_losses = 0
                        if consecutive_wins > max_consecutive_wins:
                            max_consecutive_wins = consecutive_wins
                    elif trade.pnl < 0:
                        if max_loss is None or trade.pnl < max_loss:
                            max_loss = trade.pnl
                        consecutive_losses += 1
                        consecutive_wins = 0
                        if consecutive_losses > max_consecutive_losses:
                            max_consecutive_losses = consecutive_losses
                
                # 最大持仓金额
                position_size = trade.quantity * trade.price
                if largest_position is None or position_size > largest_position:
                    largest_position = position_size
            
            return {
                "max_single_profit": float(max_profit) if max_profit is not None else None,
                "max_single_loss": float(max_loss) if max_loss is not None else None,
                "max_consecutive_wins": max_consecutive_wins,
                "max_consecutive_losses": max_consecutive_losses,
                "largest_position_size": float(largest_position) if largest_position is not None else None
            }
        except Exception as e:
            self.logger.error(f"计算性能指标统计失败: {e}", exc_info=True)
            return {}
