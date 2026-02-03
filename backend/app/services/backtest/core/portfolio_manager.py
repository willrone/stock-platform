"""
组合管理器

负责管理回测过程中的资金、持仓和交易记录
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from app.core.config import settings

from ..models import BacktestConfig, Position, SignalType, Trade, TradingSignal


class PortfolioManager:
    """组合管理器"""

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.cash = config.initial_cash
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.portfolio_history: List[Dict[str, Any]] = []
        self.trade_counter = 0

        # 无成本组合跟踪（用于计算无成本收益）
        self.cash_without_cost = config.initial_cash
        self.positions_without_cost: Dict[str, Position] = {}
        self.portfolio_history_without_cost: List[Dict[str, Any]] = []

        # 成本统计
        self.total_commission = 0.0
        self.total_slippage = 0.0

    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """计算组合总价值（含成本）"""
        total_value = self.cash

        for stock_code, position in self.positions.items():
            if stock_code in current_prices:
                position.current_price = current_prices[stock_code]
                position.market_value = position.quantity * position.current_price
                position.unrealized_pnl = position.market_value - (
                    position.quantity * position.avg_cost
                )
                total_value += position.market_value

        return total_value

    def get_portfolio_value_without_cost(
        self, current_prices: Dict[str, float]
    ) -> float:
        """计算组合总价值（无成本）"""
        total_value = self.cash_without_cost

        for stock_code, position in self.positions_without_cost.items():
            if stock_code in current_prices:
                position.current_price = current_prices[stock_code]
                position.market_value = position.quantity * position.current_price
                position.unrealized_pnl = position.market_value - (
                    position.quantity * position.avg_cost
                )
                total_value += position.market_value

        return total_value

    def execute_signal(
        self, signal: TradingSignal, current_prices: Dict[str, float]
    ) -> tuple[Optional[Trade], Optional[str]]:
        """
        执行交易信号

        Returns:
            tuple[Optional[Trade], Optional[str]]: (交易对象, 失败原因)
            如果执行成功，返回 (Trade, None)
            如果执行失败，返回 (None, 失败原因)
        """
        try:
            stock_code = signal.stock_code
            current_price = current_prices.get(stock_code, signal.price)

            # 应用滑点
            if signal.signal_type == SignalType.BUY:
                execution_price = current_price * (1 + self.config.slippage_rate)
                slippage_cost_per_share = current_price * self.config.slippage_rate
            else:
                execution_price = current_price * (1 - self.config.slippage_rate)
                slippage_cost_per_share = current_price * self.config.slippage_rate

            if signal.signal_type == SignalType.BUY:
                return self._execute_buy(
                    stock_code,
                    execution_price,
                    current_price,
                    slippage_cost_per_share,
                    signal,
                )
            elif signal.signal_type == SignalType.SELL:
                return self._execute_sell(
                    stock_code,
                    execution_price,
                    current_price,
                    slippage_cost_per_share,
                    signal,
                )

            return None, "未知信号类型"

        except Exception as e:
            logger.error(f"执行交易信号失败: {signal.stock_code}, {e}")
            return None, f"执行异常: {str(e)}"

    def _execute_buy(
        self,
        stock_code: str,
        price: float,
        original_price: float,
        slippage_cost_per_share: float,
        signal: TradingSignal,
    ) -> tuple[Optional[Trade], Optional[str]]:
        """
        执行买入

        Returns:
            tuple[Optional[Trade], Optional[str]]: (交易对象, 失败原因)
        """
        # 计算可买数量
        portfolio_value = self.get_portfolio_value({stock_code: price})
        max_position_value = portfolio_value * self.config.max_position_size

        current_position = self.positions.get(stock_code)
        current_position_value = (
            current_position.market_value if current_position else 0
        )

        available_cash_for_stock = max_position_value - current_position_value
        available_cash_for_stock = min(
            available_cash_for_stock, self.cash * 0.95
        )  # 保留5%现金

        if available_cash_for_stock <= 0:
            if (
                current_position_value > 0
                and current_position_value >= max_position_value
            ):
                return (
                    None,
                    f"已达到最大持仓限制: 当前持仓 {current_position_value:.2f} >= 最大持仓 {max_position_value:.2f}",
                )
            else:
                return None, f"可用资金不足: 需要保留5%现金，可用资金 {self.cash:.2f}"

        # 计算购买数量（假设最小交易单位为100股）
        quantity = int(available_cash_for_stock / price / 100) * 100
        if quantity <= 0:
            return (
                None,
                f"可买数量不足: 可用资金 {available_cash_for_stock:.2f}，价格 {price:.2f}，无法买入100股",
            )

        # 计算实际成本
        total_cost = quantity * price
        commission = total_cost * self.config.commission_rate
        slippage_cost = quantity * slippage_cost_per_share
        total_cost_with_commission = total_cost + commission

        if total_cost_with_commission > self.cash:
            return (
                None,
                f"资金不足: 需要 {total_cost_with_commission:.2f}（含手续费 {commission:.2f}），可用 {self.cash:.2f}",
            )

        # 执行交易（含成本）
        self.cash -= total_cost_with_commission

        # 更新持仓（含成本）
        if stock_code in self.positions:
            old_position = self.positions[stock_code]
            new_quantity = old_position.quantity + quantity
            new_avg_cost = (
                (old_position.quantity * old_position.avg_cost) + total_cost
            ) / new_quantity

            self.positions[stock_code] = Position(
                stock_code=stock_code,
                quantity=new_quantity,
                avg_cost=new_avg_cost,
                current_price=price,
                market_value=new_quantity * price,
                unrealized_pnl=0,
                realized_pnl=old_position.realized_pnl,
            )
        else:
            self.positions[stock_code] = Position(
                stock_code=stock_code,
                quantity=quantity,
                avg_cost=price,
                current_price=price,
                market_value=quantity * price,
                unrealized_pnl=0,
                realized_pnl=0,
            )

        # 执行交易（无成本）- 使用原始价格，不扣除手续费和滑点
        cost_without_fees = quantity * original_price
        self.cash_without_cost -= cost_without_fees

        # 更新持仓（无成本）
        if stock_code in self.positions_without_cost:
            old_position = self.positions_without_cost[stock_code]
            new_quantity = old_position.quantity + quantity
            new_avg_cost = (
                (old_position.quantity * old_position.avg_cost) + cost_without_fees
            ) / new_quantity

            self.positions_without_cost[stock_code] = Position(
                stock_code=stock_code,
                quantity=new_quantity,
                avg_cost=new_avg_cost,
                current_price=original_price,
                market_value=new_quantity * original_price,
                unrealized_pnl=0,
                realized_pnl=old_position.realized_pnl,
            )
        else:
            self.positions_without_cost[stock_code] = Position(
                stock_code=stock_code,
                quantity=quantity,
                avg_cost=original_price,
                current_price=original_price,
                market_value=quantity * original_price,
                unrealized_pnl=0,
                realized_pnl=0,
            )

        # 更新成本统计
        self.total_commission += commission
        self.total_slippage += slippage_cost

        # 记录交易
        self.trade_counter += 1
        trade = Trade(
            trade_id=f"T{self.trade_counter:06d}",
            stock_code=stock_code,
            action="BUY",
            quantity=quantity,
            price=price,
            timestamp=signal.timestamp,
            commission=commission,
            slippage_cost=slippage_cost,
            pnl=0,
        )

        self.trades.append(trade)
        logger.info(
            f"执行买入: {stock_code}, 数量: {quantity}, 价格: {price:.2f}, 手续费: {commission:.2f}, 滑点: {slippage_cost:.2f}"
        )

        return trade, None

    def _execute_sell(
        self,
        stock_code: str,
        price: float,
        original_price: float,
        slippage_cost_per_share: float,
        signal: TradingSignal,
    ) -> tuple[Optional[Trade], Optional[str]]:
        """
        执行卖出

        Returns:
            tuple[Optional[Trade], Optional[str]]: (交易对象, 失败原因)
        """
        if stock_code not in self.positions:
            return None, "无持仓"

        position = self.positions[stock_code]
        if position.quantity <= 0:
            return None, "持仓数量为0"

        # 卖出全部持仓（含成本）
        quantity = position.quantity
        total_proceeds = quantity * price
        commission = total_proceeds * self.config.commission_rate
        slippage_cost = quantity * slippage_cost_per_share
        net_proceeds = total_proceeds - commission

        # 计算盈亏
        cost_basis = quantity * position.avg_cost
        pnl = net_proceeds - cost_basis

        # 执行交易（含成本）
        self.cash += net_proceeds

        # 更新持仓（含成本）
        position.realized_pnl += pnl
        del self.positions[stock_code]

        # 执行交易（无成本）- 使用原始价格，不扣除手续费和滑点
        proceeds_without_fees = quantity * original_price
        position_without_cost = self.positions_without_cost[stock_code]
        cost_basis_without_cost = quantity * position_without_cost.avg_cost
        pnl_without_cost = proceeds_without_fees - cost_basis_without_cost

        self.cash_without_cost += proceeds_without_fees
        position_without_cost.realized_pnl += pnl_without_cost
        del self.positions_without_cost[stock_code]

        # 更新成本统计
        self.total_commission += commission
        self.total_slippage += slippage_cost

        # 记录交易
        self.trade_counter += 1
        trade = Trade(
            trade_id=f"T{self.trade_counter:06d}",
            stock_code=stock_code,
            action="SELL",
            quantity=quantity,
            price=price,
            timestamp=signal.timestamp,
            commission=commission,
            slippage_cost=slippage_cost,
            pnl=pnl,
        )

        self.trades.append(trade)
        logger.info(
            f"执行卖出: {stock_code}, 数量: {quantity}, 价格: {price:.2f}, 盈亏: {pnl:.2f}, 手续费: {commission:.2f}, 滑点: {slippage_cost:.2f}"
        )

        return trade, None

    def record_portfolio_snapshot(
        self, date: datetime, current_prices: Dict[str, float]
    ):
        """记录组合快照"""
        portfolio_value = self.get_portfolio_value(current_prices)
        portfolio_value_without_cost = self.get_portfolio_value_without_cost(
            current_prices
        )

        # --- sanity log (debug aid) ---
        # 性能注意：大规模回测会极其频繁调用该函数；默认关闭该日志，避免刷屏/IO 成为瓶颈。
        try:
            if getattr(settings, "ENABLE_PORTFOLIO_SNAPSHOT_SANITY_LOG", False):
                if len(self.positions) > 10:  # default topk is 10; for topk_buffer strategy
                    logger.warning(
                        f"[portfolio_snapshot][sanity] positions_count={len(self.positions)} date={date.strftime('%Y-%m-%d')} "
                        f"holdings={sorted(list(self.positions.keys()))}"
                    )
        except Exception:
            pass

        snapshot = {
            "date": date,
            "cash": self.cash,
            "portfolio_value": portfolio_value,
            "portfolio_value_without_cost": portfolio_value_without_cost,  # 无成本组合价值
            "positions": {
                code: {
                    "quantity": pos.quantity,
                    "avg_cost": pos.avg_cost,
                    "current_price": pos.current_price,
                    "market_value": pos.market_value,
                    "unrealized_pnl": pos.unrealized_pnl,
                }
                for code, pos in self.positions.items()
            },
            "total_trades": len(self.trades),
            "total_commission": self.total_commission,
            "total_slippage": self.total_slippage,
        }

        self.portfolio_history.append(snapshot)

        # 记录无成本组合快照
        snapshot_without_cost = {
            "date": date,
            "cash": self.cash_without_cost,
            "portfolio_value": portfolio_value_without_cost,
            "positions": {
                code: {
                    "quantity": pos.quantity,
                    "avg_cost": pos.avg_cost,
                    "current_price": pos.current_price,
                    "market_value": pos.market_value,
                    "unrealized_pnl": pos.unrealized_pnl,
                }
                for code, pos in self.positions_without_cost.items()
            },
            "total_trades": len(self.trades),
        }

        self.portfolio_history_without_cost.append(snapshot_without_cost)

    def get_performance_metrics(self) -> Dict[str, float]:
        """计算绩效指标（含成本）"""
        if not self.portfolio_history:
            return {}

        # 计算收益序列
        values = [snapshot["portfolio_value"] for snapshot in self.portfolio_history]
        returns = pd.Series(values).pct_change().dropna()

        if len(returns) == 0:
            return {}

        # 基础指标
        total_return = (
            values[-1] - self.config.initial_cash
        ) / self.config.initial_cash

        # 年化收益率
        days = (
            self.portfolio_history[-1]["date"] - self.portfolio_history[0]["date"]
        ).days
        annualized_return = (
            (1 + total_return) ** (365 / max(days, 1)) - 1 if days > 0 else 0
        )

        # 波动率
        volatility = returns.std() * np.sqrt(252)

        # 夏普比率
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0

        # 最大回撤
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()

        # 交易统计（以卖出成交为准，避免买入计入胜率）
        sell_trades = [t for t in self.trades if t.action == "SELL"]
        winning_trades = [t for t in sell_trades if t.pnl > 0]
        losing_trades = [t for t in sell_trades if t.pnl < 0]
        win_rate_denominator = len(winning_trades) + len(losing_trades)
        win_rate = (
            len(winning_trades) / win_rate_denominator
            if win_rate_denominator > 0
            else 0
        )

        avg_win = (
            float(np.mean([t.pnl for t in winning_trades])) if winning_trades else 0.0
        )
        avg_loss = (
            float(np.mean([t.pnl for t in losing_trades])) if losing_trades else 0.0
        )
        profit_factor = (
            float(abs(avg_win / avg_loss)) if avg_loss != 0 else float("inf")
        )

        metrics = {
            "total_return": float(total_return),
            "annualized_return": float(annualized_return),
            "volatility": float(volatility),
            "sharpe_ratio": float(sharpe_ratio),
            "max_drawdown": float(max_drawdown),
            "win_rate": float(win_rate),
            "profit_factor": profit_factor,
            "total_trades": len(self.trades),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "total_commission": float(self.total_commission),
            "total_slippage": float(self.total_slippage),
            "total_cost": float(self.total_commission + self.total_slippage),
        }

        # 添加无成本指标
        metrics_without_cost = self.get_performance_metrics_without_cost()
        for key, value in metrics_without_cost.items():
            metrics[f"{key}_without_cost"] = value

        return metrics

    def get_performance_metrics_without_cost(self) -> Dict[str, float]:
        """计算绩效指标（无成本）"""
        if not self.portfolio_history_without_cost:
            return {}

        # 计算收益序列
        values = [
            snapshot["portfolio_value"]
            for snapshot in self.portfolio_history_without_cost
        ]
        returns = pd.Series(values).pct_change().dropna()

        if len(returns) == 0:
            return {}

        # 基础指标
        total_return = (
            values[-1] - self.config.initial_cash
        ) / self.config.initial_cash

        # 年化收益率
        days = (
            self.portfolio_history_without_cost[-1]["date"]
            - self.portfolio_history_without_cost[0]["date"]
        ).days
        annualized_return = (
            (1 + total_return) ** (365 / max(days, 1)) - 1 if days > 0 else 0
        )

        # 波动率
        volatility = returns.std() * np.sqrt(252)

        # 信息比率（相对于含成本收益）
        if self.portfolio_history:
            values_with_cost = [
                snapshot["portfolio_value"] for snapshot in self.portfolio_history
            ]
            returns_with_cost = pd.Series(values_with_cost).pct_change().dropna()
            if len(returns_with_cost) == len(returns):
                excess_returns = returns - returns_with_cost
                tracking_error = excess_returns.std() * np.sqrt(252)
                information_ratio = (
                    excess_returns.mean() * np.sqrt(252) / tracking_error
                    if tracking_error > 0
                    else 0
                )
            else:
                information_ratio = 0.0
        else:
            information_ratio = 0.0

        # 最大回撤
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()

        return {
            "total_return": float(total_return),
            "annualized_return": float(annualized_return),
            "volatility": float(volatility),
            "information_ratio": float(information_ratio),
            "max_drawdown": float(max_drawdown),
            "mean": float(returns.mean()),
            "std": float(returns.std()),
        }
