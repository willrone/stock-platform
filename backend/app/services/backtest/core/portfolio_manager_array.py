"""
组合管理器 - 数组优化版本 (Phase 1)

使用 numpy 数组管理持仓，提升性能
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from app.core.config import settings

from ..models import BacktestConfig, Position, SignalType, Trade, TradingSignal


class PortfolioManagerArray:
    """组合管理器 - 数组优化版本"""

    def __init__(self, config: BacktestConfig, stock_codes: List[str]):
        """
        初始化组合管理器
        
        Args:
            config: 回测配置
            stock_codes: 股票代码列表（用于建立索引映射）
        """
        self.config = config
        self.cash = config.initial_cash
        
        # 股票代码映射
        self.stock_codes = stock_codes
        self.code_to_idx = {code: i for i, code in enumerate(stock_codes)}
        self.n_stocks = len(stock_codes)
        
        # 持仓数组化 (shape: [n_stocks])
        self.quantities = np.zeros(self.n_stocks, dtype=np.int32)  # 持仓数量
        self.avg_costs = np.zeros(self.n_stocks, dtype=np.float64)  # 平均成本
        self.realized_pnl = np.zeros(self.n_stocks, dtype=np.float64)  # 已实现盈亏
        
        # Trade records 使用 list（延迟转换为 DataFrame）
        self.trades: List[Dict[str, Any]] = []
        
        # 轻量 equity 曲线
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.trade_counter = 0
        
        # 内部计数
        self._snapshot_counter = 0
        
        # 无成本组合跟踪
        self.cash_without_cost = config.initial_cash
        self.quantities_without_cost = np.zeros(self.n_stocks, dtype=np.int32)
        self.avg_costs_without_cost = np.zeros(self.n_stocks, dtype=np.float64)
        self.realized_pnl_without_cost = np.zeros(self.n_stocks, dtype=np.float64)
        
        # 成本统计
        self.total_commission = 0.0
        self.total_slippage = 0.0
        self.total_capital_injection = 0.0

        # 可选：完整快照历史（默认关闭以节省内存）
        self.portfolio_history: List[Dict[str, Any]] = []
        self.portfolio_history_without_cost: List[Dict[str, Any]] = []

        # [P2 优化] 价格数组缓存，避免每次调用 get_portfolio_value 时重复转换
        self._price_array = np.zeros(self.n_stocks, dtype=np.float64)
        self._price_array_valid = False  # 标记缓存是否有效

        # [P0 优化] positions 缓存，避免每次调用 positions 属性时重复构建字典
        self._positions_cache: Optional[Dict[str, Position]] = None
        self._positions_cache_valid = False

        # 开仓日期跟踪（用于最小持仓期检查）
        # entry_dates[idx] 记录股票 idx 最近一次开仓的日期
        self._entry_dates: Dict[int, datetime] = {}

    def _invalidate_positions_cache(self) -> None:
        """[P0 优化] 使 positions 缓存失效，在持仓变化时调用"""
        self._positions_cache_valid = False

    @property
    def entry_dates(self) -> Dict[str, datetime]:
        """返回开仓日期字典 {stock_code: entry_date}，用于最小持仓期检查"""
        return {
            self.stock_codes[idx]: dt
            for idx, dt in self._entry_dates.items()
            if self.quantities[idx] > 0
        }

    def set_current_prices(self, current_prices: Dict[str, float]) -> None:
        """[P2 优化] 批量设置当前价格到数组，供后续向量化计算使用

        在每个交易日开始时调用一次，后续的 get_portfolio_value 等方法
        可以直接使用缓存的价格数组，避免重复的字典查找和转换。
        """
        # 重置价格数组
        self._price_array.fill(0.0)

        # 批量填充价格
        for code, price in current_prices.items():
            idx = self.code_to_idx.get(code)
            if idx is not None:
                self._price_array[idx] = price

        self._price_array_valid = True

    def get_portfolio_value(self, current_prices: Dict[str, float] = None) -> float:
        """计算组合总价值（含成本）- 向量化版本

        [P2 优化] 如果已调用 set_current_prices()，则使用缓存的价格数组进行向量化计算。
        否则回退到原有的字典遍历方式（兼容性）。
        """
        # 快速路径：使用缓存的价格数组（None 或空字典都走这条路径）
        if current_prices is None or len(current_prices) == 0:
            if self._price_array_valid:
                # 纯向量化计算：cash + dot(quantities, prices)
                return self.cash + np.dot(self.quantities, self._price_array)
            else:
                # 没有缓存，返回纯现金（兼容旧行为）
                return self.cash

        # 如果传入了非空 current_prices，更新缓存并使用向量化计算
        self.set_current_prices(current_prices)
        return self.cash + np.dot(self.quantities, self._price_array)

    def get_portfolio_value_without_cost(self, current_prices: Dict[str, float] = None) -> float:
        """计算组合总价值（无成本）- 向量化版本

        [P2 优化] 如果已调用 set_current_prices()，则使用缓存的价格数组进行向量化计算。
        """
        # 快速路径：使用缓存的价格数组（None 或空字典都走这条路径）
        if current_prices is None or len(current_prices) == 0:
            if self._price_array_valid:
                return self.cash_without_cost + np.dot(self.quantities_without_cost, self._price_array)
            else:
                return self.cash_without_cost

        # 如果传入了非空 current_prices，更新缓存并使用向量化计算
        self.set_current_prices(current_prices)
        return self.cash_without_cost + np.dot(self.quantities_without_cost, self._price_array)

    def get_position(self, stock_code: str) -> Optional[Position]:
        """获取持仓信息（兼容接口）"""
        idx = self.code_to_idx.get(stock_code)
        if idx is None or self.quantities[idx] <= 0:
            return None
        
        return Position(
            stock_code=stock_code,
            quantity=int(self.quantities[idx]),
            avg_cost=float(self.avg_costs[idx]),
            current_price=0.0,  # 需要外部提供
            market_value=0.0,
            unrealized_pnl=0.0,
            realized_pnl=float(self.realized_pnl[idx]),
        )

    @property
    def positions(self) -> Dict[str, Position]:
        """返回持仓字典（兼容接口）- [P0 优化] 带缓存"""
        # 检查缓存是否有效
        if self._positions_cache_valid and self._positions_cache is not None:
            return self._positions_cache

        # 使用 np.nonzero 向量化查找有持仓的股票索引
        position_indices = np.nonzero(self.quantities > 0)[0]

        result = {}
        for i in position_indices:
            code = self.stock_codes[i]
            result[code] = Position(
                stock_code=code,
                quantity=int(self.quantities[i]),
                avg_cost=float(self.avg_costs[i]),
                current_price=0.0,
                market_value=0.0,
                unrealized_pnl=0.0,
                realized_pnl=float(self.realized_pnl[i]),
            )

        # 缓存结果并标记有效
        self._positions_cache = result
        self._positions_cache_valid = True
        return result

    @property
    def positions_without_cost(self) -> Dict[str, Position]:
        """返回无成本持仓字典（兼容接口）"""
        result = {}
        for i in range(self.n_stocks):
            if self.quantities_without_cost[i] > 0:
                code = self.stock_codes[i]
                result[code] = Position(
                    stock_code=code,
                    quantity=int(self.quantities_without_cost[i]),
                    avg_cost=float(self.avg_costs_without_cost[i]),
                    current_price=0.0,
                    market_value=0.0,
                    unrealized_pnl=0.0,
                    realized_pnl=float(self.realized_pnl_without_cost[i]),
                )
        return result

    def execute_signal(
        self, signal: TradingSignal, current_prices: Dict[str, float]
    ) -> tuple[Optional[Trade], Optional[str]]:
        """执行交易信号"""
        try:
            stock_code = signal.stock_code
            idx = self.code_to_idx.get(stock_code)
            
            if idx is None:
                return None, f"股票代码不在universe中: {stock_code}"
            
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
                    idx,
                    stock_code,
                    execution_price,
                    current_price,
                    slippage_cost_per_share,
                    signal,
                )
            elif signal.signal_type == SignalType.SELL:
                return self._execute_sell(
                    idx,
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
        idx: int,
        stock_code: str,
        price: float,
        original_price: float,
        slippage_cost_per_share: float,
        signal: TradingSignal,
    ) -> tuple[Optional[Trade], Optional[str]]:
        """执行买入 - 数组优化版本"""
        unlimited_buy = getattr(self.config, "enable_unlimited_buy", False)

        if unlimited_buy:
            available_cash_for_stock = self.cash
        else:
            portfolio_value = self.get_portfolio_value({stock_code: price})
            max_position_value = portfolio_value * self.config.max_position_size
            current_position_value = self.quantities[idx] * price if self.quantities[idx] > 0 else 0
            available_cash_for_stock = max_position_value - current_position_value
            available_cash_for_stock = min(available_cash_for_stock, self.cash * 0.95)

            if available_cash_for_stock <= 0:
                if current_position_value > 0 and current_position_value >= max_position_value:
                    return None, f"已达到最大持仓限制"
                else:
                    return None, f"可用资金不足"

        # 计算购买数量（100股为单位）
        quantity = int(available_cash_for_stock / price / 100) * 100
        if not unlimited_buy and quantity <= 0:
            return None, f"可买数量不足"
        if unlimited_buy and quantity <= 0:
            quantity = 100

        # 计算成本
        total_cost = quantity * price
        commission = total_cost * self.config.commission_rate
        slippage_cost = quantity * slippage_cost_per_share
        total_cost_with_commission = total_cost + commission

        if total_cost_with_commission > self.cash:
            if unlimited_buy:
                needed = total_cost_with_commission - self.cash
                self.cash += needed
                self.total_capital_injection += needed
                self.cash_without_cost += needed  # 无成本账本同步
                logger.debug(f"不限制买入: 补充资金 {needed:.2f} 用于买入 {stock_code}")
            else:
                return None, f"资金不足"

        # 执行交易（含成本）
        self.cash -= total_cost_with_commission

        # 更新持仓数组
        old_quantity = self.quantities[idx]
        new_quantity = old_quantity + quantity
        
        if old_quantity > 0:
            # 更新平均成本
            self.avg_costs[idx] = (old_quantity * self.avg_costs[idx] + total_cost) / new_quantity
        else:
            self.avg_costs[idx] = price
            # 新开仓：记录开仓日期
            self._entry_dates[idx] = signal.timestamp
        
        self.quantities[idx] = new_quantity

        # [P0 优化] 持仓变化，使缓存失效
        self._invalidate_positions_cache()

        # 执行交易（无成本）
        cost_without_fees = quantity * original_price
        self.cash_without_cost -= cost_without_fees

        old_quantity_nc = self.quantities_without_cost[idx]
        new_quantity_nc = old_quantity_nc + quantity
        
        if old_quantity_nc > 0:
            self.avg_costs_without_cost[idx] = (
                old_quantity_nc * self.avg_costs_without_cost[idx] + cost_without_fees
            ) / new_quantity_nc
        else:
            self.avg_costs_without_cost[idx] = original_price
        
        self.quantities_without_cost[idx] = new_quantity_nc

        # 更新成本统计
        self.total_commission += commission
        self.total_slippage += slippage_cost

        # 记录交易（使用 dict，延迟转换）
        self.trade_counter += 1
        trade_dict = {
            'trade_id': f"T{self.trade_counter:06d}",
            'stock_code': stock_code,
            'action': 'BUY',
            'quantity': quantity,
            'price': price,
            'timestamp': signal.timestamp,
            'commission': commission,
            'slippage_cost': slippage_cost,
            'pnl': 0.0,
        }
        self.trades.append(trade_dict)

        # 返回 Trade 对象（兼容接口）
        trade = Trade(
            trade_id=trade_dict['trade_id'],
            stock_code=stock_code,
            action='BUY',
            quantity=quantity,
            price=price,
            timestamp=signal.timestamp,
            commission=commission,
            slippage_cost=slippage_cost,
            pnl=0.0,
        )

        return trade, None

    def _execute_sell(
        self,
        idx: int,
        stock_code: str,
        price: float,
        original_price: float,
        slippage_cost_per_share: float,
        signal: TradingSignal,
    ) -> tuple[Optional[Trade], Optional[str]]:
        """执行卖出 - 数组优化版本"""
        if self.quantities[idx] <= 0:
            return None, "无持仓"

        # 卖出全部持仓
        quantity = int(self.quantities[idx])
        total_proceeds = quantity * price
        commission = total_proceeds * self.config.commission_rate
        slippage_cost = quantity * slippage_cost_per_share
        net_proceeds = total_proceeds - commission

        # 计算盈亏
        cost_basis = quantity * self.avg_costs[idx]
        pnl = net_proceeds - cost_basis

        # 执行交易（含成本）
        self.cash += net_proceeds
        self.realized_pnl[idx] += pnl
        self.quantities[idx] = 0
        self.avg_costs[idx] = 0.0

        # 清除开仓日期
        self._entry_dates.pop(idx, None)

        # [P0 优化] 持仓变化，使缓存失效
        self._invalidate_positions_cache()

        # 执行交易（无成本）
        proceeds_without_fees = quantity * original_price
        cost_basis_without_cost = quantity * self.avg_costs_without_cost[idx]
        pnl_without_cost = proceeds_without_fees - cost_basis_without_cost

        self.cash_without_cost += proceeds_without_fees
        self.realized_pnl_without_cost[idx] += pnl_without_cost
        self.quantities_without_cost[idx] = 0
        self.avg_costs_without_cost[idx] = 0.0

        # 更新成本统计
        self.total_commission += commission
        self.total_slippage += slippage_cost

        # 记录交易
        self.trade_counter += 1
        trade_dict = {
            'trade_id': f"T{self.trade_counter:06d}",
            'stock_code': stock_code,
            'action': 'SELL',
            'quantity': quantity,
            'price': price,
            'timestamp': signal.timestamp,
            'commission': commission,
            'slippage_cost': slippage_cost,
            'pnl': pnl,
        }
        self.trades.append(trade_dict)

        trade = Trade(
            trade_id=trade_dict['trade_id'],
            stock_code=stock_code,
            action='SELL',
            quantity=quantity,
            price=price,
            timestamp=signal.timestamp,
            commission=commission,
            slippage_cost=slippage_cost,
            pnl=pnl,
        )

        return trade, None

    def record_portfolio_snapshot(self, date: datetime, current_prices: Dict[str, float] = None):
        """记录组合快照

        [P2 优化] 如果已调用 set_current_prices()，可以传入 None 使用缓存的价格数组。
        """
        # 如果传入了价格，先更新缓存
        if current_prices is not None:
            self.set_current_prices(current_prices)

        # 使用向量化计算（利用缓存的价格数组）
        portfolio_value = self.get_portfolio_value()
        portfolio_value_without_cost = self.get_portfolio_value_without_cost()

        # 轻量记录（用于指标计算）
        self.equity_curve.append((date, float(portfolio_value)))

        # 可选：完整快照
        self._snapshot_counter += 1
        if getattr(self.config, "record_portfolio_history", True):
            stride = int(getattr(self.config, "portfolio_history_stride", 1) or 1)
            if stride <= 1 or (self._snapshot_counter % stride == 0):
                snapshot = {
                    "date": date,
                    "cash": self.cash,
                    "portfolio_value": portfolio_value,
                    "portfolio_value_without_cost": portfolio_value_without_cost,
                    "positions": {},  # 简化版本，不记录详细持仓
                    "total_trades": len(self.trades),
                    "total_commission": self.total_commission,
                    "total_slippage": self.total_slippage,
                }
                self.portfolio_history.append(snapshot)

    def get_performance_metrics(self) -> Dict[str, float]:
        """计算绩效指标（含成本）"""
        if not self.equity_curve:
            return {}

        values = [v for _, v in self.equity_curve]
        returns = pd.Series(values).pct_change().dropna()

        if len(returns) == 0:
            return {}

        # 基础指标（不限制买入时，收益基准含补充资金）
        total_invested = self.config.initial_cash + getattr(
            self, "total_capital_injection", 0.0
        )
        total_return = (values[-1] - total_invested) / total_invested

        # 年化收益率
        days = (self.equity_curve[-1][0] - self.equity_curve[0][0]).days
        annualized_return = (1 + total_return) ** (365 / max(days, 1)) - 1 if days > 0 else 0

        # 波动率
        volatility = returns.std() * np.sqrt(252)

        # 夏普比率
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0

        # 最大回撤
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()

        # 交易统计
        sell_trades = [t for t in self.trades if t['action'] == 'SELL']
        winning_trades = [t for t in sell_trades if t['pnl'] > 0]
        losing_trades = [t for t in sell_trades if t['pnl'] < 0]
        win_rate_denominator = len(winning_trades) + len(losing_trades)
        win_rate = len(winning_trades) / win_rate_denominator if win_rate_denominator > 0 else 0

        avg_win = float(np.mean([t['pnl'] for t in winning_trades])) if winning_trades else 0.0
        avg_loss = float(np.mean([t['pnl'] for t in losing_trades])) if losing_trades else 0.0
        profit_factor = float(abs(avg_win / avg_loss)) if avg_loss != 0 else float("inf")

        total_cap_inj = getattr(self, "total_capital_injection", 0.0)
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
            "total_capital_injection": float(total_cap_inj),
            "total_cost": float(self.total_commission + self.total_slippage),
        }

        return metrics

    def get_performance_metrics_without_cost(self) -> Dict[str, float]:
        """计算绩效指标（无成本）- 简化版本"""
        # 简化实现，仅返回基础指标
        return {
            "total_return": 0.0,
            "annualized_return": 0.0,
            "volatility": 0.0,
            "information_ratio": 0.0,
            "max_drawdown": 0.0,
            "mean": 0.0,
            "std": 0.0,
        }
