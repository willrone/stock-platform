"""
回测引擎 - 策略框架、信号生成和回测执行
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum
import talib
from loguru import logger

from backend.app.core.error_handler import TaskError, ErrorSeverity, ErrorContext
from backend.app.models.task_models import BacktestResult


class SignalType(Enum):
    """信号类型"""
    BUY = 1
    SELL = -1
    HOLD = 0


class OrderType(Enum):
    """订单类型"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


@dataclass
class TradingSignal:
    """交易信号"""
    timestamp: datetime
    stock_code: str
    signal_type: SignalType
    strength: float  # 信号强度 0-1
    price: float
    reason: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class Trade:
    """交易记录"""
    trade_id: str
    stock_code: str
    action: str  # BUY, SELL
    quantity: int
    price: float
    timestamp: datetime
    commission: float
    pnl: float = 0.0
    cumulative_pnl: float = 0.0


@dataclass
class Position:
    """持仓信息"""
    stock_code: str
    quantity: int
    avg_cost: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float


@dataclass
class BacktestConfig:
    """回测配置"""
    initial_cash: float = 100000.0
    commission_rate: float = 0.001  # 手续费率
    slippage_rate: float = 0.001   # 滑点率
    max_position_size: float = 0.2  # 最大单股持仓比例
    stop_loss_pct: float = 0.05    # 止损比例
    take_profit_pct: float = 0.15  # 止盈比例
    rebalance_frequency: str = "daily"  # 调仓频率


class BaseStrategy(ABC):
    """策略基类"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.indicators = {}
        
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame, current_date: datetime) -> List[TradingSignal]:
        """生成交易信号"""
        pass
    
    @abstractmethod
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """计算技术指标"""
        pass
    
    def validate_signal(self, signal: TradingSignal, portfolio_value: float, 
                       current_positions: Dict[str, Position]) -> bool:
        """验证信号有效性"""
        # 基础验证
        if signal.strength < 0.1:  # 信号强度太低
            return False
        
        # 检查持仓限制
        if signal.signal_type == SignalType.BUY:
            current_position = current_positions.get(signal.stock_code)
            if current_position and current_position.market_value / portfolio_value > 0.3:
                return False  # 单股持仓过大
        
        return True


class MovingAverageStrategy(BaseStrategy):
    """移动平均策略"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("MovingAverage", config)
        self.short_window = config.get('short_window', 5)
        self.long_window = config.get('long_window', 20)
        self.signal_threshold = config.get('signal_threshold', 0.02)
    
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """计算移动平均指标"""
        close_prices = data['close']
        
        indicators = {
            'sma_short': close_prices.rolling(window=self.short_window).mean(),
            'sma_long': close_prices.rolling(window=self.long_window).mean(),
            'price': close_prices
        }
        
        # 计算移动平均差值
        indicators['ma_diff'] = (indicators['sma_short'] - indicators['sma_long']) / indicators['sma_long']
        
        return indicators
    
    def generate_signals(self, data: pd.DataFrame, current_date: datetime) -> List[TradingSignal]:
        """生成移动平均交叉信号"""
        signals = []
        
        try:
            # 计算指标
            indicators = self.calculate_indicators(data)
            
            # 获取当前数据点
            current_idx = data.index.get_loc(current_date) if current_date in data.index else -1
            if current_idx < self.long_window:
                return signals  # 数据不足
            
            current_price = indicators['price'].iloc[current_idx]
            current_ma_diff = indicators['ma_diff'].iloc[current_idx]
            prev_ma_diff = indicators['ma_diff'].iloc[current_idx - 1]
            
            stock_code = data.attrs.get('stock_code', 'UNKNOWN')
            
            # 生成买入信号
            if (prev_ma_diff <= 0 and current_ma_diff > 0 and 
                abs(current_ma_diff) > self.signal_threshold):
                
                signal = TradingSignal(
                    timestamp=current_date,
                    stock_code=stock_code,
                    signal_type=SignalType.BUY,
                    strength=min(1.0, abs(current_ma_diff) * 10),
                    price=current_price,
                    reason=f"短期均线上穿长期均线，差值: {current_ma_diff:.3f}",
                    metadata={
                        'sma_short': indicators['sma_short'].iloc[current_idx],
                        'sma_long': indicators['sma_long'].iloc[current_idx],
                        'ma_diff': current_ma_diff
                    }
                )
                signals.append(signal)
            
            # 生成卖出信号
            elif (prev_ma_diff >= 0 and current_ma_diff < 0 and 
                  abs(current_ma_diff) > self.signal_threshold):
                
                signal = TradingSignal(
                    timestamp=current_date,
                    stock_code=stock_code,
                    signal_type=SignalType.SELL,
                    strength=min(1.0, abs(current_ma_diff) * 10),
                    price=current_price,
                    reason=f"短期均线下穿长期均线，差值: {current_ma_diff:.3f}",
                    metadata={
                        'sma_short': indicators['sma_short'].iloc[current_idx],
                        'sma_long': indicators['sma_long'].iloc[current_idx],
                        'ma_diff': current_ma_diff
                    }
                )
                signals.append(signal)
            
            return signals
            
        except Exception as e:
            logger.error(f"移动平均策略信号生成失败: {e}")
            return []


class RSIStrategy(BaseStrategy):
    """RSI策略"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("RSI", config)
        self.rsi_period = config.get('rsi_period', 14)
        self.oversold_threshold = config.get('oversold_threshold', 30)
        self.overbought_threshold = config.get('overbought_threshold', 70)
    
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """计算RSI指标"""
        close_prices = data['close']
        
        # 使用talib计算RSI
        rsi = pd.Series(talib.RSI(close_prices.values, timeperiod=self.rsi_period), 
                       index=close_prices.index)
        
        return {
            'rsi': rsi,
            'price': close_prices
        }
    
    def generate_signals(self, data: pd.DataFrame, current_date: datetime) -> List[TradingSignal]:
        """生成RSI信号"""
        signals = []
        
        try:
            indicators = self.calculate_indicators(data)
            
            current_idx = data.index.get_loc(current_date) if current_date in data.index else -1
            if current_idx < self.rsi_period:
                return signals
            
            current_rsi = indicators['rsi'].iloc[current_idx]
            current_price = indicators['price'].iloc[current_idx]
            stock_code = data.attrs.get('stock_code', 'UNKNOWN')
            
            # RSI超卖信号（买入）
            if current_rsi < self.oversold_threshold:
                strength = (self.oversold_threshold - current_rsi) / self.oversold_threshold
                signal = TradingSignal(
                    timestamp=current_date,
                    stock_code=stock_code,
                    signal_type=SignalType.BUY,
                    strength=min(1.0, strength),
                    price=current_price,
                    reason=f"RSI超卖: {current_rsi:.2f}",
                    metadata={'rsi': current_rsi}
                )
                signals.append(signal)
            
            # RSI超买信号（卖出）
            elif current_rsi > self.overbought_threshold:
                strength = (current_rsi - self.overbought_threshold) / (100 - self.overbought_threshold)
                signal = TradingSignal(
                    timestamp=current_date,
                    stock_code=stock_code,
                    signal_type=SignalType.SELL,
                    strength=min(1.0, strength),
                    price=current_price,
                    reason=f"RSI超买: {current_rsi:.2f}",
                    metadata={'rsi': current_rsi}
                )
                signals.append(signal)
            
            return signals
            
        except Exception as e:
            logger.error(f"RSI策略信号生成失败: {e}")
            return []


class MACDStrategy(BaseStrategy):
    """MACD策略"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("MACD", config)
        self.fast_period = config.get('fast_period', 12)
        self.slow_period = config.get('slow_period', 26)
        self.signal_period = config.get('signal_period', 9)
    
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """计算MACD指标"""
        close_prices = data['close']
        
        # 使用talib计算MACD
        macd, macd_signal, macd_hist = talib.MACD(
            close_prices.values, 
            fastperiod=self.fast_period,
            slowperiod=self.slow_period, 
            signalperiod=self.signal_period
        )
        
        return {
            'macd': pd.Series(macd, index=close_prices.index),
            'macd_signal': pd.Series(macd_signal, index=close_prices.index),
            'macd_hist': pd.Series(macd_hist, index=close_prices.index),
            'price': close_prices
        }
    
    def generate_signals(self, data: pd.DataFrame, current_date: datetime) -> List[TradingSignal]:
        """生成MACD信号"""
        signals = []
        
        try:
            indicators = self.calculate_indicators(data)
            
            current_idx = data.index.get_loc(current_date) if current_date in data.index else -1
            if current_idx < self.slow_period + self.signal_period:
                return signals
            
            current_hist = indicators['macd_hist'].iloc[current_idx]
            prev_hist = indicators['macd_hist'].iloc[current_idx - 1]
            current_price = indicators['price'].iloc[current_idx]
            stock_code = data.attrs.get('stock_code', 'UNKNOWN')
            
            # MACD金叉信号（买入）
            if prev_hist <= 0 and current_hist > 0:
                signal = TradingSignal(
                    timestamp=current_date,
                    stock_code=stock_code,
                    signal_type=SignalType.BUY,
                    strength=min(1.0, abs(current_hist) * 100),
                    price=current_price,
                    reason=f"MACD金叉，柱状图: {current_hist:.4f}",
                    metadata={
                        'macd': indicators['macd'].iloc[current_idx],
                        'macd_signal': indicators['macd_signal'].iloc[current_idx],
                        'macd_hist': current_hist
                    }
                )
                signals.append(signal)
            
            # MACD死叉信号（卖出）
            elif prev_hist >= 0 and current_hist < 0:
                signal = TradingSignal(
                    timestamp=current_date,
                    stock_code=stock_code,
                    signal_type=SignalType.SELL,
                    strength=min(1.0, abs(current_hist) * 100),
                    price=current_price,
                    reason=f"MACD死叉，柱状图: {current_hist:.4f}",
                    metadata={
                        'macd': indicators['macd'].iloc[current_idx],
                        'macd_signal': indicators['macd_signal'].iloc[current_idx],
                        'macd_hist': current_hist
                    }
                )
                signals.append(signal)
            
            return signals
            
        except Exception as e:
            logger.error(f"MACD策略信号生成失败: {e}")
            return []


class StrategyFactory:
    """策略工厂"""
    
    _strategies = {
        'moving_average': MovingAverageStrategy,
        'rsi': RSIStrategy,
        'macd': MACDStrategy
    }
    
    @classmethod
    def create_strategy(cls, strategy_name: str, config: Dict[str, Any]) -> BaseStrategy:
        """创建策略实例"""
        strategy_class = cls._strategies.get(strategy_name.lower())
        if not strategy_class:
            raise TaskError(
                message=f"未知的策略类型: {strategy_name}",
                severity=ErrorSeverity.MEDIUM
            )
        
        return strategy_class(config)
    
    @classmethod
    def get_available_strategies(cls) -> List[str]:
        """获取可用策略列表"""
        return list(cls._strategies.keys())
    
    @classmethod
    def register_strategy(cls, name: str, strategy_class: type):
        """注册新策略"""
        cls._strategies[name.lower()] = strategy_class


class PortfolioManager:
    """组合管理器"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.cash = config.initial_cash
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.portfolio_history: List[Dict[str, Any]] = []
        self.trade_counter = 0
    
    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """计算组合总价值"""
        total_value = self.cash
        
        for stock_code, position in self.positions.items():
            if stock_code in current_prices:
                position.current_price = current_prices[stock_code]
                position.market_value = position.quantity * position.current_price
                position.unrealized_pnl = position.market_value - (position.quantity * position.avg_cost)
                total_value += position.market_value
        
        return total_value
    
    def execute_signal(self, signal: TradingSignal, current_prices: Dict[str, float]) -> Optional[Trade]:
        """执行交易信号"""
        try:
            stock_code = signal.stock_code
            current_price = current_prices.get(stock_code, signal.price)
            
            # 应用滑点
            if signal.signal_type == SignalType.BUY:
                execution_price = current_price * (1 + self.config.slippage_rate)
            else:
                execution_price = current_price * (1 - self.config.slippage_rate)
            
            if signal.signal_type == SignalType.BUY:
                return self._execute_buy(stock_code, execution_price, signal)
            elif signal.signal_type == SignalType.SELL:
                return self._execute_sell(stock_code, execution_price, signal)
            
            return None
            
        except Exception as e:
            logger.error(f"执行交易信号失败: {signal.stock_code}, {e}")
            return None
    
    def _execute_buy(self, stock_code: str, price: float, signal: TradingSignal) -> Optional[Trade]:
        """执行买入"""
        # 计算可买数量
        portfolio_value = self.get_portfolio_value({stock_code: price})
        max_position_value = portfolio_value * self.config.max_position_size
        
        current_position = self.positions.get(stock_code)
        current_position_value = current_position.market_value if current_position else 0
        
        available_cash_for_stock = max_position_value - current_position_value
        available_cash_for_stock = min(available_cash_for_stock, self.cash * 0.95)  # 保留5%现金
        
        if available_cash_for_stock <= 0:
            return None
        
        # 计算购买数量（假设最小交易单位为100股）
        quantity = int(available_cash_for_stock / price / 100) * 100
        if quantity <= 0:
            return None
        
        # 计算实际成本
        total_cost = quantity * price
        commission = total_cost * self.config.commission_rate
        total_cost_with_commission = total_cost + commission
        
        if total_cost_with_commission > self.cash:
            return None
        
        # 执行交易
        self.cash -= total_cost_with_commission
        
        # 更新持仓
        if stock_code in self.positions:
            old_position = self.positions[stock_code]
            new_quantity = old_position.quantity + quantity
            new_avg_cost = ((old_position.quantity * old_position.avg_cost) + total_cost) / new_quantity
            
            self.positions[stock_code] = Position(
                stock_code=stock_code,
                quantity=new_quantity,
                avg_cost=new_avg_cost,
                current_price=price,
                market_value=new_quantity * price,
                unrealized_pnl=0,
                realized_pnl=old_position.realized_pnl
            )
        else:
            self.positions[stock_code] = Position(
                stock_code=stock_code,
                quantity=quantity,
                avg_cost=price,
                current_price=price,
                market_value=quantity * price,
                unrealized_pnl=0,
                realized_pnl=0
            )
        
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
            pnl=0
        )
        
        self.trades.append(trade)
        logger.info(f"执行买入: {stock_code}, 数量: {quantity}, 价格: {price:.2f}, 手续费: {commission:.2f}")
        
        return trade
    
    def _execute_sell(self, stock_code: str, price: float, signal: TradingSignal) -> Optional[Trade]:
        """执行卖出"""
        if stock_code not in self.positions:
            return None
        
        position = self.positions[stock_code]
        if position.quantity <= 0:
            return None
        
        # 卖出全部持仓
        quantity = position.quantity
        total_proceeds = quantity * price
        commission = total_proceeds * self.config.commission_rate
        net_proceeds = total_proceeds - commission
        
        # 计算盈亏
        cost_basis = quantity * position.avg_cost
        pnl = net_proceeds - cost_basis
        
        # 执行交易
        self.cash += net_proceeds
        
        # 更新持仓
        position.realized_pnl += pnl
        del self.positions[stock_code]
        
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
            pnl=pnl
        )
        
        self.trades.append(trade)
        logger.info(f"执行卖出: {stock_code}, 数量: {quantity}, 价格: {price:.2f}, 盈亏: {pnl:.2f}")
        
        return trade
    
    def record_portfolio_snapshot(self, date: datetime, current_prices: Dict[str, float]):
        """记录组合快照"""
        portfolio_value = self.get_portfolio_value(current_prices)
        
        snapshot = {
            'date': date,
            'cash': self.cash,
            'portfolio_value': portfolio_value,
            'positions': {code: {
                'quantity': pos.quantity,
                'avg_cost': pos.avg_cost,
                'current_price': pos.current_price,
                'market_value': pos.market_value,
                'unrealized_pnl': pos.unrealized_pnl
            } for code, pos in self.positions.items()},
            'total_trades': len(self.trades)
        }
        
        self.portfolio_history.append(snapshot)
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """计算绩效指标"""
        if not self.portfolio_history:
            return {}
        
        # 计算收益序列
        values = [snapshot['portfolio_value'] for snapshot in self.portfolio_history]
        returns = pd.Series(values).pct_change().dropna()
        
        if len(returns) == 0:
            return {}
        
        # 基础指标
        total_return = (values[-1] - self.config.initial_cash) / self.config.initial_cash
        
        # 年化收益率
        days = (self.portfolio_history[-1]['date'] - self.portfolio_history[0]['date']).days
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
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl < 0]
        
        win_rate = len(winning_trades) / len(self.trades) if self.trades else 0
        
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': len(self.trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades)
        }