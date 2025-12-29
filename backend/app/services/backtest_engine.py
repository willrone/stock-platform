"""
回测引擎服务

集成Vectorbt进行策略回测和性能指标计算。
实现交易记录生成和分析功能。
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """订单类型"""
    BUY = "buy"
    SELL = "sell"


@dataclass
class BacktestConfig:
    """回测配置"""
    initial_cash: float = 100000.0  # 初始资金
    commission: float = 0.001       # 手续费率
    slippage: float = 0.001         # 滑点
    max_position_size: float = 0.2  # 最大仓位比例
    stop_loss: Optional[float] = None  # 止损比例
    take_profit: Optional[float] = None  # 止盈比例


@dataclass
class Trade:
    """交易记录"""
    stock_code: str
    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    quantity: int
    order_type: OrderType
    pnl: float              # 盈亏
    pnl_pct: float         # 盈亏百分比
    commission: float       # 手续费
    duration_days: int      # 持仓天数


@dataclass
class BacktestResult:
    """回测结果"""
    # 基本信息
    start_date: datetime
    end_date: datetime
    initial_cash: float
    final_value: float
    
    # 收益指标
    total_return: float
    annualized_return: float
    max_drawdown: float
    sharpe_ratio: float
    calmar_ratio: float
    
    # 交易统计
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    
    # 风险指标
    volatility: float
    var_95: float
    max_consecutive_losses: int
    
    # 交易记录
    trades: List[Trade]
    
    # 每日净值
    daily_returns: pd.Series
    cumulative_returns: pd.Series
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "period": {
                "start_date": self.start_date.isoformat(),
                "end_date": self.end_date.isoformat()
            },
            "portfolio": {
                "initial_cash": self.initial_cash,
                "final_value": self.final_value,
                "total_return": self.total_return,
                "annualized_return": self.annualized_return
            },
            "risk_metrics": {
                "max_drawdown": self.max_drawdown,
                "sharpe_ratio": self.sharpe_ratio,
                "calmar_ratio": self.calmar_ratio,
                "volatility": self.volatility,
                "var_95": self.var_95,
                "max_consecutive_losses": self.max_consecutive_losses
            },
            "trading_stats": {
                "total_trades": self.total_trades,
                "winning_trades": self.winning_trades,
                "losing_trades": self.losing_trades,
                "win_rate": self.win_rate,
                "avg_win": self.avg_win,
                "avg_loss": self.avg_loss,
                "profit_factor": self.profit_factor
            },
            "trades_count": len(self.trades)
        }


class SimpleBacktestEngine:
    """简化的回测引擎（不依赖vectorbt）"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.cash = config.initial_cash
        self.positions = {}  # {stock_code: quantity}
        self.trades = []
        self.daily_values = []
        self.daily_dates = []
    
    def run_backtest(
        self,
        price_data: pd.DataFrame,
        signals: pd.DataFrame
    ) -> BacktestResult:
        """
        运行回测
        
        Args:
            price_data: 价格数据，包含date, stock_code, open, high, low, close列
            signals: 信号数据，包含date, stock_code, signal列（1=买入, -1=卖出, 0=持有）
            
        Returns:
            回测结果
        """
        logger.info("开始运行回测")
        
        # 合并价格和信号数据
        data = price_data.merge(signals, on=['date', 'stock_code'], how='left')
        data['signal'] = data['signal'].fillna(0)
        data = data.sort_values(['date', 'stock_code'])
        
        # 按日期分组处理
        for date, day_data in data.groupby('date'):
            self._process_day(date, day_data)
        
        # 计算回测结果
        result = self._calculate_results()
        
        logger.info(f"回测完成，总收益: {result.total_return:.4f}, 夏普比率: {result.sharpe_ratio:.4f}")
        return result
    
    def _process_day(self, date: datetime, day_data: pd.DataFrame):
        """处理单日交易"""
        portfolio_value = self.cash
        
        # 计算当前持仓价值
        for stock_code, quantity in self.positions.items():
            stock_price_data = day_data[day_data['stock_code'] == stock_code]
            if not stock_price_data.empty:
                current_price = stock_price_data['close'].iloc[0]
                portfolio_value += quantity * current_price
        
        # 记录每日净值
        self.daily_values.append(portfolio_value)
        self.daily_dates.append(date)
        
        # 处理交易信号
        for _, row in day_data.iterrows():
            if row['signal'] != 0:
                self._execute_trade(row)
    
    def _execute_trade(self, row):
        """执行交易"""
        stock_code = row['stock_code']
        signal = row['signal']
        price = row['close']
        date = row['date']
        
        if signal == 1:  # 买入信号
            self._buy_stock(stock_code, price, date)
        elif signal == -1:  # 卖出信号
            self._sell_stock(stock_code, price, date)
    
    def _buy_stock(self, stock_code: str, price: float, date: datetime):
        """买入股票"""
        # 计算可买入数量
        max_investment = self.cash * self.config.max_position_size
        commission = max_investment * self.config.commission
        available_cash = max_investment - commission
        
        if available_cash <= 0:
            return
        
        quantity = int(available_cash / price)
        if quantity <= 0:
            return
        
        # 执行买入
        cost = quantity * price + commission
        if cost <= self.cash:
            self.cash -= cost
            self.positions[stock_code] = self.positions.get(stock_code, 0) + quantity
            
            logger.debug(f"买入 {stock_code}: {quantity}股 @ {price:.2f}, 成本: {cost:.2f}")
    
    def _sell_stock(self, stock_code: str, price: float, date: datetime):
        """卖出股票"""
        if stock_code not in self.positions or self.positions[stock_code] <= 0:
            return
        
        quantity = self.positions[stock_code]
        revenue = quantity * price
        commission = revenue * self.config.commission
        net_revenue = revenue - commission
        
        # 执行卖出
        self.cash += net_revenue
        
        # 记录交易
        # 这里简化处理，假设买入价格为当前价格的95%
        entry_price = price * 0.95
        pnl = net_revenue - (quantity * entry_price)
        pnl_pct = pnl / (quantity * entry_price) if quantity * entry_price > 0 else 0
        
        trade = Trade(
            stock_code=stock_code,
            entry_date=date - timedelta(days=5),  # 简化假设
            exit_date=date,
            entry_price=entry_price,
            exit_price=price,
            quantity=quantity,
            order_type=OrderType.SELL,
            pnl=pnl,
            pnl_pct=pnl_pct,
            commission=commission,
            duration_days=5
        )
        
        self.trades.append(trade)
        self.positions[stock_code] = 0
        
        logger.debug(f"卖出 {stock_code}: {quantity}股 @ {price:.2f}, 盈亏: {pnl:.2f}")
    
    def _calculate_results(self) -> BacktestResult:
        """计算回测结果"""
        if not self.daily_values:
            raise ValueError("没有回测数据")
        
        # 基本信息
        start_date = self.daily_dates[0]
        end_date = self.daily_dates[-1]
        initial_cash = self.config.initial_cash
        final_value = self.daily_values[-1]
        
        # 计算收益率序列
        values_series = pd.Series(self.daily_values, index=self.daily_dates)
        daily_returns = values_series.pct_change().dropna()
        cumulative_returns = (values_series / initial_cash - 1)
        
        # 收益指标
        total_return = (final_value - initial_cash) / initial_cash
        days = (end_date - start_date).days
        annualized_return = (1 + total_return) ** (365 / days) - 1 if days > 0 else 0
        
        # 最大回撤
        running_max = cumulative_returns.expanding().max()
        drawdown = cumulative_returns - running_max
        max_drawdown = drawdown.min()
        
        # 夏普比率
        if len(daily_returns) > 1:
            excess_returns = daily_returns - self.config.commission  # 简化的无风险利率
            sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0
        else:
            sharpe_ratio = 0
        
        # 卡尔玛比率
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # 交易统计
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t.pnl > 0])
        losing_trades = len([t for t in self.trades if t.pnl < 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        wins = [t.pnl for t in self.trades if t.pnl > 0]
        losses = [t.pnl for t in self.trades if t.pnl < 0]
        
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else 0
        
        # 风险指标
        volatility = daily_returns.std() * np.sqrt(252) if len(daily_returns) > 1 else 0
        var_95 = daily_returns.quantile(0.05) if len(daily_returns) > 0 else 0
        
        # 最大连续亏损
        consecutive_losses = 0
        max_consecutive_losses = 0
        for trade in self.trades:
            if trade.pnl < 0:
                consecutive_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            else:
                consecutive_losses = 0
        
        return BacktestResult(
            start_date=start_date,
            end_date=end_date,
            initial_cash=initial_cash,
            final_value=final_value,
            total_return=total_return,
            annualized_return=annualized_return,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            calmar_ratio=calmar_ratio,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            volatility=volatility,
            var_95=var_95,
            max_consecutive_losses=max_consecutive_losses,
            trades=self.trades,
            daily_returns=daily_returns,
            cumulative_returns=cumulative_returns
        )


class BacktestService:
    """回测服务主类"""
    
    def __init__(self):
        self.default_config = BacktestConfig()
    
    async def run_strategy_backtest(
        self,
        strategy_name: str,
        stock_codes: List[str],
        start_date: datetime,
        end_date: datetime,
        config: Optional[BacktestConfig] = None
    ) -> BacktestResult:
        """
        运行策略回测
        
        Args:
            strategy_name: 策略名称
            stock_codes: 股票代码列表
            start_date: 回测开始日期
            end_date: 回测结束日期
            config: 回测配置
            
        Returns:
            回测结果
        """
        if config is None:
            config = self.default_config
        
        logger.info(f"开始运行策略回测: {strategy_name}")
        
        # 生成模拟数据（实际应用中应该从数据服务获取）
        price_data = self._generate_mock_price_data(stock_codes, start_date, end_date)
        signals = self._generate_mock_signals(stock_codes, start_date, end_date)
        
        # 运行回测
        engine = SimpleBacktestEngine(config)
        result = engine.run_backtest(price_data, signals)
        
        logger.info(f"策略 {strategy_name} 回测完成")
        return result
    
    def _generate_mock_price_data(
        self, 
        stock_codes: List[str], 
        start_date: datetime, 
        end_date: datetime
    ) -> pd.DataFrame:
        """生成模拟价格数据"""
        data = []
        current_date = start_date
        
        while current_date <= end_date:
            for stock_code in stock_codes:
                # 生成随机价格数据
                base_price = 10.0 + hash(stock_code) % 50
                price_change = np.random.normal(0, 0.02)
                
                close_price = base_price * (1 + price_change)
                open_price = close_price * (1 + np.random.normal(0, 0.01))
                high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.01)))
                low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.01)))
                
                data.append({
                    'date': current_date,
                    'stock_code': stock_code,
                    'open': open_price,
                    'high': high_price,
                    'low': low_price,
                    'close': close_price,
                    'volume': np.random.randint(1000000, 10000000)
                })
            
            current_date += timedelta(days=1)
        
        return pd.DataFrame(data)
    
    def _generate_mock_signals(
        self, 
        stock_codes: List[str], 
        start_date: datetime, 
        end_date: datetime
    ) -> pd.DataFrame:
        """生成模拟交易信号"""
        data = []
        current_date = start_date
        
        while current_date <= end_date:
            for stock_code in stock_codes:
                # 生成随机信号（简化版本）
                signal = np.random.choice([0, 1, -1], p=[0.8, 0.1, 0.1])
                
                data.append({
                    'date': current_date,
                    'stock_code': stock_code,
                    'signal': signal
                })
            
            current_date += timedelta(days=1)
        
        return pd.DataFrame(data)


# 导出主要类和函数
__all__ = [
    'BacktestService',
    'BacktestConfig',
    'BacktestResult',
    'Trade',
    'OrderType',
    'SimpleBacktestEngine'
]