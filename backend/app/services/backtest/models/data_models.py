"""
数据模型定义

包含回测过程中使用的核心数据类
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

from .enums import SignalType


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
    slippage_cost: float = 0.0  # 滑点成本
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
    slippage_rate: float = 0.001  # 滑点率
    max_position_size: float = 0.2  # 最大单股持仓比例
    stop_loss_pct: float = 0.05  # 止损比例
    take_profit_pct: float = 0.15  # 止盈比例
    rebalance_frequency: str = "daily"  # 调仓频率

    # P0-2: 最大回撤熔断（None 或 0 表示不启用）
    max_drawdown_pct: Optional[float] = None

    # 不限制买入模式：启用时忽略单股仓位限制、5%现金保留，资金不足时自动补充
    enable_unlimited_buy: bool = False

    # 性能：大规模回测默认不需要每天的完整组合快照；
    # equity 曲线单独记录，不受该开关影响。
    record_portfolio_history: bool = True
    portfolio_history_stride: int = 5  # 性能优化: 默认每5天记录一次快照，减少内存和I/O开销
    record_positions_in_history: bool = True  # False 时快照不包含 positions 明细
