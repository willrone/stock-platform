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
    open_cost: float = 0.0  # 买入成本率（官方 Qlib 风格）
    close_cost: float = 0.0  # 卖出成本率（官方 Qlib 风格）
    min_cost: float = 0.0  # 最低交易成本（官方 Qlib 风格）
    max_position_size: float = 0.2  # 最大单股持仓比例
    cash_reserve_ratio: float = 0.05  # 预留现金比例
    board_lot_size: int = 100  # 最小买入手数（A股默认100股）
    # 止损/止盈（2026-10-26 真正接线到引擎，之前是死参数）
    # 默认 0 = 关闭：3 年 5238 只股票池实测，42f 加 5% 个股止损后
    # 夏普 0.956 → 0.769、收益 +39.4% → +28.5%（只换来回撤 −12.8% → −10.6%）
    # → 紧止损对周频横截面选股是净负面（砍掉反弹），需要时显式开启。
    stop_loss_pct: float = 0.0  # 个股止损比例，0=关闭
    take_profit_pct: float = 0.0  # 个股止盈比例，0=关闭
    # ── 可成交性（2026-10-26 下沉到回测引擎）─────────────────────
    # True = 涨停买不进、跌停卖不出、停牌不可交易、流动性不足不建仓。
    # 口径与纸面盘（paper_portfolio.py）和对外信号共用
    # app/services/trading/execution_rules.py。
    # 关掉它 = 回到旧口径（回测能买涨停卖跌停，夏普虚高）。
    enable_tradability: bool = True
    #: 单日买入额占该股当日成交额的最大比例（0.5%）
    participation_rate: float = 0.005
    #: 当日成交额门槛（元），低于此值视为流动性太差
    min_turnover: float = 5_000_000.0
    # P0-3：跌停/停牌卖不出时**挂起**卖单等解封（而不是丢弃信号）
    # True = 更贴近实盘（跑不掉就是跑不掉）；关掉则卖单直接作废。
    enable_pending_sells: bool = True
    rebalance_frequency: str = "daily"  # 调仓频率

    # 性能：大规模回测默认不需要每天的完整组合快照；
    # equity 曲线单独记录，不受该开关影响。
    record_portfolio_history: bool = True
    portfolio_history_stride: int = 1  # >1 表示降采样（每 stride 天记录一次）
    record_positions_in_history: bool = True  # False 时快照不包含 positions 明细
