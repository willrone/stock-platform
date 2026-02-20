"""
回测数据持久化 — 数据契约（Pydantic 模型）

定义写入/读取接口的输入输出数据结构，
确保 tasks.result 只存精简标量摘要，不存时序数据。
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class BacktestSummary(BaseModel):
    """精简的回测摘要，存入 tasks.result（只有标量指标，不含时序数据）"""

    backtest_id: str
    strategy_name: str = ""
    stock_count: int = 0
    start_date: str = ""
    end_date: str = ""
    initial_cash: float = 0.0
    final_value: float = 0.0
    total_return: float = 0.0
    annualized_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    total_signals: int = 0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # 成本分析
    cost_statistics: Optional[Dict[str, Any]] = None
    excess_return_with_cost: Optional[Dict[str, Any]] = None
    excess_return_without_cost: Optional[Dict[str, Any]] = None


class SnapshotData(BaseModel):
    """组合快照数据"""

    date: Any  # datetime 或 str，兼容多种输入
    portfolio_value: float = 0.0
    cash: float = 0.0
    positions_count: int = 0
    total_return: float = 0.0
    drawdown: float = 0.0       # 真实计算，不再硬编码 0
    daily_return: float = 0.0   # 新增：日收益率
    positions: Optional[Dict[str, Any]] = None


class TradeData(BaseModel):
    """交易记录数据"""

    trade_id: str = ""
    stock_code: str = ""
    stock_name: Optional[str] = None
    action: str = "BUY"
    quantity: int = 0
    price: float = 0.0
    timestamp: Any = None  # datetime 或 str
    commission: float = 0.0
    pnl: Optional[float] = None
    holding_days: Optional[int] = None
    technical_indicators: Optional[Dict[str, Any]] = None


class SignalData(BaseModel):
    """信号数据"""

    signal_id: str = ""
    stock_code: str = ""
    stock_name: Optional[str] = None
    signal_type: str = "BUY"
    timestamp: Any = None
    price: float = 0.0
    strength: float = 0.0
    reason: Optional[str] = None
    executed: bool = False
    execution_reason: Optional[str] = None
