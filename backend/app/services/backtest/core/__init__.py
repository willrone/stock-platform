"""
回测核心模块

包含回测引擎的基础类和核心功能
"""

# 从 models 导入数据模型
from ..models import (
    BacktestConfig,
    OrderType,
    Position,
    SignalType,
    Trade,
    TradingSignal,
)
from ..strategies.strategy_factory import StrategyFactory

# 从策略模块导入（策略已移动到strategies目录）
from ..strategies.technical.basic_strategies import (
    MACDStrategy,
    MovingAverageStrategy,
    RSIStrategy,
)

# 从各个模块导入
from .base_strategy import BaseStrategy
from .portfolio_manager import PortfolioManager
from .risk_manager import RiskManager, PositionPriceInfo
from .strategy_portfolio import StrategyPortfolio

__all__ = [
    "SignalType",
    "OrderType",
    "TradingSignal",
    "Trade",
    "Position",
    "BacktestConfig",
    "BaseStrategy",
    "MovingAverageStrategy",
    "RSIStrategy",
    "MACDStrategy",
    "StrategyFactory",
    "PortfolioManager",
    "RiskManager",
    "PositionPriceInfo",
    "StrategyPortfolio",
]
