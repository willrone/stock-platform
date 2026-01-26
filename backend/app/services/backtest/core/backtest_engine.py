"""
回测引擎 - 兼容性入口

此文件保留作为向后兼容的入口点，实际实现已拆分到各个模块：
- models/: 数据模型和枚举
- base_strategy.py: 策略基类
- portfolio_manager.py: 组合管理器
- strategies/: 所有策略实现（基础策略在 strategies/technical/，工厂在 strategies/strategy_factory.py）
"""

# 从 models 导入数据模型和枚举
from ..models import (
    BacktestConfig,
    OrderType,
    Position,
    SignalType,
    Trade,
    TradingSignal,
)
from ..strategies.strategy_factory import StrategyFactory
from ..strategies.technical.basic_strategies import (
    MACDStrategy,
    MovingAverageStrategy,
    RSIStrategy,
)

# 从策略模块导入
from .base_strategy import BaseStrategy

# 从组合管理模块导入
from .portfolio_manager import PortfolioManager

# 保持向后兼容：导出所有公共类和枚举
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
]
