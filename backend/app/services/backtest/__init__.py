"""
回测引擎模块

该模块包含所有与策略回测和执行相关的服务，包括：
- 回测引擎核心功能
- 回测执行器和数据加载
- 交易策略实现
- 组合管理和风险控制

主要组件：
- BacktestEngine: 回测引擎核心（从 backtest_engine.py 导入）
- BacktestExecutor: 回测执行器
- DataLoader: 数据加载器
- 策略类: BaseStrategy, MovingAverageStrategy, RSIStrategy, MACDStrategy
- PortfolioManager: 组合管理器
"""

# 回测引擎
from .backtest_engine import (
    # 枚举类型
    SignalType,
    OrderType,
    
    # 数据类
    TradingSignal,
    Trade,
    Position,
    BacktestConfig,
    
    # 策略类
    BaseStrategy,
    MovingAverageStrategy,
    RSIStrategy,
    MACDStrategy,
    StrategyFactory,
    
    # 管理器
    PortfolioManager
)

# 回测执行器
from .backtest_executor import (
    BacktestExecutor,
    DataLoader
)

__all__ = [
    # 枚举类型
    'SignalType',
    'OrderType',
    
    # 数据类
    'TradingSignal',
    'Trade',
    'Position',
    'BacktestConfig',
    
    # 策略类
    'BaseStrategy',
    'MovingAverageStrategy',
    'RSIStrategy',
    'MACDStrategy',
    'StrategyFactory',
    
    # 管理器
    'PortfolioManager',
    
    # 执行器
    'BacktestExecutor',
    'DataLoader'
]