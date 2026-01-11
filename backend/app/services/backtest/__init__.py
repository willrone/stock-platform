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

新增高级策略（从 strategies.py 导入）：
- 技术分析策略: BollingerBandStrategy, StochasticStrategy, CCIStrategy
- 统计套利策略: PairsTradingStrategy, MeanReversionStrategy, CointegrationStrategy
- 因子投资策略: ValueFactorStrategy, MomentumFactorStrategy, LowVolatilityStrategy, MultiFactorStrategy
- AdvancedStrategyFactory: 高级策略工厂
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

# 高级策略
from .strategies import (
    # 技术分析策略
    BollingerBandStrategy,
    StochasticStrategy,
    CCIStrategy,
    
    # 统计套利策略
    PairsTradingStrategy,
    MeanReversionStrategy,
    CointegrationStrategy,
    
    # 因子投资策略
    ValueFactorStrategy,
    MomentumFactorStrategy,
    LowVolatilityStrategy,
    MultiFactorStrategy,
    
    # 高级策略工厂
    AdvancedStrategyFactory
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
    'DataLoader',
    
    # 技术分析策略
    'BollingerBandStrategy',
    'StochasticStrategy',
    'CCIStrategy',
    
    # 统计套利策略
    'PairsTradingStrategy',
    'MeanReversionStrategy',
    'CointegrationStrategy',
    
    # 因子投资策略
    'ValueFactorStrategy',
    'MomentumFactorStrategy',
    'LowVolatilityStrategy',
    'MultiFactorStrategy',
    
    # 高级策略工厂
    'AdvancedStrategyFactory'
]
