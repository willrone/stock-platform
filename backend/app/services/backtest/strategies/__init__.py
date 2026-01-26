"""
策略模块

包含所有交易策略实现
"""

# 高级策略
from .strategies import (  # 技术分析策略; 统计套利策略; 因子投资策略; 高级策略工厂（向后兼容）
    AdvancedStrategyFactory,
    BollingerBandStrategy,
    CCIStrategy,
    CointegrationStrategy,
    FactorStrategy,
    LowVolatilityStrategy,
    MeanReversionStrategy,
    MomentumFactorStrategy,
    MultiFactorStrategy,
    PairsTradingStrategy,
    StatisticalArbitrageStrategy,
    StochasticStrategy,
    ValueFactorStrategy,
)

# 统一的策略工厂
from .strategy_factory import StrategyFactory

# 基础技术分析策略
from .technical.basic_strategies import MACDStrategy, MovingAverageStrategy, RSIStrategy

__all__ = [
    # 基础技术分析策略
    "MovingAverageStrategy",
    "RSIStrategy",
    "MACDStrategy",
    # 高级技术分析策略
    "BollingerBandStrategy",
    "StochasticStrategy",
    "CCIStrategy",
    # 统计套利策略
    "StatisticalArbitrageStrategy",
    "PairsTradingStrategy",
    "MeanReversionStrategy",
    "CointegrationStrategy",
    # 因子投资策略
    "FactorStrategy",
    "ValueFactorStrategy",
    "MomentumFactorStrategy",
    "LowVolatilityStrategy",
    "MultiFactorStrategy",
    # 策略工厂
    "StrategyFactory",
    "AdvancedStrategyFactory",  # 向后兼容别名
]
