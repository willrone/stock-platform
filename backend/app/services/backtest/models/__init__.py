"""
数据模型模块

包含所有回测相关的数据模型和枚举类型
"""

from .enums import SignalType, OrderType
from .data_models import (
    TradingSignal,
    Trade,
    Position,
    BacktestConfig
)
from .analysis_models import (
    ExtendedRiskMetrics,
    MonthlyReturnsAnalysis,
    PositionAnalysis,
    EnhancedPositionAnalysis,
    DrawdownAnalysis,
    EnhancedBacktestResult
)

__all__ = [
    # 枚举类型
    'SignalType',
    'OrderType',
    
    # 核心数据模型
    'TradingSignal',
    'Trade',
    'Position',
    'BacktestConfig',
    
    # 分析数据模型
    'ExtendedRiskMetrics',
    'MonthlyReturnsAnalysis',
    'PositionAnalysis',
    'EnhancedPositionAnalysis',
    'DrawdownAnalysis',
    'EnhancedBacktestResult'
]
