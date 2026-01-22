"""
工具模块

包含数据适配器、性能分析器等工具类
"""

from .backtest_data_adapter import BacktestDataAdapter
from .signal_integrator import SignalIntegrator
from ..models.analysis_models import (
    ExtendedRiskMetrics,
    MonthlyReturnsAnalysis,
    PositionAnalysis,
    EnhancedPositionAnalysis,
    DrawdownAnalysis,
    EnhancedBacktestResult
)

# 性能分析器（可选）
try:
    from .performance_profiler import BacktestPerformanceProfiler, PerformanceContext
    __all__ = [
        'BacktestDataAdapter',
        'SignalIntegrator',
        'ExtendedRiskMetrics',
        'MonthlyReturnsAnalysis',
        'PositionAnalysis',
        'EnhancedPositionAnalysis',
        'DrawdownAnalysis',
        'EnhancedBacktestResult',
        'BacktestPerformanceProfiler',
        'PerformanceContext'
    ]
except ImportError:
    __all__ = [
        'BacktestDataAdapter',
        'SignalIntegrator',
        'ExtendedRiskMetrics',
        'MonthlyReturnsAnalysis',
        'PositionAnalysis',
        'EnhancedPositionAnalysis',
        'DrawdownAnalysis',
        'EnhancedBacktestResult'
    ]
