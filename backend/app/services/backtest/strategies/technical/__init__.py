"""
技术分析策略模块

包含基础技术分析策略实现
"""

from .basic_strategies import MACDStrategy, MovingAverageStrategy
from .rsi_optimized import RSIOptimizedStrategy

# 向后兼容：RSIStrategy 指向优化版
RSIStrategy = RSIOptimizedStrategy

__all__ = ["MovingAverageStrategy", "RSIStrategy", "RSIOptimizedStrategy", "MACDStrategy"]
