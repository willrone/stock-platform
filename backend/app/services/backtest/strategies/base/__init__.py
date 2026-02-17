"""
基础策略类模块
"""

from .factor_base import FactorStrategy
from .statistical_arbitrage_base import StatisticalArbitrageStrategy

__all__ = [
    "StatisticalArbitrageStrategy",
    "FactorStrategy",
]
