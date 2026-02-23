"""
因子缓存模块

提供因子计算结果的缓存功能，支持内存缓存和磁盘缓存两层架构。
"""

from .factor_cache import FactorCache

__all__ = ["FactorCache"]
