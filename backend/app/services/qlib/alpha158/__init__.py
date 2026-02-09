"""
Alpha158 因子计算模块

提供 Alpha158 因子的计算功能，包括：
- Alpha158Calculator: 主计算器类
- QlibExpressionParser: Qlib 表达式解析器
- 简化因子计算函数
"""

from .calculator import Alpha158Calculator
from .expression_parser import QlibExpressionParser
from .simplified_factors import calculate_simplified_alpha_factors

__all__ = [
    "Alpha158Calculator",
    "QlibExpressionParser",
    "calculate_simplified_alpha_factors",
]
