"""
报告模块

包含报告生成和图表数据生成功能
"""

from .report_generator import BacktestReportGenerator
from .chart_data_generator import ChartDataGenerator

__all__ = [
    'BacktestReportGenerator',
    'ChartDataGenerator'
]
