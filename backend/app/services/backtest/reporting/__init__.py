"""
报告模块

包含报告生成和图表数据生成功能
"""

from .chart_data_generator import ChartDataGenerator
from .report_generator import BacktestReportGenerator

__all__ = ["BacktestReportGenerator", "ChartDataGenerator"]
