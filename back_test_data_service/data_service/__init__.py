"""
股票数据服务模块
独立运行的数据服务，提供股票数据获取能力

功能:
- 从Tushare获取股票数据
- 存储到Parquet文件（高效列式存储）
- 提供RESTful API接口
- 定时任务自动更新数据
"""

__version__ = '2.0.0'
__author__ = 'Stock Data Service'
__description__ = '独立运行的股票数据服务'

