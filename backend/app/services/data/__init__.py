"""
数据管理模块

该模块包含简化的数据服务，只提供基本的HTTP接口调用功能。

组件:
- SimpleDataService: 简化的数据服务（只提供连接状态检查和数据获取）
"""

from .simple_data_service import SimpleDataService

__all__ = [
    "SimpleDataService",
]
