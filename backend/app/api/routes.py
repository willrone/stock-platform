"""
API路由定义（已废弃）

此文件已废弃，所有路由已按模块拆分到 app/api/v1/ 目录下：
- health.py: 健康检查
- stocks.py: 股票数据
- predictions.py: 预测服务
- tasks.py: 任务管理
- models.py: 模型管理
- backtest.py: 回测服务
- data.py: 数据管理
- monitoring.py: 监控服务
- system.py: 系统状态

请使用 app.api.v1.api.api_router 替代此文件中的 api_router
"""

from fastapi import APIRouter

# 为了向后兼容，从新的模块化路由导入
from app.api.v1.api import api_router

# 导出api_router以保持向后兼容
__all__ = ['api_router']
