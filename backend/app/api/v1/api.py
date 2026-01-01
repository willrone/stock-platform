"""
API v1 路由聚合器

将所有API路由模块聚合到一个路由器中
"""

from fastapi import APIRouter

# 导入各个模块的路由
from app.api.v1 import health, stocks, predictions, tasks, models, backtest, data, monitoring, system

# 创建API v1路由器
api_router = APIRouter()

# 包含所有模块路由
api_router.include_router(health.router)
api_router.include_router(stocks.router)
api_router.include_router(predictions.router)
api_router.include_router(tasks.router)
api_router.include_router(models.router)
api_router.include_router(backtest.router)
api_router.include_router(data.router)
api_router.include_router(monitoring.router)
api_router.include_router(system.router)