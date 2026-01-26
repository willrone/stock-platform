"""
API v1 路由聚合器

将所有API路由模块聚合到一个路由器中
"""

from fastapi import APIRouter

# 导入各个模块的路由
from app.api.v1 import (
    backtest,
    backtest_detailed,
    backtest_websocket,
    data,
    data_versioning,
    features,
    files,
    health,
    infrastructure,
    models,
    monitoring,
    optimization,
    predictions,
    qlib,
    signals,
    stocks,
    strategy_configs,
    system,
    tasks,
    training_progress,
)

# 创建API v1路由器
api_router = APIRouter()

# 包含所有模块路由
api_router.include_router(health.router)
api_router.include_router(stocks.router)
api_router.include_router(predictions.router)
api_router.include_router(tasks.router)
api_router.include_router(models.router)
api_router.include_router(backtest.router)
api_router.include_router(backtest_detailed.router)
api_router.include_router(backtest_websocket.router)
api_router.include_router(data.router)
api_router.include_router(system.router)
api_router.include_router(qlib.router)
api_router.include_router(infrastructure.router)
api_router.include_router(data_versioning.router)
api_router.include_router(features.router)
api_router.include_router(training_progress.router)
api_router.include_router(monitoring.router)
api_router.include_router(files.router)
api_router.include_router(strategy_configs.router)
api_router.include_router(optimization.router)
api_router.include_router(signals.router)
