"""
API v1 路由聚合器

将所有API路由模块聚合到一个路由器中

路由安全策略：
- Auth 路由（/auth/*）：公开，无需认证
- Health 路由（/health/*）：公开，健康检查
- 其余所有业务路由：需要 JWT 认证（Depends(get_current_user)）
"""

from fastapi import APIRouter, Depends

from app.api.v1.dependencies import get_current_user

# 导入各个模块的路由
from app.api.v1 import (
    auth,
    backtest,
    backtest_detailed,
    backtest_websocket,
    billing,
    commerce,
    dashboard,
    data,
    data_versioning,
    features,
    files,
    health,
    infrastructure,
    laya,
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

# 创建API v1路由器（顶层，不含全局认证依赖）
api_router = APIRouter()

# ── 公开路由（无需认证）──
# Auth 路由：注册/登录/获取用户信息（自带认证逻辑）
api_router.include_router(auth.router)
api_router.include_router(health.router)
api_router.include_router(billing.public_router)
api_router.include_router(billing.webhook_router)

# ── 受保护路由（需要 JWT 认证）──
protected_router = APIRouter(dependencies=[Depends(get_current_user)])

protected_router.include_router(stocks.router)
protected_router.include_router(predictions.router)
protected_router.include_router(tasks.router)
protected_router.include_router(models.router)
protected_router.include_router(backtest.router)
protected_router.include_router(backtest_detailed.router)
protected_router.include_router(backtest_websocket.router)
protected_router.include_router(data.router)
protected_router.include_router(system.router)
protected_router.include_router(qlib.router)
protected_router.include_router(infrastructure.router)
protected_router.include_router(data_versioning.router)
protected_router.include_router(features.router)
protected_router.include_router(training_progress.router)
protected_router.include_router(monitoring.router)
protected_router.include_router(files.router)
protected_router.include_router(strategy_configs.router)
protected_router.include_router(optimization.router)
protected_router.include_router(signals.router)
protected_router.include_router(dashboard.router)
protected_router.include_router(billing.router)
protected_router.include_router(commerce.router)
protected_router.include_router(laya.router)

# 将受保护路由挂载到顶层路由
api_router.include_router(protected_router)
