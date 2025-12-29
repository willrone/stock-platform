"""
API v1 路由聚合器

将所有API路由聚合到一个路由器中
"""

from fastapi import APIRouter

from app.api.routes import api_router as routes_router

# 创建API v1路由器
api_router = APIRouter()

# 包含所有路由
api_router.include_router(routes_router, tags=["股票预测平台API"])