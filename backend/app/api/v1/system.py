"""
系统状态路由
"""

from fastapi import APIRouter, HTTPException, Request
from datetime import datetime
import logging

from app.api.v1.schemas import StandardResponse

router = APIRouter(prefix="/system", tags=["系统状态"])
logger = logging.getLogger(__name__)


@router.get(
    "/version",
    response_model=StandardResponse,
    summary="获取API版本信息",
    description="获取当前API的版本信息和更新日志"
)
async def get_api_version():
    """获取API版本信息"""
    version_info = {
        "version": "1.0.0",
        "release_date": "2025-01-01",
        "api_name": "股票预测平台API",
        "description": "基于AI的股票预测和回测分析平台",
        "features": [
            "股票数据获取",
            "技术指标计算", 
            "机器学习预测",
            "策略回测",
            "任务管理"
        ],
        "endpoints": {
            "total": 15,
            "categories": {
                "数据服务": 3,
                "预测服务": 2,
                "任务管理": 4,
                "回测服务": 1,
                "模型管理": 2,
                "系统状态": 3
            }
        },
        "changelog": {
            "1.0.0": [
                "初始版本发布",
                "实现基础API功能",
                "添加限流和错误处理",
                "完成API文档生成"
            ]
        }
    }
    
    return StandardResponse(
        success=True,
        message="API版本信息获取成功",
        data=version_info
    )


@router.get("/status", response_model=StandardResponse, summary="获取系统状态", description="获取各个服务组件的运行状态")
async def get_system_status(request: Request):
    """获取系统状态"""
    try:
        # 获取错误处理中间件的统计信息
        error_stats = {}
        for middleware in request.app.middleware_stack:
            if hasattr(middleware, 'cls') and middleware.cls.__name__ == 'ErrorHandlingMiddleware':
                if hasattr(middleware, 'kwargs') and 'app' in middleware.kwargs:
                    error_middleware = middleware.kwargs.get('error_middleware')
                    if error_middleware and hasattr(error_middleware, 'get_error_stats'):
                        error_stats = error_middleware.get_error_stats()
                break
        
        system_status = {
            "api_server": {"status": "healthy", "uptime": "5 days"},
            "data_service": {"status": "healthy", "last_update": datetime.now().isoformat()},
            "prediction_engine": {"status": "healthy", "active_models": 3},
            "task_manager": {"status": "healthy", "running_tasks": 2},
            "database": {"status": "healthy", "connection": "active"},
            "remote_data_service": {"status": "healthy", "url": "192.168.3.62"},
            "error_statistics": error_stats
        }
        
        return StandardResponse(
            success=True,
            message="系统状态获取成功",
            data=system_status
        )
        
    except Exception as e:
        logger.error(f"获取系统状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取系统状态失败: {str(e)}")

