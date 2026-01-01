"""
FastAPI 应用程序入口点
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from app.api.v1.api import api_router
from app.websocket import ws_router
from app.core.config import settings
from app.core.database import init_db
from app.core.logging import setup_logging
from app.core.metrics import setup_metrics_collection, MetricsMiddleware, metrics_endpoint
from app.core.container import get_container, cleanup_container
from app.middleware.rate_limiting import RateLimitMiddleware, RateLimitConfig
from app.middleware.error_handling import ErrorHandlingMiddleware, RequestLoggingMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """应用程序生命周期管理"""
    # 启动时初始化
    setup_logging()
    await init_db()
    setup_metrics_collection()
    
    # 初始化服务容器
    await get_container()
    
    yield
    
    # 关闭时清理
    await cleanup_container()


def create_application() -> FastAPI:
    """创建 FastAPI 应用程序"""
    app = FastAPI(
        title="股票预测平台API",
        version="1.0.0",
        description="""
        ## 股票预测平台API

        基于人工智能的股票预测和回测分析平台，提供完整的量化投资解决方案。

        ### 主要功能

        * **数据服务**: 获取股票历史数据和技术指标
        * **预测服务**: 基于机器学习的股票价格预测
        * **任务管理**: 异步任务创建、监控和结果查询
        * **回测服务**: 策略回测和性能分析
        * **模型管理**: 机器学习模型的训练和管理

        ### 技术特性

        * RESTful API设计
        * 异步处理支持
        * 自动限流保护
        * 统一错误处理
        * 完整的API文档

        ### 使用说明

        1. 所有API响应都采用统一的格式
        2. 支持批量操作和并发处理
        3. 提供详细的错误信息和状态码
        4. 包含请求限流和安全保护
        """,
        openapi_url=f"{settings.API_V1_PREFIX}/openapi.json",
        docs_url=f"{settings.API_V1_PREFIX}/docs",
        redoc_url=f"{settings.API_V1_PREFIX}/redoc",
        lifespan=lifespan,
        contact={
            "name": "股票预测平台开发团队",
            "email": "support@stock-prediction.com",
        },
        license_info={
            "name": "MIT License",
            "url": "https://opensource.org/licenses/MIT",
        },
        servers=[
            {
                "url": "http://localhost:8000",
                "description": "开发环境"
            },
            {
                "url": "https://api.stock-prediction.com",
                "description": "生产环境"
            }
        ]
    )

    # 添加中间件（注意顺序很重要）
    
    # 1. 指标收集中间件（最外层）
    app.add_middleware(MetricsMiddleware)
    
    # 2. 请求日志中间件
    app.add_middleware(RequestLoggingMiddleware)
    
    # 3. 错误处理中间件
    app.add_middleware(ErrorHandlingMiddleware)
    
    # 4. 限流中间件
    rate_limit_config = RateLimitConfig(
        requests_per_minute=120,  # 增加到120次/分钟
        requests_per_hour=2000,   # 增加到2000次/小时
        burst_size=20             # 增加突发限制到20
    )
    app.add_middleware(RateLimitMiddleware, config=rate_limit_config)
    
    # 5. GZIP压缩中间件
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # 6. CORS中间件（最内层）
    if settings.CORS_ORIGINS:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins_list,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # 包含路由
    app.include_router(api_router, prefix=settings.API_V1_PREFIX)
    app.include_router(ws_router)  # WebSocket路由不需要前缀
    
    # 添加指标端点
    app.get("/metrics")(metrics_endpoint)

    # 添加异常处理器
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request, exc):
        """HTTP异常处理"""
        from app.api.v1.schemas import StandardResponse
        return JSONResponse(
            status_code=exc.status_code,
            content=StandardResponse(
                success=False,
                message=exc.detail,
                data=None
            ).model_dump()
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request, exc):
        """通用异常处理"""
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"API异常: {exc}")
        
        from app.api.v1.schemas import StandardResponse
        response = StandardResponse(
            success=False,
            message="服务器内部错误",
            data=None
        )
        return JSONResponse(
            status_code=500,
            content=response.model_dump(mode='json')
        )

    return app


app = create_application()