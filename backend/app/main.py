"""
FastAPI 应用程序入口点
"""

import json
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.v1.api import api_router
from app.api.v1.health import router as health_router
from app.core.config import settings
from app.core.container import cleanup_container, get_container
from app.core.database import init_db
from app.core.logging import setup_logging
from app.core.metrics import (
    MetricsMiddleware,
    metrics_endpoint,
    setup_metrics_collection,
)
from app.core.security import decode_access_token, decode_refresh_token
from app.middleware.error_handling import (
    ErrorHandlingMiddleware,
    RequestLoggingMiddleware,
)
from app.middleware.rate_limiting import (
    RateLimitConfig,
    RateLimitMiddleware,
    TokenBucket,
)
from app.websocket import ws_router

# 限流提示文案：前端在 429 时展示「请求过于频繁，请稍后重试」
RATE_LIMIT_MESSAGE = "请求过于频繁，请稍后重试"


class IdentityRateLimitMiddleware(RateLimitMiddleware):
    """按用户身份限流（登录用户按 user_id 独立计数，未登录回退到 IP）。

    背景：原实现按**代理 IP** 计数（60 次/分钟），国内用户走同一个出口
    IP/CDN 时全站用户共用一个桶，任意用户的流量尖峰都会把其他人顶到 429；
    而前端 services/api.ts 的 token 刷新逻辑遇到 429 会清 token 跳登录页
    → 用户被莫名踢出登录态。

    计数维度（阈值不变，仍是 60 次/分钟、1000 次/小时）：
      - ``Authorization: Bearer <access token>`` 能解出 user_id
        → 限流键 ``user:<user_id>``，每个用户一个独立桶；
      - ``POST /api/v1/auth/refresh`` 的 refresh_token 放在 body 里
        （前端裸 axios 调用不带 Authorization 头），能解出 user_id
        → 同样按 ``user:<user_id>`` 计数，429 不再落在刷新链路上；
      - 其它请求（未登录 / token 解析失败）→ 回退到 ``ip:<client ip>``，
        X-Forwarded-For 优先，行为与原来一致。

    说明：登录用户的突发令牌桶容量对齐到每分钟额度（60），使实际上限与
    配置声明的「60 次/分钟」一致；未登录 IP 桶保持原 burst_size=10 不变。
    滑窗计数（分钟/小时窗口）完全沿用基类实现，未做任何放宽或收紧。
    """

    def _get_client_id(self, request: Request) -> str:
        key = getattr(request.state, "rate_limit_key", None)
        if key:
            return str(key)
        return f"ip:{super()._get_client_id(request)}"

    async def dispatch(self, request: Request, call_next: Any) -> Any:
        # 提前解析限流键（需要读 body 解 refresh_token，故必须是 async）
        request.state.rate_limit_key = await self._resolve_rate_limit_key(request)
        response = await super().dispatch(request, call_next)
        if response.status_code == 429:
            logger.warning(
                "触发限流: key={} path={} method={}",
                request.state.rate_limit_key,
                request.url.path,
                request.method,
            )
            self._normalize_rate_limit_message(response)
        return response

    @staticmethod
    def _normalize_rate_limit_message(response: Any) -> None:
        """保证 429 响应体带明确可读的 message（前端要展示「请求过于频繁，请稍后重试」）。

        响应结构沿用基类（success/message/data/timestamp + detail），这里只统一文案；
        非限流类 429（例如配额耗尽「配额已用尽: ...」）保持原样不被覆盖。
        """
        if not isinstance(response, JSONResponse):
            return
        try:
            content = json.loads(response.body)
        except Exception:
            return
        if not isinstance(content, dict):
            return
        message = content.get("message")
        if not isinstance(message, str) or "请求过于频繁" not in message:
            return
        content["message"] = RATE_LIMIT_MESSAGE
        content["detail"] = RATE_LIMIT_MESSAGE
        body = json.dumps(content, ensure_ascii=False).encode("utf-8")
        response.body = body
        response.headers["content-length"] = str(len(body))

    async def _resolve_rate_limit_key(self, request: Request) -> str:
        """解析限流键：优先 user_id，其次 IP。解析失败静默回退，绝不抛异常。"""
        user_id = self._user_id_from_header(request)
        if user_id is None:
            user_id = await self._user_id_from_refresh_body(request)
        if user_id:
            return f"user:{user_id}"
        return f"ip:{super()._get_client_id(request)}"

    @staticmethod
    def _user_id_from_header(request: Request) -> Optional[str]:
        """从 Authorization: Bearer <access token> 解出 user_id（失败返回 None）。"""
        authorization = request.headers.get("Authorization")
        if not authorization:
            return None
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() != "bearer" or not token.strip():
            return None
        try:
            payload = decode_access_token(token.strip())
        except Exception:  # 无效/过期/类型不对 → 静默回退 IP 桶
            return None
        sub = payload.get("sub")
        return str(sub) if sub else None

    @staticmethod
    async def _user_id_from_refresh_body(request: Request) -> Optional[str]:
        """从 refresh 请求体的 refresh_token 解出 user_id（失败返回 None）。

        /api/v1/auth/refresh 的 refresh_token 在 body 里（前端裸 axios 调用
        不带 Authorization 头），不单独处理的话刷新请求仍会落到共享 IP 桶。
        """
        if request.method != "POST" or not request.url.path.endswith("/auth/refresh"):
            return None
        try:
            body = await request.body()
            refresh_token = json.loads(body).get("refresh_token")
            if not refresh_token:
                return None
            claims = decode_refresh_token(refresh_token)
        except Exception:  # body 不可读/非 JSON/token 无效 → 静默回退 IP 桶
            return None
        sub = claims.get("sub")
        return str(sub) if sub else None

    def _check_rate_limit(self, client_id: str, request: Request) -> bool:
        # 登录用户：突发容量对齐到每分钟额度，避免「10 次突发 + 1 次/秒」
        # 把有效速率压到远低于配置声明的 60 次/分钟（滑窗阈值保持不变）。
        if client_id.startswith("user:") and client_id not in self.client_buckets:
            self.client_buckets[client_id] = TokenBucket(
                capacity=self.config.requests_per_minute,
                refill_rate=self.config.requests_per_minute / 60.0,
            )
        return super()._check_rate_limit(client_id, request)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """应用程序生命周期管理"""
    # 启动时初始化
    setup_logging()
    await init_db()
    setup_metrics_collection()

    # 初始化服务容器
    await get_container()

    # 注册特征管道回调
    try:
        from app.services.events.data_sync_events import (
            register_feature_pipeline_callbacks,
        )

        register_feature_pipeline_callbacks()
    except Exception as e:
        import logging

        logger = logging.getLogger(__name__)
        logger.warning(f"注册特征管道回调失败: {e}")

    # 启动进程池执行器
    try:
        from app.services.tasks.process_executor import start_process_executor

        start_process_executor()
        import logging

        logger = logging.getLogger(__name__)
        logger.info("进程池执行器已启动")
    except Exception as e:
        import logging

        logger = logging.getLogger(__name__)
        logger.warning(f"启动进程池执行器失败: {e}")

    # 启动任务状态通知器（WebSocket同步）
    try:
        from app.services.tasks.task_notifier import task_notifier

        await task_notifier.start()
        import logging

        logger = logging.getLogger(__name__)
        logger.info("任务状态通知器已启动")
    except Exception as e:
        import logging

        logger = logging.getLogger(__name__)
        logger.warning(f"启动任务状态通知器失败: {e}")

    # 启动资源监控和任务调度器
    try:
        from app.services.infrastructure.resource_monitor import resource_monitor
        from app.services.infrastructure.task_scheduler import task_scheduler
        from app.services.tasks.task_cleanup_service import task_cleanup_service

        # 启动资源监控（30秒间隔）
        await resource_monitor.start_monitoring(30.0)

        # 启动任务调度器
        await task_scheduler.start()

        # 启动任务清理服务
        await task_cleanup_service.start()

        import logging

        logger = logging.getLogger(__name__)
        logger.info("资源监控、任务调度器和任务清理服务已启动")
    except Exception as e:
        import logging

        logger = logging.getLogger(__name__)
        logger.warning(f"启动服务失败: {e}")

    yield

    # 关闭时清理
    try:
        # 停止任务状态通知器
        from app.services.tasks.task_notifier import task_notifier

        await task_notifier.stop()
        import logging

        logger = logging.getLogger(__name__)
        logger.info("任务状态通知器已停止")
    except Exception as e:
        import logging

        logger = logging.getLogger(__name__)
        logger.warning(f"停止任务状态通知器失败: {e}")

    try:
        # 关闭进程池执行器
        from app.services.tasks.process_executor import shutdown_process_executor

        shutdown_process_executor(wait=True, timeout=30)
        import logging

        logger = logging.getLogger(__name__)
        logger.info("进程池执行器已关闭")
    except Exception as e:
        import logging

        logger = logging.getLogger(__name__)
        logger.warning(f"关闭进程池执行器失败: {e}")

    try:
        from app.services.infrastructure.resource_monitor import resource_monitor
        from app.services.infrastructure.task_scheduler import task_scheduler
        from app.services.tasks.task_cleanup_service import task_cleanup_service

        # 停止任务清理服务
        await task_cleanup_service.stop()

        # 停止资源监控和任务调度器
        await resource_monitor.stop_monitoring()
        await task_scheduler.stop()

        import logging

        logger = logging.getLogger(__name__)
        logger.info("所有服务已停止")
    except Exception as e:
        import logging

        logger = logging.getLogger(__name__)
        logger.warning(f"停止服务失败: {e}")

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
            {"url": "http://localhost:8000", "description": "开发环境"},
            {"url": "https://api.stock-prediction.com", "description": "生产环境"},
        ],
    )

    # 添加中间件（注意顺序很重要）

    # 1. 指标收集中间件（最外层）
    app.add_middleware(MetricsMiddleware)

    # 2. 请求日志中间件
    app.add_middleware(RequestLoggingMiddleware)

    # 3. 错误处理中间件
    app.add_middleware(ErrorHandlingMiddleware)

    # 4. 限流中间件（按用户身份计数，见 IdentityRateLimitMiddleware）
    rate_limit_config = RateLimitConfig(
        requests_per_minute=60,
        requests_per_hour=1000,
        burst_size=10,
    )
    app.add_middleware(IdentityRateLimitMiddleware, config=rate_limit_config)

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
    app.include_router(health_router)  # Health check at root (no auth required)

    # 添加指标端点
    app.get("/metrics")(metrics_endpoint)

    # 添加异常处理器
    @app.exception_handler(HTTPException)
    async def http_exception_handler(
        request: Request, exc: HTTPException
    ) -> JSONResponse:
        """HTTP异常处理"""
        from app.api.v1.schemas import StandardResponse

        content = StandardResponse(
            success=False, message=exc.detail, data=None
        ).model_dump(mode="json")
        content["detail"] = exc.detail
        return JSONResponse(status_code=exc.status_code, content=content)

    @app.exception_handler(StarletteHTTPException)
    async def starlette_http_exception_handler(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        """Starlette HTTP异常处理（404/405等路由层异常）"""
        from app.api.v1.schemas import StandardResponse

        content = StandardResponse(
            success=False, message=str(exc.detail), data=None
        ).model_dump(mode="json")
        content["detail"] = exc.detail
        return JSONResponse(status_code=exc.status_code, content=content)

    @app.exception_handler(Exception)
    async def general_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """通用异常处理"""
        import logging

        logger = logging.getLogger(__name__)
        logger.error(f"API异常: {exc}")

        from app.api.v1.schemas import StandardResponse

        response = StandardResponse(success=False, message="服务器内部错误", data=None)
        return JSONResponse(status_code=500, content=response.model_dump(mode="json"))

    original_openapi = app.openapi

    def _custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema
        schema = original_openapi()
        schema.setdefault("components", {}).setdefault("securitySchemes", {})[
            "Bearer Authentication"
        ] = {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "输入 JWT Token（格式: Bearer <token>）",
        }
        schema.setdefault("security", []).append({"Bearer Authentication": []})
        app.openapi_schema = schema
        return schema

    app.openapi = _custom_openapi

    return app


app = create_application()
