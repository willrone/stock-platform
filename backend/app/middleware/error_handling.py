"""
错误处理中间件

统一错误处理和日志记录
"""

import traceback
import logging
from typing import Any, Dict
from datetime import datetime
from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.routes import StandardResponse

logger = logging.getLogger(__name__)


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """错误处理中间件"""
    
    def __init__(self, app):
        super().__init__(app)
        self.error_counts = {}  # 错误统计
    
    async def dispatch(self, request: Request, call_next):
        """处理请求"""
        try:
            response = await call_next(request)
            return response
            
        except HTTPException as exc:
            return await self._handle_http_exception(request, exc)
        
        except StarletteHTTPException as exc:
            return await self._handle_http_exception(request, exc)
        
        except ValueError as exc:
            return await self._handle_validation_error(request, exc)
        
        except ConnectionError as exc:
            return await self._handle_connection_error(request, exc)
        
        except TimeoutError as exc:
            return await self._handle_timeout_error(request, exc)
        
        except Exception as exc:
            return await self._handle_general_error(request, exc)
    
    async def _handle_http_exception(
        self, 
        request: Request, 
        exc: HTTPException
    ) -> JSONResponse:
        """处理HTTP异常"""
        error_info = {
            "path": str(request.url),
            "method": request.method,
            "status_code": exc.status_code,
            "detail": exc.detail
        }
        
        logger.warning(f"HTTP异常: {error_info}")
        
        # 记录错误统计
        self._record_error(exc.status_code)
        
        return JSONResponse(
            status_code=exc.status_code,
            content=StandardResponse(
                success=False,
                message=str(exc.detail),
                data=None
            ).dict()
        )
    
    async def _handle_validation_error(
        self, 
        request: Request, 
        exc: ValueError
    ) -> JSONResponse:
        """处理验证错误"""
        error_info = {
            "path": str(request.url),
            "method": request.method,
            "error": str(exc),
            "type": "validation_error"
        }
        
        logger.warning(f"验证错误: {error_info}")
        self._record_error("validation_error")
        
        return JSONResponse(
            status_code=400,
            content=StandardResponse(
                success=False,
                message=f"数据验证失败: {str(exc)}",
                data=None
            ).dict()
        )
    
    async def _handle_connection_error(
        self, 
        request: Request, 
        exc: ConnectionError
    ) -> JSONResponse:
        """处理连接错误"""
        error_info = {
            "path": str(request.url),
            "method": request.method,
            "error": str(exc),
            "type": "connection_error"
        }
        
        logger.error(f"连接错误: {error_info}")
        self._record_error("connection_error")
        
        return JSONResponse(
            status_code=503,
            content=StandardResponse(
                success=False,
                message="服务暂时不可用，请稍后重试",
                data=None
            ).dict()
        )
    
    async def _handle_timeout_error(
        self, 
        request: Request, 
        exc: TimeoutError
    ) -> JSONResponse:
        """处理超时错误"""
        error_info = {
            "path": str(request.url),
            "method": request.method,
            "error": str(exc),
            "type": "timeout_error"
        }
        
        logger.error(f"超时错误: {error_info}")
        self._record_error("timeout_error")
        
        return JSONResponse(
            status_code=504,
            content=StandardResponse(
                success=False,
                message="请求超时，请稍后重试",
                data=None
            ).dict()
        )
    
    async def _handle_general_error(
        self, 
        request: Request, 
        exc: Exception
    ) -> JSONResponse:
        """处理通用错误"""
        error_info = {
            "path": str(request.url),
            "method": request.method,
            "error": str(exc),
            "type": type(exc).__name__,
            "traceback": traceback.format_exc()
        }
        
        logger.error(f"未处理的异常: {error_info}")
        self._record_error("internal_error")
        
        # 在生产环境中不暴露详细错误信息
        return JSONResponse(
            status_code=500,
            content=StandardResponse(
                success=False,
                message="服务器内部错误",
                data=None
            ).dict()
        )
    
    def _record_error(self, error_type: str):
        """记录错误统计"""
        if error_type not in self.error_counts:
            self.error_counts[error_type] = 0
        self.error_counts[error_type] += 1
    
    def get_error_stats(self) -> Dict[str, Any]:
        """获取错误统计"""
        return {
            "error_counts": self.error_counts.copy(),
            "total_errors": sum(self.error_counts.values()),
            "timestamp": datetime.now().isoformat()
        }


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """请求日志中间件"""
    
    def __init__(self, app):
        super().__init__(app)
    
    async def dispatch(self, request: Request, call_next):
        """处理请求"""
        start_time = datetime.now()
        
        # 记录请求信息
        request_info = {
            "method": request.method,
            "url": str(request.url),
            "client": getattr(request.client, "host", "unknown"),
            "user_agent": request.headers.get("user-agent", "unknown"),
            "start_time": start_time.isoformat()
        }
        
        logger.info(f"请求开始: {request_info}")
        
        try:
            response = await call_next(request)
            
            # 记录响应信息
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            response_info = {
                "status_code": response.status_code,
                "duration_seconds": duration,
                "end_time": end_time.isoformat()
            }
            
            logger.info(f"请求完成: {request_info['method']} {request_info['url']} - {response_info}")
            
            # 添加响应时间头
            response.headers["X-Response-Time"] = f"{duration:.3f}s"
            
            return response
            
        except Exception as exc:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            error_info = {
                "error": str(exc),
                "duration_seconds": duration,
                "end_time": end_time.isoformat()
            }
            
            logger.error(f"请求失败: {request_info['method']} {request_info['url']} - {error_info}")
            raise


# 导出主要类
__all__ = [
    'ErrorHandlingMiddleware',
    'RequestLoggingMiddleware'
]