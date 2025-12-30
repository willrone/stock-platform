"""
Prometheus指标收集模块

提供应用程序性能指标的收集和暴露功能。
"""

import time
from typing import Dict, Any
from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Request, Response
from fastapi.responses import Response as FastAPIResponse
import logging

logger = logging.getLogger(__name__)

# 定义Prometheus指标

# HTTP请求计数器
http_requests_total = Counter(
    'http_requests_total',
    'HTTP请求总数',
    ['method', 'endpoint', 'status']
)

# HTTP请求持续时间直方图
http_request_duration_seconds = Histogram(
    'http_request_duration_seconds',
    'HTTP请求持续时间（秒）',
    ['method', 'endpoint']
)

# 活跃连接数
active_connections = Gauge(
    'active_connections',
    '当前活跃连接数'
)

# 数据库连接数
database_connections_active = Gauge(
    'database_connections_active',
    '活跃数据库连接数'
)

# 任务队列大小
task_queue_size = Gauge(
    'task_queue_size',
    '任务队列中的任务数量'
)

# 预测任务计数器
prediction_tasks_total = Counter(
    'prediction_tasks_total',
    '预测任务总数',
    ['status']
)

# 模型预测时间
model_prediction_duration_seconds = Histogram(
    'model_prediction_duration_seconds',
    '模型预测时间（秒）',
    ['model_type']
)

# 数据同步计数器
data_sync_total = Counter(
    'data_sync_total',
    '数据同步总数',
    ['source', 'status']
)

# 错误计数器
errors_total = Counter(
    'errors_total',
    '错误总数',
    ['error_type', 'endpoint']
)

# 系统信息
app_info = Info(
    'app_info',
    '应用程序信息'
)

# 内存使用量
memory_usage_bytes = Gauge(
    'memory_usage_bytes',
    '内存使用量（字节）'
)

# CPU使用率
cpu_usage_percent = Gauge(
    'cpu_usage_percent',
    'CPU使用率（百分比）'
)


class MetricsCollector:
    """指标收集器"""
    
    def __init__(self):
        self.start_time = time.time()
        self._setup_app_info()
    
    def _setup_app_info(self):
        """设置应用程序信息"""
        app_info.info({
            'version': '1.0.0',
            'name': '股票预测平台',
            'description': '基于AI的股票预测和回测分析平台'
        })
    
    def record_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """记录HTTP请求指标"""
        try:
            # 清理端点路径，移除参数
            clean_endpoint = self._clean_endpoint(endpoint)
            
            # 记录请求计数
            http_requests_total.labels(
                method=method,
                endpoint=clean_endpoint,
                status=str(status_code)
            ).inc()
            
            # 记录请求持续时间
            http_request_duration_seconds.labels(
                method=method,
                endpoint=clean_endpoint
            ).observe(duration)
            
        except Exception as e:
            logger.error(f"记录请求指标失败: {e}")
    
    def record_error(self, error_type: str, endpoint: str):
        """记录错误指标"""
        try:
            clean_endpoint = self._clean_endpoint(endpoint)
            errors_total.labels(
                error_type=error_type,
                endpoint=clean_endpoint
            ).inc()
        except Exception as e:
            logger.error(f"记录错误指标失败: {e}")
    
    def record_prediction_task(self, status: str):
        """记录预测任务指标"""
        try:
            prediction_tasks_total.labels(status=status).inc()
        except Exception as e:
            logger.error(f"记录预测任务指标失败: {e}")
    
    def record_model_prediction_time(self, model_type: str, duration: float):
        """记录模型预测时间"""
        try:
            model_prediction_duration_seconds.labels(model_type=model_type).observe(duration)
        except Exception as e:
            logger.error(f"记录模型预测时间失败: {e}")
    
    def record_data_sync(self, source: str, status: str):
        """记录数据同步指标"""
        try:
            data_sync_total.labels(source=source, status=status).inc()
        except Exception as e:
            logger.error(f"记录数据同步指标失败: {e}")
    
    def update_active_connections(self, count: int):
        """更新活跃连接数"""
        try:
            active_connections.set(count)
        except Exception as e:
            logger.error(f"更新活跃连接数失败: {e}")
    
    def update_database_connections(self, count: int):
        """更新数据库连接数"""
        try:
            database_connections_active.set(count)
        except Exception as e:
            logger.error(f"更新数据库连接数失败: {e}")
    
    def update_task_queue_size(self, size: int):
        """更新任务队列大小"""
        try:
            task_queue_size.set(size)
        except Exception as e:
            logger.error(f"更新任务队列大小失败: {e}")
    
    def update_system_metrics(self):
        """更新系统指标"""
        try:
            import psutil
            
            # 更新内存使用量
            memory_info = psutil.virtual_memory()
            memory_usage_bytes.set(memory_info.used)
            
            # 更新CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_usage_percent.set(cpu_percent)
            
        except ImportError:
            logger.warning("psutil未安装，无法收集系统指标")
        except Exception as e:
            logger.error(f"更新系统指标失败: {e}")
    
    def _clean_endpoint(self, endpoint: str) -> str:
        """清理端点路径，移除动态参数"""
        # 移除查询参数
        if '?' in endpoint:
            endpoint = endpoint.split('?')[0]
        
        # 替换路径参数为占位符
        import re
        # 匹配UUID、数字ID等
        endpoint = re.sub(r'/[0-9a-f-]{36}', '/{id}', endpoint)  # UUID
        endpoint = re.sub(r'/\d+', '/{id}', endpoint)  # 数字ID
        endpoint = re.sub(r'/[A-Z0-9]{6,}\.SZ|SH', '/{stock_code}', endpoint)  # 股票代码
        
        return endpoint
    
    def get_metrics(self) -> str:
        """获取Prometheus格式的指标"""
        try:
            # 更新系统指标
            self.update_system_metrics()
            
            # 生成指标数据
            return generate_latest()
        except Exception as e:
            logger.error(f"生成指标数据失败: {e}")
            return ""


# 全局指标收集器实例
metrics_collector = MetricsCollector()


class MetricsMiddleware:
    """指标收集中间件"""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        start_time = time.time()
        method = scope["method"]
        path = scope["path"]
        
        # 跳过指标端点本身
        if path == "/metrics":
            await self.app(scope, receive, send)
            return
        
        status_code = 200
        
        async def send_wrapper(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            await send(message)
        
        try:
            await self.app(scope, receive, send_wrapper)
        except Exception as e:
            status_code = 500
            metrics_collector.record_error("internal_error", path)
            raise
        finally:
            # 记录请求指标
            duration = time.time() - start_time
            metrics_collector.record_request(method, path, status_code, duration)


async def metrics_endpoint() -> FastAPIResponse:
    """Prometheus指标端点"""
    try:
        metrics_data = metrics_collector.get_metrics()
        return FastAPIResponse(
            content=metrics_data,
            media_type=CONTENT_TYPE_LATEST
        )
    except Exception as e:
        logger.error(f"指标端点错误: {e}")
        return FastAPIResponse(
            content="# 指标收集失败\n",
            media_type=CONTENT_TYPE_LATEST,
            status_code=500
        )


def setup_metrics_collection():
    """设置指标收集"""
    logger.info("指标收集系统已启动")
    return metrics_collector