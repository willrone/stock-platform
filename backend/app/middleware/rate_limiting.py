"""
限流中间件

实现基本的限流策略，防止API滥用
"""

import time
import logging
from typing import Dict, Optional
from collections import defaultdict, deque
from fastapi import Request, Response, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class RateLimitConfig:
    """限流配置"""
    
    def __init__(
        self,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
        burst_size: int = 10
    ):
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.burst_size = burst_size


class TokenBucket:
    """令牌桶算法实现"""
    
    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = refill_rate  # 每秒补充的令牌数
        self.last_refill = time.time()
    
    def consume(self, tokens: int = 1) -> bool:
        """消费令牌"""
        now = time.time()
        
        # 补充令牌
        time_passed = now - self.last_refill
        self.tokens = min(
            self.capacity,
            self.tokens + time_passed * self.refill_rate
        )
        self.last_refill = now
        
        # 检查是否有足够的令牌
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False


class SlidingWindowCounter:
    """滑动窗口计数器"""
    
    def __init__(self, window_size: int):
        self.window_size = window_size  # 窗口大小（秒）
        self.requests = deque()
    
    def add_request(self) -> int:
        """添加请求并返回当前窗口内的请求数"""
        now = time.time()
        self.requests.append(now)
        
        # 清理过期请求
        cutoff = now - self.window_size
        while self.requests and self.requests[0] < cutoff:
            self.requests.popleft()
        
        return len(self.requests)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """限流中间件"""
    
    def __init__(self, app, config: Optional[RateLimitConfig] = None):
        super().__init__(app)
        self.config = config or RateLimitConfig()
        
        # 存储每个客户端的限流状态
        self.client_buckets: Dict[str, TokenBucket] = {}
        self.client_windows: Dict[str, Dict[str, SlidingWindowCounter]] = defaultdict(dict)
        
        # 清理过期数据的时间戳
        self.last_cleanup = time.time()
        self.cleanup_interval = 300  # 5分钟清理一次
    
    async def dispatch(self, request: Request, call_next):
        """处理请求"""
        # 获取客户端标识
        client_id = self._get_client_id(request)
        
        # 检查是否需要清理过期数据
        self._cleanup_expired_data()
        
        # 检查限流
        if not self._check_rate_limit(client_id, request):
            logger.warning(f"客户端 {client_id} 触发限流")
            raise HTTPException(
                status_code=429,
                detail="请求过于频繁，请稍后再试",
                headers={"Retry-After": "60"}
            )
        
        # 处理请求
        response = await call_next(request)
        
        # 添加限流相关的响应头
        self._add_rate_limit_headers(response, client_id)
        
        return response
    
    def _get_client_id(self, request: Request) -> str:
        """获取客户端标识"""
        # 优先使用X-Forwarded-For头
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        # 使用客户端IP
        client_host = getattr(request.client, "host", "unknown")
        return client_host
    
    def _check_rate_limit(self, client_id: str, request: Request) -> bool:
        """检查限流"""
        # 检查令牌桶（突发请求限制）
        if client_id not in self.client_buckets:
            self.client_buckets[client_id] = TokenBucket(
                capacity=self.config.burst_size,
                refill_rate=self.config.requests_per_minute / 60.0
            )
        
        bucket = self.client_buckets[client_id]
        if not bucket.consume():
            return False
        
        # 检查滑动窗口（分钟级限制）
        if "minute" not in self.client_windows[client_id]:
            self.client_windows[client_id]["minute"] = SlidingWindowCounter(60)
        
        minute_window = self.client_windows[client_id]["minute"]
        minute_requests = minute_window.add_request()
        
        if minute_requests > self.config.requests_per_minute:
            return False
        
        # 检查滑动窗口（小时级限制）
        if "hour" not in self.client_windows[client_id]:
            self.client_windows[client_id]["hour"] = SlidingWindowCounter(3600)
        
        hour_window = self.client_windows[client_id]["hour"]
        hour_requests = hour_window.add_request()
        
        if hour_requests > self.config.requests_per_hour:
            return False
        
        return True
    
    def _add_rate_limit_headers(self, response: Response, client_id: str):
        """添加限流相关的响应头"""
        if client_id in self.client_windows:
            windows = self.client_windows[client_id]
            
            # 添加剩余请求数
            if "minute" in windows:
                minute_remaining = max(
                    0, 
                    self.config.requests_per_minute - len(windows["minute"].requests)
                )
                response.headers["X-RateLimit-Remaining-Minute"] = str(minute_remaining)
            
            if "hour" in windows:
                hour_remaining = max(
                    0,
                    self.config.requests_per_hour - len(windows["hour"].requests)
                )
                response.headers["X-RateLimit-Remaining-Hour"] = str(hour_remaining)
        
        # 添加限制信息
        response.headers["X-RateLimit-Limit-Minute"] = str(self.config.requests_per_minute)
        response.headers["X-RateLimit-Limit-Hour"] = str(self.config.requests_per_hour)
    
    def _cleanup_expired_data(self):
        """清理过期数据"""
        now = time.time()
        if now - self.last_cleanup < self.cleanup_interval:
            return
        
        # 清理过期的令牌桶
        expired_clients = []
        for client_id, bucket in self.client_buckets.items():
            if now - bucket.last_refill > 3600:  # 1小时未活动
                expired_clients.append(client_id)
        
        for client_id in expired_clients:
            del self.client_buckets[client_id]
            if client_id in self.client_windows:
                del self.client_windows[client_id]
        
        self.last_cleanup = now
        
        if expired_clients:
            logger.info(f"清理了 {len(expired_clients)} 个过期客户端的限流数据")


# 导出主要类
__all__ = [
    'RateLimitMiddleware',
    'RateLimitConfig',
    'TokenBucket',
    'SlidingWindowCounter'
]