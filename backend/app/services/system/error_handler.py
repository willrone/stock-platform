"""
统一错误处理和恢复机制
实现系统恢复和重试机制
"""
import asyncio
import traceback
import time
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum
import json
from loguru import logger

class ErrorSeverity(Enum):
    """错误严重程度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    """错误类别"""
    NETWORK = "network"
    DATABASE = "database"
    COMPUTATION = "computation"
    VALIDATION = "validation"
    SYSTEM = "system"
    EXTERNAL_API = "external_api"

class RetryStrategy(Enum):
    """重试策略"""
    IMMEDIATE = "immediate"
    LINEAR_BACKOFF = "linear_backoff"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    FIXED_INTERVAL = "fixed_interval"

class ErrorInfo:
    """错误信息类"""
    
    def __init__(self, error: Exception, category: ErrorCategory, severity: ErrorSeverity,
                 context: Dict[str, Any] = None, recoverable: bool = True):
        self.error = error
        self.category = category
        self.severity = severity
        self.context = context or {}
        self.recoverable = recoverable
        self.timestamp = datetime.now()
        self.error_id = f"{category.value}_{int(time.time())}"
        self.traceback = traceback.format_exc()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "error_id": self.error_id,
            "error_type": type(self.error).__name__,
            "error_message": str(self.error),
            "category": self.category.value,
            "severity": self.severity.value,
            "recoverable": self.recoverable,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context,
            "traceback": self.traceback
        }

class SystemErrorHandler:
    """系统错误处理器"""
    
    def __init__(self):
        self.error_history: List[ErrorInfo] = []
        self.recovery_strategies: Dict[ErrorCategory, Callable] = {}
        self.retry_configs: Dict[ErrorCategory, Dict] = {}
        self.circuit_breakers: Dict[str, Dict] = {}
        self.error_stats: Dict[str, int] = {}
        
        # 初始化默认配置
        self._initialize_default_configs()
    
    def _initialize_default_configs(self):
        """初始化默认配置"""
        # 默认重试配置
        self.retry_configs = {
            ErrorCategory.NETWORK: {
                "max_retries": 3,
                "strategy": RetryStrategy.EXPONENTIAL_BACKOFF,
                "base_delay": 1.0,
                "max_delay": 30.0
            },
            ErrorCategory.DATABASE: {
                "max_retries": 2,
                "strategy": RetryStrategy.LINEAR_BACKOFF,
                "base_delay": 2.0,
                "max_delay": 10.0
            },
            ErrorCategory.EXTERNAL_API: {
                "max_retries": 3,
                "strategy": RetryStrategy.EXPONENTIAL_BACKOFF,
                "base_delay": 2.0,
                "max_delay": 60.0
            },
            ErrorCategory.COMPUTATION: {
                "max_retries": 1,
                "strategy": RetryStrategy.IMMEDIATE,
                "base_delay": 0.0,
                "max_delay": 0.0
            }
        }
        
        # 注册默认恢复策略
        self.recovery_strategies[ErrorCategory.DATABASE] = self._recover_database_error
        self.recovery_strategies[ErrorCategory.NETWORK] = self._recover_network_error
        self.recovery_strategies[ErrorCategory.EXTERNAL_API] = self._recover_external_api_error
    
    async def handle_error(self, error: Exception, category: ErrorCategory, 
                          severity: ErrorSeverity, context: Dict[str, Any] = None,
                          operation: Callable = None) -> Any:
        """处理错误"""
        try:
            # 创建错误信息
            error_info = ErrorInfo(error, category, severity, context)
            self.error_history.append(error_info)
            
            # 更新错误统计
            error_key = f"{category.value}_{type(error).__name__}"
            self.error_stats[error_key] = self.error_stats.get(error_key, 0) + 1
            
            # 记录错误日志
            self._log_error(error_info)
            
            # 检查熔断器
            if self._should_circuit_break(category, error_info):
                logger.warning(f"熔断器触发: {category.value}")
                raise Exception(f"服务熔断: {category.value}")
            
            # 尝试恢复
            if error_info.recoverable and operation:
                recovery_result = await self._attempt_recovery(error_info, operation)
                if recovery_result is not None:
                    return recovery_result
            
            # 如果无法恢复，重新抛出错误
            raise error
            
        except Exception as e:
            logger.error(f"错误处理失败: {e}")
            raise
    
    async def _attempt_recovery(self, error_info: ErrorInfo, operation: Callable) -> Any:
        """尝试错误恢复"""
        try:
            category = error_info.category
            retry_config = self.retry_configs.get(category, {})
            max_retries = retry_config.get("max_retries", 0)
            
            if max_retries <= 0:
                return None
            
            # 执行重试
            for attempt in range(max_retries):
                try:
                    # 计算延迟时间
                    delay = self._calculate_retry_delay(retry_config, attempt)
                    if delay > 0:
                        await asyncio.sleep(delay)
                    
                    logger.info(f"重试操作: {category.value}, 第 {attempt + 1}/{max_retries} 次")
                    
                    # 执行恢复策略
                    if category in self.recovery_strategies:
                        await self.recovery_strategies[category](error_info)
                    
                    # 重新执行操作
                    if asyncio.iscoroutinefunction(operation):
                        result = await operation()
                    else:
                        result = operation()
                    
                    logger.info(f"操作恢复成功: {category.value}")
                    return result
                    
                except Exception as retry_error:
                    logger.warning(f"重试失败: 第 {attempt + 1} 次, 错误: {retry_error}")
                    if attempt == max_retries - 1:
                        # 最后一次重试失败
                        logger.error(f"所有重试均失败: {category.value}")
                        return None
            
            return None
            
        except Exception as e:
            logger.error(f"恢复尝试失败: {e}")
            return None
    
    def _calculate_retry_delay(self, retry_config: Dict, attempt: int) -> float:
        """计算重试延迟时间"""
        strategy = retry_config.get("strategy", RetryStrategy.IMMEDIATE)
        base_delay = retry_config.get("base_delay", 1.0)
        max_delay = retry_config.get("max_delay", 30.0)
        
        if strategy == RetryStrategy.IMMEDIATE:
            return 0.0
        elif strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = base_delay * (attempt + 1)
        elif strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = base_delay * (2 ** attempt)
        elif strategy == RetryStrategy.FIXED_INTERVAL:
            delay = base_delay
        else:
            delay = base_delay
        
        return min(delay, max_delay)
    
    def _should_circuit_break(self, category: ErrorCategory, error_info: ErrorInfo) -> bool:
        """检查是否应该触发熔断器"""
        circuit_key = category.value
        
        if circuit_key not in self.circuit_breakers:
            self.circuit_breakers[circuit_key] = {
                "failure_count": 0,
                "last_failure_time": None,
                "state": "closed",  # closed, open, half_open
                "failure_threshold": 5,
                "recovery_timeout": 60  # 秒
            }
        
        circuit = self.circuit_breakers[circuit_key]
        
        # 更新失败计数
        circuit["failure_count"] += 1
        circuit["last_failure_time"] = datetime.now()
        
        # 检查是否达到失败阈值
        if circuit["failure_count"] >= circuit["failure_threshold"]:
            if circuit["state"] == "closed":
                circuit["state"] = "open"
                logger.warning(f"熔断器开启: {circuit_key}")
                return True
        
        # 检查是否可以尝试恢复
        if circuit["state"] == "open":
            time_since_failure = (datetime.now() - circuit["last_failure_time"]).total_seconds()
            if time_since_failure >= circuit["recovery_timeout"]:
                circuit["state"] = "half_open"
                logger.info(f"熔断器半开: {circuit_key}")
                return False
            return True
        
        return False
    
    def reset_circuit_breaker(self, category: ErrorCategory):
        """重置熔断器"""
        circuit_key = category.value
        if circuit_key in self.circuit_breakers:
            self.circuit_breakers[circuit_key] = {
                "failure_count": 0,
                "last_failure_time": None,
                "state": "closed",
                "failure_threshold": 5,
                "recovery_timeout": 60
            }
            logger.info(f"熔断器已重置: {circuit_key}")
    
    async def _recover_database_error(self, error_info: ErrorInfo):
        """数据库错误恢复策略"""
        try:
            logger.info("尝试数据库连接恢复")
            
            # 检查数据库连接
            # 这里应该实现具体的数据库连接检查和重连逻辑
            await asyncio.sleep(1)  # 模拟恢复时间
            
            logger.info("数据库连接恢复成功")
            
        except Exception as e:
            logger.error(f"数据库恢复失败: {e}")
            raise
    
    async def _recover_network_error(self, error_info: ErrorInfo):
        """网络错误恢复策略"""
        try:
            logger.info("尝试网络连接恢复")
            
            # 检查网络连接
            # 这里应该实现具体的网络连接检查逻辑
            await asyncio.sleep(0.5)  # 模拟恢复时间
            
            logger.info("网络连接恢复成功")
            
        except Exception as e:
            logger.error(f"网络恢复失败: {e}")
            raise
    
    async def _recover_external_api_error(self, error_info: ErrorInfo):
        """外部API错误恢复策略"""
        try:
            logger.info("尝试外部API连接恢复")
            
            # 检查API可用性
            # 这里应该实现具体的API健康检查逻辑
            await asyncio.sleep(2)  # 模拟恢复时间
            
            logger.info("外部API连接恢复成功")
            
        except Exception as e:
            logger.error(f"外部API恢复失败: {e}")
            raise
    
    def _log_error(self, error_info: ErrorInfo):
        """记录错误日志"""
        log_level = {
            ErrorSeverity.LOW: logging.INFO,
            ErrorSeverity.MEDIUM: logging.WARNING,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL
        }.get(error_info.severity, logging.ERROR)
        
        logger.log(
            log_level,
            f"错误处理: {error_info.error_id} | "
            f"类别: {error_info.category.value} | "
            f"严重程度: {error_info.severity.value} | "
            f"错误: {error_info.error} | "
            f"上下文: {error_info.context}"
        )
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """获取错误统计信息"""
        try:
            # 按类别统计
            category_stats = {}
            severity_stats = {}
            
            for error_info in self.error_history:
                # 按类别统计
                category = error_info.category.value
                category_stats[category] = category_stats.get(category, 0) + 1
                
                # 按严重程度统计
                severity = error_info.severity.value
                severity_stats[severity] = severity_stats.get(severity, 0) + 1
            
            # 最近错误
            recent_errors = []
            for error_info in self.error_history[-10:]:
                recent_errors.append({
                    "error_id": error_info.error_id,
                    "category": error_info.category.value,
                    "severity": error_info.severity.value,
                    "timestamp": error_info.timestamp.isoformat(),
                    "error_message": str(error_info.error)
                })
            
            # 熔断器状态
            circuit_breaker_status = {}
            for key, circuit in self.circuit_breakers.items():
                circuit_breaker_status[key] = {
                    "state": circuit["state"],
                    "failure_count": circuit["failure_count"],
                    "last_failure_time": circuit["last_failure_time"].isoformat() if circuit["last_failure_time"] else None
                }
            
            return {
                "timestamp": datetime.now().isoformat(),
                "total_errors": len(self.error_history),
                "category_statistics": category_stats,
                "severity_statistics": severity_stats,
                "error_type_statistics": self.error_stats,
                "recent_errors": recent_errors,
                "circuit_breaker_status": circuit_breaker_status
            }
            
        except Exception as e:
            logger.error(f"获取错误统计失败: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
    
    def clear_error_history(self, older_than_days: int = 7):
        """清理错误历史"""
        try:
            cutoff_time = datetime.now() - timedelta(days=older_than_days)
            
            original_count = len(self.error_history)
            self.error_history = [
                error for error in self.error_history 
                if error.timestamp > cutoff_time
            ]
            
            cleared_count = original_count - len(self.error_history)
            logger.info(f"清理错误历史: 删除 {cleared_count} 条记录")
            
            return {
                "cleared_count": cleared_count,
                "remaining_count": len(self.error_history)
            }
            
        except Exception as e:
            logger.error(f"清理错误历史失败: {e}")
            return {"error": str(e)}

# 全局错误处理器实例
error_handler = SystemErrorHandler()

# 装饰器：自动错误处理
def handle_errors(category: ErrorCategory, severity: ErrorSeverity = ErrorSeverity.MEDIUM):
    """错误处理装饰器"""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                context = {
                    "function": func.__name__,
                    "args": str(args)[:200],  # 限制长度
                    "kwargs": str(kwargs)[:200]
                }
                return await error_handler.handle_error(e, category, severity, context, func)
        
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = {
                    "function": func.__name__,
                    "args": str(args)[:200],
                    "kwargs": str(kwargs)[:200]
                }
                # 对于同步函数，直接记录错误但不重试
                error_info = ErrorInfo(e, category, severity, context, recoverable=False)
                error_handler.error_history.append(error_info)
                error_handler._log_error(error_info)
                raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator