"""
统一日志记录配置
"""

import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger
from contextvars import ContextVar

# 上下文变量，用于存储请求相关信息
request_id_var: ContextVar[Optional[str]] = ContextVar('request_id', default=None)
user_id_var: ContextVar[Optional[str]] = ContextVar('user_id', default=None)
task_id_var: ContextVar[Optional[str]] = ContextVar('task_id', default=None)


class StructuredFormatter:
    """结构化日志格式化器"""
    
    def __init__(self):
        self.service_name = "stock-prediction-platform"
        self.version = "1.0.0"
    
    def format(self, record):
        """格式化日志记录"""
        # 基础日志信息
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record["level"].name,
            "logger": record["name"],
            "message": record["message"],
            "service": self.service_name,
            "version": self.version,
            "process_id": record["process"].id,
            "thread_id": record["thread"].id,
            "file": record["file"].name,
            "function": record["function"],
            "line": record["line"]
        }
        
        # 添加上下文信息
        request_id = request_id_var.get()
        user_id = user_id_var.get()
        task_id = task_id_var.get()
        
        if request_id:
            log_entry["request_id"] = request_id
        if user_id:
            log_entry["user_id"] = user_id
        if task_id:
            log_entry["task_id"] = task_id
        
        # 添加额外字段
        if record["extra"]:
            log_entry["extra"] = record["extra"]
        
        # 添加异常信息
        if record["exception"]:
            log_entry["exception"] = {
                "type": record["exception"].type.__name__ if record["exception"].type else None,
                "value": str(record["exception"].value) if record["exception"].value else None,
                "traceback": record["exception"].traceback if record["exception"].traceback else None
            }
        
        return json.dumps(log_entry, ensure_ascii=False)


class LoggingConfig:
    """日志配置管理"""
    
    def __init__(self, log_dir: str = "data/logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.formatter = StructuredFormatter()
        self._setup_logger()
    
    def _setup_logger(self):
        """设置日志配置"""
        # 移除默认处理器
        logger.remove()
        
        # 控制台输出（开发环境）
        logger.add(
            sys.stdout,
            format=self._get_console_format(),
            level="INFO",
            colorize=True,
            backtrace=True,
            diagnose=True
        )
        
        # 应用日志文件
        logger.add(
            self.log_dir / "app.log",
            format=self.formatter.format,
            level="INFO",
            rotation="100 MB",
            retention="30 days",
            compression="gz",
            encoding="utf-8",
            backtrace=True,
            diagnose=True
        )
        
        # 错误日志文件
        logger.add(
            self.log_dir / "error.log",
            format=self.formatter.format,
            level="ERROR",
            rotation="50 MB",
            retention="90 days",
            compression="gz",
            encoding="utf-8",
            backtrace=True,
            diagnose=True
        )
        
        # 任务日志文件
        logger.add(
            self.log_dir / "tasks.log",
            format=self.formatter.format,
            level="INFO",
            rotation="50 MB",
            retention="60 days",
            compression="gz",
            encoding="utf-8",
            filter=lambda record: "task_id" in record["extra"]
        )
        
        # 审计日志文件
        logger.add(
            self.log_dir / "audit.log",
            format=self.formatter.format,
            level="INFO",
            rotation="100 MB",
            retention="365 days",  # 审计日志保留一年
            compression="gz",
            encoding="utf-8",
            filter=lambda record: record["extra"].get("audit", False)
        )
        
        # 性能日志文件
        logger.add(
            self.log_dir / "performance.log",
            format=self.formatter.format,
            level="INFO",
            rotation="50 MB",
            retention="30 days",
            compression="gz",
            encoding="utf-8",
            filter=lambda record: record["extra"].get("performance", False)
        )
    
    def _get_console_format(self) -> str:
        """获取控制台日志格式"""
        return (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )


class AuditLogger:
    """审计日志记录器"""
    
    @staticmethod
    def log_user_action(action: str, user_id: str, resource: str = None, 
                       details: Dict[str, Any] = None, success: bool = True):
        """记录用户操作"""
        audit_data = {
            "audit": True,
            "action": action,
            "user_id": user_id,
            "resource": resource,
            "details": details or {},
            "success": success,
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": request_id_var.get()
        }
        
        logger.info(f"用户操作: {action}", **audit_data)
    
    @staticmethod
    def log_data_change(table: str, operation: str, record_id: str, 
                       old_values: Dict[str, Any] = None, 
                       new_values: Dict[str, Any] = None,
                       user_id: str = None):
        """记录数据变更"""
        audit_data = {
            "audit": True,
            "data_change": True,
            "table": table,
            "operation": operation,
            "record_id": record_id,
            "old_values": old_values or {},
            "new_values": new_values or {},
            "user_id": user_id,
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": request_id_var.get()
        }
        
        logger.info(f"数据变更: {table}.{operation}", **audit_data)
    
    @staticmethod
    def log_system_event(event_type: str, description: str, 
                        severity: str = "info", details: Dict[str, Any] = None):
        """记录系统事件"""
        audit_data = {
            "audit": True,
            "system_event": True,
            "event_type": event_type,
            "description": description,
            "severity": severity,
            "details": details or {},
            "timestamp": datetime.utcnow().isoformat()
        }
        
        logger.info(f"系统事件: {event_type}", **audit_data)
    
    @staticmethod
    def log_security_event(event_type: str, user_id: str = None, 
                          ip_address: str = None, details: Dict[str, Any] = None):
        """记录安全事件"""
        audit_data = {
            "audit": True,
            "security_event": True,
            "event_type": event_type,
            "user_id": user_id,
            "ip_address": ip_address,
            "details": details or {},
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": request_id_var.get()
        }
        
        logger.warning(f"安全事件: {event_type}", **audit_data)


class PerformanceLogger:
    """性能日志记录器"""
    
    @staticmethod
    def log_api_performance(endpoint: str, method: str, duration_ms: float,
                           status_code: int, user_id: str = None):
        """记录API性能"""
        perf_data = {
            "performance": True,
            "api_performance": True,
            "endpoint": endpoint,
            "method": method,
            "duration_ms": duration_ms,
            "status_code": status_code,
            "user_id": user_id,
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": request_id_var.get()
        }
        
        logger.info(f"API性能: {method} {endpoint} - {duration_ms:.2f}ms", **perf_data)
    
    @staticmethod
    def log_task_performance(task_id: str, task_type: str, duration_seconds: float,
                           success: bool, details: Dict[str, Any] = None):
        """记录任务性能"""
        perf_data = {
            "performance": True,
            "task_performance": True,
            "task_id": task_id,
            "task_type": task_type,
            "duration_seconds": duration_seconds,
            "success": success,
            "details": details or {},
            "timestamp": datetime.utcnow().isoformat()
        }
        
        logger.info(f"任务性能: {task_type} - {duration_seconds:.2f}s", **perf_data)
    
    @staticmethod
    def log_model_performance(model_id: str, operation: str, duration_ms: float,
                            input_size: int = None, details: Dict[str, Any] = None):
        """记录模型性能"""
        perf_data = {
            "performance": True,
            "model_performance": True,
            "model_id": model_id,
            "operation": operation,
            "duration_ms": duration_ms,
            "input_size": input_size,
            "details": details or {},
            "timestamp": datetime.utcnow().isoformat()
        }
        
        logger.info(f"模型性能: {model_id}.{operation} - {duration_ms:.2f}ms", **perf_data)


class LogContext:
    """日志上下文管理器"""
    
    def __init__(self, request_id: str = None, user_id: str = None, task_id: str = None):
        self.request_id = request_id
        self.user_id = user_id
        self.task_id = task_id
        self.tokens = []
    
    def __enter__(self):
        if self.request_id:
            self.tokens.append(request_id_var.set(self.request_id))
        if self.user_id:
            self.tokens.append(user_id_var.set(self.user_id))
        if self.task_id:
            self.tokens.append(task_id_var.set(self.task_id))
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        for token in reversed(self.tokens):
            token.var.reset(token)


# 初始化日志配置
logging_config = LoggingConfig()

# 创建专用日志记录器实例
audit_logger = AuditLogger()
performance_logger = PerformanceLogger()


def get_logger(name: str = None):
    """获取日志记录器"""
    if name:
        return logger.bind(logger_name=name)
    return logger


def set_log_context(request_id: str = None, user_id: str = None, task_id: str = None):
    """设置日志上下文"""
    return LogContext(request_id=request_id, user_id=user_id, task_id=task_id)