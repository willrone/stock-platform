"""
日志配置

提供结构化日志记录，支持多种输出格式和日志级别。
包含性能监控、错误追踪和审计日志功能。
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from loguru import logger

from app.core.config import settings


class StructuredFormatter:
    """结构化日志格式化器"""

    def __init__(self, include_extra: bool = True):
        self.include_extra = include_extra

    def format(self, record: Dict[str, Any]) -> str:
        """格式化日志记录为JSON"""
        log_entry = {
            "timestamp": record["time"].isoformat(),
            "level": record["level"].name,
            "logger": record["name"],
            "module": record["module"],
            "function": record["function"],
            "line": record["line"],
            "message": record["message"],
            "process_id": record["process"].id,
            "thread_id": record["thread"].id,
        }

        # 添加额外字段
        if self.include_extra and "extra" in record:
            log_entry.update(record["extra"])

        # 添加异常信息
        if record.get("exception"):
            log_entry["exception"] = {
                "type": record["exception"].type.__name__,
                "value": str(record["exception"].value),
                "traceback": record["exception"].traceback,
            }

        return json.dumps(log_entry, ensure_ascii=False)


def setup_logging() -> None:
    """设置日志配置"""
    # 移除默认处理器
    logger.remove()

    # 创建日志目录
    log_path = Path(settings.DATA_ROOT_PATH) / "logs"
    log_path.mkdir(parents=True, exist_ok=True)

    # 控制台输出（开发环境使用彩色格式，生产环境使用JSON格式）
    if settings.ENVIRONMENT == "development":
        logger.add(
            sys.stdout,
            level=settings.LOG_LEVEL.upper(),
            format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>",
            colorize=True,
            backtrace=True,
            diagnose=True,
        )
    else:
        # 生产环境使用结构化JSON日志
        logger.add(
            sys.stdout,
            level=settings.LOG_LEVEL.upper(),
            format=StructuredFormatter().format,
            serialize=True,
        )

    # 应用程序日志文件
    logger.add(
        log_path / "app.log",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
        rotation="100 MB",
        retention="30 days",
        compression="zip",
        enqueue=True,  # 异步写入
        colorize=False,  # 文件日志禁用颜色
    )

    # 错误日志文件
    logger.add(
        log_path / "error.log",
        level="ERROR",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message} | {extra}",
        rotation="50 MB",
        retention="90 days",
        compression="zip",
        enqueue=True,
        backtrace=True,
        diagnose=True,
        colorize=False,  # 文件日志禁用颜色
    )

    # 性能日志文件
    logger.add(
        log_path / "performance.log",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {message}",
        rotation="50 MB",
        retention="7 days",
        compression="zip",
        filter=lambda record: record["extra"].get("log_type") == "performance",
        enqueue=True,
        colorize=False,  # 文件日志禁用颜色
    )

    # 审计日志文件
    logger.add(
        log_path / "audit.log",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {message}",
        rotation="100 MB",
        retention="365 days",
        compression="zip",
        filter=lambda record: record["extra"].get("log_type") == "audit",
        enqueue=True,
        colorize=False,  # 文件日志禁用颜色
    )

    # API访问日志文件
    logger.add(
        log_path / "access.log",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {message}",
        rotation="100 MB",
        retention="30 days",
        compression="zip",
        filter=lambda record: record["extra"].get("log_type") == "access",
        enqueue=True,
        colorize=False,  # 文件日志禁用颜色
    )

    # 数据同步日志文件
    logger.add(
        log_path / "data_sync.log",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}",
        rotation="50 MB",
        retention="30 days",
        compression="zip",
        filter=lambda record: record["extra"].get("log_type") == "data_sync",
        enqueue=True,
        colorize=False,  # 文件日志禁用颜色
    )

    # 任务执行日志文件
    logger.add(
        log_path / "tasks.log",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}",
        rotation="50 MB",
        retention="30 days",
        compression="zip",
        filter=lambda record: record["extra"].get("log_type") == "task",
        enqueue=True,
        colorize=False,  # 文件日志禁用颜色
    )


class LoggerMixin:
    """日志记录混入类"""

    @property
    def logger(self):
        """获取当前类的日志记录器"""
        return logger.bind(class_name=self.__class__.__name__)


def log_performance(operation: str, duration: float, **kwargs):
    """记录性能日志"""
    logger.bind(log_type="performance").info(
        f"PERFORMANCE | {operation} | {duration:.3f}s | {json.dumps(kwargs)}"
    )


def log_audit(action: str, user_id: str = None, resource: str = None, **kwargs):
    """记录审计日志"""
    audit_data = {
        "action": action,
        "user_id": user_id,
        "resource": resource,
        "timestamp": datetime.now().isoformat(),
        **kwargs,
    }
    logger.bind(log_type="audit").info(f"AUDIT | {json.dumps(audit_data)}")


def log_access(
    method: str,
    path: str,
    status_code: int,
    duration: float,
    ip: str = None,
    user_agent: str = None,
    **kwargs,
):
    """记录API访问日志"""
    access_data = {
        "method": method,
        "path": path,
        "status_code": status_code,
        "duration": duration,
        "ip": ip,
        "user_agent": user_agent,
        "timestamp": datetime.now().isoformat(),
        **kwargs,
    }
    logger.bind(log_type="access").info(f"ACCESS | {json.dumps(access_data)}")


def log_data_sync(
    operation: str,
    stock_code: str = None,
    status: str = "success",
    records: int = 0,
    **kwargs,
):
    """记录数据同步日志"""
    sync_data = {
        "operation": operation,
        "stock_code": stock_code,
        "status": status,
        "records": records,
        "timestamp": datetime.now().isoformat(),
        **kwargs,
    }
    logger.bind(log_type="data_sync").info(f"DATA_SYNC | {json.dumps(sync_data)}")


def log_task(task_id: str, task_type: str, status: str, **kwargs):
    """记录任务执行日志"""
    task_data = {
        "task_id": task_id,
        "task_type": task_type,
        "status": status,
        "timestamp": datetime.now().isoformat(),
        **kwargs,
    }
    logger.bind(log_type="task").info(f"TASK | {json.dumps(task_data)}")
