"""
统一错误处理框架
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

from loguru import logger


class ErrorType(Enum):
    """错误类型"""

    PREDICTION_ERROR = "prediction_error"
    TASK_ERROR = "task_error"
    MODEL_ERROR = "model_error"
    DATA_ERROR = "data_error"
    SYSTEM_ERROR = "system_error"
    VALIDATION_ERROR = "validation_error"
    AUTHENTICATION_ERROR = "authentication_error"
    AUTHORIZATION_ERROR = "authorization_error"
    NETWORK_ERROR = "network_error"
    STORAGE_ERROR = "storage_error"


class ErrorSeverity(Enum):
    """错误严重程度"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ErrorContext:
    """错误上下文信息"""

    user_id: Optional[str] = None
    task_id: Optional[str] = None
    model_id: Optional[str] = None
    stock_code: Optional[str] = None
    request_id: Optional[str] = None
    additional_data: Optional[Dict[str, Any]] = None


@dataclass
class RecoveryAction:
    """恢复动作"""

    action_type: str  # retry, fallback, skip, abort
    parameters: Dict[str, Any]
    description: str


class BaseError(Exception):
    """基础错误类"""

    def __init__(
        self,
        message: str,
        error_type: ErrorType,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[ErrorContext] = None,
        original_exception: Optional[Exception] = None,
        recovery_actions: Optional[List[RecoveryAction]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_type = error_type
        self.severity = severity
        self.context = context or ErrorContext()
        self.original_exception = original_exception
        self.recovery_actions = recovery_actions or []
        self.timestamp = datetime.utcnow()
        self.error_id = f"{error_type.value}_{int(self.timestamp.timestamp())}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_id": self.error_id,
            "message": self.message,
            "error_type": self.error_type.value,
            "severity": self.severity.value,
            "timestamp": self.timestamp.isoformat(),
            "context": {
                "user_id": self.context.user_id,
                "task_id": self.context.task_id,
                "model_id": self.context.model_id,
                "stock_code": self.context.stock_code,
                "request_id": self.context.request_id,
                "additional_data": self.context.additional_data,
            },
            "original_exception": str(self.original_exception)
            if self.original_exception
            else None,
            "recovery_actions": [
                {
                    "action_type": action.action_type,
                    "parameters": action.parameters,
                    "description": action.description,
                }
                for action in self.recovery_actions
            ],
        }


class PredictionError(BaseError):
    """预测相关错误"""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, ErrorType.PREDICTION_ERROR, **kwargs)


class TaskError(BaseError):
    """任务相关错误"""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, ErrorType.TASK_ERROR, **kwargs)


class ModelError(BaseError):
    """模型相关错误"""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, ErrorType.MODEL_ERROR, **kwargs)


class DataError(BaseError):
    """数据相关错误"""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, ErrorType.DATA_ERROR, **kwargs)


class SystemError(BaseError):
    """系统相关错误"""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, ErrorType.SYSTEM_ERROR, **kwargs)


class ValidationError(BaseError):
    """验证相关错误"""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, ErrorType.VALIDATION_ERROR, **kwargs)


class ErrorRecoveryManager:
    """错误恢复管理器"""

    def __init__(self):
        self.error_history: List[BaseError] = []
        self.recovery_strategies = {
            ErrorType.PREDICTION_ERROR: self._handle_prediction_error,
            ErrorType.TASK_ERROR: self._handle_task_error,
            ErrorType.MODEL_ERROR: self._handle_model_error,
            ErrorType.DATA_ERROR: self._handle_data_error,
            ErrorType.SYSTEM_ERROR: self._handle_system_error,
        }

    def handle_error(self, error: BaseError) -> List[RecoveryAction]:
        """处理错误并返回恢复动作"""
        # 记录错误
        self.error_history.append(error)
        logger.error(f"错误处理: {error.error_id} - {error.message}")

        # 获取恢复策略
        handler = self.recovery_strategies.get(error.error_type)
        if handler:
            recovery_actions = handler(error)
            error.recovery_actions.extend(recovery_actions)
            return recovery_actions

        # 默认恢复动作
        return [
            RecoveryAction(
                action_type="log_and_continue", parameters={}, description="记录错误并继续执行"
            )
        ]

    def _handle_prediction_error(self, error: PredictionError) -> List[RecoveryAction]:
        """处理预测错误"""
        actions = []

        if "model" in error.message.lower():
            # 模型相关错误，尝试使用备用模型
            actions.append(
                RecoveryAction(
                    action_type="use_fallback_model",
                    parameters={"fallback_model_id": "simple_linear"},
                    description="使用备用线性模型进行预测",
                )
            )

        if "data" in error.message.lower():
            # 数据相关错误，尝试扩大时间窗口
            actions.append(
                RecoveryAction(
                    action_type="expand_time_window",
                    parameters={"additional_days": 30},
                    description="扩大历史数据时间窗口",
                )
            )

        # 通用重试策略
        actions.append(
            RecoveryAction(
                action_type="retry_with_delay",
                parameters={"delay_seconds": 60, "max_retries": 3},
                description="延迟重试预测计算",
            )
        )

        return actions

    def _handle_task_error(self, error: TaskError) -> List[RecoveryAction]:
        """处理任务错误"""
        actions = []

        if "queue" in error.message.lower():
            # 队列相关错误
            actions.append(
                RecoveryAction(
                    action_type="requeue_task",
                    parameters={"priority": "high", "delay_seconds": 30},
                    description="重新加入任务队列",
                )
            )

        if "timeout" in error.message.lower():
            # 超时错误
            actions.append(
                RecoveryAction(
                    action_type="increase_timeout",
                    parameters={"timeout_multiplier": 2},
                    description="增加任务超时时间",
                )
            )

        return actions

    def _handle_model_error(self, error: ModelError) -> List[RecoveryAction]:
        """处理模型错误"""
        actions = []

        if "load" in error.message.lower():
            # 模型加载错误
            actions.append(
                RecoveryAction(
                    action_type="reload_model",
                    parameters={"force_reload": True},
                    description="强制重新加载模型",
                )
            )

            actions.append(
                RecoveryAction(
                    action_type="use_cached_model",
                    parameters={},
                    description="使用缓存的模型版本",
                )
            )

        if "training" in error.message.lower():
            # 训练错误
            actions.append(
                RecoveryAction(
                    action_type="adjust_hyperparameters",
                    parameters={"learning_rate": 0.001, "batch_size": 32},
                    description="调整超参数重新训练",
                )
            )

        return actions

    def _handle_data_error(self, error: DataError) -> List[RecoveryAction]:
        """处理数据错误"""
        actions = []

        if "missing" in error.message.lower():
            # 数据缺失
            actions.append(
                RecoveryAction(
                    action_type="fetch_missing_data",
                    parameters={"source": "backup_service"},
                    description="从备用数据源获取缺失数据",
                )
            )

            actions.append(
                RecoveryAction(
                    action_type="interpolate_data",
                    parameters={"method": "linear"},
                    description="使用线性插值填补缺失数据",
                )
            )

        if "invalid" in error.message.lower():
            # 数据无效
            actions.append(
                RecoveryAction(
                    action_type="clean_data",
                    parameters={"remove_outliers": True},
                    description="清理无效数据和异常值",
                )
            )

        return actions

    def _handle_system_error(self, error: SystemError) -> List[RecoveryAction]:
        """处理系统错误"""
        actions = []

        if error.severity == ErrorSeverity.CRITICAL:
            # 严重系统错误
            actions.append(
                RecoveryAction(
                    action_type="alert_administrators",
                    parameters={"channels": ["email", "sms"]},
                    description="立即通知系统管理员",
                )
            )

            actions.append(
                RecoveryAction(
                    action_type="enable_maintenance_mode",
                    parameters={},
                    description="启用维护模式",
                )
            )

        # 通用系统恢复
        actions.append(
            RecoveryAction(
                action_type="restart_service",
                parameters={"service_name": "prediction_engine"},
                description="重启相关服务",
            )
        )

        return actions

    def get_error_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """获取错误统计信息"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        recent_errors = [e for e in self.error_history if e.timestamp > cutoff_time]

        error_counts = {}
        severity_counts = {}

        for error in recent_errors:
            error_type = error.error_type.value
            severity = error.severity.value

            error_counts[error_type] = error_counts.get(error_type, 0) + 1
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

        return {
            "time_range_hours": hours,
            "total_errors": len(recent_errors),
            "error_counts_by_type": error_counts,
            "error_counts_by_severity": severity_counts,
            "most_common_error": max(error_counts.items(), key=lambda x: x[1])
            if error_counts
            else None,
            "critical_errors": len(
                [e for e in recent_errors if e.severity == ErrorSeverity.CRITICAL]
            ),
        }


# 全局错误恢复管理器实例
error_recovery_manager = ErrorRecoveryManager()


def handle_exception(func):
    """装饰器：统一异常处理"""

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except BaseError as e:
            # 已经是我们的错误类型，直接处理
            error_recovery_manager.handle_error(e)
            raise
        except Exception as e:
            # 转换为我们的错误类型
            system_error = SystemError(
                message=f"未处理的异常: {str(e)}",
                severity=ErrorSeverity.HIGH,
                original_exception=e,
            )
            error_recovery_manager.handle_error(system_error)
            raise system_error

    return wrapper


def handle_async_exception(func):
    """装饰器：异步/同步函数统一异常处理"""
    import asyncio
    import functools
    import inspect

    if inspect.iscoroutinefunction(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except BaseError as e:
                error_recovery_manager.handle_error(e)
                raise
            except Exception as e:
                system_error = SystemError(
                    message=f"未处理的异步异常: {str(e)}",
                    severity=ErrorSeverity.HIGH,
                    original_exception=e,
                )
                error_recovery_manager.handle_error(system_error)
                raise system_error
        return wrapper
    else:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except BaseError as e:
                error_recovery_manager.handle_error(e)
                raise
            except Exception as e:
                system_error = SystemError(
                    message=f"未处理的异常: {str(e)}",
                    severity=ErrorSeverity.HIGH,
                    original_exception=e,
                )
                error_recovery_manager.handle_error(system_error)
                raise system_error
        return wrapper
