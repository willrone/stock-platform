"""
应用层基础错误定义
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


class ErrorType(Enum):
    """错误类型枚举"""
    VALIDATION = "VALIDATION"  # 验证错误
    DOMAIN = "DOMAIN"  # 业务逻辑错误
    INFRA = "INFRA"  # 基础设施错误
    USER_FACING = "USER_FACING"  # 用户可见错误
    SYSTEM = "SYSTEM"  # 系统错误


class AppError(Exception):
    """应用层基础错误"""
    
    def __init__(
        self,
        message: str,
        error_type: ErrorType = ErrorType.SYSTEM,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        retryable: bool = False,
    ):
        self.message = message
        self.error_type = error_type
        self.error_code = error_code or self._generate_error_code(error_type)
        self.details = details or {}
        self.retryable = retryable
        self.timestamp = datetime.now()
        
        super().__init__(self.message)
    
    def _generate_error_code(self, error_type: ErrorType) -> str:
        """生成错误码"""
        import uuid
        return f"{error_type.value}_{uuid.uuid4().hex[:8]}"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式，用于日志记录"""
        return {
            "error_type": self.error_type.value,
            "error_code": self.error_code,
            "message": str(self),
            "details": self.details,
            "retryable": self.retryable,
            "timestamp": self.timestamp.isoformat(),
        }


class ValidationError(AppError):
    """验证错误（validation layer）"""
    
    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        expected: Optional[str] = None,
        actual: Optional[str] = None,
    ):
        details = {}
        if field:
            details["field"] = field
        if expected:
            details["expected"] = expected
        if actual:
            details["actual"] = actual
        
        super().__init__(
            message=message,
            error_type=ErrorType.VALIDATION,
            details=details,
            retryable=False,
        )


class DomainError(AppError):
    """业务逻辑错误（domain layer）"""
    
    def __init__(
        self,
        message: str,
        context: Optional[str] = None,
        recovery_suggestion: Optional[str] = None,
    ):
        details = {}
        if context:
            details["context"] = context
        if recovery_suggestion:
            details["recovery_suggestion"] = recovery_suggestion
        
        super().__init__(
            message=message,
            error_type=ErrorType.DOMAIN,
            details=details,
            retryable=False,
        )


class InfraError(AppError):
    """基础设施错误（infra layer）"""
    
    def __init__(
        self,
        message: str,
        source: Optional[str] = None,
        retryable: bool = True,
    ):
        details = {}
        if source:
            details["source"] = source
        
        super().__init__(
            message=message,
            error_type=ErrorType.INFRA,
            details=details,
            retryable=retryable,
        )


class UserFacingError(AppError):
    """用户可见错误（user-facing layer）"""
    
    def __init__(
        self,
        message: str,
        user_message: Optional[str] = None,
        error_code: Optional[str] = None,
    ):
        details = {}
        if user_message:
            details["user_message"] = user_message
        
        super().__init__(
            message=message,
            error_type=ErrorType.USER_FACING,
            error_code=error_code,
            details=details,
        )
