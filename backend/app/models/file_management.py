"""
文件管理相关数据模型
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum


class IntegrityStatus(Enum):
    """文件完整性状态"""
    VALID = "valid"
    CORRUPTED = "corrupted"
    INCOMPLETE = "incomplete"
    UNKNOWN = "unknown"


@dataclass
class DetailedFileInfo:
    """详细文件信息"""
    file_path: str
    stock_code: str
    date_range: Tuple[datetime, datetime]
    record_count: int
    file_size: int
    last_modified: datetime
    integrity_status: IntegrityStatus
    compression_ratio: float
    created_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'file_path': self.file_path,
            'stock_code': self.stock_code,
            'date_range': {
                'start': self.date_range[0].isoformat(),
                'end': self.date_range[1].isoformat()
            },
            'record_count': self.record_count,
            'file_size': self.file_size,
            'last_modified': self.last_modified.isoformat(),
            'integrity_status': self.integrity_status.value,
            'compression_ratio': self.compression_ratio,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


@dataclass
class ComprehensiveStats:
    """综合统计信息"""
    total_files: int
    total_size_bytes: int
    total_records: int
    stock_count: int
    date_range: Tuple[datetime, datetime]
    average_file_size: float
    storage_efficiency: float
    last_sync_time: Optional[datetime]
    stocks_by_size: List[Tuple[str, int]]
    monthly_distribution: Dict[str, int]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'total_files': self.total_files,
            'total_size_bytes': self.total_size_bytes,
            'total_records': self.total_records,
            'stock_count': self.stock_count,
            'date_range': {
                'start': self.date_range[0].isoformat(),
                'end': self.date_range[1].isoformat()
            },
            'average_file_size': self.average_file_size,
            'storage_efficiency': self.storage_efficiency,
            'last_sync_time': self.last_sync_time.isoformat() if self.last_sync_time else None,
            'stocks_by_size': [{'stock_code': code, 'size': size} for code, size in self.stocks_by_size],
            'monthly_distribution': self.monthly_distribution
        }


@dataclass
class FilterCriteria:
    """文件筛选条件"""
    stock_codes: Optional[List[str]] = None
    date_range: Optional[Tuple[datetime, datetime]] = None
    min_file_size: Optional[int] = None
    max_file_size: Optional[int] = None
    min_records: Optional[int] = None
    max_records: Optional[int] = None
    integrity_status: Optional[IntegrityStatus] = None
    sort_by: str = "last_modified"  # "last_modified", "file_size", "record_count", "stock_code"
    sort_order: str = "desc"  # "asc", "desc"


@dataclass
class ValidationResult:
    """文件验证结果"""
    file_path: str
    is_valid: bool
    integrity_status: IntegrityStatus
    error_messages: List[str]
    record_count: Optional[int] = None
    file_size: Optional[int] = None
    validation_time: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'file_path': self.file_path,
            'is_valid': self.is_valid,
            'integrity_status': self.integrity_status.value,
            'error_messages': self.error_messages,
            'record_count': self.record_count,
            'file_size': self.file_size,
            'validation_time': self.validation_time.isoformat() if self.validation_time else None
        }


@dataclass
class DeletionResult:
    """删除操作结果"""
    success: bool
    deleted_files: List[str]
    failed_files: List[Tuple[str, str]]  # (file_path, error_message)
    total_deleted: int
    freed_space_bytes: int
    message: str
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'success': self.success,
            'deleted_files': self.deleted_files,
            'failed_files': [{'file_path': path, 'error': error} for path, error in self.failed_files],
            'total_deleted': self.total_deleted,
            'freed_space_bytes': self.freed_space_bytes,
            'message': self.message
        }


@dataclass
class FileFilters:
    """文件过滤器"""
    stock_code: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    min_size: Optional[int] = None
    max_size: Optional[int] = None
    integrity_status: Optional[IntegrityStatus] = None
    limit: int = 100
    offset: int = 0


@dataclass
class FileListRequest:
    """文件列表请求"""
    stock_codes: Optional[List[str]] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    include_stats: bool = True
    filters: Optional[FileFilters] = None
    
    def to_filter_criteria(self) -> FilterCriteria:
        """转换为筛选条件"""
        date_range = None
        if self.start_date and self.end_date:
            date_range = (self.start_date, self.end_date)
        
        return FilterCriteria(
            stock_codes=self.stock_codes,
            date_range=date_range,
            min_file_size=self.filters.min_size if self.filters else None,
            max_file_size=self.filters.max_size if self.filters else None,
            integrity_status=self.filters.integrity_status if self.filters else None
        )


@dataclass
class FileListResponse:
    """文件列表响应"""
    files: List[DetailedFileInfo]
    total_files: int
    stats: Optional[ComprehensiveStats] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'files': [file.to_dict() for file in self.files],
            'total_files': self.total_files,
            'stats': self.stats.to_dict() if self.stats else None
        }