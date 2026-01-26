"""
数据同步相关模型
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional


class SyncMode(Enum):
    """同步模式"""

    INCREMENTAL = "incremental"
    FULL = "full"


class SyncStatus(Enum):
    """同步状态"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"


@dataclass
class BatchSyncRequest:
    """批量同步请求"""

    stock_codes: List[str]
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    force_update: bool = False
    sync_mode: SyncMode = SyncMode.INCREMENTAL
    max_concurrent: int = 3
    retry_count: int = 3

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "stock_codes": self.stock_codes,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "force_update": self.force_update,
            "sync_mode": self.sync_mode.value,
            "max_concurrent": self.max_concurrent,
            "retry_count": self.retry_count,
        }


@dataclass
class SyncOptions:
    """单个股票同步选项"""

    force_update: bool = False
    sync_mode: SyncMode = SyncMode.INCREMENTAL
    retry_count: int = 3
    timeout_seconds: int = 300


@dataclass
class SyncResult:
    """单个股票同步结果"""

    stock_code: str
    success: bool
    records_synced: int
    start_time: datetime
    end_time: datetime
    error_message: Optional[str] = None
    data_range: Optional[tuple[datetime, datetime]] = None

    @property
    def duration(self) -> timedelta:
        """同步耗时"""
        return self.end_time - self.start_time

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "stock_code": self.stock_code,
            "success": self.success,
            "records_synced": self.records_synced,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "duration_seconds": self.duration.total_seconds(),
            "error_message": self.error_message,
            "data_range": {
                "start": self.data_range[0].isoformat(),
                "end": self.data_range[1].isoformat(),
            }
            if self.data_range
            else None,
        }


@dataclass
class BatchSyncResult:
    """批量同步结果"""

    sync_id: str
    success: bool
    total_stocks: int
    successful_syncs: List[SyncResult]
    failed_syncs: List[SyncResult]
    start_time: datetime
    end_time: datetime
    message: str

    @property
    def success_count(self) -> int:
        """成功数量"""
        return len(self.successful_syncs)

    @property
    def failure_count(self) -> int:
        """失败数量"""
        return len(self.failed_syncs)

    @property
    def total_records(self) -> int:
        """总记录数"""
        return sum(result.records_synced for result in self.successful_syncs)

    @property
    def duration(self) -> timedelta:
        """总耗时"""
        return self.end_time - self.start_time

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "sync_id": self.sync_id,
            "success": self.success,
            "total_stocks": self.total_stocks,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "total_records": self.total_records,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "duration_seconds": self.duration.total_seconds(),
            "message": self.message,
            "successful_syncs": [result.to_dict() for result in self.successful_syncs],
            "failed_syncs": [result.to_dict() for result in self.failed_syncs],
        }


@dataclass
class SyncProgress:
    """同步进度"""

    sync_id: str
    total_stocks: int
    completed_stocks: int
    failed_stocks: int
    current_stock: Optional[str]
    progress_percentage: float
    estimated_remaining_time: Optional[timedelta]
    start_time: datetime
    status: SyncStatus
    last_update: datetime

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "sync_id": self.sync_id,
            "total_stocks": self.total_stocks,
            "completed_stocks": self.completed_stocks,
            "failed_stocks": self.failed_stocks,
            "current_stock": self.current_stock,
            "progress_percentage": self.progress_percentage,
            "estimated_remaining_time_seconds": self.estimated_remaining_time.total_seconds()
            if self.estimated_remaining_time
            else None,
            "start_time": self.start_time.isoformat(),
            "status": self.status.value,
            "last_update": self.last_update.isoformat(),
        }


@dataclass
class RetryResult:
    """重试结果"""

    sync_id: str
    retried_stocks: List[str]
    retry_results: List[SyncResult]
    success: bool
    message: str

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "sync_id": self.sync_id,
            "retried_stocks": self.retried_stocks,
            "retry_results": [result.to_dict() for result in self.retry_results],
            "success": self.success,
            "message": self.message,
        }


@dataclass
class SyncHistoryEntry:
    """同步历史记录"""

    sync_id: str
    request: BatchSyncRequest
    result: BatchSyncResult
    created_at: datetime

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "sync_id": self.sync_id,
            "request": self.request.to_dict(),
            "result": self.result.to_dict(),
            "created_at": self.created_at.isoformat(),
        }
