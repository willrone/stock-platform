"""
简化的股票数据模型（不依赖pydantic）
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any


@dataclass
class StockData:
    """股票基础数据"""
    stock_code: str
    date: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    adj_close: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'stock_code': self.stock_code,
            'date': self.date.isoformat(),
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'adj_close': self.adj_close
        }


@dataclass
class DataServiceStatus:
    """数据服务状态"""
    service_url: str
    is_available: bool
    last_check: datetime
    response_time_ms: Optional[float] = None
    error_message: Optional[str] = None


@dataclass
class DataSyncRequest:
    """数据同步请求"""
    stock_codes: list[str]
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    force_update: bool = False


@dataclass
class DataSyncResponse:
    """数据同步响应"""
    success: bool
    synced_stocks: list[str]
    failed_stocks: list[str]
    total_records: int
    message: str