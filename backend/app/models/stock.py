"""
股票数据模型
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class StockData(BaseModel):
    """股票基础数据"""

    model_config = ConfigDict(json_encoders={datetime: lambda v: v.isoformat()})

    stock_code: str = Field(..., description="股票代码")
    date: datetime = Field(..., description="日期")
    open: float = Field(..., description="开盘价")
    high: float = Field(..., description="最高价")
    low: float = Field(..., description="最低价")
    close: float = Field(..., description="收盘价")
    volume: int = Field(..., description="成交量")
    adj_close: Optional[float] = Field(None, description="复权收盘价")


class TechnicalIndicator(BaseModel):
    """技术指标数据"""

    model_config = ConfigDict(json_encoders={datetime: lambda v: v.isoformat()})

    stock_code: str = Field(..., description="股票代码")
    date: datetime = Field(..., description="日期")
    indicator_name: str = Field(..., description="指标名称")
    value: float = Field(..., description="指标值")
    parameters: dict = Field(default_factory=dict, description="指标参数")


class DataServiceStatus(BaseModel):
    """数据服务状态"""

    service_url: str = Field(..., description="服务URL")
    is_available: bool = Field(..., description="是否可用")
    last_check: datetime = Field(..., description="最后检查时间")
    response_time_ms: Optional[float] = Field(None, description="响应时间(毫秒)")
    error_message: Optional[str] = Field(None, description="错误信息")


class DataSyncRequest(BaseModel):
    """数据同步请求"""

    stock_codes: list[str] = Field(..., description="股票代码列表")
    start_date: Optional[datetime] = Field(None, description="开始日期")
    end_date: Optional[datetime] = Field(None, description="结束日期")
    force_update: bool = Field(False, description="是否强制更新")
    sync_mode: Optional[str] = Field("incremental", description="同步模式")
    max_concurrent: Optional[int] = Field(3, description="最大并发数")
    retry_count: Optional[int] = Field(3, description="重试次数")


class DataSyncResponse(BaseModel):
    """数据同步响应"""

    success: bool = Field(..., description="是否成功")
    synced_stocks: list[str] = Field(default_factory=list, description="已同步的股票")
    failed_stocks: list[str] = Field(default_factory=list, description="同步失败的股票")
    total_records: int = Field(0, description="总记录数")
    message: str = Field("", description="消息")
