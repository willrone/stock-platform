"""
API请求和响应模型定义
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict


class StandardResponse(BaseModel):
    """标准响应格式"""
    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="响应消息")
    data: Optional[Any] = Field(None, description="响应数据")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="响应时间")
    
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat() if isinstance(v, datetime) else v
        },
        json_schema_extra={
            "example": {
                "success": True,
                "message": "操作成功",
                "data": {},
                "timestamp": "2025-01-01T12:00:00"
            }
        }
    )
    
    def model_dump_json(self, **kwargs):
        """自定义JSON序列化，确保datetime正确序列化"""
        import json
        from datetime import datetime
        
        def json_serial(obj):
            """JSON序列化辅助函数"""
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Type {type(obj)} not serializable")
        
        data = self.model_dump(**kwargs)
        return json.dumps(data, default=json_serial, ensure_ascii=False)


class StockDataRequest(BaseModel):
    """股票数据请求"""
    stock_code: str = Field(..., description="股票代码")
    start_date: datetime = Field(..., description="开始日期")
    end_date: datetime = Field(..., description="结束日期")


class PredictionRequest(BaseModel):
    """预测请求"""
    stock_codes: List[str] = Field(..., description="股票代码列表")
    model_id: str = Field(..., description="模型ID")
    horizon: str = Field(default="short_term", description="预测时间维度")
    confidence_level: float = Field(default=0.95, description="置信水平")


class TaskCreateRequest(BaseModel):
    """任务创建请求"""
    task_name: str = Field(..., description="任务名称")
    task_type: str = Field(default="prediction", description="任务类型: prediction 或 backtest")
    stock_codes: List[str] = Field(..., description="股票代码列表")
    model_id: Optional[str] = Field(None, description="使用的模型ID（预测任务必需）")
    prediction_config: Optional[Dict[str, Any]] = Field(default=None, description="预测配置")
    backtest_config: Optional[Dict[str, Any]] = Field(default=None, description="回测配置")


class BacktestRequest(BaseModel):
    """回测请求"""
    strategy_name: str = Field(..., description="策略名称")
    stock_codes: List[str] = Field(..., description="股票代码列表")
    start_date: datetime = Field(..., description="回测开始日期")
    end_date: datetime = Field(..., description="回测结束日期")
    initial_cash: float = Field(default=100000.0, description="初始资金")


class ModelTrainingRequest(BaseModel):
    """模型训练请求"""
    model_name: str = Field(..., description="模型名称")
    model_type: str = Field(default="random_forest", description="模型类型")
    stock_codes: List[str] = Field(..., description="训练数据股票代码列表")
    start_date: str = Field(..., description="训练数据开始日期 (YYYY-MM-DD)")
    end_date: str = Field(..., description="训练数据结束日期 (YYYY-MM-DD)")
    hyperparameters: Dict[str, Any] = Field(default={}, description="超参数")
    description: Optional[str] = Field(None, description="模型描述")
    parent_model_id: Optional[str] = Field(None, description="父模型ID，用于创建新版本")
    enable_hyperparameter_tuning: bool = Field(default=False, description="是否启用超参数调优")
    hyperparameter_search_strategy: str = Field(default="random_search", description="超参数搜索策略")
    hyperparameter_search_trials: int = Field(default=10, description="超参数搜索试验次数")


class RemoteDataSyncRequest(BaseModel):
    """远端数据同步请求"""
    stock_codes: Optional[List[str]] = Field(default=None, description="要同步的股票代码列表，如果为空则同步所有股票")

