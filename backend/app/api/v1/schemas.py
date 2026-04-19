"""
API请求和响应模型定义
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class StandardResponse(BaseModel):
    """标准响应格式"""

    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="响应消息")
    data: Optional[Any] = Field(None, description="响应数据")
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(), description="响应时间"
    )

    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat() if isinstance(v, datetime) else v
        },
        json_schema_extra={
            "example": {
                "success": True,
                "message": "操作成功",
                "data": {},
                "timestamp": "2025-01-01T12:00:00",
            }
        },
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
    task_type: str = Field(
        default="prediction", description="任务类型: prediction 或 backtest"
    )
    stock_codes: List[str] = Field(..., description="股票代码列表")
    model_id: Optional[str] = Field(None, description="使用的模型ID（预测任务必需）")
    prediction_config: Optional[Dict[str, Any]] = Field(
        default=None, description="预测配置"
    )
    backtest_config: Optional[Dict[str, Any]] = Field(default=None, description="回测配置")


class BacktestRequest(BaseModel):
    """回测请求（支持单策略、组合策略和模型驱动策略）"""

    strategy_name: str = Field(..., description="策略名称，单策略时使用策略名，组合策略时使用'portfolio'，模型驱动回测可使用'model'/'signal'/'model_signal'")
    stock_codes: List[str] = Field(..., description="股票代码列表")
    start_date: datetime = Field(..., description="回测开始日期")
    end_date: datetime = Field(..., description="回测结束日期")
    initial_cash: float = Field(default=100000.0, description="初始资金")
    model_id: Optional[str] = Field(None, description="模型驱动回测使用的模型ID")
    strategy_config: Optional[Dict[str, Any]] = Field(
        default=None, description="策略配置，单策略时为策略参数，组合策略时包含strategies列表；模型驱动回测可附带 model_id/horizon/buy_threshold/sell_threshold"
    )


class ModelTrainingRequest(BaseModel):
    """模型训练请求"""

    model_name: str = Field(..., description="模型名称")
    model_type: str = Field(default="random_forest", description="模型类型")
    stock_codes: List[str] = Field(..., description="训练数据股票代码列表")
    start_date: str = Field(..., description="训练数据开始日期 (YYYY-MM-DD)")
    end_date: str = Field(..., description="训练数据结束日期 (YYYY-MM-DD)")
    hyperparameters: Dict[str, Any] = Field(default={}, description="超参数")
    selected_features: Optional[List[str]] = Field(
        None, description="选择的特征列表，如果为空则使用所有可用特征"
    )
    description: Optional[str] = Field(None, description="模型描述")
    parent_model_id: Optional[str] = Field(None, description="父模型ID，用于创建新版本")
    enable_hyperparameter_tuning: bool = Field(default=False, description="是否启用超参数调优")
    hyperparameter_search_strategy: str = Field(
        default="random_search", description="超参数搜索策略"
    )
    hyperparameter_search_trials: int = Field(default=10, description="超参数搜索试验次数")
    workflow_mode: str = Field(
        default="enhanced_local",
        description="训练工作流模式: enhanced_local 或 official_replication",
    )
    official_dataset: Optional[str] = Field(
        default=None,
        description="官方复刻模式使用的数据集: alpha158 或 alpha360",
    )
    official_market: Optional[str] = Field(
        default=None,
        description="官方复刻模式使用的市场: csi300 或 csi500",
    )


class RemoteDataSyncRequest(BaseModel):
    """远端数据同步请求"""

    stock_codes: Optional[List[str]] = Field(
        default=None, description="要同步的股票代码列表，如果为空则同步所有股票"
    )


class QlibPrecomputeRequest(BaseModel):
    """Qlib预计算请求"""

    stock_codes: Optional[List[str]] = Field(
        default=None, description="股票代码列表（可选，None则处理所有股票）"
    )
    start_date: Optional[str] = Field(default=None, description="开始日期（可选，ISO格式字符串）")
    end_date: Optional[str] = Field(default=None, description="结束日期（可选，ISO格式字符串）")
    batch_size: int = Field(default=50, description="每批处理的股票数（默认50）")
    max_workers: Optional[int] = Field(default=None, description="最大并发数（可选，None则自动选择）")


class ParamSpaceConfig(BaseModel):
    """参数空间配置"""

    type: str = Field(..., description="参数类型: int, float, categorical")
    low: Optional[float] = Field(None, description="最小值（数值类型）")
    high: Optional[float] = Field(None, description="最大值（数值类型）")
    choices: Optional[List[Any]] = Field(None, description="可选值列表（分类类型）")
    default: Optional[Any] = Field(None, description="默认值")
    enabled: bool = Field(default=True, description="是否启用优化")
    log: bool = Field(default=False, description="是否使用对数尺度（数值类型）")


class ObjectiveConfig(BaseModel):
    """优化目标配置"""

    objective_metric: Any = Field(
        ...,
        description=(
            "目标指标: "
            "'sharpe' | 'calmar' | 'ic' | 'ic_ir' | "
            "'total_return' | 'annualized_return' | "
            "'win_rate' | 'profit_factor' | "
            "'max_drawdown' | 'cost' | 'custom' "
            "| ['sharpe', 'calmar', 'ic'] (多目标)"
        ),
    )
    direction: str = Field(default="maximize", description="优化方向: maximize 或 minimize")
    objective_weights: Optional[Dict[str, float]] = Field(
        None, description="自定义权重（custom 时使用）"
    )


class OptimizationConfig(BaseModel):
    """优化配置"""

    strategy_name: str = Field(..., description="策略名称")
    param_space: Dict[str, ParamSpaceConfig] = Field(..., description="参数空间")
    objective_config: ObjectiveConfig = Field(..., description="目标函数配置")
    n_trials: int = Field(
        default=50,
        ge=1,
        le=1000,
        description="试验次数，参数空间大时建议 300–500+，上限 1000",
    )
    optimization_method: str = Field(
        default="tpe", description="优化方法: tpe, random, grid, nsga2, motpe"
    )
    timeout: Optional[int] = Field(None, description="超时时间（秒）")


class HyperparameterOptimizationRequest(BaseModel):
    """超参优化任务创建请求"""

    task_name: str = Field(..., description="任务名称")
    strategy_name: str = Field(..., description="策略名称")
    stock_codes: List[str] = Field(..., description="股票代码列表")
    start_date: datetime = Field(..., description="回测开始日期")
    end_date: datetime = Field(..., description="回测结束日期")
    param_space: Dict[str, ParamSpaceConfig] = Field(..., description="参数空间")
    objective_config: ObjectiveConfig = Field(..., description="目标函数配置")
    n_trials: int = Field(
        default=50,
        ge=1,
        le=1000,
        description="试验次数，上限 1000",
    )
    optimization_method: str = Field(default="tpe", description="优化方法")
    timeout: Optional[int] = Field(None, description="超时时间（秒）")
    backtest_config: Optional[Dict[str, Any]] = Field(
        default=None, description="回测配置（初始资金、手续费等）"
    )


class BacktestCompareRequest(BaseModel):
    """回测对比请求"""

    task_ids: List[str] = Field(
        ..., description="要对比的任务ID列表", min_length=2, max_length=5
    )
    comparison_metrics: Optional[List[str]] = Field(default=None, description="指定对比的指标")


class BacktestExportRequest(BaseModel):
    """回测报告导出请求"""

    format: str = Field(..., description="导出格式: pdf 或 excel")
    include_charts: Optional[List[str]] = Field(default=None, description="包含的图表类型")
    include_tables: Optional[List[str]] = Field(default=None, description="包含的数据表格")
    include_raw_data: bool = Field(default=False, description="是否包含原始数据")


class RebuildTaskRequest(BaseModel):
    """任务重建请求"""
    task_name: Optional[str] = Field(None, description="新任务名称，默认为 [重建] {原名}")
    config_override: Optional[Dict[str, Any]] = Field(None, description="配置覆盖，深度合并到原 config")


class TaskPredictionDTO(BaseModel):
    """任务预测结果 DTO"""

    stock_code: str
    predicted_direction: int
    predicted_return: float = 0.0
    confidence_score: float
    confidence_interval: Dict[str, float] = Field(default_factory=dict)
    risk_assessment: Dict[str, Any] = Field(default_factory=dict)


class TaskResultsDTO(BaseModel):
    """任务聚合结果 DTO"""

    total_stocks: int
    successful_predictions: int
    average_confidence: float
    predictions: List[TaskPredictionDTO] = Field(default_factory=list)
    backtest_results: Optional[Any] = None


class TaskSummaryDTO(BaseModel):
    """任务摘要 DTO"""

    task_id: str
    task_name: str
    task_type: Optional[str] = None
    status: str
    progress: float
    stock_codes: List[str] = Field(default_factory=list)
    model_id: str = ""
    created_at: str
    completed_at: Optional[str] = None
    error_message: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    original_task_id: Optional[str] = None


class TaskDetailDTO(TaskSummaryDTO):
    """任务详情 DTO"""

    results: TaskResultsDTO
    backtest_results: Optional[Any] = None
    result: Optional[Any] = None
    optimization_info: Optional[Dict[str, Any]] = None


class TaskListDTO(BaseModel):
    """任务列表 DTO"""

    tasks: List[TaskSummaryDTO]
    total: int
    limit: int
    offset: int


class TaskMutationDTO(TaskSummaryDTO):
    """任务变更 DTO"""



def normalize_api_value(value: Any) -> Any:
    """归一化枚举等 API 值。"""

    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "value"):
        return value.value
    return value



def to_iso_datetime(value: Optional[datetime]) -> Optional[str]:
    """将 datetime 转为 ISO 字符串。"""

    if value is None:
        return None
    return value.isoformat()



def build_task_summary_dto(
    task: Any,
    *,
    config: Optional[Dict[str, Any]] = None,
    model_id: str = "",
    stock_codes: Optional[List[str]] = None,
    original_task_id: Optional[str] = None,
) -> TaskSummaryDTO:
    """构建任务摘要 DTO。"""

    task_config = config if config is not None else getattr(task, "config", None)
    normalized_stock_codes = stock_codes if isinstance(stock_codes, list) else []
    return TaskSummaryDTO(
        task_id=str(getattr(task, "task_id", "")),
        task_name=str(getattr(task, "task_name", "")),
        task_type=normalize_api_value(getattr(task, "task_type", None)),
        status=str(normalize_api_value(getattr(task, "status", ""))),
        progress=float(getattr(task, "progress", 0.0) or 0.0),
        stock_codes=normalized_stock_codes,
        model_id=model_id,
        created_at=to_iso_datetime(getattr(task, "created_at", None))
        or datetime.now().isoformat(),
        completed_at=to_iso_datetime(getattr(task, "completed_at", None)),
        error_message=getattr(task, "error_message", None),
        config=task_config,
        original_task_id=original_task_id,
    )



def build_task_list_dto(
    tasks: List[Any], total: int, limit: int, offset: int
) -> TaskListDTO:
    """构建任务列表 DTO。"""

    task_items = []
    for task in tasks:
        config = getattr(task, "config", None) or {}
        stock_codes = config.get("stock_codes", [])
        model_id = config.get("model_id", "")
        task_items.append(
            build_task_summary_dto(
                task,
                config=config,
                model_id=model_id,
                stock_codes=stock_codes if isinstance(stock_codes, list) else [],
            )
        )
    return TaskListDTO(tasks=task_items, total=total, limit=limit, offset=offset)



def build_task_mutation_dto(
    task: Any,
    *,
    config: Optional[Dict[str, Any]] = None,
    stock_codes: Optional[List[str]] = None,
    model_id: str = "",
    original_task_id: Optional[str] = None,
) -> TaskMutationDTO:
    """构建任务变更响应 DTO。"""

    summary = build_task_summary_dto(
        task,
        config=config,
        model_id=model_id,
        stock_codes=stock_codes,
        original_task_id=original_task_id,
    )
    return TaskMutationDTO(**summary.model_dump())



def build_task_detail_dto(
    task: Any,
    *,
    config: Dict[str, Any],
    stock_codes: List[str],
    model_id: str,
    predictions: List[Dict[str, Any]],
    average_confidence: float,
    backtest_results: Any,
    optimization_info: Optional[Dict[str, Any]],
) -> TaskDetailDTO:
    """构建任务详情 DTO。"""

    summary = build_task_summary_dto(
        task,
        config=config,
        model_id=model_id,
        stock_codes=stock_codes,
    )
    results = TaskResultsDTO(
        total_stocks=len(stock_codes),
        successful_predictions=len(predictions),
        average_confidence=average_confidence,
        predictions=[TaskPredictionDTO(**item) for item in predictions],
        backtest_results=backtest_results,
    )
    return TaskDetailDTO(
        **summary.model_dump(),
        results=results,
        backtest_results=backtest_results,
        result=backtest_results,
        optimization_info=optimization_info,
    )
