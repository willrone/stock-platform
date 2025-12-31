"""
API路由定义

定义所有API端点和路由规则，实现请求验证和响应格式统一。
支持OpenAPI文档自动生成和API版本管理。
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Request, Query
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from pydantic import BaseModel, Field, ConfigDict
import random

import logging

# 导入依赖注入函数
from app.core.container import (
    get_data_service, 
    get_indicators_service, 
    get_parquet_manager,
    get_data_sync_engine,
    get_monitoring_service
)
from app.services.data_service import StockDataService
from app.services.technical_indicators import TechnicalIndicatorCalculator
from app.services.parquet_manager import ParquetManager
from app.models.stock import DataSyncRequest

logger = logging.getLogger(__name__)

# 创建路由器（不设置前缀，在主应用中设置）
api_router = APIRouter(
    tags=["股票预测平台API"],
    responses={
        400: {"description": "请求参数错误"},
        401: {"description": "未授权访问"},
        403: {"description": "权限不足"},
        404: {"description": "资源不存在"},
        429: {"description": "请求过于频繁"},
        500: {"description": "服务器内部错误"},
    }
)


# 请求/响应模型定义

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
    stock_codes: List[str] = Field(..., description="股票代码列表")
    model_id: str = Field(..., description="使用的模型ID")
    prediction_config: Dict[str, Any] = Field(default={}, description="预测配置")


class BacktestRequest(BaseModel):
    """回测请求"""
    strategy_name: str = Field(..., description="策略名称")
    stock_codes: List[str] = Field(..., description="股票代码列表")
    start_date: datetime = Field(..., description="回测开始日期")
    end_date: datetime = Field(..., description="回测结束日期")
    initial_cash: float = Field(default=100000.0, description="初始资金")


class StandardResponse(BaseModel):
    """标准响应格式"""
    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="响应消息")
    data: Optional[Any] = Field(None, description="响应数据")
    timestamp: datetime = Field(default_factory=datetime.now, description="响应时间")
    
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )


# 数据服务相关路由

@api_router.get("/health", response_model=StandardResponse, summary="健康检查", description="检查API服务运行状态")
async def health_check():
    """
    健康检查端点
    
    返回API服务的运行状态和版本信息。
    用于监控系统和负载均衡器检查服务可用性。
    
    Returns:
        StandardResponse: 包含服务状态信息
    """
    return StandardResponse(
        success=True,
        message="API服务运行正常",
        data={"status": "healthy", "version": "1.0.0"}
    )


@api_router.get(
    "/stocks/data", 
    response_model=StandardResponse,
    summary="获取股票数据",
    description="根据股票代码和时间范围获取历史价格数据"
)
async def get_stock_data(
    stock_code: str,
    start_date: datetime,
    end_date: datetime,
    data_service: StockDataService = Depends(get_data_service)
):
    """
    获取股票历史数据
    
    根据指定的股票代码和时间范围，从数据服务获取股票的历史价格数据。
    数据包括开盘价、最高价、最低价、收盘价和成交量。
    
    Args:
        stock_code: 股票代码，支持沪深两市格式（如000001.SZ, 600000.SH）
        start_date: 数据开始日期
        end_date: 数据结束日期
        data_service: 注入的数据服务
        
    Returns:
        StandardResponse: 包含股票历史数据
        
    Raises:
        HTTPException: 当股票代码无效或数据获取失败时
    """
    try:
        # 调用真实的数据服务
        stock_data = await data_service.get_stock_data(stock_code, start_date, end_date)
        
        if not stock_data:
            return StandardResponse(
                success=False,
                message=f"未找到股票 {stock_code} 在指定时间范围内的数据",
                data=None
            )
        
        # 转换数据格式
        data_points = []
        for item in stock_data:
            data_points.append({
                "date": item.date.isoformat(),
                "open": item.open,
                "high": item.high,
                "low": item.low,
                "close": item.close,
                "volume": item.volume,
                "adj_close": item.adj_close
            })
        
        response_data = {
            "stock_code": stock_code,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "data_points": len(data_points),
            "data": data_points
        }
        
        return StandardResponse(
            success=True,
            message="股票数据获取成功",
            data=response_data
        )
        
    except Exception as e:
        logger.error(f"获取股票数据失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取股票数据失败: {str(e)}")


@api_router.get("/stocks/{stock_code}/indicators", response_model=StandardResponse)
async def get_technical_indicators(
    stock_code: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    indicators: Optional[str] = Query(default="MA5,MA10,MA20,RSI,MACD", description="指标列表，逗号分隔"),
    data_service: StockDataService = Depends(get_data_service),
    indicators_service: TechnicalIndicatorCalculator = Depends(get_indicators_service)
):
    """获取技术指标"""
    try:
        if not start_date:
            start_date = datetime.now() - timedelta(days=60)
        if not end_date:
            end_date = datetime.now()
        
        # 解析指标列表
        indicator_list = [ind.strip() for ind in indicators.split(',')]
        
        # 获取股票数据
        stock_data = await data_service.get_stock_data(stock_code, start_date, end_date)
        
        if not stock_data:
            return StandardResponse(
                success=False,
                message=f"未找到股票 {stock_code} 的数据",
                data=None
            )
        
        # 计算技术指标
        indicator_results = indicators_service.calculate_indicators(stock_data, indicator_list)
        
        # 格式化结果
        formatted_results = []
        for result in indicator_results:
            formatted_results.append(result.to_dict())
        
        # 获取最新的指标值
        latest_indicators = {}
        if indicator_results:
            latest_result = indicator_results[-1]
            latest_indicators = latest_result.indicators
        
        response_data = {
            "stock_code": stock_code,
            "calculation_period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            },
            "indicators": latest_indicators,
            "calculation_date": end_date.isoformat(),
            "total_data_points": len(indicator_results),
            "detailed_results": formatted_results
        }
        
        return StandardResponse(
            success=True,
            message="技术指标计算成功",
            data=response_data
        )
        
    except ValueError as e:
        logger.error(f"技术指标计算参数错误: {e}")
        raise HTTPException(status_code=400, detail=f"参数错误: {str(e)}")
    except Exception as e:
        logger.error(f"计算技术指标失败: {e}")
        raise HTTPException(status_code=500, detail=f"计算技术指标失败: {str(e)}")


# 预测服务相关路由

@api_router.post("/predictions", response_model=StandardResponse)
async def create_prediction(request: PredictionRequest):
    """创建预测任务"""
    try:
        # 这里应该调用预测引擎
        # prediction_engine = get_prediction_engine()
        # results = await prediction_engine.predict_multiple_stocks(...)
        
        # 模拟预测结果
        mock_results = []
        for stock_code in request.stock_codes:
            mock_results.append({
                "stock_code": stock_code,
                "predicted_direction": 1,
                "predicted_return": 0.05,
                "confidence_score": 0.75,
                "confidence_interval": {"lower": 0.02, "upper": 0.08},
                "risk_assessment": {
                    "value_at_risk": -0.03,
                    "volatility": 0.2
                }
            })
        
        return StandardResponse(
            success=True,
            message=f"成功预测 {len(request.stock_codes)} 只股票",
            data={
                "predictions": mock_results,
                "model_id": request.model_id,
                "horizon": request.horizon
            }
        )
        
    except Exception as e:
        logger.error(f"创建预测失败: {e}")
        raise HTTPException(status_code=500, detail=f"创建预测失败: {str(e)}")


@api_router.get("/predictions/{prediction_id}", response_model=StandardResponse)
async def get_prediction_result(prediction_id: str):
    """获取预测结果"""
    try:
        # 模拟预测结果查询
        mock_result = {
            "prediction_id": prediction_id,
            "status": "completed",
            "created_at": datetime.now().isoformat(),
            "results": [
                {
                    "stock_code": "000001.SZ",
                    "predicted_direction": 1,
                    "confidence_score": 0.82
                }
            ]
        }
        
        return StandardResponse(
            success=True,
            message="预测结果获取成功",
            data=mock_result
        )
        
    except Exception as e:
        logger.error(f"获取预测结果失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取预测结果失败: {str(e)}")


# 任务管理相关路由

@api_router.post("/tasks", response_model=StandardResponse)
async def create_task(request: TaskCreateRequest, background_tasks: BackgroundTasks):
    """创建预测任务"""
    try:
        # 生成任务ID
        task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 这里应该调用任务管理服务
        # task_manager = get_task_manager()
        # task = await task_manager.create_task(...)
        
        # 模拟任务创建
        mock_task = {
            "task_id": task_id,
            "task_name": request.task_name,
            "status": "created",
            "stock_codes": request.stock_codes,
            "model_id": request.model_id,
            "created_at": datetime.now().isoformat(),
            "estimated_completion": (datetime.now() + timedelta(minutes=10)).isoformat()
        }
        
        # 添加后台任务来执行预测
        # background_tasks.add_task(execute_prediction_task, task_id)
        
        return StandardResponse(
            success=True,
            message="任务创建成功",
            data=mock_task
        )
        
    except Exception as e:
        logger.error(f"创建任务失败: {e}")
        raise HTTPException(status_code=500, detail=f"创建任务失败: {str(e)}")


@api_router.get("/tasks", response_model=StandardResponse)
async def list_tasks(
    status: Optional[str] = None,
    limit: int = 20,
    offset: int = 0
):
    """获取任务列表"""
    try:
        # 模拟任务列表
        mock_tasks = [
            {
                "task_id": f"task_202301{i:02d}",
                "task_name": f"预测任务 {i}",
                "status": "completed" if i % 2 == 0 else "running",
                "created_at": (datetime.now() - timedelta(days=i)).isoformat(),
                "progress": 100 if i % 2 == 0 else 50
            }
            for i in range(1, 11)
        ]
        
        # 应用状态过滤
        if status:
            mock_tasks = [task for task in mock_tasks if task["status"] == status]
        
        # 应用分页
        paginated_tasks = mock_tasks[offset:offset + limit]
        
        return StandardResponse(
            success=True,
            message="任务列表获取成功",
            data={
                "tasks": paginated_tasks,
                "total": len(mock_tasks),
                "limit": limit,
                "offset": offset
            }
        )
        
    except Exception as e:
        logger.error(f"获取任务列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取任务列表失败: {str(e)}")


@api_router.get("/tasks/{task_id}", response_model=StandardResponse)
async def get_task_detail(task_id: str):
    """获取任务详情"""
    try:
        # 模拟任务详情
        mock_task_detail = {
            "task_id": task_id,
            "task_name": "股票预测任务",
            "status": "completed",
            "progress": 100,
            "created_at": datetime.now().isoformat(),
            "completed_at": datetime.now().isoformat(),
            "results": {
                "total_stocks": 5,
                "successful_predictions": 5,
                "average_confidence": 0.78,
                "predictions": [
                    {
                        "stock_code": "000001.SZ",
                        "predicted_direction": 1,
                        "confidence_score": 0.85
                    }
                ]
            }
        }
        
        return StandardResponse(
            success=True,
            message="任务详情获取成功",
            data=mock_task_detail
        )
        
    except Exception as e:
        logger.error(f"获取任务详情失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取任务详情失败: {str(e)}")


# 回测相关路由

@api_router.post("/backtest", response_model=StandardResponse)
async def run_backtest(request: BacktestRequest):
    """运行回测"""
    try:
        # 这里应该调用回测服务
        # backtest_service = get_backtest_service()
        # result = await backtest_service.run_strategy_backtest(...)
        
        # 模拟回测结果
        mock_backtest_result = {
            "strategy_name": request.strategy_name,
            "period": {
                "start_date": request.start_date.isoformat(),
                "end_date": request.end_date.isoformat()
            },
            "portfolio": {
                "initial_cash": request.initial_cash,
                "final_value": request.initial_cash * 1.15,
                "total_return": 0.15,
                "annualized_return": 0.18
            },
            "risk_metrics": {
                "max_drawdown": -0.08,
                "sharpe_ratio": 1.25,
                "volatility": 0.16
            },
            "trading_stats": {
                "total_trades": 25,
                "win_rate": 0.64,
                "profit_factor": 1.8
            }
        }
        
        return StandardResponse(
            success=True,
            message="回测执行成功",
            data=mock_backtest_result
        )
        
    except Exception as e:
        logger.error(f"回测执行失败: {e}")
        raise HTTPException(status_code=500, detail=f"回测执行失败: {str(e)}")


# 模型管理相关路由

@api_router.get("/models", response_model=StandardResponse)
async def list_models():
    """获取模型列表"""
    try:
        # 模拟模型列表
        mock_models = [
            {
                "model_id": "xgboost_v1",
                "model_name": "XGBoost基线模型",
                "model_type": "xgboost",
                "version": "1.0",
                "accuracy": 0.75,
                "created_at": "2023-01-01T00:00:00",
                "status": "active"
            },
            {
                "model_id": "lstm_v1",
                "model_name": "LSTM深度学习模型",
                "model_type": "lstm",
                "version": "1.0",
                "accuracy": 0.78,
                "created_at": "2023-01-02T00:00:00",
                "status": "active"
            },
            {
                "model_id": "transformer_v1",
                "model_name": "Transformer模型",
                "model_type": "transformer",
                "version": "1.0",
                "accuracy": 0.82,
                "created_at": "2023-01-03T00:00:00",
                "status": "active"
            }
        ]
        
        return StandardResponse(
            success=True,
            message="模型列表获取成功",
            data={"models": mock_models}
        )
        
    except Exception as e:
        logger.error(f"获取模型列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取模型列表失败: {str(e)}")


@api_router.get("/models/{model_id}", response_model=StandardResponse)
async def get_model_detail(model_id: str):
    """获取模型详情"""
    try:
        # 模拟模型详情
        mock_model_detail = {
            "model_id": model_id,
            "model_name": "XGBoost基线模型",
            "model_type": "xgboost",
            "version": "1.0",
            "description": "基于XGBoost算法的股票预测模型",
            "performance_metrics": {
                "accuracy": 0.75,
                "precision": 0.73,
                "recall": 0.77,
                "f1_score": 0.75,
                "sharpe_ratio": 1.2,
                "max_drawdown": -0.12
            },
            "training_info": {
                "training_data_period": "2020-01-01 to 2022-12-31",
                "training_stocks": 100,
                "training_samples": 50000,
                "training_duration": "2 hours"
            },
            "created_at": "2023-01-01T00:00:00",
            "status": "active"
        }
        
        return StandardResponse(
            success=True,
            message="模型详情获取成功",
            data=mock_model_detail
        )
        
    except Exception as e:
        logger.error(f"获取模型详情失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取模型详情失败: {str(e)}")


# 数据管理相关路由

@api_router.get("/data/status", response_model=StandardResponse, summary="获取数据服务状态", description="获取远端数据服务连接状态和响应时间")
async def get_data_service_status(data_service: StockDataService = Depends(get_data_service)):
    """获取数据服务状态"""
    try:
        # 调用真实的数据服务状态检查
        status = await data_service.check_remote_service_status()

        # 转换响应格式以匹配前端期望
        response_data = {
            "service_url": status.service_url,
            "is_connected": status.is_available,
            "last_check": status.last_check.isoformat() if status.last_check else None,
            "response_time": status.response_time_ms,
            "error_message": status.error_message,
            "status_details": {
                "service_type": "远端数据服务",
                "protocol": "HTTP",
                "timeout_ms": data_service.timeout * 1000,
                "connection_pool": {
                    "max_connections": 10,
                    "max_keepalive": 5
                }
            }
        }

        return StandardResponse(
            success=status.is_available,
            message="数据服务状态检查完成" if status.is_available else f"数据服务不可用: {status.error_message}",
            data=response_data
        )

    except Exception as e:
        logger.error(f"获取数据服务状态失败: {e}")
        # 返回错误响应，但不抛出异常
        return StandardResponse(
            success=False,
            message=f"获取数据服务状态失败: {str(e)}",
            data={
                "service_url": "unknown",
                "is_connected": False,
                "last_check": datetime.now().isoformat(),
                "response_time": None,
                "error_message": str(e)
            }
        )


@api_router.get("/data/files", response_model=StandardResponse, summary="获取本地数据文件列表", description="获取本地Parquet文件的详细信息")
async def get_local_data_files(
    stock_code: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    min_size: Optional[int] = None,
    max_size: Optional[int] = None,
    integrity_status: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    parquet_manager: ParquetManager = Depends(get_parquet_manager)
):
    """获取本地数据文件列表"""
    try:
        from app.models.file_management import FileFilters, IntegrityStatus
        
        # 构建过滤条件
        filters = FileFilters(
            stock_code=stock_code,
            start_date=start_date,
            end_date=end_date,
            min_size=min_size,
            max_size=max_size,
            integrity_status=IntegrityStatus(integrity_status) if integrity_status else None,
            limit=limit,
            offset=offset
        )
        
        # 调用增强的Parquet管理器
        detailed_files = parquet_manager.get_detailed_file_list(filters)
        
        # 转换为API响应格式
        files_data = []
        for file_info in detailed_files:
            files_data.append({
                "file_path": file_info.file_path,
                "stock_code": file_info.stock_code,
                "date_range": {
                    "start": file_info.date_range[0].isoformat(),
                    "end": file_info.date_range[1].isoformat()
                },
                "record_count": file_info.record_count,
                "file_size": file_info.file_size,
                "last_modified": file_info.last_modified.isoformat(),
                "integrity_status": file_info.integrity_status.value,
                "compression_ratio": file_info.compression_ratio,
                "created_at": file_info.created_at.isoformat() if file_info.created_at else None
            })
        
        return StandardResponse(
            success=True,
            message="本地数据文件列表获取成功",
            data={
                "files": files_data,
                "total": len(files_data),
                "limit": limit,
                "offset": offset,
                "filters_applied": {
                    "stock_code": stock_code,
                    "date_range": {
                        "start": start_date.isoformat() if start_date else None,
                        "end": end_date.isoformat() if end_date else None
                    },
                    "size_range": {
                        "min": min_size,
                        "max": max_size
                    },
                    "integrity_status": integrity_status
                }
            }
        )
        
    except Exception as e:
        logger.error(f"获取本地数据文件列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取本地数据文件列表失败: {str(e)}")


@api_router.get("/data/stats", response_model=StandardResponse, summary="获取数据统计信息", description="获取本地数据存储的统计信息")
async def get_data_statistics(parquet_manager: ParquetManager = Depends(get_parquet_manager)):
    """获取数据统计信息"""
    try:
        # 调用真实统计功能
        comprehensive_stats = parquet_manager.get_comprehensive_stats()
        
        # 转换为API响应格式
        stats_data = {
            "total_files": comprehensive_stats.total_files,
            "total_size_bytes": comprehensive_stats.total_size_bytes,
            "total_size_mb": round(comprehensive_stats.total_size_bytes / 1024 / 1024, 2),
            "total_records": comprehensive_stats.total_records,
            "stock_count": comprehensive_stats.stock_count,
            "date_range": {
                "start": comprehensive_stats.date_range[0].isoformat(),
                "end": comprehensive_stats.date_range[1].isoformat()
            },
            "average_file_size_bytes": comprehensive_stats.average_file_size,
            "average_file_size_mb": round(comprehensive_stats.average_file_size / 1024 / 1024, 2),
            "storage_efficiency": comprehensive_stats.storage_efficiency,
            "last_sync_time": comprehensive_stats.last_sync_time.isoformat() if comprehensive_stats.last_sync_time else None,
            "top_stocks_by_size": [
                {
                    "stock_code": stock_code,
                    "size_bytes": size,
                    "size_mb": round(size / 1024 / 1024, 2)
                }
                for stock_code, size in comprehensive_stats.stocks_by_size
            ],
            "monthly_distribution": comprehensive_stats.monthly_distribution,
            "data_quality_indicators": {
                "records_per_mb": round(comprehensive_stats.storage_efficiency, 2),
                "average_compression_ratio": round(comprehensive_stats.average_file_size / (comprehensive_stats.total_records * 64) if comprehensive_stats.total_records > 0 else 0, 3)
            }
        }
        
        return StandardResponse(
            success=True,
            message="数据统计信息获取成功",
            data=stats_data
        )
        
    except Exception as e:
        logger.error(f"获取数据统计信息失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取数据统计信息失败: {str(e)}")
        logger.error(f"获取数据统计信息失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取数据统计信息失败: {str(e)}")


@api_router.post("/data/sync", response_model=StandardResponse, summary="同步数据", description="从远端服务同步指定股票的数据")
async def sync_data_from_remote(
    request: DataSyncRequest,
    sync_engine: 'DataSyncEngine' = Depends(get_data_sync_engine)
):
    """同步数据"""
    try:
        from app.models.sync_models import BatchSyncRequest, SyncMode
        
        # 构建批量同步请求
        sync_request = BatchSyncRequest(
            stock_codes=request.stock_codes,
            start_date=request.start_date,
            end_date=request.end_date,
            force_update=request.force_update,
            sync_mode=SyncMode(request.sync_mode or "incremental"),
            max_concurrent=request.max_concurrent or 3,
            retry_count=request.retry_count or 3
        )
        
        # 调用数据同步引擎
        batch_result = await sync_engine.sync_stocks_batch(sync_request)
        
        # 转换为API响应格式
        sync_data = {
            "sync_id": batch_result.sync_id,
            "success": batch_result.success,
            "total_stocks": batch_result.total_stocks,
            "success_count": batch_result.success_count,
            "failure_count": batch_result.failure_count,
            "total_records": batch_result.total_records,
            "duration_seconds": batch_result.duration.total_seconds(),
            "start_time": batch_result.start_time.isoformat(),
            "end_time": batch_result.end_time.isoformat(),
            "message": batch_result.message,
            "successful_syncs": [
                {
                    "stock_code": result.stock_code,
                    "records_synced": result.records_synced,
                    "duration_seconds": result.duration.total_seconds(),
                    "data_range": {
                        "start": result.data_range[0].isoformat(),
                        "end": result.data_range[1].isoformat()
                    } if result.data_range else None
                }
                for result in batch_result.successful_syncs
            ],
            "failed_syncs": [
                {
                    "stock_code": result.stock_code,
                    "error_message": result.error_message,
                    "duration_seconds": result.duration.total_seconds()
                }
                for result in batch_result.failed_syncs
            ]
        }
        
        return StandardResponse(
            success=batch_result.success,
            message=batch_result.message,
            data=sync_data
        )
        
    except Exception as e:
        logger.error(f"数据同步失败: {e}")
        raise HTTPException(status_code=500, detail=f"数据同步失败: {str(e)}")


@api_router.get("/data/sync/{sync_id}/progress", response_model=StandardResponse, summary="获取同步进度", description="获取指定同步任务的进度信息")
async def get_sync_progress(
    sync_id: str,
    sync_engine: 'DataSyncEngine' = Depends(get_data_sync_engine)
):
    """获取同步进度"""
    try:
        progress = await sync_engine.get_sync_progress(sync_id)
        
        if not progress:
            return StandardResponse(
                success=False,
                message=f"未找到同步任务: {sync_id}",
                data=None
            )
        
        progress_data = {
            "sync_id": progress.sync_id,
            "total_stocks": progress.total_stocks,
            "completed_stocks": progress.completed_stocks,
            "failed_stocks": progress.failed_stocks,
            "current_stock": progress.current_stock,
            "progress_percentage": progress.progress_percentage,
            "estimated_remaining_time_seconds": progress.estimated_remaining_time.total_seconds() if progress.estimated_remaining_time else None,
            "start_time": progress.start_time.isoformat(),
            "status": progress.status.value,
            "last_update": progress.last_update.isoformat()
        }
        
        return StandardResponse(
            success=True,
            message="同步进度获取成功",
            data=progress_data
        )
        
    except Exception as e:
        logger.error(f"获取同步进度失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取同步进度失败: {str(e)}")


@api_router.get("/data/sync/history", response_model=StandardResponse, summary="获取同步历史", description="获取数据同步历史记录")
async def get_sync_history(
    limit: int = 50,
    sync_engine: 'DataSyncEngine' = Depends(get_data_sync_engine)
):
    """获取同步历史"""
    try:
        history_entries = sync_engine.get_sync_history(limit)
        
        history_data = []
        for entry in history_entries:
            history_data.append({
                "sync_id": entry.sync_id,
                "request": {
                    "stock_codes": entry.request.stock_codes,
                    "start_date": entry.request.start_date.isoformat() if entry.request.start_date else None,
                    "end_date": entry.request.end_date.isoformat() if entry.request.end_date else None,
                    "force_update": entry.request.force_update,
                    "sync_mode": entry.request.sync_mode.value,
                    "max_concurrent": entry.request.max_concurrent,
                    "retry_count": entry.request.retry_count
                },
                "result": {
                    "success": entry.result.success,
                    "total_stocks": entry.result.total_stocks,
                    "success_count": entry.result.success_count,
                    "failure_count": entry.result.failure_count,
                    "total_records": entry.result.total_records,
                    "duration_seconds": entry.result.duration.total_seconds(),
                    "message": entry.result.message
                },
                "created_at": entry.created_at.isoformat()
            })
        
        return StandardResponse(
            success=True,
            message="同步历史获取成功",
            data={
                "history": history_data,
                "total": len(history_data),
                "limit": limit
            }
        )
        
    except Exception as e:
        logger.error(f"获取同步历史失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取同步历史失败: {str(e)}")


@api_router.post("/data/sync/{sync_id}/retry", response_model=StandardResponse, summary="重试失败的同步", description="重试指定同步任务中失败的股票")
async def retry_failed_syncs(
    sync_id: str,
    sync_engine: 'DataSyncEngine' = Depends(get_data_sync_engine)
):
    """重试失败的同步"""
    try:
        retry_result = await sync_engine.retry_failed_syncs(sync_id)
        
        retry_data = {
            "sync_id": retry_result.sync_id,
            "retried_stocks": retry_result.retried_stocks,
            "retry_results": [
                {
                    "stock_code": result.stock_code,
                    "success": result.success,
                    "records_synced": result.records_synced,
                    "error_message": result.error_message
                }
                for result in retry_result.retry_results
            ],
            "success": retry_result.success,
            "message": retry_result.message
        }
        
        return StandardResponse(
            success=retry_result.success,
            message=retry_result.message,
            data=retry_data
        )
        
    except Exception as e:
        logger.error(f"重试同步失败: {e}")
        raise HTTPException(status_code=500, detail=f"重试同步失败: {str(e)}")


@api_router.delete("/data/files", response_model=StandardResponse, summary="删除数据文件", description="删除指定的本地数据文件")
async def delete_data_files(
    file_paths: List[str] = Query(..., description="要删除的文件路径列表"),
    parquet_manager: ParquetManager = Depends(get_parquet_manager)
):
    """删除数据文件"""
    try:
        # 调用安全文件删除功能
        deletion_result = parquet_manager.delete_files_safely(file_paths)
        
        # 转换为API响应格式
        deletion_data = {
            "success": deletion_result.success,
            "deleted_files": deletion_result.deleted_files,
            "failed_files": [
                {
                    "file_path": file_path,
                    "error": error
                }
                for file_path, error in deletion_result.failed_files
            ],
            "total_deleted": deletion_result.total_deleted,
            "freed_space_bytes": deletion_result.freed_space_bytes,
            "freed_space_mb": round(deletion_result.freed_space_bytes / 1024 / 1024, 2),
            "message": deletion_result.message
        }
        
        return StandardResponse(
            success=deletion_result.success,
            message=deletion_result.message,
            data=deletion_data
        )
        
    except Exception as e:
        logger.error(f"删除数据文件失败: {e}")
        raise HTTPException(status_code=500, detail=f"删除数据文件失败: {str(e)}")


# 监控相关路由

@api_router.get("/monitoring/health", response_model=StandardResponse, summary="系统健康检查", description="获取所有服务的健康状态")
async def get_system_health(monitoring_service: 'DataMonitoringService' = Depends(get_monitoring_service)):
    """获取系统健康状态"""
    try:
        # 执行所有服务的健康检查
        services = ["data_service", "indicators_service", "parquet_manager", "sync_engine"]
        health_results = {}
        
        for service_name in services:
            try:
                health_status = await monitoring_service.check_service_health(service_name)
                health_results[service_name] = {
                    "healthy": health_status.is_healthy,
                    "response_time_ms": health_status.response_time_ms,
                    "last_check": health_status.last_check.isoformat(),
                    "error_message": health_status.error_message
                }
            except Exception as e:
                health_results[service_name] = {
                    "healthy": False,
                    "response_time_ms": 0,
                    "last_check": datetime.now().isoformat(),
                    "error_message": f"健康检查失败: {str(e)}"
                }
        
        # 计算整体健康状态
        overall_healthy = all(result["healthy"] for result in health_results.values())
        
        return StandardResponse(
            success=True,
            message="系统健康检查完成",
            data={
                "overall_healthy": overall_healthy,
                "services": health_results,
                "check_time": datetime.now().isoformat()
            }
        )
        
    except Exception as e:
        logger.error(f"系统健康检查失败: {e}")
        raise HTTPException(status_code=500, detail=f"系统健康检查失败: {str(e)}")


@api_router.get("/monitoring/metrics", response_model=StandardResponse, summary="性能指标", description="获取系统性能指标")
async def get_performance_metrics(
    service_name: Optional[str] = None,
    monitoring_service: 'DataMonitoringService' = Depends(get_monitoring_service)
):
    """获取性能指标"""
    try:
        if service_name:
            # 获取特定服务的指标
            metrics = monitoring_service.get_performance_metrics(service_name)
            if not metrics:
                return StandardResponse(
                    success=False,
                    message=f"未找到服务 {service_name} 的性能指标",
                    data=None
                )
            
            return StandardResponse(
                success=True,
                message="性能指标获取成功",
                data=metrics.to_dict()
            )
        else:
            # 获取所有服务的指标
            services = ["data_service", "indicators_service", "parquet_manager", "sync_engine"]
            all_metrics = {}
            
            for svc_name in services:
                metrics = monitoring_service.get_performance_metrics(svc_name)
                if metrics:
                    all_metrics[svc_name] = metrics.to_dict()
            
            return StandardResponse(
                success=True,
                message="性能指标获取成功",
                data={
                    "services": all_metrics,
                    "summary": {
                        "total_services": len(all_metrics),
                        "avg_response_time": sum(m["avg_response_time"] for m in all_metrics.values()) / len(all_metrics) if all_metrics else 0,
                        "total_requests": sum(m["request_count"] for m in all_metrics.values()),
                        "total_errors": sum(m["error_count"] for m in all_metrics.values())
                    }
                }
            )
        
    except Exception as e:
        logger.error(f"获取性能指标失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取性能指标失败: {str(e)}")


@api_router.get("/monitoring/overview", response_model=StandardResponse, summary="系统概览", description="获取系统整体概览信息")
async def get_system_overview(monitoring_service: 'DataMonitoringService' = Depends(get_monitoring_service)):
    """获取系统概览"""
    try:
        overview = monitoring_service.get_system_overview()
        
        return StandardResponse(
            success=True,
            message="系统概览获取成功",
            data=overview
        )
        
    except Exception as e:
        logger.error(f"获取系统概览失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取系统概览失败: {str(e)}")


@api_router.get("/monitoring/errors", response_model=StandardResponse, summary="错误统计", description="获取系统错误统计信息")
async def get_error_statistics(
    hours: int = 24,
    monitoring_service: 'DataMonitoringService' = Depends(get_monitoring_service)
):
    """获取错误统计"""
    try:
        error_stats = monitoring_service.get_error_statistics(hours)
        
        stats_data = []
        for stat in error_stats:
            stats_data.append({
                "error_type": stat.error_type,
                "count": stat.count,
                "last_occurrence": stat.last_occurrence.isoformat(),
                "sample_message": stat.sample_message
            })
        
        return StandardResponse(
            success=True,
            message="错误统计获取成功",
            data={
                "time_range_hours": hours,
                "total_error_types": len(stats_data),
                "total_errors": sum(stat["count"] for stat in stats_data),
                "error_statistics": stats_data
            }
        )
        
    except Exception as e:
        logger.error(f"获取错误统计失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取错误统计失败: {str(e)}")


@api_router.get("/monitoring/quality", response_model=StandardResponse, summary="数据质量检查", description="获取数据质量检查结果")
async def get_data_quality(monitoring_service: 'DataMonitoringService' = Depends(get_monitoring_service)):
    """获取数据质量检查结果"""
    try:
        quality_report = monitoring_service.check_data_quality()
        
        return StandardResponse(
            success=True,
            message="数据质量检查完成",
            data=quality_report
        )
        
    except Exception as e:
        logger.error(f"数据质量检查失败: {e}")
        raise HTTPException(status_code=500, detail=f"数据质量检查失败: {str(e)}")


@api_router.get("/monitoring/anomalies", response_model=StandardResponse, summary="异常检测", description="获取系统异常检测结果")
async def get_anomalies(monitoring_service: 'DataMonitoringService' = Depends(get_monitoring_service)):
    """获取异常检测结果"""
    try:
        anomalies = monitoring_service.detect_anomalies()
        
        # 按严重程度分组
        by_severity = {"high": [], "medium": [], "low": []}
        for anomaly in anomalies:
            severity = anomaly.get("severity", "low")
            by_severity[severity].append(anomaly)
        
        return StandardResponse(
            success=True,
            message="异常检测完成",
            data={
                "total_anomalies": len(anomalies),
                "by_severity": {
                    "high": len(by_severity["high"]),
                    "medium": len(by_severity["medium"]),
                    "low": len(by_severity["low"])
                },
                "anomalies": anomalies,
                "detection_time": datetime.now().isoformat()
            }
        )
        
    except Exception as e:
        logger.error(f"异常检测失败: {e}")
        raise HTTPException(status_code=500, detail=f"异常检测失败: {str(e)}")


# 系统状态相关路由

@api_router.get(
    "/version",
    response_model=StandardResponse,
    summary="获取API版本信息",
    description="获取当前API的版本信息和更新日志"
)
async def get_api_version():
    """
    获取API版本信息
    
    返回当前API的版本号、发布日期和主要功能特性。
    用于客户端版本兼容性检查。
    
    Returns:
        StandardResponse: 包含版本信息
    """
    version_info = {
        "version": "1.0.0",
        "release_date": "2025-01-01",
        "api_name": "股票预测平台API",
        "description": "基于AI的股票预测和回测分析平台",
        "features": [
            "股票数据获取",
            "技术指标计算", 
            "机器学习预测",
            "策略回测",
            "任务管理"
        ],
        "endpoints": {
            "total": 15,
            "categories": {
                "数据服务": 3,
                "预测服务": 2,
                "任务管理": 4,
                "回测服务": 1,
                "模型管理": 2,
                "系统状态": 3
            }
        },
        "changelog": {
            "1.0.0": [
                "初始版本发布",
                "实现基础API功能",
                "添加限流和错误处理",
                "完成API文档生成"
            ]
        }
    }
    
    return StandardResponse(
        success=True,
        message="API版本信息获取成功",
        data=version_info
    )


@api_router.get("/system/status", response_model=StandardResponse, summary="获取系统状态", description="获取各个服务组件的运行状态")
async def get_system_status(request: Request):
    """获取系统状态"""
    try:
        # 获取错误处理中间件的统计信息
        error_stats = {}
        for middleware in request.app.middleware_stack:
            if hasattr(middleware, 'cls') and middleware.cls.__name__ == 'ErrorHandlingMiddleware':
                if hasattr(middleware, 'kwargs') and 'app' in middleware.kwargs:
                    error_middleware = middleware.kwargs.get('error_middleware')
                    if error_middleware and hasattr(error_middleware, 'get_error_stats'):
                        error_stats = error_middleware.get_error_stats()
                break
        
        mock_system_status = {
            "api_server": {"status": "healthy", "uptime": "5 days"},
            "data_service": {"status": "healthy", "last_update": datetime.now().isoformat()},
            "prediction_engine": {"status": "healthy", "active_models": 3},
            "task_manager": {"status": "healthy", "running_tasks": 2},
            "database": {"status": "healthy", "connection": "active"},
            "remote_data_service": {"status": "healthy", "url": "192.168.3.62"},
            "error_statistics": error_stats
        }
        
        return StandardResponse(
            success=True,
            message="系统状态获取成功",
            data=mock_system_status
        )
        
    except Exception as e:
        logger.error(f"获取系统状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取系统状态失败: {str(e)}")


# 注意：异常处理器应该在主应用中注册，而不是在路由器中
# 这些异常处理器将在main.py中注册