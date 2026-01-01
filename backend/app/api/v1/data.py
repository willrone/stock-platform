"""
数据管理路由
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Optional, List
from datetime import datetime
import logging

from app.api.v1.schemas import StandardResponse
from app.models.stock import DataSyncRequest
from app.core.container import get_data_service, get_parquet_manager, get_data_sync_engine
from app.services.data import DataService as StockDataService, ParquetManager
from app.core.database import SessionLocal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.services.data import DataSyncEngine

router = APIRouter(prefix="/data", tags=["数据管理"])
logger = logging.getLogger(__name__)


@router.get("/status", response_model=StandardResponse, summary="获取数据服务状态", description="获取远端数据服务连接状态和响应时间")
async def get_data_service_status(data_service: StockDataService = Depends(get_data_service)):
    """获取数据服务状态"""
    try:
        status = await data_service.check_remote_service_status()
        
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


@router.get("/remote/stocks", response_model=StandardResponse, summary="获取远端服务股票列表", description="从远端数据服务获取可用的股票列表")
async def get_remote_stock_list(data_service: StockDataService = Depends(get_data_service)):
    """获取远端服务的股票列表"""
    try:
        stocks = await data_service.get_remote_stock_list()
        
        if stocks is None:
            return StandardResponse(
                success=False,
                message="无法从远端服务获取股票列表",
                data={
                    "stocks": [],
                    "total_stocks": 0
                }
            )
        
        stock_codes = [stock.get("ts_code", "") for stock in stocks if stock.get("ts_code")]
        
        return StandardResponse(
            success=True,
            message=f"成功获取远端股票列表: {len(stocks)} 只股票",
            data={
                "stocks": stocks,
                "stock_codes": stock_codes,
                "total_stocks": len(stocks)
            }
        )
    
    except Exception as e:
        logger.error(f"获取远端股票列表失败: {e}")
        return StandardResponse(
            success=False,
            message=f"获取远端股票列表失败: {str(e)}",
            data={
                "stocks": [],
                "stock_codes": [],
                "total_stocks": 0
            }
        )


@router.get("/files", response_model=StandardResponse, summary="获取本地数据文件列表", description="获取本地Parquet文件的详细信息")
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
        
        detailed_files = parquet_manager.get_detailed_file_list(filters)
        
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


@router.get("/stats", response_model=StandardResponse, summary="获取数据统计信息", description="获取本地数据存储的统计信息")
async def get_data_statistics(parquet_manager: ParquetManager = Depends(get_parquet_manager)):
    """获取数据统计信息"""
    try:
        comprehensive_stats = parquet_manager.get_comprehensive_stats()
        
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


@router.post("/sync", response_model=StandardResponse, summary="同步数据", description="从远端服务同步指定股票的数据")
async def sync_data_from_remote(
    request: DataSyncRequest,
    sync_engine: 'DataSyncEngine' = Depends(get_data_sync_engine)
):
    """同步数据"""
    try:
        from app.models.sync_models import BatchSyncRequest, SyncMode
        
        sync_request = BatchSyncRequest(
            stock_codes=request.stock_codes,
            start_date=request.start_date,
            end_date=request.end_date,
            force_update=request.force_update,
            sync_mode=SyncMode(request.sync_mode or "incremental"),
            max_concurrent=request.max_concurrent or 3,
            retry_count=request.retry_count or 3
        )
        
        batch_result = await sync_engine.sync_stocks_batch(sync_request)
        
        # 计算总记录数
        total_records = sum(r.records_synced for r in batch_result.successful_syncs)
        duration_seconds = (batch_result.end_time - batch_result.start_time).total_seconds()
        
        sync_data = {
            "sync_id": batch_result.sync_id,
            "total_stocks": batch_result.total_stocks,
            "success_count": len(batch_result.successful_syncs),
            "failure_count": len(batch_result.failed_syncs),
            "total_records": total_records,
            "sync_duration_seconds": duration_seconds,
            "message": batch_result.message,
            "successful_syncs": [
                {
                    "stock_code": result.stock_code,
                    "success": result.success,
                    "records_synced": result.records_synced,
                    "data_range": {
                        "start": result.data_range[0].isoformat() if result.data_range else None,
                        "end": result.data_range[1].isoformat() if result.data_range else None
                    } if result.data_range else None
                }
                for result in batch_result.successful_syncs
            ],
            "failed_syncs": [
                {
                    "stock_code": result.stock_code,
                    "success": result.success,
                    "records_synced": result.records_synced,
                    "error_message": result.error_message
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
        logger.error(f"同步数据失败: {e}")
        raise HTTPException(status_code=500, detail=f"同步数据失败: {str(e)}")


@router.get("/sync/{sync_id}/progress", response_model=StandardResponse, summary="获取同步进度", description="获取指定同步任务的进度信息")
async def get_sync_progress(
    sync_id: str,
    sync_engine: 'DataSyncEngine' = Depends(get_data_sync_engine)
):
    """获取同步进度"""
    try:
        progress = await sync_engine.get_sync_progress(sync_id)
        
        if not progress:
            raise HTTPException(status_code=404, detail=f"同步任务不存在: {sync_id}")
        
        # 处理estimated_remaining_time
        estimated_remaining_seconds = None
        if progress.estimated_remaining_time:
            if hasattr(progress.estimated_remaining_time, 'total_seconds'):
                estimated_remaining_seconds = progress.estimated_remaining_time.total_seconds()
            elif isinstance(progress.estimated_remaining_time, (int, float)):
                estimated_remaining_seconds = progress.estimated_remaining_time
        
        progress_data = {
            "sync_id": progress.sync_id,
            "total_stocks": progress.total_stocks,
            "completed_stocks": progress.completed_stocks,
            "failed_stocks": progress.failed_stocks,
            "current_stock": progress.current_stock,
            "progress_percentage": progress.progress_percentage,
            "estimated_remaining_time_seconds": estimated_remaining_seconds,
            "start_time": progress.start_time.isoformat(),
            "status": progress.status.value if hasattr(progress.status, 'value') else str(progress.status),
            "last_update": progress.last_update.isoformat()
        }
        
        return StandardResponse(
            success=True,
            message="同步进度获取成功",
            data=progress_data
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取同步进度失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取同步进度失败: {str(e)}")


@router.get("/sync/history", response_model=StandardResponse, summary="获取同步历史", description="获取数据同步历史记录")
async def get_sync_history(
    limit: int = 50,
    sync_engine: 'DataSyncEngine' = Depends(get_data_sync_engine)
):
    """获取同步历史"""
    try:
        # get_sync_history是同步方法，不是异步
        history = sync_engine.get_sync_history(limit)
        
        history_data = []
        for record in history:
            history_data.append({
                "sync_id": record.sync_id,
                "request": {
                    "stock_codes": record.request.stock_codes,
                    "start_date": record.request.start_date.isoformat() if record.request.start_date else None,
                    "end_date": record.request.end_date.isoformat() if record.request.end_date else None,
                    "force_update": record.request.force_update,
                    "sync_mode": record.request.sync_mode.value,
                    "max_concurrent": record.request.max_concurrent,
                    "retry_count": record.request.retry_count
                },
                "result": {
                    "success": record.result.success,
                    "total_stocks": record.result.total_stocks,
                    "success_count": record.result.success_count,
                    "failure_count": record.result.failure_count,
                    "total_records": record.result.total_records,
                    "message": record.result.message
                },
                "created_at": record.created_at.isoformat()
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


@router.post("/sync/{sync_id}/retry", response_model=StandardResponse, summary="重试失败的同步", description="重试指定同步任务中失败的股票")
async def retry_sync_failed(
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
        logger.error(f"重试失败的同步失败: {e}")
        raise HTTPException(status_code=500, detail=f"重试失败的同步失败: {str(e)}")


@router.delete("/files", response_model=StandardResponse, summary="删除数据文件", description="删除指定的本地数据文件")
async def delete_data_files(
    file_paths: List[str] = Query(..., description="要删除的文件路径列表"),
    parquet_manager: ParquetManager = Depends(get_parquet_manager)
):
    """删除数据文件"""
    try:
        delete_result = parquet_manager.delete_files(file_paths)
        
        return StandardResponse(
            success=delete_result.success,
            message=delete_result.message,
            data={
                "deleted_files": delete_result.deleted_files,
                "failed_files": [
                    {
                        "file_path": failed.file_path,
                        "error": failed.error
                    }
                    for failed in delete_result.failed_files
                ],
                "total_deleted": delete_result.total_deleted,
                "freed_space_bytes": delete_result.freed_space_bytes,
                "freed_space_mb": round(delete_result.freed_space_bytes / 1024 / 1024, 2)
            }
        )
        
    except Exception as e:
        logger.error(f"删除数据文件失败: {e}")
        raise HTTPException(status_code=500, detail=f"删除数据文件失败: {str(e)}")

