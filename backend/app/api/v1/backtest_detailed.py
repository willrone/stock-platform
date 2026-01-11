"""
回测详细结果API端点
提供回测详细数据和图表缓存的API接口
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Body
from typing import Optional, Dict, Any, List
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger
from pydantic import BaseModel, Field

from app.core.database import get_async_session
from app.repositories.backtest_detailed_repository import BacktestDetailedRepository
from app.services.backtest.chart_cache_service import ChartCacheService
from app.api.v1.schemas import StandardResponse


class CacheChartRequest(BaseModel):
    """缓存图表数据请求"""
    chart_type: str = Field(..., description="图表类型")
    chart_data: Dict[str, Any] = Field(..., description="图表数据")
    expiry_hours: Optional[int] = Field(24, description="过期时间（小时）")


router = APIRouter(prefix="/backtest-detailed", tags=["回测详细结果"])


@router.get("/{task_id}/detailed-result", response_model=StandardResponse)
async def get_detailed_backtest_result(
    task_id: str,
    session: AsyncSession = Depends(get_async_session)
):
    """获取回测详细结果"""
    logger.info(f"[API] 收到获取回测详细结果请求: task_id={task_id}")
    
    try:
        repository = BacktestDetailedRepository(session)
        logger.info(f"[API] 开始查询数据库中的详细结果...")
        
        detailed_result = await repository.get_detailed_result_by_task_id(task_id)
        
        if not detailed_result:
            logger.warning(f"[API] 未找到任务 {task_id} 的回测详细结果")
            raise HTTPException(status_code=404, detail="未找到回测详细结果")
        
        logger.info(f"[API] 成功获取回测详细结果: task_id={task_id}")
        result_dict = detailed_result.to_dict()
        logger.debug(f"[API] 详细结果数据字段: {list(result_dict.keys())}")
        
        return StandardResponse(
            success=True,
            message="获取回测详细结果成功",
            data=result_dict
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[API] 获取回测详细结果失败: task_id={task_id}, error={e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取回测详细结果失败: {str(e)}")


@router.get("/{task_id}/portfolio-snapshots", response_model=StandardResponse)
async def get_portfolio_snapshots(
    task_id: str,
    start_date: Optional[str] = Query(None, description="开始日期 (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="结束日期 (YYYY-MM-DD)"),
    limit: Optional[int] = Query(100, description="返回记录数限制"),
    session: AsyncSession = Depends(get_async_session)
):
    """获取组合快照数据"""
    logger.info(f"[API] 收到获取组合快照请求: task_id={task_id}, start_date={start_date}, end_date={end_date}, limit={limit}")
    
    try:
        repository = BacktestDetailedRepository(session)
        
        # 解析日期参数
        start_dt = datetime.fromisoformat(start_date) if start_date else None
        end_dt = datetime.fromisoformat(end_date) if end_date else None
        
        logger.info(f"[API] 开始查询组合快照数据...")
        snapshots = await repository.get_portfolio_snapshots(
            task_id=task_id,
            start_date=start_dt,
            end_date=end_dt,
            limit=limit
        )
        
        snapshots_data = [snapshot.to_dict() for snapshot in snapshots]
        logger.info(f"[API] 成功获取组合快照: task_id={task_id}, count={len(snapshots_data)}")
        
        return StandardResponse(
            success=True,
            message=f"获取组合快照成功，共{len(snapshots_data)}条记录",
            data={
                "snapshots": snapshots_data,
                "total_count": len(snapshots_data)
            }
        )
        
    except ValueError as e:
        logger.error(f"[API] 日期格式错误: {e}")
        raise HTTPException(status_code=400, detail=f"日期格式错误: {str(e)}")
    except Exception as e:
        logger.error(f"[API] 获取组合快照失败: task_id={task_id}, error={e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取组合快照失败: {str(e)}")


@router.get("/{task_id}/trade-records", response_model=StandardResponse)
async def get_trade_records(
    task_id: str,
    stock_code: Optional[str] = Query(None, description="股票代码筛选"),
    action: Optional[str] = Query(None, description="交易动作筛选 (BUY/SELL)"),
    start_date: Optional[str] = Query(None, description="开始日期 (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="结束日期 (YYYY-MM-DD)"),
    offset: int = Query(0, description="偏移量"),
    limit: int = Query(50, description="返回记录数限制"),
    order_by: str = Query("timestamp", description="排序字段"),
    order_desc: bool = Query(True, description="是否降序排列"),
    session: AsyncSession = Depends(get_async_session)
):
    """获取交易记录"""
    try:
        repository = BacktestDetailedRepository(session)
        
        # 解析日期参数
        start_dt = datetime.fromisoformat(start_date) if start_date else None
        end_dt = datetime.fromisoformat(end_date) if end_date else None
        
        trades = await repository.get_trade_records(
            task_id=task_id,
            stock_code=stock_code,
            action=action,
            start_date=start_dt,
            end_date=end_dt,
            offset=offset,
            limit=limit,
            order_by=order_by,
            order_desc=order_desc
        )
        
        # 获取总记录数
        total_count = await repository.get_trade_records_count(
            task_id=task_id,
            stock_code=stock_code,
            action=action,
            start_date=start_dt,
            end_date=end_dt
        )
        
        trades_data = [trade.to_dict() for trade in trades]
        
        return StandardResponse(
            success=True,
            message=f"获取交易记录成功，共{total_count}条记录",
            data={
                "trades": trades_data,
                "pagination": {
                    "offset": offset,
                    "limit": limit,
                    "count": total_count  # 修正：返回总记录数而不是当前页记录数
                }
            }
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"日期格式错误: {str(e)}")
    except Exception as e:
        logger.error(f"获取交易记录失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取交易记录失败: {str(e)}")


@router.get("/{task_id}/trade-statistics", response_model=StandardResponse)
async def get_trade_statistics(
    task_id: str,
    session: AsyncSession = Depends(get_async_session)
):
    """获取交易统计信息"""
    try:
        repository = BacktestDetailedRepository(session)
        stats = await repository.get_trade_statistics(task_id)
        
        return StandardResponse(
            success=True,
            message="获取交易统计成功",
            data=stats
        )
        
    except Exception as e:
        logger.error(f"获取交易统计失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取交易统计失败: {str(e)}")


@router.get("/{task_id}/benchmark-data", response_model=StandardResponse)
async def get_benchmark_data(
    task_id: str,
    benchmark_symbol: str = Query("000300.SH", description="基准代码"),
    session: AsyncSession = Depends(get_async_session)
):
    """获取基准对比数据"""
    try:
        repository = BacktestDetailedRepository(session)
        benchmark = await repository.get_benchmark_data(task_id, benchmark_symbol)
        
        if not benchmark:
            raise HTTPException(status_code=404, detail="未找到基准数据")
        
        return StandardResponse(
            success=True,
            message="获取基准数据成功",
            data=benchmark.to_dict()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取基准数据失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取基准数据失败: {str(e)}")


@router.post("/{task_id}/cache-chart", response_model=StandardResponse)
async def cache_chart_data(
    task_id: str,
    request: CacheChartRequest
):
    """缓存图表数据"""
    try:
        cache_service = ChartCacheService()
        success = await cache_service.cache_chart_data(
            task_id=task_id,
            chart_type=request.chart_type,
            chart_data=request.chart_data,
            expiry_hours=request.expiry_hours
        )
        
        if success:
            return StandardResponse(
                success=True,
                message="图表数据缓存成功",
                data={"task_id": task_id, "chart_type": request.chart_type}
            )
        else:
            raise HTTPException(status_code=500, detail="图表数据缓存失败")
        
    except Exception as e:
        logger.error(f"缓存图表数据失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"缓存图表数据失败: {str(e)}")


@router.get("/{task_id}/cached-chart/{chart_type}", response_model=StandardResponse)
async def get_cached_chart_data(
    task_id: str,
    chart_type: str
):
    """获取缓存的图表数据"""
    try:
        cache_service = ChartCacheService()
        cached_data = await cache_service.get_cached_chart_data(task_id, chart_type)
        
        if cached_data:
            return StandardResponse(
                success=True,
                message="获取缓存图表数据成功",
                data=cached_data
            )
        else:
            raise HTTPException(status_code=404, detail="未找到缓存的图表数据")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取缓存图表数据失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取缓存图表数据失败: {str(e)}")


@router.delete("/{task_id}/cache", response_model=StandardResponse)
async def invalidate_cache(
    task_id: str,
    chart_type: Optional[str] = Query(None, description="特定图表类型，不指定则清理所有")
):
    """使缓存失效"""
    try:
        cache_service = ChartCacheService()
        success = await cache_service.invalidate_cache(task_id, chart_type)
        
        if success:
            message = f"缓存清理成功"
            if chart_type:
                message += f" (图表类型: {chart_type})"
            
            return StandardResponse(
                success=True,
                message=message,
                data={"task_id": task_id, "chart_type": chart_type}
            )
        else:
            raise HTTPException(status_code=500, detail="缓存清理失败")
        
    except Exception as e:
        logger.error(f"缓存清理失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"缓存清理失败: {str(e)}")


@router.get("/cache/statistics", response_model=StandardResponse)
async def get_cache_statistics():
    """获取缓存统计信息"""
    try:
        cache_service = ChartCacheService()
        stats = await cache_service.get_cache_statistics()
        
        return StandardResponse(
            success=True,
            message="获取缓存统计成功",
            data=stats
        )
        
    except Exception as e:
        logger.error(f"获取缓存统计失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取缓存统计失败: {str(e)}")


@router.delete("/cache/cleanup", response_model=StandardResponse)
async def cleanup_expired_cache():
    """清理过期缓存"""
    try:
        cache_service = ChartCacheService()
        deleted_count = await cache_service.cleanup_expired_cache()
        
        return StandardResponse(
            success=True,
            message=f"过期缓存清理完成，删除了{deleted_count}条记录",
            data={"deleted_count": deleted_count}
        )
        
    except Exception as e:
        logger.error(f"清理过期缓存失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"清理过期缓存失败: {str(e)}")


@router.delete("/{task_id}/data", response_model=StandardResponse)
async def delete_task_data(
    task_id: str,
    session: AsyncSession = Depends(get_async_session)
):
    """删除任务的所有详细数据"""
    try:
        repository = BacktestDetailedRepository(session)
        success = await repository.delete_task_data(task_id)
        await session.commit()
        
        if success:
            return StandardResponse(
                success=True,
                message="任务数据删除成功",
                data={"task_id": task_id}
            )
        else:
            raise HTTPException(status_code=500, detail="任务数据删除失败")
        
    except HTTPException:
        raise
    except Exception as e:
        await session.rollback()
        logger.error(f"删除任务数据失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"删除任务数据失败: {str(e)}")