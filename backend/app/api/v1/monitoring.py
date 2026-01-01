"""
监控服务路由
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Optional
from datetime import datetime
import logging

from app.api.v1.schemas import StandardResponse
from app.core.container import get_monitoring_service
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.services.infrastructure import DataMonitoringService

router = APIRouter(prefix="/monitoring", tags=["监控服务"])
logger = logging.getLogger(__name__)


@router.get("/health", response_model=StandardResponse, summary="系统健康检查", description="获取所有服务的健康状态")
async def get_system_health(monitoring_service: 'DataMonitoringService' = Depends(get_monitoring_service)):
    """获取系统健康状态"""
    try:
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


@router.get("/metrics", response_model=StandardResponse, summary="性能指标", description="获取系统性能指标")
async def get_performance_metrics(
    service_name: Optional[str] = None,
    monitoring_service: 'DataMonitoringService' = Depends(get_monitoring_service)
):
    """获取性能指标"""
    try:
        if service_name:
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


@router.get("/overview", response_model=StandardResponse, summary="系统概览", description="获取系统整体概览信息")
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


@router.get("/errors", response_model=StandardResponse, summary="错误统计", description="获取系统错误统计信息")
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


@router.get("/quality", response_model=StandardResponse, summary="数据质量检查", description="获取数据质量检查结果")
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


@router.get("/anomalies", response_model=StandardResponse, summary="异常检测", description="获取系统异常检测结果")
async def get_anomalies(monitoring_service: 'DataMonitoringService' = Depends(get_monitoring_service)):
    """获取异常检测结果"""
    try:
        anomalies = monitoring_service.detect_anomalies()
        
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

