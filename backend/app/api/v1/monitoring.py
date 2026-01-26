"""
监控和告警API路由
添加监控指标查询接口，支持告警配置和历史查询
"""
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from app.api.v1.schemas import StandardResponse
from app.services.monitoring.drift_detector import drift_detector
from app.services.monitoring.performance_monitor import (
    alert_manager,
    performance_monitor,
)

router = APIRouter(prefix="/monitoring", tags=["监控告警"])


# 请求模型
class AlertConfigRequest(BaseModel):
    alert_type: str  # performance, drift, system
    metric_name: str
    threshold: float
    comparison: str  # gt, lt, gte, lte, eq
    enabled: bool = True
    notification_channels: List[str] = ["email", "websocket"]
    description: Optional[str] = None


class AlertUpdateRequest(BaseModel):
    threshold: Optional[float] = None
    comparison: Optional[str] = None
    enabled: Optional[bool] = None
    notification_channels: Optional[List[str]] = None
    description: Optional[str] = None


@router.get("/metrics", response_model=StandardResponse, summary="获取监控指标")
async def get_monitoring_metrics(
    metric_type: Optional[str] = Query(None, description="指标类型过滤"),
    model_id: Optional[str] = Query(None, description="模型ID过滤"),
    time_range: str = Query("1h", description="时间范围: 1h, 6h, 1d, 7d, 30d"),
    limit: int = Query(100, description="返回数量限制"),
):
    """获取监控指标数据"""
    try:
        # 解析时间范围
        time_ranges = {
            "1h": timedelta(hours=1),
            "6h": timedelta(hours=6),
            "1d": timedelta(days=1),
            "7d": timedelta(days=7),
            "30d": timedelta(days=30),
        }

        if time_range not in time_ranges:
            raise HTTPException(status_code=400, detail=f"不支持的时间范围: {time_range}")

        end_time = datetime.now()
        start_time = end_time - time_ranges[time_range]

        # 获取性能指标
        performance_metrics = performance_monitor.get_metrics(
            start_time=start_time, end_time=end_time, model_id=model_id, limit=limit
        )

        # 获取漂移检测指标
        drift_metrics = drift_detector.get_drift_metrics(
            start_time=start_time, end_time=end_time, model_id=model_id, limit=limit
        )

        # 组织返回数据
        metrics_data = {
            "time_range": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
                "duration": time_range,
            },
            "performance_metrics": performance_metrics,
            "drift_metrics": drift_metrics,
            "summary": {
                "total_performance_points": len(performance_metrics),
                "total_drift_points": len(drift_metrics),
            },
        }

        # 过滤指标类型
        if metric_type:
            if metric_type == "performance":
                metrics_data["drift_metrics"] = []
            elif metric_type == "drift":
                metrics_data["performance_metrics"] = []

        return StandardResponse(
            success=True, message=f"成功获取监控指标: {time_range}", data=metrics_data
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取监控指标失败: {str(e)}")


@router.get("/metrics/{model_id}", response_model=StandardResponse, summary="获取模型监控指标")
async def get_model_metrics(
    model_id: str,
    time_range: str = Query("1d", description="时间范围"),
    include_predictions: bool = Query(False, description="是否包含预测数据"),
):
    """获取特定模型的监控指标"""
    try:
        # 解析时间范围
        time_ranges = {
            "1h": timedelta(hours=1),
            "6h": timedelta(hours=6),
            "1d": timedelta(days=1),
            "7d": timedelta(days=7),
            "30d": timedelta(days=30),
        }

        if time_range not in time_ranges:
            raise HTTPException(status_code=400, detail=f"不支持的时间范围: {time_range}")

        end_time = datetime.now()
        start_time = end_time - time_ranges[time_range]

        # 获取模型性能指标
        model_metrics = performance_monitor.get_model_performance(
            model_id=model_id,
            start_time=start_time,
            end_time=end_time,
            include_predictions=include_predictions,
        )

        # 获取模型漂移指标
        model_drift = drift_detector.get_model_drift_status(
            model_id=model_id, start_time=start_time, end_time=end_time
        )

        # 计算指标摘要
        metrics_summary = {
            "model_id": model_id,
            "time_range": time_range,
            "performance_summary": {
                "total_predictions": model_metrics.get("total_predictions", 0),
                "average_accuracy": model_metrics.get("average_accuracy", 0),
                "average_latency": model_metrics.get("average_latency", 0),
                "error_rate": model_metrics.get("error_rate", 0),
            },
            "drift_summary": {
                "drift_detected": model_drift.get("drift_detected", False),
                "drift_score": model_drift.get("drift_score", 0),
                "last_drift_time": model_drift.get("last_drift_time"),
            },
        }

        return StandardResponse(
            success=True,
            message=f"成功获取模型监控指标: {model_id}",
            data={
                "model_metrics": model_metrics,
                "drift_metrics": model_drift,
                "summary": metrics_summary,
            },
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取模型监控指标失败: {str(e)}")


@router.get("/alerts", response_model=StandardResponse, summary="获取告警配置")
async def get_alert_configs(
    alert_type: Optional[str] = Query(None, description="告警类型过滤"),
    enabled: Optional[bool] = Query(None, description="启用状态过滤"),
):
    """获取告警配置列表"""
    try:
        alert_configs = alert_manager.get_alert_configs(
            alert_type=alert_type, enabled=enabled
        )

        return StandardResponse(
            success=True,
            message=f"成功获取告警配置: {len(alert_configs)} 个配置",
            data={
                "alert_configs": alert_configs,
                "total_count": len(alert_configs),
                "filters": {"alert_type": alert_type, "enabled": enabled},
            },
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取告警配置失败: {str(e)}")


@router.post("/alerts", response_model=StandardResponse, summary="创建告警配置")
async def create_alert_config(request: AlertConfigRequest):
    """创建新的告警配置"""
    try:
        alert_config = {
            "alert_type": request.alert_type,
            "metric_name": request.metric_name,
            "threshold": request.threshold,
            "comparison": request.comparison,
            "enabled": request.enabled,
            "notification_channels": request.notification_channels,
            "description": request.description,
            "created_at": datetime.now().isoformat(),
        }

        alert_id = alert_manager.create_alert_config(alert_config)

        return StandardResponse(
            success=True,
            message=f"成功创建告警配置: {alert_id}",
            data={"alert_id": alert_id, "config": alert_config},
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"创建告警配置失败: {str(e)}")


@router.get("/alerts/{alert_id}", response_model=StandardResponse, summary="获取告警配置详情")
async def get_alert_config(alert_id: str):
    """获取告警配置详情"""
    try:
        alert_config = alert_manager.get_alert_config(alert_id)

        if not alert_config:
            raise HTTPException(status_code=404, detail=f"告警配置不存在: {alert_id}")

        return StandardResponse(success=True, message="成功获取告警配置详情", data=alert_config)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取告警配置详情失败: {str(e)}")


@router.put("/alerts/{alert_id}", response_model=StandardResponse, summary="更新告警配置")
async def update_alert_config(alert_id: str, request: AlertUpdateRequest):
    """更新告警配置"""
    try:
        # 获取现有配置
        existing_config = alert_manager.get_alert_config(alert_id)
        if not existing_config:
            raise HTTPException(status_code=404, detail=f"告警配置不存在: {alert_id}")

        # 更新配置
        update_data = {}
        if request.threshold is not None:
            update_data["threshold"] = request.threshold
        if request.comparison is not None:
            update_data["comparison"] = request.comparison
        if request.enabled is not None:
            update_data["enabled"] = request.enabled
        if request.notification_channels is not None:
            update_data["notification_channels"] = request.notification_channels
        if request.description is not None:
            update_data["description"] = request.description

        update_data["updated_at"] = datetime.now().isoformat()

        success = alert_manager.update_alert_config(alert_id, update_data)

        if not success:
            raise HTTPException(status_code=500, detail="更新告警配置失败")

        # 获取更新后的配置
        updated_config = alert_manager.get_alert_config(alert_id)

        return StandardResponse(
            success=True, message=f"成功更新告警配置: {alert_id}", data=updated_config
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"更新告警配置失败: {str(e)}")


@router.delete("/alerts/{alert_id}", response_model=StandardResponse, summary="删除告警配置")
async def delete_alert_config(alert_id: str):
    """删除告警配置"""
    try:
        success = alert_manager.delete_alert_config(alert_id)

        if not success:
            raise HTTPException(status_code=404, detail=f"告警配置不存在: {alert_id}")

        return StandardResponse(
            success=True,
            message=f"成功删除告警配置: {alert_id}",
            data={"alert_id": alert_id, "deleted_at": datetime.now().isoformat()},
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除告警配置失败: {str(e)}")


@router.get("/alerts/history", response_model=StandardResponse, summary="获取告警历史")
async def get_alert_history(
    alert_type: Optional[str] = Query(None, description="告警类型过滤"),
    severity: Optional[str] = Query(None, description="严重程度过滤"),
    time_range: str = Query("7d", description="时间范围"),
    limit: int = Query(100, description="返回数量限制"),
):
    """获取告警历史记录"""
    try:
        # 解析时间范围
        time_ranges = {
            "1h": timedelta(hours=1),
            "6h": timedelta(hours=6),
            "1d": timedelta(days=1),
            "7d": timedelta(days=7),
            "30d": timedelta(days=30),
        }

        if time_range not in time_ranges:
            raise HTTPException(status_code=400, detail=f"不支持的时间范围: {time_range}")

        end_time = datetime.now()
        start_time = end_time - time_ranges[time_range]

        # 获取告警历史
        alert_history = alert_manager.get_alert_history(
            start_time=start_time,
            end_time=end_time,
            alert_type=alert_type,
            severity=severity,
            limit=limit,
        )

        # 统计信息
        stats = {
            "total_alerts": len(alert_history),
            "severity_distribution": {},
            "type_distribution": {},
            "resolved_count": 0,
            "active_count": 0,
        }

        for alert in alert_history:
            # 按严重程度统计
            alert_severity = alert.get("severity", "unknown")
            stats["severity_distribution"][alert_severity] = (
                stats["severity_distribution"].get(alert_severity, 0) + 1
            )

            # 按类型统计
            alert_type_val = alert.get("alert_type", "unknown")
            stats["type_distribution"][alert_type_val] = (
                stats["type_distribution"].get(alert_type_val, 0) + 1
            )

            # 按状态统计
            if alert.get("resolved", False):
                stats["resolved_count"] += 1
            else:
                stats["active_count"] += 1

        return StandardResponse(
            success=True,
            message=f"成功获取告警历史: {len(alert_history)} 条记录",
            data={
                "alert_history": alert_history,
                "statistics": stats,
                "time_range": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat(),
                    "duration": time_range,
                },
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取告警历史失败: {str(e)}")


@router.post(
    "/alerts/{alert_id}/resolve", response_model=StandardResponse, summary="解决告警"
)
async def resolve_alert(alert_id: str, resolution_note: Optional[str] = None):
    """标记告警为已解决"""
    try:
        success = alert_manager.resolve_alert(alert_id, resolution_note)

        if not success:
            raise HTTPException(status_code=404, detail=f"告警不存在: {alert_id}")

        return StandardResponse(
            success=True,
            message=f"成功解决告警: {alert_id}",
            data={
                "alert_id": alert_id,
                "resolved_at": datetime.now().isoformat(),
                "resolution_note": resolution_note,
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"解决告警失败: {str(e)}")


@router.get("/dashboard", response_model=StandardResponse, summary="获取监控仪表板数据")
async def get_monitoring_dashboard():
    """获取监控仪表板数据"""
    try:
        # 获取系统整体状态
        system_status = performance_monitor.get_system_status()

        # 获取活跃告警
        active_alerts = alert_manager.get_active_alerts()

        # 获取最近的性能指标
        recent_metrics = performance_monitor.get_recent_metrics(limit=50)

        # 获取漂移检测状态
        drift_status = drift_detector.get_overall_drift_status()

        dashboard_data = {
            "system_status": system_status,
            "active_alerts": {
                "count": len(active_alerts),
                "alerts": active_alerts[:10],  # 只返回前10个
                "severity_counts": {},
            },
            "performance_overview": {
                "total_models": system_status.get("total_models", 0),
                "active_models": system_status.get("active_models", 0),
                "average_latency": system_status.get("average_latency", 0),
                "error_rate": system_status.get("error_rate", 0),
            },
            "drift_overview": drift_status,
            "recent_metrics": recent_metrics[-20:]
            if recent_metrics
            else [],  # 最近20个指标点
        }

        # 统计告警严重程度
        for alert in active_alerts:
            severity = alert.get("severity", "unknown")
            dashboard_data["active_alerts"]["severity_counts"][severity] = (
                dashboard_data["active_alerts"]["severity_counts"].get(severity, 0) + 1
            )

        return StandardResponse(
            success=True, message="成功获取监控仪表板数据", data=dashboard_data
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取监控仪表板数据失败: {str(e)}")


@router.post("/test-alert", response_model=StandardResponse, summary="测试告警")
async def test_alert(
    alert_type: str = Query(..., description="告警类型"),
    metric_name: str = Query(..., description="指标名称"),
    test_value: float = Query(..., description="测试值"),
):
    """测试告警配置"""
    try:
        # 触发测试告警
        test_result = alert_manager.test_alert(
            alert_type=alert_type, metric_name=metric_name, test_value=test_value
        )

        return StandardResponse(success=True, message="告警测试完成", data=test_result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"告警测试失败: {str(e)}")
