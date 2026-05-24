"""
监控和告警API路由
添加监控指标查询接口，支持告警配置和历史查询
"""

# mypy: disable-error-code="untyped-decorator"

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel

from app.api.v1.schemas import StandardResponse
from app.core.config import settings
from app.core.container import ServiceContainer, get_container
from app.services.data.parquet_manager import ParquetManager
from app.services.infrastructure.monitoring_service import DataMonitoringService
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


_data_monitoring_service: Optional[DataMonitoringService] = None


def _resolve_data_root() -> Path:
    """解析 DATA_ROOT_PATH，兼容相对 backend 目录和绝对路径。"""
    data_root = Path(settings.DATA_ROOT_PATH)
    if data_root.is_absolute():
        return data_root
    backend_dir = Path(__file__).resolve().parents[3]
    return (backend_dir / data_root).resolve()


def _candidate_parquet_roots() -> List[Path]:
    data_root = _resolve_data_root()
    backend_dir = Path(__file__).resolve().parents[3]
    return [
        data_root / "parquet" / "stock_data",
        data_root / "parquet" / "daily",
        data_root / "parquet",
        data_root / "stocks" / "daily",
        backend_dir / "data" / "parquet" / "stock_data",
        backend_dir / "data" / "parquet",
        Path("data") / "parquet" / "stock_data",
        Path("data") / "parquet",
    ]


def _find_fast_parquet_root() -> Optional[Path]:
    for candidate in _candidate_parquet_roots():
        try:
            resolved = candidate.resolve()
        except Exception:
            resolved = candidate
        if resolved.exists() and resolved.is_dir():
            return resolved
    return None


def _stock_code_from_file(file_path: Path) -> str:
    stem = file_path.stem
    if "_" in stem:
        parts = stem.split("_")
        if len(parts) >= 2 and parts[0].isdigit():
            market = parts[1].upper()
            if market in {"SZ", "SH", "BJ"}:
                return f"{parts[0]}.{market}"
    if "." in stem:
        return stem

    for parent in file_path.parents:
        name = parent.name
        if "." in name or ("_" in name and name.split("_")[0].isdigit()):
            return name.replace("_", ".")
    return stem


def _get_fast_storage_overview() -> Dict[str, Any]:
    """快速统计 parquet 文件，不读取文件内容。"""
    root = _find_fast_parquet_root()
    if root is None:
        return {
            "root": None,
            "total_files": 0,
            "total_size": 0,
            "stock_count": 0,
            "last_modified": None,
        }

    total_size = 0
    last_modified_ts: Optional[float] = None
    stock_codes: set[str] = set()
    total_files = 0

    for file_path in root.rglob("*.parquet"):
        try:
            stat = file_path.stat()
        except OSError:
            continue
        total_files += 1
        total_size += stat.st_size
        stock_codes.add(_stock_code_from_file(file_path))
        if last_modified_ts is None or stat.st_mtime > last_modified_ts:
            last_modified_ts = stat.st_mtime

    return {
        "root": str(root),
        "total_files": total_files,
        "total_size": total_size,
        "stock_count": len(stock_codes),
        "last_modified": (
            datetime.fromtimestamp(last_modified_ts).isoformat()
            if last_modified_ts is not None
            else None
        ),
    }


def _build_parquet_manager() -> ParquetManager:
    """构建用于监控接口的本地 parquet 管理器。"""
    parquet_root = _find_fast_parquet_root()
    return ParquetManager(str(parquet_root or _resolve_data_root()))


async def get_data_monitoring_service(
    container: ServiceContainer = Depends(get_container),
) -> DataMonitoringService:
    """按需创建数据监控服务，供前端兼容接口复用。"""
    global _data_monitoring_service
    if _data_monitoring_service is None:
        _data_monitoring_service = DataMonitoringService(
            data_service=container.data_service,
            indicators_service=container.indicators_service,
            parquet_manager=_build_parquet_manager(),
        )
    return _data_monitoring_service


def _service_health_to_dict(status: Any) -> Dict[str, Any]:
    return {
        "healthy": bool(status.is_healthy),
        "response_time_ms": round(float(status.response_time_ms), 2),
        "last_check": status.last_check.isoformat(),
        "error_message": status.error_message,
    }


def _empty_performance_payload() -> Dict[str, Any]:
    return {
        "services": {},
        "summary": {
            "total_services": 0,
            "avg_response_time": 0,
            "total_requests": 0,
            "total_errors": 0,
        },
    }


def _build_fast_quality_report(storage: Dict[str, Any]) -> Dict[str, Any]:
    """基于文件系统元信息构建快速数据质量报告。"""
    total_files = int(storage.get("total_files", 0) or 0)
    stock_count = int(storage.get("stock_count", 0) or 0)
    total_size = int(storage.get("total_size", 0) or 0)
    last_modified = storage.get("last_modified")

    checks: Dict[str, Any] = {}
    issues: List[str] = []
    recommendations: List[str] = []

    checks["data_completeness"] = {
        "score": 1.0 if total_files > 0 else 0.0,
        "status": "pass" if total_files > 0 else "fail",
        "message": (
            f"发现 {total_files} 个数据文件" if total_files > 0 else "未发现数据文件"
        ),
    }
    if total_files == 0:
        issues.append("系统中没有数据文件")
        recommendations.append("执行数据同步以获取股票数据")

    days_old: Optional[int] = None
    freshness_score = 0.0
    freshness_status = "unknown"
    if isinstance(last_modified, str):
        modified_at = datetime.fromisoformat(last_modified)
        days_old = (datetime.now() - modified_at).days
        if days_old <= 1:
            freshness_score = 1.0
            freshness_status = "excellent"
        elif days_old <= 7:
            freshness_score = 0.8
            freshness_status = "good"
        elif days_old <= 30:
            freshness_score = 0.6
            freshness_status = "fair"
        else:
            freshness_score = 0.3
            freshness_status = "poor"
            issues.append(f"数据文件已 {days_old} 天未更新")
            recommendations.append("同步或刷新本地数据文件")

    checks["data_freshness"] = {
        "score": freshness_score,
        "status": freshness_status,
        "message": (
            f"数据文件最后更新于 {days_old} 天前"
            if days_old is not None
            else "无法确定数据新鲜度"
        ),
        "days_old": days_old,
    }

    checks["storage_coverage"] = {
        "score": 1.0 if total_size > 0 else 0.0,
        "status": "pass" if total_size > 0 else "unknown",
        "message": f"本地数据总大小 {round(total_size / 1024 / 1024, 2)} MB",
        "total_size": total_size,
    }

    if stock_count >= 100:
        coverage_score = 1.0
        coverage_status = "excellent"
    elif stock_count >= 50:
        coverage_score = 0.8
        coverage_status = "good"
    elif stock_count >= 10:
        coverage_score = 0.6
        coverage_status = "fair"
    elif stock_count > 0:
        coverage_score = 0.4
        coverage_status = "poor"
    else:
        coverage_score = 0.0
        coverage_status = "fail"

    checks["stock_coverage"] = {
        "score": coverage_score,
        "status": coverage_status,
        "message": f"覆盖 {stock_count} 只股票",
        "stock_count": stock_count,
    }
    if stock_count < 50:
        issues.append("股票覆盖度较低")
        recommendations.append("增加更多股票的数据同步")

    scores = [float(check["score"]) for check in checks.values()]
    return {
        "overall_score": round(sum(scores) / len(scores), 2) if scores else 0.0,
        "checks": checks,
        "issues": issues,
        "recommendations": recommendations,
        "storage": storage,
    }


@router.get("/health", response_model=StandardResponse, summary="获取系统健康状态")
async def get_system_health(
    container: ServiceContainer = Depends(get_container),
) -> Any:
    """获取前端监控页需要的系统健康状态。"""
    try:
        services: Dict[str, Any] = {}

        data_start = datetime.now()
        try:
            data_status = await asyncio.wait_for(
                container.data_service.check_remote_service_status(), timeout=2.0
            )
            data_response_ms = (datetime.now() - data_start).total_seconds() * 1000
            services["remote_data_service"] = {
                "healthy": bool(data_status.is_available),
                "response_time_ms": round(min(data_response_ms, 2000.0), 2),
                "last_check": data_status.last_check.isoformat(),
                "error_message": data_status.error_message,
                "service_url": data_status.service_url,
                "core": False,
                "optional": True,
                "status_label": "正常" if data_status.is_available else "未连接",
            }
        except asyncio.TimeoutError:
            services["remote_data_service"] = {
                "healthy": False,
                "response_time_ms": 2000.0,
                "last_check": data_start.isoformat(),
                "error_message": "远端数据服务健康检查超时",
                "core": False,
                "optional": True,
                "status_label": "未连接",
            }
        except Exception as e:
            data_response_ms = (datetime.now() - data_start).total_seconds() * 1000
            services["remote_data_service"] = {
                "healthy": False,
                "response_time_ms": round(min(data_response_ms, 2000.0), 2),
                "last_check": data_start.isoformat(),
                "error_message": f"远端数据服务健康检查失败: {str(e)}",
                "core": False,
                "optional": True,
                "status_label": "未连接",
            }

        indicators_start = datetime.now()
        services["indicators_service"] = {
            "healthy": container.indicators_service is not None,
            "response_time_ms": round(
                (datetime.now() - indicators_start).total_seconds() * 1000, 2
            ),
            "last_check": indicators_start.isoformat(),
            "error_message": None,
            "core": True,
            "optional": False,
            "status_label": "正常",
        }

        storage_start = datetime.now()
        storage = _get_fast_storage_overview()
        storage_has_files = int(storage.get("total_files", 0)) > 0
        services["parquet_manager"] = {
            "healthy": storage_has_files,
            "response_time_ms": round(
                (datetime.now() - storage_start).total_seconds() * 1000, 2
            ),
            "last_check": storage_start.isoformat(),
            "error_message": (
                None if storage_has_files else "未发现本地 parquet 数据文件"
            ),
            "core": True,
            "optional": False,
            "status_label": "正常" if storage_has_files else "异常",
        }

        core_services = [
            service for service in services.values() if bool(service.get("core", True))
        ]
        overall_healthy = all(
            bool(service.get("healthy", False)) for service in core_services
        )
        payload = {
            "overall_healthy": overall_healthy,
            "services": services,
            "check_time": datetime.now().isoformat(),
        }

        return StandardResponse(
            success=True, message="成功获取系统健康状态", data=payload
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取系统健康状态失败: {str(e)}")


@router.get("/metrics", response_model=StandardResponse, summary="获取监控指标")
async def get_monitoring_metrics(
    request: Request,
    metric_type: Optional[str] = Query(None, description="指标类型过滤"),
    model_id: Optional[str] = Query(None, description="模型ID过滤"),
    time_range: str = Query("1h", description="时间范围: 1h, 6h, 1d, 7d, 30d"),
    limit: int = Query(100, description="返回数量限制"),
    service_name: Optional[str] = Query(None, description="服务名称过滤"),
    monitor: DataMonitoringService = Depends(get_data_monitoring_service),
) -> Any:
    """获取监控指标数据"""
    try:
        if service_name is not None or len(request.query_params) == 0:
            service_names = (
                [service_name]
                if service_name
                else ["data_service", "indicators_service", "parquet_manager"]
            )
            services: Dict[str, Any] = {}
            for name in service_names:
                metrics = monitor.get_performance_metrics(name)
                if metrics:
                    services[name] = metrics.to_dict()

            if services:
                total_requests = sum(
                    int(service.get("request_count", 0))
                    for service in services.values()
                )
                total_errors = sum(
                    int(service.get("error_count", 0)) for service in services.values()
                )
                avg_response_time = sum(
                    float(service.get("avg_response_time", 0))
                    for service in services.values()
                ) / len(services)
                metrics_payload = {
                    "services": services,
                    "summary": {
                        "total_services": len(services),
                        "avg_response_time": round(avg_response_time, 2),
                        "total_requests": total_requests,
                        "total_errors": total_errors,
                    },
                }
            else:
                metrics_payload = _empty_performance_payload()

            return StandardResponse(
                success=True, message="成功获取性能指标", data=metrics_payload
            )

        # 解析时间范围
        time_ranges = {
            "1h": timedelta(hours=1),
            "6h": timedelta(hours=6),
            "1d": timedelta(days=1),
            "7d": timedelta(days=7),
            "30d": timedelta(days=30),
        }

        if time_range not in time_ranges:
            raise HTTPException(
                status_code=400, detail=f"不支持的时间范围: {time_range}"
            )

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


@router.get("/overview", response_model=StandardResponse, summary="获取系统概览")
async def get_system_overview() -> Any:
    """获取数据监控系统概览。"""
    try:
        storage = _get_fast_storage_overview()
        overview = {
            "timestamp": datetime.now().isoformat(),
            "services": {},
            "overall_health": int(storage.get("total_files", 0)) > 0,
            "total_requests": 0,
            "total_errors": 0,
            "storage_stats": storage,
            "data_quality": _build_fast_quality_report(storage),
        }
        return StandardResponse(success=True, message="成功获取系统概览", data=overview)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取系统概览失败: {str(e)}")


@router.get("/errors", response_model=StandardResponse, summary="获取错误统计")
async def get_error_statistics(
    hours: int = Query(24, ge=1, le=168, description="统计时间范围（小时）"),
    monitor: DataMonitoringService = Depends(get_data_monitoring_service),
) -> Any:
    """获取前端监控页需要的错误统计。"""
    try:
        statistics = monitor.get_error_statistics(hours)
        error_statistics: List[Dict[str, Any]] = []
        total_errors = 0
        for stat in statistics:
            total_errors += stat.count
            error_statistics.append(
                {
                    "error_type": stat.error_type,
                    "count": stat.count,
                    "last_occurrence": stat.last_occurrence.isoformat(),
                    "sample_message": stat.sample_message,
                }
            )
        payload = {
            "time_range_hours": hours,
            "total_error_types": len(error_statistics),
            "total_errors": total_errors,
            "error_statistics": error_statistics,
        }
        return StandardResponse(success=True, message="成功获取错误统计", data=payload)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取错误统计失败: {str(e)}")


@router.get("/quality", response_model=StandardResponse, summary="获取数据质量检查")
async def get_data_quality() -> Any:
    """获取前端监控页需要的数据质量检查结果。"""
    try:
        quality = _build_fast_quality_report(_get_fast_storage_overview())
        return StandardResponse(
            success=True, message="成功获取数据质量检查", data=quality
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取数据质量检查失败: {str(e)}")


@router.get("/anomalies", response_model=StandardResponse, summary="获取异常检测结果")
async def get_anomalies(
    monitor: DataMonitoringService = Depends(get_data_monitoring_service),
) -> Any:
    """获取前端监控页需要的异常检测结果。"""
    try:
        raw_anomalies = monitor.detect_anomalies()
        anomalies = [
            {
                "type": anomaly.get("type", "unknown"),
                "severity": anomaly.get("severity", "low"),
                "description": anomaly.get("description")
                or anomaly.get("message")
                or "检测到异常",
                "detected_at": anomaly.get("detected_at", datetime.now().isoformat()),
                "affected_component": anomaly.get("affected_component")
                or anomaly.get("service")
                or "unknown",
            }
            for anomaly in raw_anomalies
        ]
        by_severity = {"high": 0, "medium": 0, "low": 0}
        for anomaly in anomalies:
            severity = str(anomaly.get("severity", "low"))
            if severity not in by_severity:
                severity = "low"
            by_severity[severity] += 1

        payload = {
            "total_anomalies": len(anomalies),
            "by_severity": by_severity,
            "anomalies": anomalies,
            "detection_time": datetime.now().isoformat(),
        }
        return StandardResponse(
            success=True, message="成功获取异常检测结果", data=payload
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取异常检测结果失败: {str(e)}")


@router.get(
    "/metrics/{model_id}", response_model=StandardResponse, summary="获取模型监控指标"
)
async def get_model_metrics(
    model_id: str,
    time_range: str = Query("1d", description="时间范围"),
    include_predictions: bool = Query(False, description="是否包含预测数据"),
) -> Any:
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
            raise HTTPException(
                status_code=400, detail=f"不支持的时间范围: {time_range}"
            )

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
) -> Any:
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
async def create_alert_config(request: AlertConfigRequest) -> Any:
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


@router.get(
    "/alerts/{alert_id}", response_model=StandardResponse, summary="获取告警配置详情"
)
async def get_alert_config(alert_id: str) -> Any:
    """获取告警配置详情"""
    try:
        alert_config = alert_manager.get_alert_config(alert_id)

        if not alert_config:
            raise HTTPException(status_code=404, detail=f"告警配置不存在: {alert_id}")

        return StandardResponse(
            success=True, message="成功获取告警配置详情", data=alert_config
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取告警配置详情失败: {str(e)}")


@router.put(
    "/alerts/{alert_id}", response_model=StandardResponse, summary="更新告警配置"
)
async def update_alert_config(alert_id: str, request: AlertUpdateRequest) -> Any:
    """更新告警配置"""
    try:
        # 获取现有配置
        existing_config = alert_manager.get_alert_config(alert_id)
        if not existing_config:
            raise HTTPException(status_code=404, detail=f"告警配置不存在: {alert_id}")

        # 更新配置
        update_data: Dict[str, Any] = {}
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


@router.delete(
    "/alerts/{alert_id}", response_model=StandardResponse, summary="删除告警配置"
)
async def delete_alert_config(alert_id: str) -> Any:
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
) -> Any:
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
            raise HTTPException(
                status_code=400, detail=f"不支持的时间范围: {time_range}"
            )

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
        stats: Dict[str, Any] = {
            "total_alerts": len(alert_history),
            "severity_distribution": {},
            "type_distribution": {},
            "resolved_count": 0,
            "active_count": 0,
        }

        for alert in alert_history:
            # 按严重程度统计
            alert_severity = str(alert.get("severity", "unknown"))
            severity_distribution = cast(Dict[str, int], stats["severity_distribution"])
            severity_distribution[alert_severity] = (
                severity_distribution.get(alert_severity, 0) + 1
            )

            # 按类型统计
            alert_type_val = str(alert.get("alert_type", "unknown"))
            type_distribution = cast(Dict[str, int], stats["type_distribution"])
            type_distribution[alert_type_val] = (
                type_distribution.get(alert_type_val, 0) + 1
            )

            # 按状态统计
            if bool(alert.get("resolved", False)):
                stats["resolved_count"] = int(stats["resolved_count"]) + 1
            else:
                stats["active_count"] = int(stats["active_count"]) + 1

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
async def resolve_alert(alert_id: str, resolution_note: Optional[str] = None) -> Any:
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
async def get_monitoring_dashboard() -> Any:
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

        dashboard_data: Dict[str, Any] = {
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
            "recent_metrics": (
                recent_metrics[-20:] if recent_metrics else []
            ),  # 最近20个指标点
        }

        # 统计告警严重程度
        active_alerts_data = cast(Dict[str, Any], dashboard_data["active_alerts"])
        severity_counts = cast(Dict[str, int], active_alerts_data["severity_counts"])
        for alert in active_alerts:
            severity = str(alert.get("severity", "unknown"))
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

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
) -> Any:
    """测试告警配置"""
    try:
        # 触发测试告警
        test_result = alert_manager.test_alert(
            alert_type=alert_type, metric_name=metric_name, test_value=test_value
        )

        return StandardResponse(success=True, message="告警测试完成", data=test_result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"告警测试失败: {str(e)}")
