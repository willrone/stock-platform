"""模型训练相关 DTO 构建工具。"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List


def _safe_json_dict(value: Any) -> Dict[str, Any]:
    """将 JSON 字段规整为字典。"""
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            import json

            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _extract_accuracy(performance_metrics: Dict[str, Any]) -> float:
    """从 performance_metrics 中提取准确率。"""
    accuracy = performance_metrics.get("accuracy", 0.0)
    if isinstance(accuracy, dict):
        accuracy = accuracy.get("value", 0.0)

    try:
        return float(accuracy or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _build_training_data_period(model: Any) -> Dict[str, str]:
    """构建训练数据周期。"""
    if not model.training_data_start or not model.training_data_end:
        return {}

    return {
        "start": model.training_data_start.isoformat(),
        "end": model.training_data_end.isoformat(),
    }


def _extract_stock_codes(model: Any) -> List[str]:
    """从评估报告中提取股票列表。"""
    evaluation_report = _safe_json_dict(getattr(model, "evaluation_report", None))
    training_data_info = evaluation_report.get("training_data_info", {})
    if not isinstance(training_data_info, dict):
        return []

    stock_codes = training_data_info.get("stock_codes", [])
    return stock_codes if isinstance(stock_codes, list) else []


def _map_model_status_to_task_status(status: str) -> str:
    """将模型状态映射为旧训练任务状态。"""
    if status == "ready":
        return "completed"
    if status == "failed":
        return "failed"
    if status == "training":
        return "running"
    return status


def build_model_list_item_dto(model: Any) -> Dict[str, Any]:
    """构建 /models 列表项 DTO。"""
    performance_metrics = _safe_json_dict(getattr(model, "performance_metrics", None))

    created_at = getattr(model, "created_at", None)
    if created_at:
        created_at_iso = created_at.isoformat()
    else:
        created_at_iso = datetime.now().isoformat()

    return {
        "model_id": model.model_id,
        "model_name": model.model_name,
        "model_type": model.model_type,
        "version": model.version,
        "accuracy": _extract_accuracy(performance_metrics),
        "created_at": created_at_iso,
        "status": model.status,
        "training_progress": model.training_progress or 0.0,
        "training_stage": model.training_stage,
    }


def build_model_detail_dto(model: Any) -> Dict[str, Any]:
    """构建 /models/{model_id} 详情 DTO。"""
    performance_metrics = _safe_json_dict(getattr(model, "performance_metrics", None))
    created_at = getattr(model, "created_at", None)

    return {
        "model_id": model.model_id,
        "model_name": model.model_name,
        "model_type": model.model_type,
        "version": model.version,
        "accuracy": _extract_accuracy(performance_metrics),
        "description": f"{model.model_type}模型 - {model.model_name}",
        "performance_metrics": performance_metrics,
        "training_info": {
            "training_data_period": _build_training_data_period(model),
            "hyperparameters": model.hyperparameters or {},
            "stock_codes": _extract_stock_codes(model),
        },
        "created_at": (
            created_at.isoformat() if created_at else datetime.now().isoformat()
        ),
        "status": model.status,
    }


def build_training_progress_dto_from_model(model: Any) -> Dict[str, Any]:
    """根据模型信息构建旧训练进度接口 DTO（兼容老调用方）。"""
    created_at = getattr(model, "created_at", None)
    updated_at = getattr(model, "updated_at", None)
    now = datetime.now()
    start_time = created_at or now

    stage = model.training_stage or model.status or "training"
    progress = float(model.training_progress or 0.0)

    return {
        "task_id": model.model_id,
        "status": _map_model_status_to_task_status(model.status),
        "progress_percentage": progress,
        "created_at": start_time.isoformat(),
        "updated_at": (updated_at or now).isoformat(),
        "elapsed_time": (now - start_time).total_seconds(),
        "current_epoch": 0,
        "total_epochs": 0,
        "current_batch": 0,
        "total_batches": 0,
        "current_loss": None,
        "best_loss": None,
        "current_accuracy": None,
        "best_accuracy": None,
        "learning_rate": None,
        "estimated_remaining": None,
        "stage": stage,
    }
