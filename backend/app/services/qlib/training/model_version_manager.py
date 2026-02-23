"""
模型版本管理器

管理滚动训练产生的多版本模型：
- 按时间戳保存模型版本
- 按日期自动选择对应版本
- 监控模型性能衰减（IC 随时间变化）
"""

import json
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from .rolling_config import IC_DECAY_WARN_THRESHOLD


@dataclass
class ModelVersion:
    """单个模型版本"""

    version_id: str
    model_path: str
    train_start: str
    train_end: str
    valid_start: str
    valid_end: str
    created_at: str
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class ICDecayReport:
    """IC 衰减监控报告"""

    version_ids: List[str]
    ic_values: List[float]
    ic_trend: str  # "stable", "declining", "improving"
    avg_ic: float
    ic_std: float
    decay_detected: bool


def save_model_version(
    model: Any,
    model_config: dict,
    version_info: dict,
    base_dir: Path,
) -> ModelVersion:
    """保存一个模型版本

    Args:
        model: 训练好的模型对象
        model_config: 模型配置
        version_info: 版本信息（含 window_id, train/valid 日期等）
        base_dir: 模��存储根目录

    Returns:
        ModelVersion 对象
    """
    versions_dir = base_dir / "rolling_versions"
    versions_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    window_id = version_info.get("window_id", 0)
    version_id = f"w{window_id}_{timestamp}"

    model_path = versions_dir / f"{version_id}.pkl"
    payload = {
        "model": model,
        "config": model_config,
        "version_info": version_info,
        "created_at": timestamp,
    }
    with open(model_path, "wb") as f:
        pickle.dump(payload, f)

    version = ModelVersion(
        version_id=version_id,
        model_path=str(model_path),
        train_start=version_info.get("train_start", ""),
        train_end=version_info.get("train_end", ""),
        valid_start=version_info.get("valid_start", ""),
        valid_end=version_info.get("valid_end", ""),
        created_at=timestamp,
        metrics=version_info.get("metrics", {}),
    )

    logger.info(
        f"模型版本已保存: {version_id} "
        f"(训练期: {version.train_start}~{version.train_end})",
    )
    return version


def save_version_manifest(
    versions: List[ModelVersion],
    base_dir: Path,
    model_id: str,
) -> str:
    """保存版本清单文件

    Args:
        versions: 所有版本列表
        base_dir: 模型存储根目录
        model_id: 模型 ID

    Returns:
        清单文件路径
    """
    manifest_path = base_dir / "rolling_versions" / "manifest.json"
    manifest = {
        "model_id": model_id,
        "total_versions": len(versions),
        "created_at": datetime.now().isoformat(),
        "versions": [
            {
                "version_id": v.version_id,
                "model_path": v.model_path,
                "train_start": v.train_start,
                "train_end": v.train_end,
                "valid_start": v.valid_start,
                "valid_end": v.valid_end,
                "created_at": v.created_at,
                "metrics": v.metrics,
            }
            for v in versions
        ],
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    logger.info(f"版本清单已保存: {manifest_path}")
    return str(manifest_path)


def load_model_for_date(
    target_date: str,
    base_dir: Path,
) -> Tuple[Any, dict]:
    """按日期自动选择对应版本的模型

    选择逻辑：找到 valid_end <= target_date 的最新版本。

    Args:
        target_date: 目标日期（YYYY-MM-DD）
        base_dir: 模型存储根目录

    Returns:
        (model, version_info) 元组
    """
    manifest_path = base_dir / "rolling_versions" / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"版本清单不存在: {manifest_path}",
        )

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    versions = manifest.get("versions", [])
    if not versions:
        raise ValueError("版本清单为空")

    # 找到 valid_end <= target_date 的最新版本
    candidates = [
        v for v in versions
        if v["valid_end"] <= target_date
    ]

    if not candidates:
        # 回退到最早的版本
        logger.warning(
            f"未找到 valid_end <= {target_date} 的版本，"
            f"使用最早版本",
        )
        selected = versions[0]
    else:
        # 选择 valid_end 最大的（最新的）
        selected = max(candidates, key=lambda v: v["valid_end"])

    model_path = selected["model_path"]
    with open(model_path, "rb") as f:
        payload = pickle.load(f)

    logger.info(
        f"加载模型版本: {selected['version_id']} "
        f"(目标日期: {target_date})",
    )
    return payload["model"], selected


def compute_ic_decay_report(
    versions: List[ModelVersion],
) -> ICDecayReport:
    """计算 IC 衰减报告

    Args:
        versions: 按时间排序的模型版本列表

    Returns:
        ICDecayReport
    """
    # 提取 IC 值，过滤 None 和 NaN
    raw_ic_values = [
        v.metrics.get("ic", 0.0) for v in versions
    ]
    ic_values = [
        float(ic) if ic is not None and np.isfinite(ic) else 0.0
        for ic in raw_ic_values
    ]
    version_ids = [v.version_id for v in versions]

    if len(ic_values) < 2:
        return ICDecayReport(
            version_ids=version_ids,
            ic_values=ic_values,
            ic_trend="stable",
            avg_ic=ic_values[0] if ic_values else 0.0,
            ic_std=0.0,
            decay_detected=False,
        )

    avg_ic = float(np.mean(ic_values))
    ic_std = float(np.std(ic_values))

    # 简单线性趋势判断
    ic_trend = _detect_ic_trend(ic_values)

    # 检测衰减：最近 IC 比平均低超过阈值
    recent_ic = float(np.mean(ic_values[-2:]))
    early_ic = float(np.mean(ic_values[:2]))
    
    # 防护：确保两个值都是有效数字
    if not (np.isfinite(early_ic) and np.isfinite(recent_ic)):
        logger.warning(
            f"IC 衰减检测跳过: early_ic={early_ic}, recent_ic={recent_ic} 包含无效值"
        )
        decay_detected = False
    else:
        decay_detected = (early_ic - recent_ic) > IC_DECAY_WARN_THRESHOLD

    if decay_detected:
        logger.warning(
            f"检测到 IC 衰减: 早期={early_ic:.4f} → "
            f"近期={recent_ic:.4f}",
        )

    return ICDecayReport(
        version_ids=version_ids,
        ic_values=ic_values,
        ic_trend=ic_trend,
        avg_ic=avg_ic,
        ic_std=ic_std,
        decay_detected=decay_detected,
    )


def _detect_ic_trend(ic_values: List[float]) -> str:
    """检测 IC 趋势方向"""
    if len(ic_values) < 3:
        return "stable"

    # 用简单线性回归斜率判断
    x = np.arange(len(ic_values), dtype=float)
    y = np.array(ic_values, dtype=float)
    slope = np.polyfit(x, y, 1)[0]

    if slope > IC_DECAY_WARN_THRESHOLD / len(ic_values):
        return "improving"
    elif slope < -IC_DECAY_WARN_THRESHOLD / len(ic_values):
        return "declining"
    return "stable"
