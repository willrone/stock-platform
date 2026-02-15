"""
滚动训练执行器

核心逻辑：按滚动窗口依次训练模型，管理版本，监控 IC 衰减。
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from .config import QlibTrainingConfig
from .model_version_manager import (
    ModelVersion,
    compute_ic_decay_report,
    save_model_version,
    save_version_manifest,
)
from .rolling_config import RollingTrainingConfig
from .rolling_window_gen import RollingWindow, generate_rolling_windows


def compute_sample_weights(
    dates: pd.DatetimeIndex,
    decay_rate: float,
) -> np.ndarray:
    """计算样本时间衰减权重

    Args:
        dates: 样本日期索引
        decay_rate: 每天衰减率（如 0.999）

    Returns:
        权重数组，近期样本权重更高
    """
    if len(dates) == 0:
        return np.array([])

    # 防御性转换：上游可能传入字符串类型的日期索引
    if not isinstance(dates, pd.DatetimeIndex):
        dates = pd.DatetimeIndex(dates)

    max_date = dates.max()
    days_ago = (max_date - dates).days
    weights = np.power(decay_rate, days_ago.values)
    return weights


def _get_date_level(index: pd.MultiIndex) -> int:
    """检测 MultiIndex 中哪个 level 包含日期

    兼容 (instrument, datetime) 和 (date, instrument) 两种格式。
    """
    for i in range(index.nlevels):
        level_values = index.get_level_values(i)
        if isinstance(level_values, pd.DatetimeIndex):
            return i
        try:
            pd.Timestamp(level_values[0])
            return i
        except (ValueError, TypeError):
            continue
    raise ValueError(
        f"MultiIndex 中未找到日期 level，names={index.names}",
    )


def _extract_dates_from_dataset(
    dataset: pd.DataFrame,
) -> pd.DatetimeIndex:
    """从数据集中提取排序后的唯一日期

    注意：索引可能是字符串类型，强制转为 DatetimeIndex
    以确保滚动窗口生成和日期切片正常工作。
    """
    if isinstance(dataset.index, pd.MultiIndex):
        date_level = _get_date_level(dataset.index)
        raw = dataset.index.get_level_values(date_level).unique().sort_values()
    else:
        raw = dataset.index.unique().sort_values()

    if not isinstance(raw, pd.DatetimeIndex):
        return pd.DatetimeIndex(raw)
    return raw


def _slice_dataset_by_dates(
    dataset: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """按日期范围切片数据集"""
    if isinstance(dataset.index, pd.MultiIndex):
        date_level = _get_date_level(dataset.index)
        dates = dataset.index.get_level_values(date_level)
        mask = (dates >= start_date) & (dates <= end_date)
        return dataset[mask]
    mask = (dataset.index >= start_date) & (dataset.index <= end_date)
    return dataset[mask]


def _compute_window_ic(
    predictions: np.ndarray,
    actuals: np.ndarray,
) -> float:
    """计算单窗口 IC（Pearson 相关系数）"""
    if len(predictions) < 2:
        return 0.0
    corr = np.corrcoef(predictions, actuals)[0, 1]
    return float(corr) if np.isfinite(corr) else 0.0


async def execute_rolling_training(
    dataset: pd.DataFrame,
    config: QlibTrainingConfig,
    rolling_config: RollingTrainingConfig,
    model_id: str,
    model_config_factory: Callable,
    train_single_fn: Callable,
    base_dir: Path,
    progress_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    """执行滚动训练

    Args:
        dataset: ��整数据集（含特征���标签）
        config: Qlib 训练配置
        rolling_config: 滚动训练配置
        model_id: 模型 ID
        model_config_factory: 创建模型配置的工厂函数
        train_single_fn: 单次训练函数
        base_dir: 模型存储根目录
        progress_callback: 进度回调

    Returns:
        滚动训练结果字典
    """
    dates = _extract_dates_from_dataset(dataset)
    windows = generate_rolling_windows(dates, rolling_config)

    if not windows:
        raise ValueError(
            "无法生成滚动窗口，数据量不足。"
            f"需要至少 {rolling_config.train_window + rolling_config.valid_window} 天",
        )

    logger.info(
        f"开始滚动训练: {len(windows)} 个窗口, "
        f"模型ID={model_id}",
    )

    versions: List[ModelVersion] = []
    all_metrics: List[Dict[str, float]] = []

    for idx, window in enumerate(windows):
        progress_pct = 10.0 + (idx / len(windows)) * 80.0
        if progress_callback:
            await progress_callback(
                model_id,
                progress_pct,
                "rolling_training",
                f"滚动窗口 {idx + 1}/{len(windows)}",
                {"window_id": window.window_id},
            )

        version, metrics = await _train_single_window(
            dataset=dataset,
            window=window,
            config=config,
            rolling_config=rolling_config,
            model_config_factory=model_config_factory,
            train_single_fn=train_single_fn,
            base_dir=base_dir,
        )

        versions.append(version)
        all_metrics.append(metrics)

    # 保存版本清单
    manifest_path = save_version_manifest(
        versions, base_dir, model_id,
    )

    # IC 衰减分析
    ic_report = compute_ic_decay_report(versions)

    result = _build_rolling_result(
        versions, all_metrics, ic_report, manifest_path,
    )

    logger.info(
        f"滚动训练完成: {len(versions)} 个版本, "
        f"平均IC={ic_report.avg_ic:.4f}, "
        f"趋势={ic_report.ic_trend}",
    )
    return result


async def _train_single_window(
    dataset: pd.DataFrame,
    window: RollingWindow,
    config: QlibTrainingConfig,
    rolling_config: RollingTrainingConfig,
    model_config_factory: Callable,
    train_single_fn: Callable,
    base_dir: Path,
) -> Tuple[ModelVersion, Dict[str, float]]:
    """训练单个滚动窗口"""
    train_data = _slice_dataset_by_dates(
        dataset, window.train_start, window.train_end,
    )
    valid_data = _slice_dataset_by_dates(
        dataset, window.valid_start, window.valid_end,
    )

    logger.info(
        f"窗口 {window.window_id}: "
        f"训练集={len(train_data)}行, 验证集={len(valid_data)}行",
    )

    # 样本时间衰减权重
    sample_weights = None
    if rolling_config.enable_sample_decay:
        train_dates = _extract_sample_dates(train_data)
        sample_weights = compute_sample_weights(
            train_dates, rolling_config.decay_rate,
        )

    # 创建模型配置并训练
    model_config = await model_config_factory(config)
    model, metrics = await train_single_fn(
        model_config, train_data, valid_data,
        config, sample_weights,
    )

    # 保存模型版本
    version_info = {
        "window_id": window.window_id,
        "train_start": str(window.train_start.date()),
        "train_end": str(window.train_end.date()),
        "valid_start": str(window.valid_start.date()),
        "valid_end": str(window.valid_end.date()),
        "metrics": metrics,
    }
    version = save_model_version(
        model, model_config, version_info, base_dir,
    )

    return version, metrics


def _extract_sample_dates(
    data: pd.DataFrame,
) -> pd.DatetimeIndex:
    """提取每个样本的日期

    注意：DataFrame 索引可能是字符串类型（如从 CSV/Parquet 读取后未转换），
    此处强制转为 DatetimeIndex 以确保下游日期运算正常。
    """
    if isinstance(data.index, pd.MultiIndex):
        date_level = _get_date_level(data.index)
        raw = data.index.get_level_values(date_level)
    else:
        raw = data.index

    if not isinstance(raw, pd.DatetimeIndex):
        return pd.DatetimeIndex(raw)
    return raw


def _build_rolling_result(
    versions: List[ModelVersion],
    all_metrics: List[Dict[str, float]],
    ic_report: Any,
    manifest_path: str,
) -> Dict[str, Any]:
    """构建滚动训练结果"""
    # 过滤 None 和 NaN 值
    ic_values = [
        float(m.get("ic", 0.0)) if m.get("ic") is not None and np.isfinite(m.get("ic", 0.0)) else 0.0
        for m in all_metrics
    ]
    mse_values = [
        float(m.get("mse", 0.0)) if m.get("mse") is not None and np.isfinite(m.get("mse", 0.0)) else 0.0
        for m in all_metrics
    ]

    return {
        "total_windows": len(versions),
        "manifest_path": manifest_path,
        "avg_ic": float(np.mean(ic_values)) if ic_values else 0.0,
        "avg_mse": float(np.mean(mse_values)) if mse_values else 0.0,
        "ic_trend": ic_report.ic_trend,
        "ic_decay_detected": ic_report.decay_detected,
        "per_window_metrics": all_metrics,
        "version_ids": [v.version_id for v in versions],
        "latest_model_path": versions[-1].model_path if versions else "",
    }
