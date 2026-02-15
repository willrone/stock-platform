"""
滚动训练集成模块

将滚动训练逻辑集成到 UnifiedQlibTrainingEngine。
从 engine.py 中拆分出来以遵守文件 ≤500 行规范。
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd
from loguru import logger

from .config import QlibTrainingConfig, QlibTrainingResult
from .qlib_check import QLIB_AVAILABLE
from .rolling_config import RollingTrainingConfig
from .rolling_executor import execute_rolling_training


async def rolling_train_single_window(
    model_config: dict,
    train_data: pd.DataFrame,
    valid_data: pd.DataFrame,
    config: QlibTrainingConfig,
    sample_weights=None,
):
    """滚动训练中单个窗口的训练函数

    Args:
        model_config: Qlib 模型配置
        train_data: 训练数据
        valid_data: 验证数据
        config: 训练配置
        sample_weights: 样本权重（时间衰减）

    Returns:
        (model, metrics) 元组
    """
    from .dataset_adapter import (
        DataFrameDatasetAdapter,
        _create_label_for_data,
    )
    from .data_preprocessing import (
        OutlierHandler,
        RobustFeatureScaler,
    )

    if not QLIB_AVAILABLE:
        raise RuntimeError("Qlib 不可用")

    from qlib.utils import init_instance_by_config

    # Bug 3 修复：在缩放前先从原始 $close 创建 label，
    # 避免 scaler 污染 $close 后再生成 label（数据泄漏）
    _create_label_for_data(
        train_data, "训练集", config.prediction_horizon,
        config.label_type, config.binary_threshold,
    )
    if valid_data is not None:
        _create_label_for_data(
            valid_data, "验证集", config.prediction_horizon,
            config.label_type, config.binary_threshold,
        )

    # Bug 3 修复：排除 label 列和 $close 列，
    # $close 是 label 的计算基础，不应作为特征输入模型
    label_source_cols = {"label", "$close", "close", "Close", "CLOSE"}
    feature_cols = [
        c for c in train_data.columns
        if c not in label_source_cols
    ]

    # 异常值处理
    outlier_handler = OutlierHandler(
        method="winsorize",
        lower_percentile=0.01,
        upper_percentile=0.99,
    )
    if "label" in train_data.columns:
        train_data = outlier_handler.handle_label_outliers(
            train_data, label_col="label",
        )

    # 标准化（仅缩放特征列，label 和 $close 不受影响）
    if feature_cols:
        scaler = RobustFeatureScaler()
        train_data = scaler.fit_transform(train_data, feature_cols)
        if valid_data is not None and len(valid_data) > 0:
            valid_data = scaler.transform(valid_data, feature_cols)

    adapter = DataFrameDatasetAdapter(
        train_data, valid_data,
        config.prediction_horizon, config,
    )

    model = init_instance_by_config(model_config)
    model.fit(adapter)

    # Bug 1 修复：传入 adapter 内部的 val_data（已含 label），
    # 而非外部的 valid_data 引用（可能因 copy 不同步）
    metrics = _evaluate_window(model, adapter)
    return model, metrics


def _compute_direction_accuracy(
    predictions: np.ndarray,
    actuals: np.ndarray,
) -> float:
    """计算方向准确率（预测涨跌方向是否正确）

    Args:
        predictions: 模型预测值
        actuals: 真实标签值

    Returns:
        方向准确率 (0.0~1.0)
    """
    if len(predictions) < 1:
        return 0.0
    threshold = 1e-6
    pred_dir = np.sign(predictions)
    true_dir = np.sign(actuals)
    # 将接近零的值视为零方向
    pred_dir[np.abs(predictions) < threshold] = 0
    true_dir[np.abs(actuals) < threshold] = 0
    return float(np.mean(pred_dir == true_dir))


def _predict_on_data(
    model: Any,
    data: pd.DataFrame,
) -> np.ndarray:
    """用模型对数据做预测，返回 numpy 数组

    Args:
        model: 已训练的 Qlib 模型
        data: 含 label 列的 DataFrame

    Returns:
        预测值数组
    """
    from .dataset_adapter import DataFrameDatasetAdapter
    adapter = DataFrameDatasetAdapter(
        data, None,
        prediction_horizon=0,
    )
    preds = model.predict(adapter)
    if hasattr(preds, "values"):
        return preds.values.flatten()
    return np.asarray(preds).flatten()


def _evaluate_window(
    model: Any,
    adapter: Any,
) -> Dict[str, float]:
    """评估单个窗口的模型性能

    计算验证集的 IC、MSE、方向准确率，以及训练集的方向准确率。

    Args:
        model: 已训练的 Qlib 模型
        adapter: DataFrameDatasetAdapter（内含 train/valid 数据）

    Returns:
        包含 ic, mse, val_accuracy, train_accuracy 的指标字典
    """
    metrics: Dict[str, float] = {
        "ic": 0.0,
        "mse": 0.0,
        "val_accuracy": 0.0,
        "train_accuracy": 0.0,
    }

    # --- 验证集指标 ---
    valid_data = getattr(adapter, "val_data", None)
    if valid_data is not None and "label" in valid_data.columns:
        try:
            val_preds = _predict_on_data(model, valid_data)
            val_actuals = valid_data["label"].values
            min_len = min(len(val_preds), len(val_actuals))
            val_preds = val_preds[:min_len]
            val_actuals = val_actuals[:min_len]

            ic = float(np.corrcoef(val_preds, val_actuals)[0, 1])
            metrics["ic"] = ic if np.isfinite(ic) else 0.0
            metrics["mse"] = float(np.mean((val_preds - val_actuals) ** 2))
            metrics["val_accuracy"] = _compute_direction_accuracy(
                val_preds, val_actuals,
            )
        except Exception as e:
            logger.warning(f"滚动窗口验证集评估失败: {e}")

    # --- 训练集方向准确率 ---
    train_data = getattr(adapter, "train_data", None)
    if train_data is None:
        train_data = getattr(adapter, "data", None)
    if train_data is not None and "label" in train_data.columns:
        try:
            train_preds = _predict_on_data(model, train_data)
            train_actuals = train_data["label"].values
            min_len = min(len(train_preds), len(train_actuals))
            metrics["train_accuracy"] = _compute_direction_accuracy(
                train_preds[:min_len], train_actuals[:min_len],
            )
        except Exception as e:
            logger.warning(f"滚动窗口训练集评估失败: {e}")

    return metrics


async def run_rolling_training(
    dataset: pd.DataFrame,
    config: QlibTrainingConfig,
    model_id: str,
    start_time: datetime,
    model_manager: Any,
    performance_monitor: Any,
    progress_callback: Optional[Callable] = None,
) -> QlibTrainingResult:
    """执行滚动训练流程

    Args:
        dataset: 完整数据集
        config: 训练配置
        model_id: 模型 ID
        start_time: 训练开始时间
        model_manager: QlibModelManager 实例
        performance_monitor: 性能监控器
        progress_callback: 进度回调

    Returns:
        QlibTrainingResult
    """
    from app.core.config import settings
    from .model_config import create_qlib_model_config

    logger.info(f"进入滚动训练模式: {model_id}")

    rolling_config = RollingTrainingConfig(
        enable_rolling=True,
        window_type=config.rolling_window_type,
        rolling_step=config.rolling_step,
        train_window=config.rolling_train_window,
        valid_window=config.rolling_valid_window,
        enable_sample_decay=config.enable_sample_decay,
        decay_rate=config.sample_decay_rate,
    )

    base_dir = Path(settings.MODEL_STORAGE_PATH) / model_id

    async def model_config_factory(cfg):
        return await create_qlib_model_config(model_manager, cfg)

    rolling_result = await execute_rolling_training(
        dataset=dataset,
        config=config,
        rolling_config=rolling_config,
        model_id=model_id,
        model_config_factory=model_config_factory,
        train_single_fn=rolling_train_single_window,
        base_dir=base_dir,
        progress_callback=progress_callback,
    )

    training_duration = (datetime.now() - start_time).total_seconds()

    if progress_callback:
        await progress_callback(
            model_id, 100.0, "completed",
            f"滚动训练完成: {rolling_result['total_windows']} 个窗口",
            {"rolling_result": rolling_result},
        )

    performance_monitor.end_stage("total_training")
    performance_monitor.print_summary()

    # 从 per_window_metrics 提取平均方向准确率
    per_win = rolling_result["per_window_metrics"]
    avg_train_acc = float(np.mean(
        [m.get("train_accuracy", 0.0) for m in per_win],
    )) if per_win else 0.0
    avg_val_acc = float(np.mean(
        [m.get("val_accuracy", 0.0) for m in per_win],
    )) if per_win else 0.0

    return QlibTrainingResult(
        model_path=rolling_result["latest_model_path"],
        model_config={"rolling": rolling_result},
        training_metrics={
            "avg_ic": rolling_result["avg_ic"],
            "accuracy": avg_train_acc,
            "direction_accuracy": avg_train_acc,
        },
        validation_metrics={
            "avg_ic": rolling_result["avg_ic"],
            "avg_mse": rolling_result["avg_mse"],
            "ic_trend": rolling_result["ic_trend"],
            "accuracy": avg_val_acc,
            "direction_accuracy": avg_val_acc,
        },
        feature_importance=None,
        training_history=[
            {
                "epoch": i + 1,
                "window": i,
                "train_loss": m.get("mse", 0.0),
                "val_loss": m.get("mse", 0.0),
                "train_accuracy": m.get("train_accuracy", 0.0),
                "val_accuracy": m.get("val_accuracy", 0.0),
                "ic": m.get("ic", 0.0),
            }
            for i, m in enumerate(per_win)
        ],
        training_duration=training_duration,
    )
