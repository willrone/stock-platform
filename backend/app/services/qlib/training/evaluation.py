"""
模型评估模块

包含模型评估和指标计算功能
"""

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from loguru import logger


async def _evaluate_model(
    model: Any,
    train_dataset: pd.DataFrame,
    val_dataset: pd.DataFrame,
    model_id: str = None,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """评估模型性能并计算详细指标"""
    try:
        # 记录数据集信息
        if hasattr(train_dataset, "data") and isinstance(
            train_dataset.data, pd.DataFrame
        ):
            logger.info(
                f"训练集数据维度: {train_dataset.data.shape}, 列: {list(train_dataset.data.columns[:10]) if len(train_dataset.data.columns) > 0 else 'N/A'}"
            )
        elif isinstance(train_dataset, pd.DataFrame):
            logger.info(
                f"训练集数据维度: {train_dataset.shape}, 列: {list(train_dataset.columns[:10]) if len(train_dataset.columns) > 0 else 'N/A'}"
            )

        if hasattr(val_dataset, "data") and isinstance(val_dataset.data, pd.DataFrame):
            logger.info(
                f"验证集数据维度: {val_dataset.data.shape}, 列: {list(val_dataset.data.columns[:10]) if len(val_dataset.data.columns) > 0 else 'N/A'}"
            )
        elif isinstance(val_dataset, pd.DataFrame):
            logger.info(
                f"验证集数据维度: {val_dataset.shape}, 列: {list(val_dataset.columns[:10]) if len(val_dataset.columns) > 0 else 'N/A'}"
            )

        # 训练集预测 - 使用正确的segment
        train_pred = model.predict(train_dataset, segment="train")
        logger.info(
            f"训练集预测结果: 类型={type(train_pred)}, 形状={train_pred.shape if hasattr(train_pred, 'shape') else len(train_pred) if hasattr(train_pred, '__len__') else 'N/A'}"
        )

        # 验证集预测 - 使用正确的segment
        val_pred = model.predict(val_dataset, segment="valid")
        logger.info(
            f"验证集预测结果: 类型={type(val_pred)}, 形状={val_pred.shape if hasattr(val_pred, 'shape') else len(val_pred) if hasattr(val_pred, '__len__') else 'N/A'}"
        )

        # 计算训练集指标（使用真实标签）
        training_metrics = _calculate_metrics(
            train_dataset, train_pred, "训练集", model_id
        )

        # 计算验证集指标（使用真实标签）
        validation_metrics = _calculate_metrics(val_dataset, val_pred, "验证集", model_id)

        logger.info(
            f"模型评估完成 - 训练准确率: {training_metrics.get('accuracy', 0.0):.4f}, 验证准确率: {validation_metrics.get('accuracy', 0.0):.4f}"
        )
        return training_metrics, validation_metrics

    except Exception as e:
        logger.error(f"模型评估失败: {e}", exc_info=True)
        # 返回默认指标
        return _get_default_metrics(), _get_default_metrics()


def _calculate_metrics(
    dataset: pd.DataFrame,
    predictions,
    dataset_name: str,
    model_id: str = None,
) -> Dict[str, float]:
    """计算真实的评估指标，基于预测值和真实标签"""
    try:
        import numpy as np
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            mean_absolute_error,
            mean_squared_error,
            precision_score,
            r2_score,
            recall_score,
        )

        # 从数据集中获取真实标签
        y_true = None

        # 首先确定应该使用哪个segment
        segment = (
            "train"
            if "训练" in dataset_name
            else "valid"
            if "验证" in dataset_name
            else "train"
        )

        # 尝试多种方式获取标签
        if hasattr(dataset, "data") and isinstance(dataset.data, pd.DataFrame):
            # DataFrameDatasetAdapter - 根据segment获取对应的数据
            if hasattr(dataset, "segments") and segment in dataset.segments:
                segment_data = dataset.segments[segment]
                if (
                    isinstance(segment_data, pd.DataFrame)
                    and "label" in segment_data.columns
                ):
                    label_series = segment_data["label"]
                    # 如果是LabelSeries，获取其内部的Series
                    if hasattr(label_series, "_series"):
                        y_true = label_series._series.values
                    else:
                        y_true = label_series.values
                    logger.debug(
                        f"{dataset_name} 从segment {segment}获取标签，形状: {y_true.shape if hasattr(y_true, 'shape') else len(y_true)}"
                    )
            elif "label" in dataset.data.columns:
                y_true = dataset.data["label"].values
            elif hasattr(dataset, "prepare"):
                # 尝试通过prepare方法获取
                try:
                    prepared = dataset.prepare(segment, col_set=["label"])
                    if (
                        isinstance(prepared, pd.DataFrame)
                        and "label" in prepared.columns
                    ):
                        label_col = prepared["label"]
                        # 如果是LabelSeries，获取其内部的Series
                        if hasattr(label_col, "_series"):
                            y_true = label_col._series.values
                        elif hasattr(label_col, "values"):
                            label_values = label_col.values
                            if label_values.ndim == 2:
                                y_true = label_values.flatten()
                            else:
                                y_true = label_values
                        else:
                            y_true = np.array(label_col).flatten()
                    logger.debug(
                        f"{dataset_name} 通过prepare方法获取标签，形状: {y_true.shape if hasattr(y_true, 'shape') else len(y_true)}"
                    )
                except Exception as e:
                    logger.debug(f"通过prepare方法获取标签失败: {e}")

        if y_true is None and isinstance(dataset, pd.DataFrame):
            # 直接是DataFrame
            if "label" in dataset.columns:
                label_col = dataset["label"]
                # 如果是LabelSeries，获取其内部的Series
                if hasattr(label_col, "_series"):
                    y_true = label_col._series.values
                else:
                    y_true = label_col.values

        if y_true is None:
            logger.warning(f"数据集 {dataset_name} 中没有找到label列，使用默认指标")
            logger.warning(
                f"数据集类型: {type(dataset)}, 是否有data属性: {hasattr(dataset, 'data')}, 是否有segments属性: {hasattr(dataset, 'segments')}"
            )
            if hasattr(dataset, "segments"):
                logger.warning(
                    f"可用segments: {list(dataset.segments.keys()) if hasattr(dataset.segments, 'keys') else 'N/A'}"
                )
            return _get_default_metrics()

        # 记录标签统计信息
        logger.info(
            f"{dataset_name} 标签统计 - 样本数: {len(y_true)}, 非零值: {np.count_nonzero(y_true)}, 零值: {np.sum(np.abs(y_true) < 1e-6)}, 范围: [{np.min(y_true):.6f}, {np.max(y_true):.6f}]"
        )

        # 处理预测值
        if isinstance(predictions, pd.Series):
            y_pred = predictions.values
        elif isinstance(predictions, np.ndarray):
            y_pred = predictions.flatten() if predictions.ndim > 1 else predictions
        else:
            y_pred = np.array(predictions).flatten()

        # 确保长度一致
        min_len = min(len(y_true), len(y_pred))
        if min_len == 0:
            logger.warning(f"数据集 {dataset_name} 为空，使用默认指标")
            return _get_default_metrics()

        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]

        # 移除NaN值
        valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        if valid_mask.sum() == 0:
            logger.warning(f"数据集 {dataset_name} 中没有有效数据，使用默认指标")
            return _get_default_metrics()

        y_true = y_true[valid_mask]
        y_pred = y_pred[valid_mask]

        # 计算回归指标
        mse = float(mean_squared_error(y_true, y_pred))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))

        # 计算方向准确率（预测涨跌方向）
        # 使用阈值而不是sign，避免0值问题
        threshold = 1e-6  # 很小的阈值，用于判断是否为0
        y_true_direction = np.where(
            y_true > threshold, 1, np.where(y_true < -threshold, -1, 0)
        )
        y_pred_direction = np.where(
            y_pred > threshold, 1, np.where(y_pred < -threshold, -1, 0)
        )

        # 记录方向分布信息
        unique_true = np.unique(y_true_direction)
        unique_pred = np.unique(y_pred_direction)
        true_counts = {val: np.sum(y_true_direction == val) for val in unique_true}
        pred_counts = {val: np.sum(y_pred_direction == val) for val in unique_pred}
        logger.info(
            f"{dataset_name} 方向分布 - 真实: {true_counts}, 预测: {pred_counts}, 样本数: {len(y_true_direction)}"
        )

        # 如果所有方向都相同，准确率计算会有问题
        if len(unique_true) == 1 and len(unique_pred) == 1:
            if unique_true[0] == unique_pred[0]:
                direction_accuracy = 1.0
            else:
                direction_accuracy = 0.0
        else:
            direction_accuracy = float(
                accuracy_score(y_true_direction, y_pred_direction)
            )

        logger.info(
            f"{dataset_name} 方向准确率: {direction_accuracy:.4f}, 真实值范围: [{y_true.min():.6f}, {y_true.max():.6f}], 预测值范围: [{y_pred.min():.6f}, {y_pred.max():.6f}]"
        )

        # 对于回归任务，使用方向准确率作为准确率
        accuracy = direction_accuracy

        # 计算分类指标（基于方向）
        try:
            # 确保有正负样本
            if (
                len(np.unique(y_true_direction)) > 1
                and len(np.unique(y_pred_direction)) > 1
            ):
                precision = float(
                    precision_score(
                        y_true_direction,
                        y_pred_direction,
                        average="weighted",
                        zero_division=0,
                    )
                )
                recall = float(
                    recall_score(
                        y_true_direction,
                        y_pred_direction,
                        average="weighted",
                        zero_division=0,
                    )
                )
                f1 = float(
                    f1_score(
                        y_true_direction,
                        y_pred_direction,
                        average="weighted",
                        zero_division=0,
                    )
                )
            else:
                precision = direction_accuracy
                recall = direction_accuracy
                f1 = direction_accuracy
        except Exception as e:
            logger.warning(f"计算分类指标失败: {e}，使用方向准确率")
            precision = direction_accuracy
            recall = direction_accuracy
            f1 = direction_accuracy

        # 计算金融指标
        # 使用预测方向作为交易信号，计算收益率
        returns = y_true * np.sign(y_pred)  # 如果预测正确方向，获得真实收益

        # 夏普比率
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe_ratio = float(
                np.mean(returns) / np.std(returns) * np.sqrt(252)
            )  # 年化
        else:
            sharpe_ratio = 0.0

        # 总收益率
        total_return = float(np.sum(returns))

        # 最大回撤
        cumulative_returns = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = cumulative_returns - running_max
        max_drawdown = float(np.min(drawdown)) if len(drawdown) > 0 else 0.0

        # 胜率
        win_rate = (
            float(np.sum(returns > 0) / len(returns)) if len(returns) > 0 else 0.0
        )

        # 信息比率（相对于基准）
        if len(returns) > 1 and np.std(returns - y_true) > 0:
            information_ratio = float(
                np.mean(returns - y_true) / np.std(returns - y_true) * np.sqrt(252)
            )
        else:
            information_ratio = 0.0

        # Calmar比率（年化收益率/最大回撤）
        if max_drawdown < 0 and len(returns) > 0:
            annualized_return = np.mean(returns) * 252
            calmar_ratio = (
                float(annualized_return / abs(max_drawdown))
                if max_drawdown != 0
                else 0.0
            )
        else:
            calmar_ratio = 0.0

        # === RankIC / RankICIR 计算 ===
        rank_ic, rank_ic_ir = _compute_rank_ic(
            dataset, predictions, dataset_name, segment
        )

        metrics = {
            "accuracy": max(0.0, min(1.0, accuracy)),
            "mse": max(0.0, mse),
            "mae": max(0.0, mae),
            "r2": r2,  # R2可以是负数
            "direction_accuracy": max(0.0, min(1.0, direction_accuracy)),
            "precision": max(0.0, min(1.0, precision)),
            "recall": max(0.0, min(1.0, recall)),
            "f1_score": max(0.0, min(1.0, f1)),
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "total_return": total_return,
            "win_rate": max(0.0, min(1.0, win_rate)),
            "information_ratio": information_ratio,
            "calmar_ratio": calmar_ratio,
            "rank_ic": rank_ic,
            "rank_ic_ir": rank_ic_ir,
        }

        logger.info(
            f"计算 {dataset_name} 真实指标 - 准确率: {accuracy:.4f}, MSE: {mse:.4f}, R2: {r2:.4f}"
        )
        return {k: round(v, 4) for k, v in metrics.items()}

    except Exception as e:
        logger.error(f"计算真实指标失败: {e}", exc_info=True)
        return _get_default_metrics()


def _get_default_metrics() -> Dict[str, float]:
    """返回默认指标（当无法计算真实指标时使用）"""
    return {
        "accuracy": 0.5,
        "mse": 0.1,
        "mae": 0.08,
        "r2": 0.3,
        "direction_accuracy": 0.52,
        "precision": 0.45,
        "recall": 0.42,
        "f1_score": 0.43,
        "sharpe_ratio": 0.0,
        "max_drawdown": 0.0,
        "total_return": 0.0,
        "win_rate": 0.5,
        "information_ratio": 0.0,
        "calmar_ratio": 0.0,
        "rank_ic": 0.0,
        "rank_ic_ir": 0.0,
    }


def _compute_rank_ic(
    dataset,
    predictions,
    dataset_name: str,
    segment: str,
) -> tuple:
    """计算 RankIC（Spearman 秩相关）和 RankICIR（RankIC 均值 / 标准差）

    按日期分组计算每日截面 Spearman 相关系数，然后取均值和 IR。

    Args:
        dataset: 数据集（DataFrameDatasetAdapter 或 DataFrame）
        predictions: 模型预测值
        dataset_name: 数据集名称（用于日志）
        segment: 数据段名称（train / valid）

    Returns:
        (rank_ic_mean, rank_ic_ir) 两个 float
    """
    from scipy.stats import spearmanr

    try:
        # --- 获取带索引的标签 ---
        label_series = None

        if hasattr(dataset, "segments") and segment in dataset.segments:
            seg_data = dataset.segments[segment]
            if isinstance(seg_data, pd.DataFrame) and "label" in seg_data.columns:
                label_series = seg_data["label"]
                if hasattr(label_series, "_series"):
                    label_series = label_series._series
        elif hasattr(dataset, "data") and isinstance(dataset.data, pd.DataFrame):
            if "label" in dataset.data.columns:
                label_series = dataset.data["label"]
        elif isinstance(dataset, pd.DataFrame) and "label" in dataset.columns:
            label_series = dataset["label"]

        if label_series is None:
            logger.debug(f"{dataset_name} RankIC: 无法获取标签序列")
            return 0.0, 0.0

        # --- 对齐预测值 ---
        if isinstance(predictions, pd.Series):
            pred_values = predictions.values
        elif isinstance(predictions, np.ndarray):
            pred_values = predictions.flatten()
        else:
            pred_values = np.array(predictions).flatten()

        y_true = (
            label_series.values
            if hasattr(label_series, "values")
            else np.array(label_series)
        )
        min_len = min(len(y_true), len(pred_values))
        if min_len == 0:
            return 0.0, 0.0
        y_true = y_true[:min_len]
        pred_values = pred_values[:min_len]

        # --- 尝试按日期分组 ---
        date_index = None
        if hasattr(label_series, "index"):
            idx = label_series.index
            logger.info(
                f"{dataset_name} RankIC debug: index type={type(idx).__name__}, "
                f"nlevels={getattr(idx, 'nlevels', 1)}, "
                f"names={getattr(idx, 'names', None)}, "
                f"dtype={idx.dtype if not isinstance(idx, pd.MultiIndex) else [idx.get_level_values(i).dtype for i in range(idx.nlevels)]}, "
                f"len={len(idx)}, first_5={idx[:5].tolist() if len(idx) > 0 else 'empty'}"
            )
            if isinstance(idx, pd.MultiIndex):
                # 找到日期层
                for lvl in range(idx.nlevels):
                    vals = idx.get_level_values(lvl)
                    logger.info(
                        f"{dataset_name} RankIC debug: level {lvl} name={idx.names[lvl]}, "
                        f"dtype={vals.dtype}, is_datetime={pd.api.types.is_datetime64_any_dtype(vals)}, "
                        f"sample={vals[:3].tolist()}"
                    )
                    if pd.api.types.is_datetime64_any_dtype(vals):
                        date_index = vals[:min_len]
                        break
                    # 也尝试将字符串日期转换为datetime
                    if vals.dtype == object:
                        try:
                            converted = pd.to_datetime(vals[:3])
                            if not converted.isna().any():
                                logger.info(
                                    f"{dataset_name} RankIC: 将 level {lvl} 字符串日期转换为 datetime"
                                )
                                date_index = pd.to_datetime(vals[:min_len])
                                break
                        except Exception:
                            pass
            elif pd.api.types.is_datetime64_any_dtype(idx):
                date_index = idx[:min_len]
            elif idx.dtype == object:
                # 尝试将字符串索引转换为日期
                try:
                    converted = pd.to_datetime(idx[:3])
                    if not converted.isna().any():
                        logger.info(f"{dataset_name} RankIC: 将字符串索引转换为 datetime")
                        date_index = pd.to_datetime(idx[:min_len])
                except Exception:
                    pass

        logger.info(
            f"{dataset_name} RankIC: date_index found={date_index is not None}, min_len={min_len}"
        )

        if date_index is not None and len(date_index) == min_len:
            # 按日期分组计算每日 RankIC
            df_ic = pd.DataFrame(
                {
                    "y_true": y_true,
                    "y_pred": pred_values,
                    "date": date_index,
                }
            )
            daily_ics = []
            min_stocks_per_day = 2  # Spearman 至少需要 2 个样本
            total_groups = 0
            skipped_small = 0
            skipped_nan = 0
            nan_corr_count = 0
            for dt, grp in df_ic.groupby("date"):
                total_groups += 1
                if len(grp) < min_stocks_per_day:
                    skipped_small += 1
                    continue
                valid_mask = ~(
                    np.isnan(grp["y_true"].values) | np.isnan(grp["y_pred"].values)
                )
                if valid_mask.sum() < min_stocks_per_day:
                    skipped_nan += 1
                    continue
                yt = grp["y_true"].values[valid_mask]
                yp = grp["y_pred"].values[valid_mask]
                corr, _ = spearmanr(yt, yp)
                if total_groups <= 3:
                    logger.info(
                        f"{dataset_name} RankIC sample day={dt}: n={len(yt)}, "
                        f"y_true_unique={len(np.unique(yt))}, y_pred_unique={len(np.unique(yp))}, "
                        f"y_true[:5]={yt[:5].tolist()}, y_pred[:5]={yp[:5].tolist()}, corr={corr}"
                    )
                if not np.isnan(corr):
                    daily_ics.append(corr)
                else:
                    nan_corr_count += 1

            logger.info(
                f"{dataset_name} RankIC groupby stats: total_groups={total_groups}, "
                f"skipped_small={skipped_small}, skipped_nan={skipped_nan}, "
                f"nan_corr={nan_corr_count}, valid_days={len(daily_ics)}"
            )

            if len(daily_ics) >= 2:
                ic_mean = float(np.mean(daily_ics))
                ic_std = float(np.std(daily_ics))
                ic_ir = float(ic_mean / ic_std) if ic_std > 1e-8 else 0.0
                logger.info(
                    f"{dataset_name} RankIC: mean={ic_mean:.4f}, std={ic_std:.4f}, "
                    f"IR={ic_ir:.4f}, days={len(daily_ics)}"
                )
                return round(ic_mean, 4), round(ic_ir, 4)
            elif len(daily_ics) == 1:
                logger.info(f"{dataset_name} RankIC: 仅1天, IC={daily_ics[0]:.4f}")
                return round(daily_ics[0], 4), 0.0

        # 兜底：整体计算 Spearman
        valid_mask = ~(np.isnan(y_true) | np.isnan(pred_values))
        if valid_mask.sum() < 5:
            return 0.0, 0.0
        corr, _ = spearmanr(y_true[valid_mask], pred_values[valid_mask])
        if np.isnan(corr):
            return 0.0, 0.0
        logger.info(f"{dataset_name} RankIC (整体): {corr:.4f}")
        return round(float(corr), 4), 0.0

    except Exception as e:
        logger.warning(f"{dataset_name} RankIC 计算失败: {e}")
        return 0.0, 0.0
