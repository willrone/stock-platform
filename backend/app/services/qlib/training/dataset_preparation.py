"""
数据集准备模块

包含数据处理、特征工程和数据集分割功能。
支持多种特征集（Alpha158 / 手工62特征 / 自定义）、
标签类型（回归 / 二分类）和数据分割方式（比例 / 硬切）。
"""

import asyncio
import multiprocessing as mp
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger

from .config import QlibTrainingConfig
from .data_preprocessing import (
    CSRankNormTransformer,
    CrossSectionalNeutralizer,
    OutlierHandler,
    RobustFeatureScaler,
)
from .qlib_check import QLIB_AVAILABLE

# 常量
DEFAULT_PREDICTION_HORIZON = 5
DEFAULT_EMBARGO_DAYS = 20
DEFAULT_BINARY_THRESHOLD = 0.003


def process_stock_data(
    stock_data: pd.DataFrame,
    stock_code: str,
    prediction_horizon: int = DEFAULT_PREDICTION_HORIZON,
) -> pd.DataFrame:
    """处理单个股票的数据，包括特征计算和标签生成"""
    try:
        processed_data = stock_data.copy()
        _compute_basic_features(processed_data)
        _compute_volume_features(processed_data)
        _generate_regression_label(
            processed_data, stock_code, prediction_horizon,
        )
        processed_data = processed_data.fillna(0)
        return processed_data
    except Exception as e:
        logger.error(f"处理股票 {stock_code} 数据时发生错误: {e}")
        return stock_data


def _compute_basic_features(data: pd.DataFrame) -> None:
    """计算基本价格特征（原地修改）"""
    if "$close" not in data.columns:
        return
    close = data["$close"]
    data["RET1"] = close.pct_change(1)
    data["RET5"] = close.pct_change(5)
    data["RET20"] = close.pct_change(20)
    data["MA5"] = close.rolling(5).mean()
    data["MA20"] = close.rolling(20).mean()
    data["STD5"] = close.rolling(5).std()
    data["STD20"] = close.rolling(20).std()


def _compute_volume_features(data: pd.DataFrame) -> None:
    """计算成交量特征（原地修改）"""
    if "$volume" not in data.columns:
        return
    volume = data["$volume"]
    data["VOL1"] = volume.pct_change(1)
    data["VOL5"] = volume.pct_change(5)


def _generate_regression_label(
    data: pd.DataFrame,
    stock_code: str,
    prediction_horizon: int,
) -> None:
    """生成回归标签：未来N天收益率（原地修改）"""
    if "$close" not in data.columns:
        return

    current_price = data["$close"]
    if isinstance(data.index, pd.MultiIndex):
        future_price = data.groupby(level=0)["$close"].shift(
            -prediction_horizon,
        )
    else:
        future_price = data["$close"].shift(-prediction_horizon)

    label_values = (future_price - current_price) / current_price
    data["label"] = (
        label_values.iloc[:, 0].values
        if hasattr(label_values, "iloc") and label_values.ndim > 1
        else label_values
    )
    data["label"] = data["label"].fillna(0)

    logger.debug(
        f"股票 {stock_code} 标签创建完成，预测周期={prediction_horizon}天，"
        f"标签范围=[{data['label'].min():.6f}, {data['label'].max():.6f}]",
    )


def _apply_binary_label(
    data: pd.DataFrame, threshold: float,
) -> None:
    """将回归标签转换为二分类标签（原地修改）

    Args:
        data: 包含 label 列的 DataFrame
        threshold: 二分类阈值，收益率 > threshold → 1，否则 → 0
    """
    if "label" not in data.columns:
        return
    original_mean = data["label"].mean()
    data["label"] = (data["label"] > threshold).astype(int)
    positive_ratio = data["label"].mean()
    logger.info(
        f"二分类标签转换完成: 阈值={threshold}, "
        f"原始均值={original_mean:.6f}, 正样本比例={positive_ratio:.4f}",
    )


def _compute_technical_62_features(
    dataset: pd.DataFrame,
) -> pd.DataFrame:
    """使用手工62特征替代 Alpha158

    Args:
        dataset: 原始 OHLCV 数据

    Returns:
        添加了62个技术指标和截面特征的 DataFrame
    """
    from .technical_feature_computer import (
        compute_cross_sectional_features,
        compute_stock_technical_features,
    )

    if isinstance(dataset.index, pd.MultiIndex):
        stock_codes = dataset.index.get_level_values(0).unique()
        processed_parts: List[pd.DataFrame] = []
        for code in stock_codes:
            stock_df = dataset.xs(code, level=0, drop_level=False)
            stock_df = compute_stock_technical_features(stock_df)
            processed_parts.append(stock_df)
        result = pd.concat(processed_parts)
        result = compute_cross_sectional_features(result)
    else:
        result = compute_stock_technical_features(dataset)

    logger.info(
        f"手工62特征计算完成，特征数: {len(result.columns)}",
    )
    return result


def _filter_custom_features(
    dataset: pd.DataFrame,
    selected_features: List[str],
) -> pd.DataFrame:
    """保留自定义特征列 + label 列

    Args:
        dataset: 完整数据集
        selected_features: 用户指定的特征列名

    Returns:
        仅包含指定特征和 label 的 DataFrame
    """
    available = set(dataset.columns)
    keep_cols = [c for c in selected_features if c in available]
    missing = [c for c in selected_features if c not in available]
    if missing:
        logger.warning(f"自定义特征中 {len(missing)} 个不存在: {missing[:10]}")
    if not keep_cols:
        logger.warning("自定义特征全部不存在，使用所有可用特征")
        return dataset
    if "label" in available:
        keep_cols.append("label")
    logger.info(f"使用 {len(keep_cols)} 个自定义特征")
    return dataset[keep_cols]


def _split_by_ratio(
    dataset: pd.DataFrame,
    validation_split: float,
    embargo_days: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """按比例时间分割数据集

    Returns:
        (train_data, val_data)
    """
    dates = _extract_sorted_dates(dataset)
    split_idx = int(len(dates) * (1 - validation_split))
    train_dates = dates[:split_idx]
    val_dates = dates[split_idx:]

    if len(train_dates) > embargo_days:
        train_dates = train_dates[:-embargo_days]
        logger.info(
            f"Embargo 期: 移除训练集末尾 {embargo_days} 天",
        )

    return _select_by_dates(dataset, train_dates, val_dates)


def _split_by_hardcut(
    dataset: pd.DataFrame,
    train_end_date: str,
    val_end_date: Optional[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """按固定日期硬切分割数据集

    Args:
        dataset: 完整数据集
        train_end_date: 训练集截止日期（含）
        val_end_date: 验证集截止日期（含），None 则取全部剩余

    Returns:
        (train_data, val_data)
    """
    dates = _extract_sorted_dates(dataset)
    train_end = pd.Timestamp(train_end_date)
    train_dates = dates[dates <= train_end]

    if val_end_date:
        val_end = pd.Timestamp(val_end_date)
        val_dates = dates[(dates > train_end) & (dates <= val_end)]
    else:
        val_dates = dates[dates > train_end]

    logger.info(
        f"硬切分割: 训练集 {len(train_dates)} 天 "
        f"(≤{train_end_date}), 验证集 {len(val_dates)} 天",
    )
    return _select_by_dates(dataset, train_dates, val_dates)


def _split_by_purged_cv(
    dataset: pd.DataFrame,
    config: QlibTrainingConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """使用 Purged K-Fold 的最后一折作为训练/验证分割

    最后一折训练集最大、验证集是最近的数据，
    同时通过 purge + embargo 防止信息泄漏。

    Args:
        dataset: 完整数据集
        config: 训练配置（含 purged_cv_splits 等）

    Returns:
        (train_data, val_data)
    """
    from .purged_cv import PurgedCVConfig, select_best_fold_split

    cv_config = PurgedCVConfig(
        n_splits=config.purged_cv_splits,
        purge_days=config.purged_cv_purge_days,
        embargo_days=config.embargo_days,
    )
    return select_best_fold_split(dataset, cv_config)


def _extract_sorted_dates(
    dataset: pd.DataFrame,
) -> pd.DatetimeIndex:
    """从数据集中提取排序后的唯一日期"""
    if isinstance(dataset.index, pd.MultiIndex):
        return dataset.index.get_level_values(1).unique().sort_values()
    return dataset.index.unique().sort_values()


def _select_by_dates(
    dataset: pd.DataFrame,
    train_dates: Any,
    val_dates: Any,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """根据日期集合筛选训练集和验证集"""
    if isinstance(dataset.index, pd.MultiIndex):
        train = dataset[
            dataset.index.get_level_values(1).isin(train_dates)
        ]
        val = dataset[
            dataset.index.get_level_values(1).isin(val_dates)
        ]
    else:
        train = dataset[dataset.index.isin(train_dates)]
        val = dataset[dataset.index.isin(val_dates)]
    return train, val


def _postprocess_data(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    config: QlibTrainingConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """异常值处理 + CSRankNorm（可选） + 中性化 + 标准化

    当 enable_cs_rank_norm=True 时：
      1. 先 winsorize 原始标签（去极端值）
      2. 再做 CSRankNorm（rank→percentile→ppf）
      3. 跳过 CSRankNorm 后的 winsorize（输出已有界）
    当 enable_cs_rank_norm=False 时：
      保持原有 winsorize 行为。

    Returns:
        (processed_train, processed_val)
    """
    enable_csrn = config.enable_cs_rank_norm if config else False

    # === 异常值处理（winsorize） ===
    outlier_handler = OutlierHandler(
        method="winsorize",
        lower_percentile=0.01,
        upper_percentile=0.99,
    )
    if "label" in train_data.columns:
        logger.info("开始处理标签异常值")
        train_data = outlier_handler.handle_label_outliers(
            train_data, label_col="label",
        )
        if val_data is not None and "label" in val_data.columns:
            val_data = outlier_handler.handle_label_outliers(
                val_data, label_col="label",
            )

    # === CSRankNorm（可选） ===
    if enable_csrn and "label" in train_data.columns:
        logger.info("启用 CSRankNorm 标签变换")
        csrn = CSRankNormTransformer()
        train_data = csrn.transform(train_data, label_col="label")
        if val_data is not None and "label" in val_data.columns:
            val_data = csrn.transform(val_data, label_col="label")

    feature_cols = [c for c in train_data.columns if c != "label"]

    # 中性化
    enable_neutralization = (
        config.enable_neutralization if config else True
    )
    if enable_neutralization and feature_cols:
        neutralizer = CrossSectionalNeutralizer()
        logger.info("开始截面中性化处理")
        train_data = neutralizer.transform(train_data, feature_cols)
        if val_data is not None and len(val_data) > 0:
            val_data = neutralizer.transform(val_data, feature_cols)

    # 标准化
    if feature_cols:
        scaler = RobustFeatureScaler()
        logger.info(f"开始特征标准化，特征列数: {len(feature_cols)}")
        train_data = scaler.fit_transform(train_data, feature_cols)
        if val_data is not None and len(val_data) > 0:
            val_data = scaler.transform(val_data, feature_cols)

    return train_data, val_data


def _process_multiindex_stocks(
    dataset: pd.DataFrame, config: QlibTrainingConfig,
) -> pd.DataFrame:
    """按股票分组处理 MultiIndex 数据"""
    stock_codes = dataset.index.get_level_values(0).unique()
    prediction_horizon = (
        config.prediction_horizon if config else DEFAULT_PREDICTION_HORIZON
    )

    processed_stocks: List[pd.DataFrame] = []
    logger.info("使用单进程处理数据")
    for stock_code in stock_codes:
        try:
            stock_data = dataset.xs(
                stock_code, level=0, drop_level=False,
            )
            if stock_data.empty:
                continue
            processed = process_stock_data(
                stock_data, stock_code, prediction_horizon,
            )
            if not processed.empty:
                processed_stocks.append(processed)
        except KeyError:
            logger.warning(f"股票 {stock_code} 不在数据中")

    if processed_stocks:
        result = pd.concat(processed_stocks)
        logger.info(f"数据处理完成，合并后形状: {result.shape}")
        return result

    logger.warning("没有处理任何股票数据")
    return dataset


async def prepare_training_datasets(
    dataset: pd.DataFrame,
    validation_split: float,
    config: QlibTrainingConfig = None,
) -> Tuple[Any, Any]:
    """准备训练和验证数据集，返回 qlib DatasetH 对象

    根据 config 中的 feature_set / label_type / split_method
    选择不同的特征计算、标签生成和数据分割方式。
    """
    if not QLIB_AVAILABLE:
        raise RuntimeError(
            "Qlib不可用，无法准备数据集。\n"
            "请安装Qlib库：pip install git+https://github.com/microsoft/qlib.git",
        )

    feature_set = config.feature_set if config else "alpha158"
    label_type = config.label_type if config else "regression"
    split_method = config.split_method if config else "ratio"

    logger.info(
        f"数据集准备: feature_set={feature_set}, "
        f"label_type={label_type}, split_method={split_method}",
    )

    # === 1. 特征计算 ===
    if feature_set == "technical_62":
        dataset = _compute_technical_62_features(dataset)
    elif feature_set == "custom" and config and config.selected_features:
        dataset = _filter_custom_features(
            dataset, config.selected_features,
        )

    # === 2. 按股票处理（alpha158 和默认路径） ===
    is_multi = (
        isinstance(dataset.index, pd.MultiIndex)
        and dataset.index.nlevels == 2
    )
    if is_multi and feature_set == "alpha158":
        dataset = _process_multiindex_stocks(dataset, config)

    # === 3. 标签生成（technical_62 需要单独生成标签） ===
    if feature_set == "technical_62" and "label" not in dataset.columns:
        _generate_labels_for_dataset(dataset, config)

    # === 4. 二分类标签转换 ===
    if label_type == "binary" and "label" in dataset.columns:
        threshold = (
            config.binary_threshold
            if config
            else DEFAULT_BINARY_THRESHOLD
        )
        _apply_binary_label(dataset, threshold)

    # === 5. 数据分割 ===
    embargo_days = config.embargo_days if config else DEFAULT_EMBARGO_DAYS
    if split_method == "hardcut" and config and config.train_end_date:
        train_data, val_data = _split_by_hardcut(
            dataset, config.train_end_date, config.val_end_date,
        )
    elif split_method == "purged_cv":
        train_data, val_data = _split_by_purged_cv(
            dataset, config,
        )
    else:
        train_data, val_data = _split_by_ratio(
            dataset, validation_split, embargo_days,
        )

    # === 6. 后处理（异常值 + 中性化 + 标准化） ===
    train_data, val_data = _postprocess_data(
        train_data, val_data, config,
    )

    # === 7. 创建 DatasetH 适配器 ===
    from .dataset_adapter import DataFrameDatasetAdapter

    prediction_horizon = (
        config.prediction_horizon if config else DEFAULT_PREDICTION_HORIZON
    )
    train_dataset = DataFrameDatasetAdapter(
        train_data, val_data, prediction_horizon, config,
    )
    val_dataset = train_dataset

    return train_dataset, val_dataset


def _generate_labels_for_dataset(
    dataset: pd.DataFrame, config: QlibTrainingConfig,
) -> None:
    """为整个数据集生成回归标签（原地修改）"""
    close_col = None
    for col in ["$close", "close", "Close"]:
        if col in dataset.columns:
            close_col = col
            break
    if close_col is None:
        logger.warning("未找到收盘价列，无法生成标签")
        return

    horizon = (
        config.prediction_horizon if config else DEFAULT_PREDICTION_HORIZON
    )
    current = dataset[close_col]
    if isinstance(dataset.index, pd.MultiIndex):
        future = dataset.groupby(level=0)[close_col].shift(-horizon)
    else:
        future = dataset[close_col].shift(-horizon)

    dataset["label"] = ((future - current) / current).fillna(0)
    logger.info(
        f"数据集标签生成完成，预测周期={horizon}天，"
        f"范围=[{dataset['label'].min():.6f}, {dataset['label'].max():.6f}]",
    )
