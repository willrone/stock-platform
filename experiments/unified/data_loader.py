"""
数据加载与分割模块

包含数据加载、Embargo 期分割、标签 Winsorize
"""
from typing import Tuple

import numpy as np
import pandas as pd
from loguru import logger

from .config import TrainingConfig
from .constants import (
    LABEL_WINSORIZE_LOWER,
    LABEL_WINSORIZE_UPPER,
    MIN_TRADING_DAYS,
)
from .features import (
    compute_all_features,
    compute_cross_sectional_features,
    compute_regression_label,
    neutralize_features,
)


def load_stock_files(config: TrainingConfig) -> pd.DataFrame:
    """
    加载并筛选股票数据

    按成交量排序取 Top N 只股票，合并为单一 DataFrame
    """
    stock_files = list(config.data_dir.glob("*.parquet"))
    logger.info(f"发现 {len(stock_files)} 个股票文件")

    valid_stocks = _filter_valid_stocks(stock_files, config)
    logger.info(f"筛选出 {len(valid_stocks)} 只有效股票")

    selected = valid_stocks[: config.n_stocks]
    logger.info(f"选择 Top {len(selected)} 只股票（按成交量排序）")

    all_data = _load_and_compute_features(selected, config)
    return all_data


def _filter_valid_stocks(
    stock_files: list, config: TrainingConfig
) -> list:
    """筛选交易日 ≥ 阈值的股票，按成交量排序"""
    valid = []
    for f in stock_files:
        try:
            df = pd.read_parquet(f)
            df = df[
                (df["date"] >= config.start_date)
                & (df["date"] < config.end_date)
            ]
            if len(df) >= MIN_TRADING_DAYS:
                valid.append((f, df["volume"].mean()))
        except Exception:
            continue

    valid.sort(key=lambda x: x[1], reverse=True)
    return valid


def _load_and_compute_features(
    selected: list, config: TrainingConfig
) -> pd.DataFrame:
    """加载选中股票并计算特征"""
    all_data = []
    for f, _ in selected:
        df = pd.read_parquet(f)
        df = df[
            (df["date"] >= config.start_date)
            & (df["date"] < config.end_date)
        ]
        df = compute_all_features(df)
        df = compute_regression_label(df)
        all_data.append(df)

    data = pd.concat(all_data, ignore_index=True)
    logger.info(f"总数据量: {len(data)} 条")
    return data


def prepare_dataset(config: TrainingConfig) -> pd.DataFrame:
    """
    完整的数据准备流程

    加载 → 特征计算 → 截面特征 → 中性化 → 清洗
    """
    data = load_stock_files(config)

    logger.info("计算截面特征...")
    data = compute_cross_sectional_features(data)

    if config.enable_neutralization:
        logger.info("执行市场中性化处理...")
        data = neutralize_features(data)

    data = data.dropna(subset=["future_return"])
    data = data.dropna()
    logger.info(f"清洗后数据量: {len(data)} 条")

    return data


def winsorize_labels(df: pd.DataFrame) -> pd.DataFrame:
    """对回归标签做 Winsorize 处理，截断极端值"""
    label = df["future_return"]
    lower = label.quantile(LABEL_WINSORIZE_LOWER)
    upper = label.quantile(LABEL_WINSORIZE_UPPER)

    outliers_count = ((label < lower) | (label > upper)).sum()
    if outliers_count > 0:
        logger.info(
            f"标签 Winsorize: [{lower:.6f}, {upper:.6f}], "
            f"截断 {outliers_count} 个极端值"
        )

    df["future_return"] = label.clip(lower=lower, upper=upper)
    return df


def split_with_embargo(
    data: pd.DataFrame, config: TrainingConfig
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    带 Embargo 期的时间分割

    训练集 | embargo 缓冲 | 验证集 | embargo 缓冲 | 测试集
    防止使用了 20 日均线等特征时的信息泄漏

    Returns:
        (train, val, test) 三个 DataFrame
    """
    train_end = pd.Timestamp(config.train_end)
    val_end = pd.Timestamp(config.val_end)
    embargo = pd.Timedelta(days=config.embargo_days)

    # 训练集：date < train_end
    train = data[data["date"] < train_end]

    # 验证集：date >= train_end + embargo 且 date < val_end
    val_start = train_end + embargo
    val = data[(data["date"] >= val_start) & (data["date"] < val_end)]

    # 测试集：date >= val_end + embargo
    test_start = val_end + embargo
    test = data[data["date"] >= test_start]

    # Winsorize 训练集标签
    train = winsorize_labels(train)

    _log_split_info(train, val, test, config.embargo_days)
    return train, val, test


def _log_split_info(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    embargo_days: int,
) -> None:
    """打印数据分割信息"""
    for name, df in [("训练集", train), ("验证集", val), ("测试集", test)]:
        if len(df) > 0:
            date_min = df["date"].min().date()
            date_max = df["date"].max().date()
            logger.info(f"{name}: {len(df)} 条 ({date_min} ~ {date_max})")
        else:
            logger.warning(f"{name}: 空")
    logger.info(f"Embargo 缓冲期: {embargo_days} 天")
