"""
Purged Group Time Series Split — 防信息泄漏的时间序列交叉验证

移植自 experiments/unified/cross_validation.py，适配 Qlib 训练引擎。

参考: Marcos López de Prado《Advances in Financial Machine Learning》

核心思想:
  - 按时间顺序分割 fold，保证训练集在验证集之前
  - 训练集尾部移除 purge_days（防止特征窗口泄漏）
  - 验证集尾部移除 embargo_days（防止标签泄漏到下一折）
"""

from dataclasses import dataclass, field
from typing import Generator, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

# === 常量 ===
DEFAULT_N_SPLITS = 5
DEFAULT_PURGE_DAYS = 20
DEFAULT_EMBARGO_DAYS = 20


@dataclass
class PurgedCVConfig:
    """Purged K-Fold 配置"""

    n_splits: int = DEFAULT_N_SPLITS
    purge_days: int = DEFAULT_PURGE_DAYS
    embargo_days: int = DEFAULT_EMBARGO_DAYS


@dataclass
class CVFoldResult:
    """单折交叉验证结果"""

    fold_index: int
    train_size: int
    val_size: int
    train_date_range: Tuple[str, str]
    val_date_range: Tuple[str, str]
    metrics: dict = field(default_factory=dict)


class PurgedGroupTimeSeriesSplit:
    """
    Purged Group Time Series Split

    与 sklearn TimeSeriesSplit 的区别:
      1. 按日期分组（同一天的所有股票属于同一组）
      2. 训练集尾部移除 purge_days 天
      3. 验证集尾部移除 embargo_days 天

    支持 MultiIndex (instrument, datetime) 格式的 DataFrame。
    """

    def __init__(self, config: Optional[PurgedCVConfig] = None):
        cfg = config or PurgedCVConfig()
        self.n_splits = cfg.n_splits
        self.purge_days = cfg.purge_days
        self.embargo_days = cfg.embargo_days

    def split(
        self, data: pd.DataFrame,
    ) -> Generator[
        Tuple[pd.DataFrame, pd.DataFrame], None, None,
    ]:
        """
        生成 purged 时间序列分割

        Args:
            data: MultiIndex (instrument, datetime) 或普通 DatetimeIndex

        Yields:
            (train_df, val_df) 元组
        """
        dates = _extract_sorted_dates(data)
        boundaries = self._compute_fold_boundaries(dates)

        for fold_idx, (t_end, v_start, v_end) in enumerate(
            boundaries,
        ):
            train_df, val_df = _select_by_index_range(
                data, dates, t_end, v_start, v_end,
            )
            if train_df.empty or val_df.empty:
                logger.warning(
                    f"Fold {fold_idx}: 训练集或验证集为空，跳过",
                )
                continue

            _log_fold_info(
                fold_idx, train_df, val_df, dates,
                t_end, v_start, v_end,
                self.purge_days, self.embargo_days,
            )
            yield train_df, val_df

    def _compute_fold_boundaries(
        self, dates: np.ndarray,
    ) -> List[Tuple[int, int, int]]:
        """计算每折的边界索引"""
        n_dates = len(dates)
        fold_size = n_dates // (self.n_splits + 1)
        gap = self.purge_days + self.embargo_days

        if fold_size <= gap:
            raise ValueError(
                f"数据量不足: 每折 {fold_size} 天, "
                f"purge+embargo = {gap} 天",
            )

        boundaries: List[Tuple[int, int, int]] = []
        for i in range(self.n_splits):
            train_end = fold_size * (i + 1) - self.purge_days
            val_start = fold_size * (i + 1)
            val_end = min(
                fold_size * (i + 2) - self.embargo_days,
                n_dates,
            )
            if train_end <= 0 or val_start >= n_dates:
                continue
            if val_end <= val_start:
                continue
            boundaries.append((train_end, val_start, val_end))

        logger.info(f"PurgedCV: 生成 {len(boundaries)} 个有效折")
        return boundaries


# === 辅助函数 ===


def _extract_sorted_dates(
    data: pd.DataFrame,
) -> np.ndarray:
    """从 DataFrame 中提取排序后的唯一日期"""
    if isinstance(data.index, pd.MultiIndex):
        dates = data.index.get_level_values(1).unique()
    else:
        dates = data.index.unique()
    return np.sort(dates.values)


def _select_by_index_range(
    data: pd.DataFrame,
    dates: np.ndarray,
    train_end_idx: int,
    val_start_idx: int,
    val_end_idx: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """根据日期索引范围筛选训练集和验证集"""
    train_dates = set(dates[:train_end_idx])
    val_dates = set(dates[val_start_idx:val_end_idx])

    if isinstance(data.index, pd.MultiIndex):
        dt_level = data.index.get_level_values(1)
        train = data[dt_level.isin(train_dates)]
        val = data[dt_level.isin(val_dates)]
    else:
        train = data[data.index.isin(train_dates)]
        val = data[data.index.isin(val_dates)]

    return train, val


def _log_fold_info(
    fold_idx: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    dates: np.ndarray,
    t_end: int,
    v_start: int,
    v_end: int,
    purge_days: int,
    embargo_days: int,
) -> None:
    """打印折信息"""
    logger.info(
        f"Fold {fold_idx}: "
        f"训练 {len(train_df)} 条 ({dates[0]}~{dates[t_end - 1]}) | "
        f"purge {purge_days}d | "
        f"验证 {len(val_df)} 条 ({dates[v_start]}~{dates[v_end - 1]}) | "
        f"embargo {embargo_days}d",
    )


def select_best_fold_split(
    data: pd.DataFrame,
    config: Optional[PurgedCVConfig] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    使用 Purged CV 的最后一折作为训练/验证分割

    这是最简单的集成方式：不改变训练流程，
    只是用 purged split 替代简单的 ratio split。
    最后一折的训练集最大，验证集是最近的数据。

    Args:
        data: 完整数据集
        config: PurgedCV 配置

    Returns:
        (train_data, val_data)
    """
    splitter = PurgedGroupTimeSeriesSplit(config)
    last_train, last_val = None, None

    for train_df, val_df in splitter.split(data):
        last_train, last_val = train_df, val_df

    if last_train is None or last_val is None:
        raise ValueError(
            "PurgedCV 未能生成有效分割，数据量可能不足",
        )

    logger.info(
        f"PurgedCV 最后一折: 训练集 {len(last_train)} 条, "
        f"验证集 {len(last_val)} 条",
    )
    return last_train, last_val
