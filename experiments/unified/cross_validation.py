"""
Purged Group Time Series Split 交叉验证

参考 Marcos López de Prado《Advances in Financial Machine Learning》
实现带 purge 期和 embargo 期的时间序列交叉验证，防止信息泄漏。

核心思想：
  - 按时间顺序分割 fold，保证训练集在验证集之前
  - 在训练集和验证集之间加入 purge 期（移除训练集尾部数据）
  - 在验证集之后加入 embargo 期（移除验证集尾部数据，防止下一折泄漏）
"""
from dataclasses import dataclass
from typing import Generator, List, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from .constants import EMBARGO_DAYS


@dataclass
class CVFoldResult:
    """单折交叉验证结果"""

    fold_index: int
    train_size: int
    val_size: int
    train_date_range: Tuple[str, str]
    val_date_range: Tuple[str, str]
    metrics: dict


@dataclass
class PurgedCVConfig:
    """Purged K-Fold 配置"""

    n_splits: int = 5
    purge_days: int = EMBARGO_DAYS
    embargo_days: int = EMBARGO_DAYS


class PurgedGroupTimeSeriesSplit:
    """
    Purged Group Time Series Split

    与 sklearn TimeSeriesSplit 的区别：
      1. 按日期分组（同一天的所有股票属于同一组）
      2. 训练集尾部移除 purge_days 天（防止特征窗口泄漏）
      3. 验证集尾部移除 embargo_days 天（防止标签泄漏到下一折）

    示例（5 折，purge=20 天，embargo=20 天）：
      Fold 1: train=[d0, d200-20] | purge | val=[d200, d400-20] | embargo
      Fold 2: train=[d0, d400-20] | purge | val=[d400, d600-20] | embargo
      ...
    """

    def __init__(self, config: PurgedCVConfig):
        self.n_splits = config.n_splits
        self.purge_days = config.purge_days
        self.embargo_days = config.embargo_days

    def split(
        self, data: pd.DataFrame, date_col: str = "date"
    ) -> Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]:
        """
        生成 purged 时间序列分割

        Args:
            data: 包含日期列的 DataFrame
            date_col: 日期列名

        Yields:
            (train_df, val_df) 元组
        """
        dates = self._extract_sorted_dates(data, date_col)
        fold_boundaries = self._compute_fold_boundaries(dates)

        for fold_idx, (train_end, val_start, val_end) in enumerate(
            fold_boundaries
        ):
            train_df, val_df = self._build_fold_datasets(
                data, dates, date_col, train_end, val_start, val_end
            )

            if train_df.empty or val_df.empty:
                logger.warning(f"Fold {fold_idx}: 训练集或验证集为空，跳过")
                continue

            self._log_fold_info(fold_idx, train_df, val_df, date_col)
            yield train_df, val_df

    def _extract_sorted_dates(
        self, data: pd.DataFrame, date_col: str
    ) -> np.ndarray:
        """提取并排序唯一日期"""
        dates = np.sort(data[date_col].unique())
        logger.info(
            f"PurgedCV: {len(dates)} 个交易日, "
            f"范围 {dates[0]} ~ {dates[-1]}"
        )
        return dates

    def _compute_fold_boundaries(
        self, dates: np.ndarray
    ) -> List[Tuple[int, int, int]]:
        """
        计算每折的边界索引

        Returns:
            [(train_end_idx, val_start_idx, val_end_idx), ...]
        """
        n_dates = len(dates)
        # 验证集大小 = 总天数 / (n_splits + 1)
        # 第一折训练集占 1 份，之后每折训练集增加 1 份
        fold_size = n_dates // (self.n_splits + 1)

        if fold_size <= self.purge_days + self.embargo_days:
            raise ValueError(
                f"数据量不足: 每折 {fold_size} 天, "
                f"但 purge({self.purge_days}) + embargo({self.embargo_days}) "
                f"= {self.purge_days + self.embargo_days} 天"
            )

        boundaries = []
        for i in range(self.n_splits):
            # 训练集结束位置（含 purge 移除）
            train_end_raw = fold_size * (i + 1)
            train_end = train_end_raw - self.purge_days

            # 验证集起止位置
            val_start = train_end_raw
            val_end_raw = fold_size * (i + 2)
            val_end = min(val_end_raw - self.embargo_days, n_dates)

            if train_end <= 0 or val_start >= n_dates or val_end <= val_start:
                continue

            boundaries.append((train_end, val_start, val_end))

        logger.info(f"PurgedCV: 生成 {len(boundaries)} 个有效折")
        return boundaries

    def _build_fold_datasets(
        self,
        data: pd.DataFrame,
        dates: np.ndarray,
        date_col: str,
        train_end_idx: int,
        val_start_idx: int,
        val_end_idx: int,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """根据边界索引构建训练集和验证集"""
        train_dates = set(dates[:train_end_idx])
        val_dates = set(dates[val_start_idx:val_end_idx])

        train_mask = data[date_col].isin(train_dates)
        val_mask = data[date_col].isin(val_dates)

        return data[train_mask].copy(), data[val_mask].copy()

    def _log_fold_info(
        self,
        fold_idx: int,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        date_col: str,
    ) -> None:
        """打印折信息"""
        train_start = train_df[date_col].min()
        train_end = train_df[date_col].max()
        val_start = val_df[date_col].min()
        val_end = val_df[date_col].max()

        logger.info(
            f"Fold {fold_idx}: "
            f"训练 {len(train_df)} 条 ({train_start}~{train_end}) | "
            f"purge {self.purge_days}d | "
            f"验证 {len(val_df)} 条 ({val_start}~{val_end}) | "
            f"embargo {self.embargo_days}d"
        )


def run_purged_cv(
    data: pd.DataFrame,
    train_fn: callable,
    config: PurgedCVConfig,
    date_col: str = "date",
) -> List[CVFoldResult]:
    """
    执行 Purged K-Fold 交叉验证

    Args:
        data: 完整数据集（含特征和标签）
        train_fn: 训练函数，签名 (train_df, val_df) -> dict（返回指标）
        config: CV 配置
        date_col: 日期列名

    Returns:
        每折的验证结果列表
    """
    splitter = PurgedGroupTimeSeriesSplit(config)
    results: List[CVFoldResult] = []

    for fold_idx, (train_df, val_df) in enumerate(
        splitter.split(data, date_col)
    ):
        logger.info(f"=== Purged CV Fold {fold_idx + 1}/{config.n_splits} ===")

        metrics = train_fn(train_df, val_df)

        result = CVFoldResult(
            fold_index=fold_idx,
            train_size=len(train_df),
            val_size=len(val_df),
            train_date_range=(
                str(train_df[date_col].min()),
                str(train_df[date_col].max()),
            ),
            val_date_range=(
                str(val_df[date_col].min()),
                str(val_df[date_col].max()),
            ),
            metrics=metrics,
        )
        results.append(result)

        _log_fold_metrics(fold_idx, metrics)

    _log_cv_summary(results)
    return results


def _log_fold_metrics(fold_idx: int, metrics: dict) -> None:
    """打印单折指标"""
    parts = [f"{k}={v:.4f}" for k, v in metrics.items() if isinstance(v, float)]
    logger.info(f"Fold {fold_idx} 指标: {', '.join(parts)}")


def _log_cv_summary(results: List[CVFoldResult]) -> None:
    """打印 CV 汇总"""
    if not results:
        logger.warning("CV 无有效结果")
        return

    all_metrics = {}
    for r in results:
        for k, v in r.metrics.items():
            if isinstance(v, (int, float)):
                all_metrics.setdefault(k, []).append(v)

    logger.info("=== Purged CV 汇总 ===")
    for k, values in all_metrics.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        logger.info(f"  {k}: {mean_val:.4f} ± {std_val:.4f}")
