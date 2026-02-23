"""
缺失值处理器

提供智能的缺失值处理策略，区分不同类型的缺失值并采用相应的填充方法。
"""

from typing import Dict, List, Optional

import pandas as pd
from loguru import logger


class MissingValueHandler:
    """缺失值处理器

    根据缺失值的类型和比例采用不同的填充策略：
    - 价格数据：前向填充（处理停牌等情况）
    - 技术指标：根据缺失比例选择填充方法
    - 高缺失率列：使用中位数填充

    Attributes:
        high_missing_threshold: 高缺失率阈值
        price_columns: 价格相关列名列表
    """

    # 默认价格列
    DEFAULT_PRICE_COLUMNS = ["$open", "$high", "$low", "$close", "$volume"]

    def __init__(
        self,
        high_missing_threshold: float = 0.5,
        price_columns: Optional[List[str]] = None,
    ):
        """初始化缺失值处理器

        Args:
            high_missing_threshold: 高缺失率阈值，超过此比例使用中位数填充
            price_columns: 价格相关列名列表
        """
        self.high_missing_threshold = high_missing_threshold
        self.price_columns = price_columns or self.DEFAULT_PRICE_COLUMNS

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理 DataFrame 中的缺失值

        Args:
            df: 输入 DataFrame

        Returns:
            处理后的 DataFrame
        """
        if df.empty:
            return df

        df_filled = df.copy()

        # 确保数据按时间排序（避免未来信息泄漏）
        df_filled = self._ensure_sorted(df_filled)

        # 处理价格列
        df_filled = self._handle_price_columns(df_filled)

        # 处理技术指标列
        df_filled = self._handle_indicator_columns(df_filled)

        # 记录处理结果
        self._log_processing_result(df, df_filled)

        return df_filled

    def _ensure_sorted(self, df: pd.DataFrame) -> pd.DataFrame:
        """确保数据按时间排序

        Args:
            df: 输入 DataFrame

        Returns:
            排序后的 DataFrame
        """
        if isinstance(df.index, pd.MultiIndex):
            return df.sort_index()
        elif df.index.name in ["datetime", "date", "time"] or isinstance(
            df.index, pd.DatetimeIndex
        ):
            return df.sort_index()
        return df

    def _handle_price_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理价格列的缺失值

        使用前向填充，然后后向填���（处理开头缺失）

        Args:
            df: 输入 DataFrame

        Returns:
            处理后的 DataFrame
        """
        for col in self.price_columns:
            if col in df.columns:
                df[col] = df[col].ffill().bfill()
        return df

    def _handle_indicator_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理技术指标列的缺失值

        根据缺失比例选择填充方法：
        - 高缺失率（>50%）：使用中位数填充
        - 低缺失率：使用前向填充 + 中位数

        Args:
            df: 输入 DataFrame

        Returns:
            处理后的 DataFrame
        """
        indicator_cols = [
            col for col in df.columns if col not in self.price_columns + ["label"]
        ]

        for col in indicator_cols:
            if col not in df.columns:
                continue

            col_data = df[col]
            missing_mask = col_data.isna()

            if not missing_mask.any():
                continue

            missing_ratio = missing_mask.sum() / len(col_data)

            if missing_ratio > self.high_missing_threshold:
                df[col] = self._fill_with_median(col_data)
                logger.debug(f"列 {col} 缺失率 {missing_ratio:.2%}，使用中位数填充")
            else:
                df[col] = self._fill_with_ffill_and_median(col_data)
                logger.debug(f"列 {col} 缺失率 {missing_ratio:.2%}，使用前向填充+中位数")

        return df

    def _fill_with_median(self, series: pd.Series) -> pd.Series:
        """使用中位数填充

        Args:
            series: 输入 Series

        Returns:
            填充后的 Series
        """
        median_value = series.median()
        if pd.notna(median_value):
            return series.fillna(median_value)
        return series.fillna(0)

    def _fill_with_ffill_and_median(self, series: pd.Series) -> pd.Series:
        """使用前向填充 + 中位数填充

        Args:
            series: 输入 Series

        Returns:
            填充后的 Series
        """
        filled = series.ffill().bfill()

        if filled.isna().any():
            median_value = filled.median()
            if pd.notna(median_value):
                filled = filled.fillna(median_value)
            else:
                filled = filled.fillna(0)

        return filled

    def _log_processing_result(
        self, df_before: pd.DataFrame, df_after: pd.DataFrame
    ) -> None:
        """记录缺失值处理结果

        Args:
            df_before: 处理前的 DataFrame
            df_after: 处理后的 DataFrame
        """
        missing_before = df_before.isnull().sum()
        missing_after = df_after.isnull().sum()

        if missing_before.sum() > 0:
            before_dict = missing_before[missing_before > 0].to_dict()
            after_dict = missing_after[missing_after > 0].to_dict()
            logger.debug(f"缺失值处理完成 - 处理前: {before_dict}, 处理后: {after_dict}")

    def get_missing_stats(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """获取缺失值统计信息

        Args:
            df: 输入 DataFrame

        Returns:
            缺失值统计字典
        """
        stats = {}
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            total_count = len(df)
            stats[col] = {
                "missing_count": missing_count,
                "missing_ratio": missing_count / total_count if total_count > 0 else 0,
            }
        return stats
