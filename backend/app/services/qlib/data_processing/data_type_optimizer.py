"""
数据类型优化器

优化 DataFrame 的数据类型以节省内存并提高计算效率。
"""

from typing import Dict, List, Optional

import pandas as pd
from loguru import logger


class DataTypeOptimizer:
    """数据类型优化器

    将 DataFrame 的列转换为更高效的数据类型。

    Attributes:
        price_columns: 价格相关列名列表
        volume_column: 成交量列名
    """

    # 默认价格列
    DEFAULT_PRICE_COLUMNS = ["$open", "$high", "$low", "$close"]

    # 默认成交量列
    DEFAULT_VOLUME_COLUMN = "$volume"

    def __init__(
        self,
        price_columns: Optional[List[str]] = None,
        volume_column: Optional[str] = None,
    ):
        """初始化数据类型优化器

        Args:
            price_columns: 价格相关列名列表
            volume_column: 成交量列名
        """
        self.price_columns = price_columns or self.DEFAULT_PRICE_COLUMNS
        self.volume_column = volume_column or self.DEFAULT_VOLUME_COLUMN

    def optimize(self, df: pd.DataFrame) -> pd.DataFrame:
        """优化 DataFrame 的数据类型

        Args:
            df: 输入 DataFrame

        Returns:
            优化后的 DataFrame
        """
        if df.empty:
            return df

        df_optimized = df.copy()

        # 优化价格列（float32）
        df_optimized = self._optimize_price_columns(df_optimized)

        # 优化成交量列（int64）
        df_optimized = self._optimize_volume_column(df_optimized)

        # 优化其他数值列（float32）
        df_optimized = self._optimize_indicator_columns(df_optimized)

        logger.debug("数据类型优化完成")
        return df_optimized

    def _optimize_price_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """优化价格列为 float32

        Args:
            df: 输入 DataFrame

        Returns:
            优化后的 DataFrame
        """
        for col in self.price_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
        return df

    def _optimize_volume_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """优化成交量列为 int64

        Args:
            df: 输入 DataFrame

        Returns:
            优化后的 DataFrame
        """
        if self.volume_column in df.columns:
            df[self.volume_column] = pd.to_numeric(
                df[self.volume_column], errors="coerce"
            ).astype("int64")
        return df

    def _optimize_indicator_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """优化技术指标列为 float32

        Args:
            df: 输入 DataFrame

        Returns:
            优化后的 DataFrame
        """
        exclude_cols = set(self.price_columns + [self.volume_column])
        indicator_cols = [col for col in df.columns if col not in exclude_cols]

        for col in indicator_cols:
            if df[col].dtype in ["float64", "object"]:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

        return df

    def get_memory_usage(self, df: pd.DataFrame) -> Dict[str, float]:
        """获取 DataFrame 的内存使用情况

        Args:
            df: 输入 DataFrame

        Returns:
            内存使用统计字典
        """
        memory_usage = df.memory_usage(deep=True)
        total_mb = memory_usage.sum() / (1024 * 1024)

        return {
            "total_mb": total_mb,
            "per_column_mb": {
                col: memory_usage[col] / (1024 * 1024) for col in df.columns
            },
            "index_mb": memory_usage.get("Index", 0) / (1024 * 1024),
        }
