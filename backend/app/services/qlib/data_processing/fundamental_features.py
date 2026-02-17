"""
基本面特征计算器

计算股票的基本面特征，包括价格变化率、成交量变化率、波动率等。
"""

from typing import List, Optional

import pandas as pd
from loguru import logger


class FundamentalFeatureCalculator:
    """��本面特征计算器

    计算以下特征：
    - 价格变化率（1日、5日、20日）
    - 成交量变化率和均量比
    - 波动率（5日、20日）
    - 价格位置（相对于近期高低点）

    Attributes:
        periods: 计算周期列表
    """

    DEFAULT_PERIODS = [1, 5, 20]

    def __init__(self, periods: Optional[List[int]] = None):
        """初始化基本面特征计算器

        Args:
            periods: 计算周期列表
        """
        self.periods = periods or self.DEFAULT_PERIODS

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算基本面特征

        Args:
            df: 输入 DataFrame，需要包含 close, volume, high, low 列

        Returns:
            添加了基本面特征的 DataFrame
        """
        if df.empty:
            return df

        df = df.copy()

        # 获取列名（支持带 $ 前缀和不带前缀）
        close_col = self._find_column(df, ["close", "$close"])
        volume_col = self._find_column(df, ["volume", "$volume"])
        high_col = self._find_column(df, ["high", "$high"])
        low_col = self._find_column(df, ["low", "$low"])

        if close_col:
            df = self._add_price_change_features(df, close_col)
            df = self._add_volatility_features(df, close_col)

        if volume_col:
            df = self._add_volume_features(df, volume_col)

        if all([close_col, high_col, low_col]):
            df = self._add_price_position_feature(df, close_col, high_col, low_col)

        logger.debug(f"基本面特征计算完成，新增 {len(df.columns) - len(df.columns)} 列")
        return df

    def _find_column(self, df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        """查找列名

        Args:
            df: DataFrame
            candidates: 候选列名列表

        Returns:
            找到的列名，未找到返回 None
        """
        for col in candidates:
            if col in df.columns:
                return col
        return None

    def _add_price_change_features(
        self, df: pd.DataFrame, close_col: str
    ) -> pd.DataFrame:
        """添加价格变化率特征

        Args:
            df: 输入 DataFrame
            close_col: 收盘价列名

        Returns:
            添加特征后的 DataFrame
        """
        for period in self.periods:
            feature_name = f"price_change_{period}d" if period > 1 else "price_change"
            df[feature_name] = df[close_col].pct_change(periods=period)

        return df

    def _add_volume_features(self, df: pd.DataFrame, volume_col: str) -> pd.DataFrame:
        """添加成交量特征

        Args:
            df: 输入 DataFrame
            volume_col: 成交量列名

        Returns:
            添加特征后的 DataFrame
        """
        # 成交量变化率
        df["volume_change"] = df[volume_col].pct_change()

        # 成交量与20日均量的比值
        df["volume_ma_ratio"] = df[volume_col] / df[volume_col].rolling(20).mean()

        return df

    def _add_volatility_features(
        self, df: pd.DataFrame, close_col: str
    ) -> pd.DataFrame:
        """添加波动率特征

        Args:
            df: 输入 DataFrame
            close_col: 收盘价列名

        Returns:
            添加特征后的 DataFrame
        """
        # 计算日收益���
        returns = df[close_col].pct_change()

        # 5日和20日波动率
        df["volatility_5d"] = returns.rolling(5).std()
        df["volatility_20d"] = returns.rolling(20).std()

        return df

    def _add_price_position_feature(
        self,
        df: pd.DataFrame,
        close_col: str,
        high_col: str,
        low_col: str,
    ) -> pd.DataFrame:
        """添加价格位置特征

        计算当前价格在近20日高低点区间中的位置

        Args:
            df: 输入 DataFrame
            close_col: 收盘价列名
            high_col: 最高价列名
            low_col: 最低价列名

        Returns:
            添加特征后的 DataFrame
        """
        rolling_high = df[high_col].rolling(20).max()
        rolling_low = df[low_col].rolling(20).min()

        # 价格位置 = (当前价 - 最低价) / (最高价 - 最低价)
        price_range = rolling_high - rolling_low
        df["price_position"] = (df[close_col] - rolling_low) / price_range.replace(
            0, float("nan")
        )

        return df

    def get_feature_names(self) -> List[str]:
        """获取所有特征名称

        Returns:
            特征名称列表
        """
        features = ["price_change"]
        for period in self.periods:
            if period > 1:
                features.append(f"price_change_{period}d")

        features.extend(
            [
                "volume_change",
                "volume_ma_ratio",
                "volatility_5d",
                "volatility_20d",
                "price_position",
            ]
        )

        return features
