"""
标签变换模块

CSRankNorm：截面排名标准化（Cross-Sectional Rank Normalization）
参考 Qlib CSRankNorm 处理器
"""
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import norm

# 截面最少有效股票数，低于此值跳过该日
MIN_CROSS_SECTION_SIZE = 10

# rank percentile 的裁剪边界，避免 ppf(0) 和 ppf(1) 产生 ±inf
RANK_CLIP_LOWER = 0.001
RANK_CLIP_UPPER = 0.999


def cs_rank_norm(
    df: pd.DataFrame,
    label_col: str = "future_return",
    date_col: str = "date",
    output_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    对标签做截面排名标准化（CSRankNorm）

    每个交易日截面内：
    1. 对 label_col 做 rank（NaN 自动跳过）
    2. 转为百分位 (0, 1)
    3. 通过 inverse normal CDF 映射到 N(0,1)

    Args:
        df: 包含 date_col 和 label_col 的 DataFrame
        label_col: 原始标签列名
        date_col: 日期列名
        output_col: 输出列名，默认覆盖 label_col

    Returns:
        添加/覆盖了标准化标签的 DataFrame
    """
    if output_col is None:
        output_col = label_col

    df = df.copy()
    df[output_col] = df.groupby(date_col)[label_col].transform(
        _rank_norm_single_day,
    )

    valid_count = df[output_col].notna().sum()
    total_count = len(df)
    logger.info(
        f"CSRankNorm 完成: {valid_count}/{total_count} 有效, "
        f"均值={df[output_col].mean():.4f}, "
        f"标准差={df[output_col].std():.4f}",
    )
    return df


def _rank_norm_single_day(series: pd.Series) -> pd.Series:
    """单日截面的 rank → percentile → inverse normal"""
    valid = series.dropna()
    if len(valid) < MIN_CROSS_SECTION_SIZE:
        return pd.Series(np.nan, index=series.index)

    ranked = series.rank(method="average", na_option="keep")
    n_valid = valid.count()
    percentile = (ranked - 0.5) / n_valid
    percentile = percentile.clip(RANK_CLIP_LOWER, RANK_CLIP_UPPER)

    return pd.Series(norm.ppf(percentile), index=series.index)
