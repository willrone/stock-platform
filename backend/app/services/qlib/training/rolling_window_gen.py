"""
滚动窗口生成器

负责生成滚动训练的时间窗口切片。
支持 Sliding Window（固定窗口）和 Expanding Window（扩展窗口）。
"""

from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd
from loguru import logger

from .rolling_config import (
    MIN_TRAIN_SAMPLES,
    RollingTrainingConfig,
    RollingWindowType,
)


@dataclass
class RollingWindow:
    """单个滚动窗口"""

    window_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    valid_start: pd.Timestamp
    valid_end: pd.Timestamp

    def __repr__(self) -> str:
        return (
            f"Window({self.window_id}: "
            f"train={self.train_start.date()}~{self.train_end.date()}, "
            f"valid={self.valid_start.date()}~{self.valid_end.date()})"
        )


def generate_rolling_windows(
    dates: pd.DatetimeIndex,
    config: RollingTrainingConfig,
) -> List[RollingWindow]:
    """生成滚动训练窗口列表

    Args:
        dates: 排序后的唯一交易日期
        config: 滚动训练配置

    Returns:
        滚动窗口列表
    """
    sorted_dates = dates.sort_values()
    total_days = len(sorted_dates)
    window_type = RollingWindowType(config.window_type)

    min_required = config.train_window + config.valid_window
    if total_days < min_required:
        logger.warning(
            f"数据天数({total_days})不足最小要求({min_required})，"
            f"无法生成滚动窗口",
        )
        return []

    windows: List[RollingWindow] = []
    window_id = 0

    if window_type == RollingWindowType.SLIDING:
        windows = _generate_sliding_windows(
            sorted_dates, config, window_id,
        )
    else:
        windows = _generate_expanding_windows(
            sorted_dates, config, window_id,
        )

    logger.info(
        f"生成 {len(windows)} 个滚动窗口 "
        f"(类型={config.window_type}, "
        f"步长={config.rolling_step}, "
        f"训练窗口={config.train_window}天)",
    )
    return windows


def _generate_sliding_windows(
    dates: pd.DatetimeIndex,
    config: RollingTrainingConfig,
    start_id: int,
) -> List[RollingWindow]:
    """生成固定窗口滑动的窗口列表"""
    windows: List[RollingWindow] = []
    total = len(dates)
    window_id = start_id
    cursor = 0

    while True:
        train_start_idx = cursor
        train_end_idx = cursor + config.train_window - 1
        valid_start_idx = train_end_idx + 1
        valid_end_idx = valid_start_idx + config.valid_window - 1

        if valid_end_idx >= total:
            break

        windows.append(RollingWindow(
            window_id=window_id,
            train_start=dates[train_start_idx],
            train_end=dates[train_end_idx],
            valid_start=dates[valid_start_idx],
            valid_end=dates[valid_end_idx],
        ))
        window_id += 1
        cursor += config.rolling_step

    return windows


def _generate_expanding_windows(
    dates: pd.DatetimeIndex,
    config: RollingTrainingConfig,
    start_id: int,
) -> List[RollingWindow]:
    """生成扩展窗口的窗口列表"""
    windows: List[RollingWindow] = []
    total = len(dates)
    window_id = start_id

    # 第一个窗口从 train_window 开始
    cursor = config.train_window

    while True:
        train_start_idx = 0  # 扩展窗口始终从头开始
        train_end_idx = cursor - 1
        valid_start_idx = cursor
        valid_end_idx = valid_start_idx + config.valid_window - 1

        if valid_end_idx >= total:
            break

        actual_train_days = train_end_idx - train_start_idx + 1
        if actual_train_days < MIN_TRAIN_SAMPLES:
            cursor += config.rolling_step
            continue

        windows.append(RollingWindow(
            window_id=window_id,
            train_start=dates[train_start_idx],
            train_end=dates[train_end_idx],
            valid_start=dates[valid_start_idx],
            valid_end=dates[valid_end_idx],
        ))
        window_id += 1
        cursor += config.rolling_step

    return windows
