"""
Parquet文件管理系统
实现按股票代码和时间范围的目录结构组织，以及Parquet文件的读写和索引
"""

import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger

from app.models.file_management import (
    ComprehensiveStats,
    DeletionResult,
    DetailedFileInfo,
    FileFilters,
    FilterCriteria,
    IntegrityStatus,
    ValidationResult,
)
from app.models.stock_simple import StockData


def _read_parquet_with_fallback(file_path):
    """
    读取 parquet 文件，如果 pyarrow 失败则尝试 fastparquet
    用于处理包含 arrow.py_extension_type 的旧版本 parquet 文件
    """
    try:
        return pd.read_parquet(file_path, engine="pyarrow")
    except (IOError, OSError) as e:
        logger.debug(
            f"使用 pyarrow 引擎读取失败（IO 错误）：{e}，尝试使用 fastparquet",
            extra={
                "error_type": "INFRA",
                "error_code": "PARQUET_READ_IO_ERROR",
                "file_path": str(file_path),
            }
        )
        try:
            return pd.read_parquet(file_path, engine="fastparquet")
        except (IOError, OSError, Exception) as e2:
            logger.error(
                f"使用 fastparquet 引擎也失败：{e2}, file_path: {file_path}",
                extra={
                    "error_type": "INFRA",
                    "error_code": "PARQUET_READ_FAILED",
                    "file_path": str(file_path),
                }
            )
            from app.core.errors import InfraError
            raise InfraError(
                message=f"读取 Parquet 文件失败：{file_path}",
                source="parquet_engine",
                details={
                    "pyarrow_error": str(e),
                    "fastparquet_error": str(e2),
                    "file_path": str(file_path),
                },
            ) from e2