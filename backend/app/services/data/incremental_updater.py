"""
增量更新机制
检测数据变化，只计算新增或修改的部分
"""

from __future__ import annotations  # 延迟评估类型注解

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
from loguru import logger

from app.core.config import settings
from app.services.data.qlib_format_converter import QlibFormatConverter
from app.services.data.stock_data_loader import StockDataLoader


class IncrementalUpdater:
    """增量更新器"""

    def __init__(self):
        self.data_loader = StockDataLoader()
        self.format_converter = QlibFormatConverter()

        # 数据路径
        self.parquet_data_path = (
            Path(settings.DATA_ROOT_PATH) / "parquet" / "stock_data"
        )
        self.qlib_data_path = Path(settings.QLIB_DATA_PATH) / "features" / "day"
        self.qlib_data_path.mkdir(parents=True, exist_ok=True)

    def detect_changes(
        self, stock_codes: Optional[List[str]] = None
    ) -> "Dict[str, Dict[str, Any]]":
        """
        检测数据变化

        Args:
            stock_codes: 股票代码列表（None则检测所有股票）

        Returns:
            变化信息字典 {stock_code: {'action': 'new'|'update'|'none', 'parquet_mtime': ..., 'qlib_mtime': ..., 'date_range': ...}}
        """
        try:
            changes = {}

            # 获取所有股票代码
            if stock_codes is None:
                stock_codes = self._get_all_stock_codes()

            for stock_code in stock_codes:
                change_info = self._detect_stock_changes(stock_code)
                if change_info["action"] != "none":
                    changes[stock_code] = change_info

            logger.info(f"检测到 {len(changes)} 只股票有变化")
            return changes

        except Exception as e:
            logger.error(f"检测数据变化失败: {e}")
            return {}

    def _get_all_stock_codes(self) -> List[str]:
        """获取所有股票代码"""
        try:
            stock_codes = []
            for file_path in self.parquet_data_path.glob("*.parquet"):
                file_name = file_path.stem
                if "_" in file_name:
                    parts = file_name.split("_")
                    if len(parts) >= 2:
                        code = parts[0]
                        market = parts[1].upper()
                        if market in ["SZ", "SH", "BJ"]:
                            stock_code = f"{code}.{market}"
                            stock_codes.append(stock_code)
            return sorted(set(stock_codes))
        except Exception as e:
            logger.error(f"获取股票列表失败: {e}")
            return []

    def _detect_stock_changes(self, stock_code: str) -> "Dict[str, Any]":
        """
        检测单只股票的数据变化

        Returns:
            变化信息字典
        """
        try:
            safe_code = stock_code.replace(".", "_")
            parquet_file = self.parquet_data_path / f"{safe_code}.parquet"
            qlib_file = self.qlib_data_path / f"{safe_code}.parquet"

            # 检查Parquet文件是否存在
            if not parquet_file.exists():
                return {
                    "action": "none",
                    "parquet_mtime": None,
                    "qlib_mtime": None,
                    "date_range": None,
                }

            parquet_mtime = datetime.fromtimestamp(parquet_file.stat().st_mtime)

            # 检查Qlib文件是否存在
            if not qlib_file.exists():
                # 新股票，需要计算
                parquet_data = self.data_loader.load_stock_data(stock_code)
                date_range = self._get_date_range(parquet_data)
                return {
                    "action": "new",
                    "parquet_mtime": parquet_mtime,
                    "qlib_mtime": None,
                    "date_range": date_range,
                }

            qlib_mtime = datetime.fromtimestamp(qlib_file.stat().st_mtime)

            # 比较修改时间
            if parquet_mtime > qlib_mtime:
                # Parquet文件更新了，需要重新计算
                parquet_data = self.data_loader.load_stock_data(stock_code)
                parquet_date_range = self._get_date_range(parquet_data)

                # 检查日期范围是否扩展
                qlib_data = self.format_converter.load_qlib_data(qlib_file, stock_code)
                qlib_date_range = self._get_date_range_from_qlib(qlib_data)

                if parquet_date_range and qlib_date_range:
                    # 检查是否有新日期
                    if (
                        parquet_date_range[1] > qlib_date_range[1]
                        or parquet_date_range[0] < qlib_date_range[0]
                    ):
                        return {
                            "action": "update",
                            "parquet_mtime": parquet_mtime,
                            "qlib_mtime": qlib_mtime,
                            "date_range": parquet_date_range,
                            "new_dates": self._get_new_dates(
                                parquet_date_range, qlib_date_range
                            ),
                        }
                    else:
                        # 数据被修改但日期范围没变，需要重新计算
                        return {
                            "action": "update",
                            "parquet_mtime": parquet_mtime,
                            "qlib_mtime": qlib_mtime,
                            "date_range": parquet_date_range,
                        }
                else:
                    return {
                        "action": "update",
                        "parquet_mtime": parquet_mtime,
                        "qlib_mtime": qlib_mtime,
                        "date_range": parquet_date_range,
                    }

            return {
                "action": "none",
                "parquet_mtime": parquet_mtime,
                "qlib_mtime": qlib_mtime,
                "date_range": None,
            }

        except Exception as e:
            logger.warning(f"检测股票 {stock_code} 变化失败: {e}")
            return {
                "action": "none",
                "parquet_mtime": None,
                "qlib_mtime": None,
                "date_range": None,
            }

    def _get_date_range(
        self, data: pd.DataFrame
    ) -> Optional[Tuple[datetime, datetime]]:
        """获取数据的日期范围"""
        if data.empty:
            return None

        dates = data.index
        return (dates.min().to_pydatetime(), dates.max().to_pydatetime())

    def _get_date_range_from_qlib(
        self, qlib_data: pd.DataFrame
    ) -> Optional[Tuple[datetime, datetime]]:
        """从Qlib数据获取日期范围"""
        if qlib_data.empty:
            return None

        if isinstance(qlib_data.index, pd.MultiIndex):
            dates = qlib_data.index.get_level_values(1)
        else:
            dates = qlib_data.index

        return (dates.min().to_pydatetime(), dates.max().to_pydatetime())

    def _get_new_dates(
        self,
        parquet_range: Tuple[datetime, datetime],
        qlib_range: Tuple[datetime, datetime],
    ) -> List[datetime]:
        """获取新增的日期"""
        new_dates = []

        # 检查是否有新日期在qlib_range之后
        if parquet_range[1] > qlib_range[1]:
            # 生成新日期列表
            current_date = qlib_range[1] + pd.Timedelta(days=1)
            while current_date <= parquet_range[1]:
                # 只包含交易日（排除周末）
                if current_date.weekday() < 5:
                    new_dates.append(current_date)
                current_date += pd.Timedelta(days=1)

        # 检查是否有新日期在qlib_range之前（历史数据补充）
        if parquet_range[0] < qlib_range[0]:
            current_date = parquet_range[0]
            while current_date < qlib_range[0]:
                if current_date.weekday() < 5:
                    new_dates.append(current_date)
                current_date += pd.Timedelta(days=1)

        return sorted(new_dates)

    def get_stocks_to_update(
        self, stock_codes: Optional[List[str]] = None, force_update: bool = False
    ) -> List[str]:
        """
        获取需要更新的股票列表

        Args:
            stock_codes: 股票代码列表（None则检测所有股票）
            force_update: 是否强制更新所有股票

        Returns:
            需要更新的股票代码列表
        """
        if force_update:
            if stock_codes is None:
                return self._get_all_stock_codes()
            return stock_codes

        changes = self.detect_changes(stock_codes)
        return [
            code
            for code, info in changes.items()
            if info["action"] in ["new", "update"]
        ]

    def merge_incremental_data(
        self, existing_data: pd.DataFrame, new_data: pd.DataFrame, stock_code: str
    ) -> pd.DataFrame:
        """
        合并增量数据到现有数据

        Args:
            existing_data: 现有的Qlib格式数据
            new_data: 新计算的Qlib格式数据
            stock_code: 股票代码

        Returns:
            合并后的数据
        """
        try:
            if existing_data.empty:
                return new_data

            if new_data.empty:
                return existing_data

            # 确保都是MultiIndex格式
            if not isinstance(existing_data.index, pd.MultiIndex):
                existing_data = existing_data.copy()
                existing_data["stock_code"] = stock_code
                existing_data = existing_data.set_index("stock_code", append=True)
                existing_data = existing_data.swaplevel(0, 1)

            if not isinstance(new_data.index, pd.MultiIndex):
                new_data = new_data.copy()
                new_data["stock_code"] = stock_code
                new_data = new_data.set_index("stock_code", append=True)
                new_data = new_data.swaplevel(0, 1)

            # 合并数据（新数据覆盖旧数据）
            # 先删除旧数据中与新数据重叠的部分
            new_dates = (
                new_data.index.get_level_values(1)
                if isinstance(new_data.index, pd.MultiIndex)
                else new_data.index
            )
            existing_dates = (
                existing_data.index.get_level_values(1)
                if isinstance(existing_data.index, pd.MultiIndex)
                else existing_data.index
            )

            # 保留不在新数据日期范围内的旧数据
            overlap_mask = existing_dates.isin(new_dates)
            existing_data_clean = existing_data[~overlap_mask]

            # 合并新旧数据
            merged_data = pd.concat([existing_data_clean, new_data])
            merged_data = merged_data.sort_index()

            # 去重（保留新数据）
            merged_data = merged_data[~merged_data.index.duplicated(keep="last")]

            logger.debug(
                f"合并数据完成: {stock_code}, 旧数据: {len(existing_data)}, 新数据: {len(new_data)}, 合并后: {len(merged_data)}"
            )

            return merged_data

        except Exception as e:
            logger.error(f"合并增量数据失败 {stock_code}: {e}")
            # 如果合并失败，返回新数据
            return new_data
