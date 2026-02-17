"""
Qlib交易日历生成器
从实际数据中提取交易日并生成Qlib格式的交易日历文件
"""
from datetime import datetime
from pathlib import Path
from typing import Set

import pandas as pd
from loguru import logger

from app.core.config import settings


class QlibCalendarGenerator:
    """Qlib交易日历生成器"""

    def __init__(self):
        self.qlib_data_path = Path(settings.QLIB_DATA_PATH)
        self.calendar_dir = self.qlib_data_path / "calendars"
        self.calendar_file = self.calendar_dir / "day.txt"

    def generate_calendar_from_data(self, stock_data_path: Path = None) -> bool:
        """
        从股票数据中提取交易日并生成日历文件

        Args:
            stock_data_path: 股票数据路径（Parquet文件目录），如果为None则从预计算数据中提取

        Returns:
            是否成功生成
        """
        try:
            # 收集所有交易日
            trading_dates: Set[datetime] = set()

            if stock_data_path is None:
                # 从预计算数据中提取
                precomputed_path = self.qlib_data_path / "features" / "day"
                if precomputed_path.exists():
                    logger.info(f"从预计算数据中提取交易日: {precomputed_path}")
                    for parquet_file in precomputed_path.glob("*.parquet"):
                        try:
                            df = pd.read_parquet(parquet_file)
                            if isinstance(df.index, pd.MultiIndex):
                                # MultiIndex: (stock_code, date)
                                dates = df.index.get_level_values(1).unique()
                            else:
                                # 单层索引: date
                                dates = df.index.unique()
                            trading_dates.update(pd.to_datetime(dates))
                        except Exception as e:
                            logger.warning(f"读取文件失败 {parquet_file}: {e}")
                            continue
            else:
                # 从Parquet数据中提取
                logger.info(f"从Parquet数据中提取交易日: {stock_data_path}")
                if stock_data_path.is_dir():
                    for parquet_file in stock_data_path.glob("*.parquet"):
                        try:
                            df = pd.read_parquet(parquet_file)
                            if "date" in df.columns:
                                dates = pd.to_datetime(df["date"].unique())
                            elif isinstance(df.index, pd.DatetimeIndex):
                                dates = df.index.unique()
                            else:
                                continue
                            trading_dates.update(dates)
                        except Exception as e:
                            logger.warning(f"读取文件失败 {parquet_file}: {e}")
                            continue
                elif stock_data_path.is_file():
                    try:
                        df = pd.read_parquet(stock_data_path)
                        if "date" in df.columns:
                            dates = pd.to_datetime(df["date"].unique())
                        elif isinstance(df.index, pd.DatetimeIndex):
                            dates = df.index.unique()
                        else:
                            dates = []
                        trading_dates.update(dates)
                    except Exception as e:
                        logger.warning(f"读取文件失败 {stock_data_path}: {e}")

            if not trading_dates:
                logger.warning("未找到任何交易日数据")
                return False

            # 转换为日期字符串并排序
            date_strings = sorted([d.strftime("%Y%m%d") for d in trading_dates])

            # 创建目录
            self.calendar_dir.mkdir(parents=True, exist_ok=True)

            # 写入日历文件（Qlib格式：每行一个日期，格式为YYYYMMDD）
            with open(self.calendar_file, "w") as f:
                for date_str in date_strings:
                    f.write(f"{date_str}\n")

            logger.info(f"交易日历文件已生成: {self.calendar_file}, 包含 {len(date_strings)} 个交易日")
            logger.info(f"日期范围: {date_strings[0]} 至 {date_strings[-1]}")

            return True

        except Exception as e:
            logger.error(f"生成交易日历文件失败: {e}")
            return False

    def generate_calendar_from_date_range(
        self, start_date: datetime, end_date: datetime, exclude_weekends: bool = True
    ) -> bool:
        """
        从日期范围生成交易日历（排除周末）

        Args:
            start_date: 开始日期
            end_date: 结束日期
            exclude_weekends: 是否排除周末

        Returns:
            是否成功生成
        """
        try:
            # 生成日期范围
            if exclude_weekends:
                # 只包含工作日（周一到周五）
                dates = pd.bdate_range(start=start_date, end=end_date)
            else:
                dates = pd.date_range(start=start_date, end=end_date, freq="D")

            # 转换为日期字符串并排序
            date_strings = sorted([d.strftime("%Y%m%d") for d in dates])

            # 创建目录
            self.calendar_dir.mkdir(parents=True, exist_ok=True)

            # 写入日历文件
            with open(self.calendar_file, "w") as f:
                for date_str in date_strings:
                    f.write(f"{date_str}\n")

            logger.info(f"交易日历文件已生成: {self.calendar_file}, 包含 {len(date_strings)} 个交易日")
            logger.info(f"日期范围: {date_strings[0]} 至 {date_strings[-1]}")

            return True

        except Exception as e:
            logger.error(f"生成交易日历文件失败: {e}")
            return False

    def ensure_calendar_exists(self) -> bool:
        """
        确保交易日历文件存在，如果不存在则生成

        Returns:
            日历文件是否存在
        """
        if self.calendar_file.exists():
            logger.debug(f"交易日历文件已存在: {self.calendar_file}")
            return True

        logger.info("交易日历文件不存在，尝试生成...")

        # 尝试从预计算数据生成
        if self.generate_calendar_from_data():
            return True

        # 如果失败，生成一个默认的日期范围（最近5年）
        from datetime import datetime, timedelta

        end_date = datetime.now()
        start_date = end_date - timedelta(days=5 * 365)

        logger.info(f"使用默认日期范围生成交易日历: {start_date.date()} 至 {end_date.date()}")
        return self.generate_calendar_from_date_range(start_date, end_date)
