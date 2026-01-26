"""
股票数据加载工具
统一使用新格式：data/parquet/stock_data/{safe_code}.parquet
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

from app.core.config import settings


class StockDataLoader:
    """股票数据加载器，统一使用新格式路径"""

    def __init__(self, data_root: Optional[str] = None):
        """
        初始化数据加载器

        Args:
            data_root: 数据根目录，如果为None则使用配置中的路径
        """
        if data_root is None:
            self.data_root = Path(settings.DATA_ROOT_PATH)
        else:
            self.data_root = Path(data_root)

    def load_stock_data(
        self,
        stock_code: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        加载股票数据

        路径格式：data/parquet/stock_data/{safe_code}.parquet

        Args:
            stock_code: 股票代码（如 000001.SZ）
            start_date: 开始日期（可选）
            end_date: 结束日期（可选）

        Returns:
            股票数据DataFrame，索引为日期
        """
        try:
            # 转换股票代码格式（将 . 替换为 _）
            safe_code = stock_code.replace(".", "_")
            file_path = (
                self.data_root / "parquet" / "stock_data" / f"{safe_code}.parquet"
            )

            if not file_path.exists():
                logger.warning(f"股票数据文件不存在: {file_path}")
                return pd.DataFrame()

            # 读取数据
            df = pd.read_parquet(file_path)

            # 确保日期列存在并设置为索引
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date")
            elif df.index.name == "date" or df.index.dtype == "datetime64[ns]":
                # 索引已经是日期
                pass
            else:
                logger.warning(f"数据文件 {file_path} 缺少日期列或日期索引")
                return pd.DataFrame()

            # 过滤日期范围
            if start_date is not None:
                df = df[df.index >= pd.Timestamp(start_date)]
            if end_date is not None:
                df = df[df.index <= pd.Timestamp(end_date)]

            # 排序并去重
            df = df.sort_index()
            df = df[~df.index.duplicated(keep="first")]

            logger.info(f"加载股票数据: {stock_code}, 记录数: {len(df)}")
            return df

        except Exception as e:
            logger.error(f"加载股票数据失败 {stock_code}: {e}")
            return pd.DataFrame()

    def check_data_exists(self, stock_code: str) -> bool:
        """
        检查股票数据是否存在

        Returns:
            如果数据存在返回True，否则返回False
        """
        safe_code = stock_code.replace(".", "_")
        file_path = self.data_root / "parquet" / "stock_data" / f"{safe_code}.parquet"
        return file_path.exists()
