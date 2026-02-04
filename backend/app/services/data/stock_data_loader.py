"""
股票数据加载工具
统一使用新格式：data/parquet/stock_data/{safe_code}.parquet
支持优先加载 Qlib 预计算特征（data/qlib_data/features/day/{safe_code}.parquet）
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

from ...core.config import settings


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
            data_path = Path(data_root)
            # 确保使用绝对路径（多进程环境下相对路径会失效）
            if not data_path.is_absolute():
                # 相对路径：从项目根目录解析
                project_root = Path(__file__).parent.parent.parent.parent.parent
                data_path = (project_root / data_root).resolve()
            self.data_root = data_path
        
        # Qlib 预计算特征路径
        self.qlib_features_path = Path(settings.QLIB_DATA_PATH) / "features" / "day"

    def _load_precomputed_features(self, stock_code: str) -> Optional[pd.DataFrame]:
        """
        尝试加载 Qlib 预计算特征
        
        Args:
            stock_code: 股票代码（如 000001.SZ）
            
        Returns:
            预计算特征 DataFrame，如果不存在返回 None
        """
        safe_code = stock_code.replace(".", "_")
        feature_file = self.qlib_features_path / f"{safe_code}.parquet"
        
        if not feature_file.exists():
            return None
        
        try:
            df = pd.read_parquet(feature_file)
            
            # 处理 MultiIndex (stock_code, date) 或 (date, stock_code)
            if isinstance(df.index, pd.MultiIndex):
                # 重置索引，提取 date 列
                df = df.reset_index()
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"])
                    df = df.set_index("date")
                elif "level_1" in df.columns:  # 可能是 (stock_code, date)
                    df = df.rename(columns={"level_1": "date"})
                    df["date"] = pd.to_datetime(df["date"])
                    df = df.set_index("date")
                
                # 移除 stock_code 列（如果存在）
                if "stock_code" in df.columns:
                    df = df.drop(columns=["stock_code"])
            
            # 确保索引是日期类型
            if df.index.dtype != "datetime64[ns]":
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"])
                    df = df.set_index("date")
            
            logger.debug(f"✅ 加载预计算特征: {stock_code}, {len(df)} 行, {len(df.columns)} 列")
            return df
            
        except Exception as e:
            logger.warning(f"加载预计算特征失败 {stock_code}: {e}")
            return None

    def load_stock_data(
        self,
        stock_code: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        加载股票数据（优先使用预计算特征）

        加载策略：
        1. 尝试加载 Qlib 预计算特征（包含 OHLCV + 技术指标）
        2. 如果预计算特征包含 $open/$close 等，直接还原 OHLCV
        3. 否则加载原始 parquet 并合并预计算特征
        4. 如果预计算特征不存在，回退到原始 parquet

        Args:
            stock_code: 股票代码（如 000001.SZ）
            start_date: 开始日期（可选）
            end_date: 结束日期（可选）

        Returns:
            股票数据DataFrame，索引为日期
        """
        try:
            # 1. 尝试加载预计算特征
            precomputed = self._load_precomputed_features(stock_code)
            
            if precomputed is not None and not precomputed.empty:
                # 检查是否包含 OHLCV（Qlib 格式：$open, $close 等）
                has_ohlcv = all(
                    col in precomputed.columns 
                    for col in ["$open", "$high", "$low", "$close", "$volume"]
                )
                
                if has_ohlcv:
                    # 直接使用预计算数据，还原列名
                    df = precomputed.copy()
                    df = df.rename(columns={
                        "$open": "open",
                        "$high": "high",
                        "$low": "low",
                        "$close": "close",
                        "$volume": "volume",
                    })
                    logger.info(f"✅ 使用预计算特征（含 OHLCV）: {stock_code}, {len(df)} 行")
                else:
                    # 预计算特征不含 OHLCV，需要加载原始数据并合并
                    base_df = self._load_base_data(stock_code)
                    if base_df.empty:
                        logger.warning(f"原始数据不存在，仅使用预计算特征: {stock_code}")
                        df = precomputed
                    else:
                        # 合并：base_df (OHLCV) + precomputed (技术指标)
                        df = base_df.join(precomputed, how="left", rsuffix="_precomputed")
                        
                        # 处理列名冲突（保留原始列，删除 _precomputed 后缀列）
                        conflict_cols = [col for col in df.columns if col.endswith("_precomputed")]
                        if conflict_cols:
                            df = df.drop(columns=conflict_cols)
                        
                        logger.info(f"✅ 合并预计算特征: {stock_code}, {len(df)} 行")
            else:
                # 2. 回退到原始数据
                df = self._load_base_data(stock_code)
                if df.empty:
                    return pd.DataFrame()
                logger.info(f"⚠️ 未找到预计算特征，使用原始数据: {stock_code}")
            
            # 3. 过滤日期范围
            if start_date is not None:
                df = df[df.index >= pd.Timestamp(start_date)]
            if end_date is not None:
                df = df[df.index <= pd.Timestamp(end_date)]
            
            # 4. 排序并去重
            df = df.sort_index()
            df = df[~df.index.duplicated(keep="first")]
            
            return df

        except Exception as e:
            logger.error(f"加载股票数据失败 {stock_code}: {e}")
            return pd.DataFrame()

    def _load_base_data(self, stock_code: str) -> pd.DataFrame:
        """
        加载原始股票数据（OHLCV）
        
        Args:
            stock_code: 股票代码
            
        Returns:
            原始数据 DataFrame
        """
        safe_code = stock_code.replace(".", "_")
        file_path = self.data_root / "parquet" / "stock_data" / f"{safe_code}.parquet"
        
        if not file_path.exists():
            logger.warning(f"股票数据文件不存在: {file_path}")
            return pd.DataFrame()
        
        df = pd.read_parquet(file_path)
        
        # 确保日期列存在并设置为索引
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        elif df.index.name == "date" or df.index.dtype == "datetime64[ns]":
            pass
        else:
            logger.warning(f"数据文件 {file_path} 缺少日期列或日期��引")
            return pd.DataFrame()
        
        return df

    def check_data_exists(self, stock_code: str) -> bool:
        """
        检查股票数据是否存在

        Returns:
            如果数据存在返回True，否则返回False
        """
        safe_code = stock_code.replace(".", "_")
        file_path = self.data_root / "parquet" / "stock_data" / f"{safe_code}.parquet"
        return file_path.exists()
