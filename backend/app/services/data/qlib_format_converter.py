"""
Qlib数据格式转换工具
将Parquet格式的股票数据转换为Qlib标准格式（MultiIndex DataFrame）
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd
from loguru import logger


class QlibFormatConverter:
    """Qlib数据格式转换器"""

    # Qlib标准列名映射
    COLUMN_MAPPING = {
        "open": "$open",
        "high": "$high",
        "low": "$low",
        "close": "$close",
        "volume": "$volume",
    }

    def __init__(self) -> None:
        """初始化转换器"""

    def convert_parquet_to_qlib(
        self, stock_data: pd.DataFrame, stock_code: str
    ) -> pd.DataFrame:
        """
        将单股票的Parquet数据转换为Qlib格式

        Args:
            stock_data: 单股票DataFrame，索引为日期，列名为 open, high, low, close, volume
            stock_code: 股票代码（如 000001.SZ）

        Returns:
            Qlib格式的DataFrame，MultiIndex为 (stock_code, date)，列名为 $open, $high, $low, $close, $volume
        """
        try:
            # 复制数据避免修改原数据
            df = stock_data.copy()

            # 确保索引是日期类型
            if not isinstance(df.index, pd.DatetimeIndex):
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"])
                    df = df.set_index("date")
                else:
                    raise ValueError(f"数据缺少日期索引或日期列: {stock_code}")

            # 确保必需的列存在
            required_cols = ["open", "high", "low", "close", "volume"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"股票 {stock_code} 缺少必需列: {missing_cols}")

            # 选择并重命名列
            qlib_cols = {}
            for old_col, new_col in self.COLUMN_MAPPING.items():
                if old_col in df.columns:
                    qlib_cols[new_col] = df[old_col]

            # 创建新的DataFrame
            qlib_df = pd.DataFrame(qlib_cols, index=df.index)

            # 添加stock_code列
            qlib_df["stock_code"] = stock_code

            # 构建MultiIndex: (stock_code, date)
            qlib_df = qlib_df.set_index("stock_code", append=True)
            qlib_df = qlib_df.swaplevel(0, 1)  # 交换层级，使stock_code在前
            qlib_df.index = qlib_df.index.set_names(["stock_code", "date"])
            qlib_df = qlib_df.sort_index()

            # 数据类型优化
            qlib_df = self._optimize_dtypes(qlib_df)

            logger.debug(
                f"转换完成: {stock_code}, 形状: {qlib_df.shape}, 日期范围: {qlib_df.index.get_level_values(1).min()} - {qlib_df.index.get_level_values(1).max()}"
            )

            return qlib_df

        except Exception as e:
            logger.error(f"转换股票 {stock_code} 数据失败: {e}")
            raise

    def convert_multiple_stocks_to_qlib(
        self, stock_data_dict: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        将多只股票的Parquet数据转换为Qlib格式并合并

        Args:
            stock_data_dict: 股票代码 -> DataFrame的字典

        Returns:
            合并后的Qlib格式DataFrame，MultiIndex为 (stock_code, date)
        """
        try:
            qlib_dataframes = []

            for stock_code, stock_data in stock_data_dict.items():
                try:
                    qlib_df = self.convert_parquet_to_qlib(stock_data, stock_code)
                    if not qlib_df.empty:
                        qlib_dataframes.append(qlib_df)
                except Exception as e:
                    logger.warning(f"跳过股票 {stock_code}: {e}")
                    continue

            if not qlib_dataframes:
                logger.warning("没有成功转换的股票数据")
                return pd.DataFrame()

            # 合并所有股票数据
            combined_df = pd.concat(qlib_dataframes, axis=0)
            combined_df = combined_df.sort_index()

            logger.info(
                f"合并完成: {len(qlib_dataframes)} 只股票, 总记录数: {len(combined_df)}"
            )

            return combined_df

        except Exception as e:
            logger.error(f"批量转换失败: {e}")
            raise

    def add_indicators_to_qlib(
        self,
        qlib_base: pd.DataFrame,
        indicators: pd.DataFrame,
        stock_code: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        将指标数据添加到Qlib格式的DataFrame中

        Args:
            qlib_base: 基础Qlib格式DataFrame（MultiIndex: stock_code, date）
            indicators: 指标DataFrame（索引为日期，列名为指标名）
            stock_code: 股票代码（如果indicators是单股票数据）

        Returns:
            添加了指标的Qlib格式DataFrame
        """
        try:
            # 确保基础索引命名一致
            if isinstance(qlib_base.index, pd.MultiIndex):
                if qlib_base.index.names != ["stock_code", "date"]:
                    qlib_base = qlib_base.copy()
                    qlib_base.index = qlib_base.index.set_names(["stock_code", "date"])
            else:
                raise ValueError("qlib_base必须为MultiIndex (stock_code, date)")

            # 如果indicators是单股票数据（索引为日期），需要转换为MultiIndex
            if not isinstance(indicators.index, pd.MultiIndex):
                if stock_code is None:
                    raise ValueError("单股票指标数据需要提供stock_code")

                # 添加stock_code到索引
                indicators = indicators.copy()
                if indicators.index.name is None:
                    indicators.index = indicators.index.set_names("date")
                indicators["stock_code"] = stock_code
                indicators = indicators.set_index("stock_code", append=True)
                indicators = indicators.swaplevel(0, 1)
                indicators.index = indicators.index.set_names(["stock_code", "date"])
            else:
                # MultiIndex: 对齐索引层级顺序和名称
                indicators = indicators.copy()
                if len(indicators.index.levels) != 2:
                    raise ValueError("指标数据MultiIndex必须是2层 (stock_code, date)")
                names = indicators.index.names
                if names != ["stock_code", "date"]:
                    # 尝试根据dtype推断日期层
                    level0 = indicators.index.get_level_values(0)
                    level1 = indicators.index.get_level_values(1)
                    is_dt0 = pd.api.types.is_datetime64_any_dtype(level0)
                    is_dt1 = pd.api.types.is_datetime64_any_dtype(level1)
                    if is_dt0 and not is_dt1:
                        indicators = indicators.swaplevel(0, 1)
                    indicators.index = indicators.index.set_names(
                        ["stock_code", "date"]
                    )

            # 检查列名冲突
            overlapping_cols = qlib_base.columns.intersection(indicators.columns)
            if len(overlapping_cols) > 0:
                logger.warning(
                    f"检测到列名冲突: {list(overlapping_cols)}，将只添加不存在的列"
                )
                # 只添加不存在的列
                new_cols = indicators.columns.difference(qlib_base.columns)
                if len(new_cols) > 0:
                    indicators = indicators[new_cols]
                else:
                    logger.warning("所有指标列都已存在，跳过添加")
                    return qlib_base

            # 合并指标到基础数据
            # 使用outer join确保所有日期都保留
            result = qlib_base.join(indicators, how="outer")

            # 排序索引
            result = result.sort_index()

            logger.debug(
                f"添加指标完成: 基础列数 {len(qlib_base.columns)}, 指标列数 {len(indicators.columns)}, 结果列数 {len(result.columns)}"
            )

            return result

        except Exception as e:
            logger.error(f"添加指标失败: {e}")
            raise

    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        优化数据类型以节省内存

        Args:
            df: 输入DataFrame

        Returns:
            优化后的DataFrame
        """
        try:
            df = df.copy()

            # 价格列使用float32（节省内存）
            price_cols = ["$open", "$high", "$low", "$close"]
            for col in price_cols:
                if col in df.columns:
                    df[col] = df[col].astype(np.float32)

            # 成交量列使用int64（保持精度）
            if "$volume" in df.columns:
                df["$volume"] = df["$volume"].astype(np.int64)

            # 指标列使用float32
            for col in df.columns:
                if (
                    col not in price_cols
                    and col != "$volume"
                    and df[col].dtype == "float64"
                ):
                    df[col] = df[col].astype(np.float32)

            return df

        except Exception as e:
            logger.warning(f"数据类型优化失败: {e}，使用原始类型")
            return df

    def save_qlib_data(
        self,
        qlib_data: pd.DataFrame,
        output_path: Path,
        stock_code: Optional[str] = None,
        format: str = "parquet",
    ) -> Path:
        """
        保存Qlib格式数据到文件

        Args:
            qlib_data: Qlib格式DataFrame
            output_path: 输出路径（目录或文件）
            stock_code: 股票代码（如果按股票保存）
            format: 保存格式（'parquet' 或 'csv'）

        Returns:
            保存的文件路径
        """
        try:
            output_path = Path(output_path)

            # 如果output_path是目录，根据stock_code生成文件名
            if output_path.is_dir() or not output_path.suffix:
                if stock_code is None:
                    raise ValueError("目录路径需要提供stock_code")
                safe_code = stock_code.replace(".", "_")
                output_path = output_path / f"{safe_code}.{format}"

            # 确保目录存在
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # 保存数据
            if format == "parquet":
                qlib_data.to_parquet(output_path, compression="snappy", index=True)
                # 确保文件写入完成并同步到磁盘
                import os

                try:
                    # 强制同步文件系统缓存
                    file_fd = os.open(str(output_path), os.O_RDONLY)
                    os.fsync(file_fd)
                    os.close(file_fd)
                except Exception:
                    pass  # 如果同步失败，继续执行
            elif format == "csv":
                qlib_data.to_csv(output_path, index=True)
            else:
                raise ValueError(f"不支持的格式: {format}")

            logger.info(f"保存Qlib数据: {output_path}, 形状: {qlib_data.shape}")

            return output_path

        except Exception as e:
            logger.error(f"保存Qlib数据失败: {e}")
            raise

    def load_qlib_data(
        self,
        file_path: Path,
        stock_code: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        columns: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """
        从文件加载Qlib格式数据

        Args:
            file_path: 文件路径
            stock_code: 股票代码（可选，用于过滤）
            start_date: 开始日期（可选）
            end_date: 结束日期（可选）
            columns: 仅读取/返回指定列（可选，Parquet 下推，减少宽表 I/O）

        Returns:
            Qlib格式DataFrame
        """
        try:
            file_path = Path(file_path)

            if not file_path.exists():
                logger.warning(f"文件不存在: {file_path}")
                return pd.DataFrame()

            df = self._load_raw_qlib_data(file_path, columns=columns)
            df = self._ensure_multiindex(df, file_path, stock_code)
            df = self._filter_by_stock_code(df, stock_code)
            if df.empty:
                return df

            df = self._filter_by_date_range(df, start_date, end_date)

            logger.debug(f"加载Qlib数据: {file_path}, 形状: {df.shape}")
            return df

        except Exception as e:
            logger.error(f"加载Qlib数据失败: {e}")
            return pd.DataFrame()

    def _load_raw_qlib_data(
        self, file_path: Path, columns: Optional[Sequence[str]] = None
    ) -> pd.DataFrame:
        if file_path.suffix == ".parquet":
            return self._read_parquet_with_fallback(file_path, columns=columns)
        if file_path.suffix == ".csv":
            return pd.read_csv(file_path, index_col=[0, 1], parse_dates=True)
        raise ValueError(f"不支持的文件格式: {file_path.suffix}")

    def _read_parquet_with_fallback(
        self, file_path: Path, columns: Optional[Sequence[str]] = None
    ) -> pd.DataFrame:
        try:
            return pd.read_parquet(file_path, engine="pyarrow", columns=columns)
        except Exception as e:
            logger.debug(f"使用pyarrow读取失败: {e}，尝试fastparquet")
            try:
                return pd.read_parquet(file_path, engine="fastparquet", columns=columns)
            except Exception as e2:
                logger.warning(f"使用fastparquet也失败: {e2}，尝试默认引擎")
                return pd.read_parquet(file_path, columns=columns)

    def _ensure_multiindex(
        self, df: pd.DataFrame, file_path: Path, stock_code: Optional[str]
    ) -> pd.DataFrame:
        if isinstance(df.index, pd.MultiIndex):
            return df

        if df.index.nlevels == 1 and len(df.columns) > 0:
            logger.warning(f"数据索引不是MultiIndex，尝试重建: {file_path}")
            if stock_code is None:
                raise ValueError(f"数据格式错误，无法重建MultiIndex: {file_path}")

            df.index = pd.MultiIndex.from_tuples(
                [(stock_code, idx) for idx in df.index],
                names=["stock_code", "date"],
            )
            return df

        raise ValueError(f"数据格式错误，期望MultiIndex: {file_path}")

    def _filter_by_stock_code(
        self, df: pd.DataFrame, stock_code: Optional[str]
    ) -> pd.DataFrame:
        if stock_code is None:
            return df

        try:
            return df.xs(stock_code, level=0, drop_level=False)
        except KeyError:
            logger.warning(f"股票 {stock_code} 不在数据中")
            return pd.DataFrame()

    def _filter_by_date_range(
        self,
        df: pd.DataFrame,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
    ) -> pd.DataFrame:
        if start_date is None and end_date is None:
            return df

        try:
            date_level = df.index.get_level_values(1).to_numpy()
            mask: np.ndarray[Any, np.dtype[np.bool_]] = np.ones(len(df), dtype=bool)

            if start_date is not None:
                start_ts = pd.Timestamp(start_date)
                mask = mask & (date_level >= start_ts)
            if end_date is not None:
                end_ts = pd.Timestamp(end_date)
                mask = mask & (date_level <= end_ts)

            return df.iloc[mask]
        except Exception as e:
            logger.warning(f"日期过滤失败: {e}，返回全部数据")
            return df
