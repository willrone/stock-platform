"""
Qlib 格式转换器

将不同来源的股票数据转换为 Qlib 标准格式。
支持多种输入格式，包括 CSV、DataFrame 等。
"""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from loguru import logger

from .column_standardizer import ColumnStandardizer


class DataSourceType(Enum):
    """数据源类型枚举"""

    TUSHARE = "tushare"
    AKSHARE = "akshare"
    BAOSTOCK = "baostock"
    YAHOO = "yahoo"
    CUSTOM = "custom"


class QlibFormatConverter:
    """Qlib 格式转换器

    将不同格式的股票数据转换为 Qlib 标准格式。
    Qlib 标准格式要求：
    - 索引：MultiIndex (datetime, instrument) 或 DatetimeIndex
    - 必需列：$open, $close, $high, $low, $volume
    - 可选列：$factor, $vwap, $change 等

    Attributes:
        column_standardizer: 列名标准化器
        date_column: 日期列名
        instrument_column: 股票代码列名
    """

    # Qlib 标准列名前缀
    QLIB_PREFIX = "$"

    # Qlib 必需的 OHLCV 列
    REQUIRED_COLUMNS = ["open", "close", "high", "low", "volume"]

    # 可选列
    OPTIONAL_COLUMNS = ["factor", "vwap", "change", "amount", "turn"]

    def __init__(
        self,
        date_column: str = "date",
        instrument_column: str = "instrument",
        source_type: DataSourceType = DataSourceType.CUSTOM,
    ):
        """初始化 Qlib 格式转换器

        Args:
            date_column: 输入数据中的日期列名
            instrument_column: 输入数据中的股票代码列名
            source_type: 数据源类型，用于自动识别列名映射
        """
        self.date_column = date_column
        self.instrument_column = instrument_column
        self.source_type = source_type
        self.column_standardizer = ColumnStandardizer(source_type=source_type)

    def convert(
        self,
        data: pd.DataFrame,
        date_column: Optional[str] = None,
        instrument_column: Optional[str] = None,
        column_mapping: Optional[Dict[str, str]] = None,
        add_qlib_prefix: bool = True,
    ) -> pd.DataFrame:
        """将 DataFrame 转换为 Qlib 标准格式

        Args:
            data: 输入的股票数据 DataFrame
            date_column: 日期列名，如果为 None 则使用初始化时的设置
            instrument_column: 股票代码列名，如果为 None 则使用初始化时的设置
            column_mapping: 自定义列名映射，格式为 {原列名: 标准列名}
            add_qlib_prefix: 是否添加 Qlib 的 $ 前缀

        Returns:
            转换后的 Qlib 格式 DataFrame

        Raises:
            ValueError: 当缺少必需列或数据格式不正确时
        """
        if data.empty:
            logger.warning("输入数据为空")
            return data.copy()

        date_col = date_column or self.date_column
        instrument_col = instrument_column or self.instrument_column

        # 复制数据避免修改原始数据
        df = data.copy()

        # 1. 标准化列名
        df = self.column_standardizer.standardize(df, column_mapping=column_mapping)

        # 2. 处理日期列
        df = self._process_date_column(df, date_col)

        # 3. 处理股票代码列
        df = self._process_instrument_column(df, instrument_col)

        # 4. 验证必需列
        self._validate_required_columns(df)

        # 5. 转换数据类型
        df = self._convert_dtypes(df)

        # 6. 添加 Qlib 前缀
        if add_qlib_prefix:
            df = self._add_qlib_prefix(df)

        # 7. 设置 MultiIndex
        df = self._set_multiindex(df, date_col, instrument_col)

        # 8. 排序
        df = df.sort_index()

        logger.info(
            f"数据转换完成: {len(df)} 行, "
            f"日期范围: {df.index.get_level_values(0).min()} - "
            f"{df.index.get_level_values(0).max()}"
        )

        return df

    def convert_from_csv(
        self,
        file_path: Union[str, Path],
        date_column: Optional[str] = None,
        instrument_column: Optional[str] = None,
        column_mapping: Optional[Dict[str, str]] = None,
        **read_csv_kwargs: Any,
    ) -> pd.DataFrame:
        """从 CSV 文件读取并转换为 Qlib 格式

        Args:
            file_path: CSV 文件路径
            date_column: 日期列名
            instrument_column: 股票代码列名
            column_mapping: 自定义列名映射
            **read_csv_kwargs: 传递给 pd.read_csv 的额外参数

        Returns:
            转换后的 Qlib 格式 DataFrame
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        logger.info(f"从 CSV 文件读取数据: {file_path}")
        df = pd.read_csv(file_path, **read_csv_kwargs)

        return self.convert(
            df,
            date_column=date_column,
            instrument_column=instrument_column,
            column_mapping=column_mapping,
        )

    def convert_multiple_files(
        self,
        file_paths: List[Union[str, Path]],
        date_column: Optional[str] = None,
        instrument_column: Optional[str] = None,
        column_mapping: Optional[Dict[str, str]] = None,
    ) -> pd.DataFrame:
        """转换多个文件并合并

        Args:
            file_paths: CSV 文件路径列表
            date_column: 日期列名
            instrument_column: 股票代码列名
            column_mapping: 自定义列名映射

        Returns:
            合并后的 Qlib 格式 DataFrame
        """
        dfs = []
        for file_path in file_paths:
            try:
                df = self.convert_from_csv(
                    file_path,
                    date_column=date_column,
                    instrument_column=instrument_column,
                    column_mapping=column_mapping,
                )
                dfs.append(df)
            except Exception as e:
                logger.warning(f"处理文件 {file_path} 失败: {e}")
                continue

        if not dfs:
            raise ValueError("没有成功转换任何文件")

        result = pd.concat(dfs)
        result = result[~result.index.duplicated(keep="last")]
        result = result.sort_index()

        logger.info(f"合并 {len(dfs)} 个文件完成，共 {len(result)} 行数据")
        return result

    def _process_date_column(self, df: pd.DataFrame, date_col: str) -> pd.DataFrame:
        """处理日期列

        Args:
            df: 输入 DataFrame
            date_col: 日期列名

        Returns:
            处理后的 DataFrame
        """
        # 尝试多种可能的日期列名
        possible_date_cols = [date_col, "date", "datetime", "trade_date", "time"]

        found_col = None
        for col in possible_date_cols:
            if col in df.columns:
                found_col = col
                break

        if found_col is None:
            # 检查索引是否为日期类型
            if isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index()
                df = df.rename(columns={df.columns[0]: date_col})
                found_col = date_col
            else:
                raise ValueError(
                    f"找不到日期列，尝试过的列名: {possible_date_cols}"
                )

        # 重命名为标准日期列名
        if found_col != date_col:
            df = df.rename(columns={found_col: date_col})

        # 转换为 datetime 类型
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df[date_col] = pd.to_datetime(df[date_col])

        return df

    def _process_instrument_column(
        self, df: pd.DataFrame, instrument_col: str
    ) -> pd.DataFrame:
        """处理股票代码列

        Args:
            df: 输入 DataFrame
            instrument_col: 股票代码列名

        Returns:
            处理后的 DataFrame
        """
        # 尝试多种可能的股票代码列名
        possible_instrument_cols = [
            instrument_col,
            "instrument",
            "code",
            "stock_code",
            "symbol",
            "ts_code",
            "ticker",
        ]

        found_col = None
        for col in possible_instrument_cols:
            if col in df.columns:
                found_col = col
                break

        if found_col is None:
            raise ValueError(
                f"找不到股票代码列，尝试过的列名: {possible_instrument_cols}"
            )

        # 重命名为标准列名
        if found_col != instrument_col:
            df = df.rename(columns={found_col: instrument_col})

        # 标准化股票代码格式
        df[instrument_col] = df[instrument_col].astype(str)
        df[instrument_col] = df[instrument_col].apply(self._normalize_instrument_code)

        return df

    def _normalize_instrument_code(self, code: str) -> str:
        """标准化股票代码格式

        将各种格式的股票代码转换为 Qlib 标准格式（如 SH600000, SZ000001）

        Args:
            code: 原始股票代码

        Returns:
            标准化后的股票代码
        """
        code = str(code).strip().upper()

        # 已经是标准格式
        if code.startswith(("SH", "SZ")):
            return code

        # Tushare 格式: 600000.SH -> SH600000
        if "." in code:
            parts = code.split(".")
            if len(parts) == 2:
                num, exchange = parts
                if exchange in ("SH", "SS"):
                    return f"SH{num}"
                elif exchange == "SZ":
                    return f"SZ{num}"

        # 纯数字格式，根据规则判断交易所
        if code.isdigit():
            if code.startswith(("6", "9")):
                return f"SH{code}"
            elif code.startswith(("0", "3", "2")):
                return f"SZ{code}"

        # 无法识别的格式，保持原样
        logger.warning(f"无法识别的股票代码格式: {code}")
        return code

    def _validate_required_columns(self, df: pd.DataFrame) -> None:
        """验证必需列是否存在

        Args:
            df: 输入 DataFrame

        Raises:
            ValueError: 当缺少必需列时
        """
        missing_cols = []
        for col in self.REQUIRED_COLUMNS:
            if col not in df.columns:
                missing_cols.append(col)

        if missing_cols:
            raise ValueError(f"缺少必需列: {missing_cols}")

    def _convert_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """转换数据类型

        Args:
            df: 输入 DataFrame

        Returns:
            转换后的 DataFrame
        """
        # OHLCV 列转换为 float64
        numeric_cols = self.REQUIRED_COLUMNS + self.OPTIONAL_COLUMNS
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float64)

        return df

    def _add_qlib_prefix(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加 Qlib 列名前缀

        Args:
            df: 输入 DataFrame

        Returns:
            添加前缀后的 DataFrame
        """
        rename_map = {}
        all_cols = self.REQUIRED_COLUMNS + self.OPTIONAL_COLUMNS

        for col in all_cols:
            if col in df.columns:
                new_name = f"{self.QLIB_PREFIX}{col}"
                rename_map[col] = new_name

        return df.rename(columns=rename_map)

    def _set_multiindex(
        self, df: pd.DataFrame, date_col: str, instrument_col: str
    ) -> pd.DataFrame:
        """设置 MultiIndex

        Args:
            df: 输入 DataFrame
            date_col: 日期列名
            instrument_col: 股票代码列名

        Returns:
            设置 MultiIndex 后的 DataFrame
        """
        if date_col in df.columns and instrument_col in df.columns:
            df = df.set_index([date_col, instrument_col])
        elif date_col in df.columns:
            df = df.set_index(date_col)

        return df

    def to_qlib_bin_format(
        self,
        df: pd.DataFrame,
        output_dir: Union[str, Path],
        freq: str = "day",
    ) -> None:
        """将数据导出为 Qlib 二进制格式

        Args:
            df: Qlib 格式的 DataFrame
            output_dir: 输出目录
            freq: 数据频率，'day' 或 'min'
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 按股票代码分组导出
        if isinstance(df.index, pd.MultiIndex):
            instruments = df.index.get_level_values(1).unique()
        else:
            raise ValueError("DataFrame 必须有 MultiIndex (datetime, instrument)")

        for instrument in instruments:
            instrument_data = df.xs(instrument, level=1)
            instrument_dir = output_dir / instrument
            instrument_dir.mkdir(parents=True, exist_ok=True)

            # 导出每个特征列
            for col in instrument_data.columns:
                feature_file = instrument_dir / f"{col}.{freq}.bin"
                values = instrument_data[col].values.astype(np.float32)
                values.tofile(feature_file)

        logger.info(f"导出 {len(instruments)} 只股票数据到 {output_dir}")
