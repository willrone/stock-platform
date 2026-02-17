"""
列名标准化器

将不同数据源的列名标准化为统一格式。
支持 Tushare、AKShare、BaoStock、Yahoo Finance 等数据源。
"""

from typing import Dict, List, Optional

import pandas as pd
from loguru import logger


class DataSourceType:
    """数据源类型常量"""

    TUSHARE = "tushare"
    AKSHARE = "akshare"
    BAOSTOCK = "baostock"
    YAHOO = "yahoo"
    CUSTOM = "custom"


class ColumnStandardizer:
    """列名标准化器

    将不同数据源的列名映射到标准列名。

    标准列名：
    - open: 开盘价
    - close: 收盘价
    - high: 最高价
    - low: 最低价
    - volume: 成交量
    - amount: 成交额
    - factor: 复权因子
    - vwap: 成交量加权平均价
    - change: 涨跌幅
    - turn: 换手率

    Attributes:
        source_type: 数据源类型
        column_mappings: 各数据源的列名映射配置
    """

    # 标准列名
    STANDARD_COLUMNS = [
        "open",
        "close",
        "high",
        "low",
        "volume",
        "amount",
        "factor",
        "vwap",
        "change",
        "turn",
        "pre_close",
        "adj_factor",
    ]

    # 各数据源的列名映射
    SOURCE_MAPPINGS: Dict[str, Dict[str, str]] = {
        DataSourceType.TUSHARE: {
            "ts_code": "instrument",
            "trade_date": "date",
            "open": "open",
            "close": "close",
            "high": "high",
            "low": "low",
            "vol": "volume",
            "amount": "amount",
            "pre_close": "pre_close",
            "change": "change",
            "pct_chg": "pct_change",
            "adj_factor": "factor",
            "turnover_rate": "turn",
        },
        DataSourceType.AKSHARE: {
            "日期": "date",
            "股票代码": "instrument",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
            "涨跌幅": "change",
            "换手率": "turn",
            "振幅": "amplitude",
        },
        DataSourceType.BAOSTOCK: {
            "date": "date",
            "code": "instrument",
            "open": "open",
            "close": "close",
            "high": "high",
            "low": "low",
            "volume": "volume",
            "amount": "amount",
            "adjustflag": "adj_flag",
            "turn": "turn",
            "pctChg": "change",
            "preclose": "pre_close",
            "isST": "is_st",
        },
        DataSourceType.YAHOO: {
            "Date": "date",
            "Open": "open",
            "Close": "close",
            "High": "high",
            "Low": "low",
            "Volume": "volume",
            "Adj Close": "adj_close",
        },
        DataSourceType.CUSTOM: {},
    }

    # 通用列名别名（用于自动识别）
    COLUMN_ALIASES: Dict[str, List[str]] = {
        "date": [
            "date",
            "datetime",
            "trade_date",
            "日期",
            "交易日期",
            "time",
            "Date",
            "DATE",
        ],
        "instrument": [
            "instrument",
            "code",
            "stock_code",
            "symbol",
            "ts_code",
            "ticker",
            "股票代码",
            "代码",
        ],
        "open": ["open", "Open", "OPEN", "开���", "开盘价"],
        "close": ["close", "Close", "CLOSE", "收盘", "收盘价"],
        "high": ["high", "High", "HIGH", "最高", "最高价"],
        "low": ["low", "Low", "LOW", "最低", "最低价"],
        "volume": [
            "volume",
            "Volume",
            "VOLUME",
            "vol",
            "成交量",
            "交易量",
        ],
        "amount": ["amount", "Amount", "AMOUNT", "成交额", "交易额", "turnover"],
        "change": [
            "change",
            "Change",
            "pct_chg",
            "pctChg",
            "涨跌幅",
            "涨跌",
            "pct_change",
        ],
        "turn": ["turn", "turnover_rate", "换手率", "turnover"],
        "factor": ["factor", "adj_factor", "adjustflag", "复权因子"],
        "pre_close": ["pre_close", "preclose", "昨收", "昨收价", "前收盘"],
        "vwap": ["vwap", "VWAP", "成交均价", "均价"],
    }

    def __init__(self, source_type: str = DataSourceType.CUSTOM):
        """初始化列名标准化器

        Args:
            source_type: 数据源类型
        """
        self.source_type = source_type
        self._build_reverse_mapping()

    def _build_reverse_mapping(self) -> None:
        """构建反向映射（别名 -> 标准名）"""
        self._alias_to_standard: Dict[str, str] = {}
        for standard_name, aliases in self.COLUMN_ALIASES.items():
            for alias in aliases:
                self._alias_to_standard[alias.lower()] = standard_name

    def standardize(
        self,
        df: pd.DataFrame,
        column_mapping: Optional[Dict[str, str]] = None,
        inplace: bool = False,
    ) -> pd.DataFrame:
        """标准化 DataFrame 的列名

        Args:
            df: 输入 DataFrame
            column_mapping: 自定义列名映射，优先级最高
            inplace: 是否原地修改

        Returns:
            标准化后的 DataFrame
        """
        if not inplace:
            df = df.copy()

        # 构建最终的列名映射
        final_mapping = self._build_final_mapping(df.columns.tolist(), column_mapping)

        if final_mapping:
            df = df.rename(columns=final_mapping)
            logger.debug(f"列名映射: {final_mapping}")

        return df

    def _build_final_mapping(
        self,
        columns: List[str],
        custom_mapping: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        """构建最终的列名映射

        优先级：自定义映射 > 数据源映射 > 自动识别

        Args:
            columns: 原始列名列表
            custom_mapping: 自定义列名映射

        Returns:
            最终的列名映射字典
        """
        final_mapping: Dict[str, str] = {}

        # 获取数据源特定的映射
        source_mapping = self.SOURCE_MAPPINGS.get(self.source_type, {})

        for col in columns:
            new_name = None

            # 1. 优先使用自定义映射
            if custom_mapping and col in custom_mapping:
                new_name = custom_mapping[col]

            # 2. 使用数据源特定映射
            elif col in source_mapping:
                new_name = source_mapping[col]

            # 3. 自动识别（通过别名）
            else:
                new_name = self._auto_detect_column(col)

            # 只有当新名称与原名称不同时才添加映射
            if new_name and new_name != col:
                final_mapping[col] = new_name

        return final_mapping

    def _auto_detect_column(self, column_name: str) -> Optional[str]:
        """自动识别列名

        Args:
            column_name: 原始列名

        Returns:
            标准列名，如果无法识别则返回 None
        """
        # 转换为小写进行匹配
        col_lower = column_name.lower().strip()

        # 直接匹配
        if col_lower in self._alias_to_standard:
            return self._alias_to_standard[col_lower]

        # 注意：不再使用模糊匹配（包含关系）。
        # 之前的模糊匹配会把 price_change → change、volume_change → volume 等
        # 技术指标/衍生列错误映射为 OHLCV 标准列，导致重复列名。
        # 如果需要支持新的数据源列名，请在 COLUMN_ALIASES 中添加精确别名。

        return None

    def get_standard_columns(self, df: pd.DataFrame) -> List[str]:
        """获取 DataFrame 中已标准化的列名

        Args:
            df: 输入 DataFrame

        Returns:
            已标准化的列名列表
        """
        return [col for col in df.columns if col in self.STANDARD_COLUMNS]

    def get_missing_columns(
        self, df: pd.DataFrame, required: Optional[List[str]] = None
    ) -> List[str]:
        """获取缺失的必需列

        Args:
            df: 输入 DataFrame
            required: 必需列列表，默认为 OHLCV

        Returns:
            缺失的列名列表
        """
        if required is None:
            required = ["open", "close", "high", "low", "volume"]

        return [col for col in required if col not in df.columns]

    def validate_columns(
        self, df: pd.DataFrame, required: Optional[List[str]] = None
    ) -> bool:
        """验证必需列是否存在

        Args:
            df: 输入 DataFrame
            required: 必需列列表

        Returns:
            是否所有必需列都存在
        """
        missing = self.get_missing_columns(df, required)
        if missing:
            logger.warning(f"缺失必需列: {missing}")
            return False
        return True

    def add_source_mapping(self, source_type: str, mapping: Dict[str, str]) -> None:
        """添加新的数据源映射

        Args:
            source_type: 数据源类型名称
            mapping: 列名映射字典
        """
        self.SOURCE_MAPPINGS[source_type] = mapping
        logger.info(f"添加数据源映射: {source_type}")

    def get_column_info(self, df: pd.DataFrame) -> Dict[str, Dict[str, str]]:
        """获取列信息摘要

        Args:
            df: 输入 DataFrame

        Returns:
            列信息字典，包含原始列名、标准列名、数据类型
        """
        info = {}
        for col in df.columns:
            standard_name = self._auto_detect_column(col)
            info[col] = {
                "original": col,
                "standard": standard_name or "unknown",
                "dtype": str(df[col].dtype),
            }
        return info
