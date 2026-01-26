"""
数据加载器

负责加载回测所需的历史数据
优先从Qlib预计算结果读取，如果不可用则fallback到Parquet现场计算
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger

from app.core.config import settings
from app.core.error_handler import ErrorContext, ErrorSeverity, TaskError


class DataLoader:
    """数据加载器"""

    def __init__(
        self, data_dir: str = "backend/data", max_workers: Optional[int] = None
    ):
        self.data_dir = Path(data_dir)
        self.max_workers = max_workers  # 用于并行加载数据
        self.qlib_data_path = Path(settings.QLIB_DATA_PATH) / "features" / "day"

    def load_stock_data(
        self, stock_code: str, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """
        加载股票历史数据，优先从预计算结果读取

        Args:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            包含OHLCV和所有指标的DataFrame
        """
        try:
            # 1. 优先尝试从Qlib预计算目录加载
            precomputed_data = self._load_from_precomputed(
                stock_code, start_date, end_date
            )
            if precomputed_data is not None and not precomputed_data.empty:
                logger.info(
                    f"从预计算结果加载: {stock_code}, 指标数: {len(precomputed_data.columns)}"
                )
                return precomputed_data

            # 2. Fallback：从Parquet加载基础数据（现场计算指标）
            logger.info(f"预计算结果不可用，从Parquet加载并计算: {stock_code}")
            return self._load_from_parquet_and_calculate(
                stock_code, start_date, end_date
            )

        except TaskError:
            raise
        except Exception as e:
            raise TaskError(
                message=f"加载股票数据失败: {str(e)}",
                severity=ErrorSeverity.HIGH,
                context=ErrorContext(stock_code=stock_code),
                original_exception=e,
            )

    def _load_from_precomputed(
        self, stock_code: str, start_date: datetime, end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """
        从Qlib预计算目录加载数据

        Args:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            预计算数据DataFrame，如果不存在或加载失败返回None
        """
        try:
            from app.services.data.qlib_format_converter import QlibFormatConverter

            converter = QlibFormatConverter()
            safe_code = stock_code.replace(".", "_")

            # 尝试从单股票文件加载
            stock_file = self.qlib_data_path / f"{safe_code}.parquet"
            if stock_file.exists():
                qlib_data = converter.load_qlib_data(
                    stock_file,
                    stock_code=stock_code,
                    start_date=start_date,
                    end_date=end_date,
                )

                if not qlib_data.empty:
                    # 转换为回测需要的格式（单股票DataFrame，索引为日期）
                    # 从MultiIndex中提取单股票数据
                    if isinstance(qlib_data.index, pd.MultiIndex):
                        try:
                            stock_data = qlib_data.xs(
                                stock_code, level=0, drop_level=False
                            )
                            # 将日期索引提取出来
                            stock_data.index = stock_data.index.get_level_values(1)
                        except KeyError:
                            # 如果MultiIndex中没有该股票，尝试直接使用
                            if qlib_data.index.nlevels == 2:
                                stock_data = qlib_data.copy()
                                stock_data.index = stock_data.index.get_level_values(1)
                            else:
                                return None
                    else:
                        stock_data = qlib_data.copy()

                    # 列名映射：$close -> close等（回测策略期望的格式）
                    column_mapping = {
                        "$open": "open",
                        "$high": "high",
                        "$low": "low",
                        "$close": "close",
                        "$volume": "volume",
                    }
                    stock_data = stock_data.rename(columns=column_mapping)

                    # 确保必需的列存在
                    required_cols = ["open", "high", "low", "close", "volume"]
                    if all(col in stock_data.columns for col in required_cols):
                        # 添加股票代码属性
                        stock_data.attrs["stock_code"] = stock_code
                        stock_data.attrs["from_precomputed"] = True
                        return stock_data

            # 尝试从合并文件加载
            all_stocks_file = self.qlib_data_path / "all_stocks.parquet"
            if all_stocks_file.exists():
                qlib_data = converter.load_qlib_data(
                    all_stocks_file,
                    stock_code=stock_code,
                    start_date=start_date,
                    end_date=end_date,
                )

                if not qlib_data.empty:
                    # 转换为回测需要的格式
                    if isinstance(qlib_data.index, pd.MultiIndex):
                        try:
                            stock_data = qlib_data.xs(
                                stock_code, level=0, drop_level=False
                            )
                            stock_data.index = stock_data.index.get_level_values(1)
                        except KeyError:
                            return None
                    else:
                        stock_data = qlib_data.copy()

                    # 列名映射
                    column_mapping = {
                        "$open": "open",
                        "$high": "high",
                        "$low": "low",
                        "$close": "close",
                        "$volume": "volume",
                    }
                    stock_data = stock_data.rename(columns=column_mapping)

                    required_cols = ["open", "high", "low", "close", "volume"]
                    if all(col in stock_data.columns for col in required_cols):
                        stock_data.attrs["stock_code"] = stock_code
                        stock_data.attrs["from_precomputed"] = True
                        return stock_data

            return None

        except Exception as e:
            logger.warning(f"从预计算结果加载失败 {stock_code}: {e}")
            return None

    def _load_from_parquet_and_calculate(
        self, stock_code: str, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """
        从Parquet加载基础数据（Fallback方法）

        Args:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            包含基础OHLCV数据的DataFrame（指标需要策略中计算）
        """
        # 使用统一的数据加载器
        from app.services.data.stock_data_loader import StockDataLoader

        loader = StockDataLoader(data_root=str(self.data_dir))

        # 加载数据
        data = loader.load_stock_data(
            stock_code, start_date=start_date, end_date=end_date
        )

        if data.empty:
            raise TaskError(
                message=f"未找到股票数据文件: {stock_code}",
                severity=ErrorSeverity.HIGH,
                context=ErrorContext(stock_code=stock_code),
            )

        if len(data) == 0:
            raise TaskError(
                message=f"指定日期范围内无数据: {stock_code}, {start_date} - {end_date}",
                severity=ErrorSeverity.MEDIUM,
                context=ErrorContext(stock_code=stock_code),
            )

        # 验证必需的列
        required_columns = ["open", "high", "low", "close", "volume"]
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise TaskError(
                message=f"数据缺少必需列: {missing_columns}",
                severity=ErrorSeverity.HIGH,
                context=ErrorContext(stock_code=stock_code),
            )

        # 添加股票代码属性
        data.attrs["stock_code"] = stock_code
        data.attrs["from_precomputed"] = False

        logger.info(
            f"从Parquet加载股票数据: {stock_code}, 数据量: {len(data)}, 日期范围: {data.index[0]} - {data.index[-1]}"
        )
        return data

    def load_multiple_stocks(
        self,
        stock_codes: List[str],
        start_date: datetime,
        end_date: datetime,
        parallel: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """
        加载多只股票数据，优先从预计算结果读取

        Args:
            stock_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            parallel: 是否并行加载（默认True）
        """
        stock_data = {}
        failed_stocks = []
        precomputed_count = 0

        if parallel and len(stock_codes) > 1 and self.max_workers:
            # 并行加载多只股票数据
            max_workers = min(self.max_workers, len(stock_codes))
            logger.info(f"并行加载 {len(stock_codes)} 只股票数据，使用 {max_workers} 个线程")

            def load_single_stock(
                stock_code: str,
            ) -> Tuple[str, Optional[pd.DataFrame], Optional[str], bool]:
                """加载单只股票数据，返回 (stock_code, data, error, from_precomputed)"""
                try:
                    data = self.load_stock_data(stock_code, start_date, end_date)
                    from_precomputed = data.attrs.get("from_precomputed", False)
                    return (stock_code, data, None, from_precomputed)
                except Exception as e:
                    error_msg = str(e)
                    logger.error(f"加载股票数据失败: {stock_code}, 错误: {error_msg}")
                    return (stock_code, None, error_msg, False)

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(load_single_stock, code): code
                    for code in stock_codes
                }

                for future in as_completed(futures):
                    stock_code, data, error, from_precomputed = future.result()
                    if data is not None:
                        stock_data[stock_code] = data
                        if from_precomputed:
                            precomputed_count += 1
                    else:
                        failed_stocks.append(stock_code)
        else:
            # 顺序加载（兼容旧逻辑）
            for stock_code in stock_codes:
                try:
                    data = self.load_stock_data(stock_code, start_date, end_date)
                    stock_data[stock_code] = data
                    if data.attrs.get("from_precomputed", False):
                        precomputed_count += 1
                except Exception as e:
                    logger.error(f"加载股票数据失败: {stock_code}, 错误: {e}")
                    failed_stocks.append(stock_code)
                    continue

        if precomputed_count > 0:
            logger.info(f"从预计算结果加载了 {precomputed_count}/{len(stock_data)} 只股票的数据")

        if failed_stocks:
            logger.warning(f"部分股票数据加载失败: {failed_stocks}")

        if not stock_data:
            raise TaskError(message="所有股票数据加载失败", severity=ErrorSeverity.HIGH)

        return stock_data
