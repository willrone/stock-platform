"""Build a clean Qlib provider tree for official-replication workflows.

The source of truth remains the local raw parquet data derived from Tushare.
This builder intentionally exports only the raw OHLCV fields required by the
canonical Qlib handlers so official-replication runs do not accidentally reuse
locally enhanced technical indicators.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from loguru import logger

from app.core.config import settings
from app.services.data.qlib_bin_converter import QlibBinConverter
from app.services.data.stock_data_loader import StockDataLoader


class OfficialQlibDataBuilder:
    def __init__(
        self,
        official_qlib_data_path: Optional[str | Path] = None,
        data_loader: Optional[Any] = None,
        bin_converter: Optional[Any] = None,
    ):
        self.official_qlib_data_path = Path(
            official_qlib_data_path or settings.OFFICIAL_QLIB_DATA_PATH
        )
        self.official_qlib_data_path.mkdir(parents=True, exist_ok=True)
        self.data_loader = data_loader or StockDataLoader()
        self.bin_converter = bin_converter or QlibBinConverter()

    def prepare_stocks(self, stock_codes: list[str]) -> dict[str, list[str]]:
        success: list[str] = []
        failed: list[str] = []
        for stock_code in stock_codes:
            try:
                base_data = self.data_loader._load_base_data(stock_code)
                if base_data is None or base_data.empty:
                    logger.warning(f"官方Qlib构建缺少原始数据: {stock_code}")
                    failed.append(stock_code)
                    continue
                result = self.bin_converter.convert_parquet_to_bin(
                    base_data[["open", "high", "low", "close", "volume"]],
                    stock_code,
                    self.official_qlib_data_path,
                )
                if result is None:
                    failed.append(stock_code)
                    continue
                success.append(stock_code)
            except Exception as exc:
                logger.error(f"官方Qlib构建失败 {stock_code}: {exc}")
                failed.append(stock_code)
        return {"success": success, "failed": failed}
