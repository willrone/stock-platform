"""Build a clean Qlib provider tree for official-replication workflows.

The source of truth remains the local raw parquet data derived from Tushare.
This builder intentionally exports only the raw OHLCV fields required by the
canonical Qlib handlers so official-replication runs do not accidentally reuse
locally enhanced technical indicators.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from loguru import logger

from app.core.config import settings
from app.services.data.qlib_bin_converter import QlibBinConverter
from app.services.data.stock_data_loader import StockDataLoader


class OfficialQlibDataBuilder:
    def __init__(
        self,
        official_qlib_data_path: Optional[Union[str, Path]] = None,
        data_loader: Optional[Any] = None,
        bin_converter: Optional[Any] = None,
    ):
        self.official_qlib_data_path = Path(
            official_qlib_data_path or settings.OFFICIAL_QLIB_DATA_PATH
        )
        self.official_qlib_data_path.mkdir(parents=True, exist_ok=True)
        self.data_loader = data_loader or StockDataLoader()
        self.bin_converter = bin_converter or QlibBinConverter()

    def discover_available_stock_codes(self, limit: Optional[int] = None) -> List[str]:
        """Discover locally available OHLCV parquet stock codes."""
        stock_data_dir = self.data_loader.data_root / "parquet" / "stock_data"
        if not stock_data_dir.exists():
            logger.warning(f"官方Qlib构建股票数据目录不存在: {stock_data_dir}")
            return []

        codes: List[str] = []
        for file_path in sorted(stock_data_dir.glob("*.parquet")):
            stem = file_path.stem
            if "_" not in stem:
                continue
            code, market = stem.rsplit("_", 1)
            market = market.upper()
            if market in {"SZ", "SH", "BJ"}:
                codes.append(f"{code}.{market}")
            if limit is not None and len(codes) >= limit:
                break
        return codes

    def prepare_stocks(self, stock_codes: List[str]) -> Dict[str, List[str]]:
        success: List[str] = []
        failed: List[str] = []
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
