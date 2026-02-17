"""
Qlib Bin格式转换工具
将Parquet格式的股票数据转换为Qlib Bin格式，供Alpha158 handler使用
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


class QlibBinConverter:
    """Qlib Bin格式转换器"""

    def __init__(self):
        """初始化转换器"""

    def convert_parquet_to_bin(
        self,
        parquet_data: pd.DataFrame,
        stock_code: str,
        qlib_data_path: Path,
    ) -> Optional[Path]:
        """
        将Parquet数据转换为Qlib Bin格式

        Args:
            parquet_data: DataFrame，索引为日期，列名为 open/high/low/close/volume 或 $open 等
            stock_code: 股票代码（如 002463.SZ）
            qlib_data_path: Qlib数据根目录（包含features/day、calendars、instruments）

        Returns:
            任意一个生成的bin文件路径（用于确认成功）
        """
        try:
            import qlib
            from qlib.config import REG_CN
            from qlib.data.storage.file_storage import FileFeatureStorage

            qlib_data_path = Path(qlib_data_path).resolve()
            qlib_data_path.mkdir(parents=True, exist_ok=True)

            # 确保Qlib初始化
            try:
                qlib.init(
                    region=REG_CN, provider_uri=str(qlib_data_path), auto_mount=False
                )
            except Exception:
                # 已初始化或重复初始化失败时继续
                pass

            # 标准化列名和索引
            data = parquet_data.copy()
            if isinstance(data.index, pd.MultiIndex):
                data = data.droplevel(0)
            if "date" in data.columns:
                data["date"] = pd.to_datetime(data["date"])
                data = data.set_index("date")
            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index)
            data = data.sort_index()

            # 支持$前缀列
            col_map = {}
            for col in ["open", "high", "low", "close", "volume"]:
                if col in data.columns:
                    col_map[col] = col
                elif f"${col}" in data.columns:
                    col_map[col] = f"${col}"
            missing = [
                c
                for c in ["open", "high", "low", "close", "volume"]
                if c not in col_map
            ]
            if missing:
                logger.warning(f"转换bin失败 {stock_code}: 缺少列 {missing}")
                return None
            data = data[
                [col_map[c] for c in ["open", "high", "low", "close", "volume"]]
            ]
            data.columns = ["open", "high", "low", "close", "volume"]

            # 确保交易日历文件存在，缺失则基于当前数据补齐
            calendar_file = qlib_data_path / "calendars" / "day.txt"
            calendar_file.parent.mkdir(parents=True, exist_ok=True)
            dates = pd.DatetimeIndex(data.index.unique()).sort_values()
            if calendar_file.exists():
                raw_lines = [
                    l.strip()
                    for l in calendar_file.read_text().splitlines()
                    if l.strip()
                ]
                valid_lines = [l for l in raw_lines if len(l) == 8 and l.isdigit()]
                existing = pd.to_datetime(valid_lines, format="%Y%m%d", errors="coerce")
                existing = pd.DatetimeIndex(existing).dropna()
                calendar = existing.union(dates).sort_values()
            else:
                calendar = dates
            calendar_file.write_text(
                "\n".join(calendar.strftime("%Y%m%d")) + "\n", encoding="utf-8"
            )

            # 对齐到交易日历
            data = data.reindex(calendar)

            # 写入bin
            instrument = stock_code.replace(".", "_").lower()
            features_dir = qlib_data_path / "features" / instrument
            features_dir.mkdir(parents=True, exist_ok=True)

            for field in ["open", "high", "low", "close", "volume"]:
                storage = FileFeatureStorage(
                    instrument=instrument,
                    field=field,
                    freq="day",
                    provider_uri={"day": str(qlib_data_path)},
                )
                # Ensure a clean write to avoid rewrite path using float index from existing bins
                if storage.uri.exists():
                    storage.uri.unlink()
                storage.write(data[field].astype(np.float32).values, index=0)

            # 更新instruments/all.txt
            inst_file = qlib_data_path / "instruments" / "all.txt"
            inst_file.parent.mkdir(parents=True, exist_ok=True)
            valid_dates = data["close"].dropna().index
            if len(valid_dates) > 0:
                start_str = valid_dates.min().strftime("%Y-%m-%d")
                end_str = valid_dates.max().strftime("%Y-%m-%d")
                new_line = f"{instrument}\t{start_str}\t{end_str}\n"
                if inst_file.exists():
                    lines = [
                        l
                        for l in inst_file.read_text(encoding="utf-8").splitlines()
                        if l.strip()
                    ]
                    updated = False
                    out_lines = []
                    for line in lines:
                        code = line.split()[0]
                        if code == instrument:
                            out_lines.append(new_line.rstrip("\n"))
                            updated = True
                        else:
                            out_lines.append(line)
                    if not updated:
                        out_lines.append(new_line.rstrip("\n"))
                    inst_file.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
                else:
                    inst_file.write_text(new_line, encoding="utf-8")

            return features_dir / "close.day.bin"

        except Exception as e:
            logger.error(f"转换为bin格式失败: {e}", exc_info=True)
            return None
