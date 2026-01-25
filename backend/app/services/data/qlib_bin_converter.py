"""
Qlib Bin格式转换工具
将Parquet格式的股票数据转换为Qlib Bin格式，供Alpha158 handler使用
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from loguru import logger
import subprocess
import sys
import os

from app.core.config import settings


class QlibBinConverter:
    """Qlib Bin格式转换器"""
    
    def __init__(self):
        """初始化转换器"""
        pass
    
    def convert_parquet_to_bin(
        self,
        parquet_data: pd.DataFrame,
        stock_code: str,
        output_dir: Path
    ) -> Path:
        """
        将Parquet数据转换为Qlib Bin格式
        
        Args:
            parquet_data: Parquet格式的DataFrame，索引为日期，列名为 open, high, low, close, volume
            stock_code: 股票代码（如 002463.SZ）
            output_dir: 输出目录（qlib features/day目录）
            
        Returns:
            保存的bin文件路径
        """
        try:
            # 创建临时CSV文件（qlib的dump_bin需要CSV格式）
            import tempfile
            temp_csv = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
            temp_csv_path = Path(temp_csv.name)
            
            # 准备CSV数据（qlib期望的格式）
            csv_data = parquet_data.copy()
            csv_data = csv_data.reset_index()
            csv_data['instrument'] = stock_code
            # 确保日期格式正确
            if 'date' in csv_data.columns:
                csv_data['date'] = pd.to_datetime(csv_data['date']).dt.strftime('%Y%m%d')
            # 重命名列为qlib期望的格式
            csv_data = csv_data.rename(columns={
                'date': 'date',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume',
            })
            # 确保列顺序
            csv_data = csv_data[['instrument', 'date', 'open', 'high', 'low', 'close', 'volume']]
            
            # 保存为CSV
            csv_data.to_csv(temp_csv_path, index=False)
            temp_csv.close()
            
            logger.debug(f"创建临时CSV文件: {temp_csv_path}")
            
            # 使用qlib的dump_bin脚本转换为bin格式
            # 注意：需要找到qlib的dump_bin脚本
            qlib_scripts_path = self._find_qlib_scripts()
            if qlib_scripts_path:
                dump_bin_script = qlib_scripts_path / "dump_bin.py"
                if dump_bin_script.exists():
                    # 运行dump_bin脚本
                    output_dir.mkdir(parents=True, exist_ok=True)
                    cmd = [
                        sys.executable,
                        str(dump_bin_script),
                        "dump",
                        "--csv_path", str(temp_csv_path.parent),
                        "--qlib_dir", str(output_dir.parent.parent),
                        "--include_fields", "open,close,high,low,volume"
                    ]
                    logger.debug(f"运行dump_bin命令: {' '.join(cmd)}")
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                    if result.returncode == 0:
                        logger.info(f"成功转换为bin格式: {stock_code}")
                    else:
                        logger.warning(f"dump_bin转换失败: {result.stderr}")
                else:
                    logger.warning(f"找不到dump_bin脚本: {dump_bin_script}")
            else:
                logger.warning("找不到qlib scripts目录，无法使用dump_bin")
            
            # 清理临时文件
            try:
                temp_csv_path.unlink()
            except Exception:
                pass
            
            # 返回bin文件路径（qlib的bin文件通常保存在features/day目录下）
            bin_file = output_dir / f"{stock_code.replace('.', '_')}.bin"
            return bin_file if bin_file.exists() else None
            
        except Exception as e:
            logger.error(f"转换为bin格式失败: {e}", exc_info=True)
            return None
    
    def _find_qlib_scripts(self) -> Optional[Path]:
        """查找qlib的scripts目录"""
        try:
            import qlib
            qlib_path = Path(qlib.__file__).parent
            scripts_path = qlib_path.parent / "scripts"
            if scripts_path.exists():
                return scripts_path
            # 尝试其他可能的位置
            possible_paths = [
                qlib_path / "scripts",
                Path(sys.prefix) / "lib" / "python*" / "site-packages" / "qlib" / "scripts",
            ]
            for path_pattern in possible_paths:
                import glob
                matches = glob.glob(str(path_pattern))
                if matches:
                    return Path(matches[0])
        except Exception as e:
            logger.debug(f"查找qlib scripts失败: {e}")
        return None
