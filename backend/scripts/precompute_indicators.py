#!/usr/bin/env python3
"""
预计算技术指标并写入 Qlib features 目录

将 DataLoader 中临时计算的常用指标（MA20/MA50/MA60/STD20/STD60/RSI14）
持久化到 Qlib 格式，加速后续回测加载
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import List, Optional

import pandas as pd
from loguru import logger

# 添加项目根目录到路径
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from app.core.config import settings
from app.services.data.stock_data_loader import StockDataLoader
from app.services.data.qlib_format_converter import QlibFormatConverter


class IndicatorPrecomputer:
    """技术指标预计算器"""

    def __init__(self, data_root: str = "backend/data"):
        self.data_root = Path(data_root)
        self.qlib_data_path = Path(settings.QLIB_DATA_PATH) / "features" / "day"
        self.qlib_data_path.mkdir(parents=True, exist_ok=True)
        
        self.loader = StockDataLoader(data_root=str(self.data_root))
        self.converter = QlibFormatConverter()

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算常用技术指标
        
        Args:
            data: 包含 OHLCV 的 DataFrame
            
        Returns:
            添加了指标列的 DataFrame
        """
        result = data.copy()
        close = result["close"]
        
        try:
            # 均线
            for period in (20, 50, 60):
                col = f"MA{period}"
                if col not in result.columns:
                    result[col] = close.rolling(window=period).mean()
            
            # 标准差
            for period in (20, 60):
                col = f"STD{period}"
                if col not in result.columns:
                    result[col] = close.rolling(window=period).std()
            
            # RSI14 (Wilder 简化版)
            if "RSI14" not in result.columns:
                delta = close.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                result["RSI14"] = 100 - (100 / (1 + rs))
                
        except Exception as e:
            logger.error(f"计算指标失败: {e}")
            raise
            
        return result

    def precompute_stock(
        self,
        stock_code: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> bool:
        """
        预计算单只股票的指标并保存
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期（None = 全部历史）
            end_date: 结束日期（None = 最新）
            
        Returns:
            是否成功
        """
        try:
            # 1. 加载原始数据
            logger.info(f"加载股票数据: {stock_code}")
            data = self.loader.load_stock_data(
                stock_code,
                start_date=start_date,
                end_date=end_date
            )
            
            if data.empty:
                logger.warning(f"股票数据为空: {stock_code}")
                return False
            
            # 2. 计算指标
            logger.info(f"计算技术指标: {stock_code}")
            data_with_indicators = self.calculate_indicators(data)
            
            # 3. 转换为 Qlib 格式
            logger.info(f"转换为 Qlib 格式: {stock_code}")
            
            # 添加 stock_code 列（Qlib MultiIndex 需要）
            data_with_indicators["stock_code"] = stock_code
            data_with_indicators = data_with_indicators.set_index(
                ["stock_code", data_with_indicators.index]
            )
            
            # 列名映射：close -> $close 等
            column_mapping = {
                "open": "$open",
                "high": "$high",
                "low": "$low",
                "close": "$close",
                "volume": "$volume",
            }
            data_with_indicators = data_with_indicators.rename(columns=column_mapping)
            
            # 4. 保存到 Qlib features 目录
            safe_code = stock_code.replace(".", "_")
            output_file = self.qlib_data_path / f"{safe_code}.parquet"
            
            logger.info(f"保存到: {output_file}")
            self.converter.save_qlib_data(data_with_indicators, output_file)
            
            logger.success(
                f"✅ {stock_code}: {len(data_with_indicators)} 行, "
                f"{len(data_with_indicators.columns)} 列"
            )
            return True
            
        except Exception as e:
            logger.error(f"❌ {stock_code} 预计算失败: {e}")
            return False

    def precompute_multiple(
        self,
        stock_codes: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> dict:
        """
        批量预计算多只股票
        
        Returns:
            {"success": [...], "failed": [...]}
        """
        success = []
        failed = []
        
        total = len(stock_codes)
        for i, stock_code in enumerate(stock_codes, 1):
            logger.info(f"[{i}/{total}] 处理: {stock_code}")
            
            if self.precompute_stock(stock_code, start_date, end_date):
                success.append(stock_code)
            else:
                failed.append(stock_code)
        
        logger.info(f"\n📊 完成: 成功 {len(success)}, 失败 {len(failed)}")
        if failed:
            logger.warning(f"失败列表: {failed}")
            
        return {"success": success, "failed": failed}


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="预计算技术指标到 Qlib features")
    parser.add_argument(
        "--stocks",
        nargs="+",
        help="股票代码列表（空格分隔）",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="处理所有可用股票",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help="开始日期 (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="结束日期 (YYYY-MM-DD)",
    )
    
    args = parser.parse_args()
    
    # 解析日期
    start_date = None
    end_date = None
    if args.start_date:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    if args.end_date:
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
    
    # 初始化预计算器
    precomputer = IndicatorPrecomputer()
    
    # 确定要处理的股票列表
    if args.all:
        # 扫描 data 目录获取所有股票
        data_dir = Path("backend/data")
        stock_files = list(data_dir.glob("*.parquet"))
        stock_codes = [f.stem for f in stock_files]
        logger.info(f"找到 {len(stock_codes)} 只股票")
    elif args.stocks:
        stock_codes = args.stocks
    else:
        parser.error("必须指定 --stocks 或 --all")
        return
    
    # 执行预计算
    result = precomputer.precompute_multiple(
        stock_codes,
        start_date=start_date,
        end_date=end_date,
    )
    
    # 输出结果
    print(f"\n✅ 成功: {len(result['success'])}")
    print(f"❌ 失败: {len(result['failed'])}")


if __name__ == "__main__":
    main()
