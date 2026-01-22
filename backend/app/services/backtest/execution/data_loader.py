"""
数据加载器

负责加载回测所需的历史数据
"""

import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
from loguru import logger
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.core.error_handler import TaskError, ErrorSeverity, ErrorContext


class DataLoader:
    """数据加载器"""
    
    def __init__(self, data_dir: str = "backend/data", max_workers: Optional[int] = None):
        self.data_dir = Path(data_dir)
        self.max_workers = max_workers  # 用于并行加载数据
    
    def load_stock_data(self, stock_code: str, start_date: datetime, 
                       end_date: datetime) -> pd.DataFrame:
        """加载股票历史数据"""
        try:
            # 使用统一的数据加载器
            from app.services.data.stock_data_loader import StockDataLoader
            loader = StockDataLoader(data_root=str(self.data_dir))
            
            # 加载数据
            data = loader.load_stock_data(stock_code, start_date=start_date, end_date=end_date)
            
            if data.empty:
                raise TaskError(
                    message=f"未找到股票数据文件: {stock_code}",
                    severity=ErrorSeverity.HIGH,
                    context=ErrorContext(stock_code=stock_code)
                )
            
            if len(data) == 0:
                raise TaskError(
                    message=f"指定日期范围内无数据: {stock_code}, {start_date} - {end_date}",
                    severity=ErrorSeverity.MEDIUM,
                    context=ErrorContext(stock_code=stock_code)
                )
            
            # 验证必需的列
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                raise TaskError(
                    message=f"数据缺少必需列: {missing_columns}",
                    severity=ErrorSeverity.HIGH,
                    context=ErrorContext(stock_code=stock_code)
                )
            
            # 添加股票代码属性
            data.attrs['stock_code'] = stock_code
            
            logger.info(f"加载股票数据成功: {stock_code}, 数据量: {len(data)}, 日期范围: {data.index[0]} - {data.index[-1]}")
            return data
            
        except TaskError:
            raise
        except Exception as e:
            raise TaskError(
                message=f"加载股票数据失败: {str(e)}",
                severity=ErrorSeverity.HIGH,
                context=ErrorContext(stock_code=stock_code),
                original_exception=e
            )
    
    def load_multiple_stocks(self, stock_codes: List[str], start_date: datetime,
                           end_date: datetime, parallel: bool = True) -> Dict[str, pd.DataFrame]:
        """
        加载多只股票数据
        
        Args:
            stock_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            parallel: 是否并行加载（默认True）
        """
        stock_data = {}
        failed_stocks = []
        
        if parallel and len(stock_codes) > 1 and self.max_workers:
            # 并行加载多只股票数据
            max_workers = min(self.max_workers, len(stock_codes))
            logger.info(f"并行加载 {len(stock_codes)} 只股票数据，使用 {max_workers} 个线程")
            
            def load_single_stock(stock_code: str) -> Tuple[str, Optional[pd.DataFrame], Optional[str]]:
                """加载单只股票数据，返回 (stock_code, data, error)"""
                try:
                    data = self.load_stock_data(stock_code, start_date, end_date)
                    return (stock_code, data, None)
                except Exception as e:
                    error_msg = str(e)
                    logger.error(f"加载股票数据失败: {stock_code}, 错误: {error_msg}")
                    return (stock_code, None, error_msg)
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(load_single_stock, code): code for code in stock_codes}
                
                for future in as_completed(futures):
                    stock_code, data, error = future.result()
                    if data is not None:
                        stock_data[stock_code] = data
                    else:
                        failed_stocks.append(stock_code)
        else:
            # 顺序加载（兼容旧逻辑）
            for stock_code in stock_codes:
                try:
                    data = self.load_stock_data(stock_code, start_date, end_date)
                    stock_data[stock_code] = data
                except Exception as e:
                    logger.error(f"加载股票数据失败: {stock_code}, 错误: {e}")
                    failed_stocks.append(stock_code)
                    continue
        
        if failed_stocks:
            logger.warning(f"部分股票数据加载失败: {failed_stocks}")
        
        if not stock_data:
            raise TaskError(
                message="所有股票数据加载失败",
                severity=ErrorSeverity.HIGH
            )
        
        return stock_data
