"""
股票数据服务
实现与远端数据服务的集成和本地Parquet文件管理
"""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import httpx
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger

from app.core.config import settings
from app.models.stock import (
    DataServiceStatus,
    DataSyncRequest,
    DataSyncResponse,
    StockData,
)


class StockDataService:
    """股票数据服务"""
    
    def __init__(self):
        self.remote_url = settings.REMOTE_DATA_SERVICE_URL
        self.timeout = settings.REMOTE_DATA_SERVICE_TIMEOUT
        self.data_path = Path(settings.PARQUET_DATA_PATH)
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        # HTTP客户端
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
        )
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.client.aclose()
    
    async def check_remote_service_status(self) -> DataServiceStatus:
        """检查远端数据服务状态"""
        start_time = datetime.now()
        
        try:
            response = await self.client.get(f"{self.remote_url}/health")
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            
            if response.status_code == 200:
                return DataServiceStatus(
                    service_url=self.remote_url,
                    is_available=True,
                    last_check=start_time,
                    response_time_ms=response_time
                )
            else:
                return DataServiceStatus(
                    service_url=self.remote_url,
                    is_available=False,
                    last_check=start_time,
                    response_time_ms=response_time,
                    error_message=f"HTTP {response.status_code}: {response.text}"
                )
        
        except Exception as e:
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            logger.error(f"远端数据服务检查失败: {e}")
            
            return DataServiceStatus(
                service_url=self.remote_url,
                is_available=False,
                last_check=start_time,
                response_time_ms=response_time,
                error_message=str(e)
            )
    
    def get_local_parquet_path(self, stock_code: str, year: int) -> Path:
        """获取本地Parquet文件路径"""
        stock_dir = self.data_path / "daily" / stock_code
        stock_dir.mkdir(parents=True, exist_ok=True)
        return stock_dir / f"{year}.parquet"
    
    def check_local_data_exists(
        self, 
        stock_code: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> tuple[bool, List[int]]:
        """
        检查本地数据是否存在
        返回: (是否完全存在, 缺失的年份列表)
        """
        years = list(range(start_date.year, end_date.year + 1))
        missing_years = []
        
        for year in years:
            parquet_path = self.get_local_parquet_path(stock_code, year)
            if not parquet_path.exists():
                missing_years.append(year)
                continue
            
            # 检查文件中的数据范围
            try:
                table = pq.read_table(parquet_path)
                df = table.to_pandas()
                
                if df.empty:
                    missing_years.append(year)
                    continue
                
                # 检查日期范围
                df['date'] = pd.to_datetime(df['date'])
                file_start = df['date'].min()
                file_end = df['date'].max()
                
                # 检查是否覆盖所需的日期范围
                year_start = datetime(year, 1, 1)
                year_end = datetime(year, 12, 31)
                
                if year == start_date.year:
                    year_start = start_date
                if year == end_date.year:
                    year_end = end_date
                
                if file_start > year_start or file_end < year_end:
                    missing_years.append(year)
            
            except Exception as e:
                logger.warning(f"检查本地文件失败 {parquet_path}: {e}")
                missing_years.append(year)
        
        return len(missing_years) == 0, missing_years
    
    async def fetch_remote_data(
        self, 
        stock_code: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """从远端服务获取股票数据"""
        try:
            params = {
                "stock_code": stock_code,
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d")
            }
            
            response = await self.client.get(
                f"{self.remote_url}/api/stock/daily",
                params=params
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if not data or "data" not in data:
                    logger.warning(f"远端服务返回空数据: {stock_code}")
                    return None
                
                # 转换为DataFrame
                df = pd.DataFrame(data["data"])
                
                # 数据验证和清理
                if df.empty:
                    logger.warning(f"远端服务返回空DataFrame: {stock_code}")
                    return None
                
                # 确保必需的列存在
                required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if missing_columns:
                    logger.error(f"远端数据缺少必需列 {missing_columns}: {stock_code}")
                    return None
                
                # 数据类型转换
                df['date'] = pd.to_datetime(df['date'])
                df['stock_code'] = stock_code
                
                # 数值列转换
                numeric_columns = ['open', 'high', 'low', 'close', 'volume']
                for col in numeric_columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # 删除无效数据
                df = df.dropna(subset=numeric_columns)
                
                # 按日期排序
                df = df.sort_values('date')
                
                logger.info(f"从远端获取数据成功: {stock_code}, {len(df)} 条记录")
                return df
            
            else:
                logger.error(f"远端服务请求失败: {response.status_code} - {response.text}")
                return None
        
        except Exception as e:
            logger.error(f"从远端获取数据失败 {stock_code}: {e}")
            return None
    
    def save_to_parquet(self, df: pd.DataFrame, stock_code: str) -> bool:
        """将数据保存为Parquet格式"""
        try:
            if df.empty:
                logger.warning(f"尝试保存空DataFrame: {stock_code}")
                return False
            
            # 按年份分组保存
            df['year'] = df['date'].dt.year
            
            for year, year_df in df.groupby('year'):
                parquet_path = self.get_local_parquet_path(stock_code, year)
                
                # 准备数据
                year_df = year_df.drop('year', axis=1)
                
                # 定义schema
                schema = pa.schema([
                    pa.field('stock_code', pa.string()),
                    pa.field('date', pa.timestamp('ns')),
                    pa.field('open', pa.float64()),
                    pa.field('high', pa.float64()),
                    pa.field('low', pa.float64()),
                    pa.field('close', pa.float64()),
                    pa.field('volume', pa.int64()),
                    pa.field('adj_close', pa.float64()),
                ])
                
                # 确保adj_close列存在
                if 'adj_close' not in year_df.columns:
                    year_df['adj_close'] = year_df['close']
                
                # 转换为Arrow表
                table = pa.Table.from_pandas(year_df, schema=schema)
                
                # 保存到Parquet文件
                pq.write_table(table, parquet_path, compression='snappy')
                
                logger.info(f"保存Parquet文件成功: {parquet_path}, {len(year_df)} 条记录")
            
            return True
        
        except Exception as e:
            logger.error(f"保存Parquet文件失败 {stock_code}: {e}")
            return False
    
    def load_from_parquet(
        self, 
        stock_code: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """从本地Parquet文件加载数据"""
        try:
            years = list(range(start_date.year, end_date.year + 1))
            dfs = []
            
            for year in years:
                parquet_path = self.get_local_parquet_path(stock_code, year)
                
                if not parquet_path.exists():
                    continue
                
                # 读取Parquet文件
                table = pq.read_table(parquet_path)
                year_df = table.to_pandas()
                
                if year_df.empty:
                    continue
                
                # 日期过滤
                year_df['date'] = pd.to_datetime(year_df['date'])
                year_df = year_df[
                    (year_df['date'] >= start_date) & 
                    (year_df['date'] <= end_date)
                ]
                
                if not year_df.empty:
                    dfs.append(year_df)
            
            if not dfs:
                return None
            
            # 合并所有年份的数据
            df = pd.concat(dfs, ignore_index=True)
            df = df.sort_values('date')
            
            logger.info(f"从本地加载数据成功: {stock_code}, {len(df)} 条记录")
            return df
        
        except Exception as e:
            logger.error(f"从本地加载数据失败 {stock_code}: {e}")
            return None
    
    async def get_stock_data(
        self, 
        stock_code: str, 
        start_date: datetime, 
        end_date: datetime,
        force_remote: bool = False
    ) -> Optional[List[StockData]]:
        """
        获取股票数据（本地优先策略）
        """
        logger.info(f"获取股票数据: {stock_code}, {start_date} - {end_date}")
        
        # 检查本地数据是否存在
        if not force_remote:
            data_exists, missing_years = self.check_local_data_exists(
                stock_code, start_date, end_date
            )
            
            if data_exists:
                # 本地数据完整，直接加载
                df = self.load_from_parquet(stock_code, start_date, end_date)
                if df is not None and not df.empty:
                    return self._dataframe_to_stock_data(df)
        
        # 本地数据不完整或强制从远端获取，尝试从远端获取
        logger.info(f"从远端获取数据: {stock_code}")
        
        # 检查远端服务状态
        service_status = await self.check_remote_service_status()
        
        if not service_status.is_available:
            logger.warning(f"远端服务不可用，尝试使用本地数据: {service_status.error_message}")
            
            # 尝试加载本地数据作为降级方案
            df = self.load_from_parquet(stock_code, start_date, end_date)
            if df is not None and not df.empty:
                return self._dataframe_to_stock_data(df)
            else:
                logger.error(f"远端服务不可用且本地无数据: {stock_code}")
                return None
        
        # 从远端获取数据
        df = await self.fetch_remote_data(stock_code, start_date, end_date)
        
        if df is None or df.empty:
            logger.warning(f"远端获取数据失败，尝试使用本地数据: {stock_code}")
            
            # 降级到本地数据
            df = self.load_from_parquet(stock_code, start_date, end_date)
            if df is not None and not df.empty:
                return self._dataframe_to_stock_data(df)
            else:
                return None
        
        # 保存到本地
        if self.save_to_parquet(df, stock_code):
            logger.info(f"数据保存成功: {stock_code}")
        else:
            logger.warning(f"数据保存失败: {stock_code}")
        
        return self._dataframe_to_stock_data(df)
    
    def _dataframe_to_stock_data(self, df: pd.DataFrame) -> List[StockData]:
        """将DataFrame转换为StockData列表"""
        stock_data_list = []
        
        for _, row in df.iterrows():
            stock_data = StockData(
                stock_code=row['stock_code'],
                date=row['date'],
                open=float(row['open']),
                high=float(row['high']),
                low=float(row['low']),
                close=float(row['close']),
                volume=int(row['volume']),
                adj_close=float(row.get('adj_close', row['close']))
            )
            stock_data_list.append(stock_data)
        
        return stock_data_list
    
    async def sync_multiple_stocks(
        self, 
        request: DataSyncRequest
    ) -> DataSyncResponse:
        """批量同步多只股票数据"""
        logger.info(f"开始批量同步: {len(request.stock_codes)} 只股票")
        
        # 默认日期范围
        end_date = request.end_date or datetime.now()
        start_date = request.start_date or (end_date - timedelta(days=365))
        
        synced_stocks = []
        failed_stocks = []
        total_records = 0
        
        # 并发控制
        semaphore = asyncio.Semaphore(3)  # 最多3个并发请求
        
        async def sync_single_stock(stock_code: str):
            async with semaphore:
                try:
                    data = await self.get_stock_data(
                        stock_code, 
                        start_date, 
                        end_date,
                        force_remote=request.force_update
                    )
                    
                    if data:
                        synced_stocks.append(stock_code)
                        return len(data)
                    else:
                        failed_stocks.append(stock_code)
                        return 0
                
                except Exception as e:
                    logger.error(f"同步股票失败 {stock_code}: {e}")
                    failed_stocks.append(stock_code)
                    return 0
        
        # 并发执行同步任务
        tasks = [sync_single_stock(code) for code in request.stock_codes]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 统计结果
        for result in results:
            if isinstance(result, int):
                total_records += result
        
        success = len(failed_stocks) == 0
        message = f"同步完成: 成功 {len(synced_stocks)}, 失败 {len(failed_stocks)}"
        
        logger.info(message)
        
        return DataSyncResponse(
            success=success,
            synced_stocks=synced_stocks,
            failed_stocks=failed_stocks,
            total_records=total_records,
            message=message
        )


# 全局数据服务实例
stock_data_service = StockDataService()