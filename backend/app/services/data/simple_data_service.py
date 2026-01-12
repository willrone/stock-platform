"""
简化的股票数据服务
只提供基本的HTTP接口调用功能：检查连接状态和获取数据
"""

import httpx
import pandas as pd
from datetime import datetime
from typing import List, Optional, Dict, Any
from loguru import logger

from app.core.config import settings
from app.models.stock import DataServiceStatus, StockData


class SimpleDataService:
    """简化的股票数据服务 - 只提供基本的HTTP调用功能"""
    
    def __init__(self):
        self.remote_url = settings.REMOTE_DATA_SERVICE_URL
        self.timeout = settings.REMOTE_DATA_SERVICE_TIMEOUT
        self.client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """获取HTTP客户端"""
        if self.client is None or self.client.is_closed:
            # 创建一个不使用代理的HTTP传输
            transport = httpx.AsyncHTTPTransport(proxy=None)
            self.client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
                transport=transport  # 使用不带代理的传输
            )
        return self.client
    
    async def check_remote_service_status(self) -> DataServiceStatus:
        """检查远端数据服务状态"""
        start_time = datetime.now()
        health_url = f"{self.remote_url}/api/data/health"
        
        try:
            client = await self._get_client()
            response = await client.get(health_url)
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
            error_msg = str(e)
            logger.warning(f"远端数据服务健康检查失败: {error_msg}")
            
            return DataServiceStatus(
                service_url=self.remote_url,
                is_available=False,
                last_check=start_time,
                response_time_ms=response_time,
                error_message=error_msg
            )
    
    async def get_stock_data(
        self,
        stock_code: str,
        start_date: datetime,
        end_date: datetime
    ) -> Optional[List[StockData]]:
        """获取股票数据，优先从本地parquet文件读取，如果本地没有则从远端服务获取"""
        try:
            # 优先从本地parquet文件读取
            from app.services.data.stock_data_loader import StockDataLoader
            from app.core.config import settings
            
            logger.info(f"尝试从本地加载股票数据: {stock_code}, 时间范围: {start_date.date()} 至 {end_date.date()}")
            loader = StockDataLoader(data_root=settings.DATA_ROOT_PATH)
            stock_df = loader.load_stock_data(stock_code, start_date=start_date, end_date=end_date)
            
            if not stock_df.empty and len(stock_df) > 0:
                # 从本地文件成功加载数据
                stock_data_list = []
                for date, row in stock_df.iterrows():
                    stock_data_list.append(StockData(
                        stock_code=stock_code,
                        date=date if isinstance(date, datetime) else pd.Timestamp(date).to_pydatetime(),
                        open=float(row.get('open', 0)),
                        high=float(row.get('high', 0)),
                        low=float(row.get('low', 0)),
                        close=float(row.get('close', 0)),
                        volume=int(row.get('volume', 0)),
                        adj_close=float(row.get('adj_close', row.get('close', 0))) if 'adj_close' in row else None
                    ))
                
                logger.info(f"从本地成功加载股票数据: {stock_code}, {len(stock_data_list)} 条记录")
                return stock_data_list
            
            # 本地没有数据，尝试从远端服务获取
            logger.info(f"本地无数据，尝试从远端服务获取: {stock_code}")
            return await self._get_stock_data_from_remote(stock_code, start_date, end_date)
            
        except Exception as e:
            logger.error(f"获取股票数据异常: {stock_code}, 错误类型: {type(e).__name__}, 错误信息: {str(e)}", exc_info=True)
            # 如果本地加载失败，尝试从远端获取
            try:
                logger.info(f"本地加载失败，尝试从远端服务获取: {stock_code}")
                return await self._get_stock_data_from_remote(stock_code, start_date, end_date)
            except Exception as remote_error:
                logger.error(f"从远端获取股票数据也失败: {stock_code}, {str(remote_error)}")
                return None
    
    async def _get_stock_data_from_remote(
        self,
        stock_code: str,
        start_date: datetime,
        end_date: datetime
    ) -> Optional[List[StockData]]:
        """从远端服务获取股票数据"""
        try:
            # 格式化日期
            start_date_str = start_date.strftime('%Y-%m-%d')
            end_date_str = end_date.strftime('%Y-%m-%d')
            
            # 调用远端API
            url = f"{self.remote_url}/api/data/stock/{stock_code}/daily"
            params = {
                'start_date': start_date_str,
                'end_date': end_date_str
            }
            
            client = await self._get_client()
            response = await client.get(url, params=params)
            
            if response.status_code != 200:
                logger.error(f"从远端获取股票数据失败: {stock_code}, HTTP {response.status_code}")
                return None
            
            data = response.json()
            if not data.get('success', False):
                logger.error(f"从远端获取股票数据失败: {stock_code}, {data.get('error', '未知错误')}")
                return None
            
            # 转换为StockData格式
            stock_data_list = []
            for item in data.get('data', []):
                stock_data_list.append(StockData(
                    stock_code=stock_code,
                    date=datetime.strptime(item['date'], '%Y-%m-%d'),
                    open=float(item['open']),
                    high=float(item['high']),
                    low=float(item['low']),
                    close=float(item['close']),
                    volume=int(item['volume'])
                ))
            
            logger.info(f"从远端成功获取股票数据: {stock_code}, {len(stock_data_list)} 条记录")
            return stock_data_list
            
        except Exception as e:
            logger.error(f"从远端获取股票数据异常: {stock_code}, 错误类型: {type(e).__name__}, 错误信息: {str(e)}", exc_info=True)
            return None
    
    async def get_remote_stock_list(self) -> Optional[List[Dict[str, Any]]]:
        """从远端服务获取股票列表"""
        try:
            url = f"{self.remote_url}/api/data/stock_data_status"
            client = await self._get_client()
            response = await client.get(url)
            
            if response.status_code != 200:
                logger.error(f"获取股票列表失败: HTTP {response.status_code}")
                return None
            
            data = response.json()
            stocks = data.get('stocks', [])
            
            logger.info(f"成功获取股票列表: {len(stocks)} 只股票")
            return stocks
            
        except Exception as e:
            logger.error(f"获取股票列表异常: {str(e)}")
            return None
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口 - 关闭HTTP客户端"""
        if self.client and not self.client.is_closed:
            await self.client.aclose()
            self.client = None

