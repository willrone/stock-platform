"""
简化的股票数据服务
只提供基本的HTTP接口调用功能：检查连接状态和获取数据
"""

import httpx
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
            self.client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
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
                logger.error(f"获取股票数据失败: {stock_code}, HTTP {response.status_code}")
                return None
            
            data = response.json()
            if not data.get('success', False):
                logger.error(f"获取股票数据失败: {stock_code}, {data.get('error', '未知错误')}")
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
            
            logger.info(f"成功获取股票数据: {stock_code}, {len(stock_data_list)} 条记录")
            return stock_data_list
            
        except Exception as e:
            logger.error(f"获取股票数据异常: {stock_code}, {str(e)}")
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

