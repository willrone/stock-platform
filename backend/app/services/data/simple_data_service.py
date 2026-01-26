"""
简化的股票数据服务
只提供基本的HTTP接口调用功能：检查连接状态和获取数据
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import httpx
import pandas as pd
from loguru import logger

from app.core.config import settings
from app.models.stock import DataServiceStatus, StockData


class SimpleDataService:
    """简化的股票数据服务 - 只提供基本的HTTP调用功能"""

    def __init__(self):
        self.remote_url = settings.REMOTE_DATA_SERVICE_URL
        self.timeout = settings.REMOTE_DATA_SERVICE_TIMEOUT
        self.client: Optional[httpx.AsyncClient] = None
        self._cached_working_url: Optional[str] = None  # 缓存可用的URL

    async def _get_client(self) -> httpx.AsyncClient:
        """获取HTTP客户端"""
        if self.client is None or self.client.is_closed:
            # 创建一个不使用代理的HTTP传输
            transport = httpx.AsyncHTTPTransport(proxy=None)
            self.client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
                transport=transport,  # 使用不带代理的传输
            )
        return self.client

    def _extract_port_from_url(self, url: str) -> int:
        """从URL中提取端口号"""
        try:
            parsed = urlparse(url)
            return parsed.port or (443 if parsed.scheme == "https" else 5002)
        except Exception:
            return 5002

    async def _try_connect(
        self, base_url: str, path: str
    ) -> Tuple[bool, Optional[httpx.Response], str]:
        """尝试连接到指定的URL，返回 (是否成功, 响应对象, 错误信息)"""
        url = f"{base_url.rstrip('/')}/{path.lstrip('/')}"
        try:
            client = await self._get_client()
            response = await client.get(url, timeout=self.timeout)
            if response.status_code == 200:
                return True, response, ""
            return (
                False,
                response,
                f"HTTP {response.status_code}: {response.text[:200]}",
            )
        except httpx.TimeoutException as e:
            return False, None, f"连接超时: {str(e)}"
        except httpx.ConnectError as e:
            return False, None, f"连接错误: {str(e)}"
        except Exception as e:
            return False, None, f"未知错误: {str(e)}"

    async def _get_working_url(self, path: str = "api/data/health") -> Optional[str]:
        """
        获取可用的服务URL（本地优先，远程回退）

        Args:
            path: 要测试的API路径

        Returns:
            可用的URL，如果都失败则返回None
        """
        # 如果已经缓存了可用的URL，先尝试使用它
        if self._cached_working_url:
            success, _, _ = await self._try_connect(self._cached_working_url, path)
            if success:
                return self._cached_working_url
            self._cached_working_url = None  # 缓存失效，清除

        # 提取端口号
        port = self._extract_port_from_url(self.remote_url)

        # 构建要尝试的URL列表（本地优先）
        urls_to_try = [
            f"http://localhost:{port}",
            f"http://127.0.0.1:{port}",
        ]

        # 如果配置的URL不是localhost或127.0.0.1，添加到列表末尾
        parsed_remote = urlparse(self.remote_url)
        if parsed_remote.hostname not in ["localhost", "127.0.0.1"]:
            urls_to_try.append(self.remote_url)

        # 尝试每个URL
        for url in urls_to_try:
            success, _, _ = await self._try_connect(url, path)
            if success:
                logger.info(f"成功连接到数据服务: {url}")
                self._cached_working_url = url
                return url

        logger.error("所有连接尝试都失败")
        return None

    async def check_remote_service_status(self) -> DataServiceStatus:
        """检查远端数据服务状态（本地优先，远程回退）"""
        start_time = datetime.now()
        working_url = await self._get_working_url("api/data/health")

        if working_url is None:
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            return DataServiceStatus(
                service_url=self.remote_url,
                is_available=False,
                last_check=start_time,
                response_time_ms=response_time,
                error_message="无法连接到数据服务：所有连接尝试都失败",
            )

        try:
            success, response, error_msg = await self._try_connect(
                working_url, "api/data/health"
            )
            response_time = (datetime.now() - start_time).total_seconds() * 1000

            if success and response:
                logger.info(
                    f"数据服务健康检查成功，使用URL: {working_url}，响应时间: {response_time:.2f}ms"
                )
                return DataServiceStatus(
                    service_url=working_url,
                    is_available=True,
                    last_check=start_time,
                    response_time_ms=response_time,
                )

            return DataServiceStatus(
                service_url=working_url,
                is_available=False,
                last_check=start_time,
                response_time_ms=response_time,
                error_message=error_msg or "未知错误",
            )
        except Exception as e:
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            logger.error(
                f"远端数据服务健康检查异常: {e}，URL: {working_url}，响应时间: {response_time:.2f}ms",
                exc_info=True,
            )
            return DataServiceStatus(
                service_url=working_url or self.remote_url,
                is_available=False,
                last_check=start_time,
                response_time_ms=response_time,
                error_message=str(e),
            )

    async def get_stock_data(
        self, stock_code: str, start_date: datetime, end_date: datetime
    ) -> Optional[List[StockData]]:
        """获取股票数据，优先从本地parquet文件读取，失败则从远端服务获取"""
        try:
            from app.services.data.stock_data_loader import StockDataLoader

            loader = StockDataLoader(data_root=settings.DATA_ROOT_PATH)
            stock_df = loader.load_stock_data(
                stock_code, start_date=start_date, end_date=end_date
            )

            if not stock_df.empty:
                stock_data_list = []
                for date, row in stock_df.iterrows():
                    stock_data_list.append(
                        StockData(
                            stock_code=stock_code,
                            date=date
                            if isinstance(date, datetime)
                            else pd.Timestamp(date).to_pydatetime(),
                            open=float(row.get("open", 0)),
                            high=float(row.get("high", 0)),
                            low=float(row.get("low", 0)),
                            close=float(row.get("close", 0)),
                            volume=int(row.get("volume", 0)),
                            adj_close=float(row.get("adj_close", row.get("close", 0)))
                            if "adj_close" in row
                            else None,
                        )
                    )
                logger.info(f"从本地成功加载股票数据: {stock_code}, {len(stock_data_list)} 条记录")
                return stock_data_list
        except Exception as e:
            logger.debug(f"本地加载失败: {stock_code}, {e}")

        # 本地无数据或加载失败，尝试从远端获取
        return await self._get_stock_data_from_remote(stock_code, start_date, end_date)

    async def _get_stock_data_from_remote(
        self, stock_code: str, start_date: datetime, end_date: datetime
    ) -> Optional[List[StockData]]:
        """从远端服务获取股票数据"""
        try:
            working_url = await self._get_working_url("api/data/health")
            if working_url is None:
                return None

            full_url = f"{working_url.rstrip('/')}/api/data/stock/{stock_code}/daily"
            params = {
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
            }

            client = await self._get_client()
            response = await client.get(full_url, params=params)

            if response.status_code != 200:
                return None

            data = response.json()
            if not data.get("success", False):
                return None

            stock_data_list = [
                StockData(
                    stock_code=stock_code,
                    date=datetime.strptime(item["date"], "%Y-%m-%d"),
                    open=float(item["open"]),
                    high=float(item["high"]),
                    low=float(item["low"]),
                    close=float(item["close"]),
                    volume=int(item["volume"]),
                )
                for item in data.get("data", [])
            ]

            logger.info(f"从远端成功获取股票数据: {stock_code}, {len(stock_data_list)} 条记录")
            return stock_data_list
        except Exception as e:
            logger.error(f"从远端获取股票数据失败: {stock_code}, {e}", exc_info=True)
            return None

    async def get_remote_stock_list(self) -> Optional[List[Dict[str, Any]]]:
        """从远端服务获取股票列表"""
        try:
            working_url = await self._get_working_url("api/data/health")
            if working_url is None:
                return None

            full_url = f"{working_url.rstrip('/')}/api/data/stock_data_status"
            client = await self._get_client()
            response = await client.get(full_url)

            if response.status_code != 200:
                logger.error(f"获取股票列表失败: HTTP {response.status_code}")
                return None

            data = response.json()
            stocks = data.get("stocks", [])
            logger.info(f"成功获取股票列表: {len(stocks)} 只股票, URL: {working_url}")
            return stocks
        except (httpx.TimeoutException, httpx.ConnectError) as e:
            logger.error(f"获取股票列表连接错误: {e}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"获取股票列表异常: {e}", exc_info=True)
            return None

    async def __aenter__(self):
        """异步上下文管理器入口"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口 - 关闭HTTP客户端"""
        if self.client and not self.client.is_closed:
            await self.client.aclose()
            self.client = None
