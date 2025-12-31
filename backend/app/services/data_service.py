"""
股票数据服务
实现与远端数据服务的集成和本地Parquet文件管理
"""

import asyncio
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any
from enum import Enum

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
from app.services.cache_service import cache_manager
from app.services.connection_pool import connection_pool_manager, PoolConfig


class RetryStrategy(Enum):
    """重试策略"""
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIXED_DELAY = "fixed_delay"


class FallbackStrategy(Enum):
    """降级策略"""
    LOCAL_DATA = "local_data"
    CACHED_DATA = "cached_data"
    MOCK_DATA = "mock_data"
    FAIL_FAST = "fail_fast"


class ServiceHealthLevel(Enum):
    """服务健康等级"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


class RetryConfig:
    """重试配置"""
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF,
        jitter: bool = True
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.strategy = strategy
        self.jitter = jitter
    
    def get_delay(self, attempt: int) -> float:
        """计算重试延迟时间"""
        if self.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = self.base_delay * (2 ** attempt)
        elif self.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = self.base_delay * (attempt + 1)
        else:  # FIXED_DELAY
            delay = self.base_delay
        
        # 限制最大延迟
        delay = min(delay, self.max_delay)
        
        # 添加随机抖动
        if self.jitter:
            delay = delay * (0.5 + random.random() * 0.5)
        
        return delay


class CircuitBreaker:
    """熔断器"""
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open
    
    def can_execute(self) -> bool:
        """检查是否可以执行请求"""
        if self.state == "closed":
            return True
        elif self.state == "open":
            if self.last_failure_time and \
               (datetime.now() - self.last_failure_time).total_seconds() > self.recovery_timeout:
                self.state = "half_open"
                return True
            return False
        else:  # half_open
            return True
    
    def record_success(self):
        """记录成功"""
        self.failure_count = 0
        self.state = "closed"
    
    def record_failure(self):
        """记录失败"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"


class StockDataService:
    """股票数据服务"""
    
    def __init__(self):
        self.remote_url = settings.REMOTE_DATA_SERVICE_URL
        self.timeout = settings.REMOTE_DATA_SERVICE_TIMEOUT
        self.data_path = Path(settings.PARQUET_DATA_PATH)
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        # 初始化连接池
        self._init_connection_pool()
        
        # 初始化缓存
        self.cache = cache_manager.get_cache('stock_data')
        
        # 错误处理和降级策略配置
        self.retry_config = RetryConfig(
            max_retries=3,
            base_delay=1.0,
            max_delay=30.0,
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF
        )
        
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60.0
        )
        
        self.fallback_strategies = [
            FallbackStrategy.LOCAL_DATA,
            FallbackStrategy.CACHED_DATA
        ]
        
        # 服务健康状态
        self.health_level = ServiceHealthLevel.HEALTHY
        self.last_health_check = None
        self.consecutive_failures = 0
        
        # 数据缓存（保留原有的内存缓存作为L1缓存）
        self._data_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_ttl = 300  # 5分钟缓存
    
    def _init_connection_pool(self):
        """初始化连接池"""
        try:
            # 创建HTTP连接池配置
            pool_config = PoolConfig(
                max_connections=20,
                min_connections=5,
                max_keepalive_connections=10,
                keepalive_expiry=30.0,
                timeout=self.timeout,
                retry_attempts=3,
                retry_delay=1.0
            )
            
            # 创建HTTP连接池
            asyncio.create_task(
                connection_pool_manager.create_http_pool('stock_data', pool_config)
            )
            
            logger.info("股票数据服务连接池初始化完成")
        except Exception as e:
            logger.error(f"连接池初始化失败: {e}")
            # 使用默认的httpx客户端作为后备
            self.client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
            )
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        # 获取HTTP连接池
        self.http_pool = await connection_pool_manager.get_http_pool('stock_data')
        if not self.http_pool:
            # 如果连接池不存在，创建一个
            pool_config = PoolConfig(
                max_connections=20,
                min_connections=5,
                max_keepalive_connections=10,
                keepalive_expiry=30.0,
                timeout=self.timeout
            )
            self.http_pool = await connection_pool_manager.create_http_pool('stock_data', pool_config)
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        # 连接池由管理器统一管理，这里不需要关闭
        pass
    
    def _update_health_level(self, success: bool):
        """更新服务健康等级"""
        if success:
            self.consecutive_failures = 0
            if self.health_level != ServiceHealthLevel.HEALTHY:
                logger.info("服务健康状态恢复正常")
                self.health_level = ServiceHealthLevel.HEALTHY
        else:
            self.consecutive_failures += 1
            
            if self.consecutive_failures >= 10:
                self.health_level = ServiceHealthLevel.CRITICAL
            elif self.consecutive_failures >= 5:
                self.health_level = ServiceHealthLevel.UNHEALTHY
            elif self.consecutive_failures >= 2:
                self.health_level = ServiceHealthLevel.DEGRADED
            
            logger.warning(f"服务健康等级: {self.health_level.value}, 连续失败: {self.consecutive_failures}")
    
    async def _execute_with_retry(self, func, *args, **kwargs):
        """带重试的执行函数"""
        last_exception = None
        
        for attempt in range(self.retry_config.max_retries + 1):
            try:
                # 检查熔断器状态
                if not self.circuit_breaker.can_execute():
                    raise Exception("熔断器开启，服务暂时不可用")
                
                result = await func(*args, **kwargs)
                
                # 记录成功
                self.circuit_breaker.record_success()
                self._update_health_level(True)
                
                return result
                
            except Exception as e:
                last_exception = e
                self.circuit_breaker.record_failure()
                self._update_health_level(False)
                
                if attempt < self.retry_config.max_retries:
                    delay = self.retry_config.get_delay(attempt)
                    logger.warning(f"请求失败，{delay:.2f}秒后重试 (第{attempt + 1}次): {e}")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"重试次数已用完，最终失败: {e}")
        
        raise last_exception
    
    async def _try_fallback_strategies(self, stock_code: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """尝试降级策略"""
        for strategy in self.fallback_strategies:
            try:
                if strategy == FallbackStrategy.LOCAL_DATA:
                    logger.info(f"尝试本地数据降级策略: {stock_code}")
                    df = await self.load_from_local(stock_code, start_date, end_date)
                    if df is not None and not df.empty:
                        logger.info(f"本地数据降级成功: {stock_code}")
                        return df
                
                elif strategy == FallbackStrategy.CACHED_DATA:
                    logger.info(f"尝试缓存数据降级策略: {stock_code}")
                    cached_data = self._get_cached_data(stock_code, start_date, end_date)
                    if cached_data is not None:
                        logger.info(f"缓存数据降级成功: {stock_code}")
                        return cached_data
                
                elif strategy == FallbackStrategy.MOCK_DATA:
                    logger.info(f"尝试模拟数据降级策略: {stock_code}")
                    mock_data = self._generate_mock_data(stock_code, start_date, end_date)
                    if mock_data is not None:
                        logger.warning(f"使用模拟数据: {stock_code}")
                        return mock_data
                        
            except Exception as e:
                logger.warning(f"降级策略 {strategy.value} 失败: {e}")
                continue
        
        logger.error(f"所有降级策略都失败: {stock_code}")
        return None
    
    def _get_cached_data(self, stock_code: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """获取缓存数据"""
        # 首先检查L2缓存（持久化缓存）
        cache_key = f"stock_data:{stock_code}:{start_date.date()}:{end_date.date()}"
        cached_data = self.cache.get(cache_key)
        
        if cached_data is not None:
            logger.info(f"从L2缓存获取数据: {stock_code}")
            return cached_data
        
        # 然后检查L1缓存（内存缓存）
        if cache_key in self._data_cache:
            cached_item = self._data_cache[cache_key]
            cache_time = cached_item.get('timestamp', datetime.min)
            
            # 检查缓存是否过期
            if (datetime.now() - cache_time).total_seconds() < self._cache_ttl:
                data = cached_item.get('data')
                # 将数据提升到L2缓存
                if data is not None:
                    self.cache.put(cache_key, data, ttl=self._cache_ttl)
                    logger.info(f"从L1缓存获取数据并提升到L2缓存: {stock_code}")
                return data
            else:
                # 清理过期的L1缓存
                del self._data_cache[cache_key]
        
        return None
    
    def _cache_data(self, stock_code: str, start_date: datetime, end_date: datetime, data: pd.DataFrame):
        """缓存数据"""
        cache_key = f"stock_data:{stock_code}:{start_date.date()}:{end_date.date()}"
        
        # 存储到L2缓存
        self.cache.put(cache_key, data.copy(), ttl=self._cache_ttl)
        
        # 存储到L1缓存
        self._data_cache[cache_key] = {
            'data': data.copy(),
            'timestamp': datetime.now()
        }
        
        # 限制L1缓存大小
        if len(self._data_cache) > 100:
            # 删除最旧的缓存项
            oldest_key = min(self._data_cache.keys(), 
                           key=lambda k: self._data_cache[k]['timestamp'])
            del self._data_cache[oldest_key]
        
        logger.info(f"数据已缓存到L1和L2缓存: {stock_code}")
    
    def _generate_mock_data(self, stock_code: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """生成模拟数据（仅在紧急情况下使用）"""
        try:
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            
            # 生成基于随机游走的模拟价格数据
            base_price = 100.0
            prices = [base_price]
            
            for _ in range(len(date_range) - 1):
                change = random.uniform(-0.05, 0.05)  # ±5%的随机变化
                new_price = prices[-1] * (1 + change)
                prices.append(max(new_price, 1.0))  # 确保价格不为负
            
            data = []
            for i, date in enumerate(date_range):
                price = prices[i]
                high = price * random.uniform(1.0, 1.03)
                low = price * random.uniform(0.97, 1.0)
                volume = random.randint(1000000, 10000000)
                
                data.append({
                    'stock_code': stock_code,
                    'date': date,
                    'open': price,
                    'high': high,
                    'low': low,
                    'close': price,
                    'volume': volume
                })
            
            return pd.DataFrame(data)
            
        except Exception as e:
            logger.error(f"生成模拟数据失败: {e}")
            return None
    
    async def check_remote_service_status(self) -> DataServiceStatus:
        """检查远端数据服务状态"""
        start_time = datetime.now()
        
        async def _check_health():
            # 使用连接池发送请求
            if hasattr(self, 'http_pool') and self.http_pool:
                async with self.http_pool.request('GET', f"{self.remote_url}/api/data/health") as response:
                    return response
            else:
                # 后备方案：使用默认客户端
                if not hasattr(self, 'client'):
                    self.client = httpx.AsyncClient(
                        timeout=httpx.Timeout(self.timeout),
                        limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
                    )
                response = await self.client.get(f"{self.remote_url}/api/data/health")
                return response
        
        try:
            response = await self._execute_with_retry(_check_health)
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            
            if response.status_code == 200:
                self.last_health_check = datetime.now()
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
        
        async def _fetch_data():
            params = {
                "stock_code": stock_code,
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d")
            }
            
            # 使用连接池发送请求
            if hasattr(self, 'http_pool') and self.http_pool:
                async with self.http_pool.request(
                    'GET', 
                    f"{self.remote_url}/api/stock/daily",
                    params=params
                ) as response:
                    if response.status_code != 200:
                        raise Exception(f"HTTP {response.status_code}: {response.text}")
                    return response
            else:
                # 后备方案：使用默认客户端
                if not hasattr(self, 'client'):
                    self.client = httpx.AsyncClient(
                        timeout=httpx.Timeout(self.timeout),
                        limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
                    )
                response = await self.client.get(
                    f"{self.remote_url}/api/stock/daily",
                    params=params
                )
                
                if response.status_code != 200:
                    raise Exception(f"HTTP {response.status_code}: {response.text}")
                
                return response
        
        try:
            # 使用重试机制获取数据
            response = await self._execute_with_retry(_fetch_data)
            
            data = response.json()
            
            if not data or "data" not in data:
                logger.warning(f"远端服务返回空数据: {stock_code}")
                return None
            
            # 转换为DataFrame
            df = pd.DataFrame(data["data"])
            
            # 数据验证和清理
            df = self._validate_and_clean_data(df, stock_code)
            
            if df is not None and not df.empty:
                # 缓存成功获取的数据
                self._cache_data(stock_code, start_date, end_date, df)
                logger.info(f"从远端获取数据成功: {stock_code}, {len(df)} 条记录")
            
            return df
        
        except Exception as e:
            logger.error(f"从远端获取数据失败 {stock_code}: {e}")
            
            # 尝试降级策略
            fallback_data = await self._try_fallback_strategies(stock_code, start_date, end_date)
            if fallback_data is not None:
                logger.info(f"降级策略成功获取数据: {stock_code}")
                return fallback_data
            
            return None
    
    def _validate_and_clean_data(self, df: pd.DataFrame, stock_code: str) -> Optional[pd.DataFrame]:
        """验证和清理数据"""
        try:
            if df.empty:
                logger.warning(f"数据为空: {stock_code}")
                return None
            
            # 确保必需的列存在
            required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                logger.error(f"数据缺少必需列 {missing_columns}: {stock_code}")
                return None
            
            # 数据类型转换
            df['date'] = pd.to_datetime(df['date'])
            df['stock_code'] = stock_code
            
            # 数值列转换
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 删除无效数据
            original_len = len(df)
            df = df.dropna(subset=numeric_columns)
            
            if len(df) < original_len:
                logger.warning(f"清理了 {original_len - len(df)} 条无效数据: {stock_code}")
            
            # 数据合理性检查
            df = self._validate_price_data(df, stock_code)
            
            # 按日期排序
            df = df.sort_values('date')
            
            return df
            
        except Exception as e:
            logger.error(f"数据验证和清理失败 {stock_code}: {e}")
            return None
    
    def _validate_price_data(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        """验证价格数据的合理性"""
        try:
            # 检查价格是否为正数
            price_columns = ['open', 'high', 'low', 'close']
            for col in price_columns:
                invalid_mask = df[col] <= 0
                if invalid_mask.any():
                    logger.warning(f"发现非正价格数据: {stock_code}, 列: {col}, 数量: {invalid_mask.sum()}")
                    df = df[~invalid_mask]
            
            # 检查高低价关系
            invalid_high_low = df['high'] < df['low']
            if invalid_high_low.any():
                logger.warning(f"发现高价低于低价的数据: {stock_code}, 数量: {invalid_high_low.sum()}")
                df = df[~invalid_high_low]
            
            # 检查开盘收盘价是否在高低价范围内
            invalid_open = (df['open'] > df['high']) | (df['open'] < df['low'])
            invalid_close = (df['close'] > df['high']) | (df['close'] < df['low'])
            
            if invalid_open.any():
                logger.warning(f"发现开盘价超出高低价范围: {stock_code}, 数量: {invalid_open.sum()}")
                df = df[~invalid_open]
            
            if invalid_close.any():
                logger.warning(f"发现收盘价超出高低价范围: {stock_code}, 数量: {invalid_close.sum()}")
                df = df[~invalid_close]
            
            # 检查成交量
            invalid_volume = df['volume'] < 0
            if invalid_volume.any():
                logger.warning(f"发现负成交量: {stock_code}, 数量: {invalid_volume.sum()}")
                df = df[~invalid_volume]
            
            # 检查异常价格波动（单日涨跌幅超过50%）
            if len(df) > 1:
                df_sorted = df.sort_values('date')
                price_change = df_sorted['close'].pct_change().abs()
                extreme_change = price_change > 0.5
                
                if extreme_change.any():
                    logger.warning(f"发现极端价格波动: {stock_code}, 数量: {extreme_change.sum()}")
                    # 可以选择保留或删除这些数据，这里选择保留但记录警告
            
            return df
            
        except Exception as e:
            logger.error(f"价格数据验证失败 {stock_code}: {e}")
            return df
    
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
    
    async def load_from_local(
        self, 
        stock_code: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """从本地Parquet文件加载数据"""
        return self.load_from_parquet(stock_code, start_date, end_date)
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
        获取股票数据（智能降级策略）
        """
        logger.info(f"获取股票数据: {stock_code}, {start_date} - {end_date}")
        
        # 根据服务健康等级调整策略
        if self.health_level == ServiceHealthLevel.CRITICAL:
            logger.warning(f"服务处于严重状态，优先使用本地数据: {stock_code}")
            force_remote = False
        elif self.health_level == ServiceHealthLevel.UNHEALTHY:
            logger.warning(f"服务不健康，降低远端请求优先级: {stock_code}")
        
        # 检查本地数据是否存在
        if not force_remote:
            data_exists, missing_years = self.check_local_data_exists(
                stock_code, start_date, end_date
            )
            
            if data_exists:
                # 本地数据完整，直接加载
                df = await self.load_from_local(stock_code, start_date, end_date)
                if df is not None and not df.empty:
                    logger.info(f"使用本地数据: {stock_code}")
                    return self._dataframe_to_stock_data(df)
        
        # 本地数据不完整或强制从远端获取，尝试从远端获取
        logger.info(f"尝试从远端获取数据: {stock_code}")
        
        # 检查远端服务状态
        service_status = await self.check_remote_service_status()
        
        if not service_status.is_available:
            logger.warning(f"远端服务不可用: {service_status.error_message}")
            
            # 尝试降级策略
            fallback_data = await self._try_fallback_strategies(stock_code, start_date, end_date)
            if fallback_data is not None:
                return self._dataframe_to_stock_data(fallback_data)
            else:
                logger.error(f"所有降级策略都失败: {stock_code}")
                return None
        
        # 从远端获取数据（已包含重试和降级逻辑）
        df = await self.fetch_remote_data(stock_code, start_date, end_date)
        
        if df is None or df.empty:
            logger.error(f"获取数据失败: {stock_code}")
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