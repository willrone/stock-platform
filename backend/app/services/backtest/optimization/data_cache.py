"""
回测数据缓存模块

避免每个 Optuna trial 重复加载股票数据，显著提升优化效率
"""

import asyncio
import hashlib
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger


class BacktestDataCache:
    """
    回测数据缓存（单例模式）

    功能：
    1. 预加载股票数据到内存
    2. 缓存回测中间结果
    3. 线程安全的并发访问
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._data_cache: Dict[str, pd.DataFrame] = {}
        self._result_cache: Dict[str, Any] = {}
        self._cache_lock = threading.RLock()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "preloaded_stocks": 0,
            "cached_results": 0,
        }
        self._initialized = True
        logger.info("BacktestDataCache 初始化完成")

    @classmethod
    def get_instance(cls) -> "BacktestDataCache":
        """获取缓存单例"""
        return cls()

    def _make_data_key(
        self, stock_code: str, start_date: datetime, end_date: datetime
    ) -> str:
        """生成数据缓存键"""
        start_str = (
            start_date.strftime("%Y%m%d")
            if isinstance(start_date, datetime)
            else str(start_date)[:10]
        )
        end_str = (
            end_date.strftime("%Y%m%d")
            if isinstance(end_date, datetime)
            else str(end_date)[:10]
        )
        return f"data_{stock_code}_{start_str}_{end_str}"

    def _make_result_key(self, params: Dict[str, Any], context: str = "") -> str:
        """生成结果缓存键（基于参数哈希）"""
        # 将参数转换为稳定的字符串表示
        param_str = str(sorted(params.items()))
        hash_val = hashlib.md5(f"{context}_{param_str}".encode()).hexdigest()[:16]
        return f"result_{hash_val}"

    def preload_stock_data(
        self,
        stock_codes: List[str],
        start_date: datetime,
        end_date: datetime,
        data_loader: callable,
    ) -> int:
        """
        预加载股票数据到缓存

        Args:
            stock_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            data_loader: 数据加载函数，签名为 (stock_code, start_date, end_date) -> DataFrame

        Returns:
            成功加载的股票数量
        """
        loaded_count = 0

        with self._cache_lock:
            for code in stock_codes:
                cache_key = self._make_data_key(code, start_date, end_date)

                if cache_key in self._data_cache:
                    logger.debug(f"股票 {code} 数据已在缓存中")
                    continue

                try:
                    data = data_loader(code, start_date, end_date)
                    if data is not None and not data.empty:
                        self._data_cache[cache_key] = data
                        loaded_count += 1
                        logger.debug(f"预加载股票 {code} 数据成功，{len(data)} 条记录")
                except Exception as e:
                    logger.warning(f"预加载股票 {code} 数据失败: {e}")

            self._stats["preloaded_stocks"] = len(self._data_cache)

        logger.info(f"预加载完成: {loaded_count}/{len(stock_codes)} 只股票")
        return loaded_count

    async def preload_async(
        self,
        stock_codes: List[str],
        start_date: datetime,
        end_date: datetime,
        data_loader: Optional[callable] = None,
    ) -> int:
        """异步预加载（供优化器使用）。

        `StrategyHyperparameterOptimizer` 只调用 `preload_async(stock_codes, start_date, end_date)`。
        为了保持兼容：
        - 如果提供 data_loader，则在后台线程中执行同步预加载。
        - 如果未提供 data_loader，则作为 no-op（返回 0），让优化流程继续走。
          （数据加载会由回测执行器在后续步骤自行完成，只是少了预热缓存。）
        """
        if data_loader is None:
            logger.info("preload_async: 未提供 data_loader，跳过预加载（兼容模式）")
            return 0

        return await asyncio.to_thread(
            self.preload_stock_data, stock_codes, start_date, end_date, data_loader
        )

    def get_stock_data(
        self,
        stock_code: str,
        start_date: datetime,
        end_date: datetime,
        data_loader: Optional[callable] = None,
    ) -> Optional[pd.DataFrame]:
        """
        获取股票数据（优先从缓存）

        Args:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            data_loader: 可选的数据加载函数（缓存未命中时使用）

        Returns:
            股票数据 DataFrame 或 None
        """
        cache_key = self._make_data_key(stock_code, start_date, end_date)

        with self._cache_lock:
            if cache_key in self._data_cache:
                self._stats["hits"] += 1
                return self._data_cache[cache_key].copy()

            self._stats["misses"] += 1

            # 缓存未命中，尝试加载
            if data_loader is not None:
                try:
                    data = data_loader(stock_code, start_date, end_date)
                    if data is not None and not data.empty:
                        self._data_cache[cache_key] = data
                        return data.copy()
                except Exception as e:
                    logger.warning(f"加载股票 {stock_code} 数据失败: {e}")

        return None

    def cache_result(
        self, params: Dict[str, Any], result: Any, context: str = ""
    ) -> None:
        """
        缓存回测结果

        Args:
            params: 策略参数
            result: 回测结果
            context: 上下文标识（如策略名称）
        """
        cache_key = self._make_result_key(params, context)

        with self._cache_lock:
            self._result_cache[cache_key] = result
            self._stats["cached_results"] = len(self._result_cache)

    def get_cached_result(
        self, params: Dict[str, Any], context: str = ""
    ) -> Optional[Any]:
        """
        获取缓存的回测结果

        Args:
            params: 策略参数
            context: 上下文标识

        Returns:
            缓存的结果或 None
        """
        cache_key = self._make_result_key(params, context)

        with self._cache_lock:
            return self._result_cache.get(cache_key)

    def clear(self, clear_data: bool = True, clear_results: bool = True) -> None:
        """
        清空缓存

        Args:
            clear_data: 是否清空数据缓存
            clear_results: 是否清空结果缓存
        """
        with self._cache_lock:
            if clear_data:
                self._data_cache.clear()
                self._stats["preloaded_stocks"] = 0
            if clear_results:
                self._result_cache.clear()
                self._stats["cached_results"] = 0

            self._stats["hits"] = 0
            self._stats["misses"] = 0

        logger.info(f"缓存已清空 (data={clear_data}, results={clear_results})")

    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        with self._cache_lock:
            total_requests = self._stats["hits"] + self._stats["misses"]
            hit_rate = (
                self._stats["hits"] / total_requests if total_requests > 0 else 0.0
            )

            return {
                **self._stats,
                "hit_rate": hit_rate,
                "data_cache_size": len(self._data_cache),
                "result_cache_size": len(self._result_cache),
            }

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"BacktestDataCache("
            f"stocks={stats['preloaded_stocks']}, "
            f"results={stats['cached_results']}, "
            f"hit_rate={stats['hit_rate']:.1%})"
        )


# 便捷函数
def get_data_cache() -> BacktestDataCache:
    """获取全局数据缓存实例"""
    return BacktestDataCache.get_instance()
