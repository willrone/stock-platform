"""
因子缓存管理模块

提供因子计算结果的两层缓存机制：
1. 内存缓存：快速访问，容量有限
2. 磁盘缓存：持久化存储，使用 Parquet 格式
"""

import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger


class FactorCache:
    """因子计算结果缓存 - 优化版

    支持两层缓存架构：
    - 内存缓存：使用字典存储，支持 LRU 淘汰策略
    - 磁盘缓存：使用 Parquet 格式存储，支持自动清理

    Attributes:
        cache_dir: 磁盘缓存目录
        max_cache_size: 最大磁盘缓存文件数
        default_ttl: 默认缓存过期时间
        memory_cache: 内存缓存字典
        max_memory_cache_size: 最大内存缓存项数
    """

    def __init__(self, cache_dir: str = "./data/qlib_cache"):
        """初始化因子缓存

        Args:
            cache_dir: 缓存目录路径
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 缓存配置
        self.max_cache_size = 50  # 最大缓存文件数
        self.default_ttl = timedelta(hours=24)  # 默认缓存过期时间

        # 内存缓存层
        self.memory_cache: Dict[str, Dict[str, Any]] = {}
        self.max_memory_cache_size = 10  # 最大内存缓存项数
        self.memory_cache_stats = {"hits": 0, "misses": 0, "evictions": 0}

        logger.info(f"因子缓存初始化: {self.cache_dir}, 内存缓存大小: {self.max_memory_cache_size}")

    def get_cache_key(
        self, stock_codes: List[str], date_range: Tuple[datetime, datetime]
    ) -> str:
        """生成缓存键

        使用股票代码的哈希值和日期范围生成唯一的缓存键。
        对股票代码排序以确保相同股票集合生成相同的缓存键。

        Args:
            stock_codes: 股票代码列表
            date_range: 日期范围元组 (start_date, end_date)

        Returns:
            缓存键字符串
        """
        # 对股票代码排序，确保相同股票集合生成相同的缓存键
        sorted_codes = sorted(stock_codes)
        codes_str = "_".join(sorted_codes)
        # 使用更高效的哈希算法
        codes_hash = hashlib.sha1(codes_str.encode()).hexdigest()[:12]
        start_str = date_range[0].strftime("%Y%m%d")
        end_str = date_range[1].strftime("%Y%m%d")
        return f"alpha_{codes_hash}_{start_str}_{end_str}"

    def get_cached_factors(self, cache_key: str) -> Optional[pd.DataFrame]:
        """获取缓存的因子数据

        优先从内存缓存获取，如果内存缓存未命中则从磁盘缓存获取。
        从磁盘加载的数据会自动添加到内存缓存。

        Args:
            cache_key: 缓存键

        Returns:
            缓存的因子数据，如果未命中返回 None
        """
        # 1. 先从内存缓存获取
        if cache_key in self.memory_cache:
            cache_item = self.memory_cache[cache_key]
            factors = cache_item["data"]
            timestamp = cache_item["timestamp"]

            # 检查内存缓存是否过期
            if datetime.now() - timestamp < self.default_ttl:
                self.memory_cache_stats["hits"] += 1
                logger.debug(f"内存缓存命中: {cache_key}, 数据量: {len(factors)}")
                return factors
            else:
                # 内存缓存过期，删除
                del self.memory_cache[cache_key]
                self.memory_cache_stats["misses"] += 1
                logger.debug(f"内存缓存过期: {cache_key}")
        else:
            self.memory_cache_stats["misses"] += 1

        # 2. 从磁盘缓存获取
        cache_file = self.cache_dir / f"{cache_key}.parquet"
        if cache_file.exists():
            try:
                # 检查文件是否过期
                file_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
                if datetime.now() - file_time > self.default_ttl:
                    logger.debug(f"磁盘缓存已过期: {cache_key}")
                    cache_file.unlink()
                    return None

                factors = pd.read_parquet(cache_file)
                logger.info(f"磁盘缓存命中: {cache_key}, 数据量: {len(factors)}")

                # 将数据加载到内存缓存
                self._add_to_memory_cache(cache_key, factors)

                return factors
            except Exception as e:
                logger.warning(f"读取磁盘缓存失败: {e}")
                # 删除损坏的缓存文件
                try:
                    cache_file.unlink()
                except Exception:
                    pass
        return None

    def save_factors(self, cache_key: str, factors: pd.DataFrame) -> None:
        """保存因子数据到缓存

        同时保存到内存缓存和磁盘缓存。

        Args:
            cache_key: 缓存键
            factors: 因子数据 DataFrame
        """
        try:
            # 1. 保存到内存缓存
            self._add_to_memory_cache(cache_key, factors)

            # 2. 保存到磁盘缓存
            cache_file = self.cache_dir / f"{cache_key}.parquet"
            # 优化：使用更快的压缩方式
            factors.to_parquet(cache_file, compression="snappy")

            # 清理旧缓存
            self._cleanup_old_cache()

            logger.info(f"因子数据缓存成功: {cache_key}, 数据量: {len(factors)}")
        except Exception as e:
            logger.warning(f"保存因子缓存失败: {e}")

    def _add_to_memory_cache(self, cache_key: str, factors: pd.DataFrame) -> None:
        """添加数据到内存缓存

        使用简单的 FIFO 淘汰策略，当缓存满时删除最早添加的项。

        Args:
            cache_key: 缓存键
            factors: 因子数据 DataFrame
        """
        # 检查内存缓存大小
        if len(self.memory_cache) >= self.max_memory_cache_size:
            # 删除最旧的缓存项
            oldest_key = next(iter(self.memory_cache))
            del self.memory_cache[oldest_key]
            self.memory_cache_stats["evictions"] += 1
            logger.debug(f"内存缓存淘汰: {oldest_key}")

        # 添加到内存缓存
        self.memory_cache[cache_key] = {"data": factors, "timestamp": datetime.now()}

    def _cleanup_old_cache(self) -> None:
        """清理旧缓存文件

        当磁盘缓存文件数超过最大限制时，按修改时间排序删除最旧的文件。
        """
        try:
            cache_files = list(self.cache_dir.glob("*.parquet"))
            if len(cache_files) <= self.max_cache_size:
                return

            # 按修改时间排序，删除最旧的文件
            cache_files.sort(key=lambda f: f.stat().st_mtime)
            files_to_remove = len(cache_files) - self.max_cache_size

            for i in range(files_to_remove):
                cache_files[i].unlink()
                logger.debug(f"删除旧缓存文件: {cache_files[i].name}")

        except Exception as e:
            logger.warning(f"清理缓存失败: {e}")

    def get_cache_stats(self) -> Dict[str, int]:
        """获取缓存统计信息

        Returns:
            包含缓存统计信息的字典，包括：
            - memory_cache_size: 当前内存缓存项数
            - disk_cache_size: 当前磁盘缓存文件数
            - memory_cache_hits: 内存缓存命中次数
            - memory_cache_misses: 内存缓存未命中次数
            - memory_cache_evictions: 内存缓存淘汰次数
            - max_memory_cache_size: 最大内存缓存项数
            - max_disk_cache_size: 最大磁盘缓存文件数
        """
        # 计算磁盘缓存文件数
        try:
            disk_cache_count = len(list(self.cache_dir.glob("*.parquet")))
        except Exception:
            disk_cache_count = 0

        return {
            "memory_cache_size": len(self.memory_cache),
            "disk_cache_size": disk_cache_count,
            "memory_cache_hits": self.memory_cache_stats["hits"],
            "memory_cache_misses": self.memory_cache_stats["misses"],
            "memory_cache_evictions": self.memory_cache_stats["evictions"],
            "max_memory_cache_size": self.max_memory_cache_size,
            "max_disk_cache_size": self.max_cache_size,
        }

    def clear_cache(self, memory_only: bool = False) -> None:
        """清除缓存

        Args:
            memory_only: 如果为 True，只清除内存缓存；否则同时清除磁盘缓存
        """
        # 清除内存缓存
        self.memory_cache.clear()
        self.memory_cache_stats = {"hits": 0, "misses": 0, "evictions": 0}
        logger.info("内存缓存已清除")

        # 清除磁盘缓存
        if not memory_only:
            try:
                for cache_file in self.cache_dir.glob("*.parquet"):
                    cache_file.unlink()
                logger.info("磁盘缓存已清除")
            except Exception as e:
                logger.warning(f"清除磁盘缓存失败: {e}")
