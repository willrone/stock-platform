"""Qlib Alpha 因子缓存模块。"""

import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger


class FactorCache:
    """Alpha 因子计算结果缓存。"""

    def __init__(self, cache_dir: str = "./data/qlib_cache") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_cache_size = 50
        self.default_ttl = timedelta(hours=24)
        self.memory_cache: dict[str, dict[str, object]] = {}
        self.max_memory_cache_size = 10
        self.memory_cache_stats = {"hits": 0, "misses": 0, "evictions": 0}
        logger.info(f"因子缓存初始化: {self.cache_dir}, 内存缓存大小: {self.max_memory_cache_size}")

    def get_cache_key(
        self,
        stock_codes: list[str],
        date_range: tuple[datetime, datetime],
    ) -> str:
        """生成稳定缓存键。"""
        sorted_codes = sorted(stock_codes)
        codes_str = "_".join(sorted_codes)
        codes_hash = hashlib.sha1(codes_str.encode()).hexdigest()[:12]
        start_str = date_range[0].strftime("%Y%m%d")
        end_str = date_range[1].strftime("%Y%m%d")
        return f"alpha_{codes_hash}_{start_str}_{end_str}"

    def get_cached_factors(self, cache_key: str) -> Optional[pd.DataFrame]:
        """获取缓存因子数据。"""
        memory_factors = self._get_from_memory_cache(cache_key)
        if memory_factors is not None:
            return memory_factors
        return self._get_from_disk_cache(cache_key)

    def _get_from_memory_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """从内存缓存读取因子数据。"""
        cache_item = self.memory_cache.get(cache_key)
        if cache_item is None:
            self.memory_cache_stats["misses"] += 1
            return None

        timestamp = cache_item["timestamp"]
        if not isinstance(timestamp, datetime):
            self.memory_cache.pop(cache_key, None)
            self.memory_cache_stats["misses"] += 1
            return None

        if datetime.now() - timestamp >= self.default_ttl:
            self.memory_cache.pop(cache_key, None)
            self.memory_cache_stats["misses"] += 1
            logger.debug(f"内存缓存过期: {cache_key}")
            return None

        factors = cache_item["data"]
        if isinstance(factors, pd.DataFrame):
            self.memory_cache_stats["hits"] += 1
            logger.debug(f"内存缓存命中: {cache_key}, 数据量: {len(factors)}")
            return factors

        self.memory_cache.pop(cache_key, None)
        self.memory_cache_stats["misses"] += 1
        return None

    def _get_from_disk_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """从磁盘缓存读取因子数据。"""
        cache_file = self.cache_dir / f"{cache_key}.parquet"
        if not cache_file.exists():
            return None

        try:
            file_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
            if datetime.now() - file_time > self.default_ttl:
                logger.debug(f"磁盘缓存已过期: {cache_key}")
                cache_file.unlink()
                return None

            factors = pd.read_parquet(cache_file)
            logger.info(f"磁盘缓存命中: {cache_key}, 数据量: {len(factors)}")
            self._add_to_memory_cache(cache_key, factors)
            return factors
        except Exception as exc:
            logger.warning(f"读取磁盘缓存失败: {exc}")
            try:
                cache_file.unlink()
            except OSError:
                pass
            return None

    def save_factors(self, cache_key: str, factors: pd.DataFrame) -> None:
        """保存因子数据到内存与磁盘缓存。"""
        try:
            self._add_to_memory_cache(cache_key, factors)
            cache_file = self.cache_dir / f"{cache_key}.parquet"
            factors.to_parquet(cache_file, compression="snappy")
            self._cleanup_old_cache()
            logger.info(f"因子数据缓存成功: {cache_key}, 数据量: {len(factors)}")
        except Exception as exc:
            logger.warning(f"保存因子缓存失败: {exc}")

    def _add_to_memory_cache(self, cache_key: str, factors: pd.DataFrame) -> None:
        """将因子数据写入内存缓存。"""
        if len(self.memory_cache) >= self.max_memory_cache_size:
            oldest_key = next(iter(self.memory_cache))
            del self.memory_cache[oldest_key]
            self.memory_cache_stats["evictions"] += 1
            logger.debug(f"内存缓存淘汰: {oldest_key}")

        self.memory_cache[cache_key] = {
            "data": factors,
            "timestamp": datetime.now(),
        }

    def _cleanup_old_cache(self) -> None:
        """清理超出上限的旧磁盘缓存。"""
        try:
            cache_files = list(self.cache_dir.glob("*.parquet"))
            if len(cache_files) <= self.max_cache_size:
                return

            cache_files.sort(key=lambda file_path: file_path.stat().st_mtime)
            files_to_remove = len(cache_files) - self.max_cache_size
            for cache_file in cache_files[:files_to_remove]:
                cache_file.unlink()
                logger.debug(f"删除旧缓存文件: {cache_file.name}")
        except Exception as exc:
            logger.warning(f"清理缓存失败: {exc}")

    def get_cache_stats(self) -> dict[str, int]:
        """获取缓存统计信息。"""
        try:
            disk_cache_count = len(list(self.cache_dir.glob("*.parquet")))
        except OSError:
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
        """清空缓存。"""
        self.memory_cache.clear()
        self.memory_cache_stats = {"hits": 0, "misses": 0, "evictions": 0}
        logger.info("内存缓存已清除")

        if memory_only:
            return

        try:
            for cache_file in self.cache_dir.glob("*.parquet"):
                cache_file.unlink()
            logger.info("磁盘缓存已清除")
        except Exception as exc:
            logger.warning(f"清除磁盘缓存失败: {exc}")
