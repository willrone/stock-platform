"""
内存缓存服务
实现LRU缓存机制和缓存失效策略
"""

import asyncio
import time
import threading
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, List, Tuple, Callable
from dataclasses import dataclass, field
from collections import OrderedDict
from enum import Enum
import weakref
import gc
import psutil
from loguru import logger


class CachePolicy(Enum):
    """缓存策略"""
    LRU = "lru"  # 最近最少使用
    LFU = "lfu"  # 最少使用频率
    TTL = "ttl"  # 基于时间过期
    FIFO = "fifo"  # 先进先出


@dataclass
class CacheEntry:
    """缓存条目"""
    key: str
    value: Any
    created_at: float
    last_accessed: float
    access_count: int = 0
    ttl: Optional[float] = None
    size_bytes: int = 0
    
    def is_expired(self) -> bool:
        """检查是否过期"""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl
    
    def touch(self):
        """更新访问时间和次数"""
        self.last_accessed = time.time()
        self.access_count += 1


@dataclass
class CacheStats:
    """缓存统计信息"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size: int = 0
    memory_usage_bytes: int = 0
    hit_rate: float = 0.0
    
    def update_hit_rate(self):
        """更新命中率"""
        total = self.hits + self.misses
        self.hit_rate = self.hits / total if total > 0 else 0.0


class LRUCache:
    """LRU缓存实现"""
    
    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: Optional[float] = None,
        memory_limit_mb: int = 100
    ):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.memory_limit_bytes = memory_limit_mb * 1024 * 1024
        
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._stats = CacheStats()
        
        # 内存监控
        self._memory_usage = 0
        self._cleanup_threshold = 0.8  # 80%内存使用率时开始清理
        
        # 启动后台清理任务
        self._cleanup_task = None
        self._start_cleanup_task()
    
    def _start_cleanup_task(self):
        """启动后台清理任务"""
        def cleanup_loop():
            while True:
                try:
                    self._cleanup_expired()
                    self._check_memory_usage()
                    time.sleep(60)  # 每分钟清理一次
                except Exception as e:
                    logger.error(f"缓存清理任务失败: {e}")
                    time.sleep(10)
        
        cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
        cleanup_thread.start()
    
    def _estimate_size(self, obj: Any) -> int:
        """估算对象大小"""
        try:
            import sys
            return sys.getsizeof(obj)
        except:
            return 1024  # 默认1KB
    
    def _cleanup_expired(self):
        """清理过期条目"""
        with self._lock:
            expired_keys = []
            for key, entry in self._cache.items():
                if entry.is_expired():
                    expired_keys.append(key)
            
            for key in expired_keys:
                self._remove_entry(key)
                self._stats.evictions += 1
            
            if expired_keys:
                logger.info(f"清理了 {len(expired_keys)} 个过期缓存条目")
    
    def _check_memory_usage(self):
        """检查内存使用情况"""
        if self._memory_usage > self.memory_limit_bytes * self._cleanup_threshold:
            self._evict_by_memory_pressure()
    
    def _evict_by_memory_pressure(self):
        """基于内存压力进行驱逐"""
        with self._lock:
            # 按访问时间排序，驱逐最久未访问的条目
            entries = list(self._cache.items())
            entries.sort(key=lambda x: x[1].last_accessed)
            
            target_size = int(self.memory_limit_bytes * 0.7)  # 清理到70%
            evicted_count = 0
            
            for key, entry in entries:
                if self._memory_usage <= target_size:
                    break
                
                self._remove_entry(key)
                evicted_count += 1
                self._stats.evictions += 1
            
            if evicted_count > 0:
                logger.info(f"内存压力清理了 {evicted_count} 个缓存条目")
    
    def _remove_entry(self, key: str):
        """移除缓存条目"""
        if key in self._cache:
            entry = self._cache.pop(key)
            self._memory_usage -= entry.size_bytes
            self._stats.size -= 1
    
    def _evict_lru(self):
        """驱逐最近最少使用的条目"""
        if self._cache:
            key, entry = self._cache.popitem(last=False)  # FIFO
            self._memory_usage -= entry.size_bytes
            self._stats.size -= 1
            self._stats.evictions += 1
            logger.debug(f"LRU驱逐缓存条目: {key}")
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        with self._lock:
            if key not in self._cache:
                self._stats.misses += 1
                self._stats.update_hit_rate()
                return None
            
            entry = self._cache[key]
            
            # 检查是否过期
            if entry.is_expired():
                self._remove_entry(key)
                self._stats.misses += 1
                self._stats.update_hit_rate()
                return None
            
            # 更新访问信息
            entry.touch()
            
            # 移动到末尾（最近使用）
            self._cache.move_to_end(key)
            
            self._stats.hits += 1
            self._stats.update_hit_rate()
            
            return entry.value
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """设置缓存值"""
        with self._lock:
            # 估算大小
            size_bytes = self._estimate_size(value)
            
            # 检查内存限制
            if size_bytes > self.memory_limit_bytes:
                logger.warning(f"缓存值太大，无法存储: {key}, 大小: {size_bytes} bytes")
                return False
            
            # 如果键已存在，先移除旧值
            if key in self._cache:
                self._remove_entry(key)
            
            # 确保有足够空间
            while (len(self._cache) >= self.max_size or 
                   self._memory_usage + size_bytes > self.memory_limit_bytes):
                if not self._cache:
                    break
                self._evict_lru()
            
            # 创建新条目
            now = time.time()
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=now,
                last_accessed=now,
                access_count=1,
                ttl=ttl or self.default_ttl,
                size_bytes=size_bytes
            )
            
            # 添加到缓存
            self._cache[key] = entry
            self._memory_usage += size_bytes
            self._stats.size += 1
            
            return True
    
    def delete(self, key: str) -> bool:
        """删除缓存条目"""
        with self._lock:
            if key in self._cache:
                self._remove_entry(key)
                return True
            return False
    
    def clear(self):
        """清空缓存"""
        with self._lock:
            self._cache.clear()
            self._memory_usage = 0
            self._stats.size = 0
    
    def get_stats(self) -> CacheStats:
        """获取缓存统计信息"""
        with self._lock:
            stats = CacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                evictions=self._stats.evictions,
                size=len(self._cache),
                memory_usage_bytes=self._memory_usage,
                hit_rate=self._stats.hit_rate
            )
            return stats
    
    def get_keys(self) -> List[str]:
        """获取所有键"""
        with self._lock:
            return list(self._cache.keys())
    
    def contains(self, key: str) -> bool:
        """检查键是否存在"""
        with self._lock:
            if key not in self._cache:
                return False
            
            entry = self._cache[key]
            if entry.is_expired():
                self._remove_entry(key)
                return False
            
            return True


class CacheManager:
    """缓存管理器"""
    
    def __init__(self):
        self._caches: Dict[str, LRUCache] = {}
        self._lock = threading.RLock()
        
        # 默认缓存配置
        self._default_configs = {
            'stock_data': {
                'max_size': 1000,
                'default_ttl': 3600,  # 1小时
                'memory_limit_mb': 200
            },
            'indicators': {
                'max_size': 500,
                'default_ttl': 1800,  # 30分钟
                'memory_limit_mb': 100
            },
            'api_responses': {
                'max_size': 200,
                'default_ttl': 300,  # 5分钟
                'memory_limit_mb': 50
            }
        }
    
    def get_cache(self, name: str) -> LRUCache:
        """获取或创建缓存实例"""
        with self._lock:
            if name not in self._caches:
                config = self._default_configs.get(name, {
                    'max_size': 100,
                    'default_ttl': 600,
                    'memory_limit_mb': 50
                })
                
                self._caches[name] = LRUCache(**config)
                logger.info(f"创建缓存实例: {name}, 配置: {config}")
            
            return self._caches[name]
    
    def get_global_stats(self) -> Dict[str, Any]:
        """获取全局缓存统计"""
        with self._lock:
            total_stats = {
                'total_caches': len(self._caches),
                'total_hits': 0,
                'total_misses': 0,
                'total_evictions': 0,
                'total_size': 0,
                'total_memory_usage_mb': 0,
                'overall_hit_rate': 0.0,
                'cache_details': {}
            }
            
            for name, cache in self._caches.items():
                stats = cache.get_stats()
                total_stats['total_hits'] += stats.hits
                total_stats['total_misses'] += stats.misses
                total_stats['total_evictions'] += stats.evictions
                total_stats['total_size'] += stats.size
                total_stats['total_memory_usage_mb'] += stats.memory_usage_bytes / (1024 * 1024)
                
                total_stats['cache_details'][name] = {
                    'hits': stats.hits,
                    'misses': stats.misses,
                    'evictions': stats.evictions,
                    'size': stats.size,
                    'memory_usage_mb': stats.memory_usage_bytes / (1024 * 1024),
                    'hit_rate': stats.hit_rate
                }
            
            # 计算总体命中率
            total_requests = total_stats['total_hits'] + total_stats['total_misses']
            if total_requests > 0:
                total_stats['overall_hit_rate'] = total_stats['total_hits'] / total_requests
            
            return total_stats
    
    def clear_all(self):
        """清空所有缓存"""
        with self._lock:
            for cache in self._caches.values():
                cache.clear()
            logger.info("清空了所有缓存")
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """获取内存使用情况"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'process_memory_mb': memory_info.rss / (1024 * 1024),
            'cache_memory_mb': sum(
                cache.get_stats().memory_usage_bytes / (1024 * 1024)
                for cache in self._caches.values()
            ),
            'system_memory_percent': psutil.virtual_memory().percent
        }


# 全局缓存管理器实例
cache_manager = CacheManager()


def cached(
    cache_name: str = 'default',
    ttl: Optional[float] = None,
    key_func: Optional[Callable] = None
):
    """缓存装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # 生成缓存键
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"
            
            # 尝试从缓存获取
            cache = cache_manager.get_cache(cache_name)
            result = cache.get(cache_key)
            
            if result is not None:
                return result
            
            # 执行函数并缓存结果
            result = func(*args, **kwargs)
            cache.put(cache_key, result, ttl)
            
            return result
        
        return wrapper
    return decorator


async def async_cached(
    cache_name: str = 'default',
    ttl: Optional[float] = None,
    key_func: Optional[Callable] = None
):
    """异步缓存装饰器"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # 生成缓存键
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"
            
            # 尝试从缓存获取
            cache = cache_manager.get_cache(cache_name)
            result = cache.get(cache_key)
            
            if result is not None:
                return result
            
            # 执行异步函数并缓存结果
            result = await func(*args, **kwargs)
            cache.put(cache_key, result, ttl)
            
            return result
        
        return wrapper
    return decorator