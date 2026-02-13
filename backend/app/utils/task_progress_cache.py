"""
任务进度内存缓存

将高频进度更新缓存在内存中，只在关键节点写入数据库，
减少 SQLite 并发写入压力。

关键写入节点：
- 任务开始（status=running）
- 每 10% 进度变化
- 任务完成/失败（status=completed/failed）
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from loguru import logger

# 进度写入间隔：每 10% 写一次 DB
PROGRESS_FLUSH_INTERVAL_PERCENT = 10.0
# 最小时间间隔（秒），避免短时间内多次写入
MIN_FLUSH_TIME_INTERVAL_S = 5.0


@dataclass
class CachedProgress:
    """单个任务的缓存进度数据"""

    task_id: str
    progress: float = 0.0
    result_data: Dict[str, Any] = field(default_factory=dict)
    last_flushed_progress: float = 0.0
    last_flush_time: float = 0.0
    is_dirty: bool = False


class TaskProgressCache:
    """
    任务进度内存缓存

    线程安全，支持多任务并发更新。
    只在关键节点（每 10% 进度、任务开始/结束）写入数据库。
    """

    def __init__(self):
        self._cache: Dict[str, CachedProgress] = {}
        self._lock = threading.Lock()

    def update_progress(
        self,
        task_id: str,
        progress: float,
        result_data: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        更新任务进度到内存缓存

        Args:
            task_id: 任务 ID
            progress: 当前进度百分比 (0-100)
            result_data: 附加的结果数据（如 progress_data）

        Returns:
            是否需要写入数据库（达到 10% 阈值）
        """
        with self._lock:
            cached = self._cache.get(task_id)
            if cached is None:
                cached = CachedProgress(task_id=task_id)
                self._cache[task_id] = cached

            cached.progress = progress
            if result_data is not None:
                cached.result_data = result_data
            cached.is_dirty = True

            return self._should_flush(cached)

    def get_progress(self, task_id: str) -> Optional[CachedProgress]:
        """
        从缓存获取任务进度

        Args:
            task_id: 任务 ID

        Returns:
            缓存的进度数据，不存在则返回 None
        """
        with self._lock:
            return self._cache.get(task_id)

    def mark_flushed(self, task_id: str) -> None:
        """标记任务进度已写入数据库"""
        with self._lock:
            cached = self._cache.get(task_id)
            if cached is not None:
                cached.last_flushed_progress = cached.progress
                cached.last_flush_time = time.monotonic()
                cached.is_dirty = False

    def remove(self, task_id: str) -> None:
        """移除任务缓存（任务完成/失败后调用）"""
        with self._lock:
            self._cache.pop(task_id, None)

    def get_all_dirty(self) -> Dict[str, CachedProgress]:
        """获取所有需要写入的脏数据（用于异常退出前的紧急刷盘）"""
        with self._lock:
            return {
                tid: cp for tid, cp in self._cache.items() if cp.is_dirty
            }

    def _should_flush(self, cached: CachedProgress) -> bool:
        """判断是否应该写入数据库"""
        progress_delta = cached.progress - cached.last_flushed_progress
        time_delta = time.monotonic() - cached.last_flush_time

        # 时间间隔太短，跳过
        if time_delta < MIN_FLUSH_TIME_INTERVAL_S:
            return False

        # 进度变化达到 10%
        if progress_delta >= PROGRESS_FLUSH_INTERVAL_PERCENT:
            return True

        return False


# 全局单例
task_progress_cache = TaskProgressCache()
