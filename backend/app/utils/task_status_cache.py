from typing import Optional

import redis.asyncio as redis

from app.core.config import settings


class TaskStatusCache:
    _redis: Optional[redis.Redis] = None

    @classmethod
    async def get_redis(cls) -> redis.Redis:
        if cls._redis is None:
            # 假设 settings.REDIS_URL 存在，否则用默认值
            url = getattr(settings, "REDIS_URL", "redis://localhost:6379/0")
            cls._redis = redis.from_url(url, encoding="utf-8", decode_responses=True)
        return cls._redis

    @classmethod
    async def is_task_running(cls, task_id: str) -> bool:
        """从 Redis 检查任务状态"""
        try:
            r = await cls.get_redis()
            status = await r.get(f"task:status:{task_id}")
            # 如果没有 key，说明可能还没缓存，暂且允许运行，或者依赖 DB 的定期同步
            if status is None:
                return True
            return status == "running"
        except Exception:
            return True

    @classmethod
    async def delete_status(cls, task_id: str) -> None:
        """删除任务状态缓存"""
        try:
            r = await cls.get_redis()
            await r.delete(f"task:status:{task_id}")
        except Exception:
            pass


# 导出单例实例
task_status_cache = TaskStatusCache()
