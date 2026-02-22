"""
进程池任务执行器

使用ProcessPoolExecutor执行CPU密集型任务，避免阻塞主进程的API请求。
每个进程独立创建所需资源（数据库连接、服务实例等），不依赖全局变量。
"""

import asyncio
from concurrent.futures import Future, ProcessPoolExecutor
from typing import Any, Callable, Dict, Optional

from loguru import logger

from app.core.config import settings


class ProcessTaskExecutor:
    """进程池任务执行器"""

    def __init__(self, max_workers: Optional[int] = None):
        """
        初始化进程池执行器

        Args:
            max_workers: 最大工作进程数，默认使用配置中的PROCESS_POOL_SIZE
        """
        self.max_workers = max_workers or getattr(settings, "PROCESS_POOL_SIZE", 3)
        self.executor: Optional[ProcessPoolExecutor] = None
        self.is_running = False

        # 统计信息
        self.stats = {
            "total_submitted": 0,
            "total_completed": 0,
            "total_failed": 0,
            "active_tasks": 0,
        }

    def start(self):
        """启动进程池"""
        if self.is_running:
            logger.warning("进程池已经运行中")
            return

        self.executor = ProcessPoolExecutor(max_workers=self.max_workers)
        self.is_running = True
        logger.info(f"进程池执行器已启动，工作进程数: {self.max_workers}")

    def shutdown(self, wait: bool = True, timeout: Optional[float] = None):
        """关闭进程池"""
        if not self.is_running or not self.executor:
            return

        self.is_running = False

        # 先尝试优雅关闭
        try:
            # 取消所有未完成的任务
            if hasattr(self.executor, "_processes"):
                # 获取所有活跃的Future
                import concurrent.futures

                for future in concurrent.futures.as_completed([]):
                    try:
                        future.cancel()
                    except:
                        pass

            # 关闭进程池
            if wait:
                # 等待任务完成，但设置超时
                timeout or 30.0
                self.executor.shutdown(wait=True)
            else:
                # 不等待，立即关闭
                self.executor.shutdown(wait=False)
        except Exception as e:
            logger.warning(f"关闭进程池时出错: {e}")
            # 强制关闭
            try:
                self.executor.shutdown(wait=False)
            except:
                pass

        logger.info("进程池执行器已关闭")

    def submit(self, fn: Callable, *args, **kwargs) -> Future:
        """
        提交任务到进程池

        Args:
            fn: 要执行的函数（必须是可序列化的）
            *args: 位置参数
            **kwargs: 关键字参数

        Returns:
            Future对象，可用于获取结果
        """
        if not self.is_running or not self.executor:
            raise RuntimeError("进程池未启动，请先调用start()")

        self.stats["total_submitted"] += 1
        self.stats["active_tasks"] += 1

        future = self.executor.submit(fn, *args, **kwargs)

        # 添加回调来更新统计信息
        def on_done(f: Future):
            self.stats["active_tasks"] -= 1
            try:
                f.result()  # 获取结果，如果有异常会抛出
                self.stats["total_completed"] += 1
            except Exception as e:
                self.stats["total_failed"] += 1
                logger.error(f"任务执行失败: {e}")

        future.add_done_callback(on_done)

        logger.debug(f"任务已提交到进程池，当前活跃任务数: {self.stats['active_tasks']}")
        return future

    async def submit_async(
        self, fn: Callable, *args, timeout: Optional[float] = None, **kwargs
    ) -> Any:
        """
        异步提交任务到进程池并等待结果

        Args:
            fn: 要执行的函数
            *args: 位置参数
            timeout: 超时时间（秒），None 表示不限时。超时后抛出 TimeoutError。
            **kwargs: 关键字参数

        Returns:
            任务执行结果
        """
        loop = asyncio.get_event_loop()
        future = self.submit(fn, *args, **kwargs)

        if timeout is not None:
            # 使用 asyncio.wait_for 实现超时
            try:
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, future.result),
                    timeout=timeout,
                )
                return result
            except asyncio.TimeoutError:
                # 尝试取消 future
                future.cancel()
                raise TimeoutError(
                    f"子进程任务超时（{timeout}s），已取消。"
                    f"可能原因：子进程死锁或数据量过大。"
                )
        else:
            # 无超时，保持原有行为
            result = await loop.run_in_executor(None, future.result)
            return result

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self.stats,
            "max_workers": self.max_workers,
            "is_running": self.is_running,
        }


# 全局进程池执行器实例
_process_executor: Optional[ProcessTaskExecutor] = None


def get_process_executor() -> ProcessTaskExecutor:
    """获取全局进程池执行器实例"""
    global _process_executor
    if _process_executor is None:
        _process_executor = ProcessTaskExecutor()
    return _process_executor


def start_process_executor():
    """启动全局进程池执行器"""
    executor = get_process_executor()
    executor.start()


def shutdown_process_executor(wait: bool = True, timeout: Optional[float] = None):
    """关闭全局进程池执行器"""
    global _process_executor
    if _process_executor:
        _process_executor.shutdown(wait=wait, timeout=timeout)
        _process_executor = None
