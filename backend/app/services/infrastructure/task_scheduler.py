"""
智能任务调度器
基于资源使用情况和任务优先级进行智能调度
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import heapq
import json
from uuid import uuid4

from .resource_monitor import ResourceMonitor, resource_monitor

logger = logging.getLogger(__name__)

class TaskPriority(Enum):
    """任务优先级"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class ResourceRequirement:
    """资源需求"""
    memory_gb: float = 0.0
    cpu_percent: float = 0.0
    gpu_memory_gb: float = 0.0
    estimated_duration_minutes: float = 60.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'memory_gb': self.memory_gb,
            'cpu_percent': self.cpu_percent,
            'gpu_memory_gb': self.gpu_memory_gb,
            'estimated_duration_minutes': self.estimated_duration_minutes
        }

@dataclass
class ScheduledTask:
    """调度任务"""
    task_id: str
    name: str
    priority: TaskPriority
    resource_requirement: ResourceRequirement
    task_func: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    scheduled_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    
    def __lt__(self, other):
        """用于优先队列排序"""
        # 优先级高的任务排在前面，创建时间早的排在前面
        if self.priority.value != other.priority.value:
            return self.priority.value > other.priority.value
        return self.created_at < other.created_at
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'task_id': self.task_id,
            'name': self.name,
            'priority': self.priority.name,
            'resource_requirement': self.resource_requirement.to_dict(),
            'created_at': self.created_at.isoformat(),
            'scheduled_at': self.scheduled_at.isoformat() if self.scheduled_at else None,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'status': self.status.value,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries,
            'error': self.error
        }

class TaskScheduler:
    """智能任务调度器"""
    
    def __init__(self, 
                 resource_monitor: ResourceMonitor,
                 max_concurrent_tasks: int = 3,
                 check_interval: float = 10.0):
        """
        初始化任务调度器
        
        Args:
            resource_monitor: 资源监控器
            max_concurrent_tasks: 最大并发任务数
            check_interval: 检查间隔（秒）
        """
        self.resource_monitor = resource_monitor
        self.max_concurrent_tasks = max_concurrent_tasks
        self.check_interval = check_interval
        
        # 任务队列（优先队列）
        self.pending_tasks: List[ScheduledTask] = []
        self.running_tasks: Dict[str, ScheduledTask] = {}
        self.completed_tasks: Dict[str, ScheduledTask] = {}
        
        # 调度器状态
        self._running = False
        self._scheduler_task: Optional[asyncio.Task] = None
        
        # 回调函数
        self.task_callbacks: Dict[str, List[Callable]] = {
            'task_started': [],
            'task_completed': [],
            'task_failed': [],
            'task_cancelled': []
        }
    
    def add_callback(self, event: str, callback: Callable):
        """添加事件回调函数"""
        if event in self.task_callbacks:
            self.task_callbacks[event].append(callback)
    
    def remove_callback(self, event: str, callback: Callable):
        """移除事件回调函数"""
        if event in self.task_callbacks and callback in self.task_callbacks[event]:
            self.task_callbacks[event].remove(callback)
    
    async def _notify_callbacks(self, event: str, task: ScheduledTask):
        """通知回调函数"""
        for callback in self.task_callbacks.get(event, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(task)
                else:
                    callback(task)
            except Exception as e:
                logger.error(f"任务回调函数执行失败: {e}")
    
    def schedule_task(self,
                     name: str,
                     task_func: Callable,
                     resource_requirement: ResourceRequirement,
                     priority: TaskPriority = TaskPriority.NORMAL,
                     args: tuple = (),
                     kwargs: dict = None,
                     max_retries: int = 3) -> str:
        """
        调度任务
        
        Args:
            name: 任务名称
            task_func: 任务函数
            resource_requirement: 资源需求
            priority: 任务优先级
            args: 任务参数
            kwargs: 任务关键字参数
            max_retries: 最大重试次数
            
        Returns:
            任务ID
        """
        task_id = str(uuid4())
        task = ScheduledTask(
            task_id=task_id,
            name=name,
            priority=priority,
            resource_requirement=resource_requirement,
            task_func=task_func,
            args=args,
            kwargs=kwargs or {},
            max_retries=max_retries
        )
        
        # 添加到优先队列
        heapq.heappush(self.pending_tasks, task)
        
        logger.info(f"任务已调度: {name} (ID: {task_id}, 优先级: {priority.name})")
        return task_id
    
    def cancel_task(self, task_id: str) -> bool:
        """
        取消任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            是否成功取消
        """
        # 检查待执行任务
        for i, task in enumerate(self.pending_tasks):
            if task.task_id == task_id:
                task.status = TaskStatus.CANCELLED
                self.pending_tasks.pop(i)
                heapq.heapify(self.pending_tasks)  # 重新堆化
                self.completed_tasks[task_id] = task
                logger.info(f"任务已取消: {task.name} (ID: {task_id})")
                return True
        
        # 检查正在运行的任务
        if task_id in self.running_tasks:
            task = self.running_tasks[task_id]
            task.status = TaskStatus.CANCELLED
            # 注意：这里不能直接取消正在运行的任务，需要任务函数自己检查状态
            logger.warning(f"任务正在运行，无法直接取消: {task.name} (ID: {task_id})")
            return False
        
        return False
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务状态"""
        # 检查待执行任务
        for task in self.pending_tasks:
            if task.task_id == task_id:
                return task.to_dict()
        
        # 检查正在运行的任务
        if task_id in self.running_tasks:
            return self.running_tasks[task_id].to_dict()
        
        # 检查已完成任务
        if task_id in self.completed_tasks:
            return self.completed_tasks[task_id].to_dict()
        
        return None
    
    def get_all_tasks(self) -> Dict[str, List[Dict[str, Any]]]:
        """获取所有任务状态"""
        return {
            'pending': [task.to_dict() for task in self.pending_tasks],
            'running': [task.to_dict() for task in self.running_tasks.values()],
            'completed': [task.to_dict() for task in self.completed_tasks.values()]
        }
    
    def _can_schedule_task(self, task: ScheduledTask) -> bool:
        """检查是否可以调度任务"""
        # 检查并发任务数限制
        if len(self.running_tasks) >= self.max_concurrent_tasks:
            return False
        
        # 检查资源可用性
        resource_check = self.resource_monitor.is_resource_available(
            required_memory_gb=task.resource_requirement.memory_gb,
            required_cpu_percent=task.resource_requirement.cpu_percent,
            required_gpu_memory_gb=task.resource_requirement.gpu_memory_gb
        )
        
        return resource_check['available']
    
    async def _execute_task(self, task: ScheduledTask):
        """执行任务"""
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()
        self.running_tasks[task.task_id] = task
        
        logger.info(f"开始执行任务: {task.name} (ID: {task.task_id})")
        await self._notify_callbacks('task_started', task)
        
        try:
            # 执行任务函数
            if asyncio.iscoroutinefunction(task.task_func):
                result = await task.task_func(*task.args, **task.kwargs)
            else:
                result = task.task_func(*task.args, **task.kwargs)
            
            # 任务成功完成
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            task.result = result
            
            logger.info(f"任务执行成功: {task.name} (ID: {task.task_id})")
            await self._notify_callbacks('task_completed', task)
            
        except Exception as e:
            # 任务执行失败
            task.error = str(e)
            task.retry_count += 1
            
            logger.error(f"任务执行失败: {task.name} (ID: {task.task_id}), 错误: {e}")
            
            # 检查是否需要重试
            if task.retry_count <= task.max_retries:
                logger.info(f"任务将重试: {task.name} (ID: {task.task_id}), 重试次数: {task.retry_count}/{task.max_retries}")
                task.status = TaskStatus.PENDING
                task.started_at = None
                # 重新加入待执行队列
                heapq.heappush(self.pending_tasks, task)
            else:
                task.status = TaskStatus.FAILED
                task.completed_at = datetime.now()
                logger.error(f"任务重试次数已用完，标记为失败: {task.name} (ID: {task.task_id})")
                await self._notify_callbacks('task_failed', task)
        
        finally:
            # 从运行任务中移除
            if task.task_id in self.running_tasks:
                del self.running_tasks[task.task_id]
            
            # 如果任务已完成或失败，添加到完成列表
            if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                self.completed_tasks[task.task_id] = task
    
    async def _scheduler_loop(self):
        """调度器主循环"""
        while self._running:
            try:
                # 检查是否有可以调度的任务
                while self.pending_tasks and self._can_schedule_task(self.pending_tasks[0]):
                    task = heapq.heappop(self.pending_tasks)
                    
                    # 检查任务是否被取消
                    if task.status == TaskStatus.CANCELLED:
                        continue
                    
                    # 创建任务执行协程
                    asyncio.create_task(self._execute_task(task))
                
                await asyncio.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"调度器循环出错: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def start(self):
        """启动调度器"""
        if self._running:
            logger.warning("任务调度器已经在运行")
            return
        
        self._running = True
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        logger.info("任务调度器已启动")
    
    async def stop(self):
        """停止调度器"""
        if not self._running:
            return
        
        self._running = False
        
        # 取消调度器任务
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
        
        # 等待所有运行中的任务完成
        if self.running_tasks:
            logger.info(f"等待 {len(self.running_tasks)} 个任务完成...")
            # 这里可以添加超时机制
            while self.running_tasks:
                await asyncio.sleep(1)
        
        logger.info("任务调度器已停止")
    
    def get_scheduler_stats(self) -> Dict[str, Any]:
        """获取调度器统计信息"""
        total_completed = len([t for t in self.completed_tasks.values() if t.status == TaskStatus.COMPLETED])
        total_failed = len([t for t in self.completed_tasks.values() if t.status == TaskStatus.FAILED])
        total_cancelled = len([t for t in self.completed_tasks.values() if t.status == TaskStatus.CANCELLED])
        
        return {
            'running': self._running,
            'pending_tasks': len(self.pending_tasks),
            'running_tasks': len(self.running_tasks),
            'completed_tasks': total_completed,
            'failed_tasks': total_failed,
            'cancelled_tasks': total_cancelled,
            'max_concurrent_tasks': self.max_concurrent_tasks,
            'check_interval': self.check_interval
        }

# 全局任务调度器实例
task_scheduler = TaskScheduler(resource_monitor)