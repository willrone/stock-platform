"""
数据同步事件管理器

提供数据同步完成后的回调机制，用于触发特征计算等后续处理
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from loguru import logger


class DataSyncEventType(Enum):
    """数据同步事件类型"""

    SYNC_STARTED = "sync_started"
    SYNC_COMPLETED = "sync_completed"
    SYNC_FAILED = "sync_failed"
    INCREMENTAL_UPDATE = "incremental_update"
    FULL_REFRESH = "full_refresh"


@dataclass
class DataSyncEvent:
    """数据同步事件"""

    event_type: DataSyncEventType
    stock_code: str
    date_range: Tuple[datetime, datetime]
    sync_type: str  # "incremental", "full", "manual"
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "event_type": self.event_type.value,
            "stock_code": self.stock_code,
            "date_range": [
                self.date_range[0].isoformat(),
                self.date_range[1].isoformat(),
            ],
            "sync_type": self.sync_type,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "error_message": self.error_message,
        }


class DataSyncEventManager:
    """数据同步事件管理器"""

    def __init__(self):
        # 事件监听器注册表
        self._listeners: Dict[DataSyncEventType, List[Callable]] = {
            event_type: [] for event_type in DataSyncEventType
        }

        # 事件历史记录
        self._event_history: List[DataSyncEvent] = []
        self._max_history_size = 1000

        # 统计信息
        self._stats = {
            "total_events": 0,
            "events_by_type": {event_type.value: 0 for event_type in DataSyncEventType},
            "last_event_time": None,
        }

        logger.info("数据同步事件管理器初始化完成")

    def register_listener(
        self, event_type: DataSyncEventType, callback: Callable[[DataSyncEvent], Any]
    ):
        """注册事件监听器"""
        if event_type not in self._listeners:
            self._listeners[event_type] = []

        self._listeners[event_type].append(callback)
        logger.info(f"注册事件监听器: {event_type.value}, 回调: {callback.__name__}")

    def unregister_listener(
        self, event_type: DataSyncEventType, callback: Callable[[DataSyncEvent], Any]
    ):
        """取消注册事件监听器"""
        if event_type in self._listeners and callback in self._listeners[event_type]:
            self._listeners[event_type].remove(callback)
            logger.info(f"取消注册事件监听器: {event_type.value}, 回调: {callback.__name__}")

    async def emit_event(self, event: DataSyncEvent):
        """发出事件"""
        try:
            # 记录事件
            self._event_history.append(event)
            if len(self._event_history) > self._max_history_size:
                self._event_history.pop(0)

            # 更新统计
            self._stats["total_events"] += 1
            self._stats["events_by_type"][event.event_type.value] += 1
            self._stats["last_event_time"] = event.timestamp.isoformat()

            logger.info(f"发出数据同步事件: {event.event_type.value}, 股票: {event.stock_code}")

            # 通知监听器
            listeners = self._listeners.get(event.event_type, [])
            if listeners:
                # 并发执行所有监听器
                tasks = []
                for listener in listeners:
                    try:
                        if asyncio.iscoroutinefunction(listener):
                            tasks.append(listener(event))
                        else:
                            # 同步函数在线程池中执行
                            loop = asyncio.get_event_loop()
                            tasks.append(loop.run_in_executor(None, listener, event))
                    except Exception as e:
                        logger.error(f"创建监听器任务失败 {listener.__name__}: {e}")

                if tasks:
                    # 等待所有监听器完成，但不让单个失败影响其他
                    results = await asyncio.gather(*tasks, return_exceptions=True)

                    for i, result in enumerate(results):
                        if isinstance(result, Exception):
                            listener_name = (
                                listeners[i].__name__
                                if i < len(listeners)
                                else "unknown"
                            )
                            logger.error(f"事件监听器执行失败 {listener_name}: {result}")
                        else:
                            logger.debug(f"事件监听器执行成功: {listeners[i].__name__}")

            logger.debug(f"事件处理完成: {event.event_type.value}, 监听器数量: {len(listeners)}")

        except Exception as e:
            logger.error(f"发出事件失败: {e}")

    async def emit_sync_started(
        self,
        stock_code: str,
        date_range: Tuple[datetime, datetime],
        sync_type: str = "manual",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """发出同步开始事件"""
        event = DataSyncEvent(
            event_type=DataSyncEventType.SYNC_STARTED,
            stock_code=stock_code,
            date_range=date_range,
            sync_type=sync_type,
            timestamp=datetime.now(),
            metadata=metadata,
        )
        await self.emit_event(event)

    async def emit_sync_completed(
        self,
        stock_code: str,
        date_range: Tuple[datetime, datetime],
        sync_type: str = "manual",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """发出同步完成事件"""
        event = DataSyncEvent(
            event_type=DataSyncEventType.SYNC_COMPLETED,
            stock_code=stock_code,
            date_range=date_range,
            sync_type=sync_type,
            timestamp=datetime.now(),
            metadata=metadata,
        )
        await self.emit_event(event)

    async def emit_sync_failed(
        self,
        stock_code: str,
        date_range: Tuple[datetime, datetime],
        sync_type: str = "manual",
        error_message: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """发出同步失败事件"""
        event = DataSyncEvent(
            event_type=DataSyncEventType.SYNC_FAILED,
            stock_code=stock_code,
            date_range=date_range,
            sync_type=sync_type,
            timestamp=datetime.now(),
            metadata=metadata,
            error_message=error_message,
        )
        await self.emit_event(event)

    async def emit_incremental_update(
        self,
        stock_code: str,
        date_range: Tuple[datetime, datetime],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """发出增量更新事件"""
        event = DataSyncEvent(
            event_type=DataSyncEventType.INCREMENTAL_UPDATE,
            stock_code=stock_code,
            date_range=date_range,
            sync_type="incremental",
            timestamp=datetime.now(),
            metadata=metadata,
        )
        await self.emit_event(event)

    async def emit_full_refresh(
        self,
        stock_code: str,
        date_range: Tuple[datetime, datetime],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """发出全量刷新事件"""
        event = DataSyncEvent(
            event_type=DataSyncEventType.FULL_REFRESH,
            stock_code=stock_code,
            date_range=date_range,
            sync_type="full",
            timestamp=datetime.now(),
            metadata=metadata,
        )
        await self.emit_event(event)

    def get_event_history(
        self,
        stock_code: Optional[str] = None,
        event_type: Optional[DataSyncEventType] = None,
        limit: int = 100,
    ) -> List[DataSyncEvent]:
        """获取事件历史"""
        events = self._event_history.copy()

        # 过滤条件
        if stock_code:
            events = [e for e in events if e.stock_code == stock_code]

        if event_type:
            events = [e for e in events if e.event_type == event_type]

        # 按时间倒序排列，返回最新的events
        events.sort(key=lambda x: x.timestamp, reverse=True)
        return events[:limit]

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self._stats,
            "registered_listeners": {
                event_type.value: len(listeners)
                for event_type, listeners in self._listeners.items()
            },
            "history_size": len(self._event_history),
        }

    def clear_history(self):
        """清空事件历史"""
        self._event_history.clear()
        logger.info("事件历史已清空")


# 全局事件管理器实例
_global_event_manager: Optional[DataSyncEventManager] = None


def get_data_sync_event_manager() -> DataSyncEventManager:
    """获取全局数据同步事件管理器"""
    global _global_event_manager
    if _global_event_manager is None:
        _global_event_manager = DataSyncEventManager()
    return _global_event_manager


def register_feature_pipeline_callbacks():
    """注册特征管道回调"""
    from ..features.feature_pipeline import FeaturePipeline

    event_manager = get_data_sync_event_manager()
    feature_pipeline = FeaturePipeline()

    async def on_sync_completed(event: DataSyncEvent):
        """同步完成回调"""
        try:
            await feature_pipeline.on_data_sync_complete(
                stock_code=event.stock_code,
                date_range=event.date_range,
                sync_type=event.sync_type,
            )
        except Exception as e:
            logger.error(f"特征管道回调执行失败: {e}")

    async def on_incremental_update(event: DataSyncEvent):
        """增量更新回调"""
        try:
            await feature_pipeline.on_data_sync_complete(
                stock_code=event.stock_code,
                date_range=event.date_range,
                sync_type="incremental",
            )
        except Exception as e:
            logger.error(f"特征管道增量更新回调执行失败: {e}")

    # 注册回调
    event_manager.register_listener(DataSyncEventType.SYNC_COMPLETED, on_sync_completed)
    event_manager.register_listener(
        DataSyncEventType.INCREMENTAL_UPDATE, on_incremental_update
    )
    event_manager.register_listener(DataSyncEventType.FULL_REFRESH, on_sync_completed)

    logger.info("特征管道回调注册完成")
