"""
任务状态通知服务 - 集成WebSocket实现任务状态的实时推送
"""

import asyncio
import json
from typing import Dict, List, Optional, Any, Set
from datetime import datetime
from dataclasses import dataclass, asdict
from loguru import logger

from app.services.infrastructure import websocket_manager, WebSocketMessage
from app.models.task_models import TaskStatus, TaskType
from app.core.error_handler import TaskError, ErrorSeverity, ErrorContext


@dataclass
class TaskStatusNotification:
    """任务状态通知"""
    task_id: str
    task_name: str
    task_type: str
    status: str
    progress: float
    user_id: str
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    estimated_remaining_seconds: Optional[int] = None
    current_step: Optional[str] = None
    result_summary: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


@dataclass
class TaskProgressNotification:
    """任务进度通知"""
    task_id: str
    progress: float
    current_step: str
    estimated_remaining_seconds: Optional[int] = None
    details: Optional[Dict[str, Any]] = None


class TaskNotificationService:
    """任务通知服务"""
    
    def __init__(self):
        self.websocket_manager = websocket_manager
        
        # 任务订阅管理
        self.task_subscribers: Dict[str, Set[str]] = {}  # task_id -> set(user_ids)
        self.user_subscriptions: Dict[str, Set[str]] = {}  # user_id -> set(task_ids)
        
        # 通知历史（用于离线用户）
        self.notification_history: Dict[str, List[TaskStatusNotification]] = {}
        self.max_history_per_user = 50
        
        # 通知统计
        self.notification_stats = {
            "total_sent": 0,
            "status_notifications": 0,
            "progress_notifications": 0,
            "failed_notifications": 0
        }
    
    async def notify_task_created(self, task_id: str, task_name: str, task_type: TaskType,
                                user_id: str, config: Dict[str, Any]):
        """通知任务创建"""
        try:
            notification = TaskStatusNotification(
                task_id=task_id,
                task_name=task_name,
                task_type=task_type.value,
                status=TaskStatus.CREATED.value,
                progress=0.0,
                user_id=user_id,
                created_at=datetime.utcnow().isoformat(),
                result_summary={"config_summary": self._summarize_config(task_type, config)}
            )
            
            await self._send_task_notification(notification, "task_created")
            
            # 自动订阅用户到任务通知
            await self.subscribe_user_to_task(user_id, task_id)
            
            logger.info(f"任务创建通知已发送: {task_id}")
            
        except Exception as e:
            logger.error(f"发送任务创建通知失败: {task_id}, 错误: {e}")
            self.notification_stats["failed_notifications"] += 1
    
    async def notify_task_status_change(self, task_id: str, task_name: str, task_type: str,
                                      old_status: str, new_status: str, progress: float,
                                      user_id: str, created_at: datetime,
                                      started_at: Optional[datetime] = None,
                                      completed_at: Optional[datetime] = None,
                                      result: Optional[Dict[str, Any]] = None,
                                      error_message: Optional[str] = None):
        """通知任务状态变化"""
        try:
            notification = TaskStatusNotification(
                task_id=task_id,
                task_name=task_name,
                task_type=task_type,
                status=new_status,
                progress=progress,
                user_id=user_id,
                created_at=created_at.isoformat(),
                started_at=started_at.isoformat() if started_at else None,
                completed_at=completed_at.isoformat() if completed_at else None,
                result_summary=self._summarize_result(result) if result else None,
                error_message=error_message
            )
            
            await self._send_task_notification(notification, "task_status_changed")
            
            # 如果任务完成或失败，发送特殊通知
            if new_status in [TaskStatus.COMPLETED.value, TaskStatus.FAILED.value]:
                await self._send_task_completion_notification(notification)
            
            logger.info(f"任务状态变化通知已发送: {task_id}, {old_status} -> {new_status}")
            
        except Exception as e:
            logger.error(f"发送任务状态变化通知失败: {task_id}, 错误: {e}")
            self.notification_stats["failed_notifications"] += 1
    
    async def notify_task_progress(self, task_id: str, progress: float, current_step: str,
                                 estimated_remaining_seconds: Optional[int] = None,
                                 details: Optional[Dict[str, Any]] = None):
        """通知任务进度更新"""
        try:
            notification = TaskProgressNotification(
                task_id=task_id,
                progress=progress,
                current_step=current_step,
                estimated_remaining_seconds=estimated_remaining_seconds,
                details=details
            )
            
            # 发送给订阅该任务的用户
            subscribers = self.task_subscribers.get(task_id, set())
            if subscribers:
                message = WebSocketMessage(
                    type="task_progress",
                    data={
                        "task_id": task_id,
                        "progress": progress,
                        "current_step": current_step,
                        "estimated_remaining_seconds": estimated_remaining_seconds,
                        "details": details,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                )
                
                # 发送给所有订阅用户
                for user_id in subscribers:
                    await self.websocket_manager.notify_user(user_id, "task_progress", message.data)
                
                self.notification_stats["progress_notifications"] += 1
                logger.debug(f"任务进度通知已发送: {task_id}, 进度: {progress:.1f}%, 订阅者: {len(subscribers)}")
            
        except Exception as e:
            logger.error(f"发送任务进度通知失败: {task_id}, 错误: {e}")
            self.notification_stats["failed_notifications"] += 1
    
    async def subscribe_user_to_task(self, user_id: str, task_id: str):
        """订阅用户到任务通知"""
        try:
            # 添加到任务订阅者列表
            if task_id not in self.task_subscribers:
                self.task_subscribers[task_id] = set()
            self.task_subscribers[task_id].add(user_id)
            
            # 添加到用户订阅列表
            if user_id not in self.user_subscriptions:
                self.user_subscriptions[user_id] = set()
            self.user_subscriptions[user_id].add(task_id)
            
            logger.debug(f"用户订阅任务通知: {user_id} -> {task_id}")
            
        except Exception as e:
            logger.error(f"订阅任务通知失败: {user_id} -> {task_id}, 错误: {e}")
    
    async def unsubscribe_user_from_task(self, user_id: str, task_id: str):
        """取消用户对任务的订阅"""
        try:
            # 从任务订阅者列表移除
            if task_id in self.task_subscribers:
                self.task_subscribers[task_id].discard(user_id)
                if not self.task_subscribers[task_id]:
                    del self.task_subscribers[task_id]
            
            # 从用户订阅列表移除
            if user_id in self.user_subscriptions:
                self.user_subscriptions[user_id].discard(task_id)
                if not self.user_subscriptions[user_id]:
                    del self.user_subscriptions[user_id]
            
            logger.debug(f"用户取消订阅任务通知: {user_id} -> {task_id}")
            
        except Exception as e:
            logger.error(f"取消订阅任务通知失败: {user_id} -> {task_id}, 错误: {e}")
    
    async def get_user_task_subscriptions(self, user_id: str) -> List[str]:
        """获取用户订阅的任务列表"""
        return list(self.user_subscriptions.get(user_id, set()))
    
    async def get_task_subscribers(self, task_id: str) -> List[str]:
        """获取任务的订阅者列表"""
        return list(self.task_subscribers.get(task_id, set()))
    
    async def send_system_notification(self, message: str, notification_type: str = "info",
                                     target_users: Optional[List[str]] = None):
        """发送系统通知"""
        try:
            notification_data = {
                "type": "system_notification",
                "notification_type": notification_type,
                "message": message,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            if target_users:
                # 发送给指定用户
                for user_id in target_users:
                    await self.websocket_manager.notify_user(user_id, "system_notification", notification_data)
            else:
                # 广播给所有用户
                await self.websocket_manager.broadcast_system_alert(
                    alert_type="system_notification",
                    message=message,
                    severity=notification_type
                )
            
            logger.info(f"系统通知已发送: {message}, 目标用户: {len(target_users) if target_users else '所有用户'}")
            
        except Exception as e:
            logger.error(f"发送系统通知失败: {e}")
            self.notification_stats["failed_notifications"] += 1
    
    async def _send_task_notification(self, notification: TaskStatusNotification, 
                                    notification_type: str):
        """发送任务通知"""
        try:
            # 添加到历史记录
            self._add_to_history(notification)
            
            # 发送WebSocket通知
            message_data = asdict(notification)
            message_data["notification_type"] = notification_type
            message_data["timestamp"] = datetime.utcnow().isoformat()
            
            # 发送给任务订阅者
            subscribers = self.task_subscribers.get(notification.task_id, set())
            if subscribers:
                for user_id in subscribers:
                    await self.websocket_manager.notify_user(user_id, "task_notification", message_data)
            
            # 总是发送给任务创建者
            await self.websocket_manager.notify_user(notification.user_id, "task_notification", message_data)
            
            self.notification_stats["status_notifications"] += 1
            self.notification_stats["total_sent"] += 1
            
        except Exception as e:
            logger.error(f"发送任务通知失败: {notification.task_id}, 错误: {e}")
            self.notification_stats["failed_notifications"] += 1
    
    async def _send_task_completion_notification(self, notification: TaskStatusNotification):
        """发送任务完成特殊通知"""
        try:
            completion_type = "task_completed" if notification.status == TaskStatus.COMPLETED.value else "task_failed"
            
            # 创建完成通知消息
            completion_message = {
                "task_id": notification.task_id,
                "task_name": notification.task_name,
                "task_type": notification.task_type,
                "status": notification.status,
                "completion_time": notification.completed_at,
                "duration": self._calculate_duration(notification),
                "result_summary": notification.result_summary,
                "error_message": notification.error_message,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # 发送给任务创建者
            await self.websocket_manager.notify_user(
                notification.user_id, 
                completion_type, 
                completion_message
            )
            
            logger.info(f"任务完成通知已发送: {notification.task_id}, 状态: {notification.status}")
            
        except Exception as e:
            logger.error(f"发送任务完成通知失败: {notification.task_id}, 错误: {e}")
    
    def _add_to_history(self, notification: TaskStatusNotification):
        """添加通知到历史记录"""
        user_id = notification.user_id
        
        if user_id not in self.notification_history:
            self.notification_history[user_id] = []
        
        self.notification_history[user_id].append(notification)
        
        # 限制历史记录数量
        if len(self.notification_history[user_id]) > self.max_history_per_user:
            self.notification_history[user_id] = self.notification_history[user_id][-self.max_history_per_user:]
    
    def _summarize_config(self, task_type: TaskType, config: Dict[str, Any]) -> Dict[str, Any]:
        """总结任务配置"""
        summary = {}
        
        if task_type == TaskType.PREDICTION:
            summary = {
                "stock_count": len(config.get('stock_codes', [])),
                "model_id": config.get('model_id', 'unknown'),
                "horizon": config.get('horizon', 'short_term')
            }
        elif task_type == TaskType.BACKTEST:
            summary = {
                "strategy": config.get('strategy_name', 'unknown'),
                "stock_count": len(config.get('stock_codes', [])),
                "period": f"{config.get('start_date', '')} - {config.get('end_date', '')}"
            }
        elif task_type == TaskType.TRAINING:
            summary = {
                "model_name": config.get('model_name', 'unknown'),
                "model_type": config.get('model_type', 'unknown'),
                "stock_count": len(config.get('stock_codes', []))
            }
        
        return summary
    
    def _summarize_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """总结任务结果"""
        summary = {}
        
        # 通用字段
        if 'execution_time' in result:
            summary['execution_time'] = f"{result['execution_time']:.1f}秒"
        
        # 预测任务结果
        if 'successful_predictions' in result:
            summary['successful_predictions'] = result['successful_predictions']
            summary['failed_predictions'] = result.get('failed_predictions', 0)
        
        # 回测任务结果
        if 'total_return' in result:
            summary['total_return'] = f"{result['total_return']:.2%}"
            summary['sharpe_ratio'] = f"{result.get('sharpe_ratio', 0):.2f}"
        
        # 训练任务结果
        if 'performance_metrics' in result:
            metrics = result['performance_metrics']
            summary['accuracy'] = f"{metrics.get('accuracy', 0):.3f}"
            summary['mse'] = f"{metrics.get('mse', 0):.4f}"
        
        return summary
    
    def _calculate_duration(self, notification: TaskStatusNotification) -> Optional[str]:
        """计算任务执行时长"""
        try:
            if notification.started_at and notification.completed_at:
                start_time = datetime.fromisoformat(notification.started_at)
                end_time = datetime.fromisoformat(notification.completed_at)
                duration = (end_time - start_time).total_seconds()
                
                if duration < 60:
                    return f"{duration:.1f}秒"
                elif duration < 3600:
                    return f"{duration/60:.1f}分钟"
                else:
                    return f"{duration/3600:.1f}小时"
            
            return None
            
        except Exception:
            return None
    
    async def get_user_notification_history(self, user_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """获取用户的通知历史"""
        try:
            history = self.notification_history.get(user_id, [])
            recent_history = history[-limit:] if len(history) > limit else history
            
            return [asdict(notification) for notification in reversed(recent_history)]
            
        except Exception as e:
            logger.error(f"获取用户通知历史失败: {user_id}, 错误: {e}")
            return []
    
    def get_notification_statistics(self) -> Dict[str, Any]:
        """获取通知统计信息"""
        return {
            **self.notification_stats,
            "active_subscriptions": len(self.task_subscribers),
            "users_with_subscriptions": len(self.user_subscriptions),
            "users_with_history": len(self.notification_history),
            "websocket_stats": self.websocket_manager.get_connection_stats()
        }
    
    async def cleanup_completed_task_subscriptions(self, task_id: str):
        """清理已完成任务的订阅"""
        try:
            # 移除任务订阅
            subscribers = self.task_subscribers.pop(task_id, set())
            
            # 从用户订阅中移除该任务
            for user_id in subscribers:
                if user_id in self.user_subscriptions:
                    self.user_subscriptions[user_id].discard(task_id)
                    if not self.user_subscriptions[user_id]:
                        del self.user_subscriptions[user_id]
            
            logger.debug(f"已清理任务订阅: {task_id}, 影响用户: {len(subscribers)}")
            
        except Exception as e:
            logger.error(f"清理任务订阅失败: {task_id}, 错误: {e}")


# 全局任务通知服务实例
task_notification_service = TaskNotificationService()