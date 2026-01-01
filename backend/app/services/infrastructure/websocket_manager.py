"""
WebSocket连接管理器
"""

import json
import asyncio
from typing import Dict, List, Set, Optional, Any
from datetime import datetime
from fastapi import WebSocket, WebSocketDisconnect
from loguru import logger
from dataclasses import dataclass, asdict


@dataclass
class WebSocketMessage:
    """WebSocket消息格式"""
    type: str  # task_status, system_alert, notification
    data: Dict[str, Any]
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat()
    
    def to_json(self) -> str:
        return json.dumps(asdict(self))


@dataclass
class ClientConnection:
    """客户端连接信息"""
    websocket: WebSocket
    user_id: Optional[str] = None
    subscribed_tasks: Set[str] = None
    connected_at: datetime = None
    
    def __post_init__(self):
        if self.subscribed_tasks is None:
            self.subscribed_tasks = set()
        if self.connected_at is None:
            self.connected_at = datetime.utcnow()


class WebSocketManager:
    """WebSocket连接管理器"""
    
    def __init__(self):
        # 存储所有活跃连接 {connection_id: ClientConnection}
        self.active_connections: Dict[str, ClientConnection] = {}
        # 用户ID到连接ID的映射 {user_id: set(connection_ids)}
        self.user_connections: Dict[str, Set[str]] = {}
        # 任务订阅映射 {task_id: set(connection_ids)}
        self.task_subscriptions: Dict[str, Set[str]] = {}
        # 消息队列，用于离线消息
        self.message_queue: Dict[str, List[WebSocketMessage]] = {}
        
    async def connect(self, websocket: WebSocket, connection_id: str, user_id: Optional[str] = None) -> bool:
        """建立WebSocket连接"""
        try:
            await websocket.accept()
            
            connection = ClientConnection(
                websocket=websocket,
                user_id=user_id
            )
            
            self.active_connections[connection_id] = connection
            
            if user_id:
                if user_id not in self.user_connections:
                    self.user_connections[user_id] = set()
                self.user_connections[user_id].add(connection_id)
                
                # 发送离线消息
                await self._send_queued_messages(connection_id, user_id)
            
            logger.info(f"WebSocket连接建立: {connection_id}, 用户: {user_id}")
            
            # 发送连接成功消息
            welcome_message = WebSocketMessage(
                type="connection_established",
                data={
                    "connection_id": connection_id,
                    "user_id": user_id,
                    "message": "WebSocket连接建立成功"
                }
            )
            await self._send_to_connection(connection_id, welcome_message)
            
            return True
            
        except Exception as e:
            logger.error(f"WebSocket连接失败: {connection_id}, 错误: {e}")
            return False
    
    async def disconnect(self, connection_id: str):
        """断开WebSocket连接"""
        if connection_id not in self.active_connections:
            return
        
        connection = self.active_connections[connection_id]
        user_id = connection.user_id
        
        # 清理连接记录
        del self.active_connections[connection_id]
        
        if user_id and user_id in self.user_connections:
            self.user_connections[user_id].discard(connection_id)
            if not self.user_connections[user_id]:
                del self.user_connections[user_id]
        
        # 清理任务订阅
        for task_id, subscribers in self.task_subscriptions.items():
            subscribers.discard(connection_id)
        
        # 清理空的订阅集合
        self.task_subscriptions = {
            task_id: subscribers 
            for task_id, subscribers in self.task_subscriptions.items() 
            if subscribers
        }
        
        logger.info(f"WebSocket连接断开: {connection_id}, 用户: {user_id}")
    
    async def subscribe_task(self, connection_id: str, task_id: str):
        """订阅任务状态更新"""
        if connection_id not in self.active_connections:
            logger.warning(f"尝试订阅任务但连接不存在: {connection_id}")
            return
        
        connection = self.active_connections[connection_id]
        connection.subscribed_tasks.add(task_id)
        
        if task_id not in self.task_subscriptions:
            self.task_subscriptions[task_id] = set()
        self.task_subscriptions[task_id].add(connection_id)
        
        logger.info(f"连接 {connection_id} 订阅任务: {task_id}")
    
    async def unsubscribe_task(self, connection_id: str, task_id: str):
        """取消订阅任务状态更新"""
        if connection_id not in self.active_connections:
            return
        
        connection = self.active_connections[connection_id]
        connection.subscribed_tasks.discard(task_id)
        
        if task_id in self.task_subscriptions:
            self.task_subscriptions[task_id].discard(connection_id)
            if not self.task_subscriptions[task_id]:
                del self.task_subscriptions[task_id]
        
        logger.info(f"连接 {connection_id} 取消订阅任务: {task_id}")
    
    async def notify_task_status(self, task_id: str, status: str, progress: float = None, 
                                result: Dict[str, Any] = None, error_message: str = None):
        """推送任务状态变化"""
        message_data = {
            "task_id": task_id,
            "status": status,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if progress is not None:
            message_data["progress"] = progress
        if result is not None:
            message_data["result"] = result
        if error_message is not None:
            message_data["error_message"] = error_message
        
        message = WebSocketMessage(
            type="task_status",
            data=message_data
        )
        
        # 发送给订阅该任务的所有连接
        if task_id in self.task_subscriptions:
            subscribers = list(self.task_subscriptions[task_id])
            await self._broadcast_to_connections(subscribers, message)
            logger.info(f"任务状态通知已发送: {task_id}, 状态: {status}, 订阅者: {len(subscribers)}")
    
    async def notify_user(self, user_id: str, message_type: str, data: Dict[str, Any]):
        """向特定用户发送通知"""
        message = WebSocketMessage(
            type=message_type,
            data=data
        )
        
        if user_id in self.user_connections:
            connections = list(self.user_connections[user_id])
            await self._broadcast_to_connections(connections, message)
            logger.info(f"用户通知已发送: {user_id}, 类型: {message_type}")
        else:
            # 用户离线，将消息加入队列
            if user_id not in self.message_queue:
                self.message_queue[user_id] = []
            self.message_queue[user_id].append(message)
            logger.info(f"用户离线，消息已加入队列: {user_id}, 类型: {message_type}")
    
    async def broadcast_system_alert(self, alert_type: str, message: str, severity: str = "info"):
        """广播系统告警"""
        alert_message = WebSocketMessage(
            type="system_alert",
            data={
                "alert_type": alert_type,
                "message": message,
                "severity": severity,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        # 发送给所有连接
        connections = list(self.active_connections.keys())
        await self._broadcast_to_connections(connections, alert_message)
        logger.info(f"系统告警已广播: {alert_type}, 严重程度: {severity}, 接收者: {len(connections)}")
    
    async def _send_to_connection(self, connection_id: str, message: WebSocketMessage):
        """向单个连接发送消息"""
        if connection_id not in self.active_connections:
            return
        
        connection = self.active_connections[connection_id]
        try:
            await connection.websocket.send_text(message.to_json())
        except Exception as e:
            logger.error(f"发送消息失败: {connection_id}, 错误: {e}")
            # 连接可能已断开，清理连接
            await self.disconnect(connection_id)
    
    async def _broadcast_to_connections(self, connection_ids: List[str], message: WebSocketMessage):
        """向多个连接广播消息"""
        if not connection_ids:
            return
        
        # 并发发送消息
        tasks = []
        for connection_id in connection_ids:
            if connection_id in self.active_connections:
                task = asyncio.create_task(self._send_to_connection(connection_id, message))
                tasks.append(task)
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _send_queued_messages(self, connection_id: str, user_id: str):
        """发送离线消息队列中的消息"""
        if user_id not in self.message_queue:
            return
        
        messages = self.message_queue[user_id]
        for message in messages:
            await self._send_to_connection(connection_id, message)
        
        # 清空消息队列
        del self.message_queue[user_id]
        logger.info(f"已发送 {len(messages)} 条离线消息给用户: {user_id}")
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """获取连接统计信息"""
        return {
            "total_connections": len(self.active_connections),
            "user_connections": len(self.user_connections),
            "task_subscriptions": len(self.task_subscriptions),
            "queued_messages": sum(len(messages) for messages in self.message_queue.values()),
            "connections_by_user": {
                user_id: len(connections) 
                for user_id, connections in self.user_connections.items()
            }
        }
    
    async def handle_client_message(self, connection_id: str, message: str):
        """处理客户端发送的消息"""
        try:
            data = json.loads(message)
            message_type = data.get("type")
            
            if message_type == "subscribe_task":
                task_id = data.get("task_id")
                if task_id:
                    await self.subscribe_task(connection_id, task_id)
            
            elif message_type == "unsubscribe_task":
                task_id = data.get("task_id")
                if task_id:
                    await self.unsubscribe_task(connection_id, task_id)
            
            elif message_type == "ping":
                # 心跳检测
                pong_message = WebSocketMessage(
                    type="pong",
                    data={"message": "pong"}
                )
                await self._send_to_connection(connection_id, pong_message)
            
            else:
                logger.warning(f"未知消息类型: {message_type}")
                
        except json.JSONDecodeError:
            logger.error(f"无效的JSON消息: {message}")
        except Exception as e:
            logger.error(f"处理客户端消息失败: {e}")


# 全局WebSocket管理器实例
websocket_manager = WebSocketManager()