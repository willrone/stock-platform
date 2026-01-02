"""
WebSocket支持

实现实时通信功能，包括：
- 任务状态实时更新
- 系统状态监控
- 实时数据推送
- 连接管理
"""

import json
from typing import Dict, Set, Any
from datetime import datetime
from loguru import logger

from fastapi import WebSocket, WebSocketDisconnect
from fastapi.routing import APIRouter

# WebSocket连接管理器
class ConnectionManager:
    """WebSocket连接管理器"""
    
    def __init__(self):
        # 存储活跃连接
        self.active_connections: Set[WebSocket] = set()
        # 存储任务订阅关系
        self.task_subscriptions: Dict[str, Set[WebSocket]] = {}
        # 存储系统状态订阅
        self.system_subscriptions: Set[WebSocket] = set()
    
    async def connect(self, websocket: WebSocket):
        """接受WebSocket连接"""
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"WebSocket连接建立，当前连接数: {len(self.active_connections)}")
        
        # 发送欢迎消息
        await self.send_personal_message(websocket, {
            "type": "connection",
            "message": "WebSocket连接成功",
            "timestamp": datetime.now().isoformat()
        })
    
    def disconnect(self, websocket: WebSocket):
        """断开WebSocket连接"""
        self.active_connections.discard(websocket)
        
        # 清理任务订阅
        for task_id, subscribers in self.task_subscriptions.items():
            subscribers.discard(websocket)
        
        # 清理系统状态订阅
        self.system_subscriptions.discard(websocket)
        
        logger.info(f"WebSocket连接断开，当前连接数: {len(self.active_connections)}")
    
    async def send_personal_message(self, websocket: WebSocket, message: Dict[str, Any]):
        """发送个人消息"""
        try:
            await websocket.send_text(json.dumps(message, ensure_ascii=False))
        except Exception as e:
            logger.error(f"发送个人消息失败: {e}")
            self.disconnect(websocket)
    
    async def broadcast(self, message: Dict[str, Any]):
        """广播消息给所有连接"""
        if not self.active_connections:
            return
        
        message_text = json.dumps(message, ensure_ascii=False)
        disconnected = set()
        
        for connection in self.active_connections:
            try:
                await connection.send_text(message_text)
            except Exception as e:
                logger.error(f"广播消息失败: {e}")
                disconnected.add(connection)
        
        # 清理断开的连接
        for connection in disconnected:
            self.disconnect(connection)
    
    async def send_to_task_subscribers(self, task_id: str, message: Dict[str, Any]):
        """发送消息给任务订阅者"""
        if task_id not in self.task_subscriptions:
            return
        
        subscribers = self.task_subscriptions[task_id].copy()
        message_text = json.dumps(message, ensure_ascii=False)
        disconnected = set()
        
        for connection in subscribers:
            try:
                await connection.send_text(message_text)
            except Exception as e:
                logger.error(f"发送任务消息失败: {e}")
                disconnected.add(connection)
        
        # 清理断开的连接
        for connection in disconnected:
            self.task_subscriptions[task_id].discard(connection)
    
    async def send_to_system_subscribers(self, message: Dict[str, Any]):
        """发送消息给系统状态订阅者"""
        if not self.system_subscriptions:
            return
        
        message_text = json.dumps(message, ensure_ascii=False)
        disconnected = set()
        
        for connection in self.system_subscriptions.copy():
            try:
                await connection.send_text(message_text)
            except Exception as e:
                logger.error(f"发送系统消息失败: {e}")
                disconnected.add(connection)
        
        # 清理断开的连接
        for connection in disconnected:
            self.system_subscriptions.discard(connection)
    
    def subscribe_to_task(self, websocket: WebSocket, task_id: str):
        """订阅任务更新"""
        if task_id not in self.task_subscriptions:
            self.task_subscriptions[task_id] = set()
        self.task_subscriptions[task_id].add(websocket)
        logger.info(f"客户端订阅任务 {task_id}")
    
    def subscribe_to_model_training(self, websocket: WebSocket, model_id: str):
        """订阅模型训练更新"""
        # 使用任务订阅机制，但用model_id作为key
        if model_id not in self.task_subscriptions:
            self.task_subscriptions[model_id] = set()
        self.task_subscriptions[model_id].add(websocket)
        logger.info(f"客户端订阅模型训练 {model_id}")
    
    def unsubscribe_from_task(self, websocket: WebSocket, task_id: str):
        """取消订阅任务更新"""
        if task_id in self.task_subscriptions:
            self.task_subscriptions[task_id].discard(websocket)
            if not self.task_subscriptions[task_id]:
                del self.task_subscriptions[task_id]
        logger.info(f"客户端取消订阅任务 {task_id}")
    
    def subscribe_to_system(self, websocket: WebSocket):
        """订阅系统状态"""
        self.system_subscriptions.add(websocket)
        logger.info("客户端订阅系统状态")
    
    def unsubscribe_from_system(self, websocket: WebSocket):
        """取消订阅系统状态"""
        self.system_subscriptions.discard(websocket)
        logger.info("客户端取消订阅系统状态")
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """获取连接统计信息"""
        return {
            "total_connections": len(self.active_connections),
            "task_subscriptions": len(self.task_subscriptions),
            "system_subscriptions": len(self.system_subscriptions),
            "timestamp": datetime.now().isoformat()
        }


# 全局连接管理器实例
manager = ConnectionManager()

# WebSocket路由
ws_router = APIRouter()

@ws_router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket端点"""
    await manager.connect(websocket)
    
    try:
        while True:
            # 接收客户端消息
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                await handle_websocket_message(websocket, message)
            except json.JSONDecodeError:
                await manager.send_personal_message(websocket, {
                    "type": "error",
                    "message": "无效的JSON格式",
                    "timestamp": datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"处理WebSocket消息失败: {e}")
                await manager.send_personal_message(websocket, {
                    "type": "error",
                    "message": "消息处理失败",
                    "timestamp": datetime.now().isoformat()
                })
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket连接异常: {e}")
        manager.disconnect(websocket)


async def handle_websocket_message(websocket: WebSocket, message: Dict[str, Any]):
    """处理WebSocket消息"""
    message_type = message.get("type")
    
    if message_type == "subscribe:task":
        # 订阅任务更新
        task_id = message.get("task_id")
        if task_id:
            manager.subscribe_to_task(websocket, task_id)
            await manager.send_personal_message(websocket, {
                "type": "subscription",
                "message": f"已订阅任务 {task_id}",
                "task_id": task_id,
                "timestamp": datetime.now().isoformat()
            })
    
    elif message_type == "unsubscribe:task":
        # 取消订阅任务更新
        task_id = message.get("task_id")
        if task_id:
            manager.unsubscribe_from_task(websocket, task_id)
            await manager.send_personal_message(websocket, {
                "type": "unsubscription",
                "message": f"已取消订阅任务 {task_id}",
                "task_id": task_id,
                "timestamp": datetime.now().isoformat()
            })
    
    elif message_type == "subscribe:system":
        # 订阅系统状态
        manager.subscribe_to_system(websocket)
        await manager.send_personal_message(websocket, {
            "type": "subscription",
            "message": "已订阅系统状态",
            "timestamp": datetime.now().isoformat()
        })
    
    elif message_type == "unsubscribe:system":
        # 取消订阅系统状态
        manager.unsubscribe_from_system(websocket)
        await manager.send_personal_message(websocket, {
            "type": "unsubscription",
            "message": "已取消订阅系统状态",
            "timestamp": datetime.now().isoformat()
        })
    
    elif message_type == "subscribe:model_training":
        # 订阅模型训练更新
        model_id = message.get("model_id")
        if model_id:
            manager.subscribe_to_model_training(websocket, model_id)
            await manager.send_personal_message(websocket, {
                "type": "subscription",
                "message": f"已订阅模型训练 {model_id}",
                "model_id": model_id,
                "timestamp": datetime.now().isoformat()
            })
    
    elif message_type == "ping":
        # 心跳检测
        await manager.send_personal_message(websocket, {
            "type": "pong",
            "timestamp": datetime.now().isoformat()
        })
    
    else:
        await manager.send_personal_message(websocket, {
            "type": "error",
            "message": f"未知的消息类型: {message_type}",
            "timestamp": datetime.now().isoformat()
        })


# 任务状态更新通知函数
async def notify_task_created(task_id: str, task_name: str):
    """通知任务创建"""
    await manager.send_to_task_subscribers(task_id, {
        "type": "task:created",
        "task_id": task_id,
        "task_name": task_name,
        "timestamp": datetime.now().isoformat()
    })


async def notify_task_progress(task_id: str, progress: float, status: str, current_stock: str = None):
    """通知任务进度更新"""
    message = {
        "type": "task:progress",
        "task_id": task_id,
        "progress": progress,
        "status": status,
        "timestamp": datetime.now().isoformat()
    }
    
    if current_stock:
        message["current_stock"] = current_stock
    
    await manager.send_to_task_subscribers(task_id, message)


async def notify_task_completed(task_id: str, results: Dict[str, Any]):
    """通知任务完成"""
    await manager.send_to_task_subscribers(task_id, {
        "type": "task:completed",
        "task_id": task_id,
        "results": results,
        "timestamp": datetime.now().isoformat()
    })


async def notify_task_failed(task_id: str, error: str):
    """通知任务失败"""
    await manager.send_to_task_subscribers(task_id, {
        "type": "task:failed",
        "task_id": task_id,
        "error": error,
        "timestamp": datetime.now().isoformat()
    })


async def notify_system_status(status: Dict[str, Any]):
    """通知系统状态更新"""
    await manager.send_to_system_subscribers({
        "type": "system:status",
        "status": status,
        "timestamp": datetime.now().isoformat()
    })


async def notify_system_alert(level: str, message: str):
    """通知系统警告"""
    await manager.send_to_system_subscribers({
        "type": "system:alert",
        "level": level,
        "message": message,
        "timestamp": datetime.now().isoformat()
    })


async def notify_data_updated(stock_code: str):
    """通知数据更新"""
    await manager.broadcast({
        "type": "data:updated",
        "stock_code": stock_code,
        "timestamp": datetime.now().isoformat()
    })


async def notify_model_training_progress(
    model_id: str,
    progress: float,
    stage: str,
    message: str = None,
    metrics: Dict[str, Any] = None
):
    """通知模型训练进度"""
    await manager.send_to_task_subscribers(model_id, {
        "type": "model:training:progress",
        "model_id": model_id,
        "progress": progress,
        "stage": stage,
        "message": message,
        "metrics": metrics or {},
        "timestamp": datetime.now().isoformat()
    })


async def notify_model_training_completed(model_id: str, metrics: Dict[str, Any]):
    """通知模型训练完成"""
    await manager.send_to_task_subscribers(model_id, {
        "type": "model:training:completed",
        "model_id": model_id,
        "metrics": metrics,
        "timestamp": datetime.now().isoformat()
    })


async def notify_model_training_failed(model_id: str, error: str):
    """通知模型训练失败"""
    await manager.send_to_task_subscribers(model_id, {
        "type": "model:training:failed",
        "model_id": model_id,
        "error": error,
        "timestamp": datetime.now().isoformat()
    })


# 导出主要组件
__all__ = [
    'manager',
    'ws_router',
    'notify_task_created',
    'notify_task_progress',
    'notify_task_completed',
    'notify_task_failed',
    'notify_system_status',
    'notify_system_alert',
    'notify_data_updated',
    'notify_model_training_progress',
    'notify_model_training_completed',
    'notify_model_training_failed'
]