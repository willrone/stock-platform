"""
回测WebSocket端点

提供回测进度的实时WebSocket通信
"""

import json
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime
from loguru import logger

from fastapi import WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.routing import APIRouter

from app.services.backtest.backtest_progress_monitor import backtest_progress_monitor
from app.services.infrastructure.websocket_manager import websocket_manager
from app.repositories.task_repository import TaskRepository
from app.core.database import SessionLocal
from sqlalchemy.orm import Session


router = APIRouter(prefix="/backtest", tags=["backtest-websocket"])


def get_db():
    """获取数据库会话依赖"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class BacktestWebSocketManager:
    """回测WebSocket连接管理器"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.task_subscriptions: Dict[str, set] = {}  # task_id -> set of connection_ids
        self.connection_tasks: Dict[str, str] = {}  # connection_id -> task_id
    
    async def connect(self, websocket: WebSocket, connection_id: str, task_id: str) -> bool:
        """建立WebSocket连接"""
        try:
            await websocket.accept()
            
            self.active_connections[connection_id] = websocket
            self.connection_tasks[connection_id] = task_id
            
            # 订阅任务
            if task_id not in self.task_subscriptions:
                self.task_subscriptions[task_id] = set()
            self.task_subscriptions[task_id].add(connection_id)
            
            logger.info(f"回测WebSocket连接建立: {connection_id}, 任务: {task_id}")
            
            # 发送连接成功消息
            await self.send_to_connection(connection_id, {
                "type": "connection_established",
                "task_id": task_id,
                "connection_id": connection_id,
                "message": "回测进度WebSocket连接建立成功",
                "timestamp": datetime.utcnow().isoformat()
            })
            
            # 发送当前进度状态（如果存在）
            progress_data = backtest_progress_monitor.get_progress_data(task_id)
            if progress_data:
                await self.send_progress_update(task_id, progress_data)
            
            return True
            
        except Exception as e:
            logger.error(f"回测WebSocket连接失败: {connection_id}, 错误: {e}")
            return False
    
    async def disconnect(self, connection_id: str):
        """断开WebSocket连接"""
        if connection_id not in self.active_connections:
            return
        
        task_id = self.connection_tasks.get(connection_id)
        
        # 清理连接记录
        del self.active_connections[connection_id]
        if connection_id in self.connection_tasks:
            del self.connection_tasks[connection_id]
        
        # 清理任务订阅
        if task_id and task_id in self.task_subscriptions:
            self.task_subscriptions[task_id].discard(connection_id)
            if not self.task_subscriptions[task_id]:
                del self.task_subscriptions[task_id]
        
        logger.info(f"回测WebSocket连接断开: {connection_id}, 任务: {task_id}")
    
    async def send_to_connection(self, connection_id: str, message: Dict[str, Any]):
        """向单个连接发送消息"""
        if connection_id not in self.active_connections:
            return
        
        websocket = self.active_connections[connection_id]
        try:
            await websocket.send_text(json.dumps(message, ensure_ascii=False))
        except Exception as e:
            logger.error(f"发送回测WebSocket消息失败: {connection_id}, 错误: {e}")
            await self.disconnect(connection_id)
    
    async def send_to_task_subscribers(self, task_id: str, message: Dict[str, Any]):
        """发送消息给任务订阅者"""
        if task_id not in self.task_subscriptions:
            return
        
        subscribers = list(self.task_subscriptions[task_id])
        for connection_id in subscribers:
            await self.send_to_connection(connection_id, message)
    
    async def send_progress_update(self, task_id: str, progress_data):
        """发送进度更新"""
        message = {
            "type": "progress_update",
            "task_id": task_id,
            "backtest_id": progress_data.backtest_id,
            "overall_progress": progress_data.overall_progress,
            "current_stage": progress_data.current_stage,
            "processed_days": progress_data.processed_trading_days,
            "total_days": progress_data.total_trading_days,
            "current_date": progress_data.current_date,
            "processing_speed": progress_data.processing_speed,
            "estimated_completion": progress_data.estimated_completion.isoformat() if progress_data.estimated_completion else None,
            "elapsed_time": str(progress_data.elapsed_time) if progress_data.elapsed_time else None,
            "portfolio_value": progress_data.current_portfolio_value,
            "signals_generated": progress_data.total_signals_generated,
            "trades_executed": progress_data.total_trades_executed,
            "warnings_count": len(progress_data.warnings),
            "error_message": progress_data.error_message,
            "stages": [
                {
                    "name": stage.stage_name,
                    "description": stage.stage_description,
                    "progress": stage.progress,
                    "status": stage.status,
                    "start_time": stage.start_time.isoformat() if stage.start_time else None,
                    "end_time": stage.end_time.isoformat() if stage.end_time else None,
                    "details": stage.details
                }
                for stage in progress_data.stages
            ],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self.send_to_task_subscribers(task_id, message)
    
    async def send_error_notification(self, task_id: str, error_message: str):
        """发送错误通知"""
        message = {
            "type": "backtest_error",
            "task_id": task_id,
            "error_message": error_message,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self.send_to_task_subscribers(task_id, message)
    
    async def send_completion_notification(self, task_id: str, results: Dict[str, Any]):
        """发送完成通知"""
        message = {
            "type": "backtest_completed",
            "task_id": task_id,
            "results": results,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self.send_to_task_subscribers(task_id, message)
    
    async def send_cancellation_notification(self, task_id: str, reason: str):
        """发送取消通知"""
        message = {
            "type": "backtest_cancelled",
            "task_id": task_id,
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self.send_to_task_subscribers(task_id, message)
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """获取连接统计信息"""
        return {
            "total_connections": len(self.active_connections),
            "task_subscriptions": len(self.task_subscriptions),
            "active_backtests": len(backtest_progress_monitor.get_all_active_backtests()),
            "timestamp": datetime.utcnow().isoformat()
        }


# 全局回测WebSocket管理器
backtest_ws_manager = BacktestWebSocketManager()


@router.websocket("/ws/{task_id}")
async def backtest_progress_websocket(
    websocket: WebSocket, 
    task_id: str,
    session: Session = Depends(get_db)
):
    """回测进度WebSocket端点"""
    connection_id = f"bt_{task_id}_{datetime.utcnow().timestamp()}"
    
    # 验证任务存在
    try:
        task_repository = TaskRepository(session)
        task = task_repository.get_task_by_id(task_id)
        
        if not task:
            await websocket.close(code=4004, reason="任务不存在")
            return
        
        if task.task_type != "backtest":
            await websocket.close(code=4005, reason="任务类型不是回测")
            return
            
    except Exception as e:
        logger.error(f"验证回测任务失败: {task_id}, 错误: {e}")
        await websocket.close(code=4000, reason="服务器内部错误")
        return
    
    # 建立连接
    connected = await backtest_ws_manager.connect(websocket, connection_id, task_id)
    if not connected:
        return
    
    try:
        while True:
            # 接收客户端消息
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                await handle_backtest_websocket_message(connection_id, task_id, message)
            except json.JSONDecodeError:
                await backtest_ws_manager.send_to_connection(connection_id, {
                    "type": "error",
                    "message": "无效的JSON格式",
                    "timestamp": datetime.utcnow().isoformat()
                })
            except Exception as e:
                logger.error(f"处理回测WebSocket消息失败: {e}")
                await backtest_ws_manager.send_to_connection(connection_id, {
                    "type": "error",
                    "message": "消息处理失败",
                    "timestamp": datetime.utcnow().isoformat()
                })
    
    except WebSocketDisconnect:
        await backtest_ws_manager.disconnect(connection_id)
    except Exception as e:
        logger.error(f"回测WebSocket连接异常: {e}")
        await backtest_ws_manager.disconnect(connection_id)


async def handle_backtest_websocket_message(connection_id: str, task_id: str, message: Dict[str, Any]):
    """处理回测WebSocket消息"""
    message_type = message.get("type")
    
    if message_type == "ping":
        # 心跳检测
        await backtest_ws_manager.send_to_connection(connection_id, {
            "type": "pong",
            "timestamp": datetime.utcnow().isoformat()
        })
    
    elif message_type == "get_current_progress":
        # 获取当前进度
        progress_data = backtest_progress_monitor.get_progress_data(task_id)
        if progress_data:
            await backtest_ws_manager.send_progress_update(task_id, progress_data)
        else:
            await backtest_ws_manager.send_to_connection(connection_id, {
                "type": "no_progress_data",
                "message": "当前没有进度数据",
                "timestamp": datetime.utcnow().isoformat()
            })
    
    elif message_type == "cancel_backtest":
        # 取消回测
        reason = message.get("reason", "用户取消")
        await backtest_progress_monitor.cancel_backtest(task_id, reason)
        await backtest_ws_manager.send_cancellation_notification(task_id, reason)
    
    else:
        await backtest_ws_manager.send_to_connection(connection_id, {
            "type": "error",
            "message": f"未知的消息类型: {message_type}",
            "timestamp": datetime.utcnow().isoformat()
        })


@router.get("/ws/stats")
async def get_backtest_websocket_stats():
    """获取回测WebSocket统计信息"""
    return {
        "success": True,
        "data": backtest_ws_manager.get_connection_stats()
    }


@router.get("/progress/{task_id}")
async def get_backtest_progress(
    task_id: str,
    session: Session = Depends(get_db)
):
    """获取回测进度（HTTP接口）"""
    try:
        # 验证任务存在
        task_repository = TaskRepository(session)
        task = task_repository.get_task_by_id(task_id)
        
        if not task:
            raise HTTPException(status_code=404, detail="任务不存在")
        
        if task.task_type != "backtest":
            raise HTTPException(status_code=400, detail="任务类型不是回测")
        
        # 获取进度数据
        progress_data = backtest_progress_monitor.get_progress_data(task_id)
        
        if not progress_data:
            return {
                "success": True,
                "message": "当前没有进度数据",
                "data": None
            }
        
        return {
            "success": True,
            "message": "获取进度数据成功",
            "data": {
                "task_id": progress_data.task_id,
                "backtest_id": progress_data.backtest_id,
                "overall_progress": progress_data.overall_progress,
                "current_stage": progress_data.current_stage,
                "processed_days": progress_data.processed_trading_days,
                "total_days": progress_data.total_trading_days,
                "current_date": progress_data.current_date,
                "processing_speed": progress_data.processing_speed,
                "estimated_completion": progress_data.estimated_completion.isoformat() if progress_data.estimated_completion else None,
                "elapsed_time": str(progress_data.elapsed_time) if progress_data.elapsed_time else None,
                "portfolio_value": progress_data.current_portfolio_value,
                "signals_generated": progress_data.total_signals_generated,
                "trades_executed": progress_data.total_trades_executed,
                "warnings_count": len(progress_data.warnings),
                "warnings": progress_data.warnings[-5:],  # 最近5个警告
                "error_message": progress_data.error_message,
                "stages": [
                    {
                        "name": stage.stage_name,
                        "description": stage.stage_description,
                        "progress": stage.progress,
                        "status": stage.status,
                        "start_time": stage.start_time.isoformat() if stage.start_time else None,
                        "end_time": stage.end_time.isoformat() if stage.end_time else None,
                        "details": stage.details
                    }
                    for stage in progress_data.stages
                ]
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取回测进度失败: {task_id}, 错误: {e}")
        raise HTTPException(status_code=500, detail=f"获取回测进度失败: {str(e)}")


@router.post("/cancel/{task_id}")
async def cancel_backtest(
    task_id: str,
    reason: str = "用户取消",
    session: Session = Depends(get_db)
):
    """取消回测任务"""
    try:
        # 验证任务存在
        task_repository = TaskRepository(session)
        task = task_repository.get_task_by_id(task_id)
        
        if not task:
            raise HTTPException(status_code=404, detail="任务不存在")
        
        if task.task_type != "backtest":
            raise HTTPException(status_code=400, detail="任务类型不是回测")
        
        # 取消回测
        await backtest_progress_monitor.cancel_backtest(task_id, reason)
        
        # 通知WebSocket客户端
        await backtest_ws_manager.send_cancellation_notification(task_id, reason)
        
        return {
            "success": True,
            "message": "回测任务已取消",
            "data": {
                "task_id": task_id,
                "reason": reason,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"取消回测任务失败: {task_id}, 错误: {e}")
        raise HTTPException(status_code=500, detail=f"取消回测任务失败: {str(e)}")


# 导出主要组件
__all__ = [
    'router',
    'backtest_ws_manager',
    'BacktestWebSocketManager'
]