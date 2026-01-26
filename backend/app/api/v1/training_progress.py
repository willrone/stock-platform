"""
训练进度API路由
添加进度查询和报告接口，支持训练控制操作
"""
import asyncio
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from app.api.v1.schemas import StandardResponse

# from app.services.tasks.task_manager import task_manager  # 将在运行时导入
from app.services.models.model_lifecycle_manager import model_lifecycle_manager

router = APIRouter(prefix="/training", tags=["训练进度"])


# WebSocket连接管理
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        try:
            await websocket.send_text(message)
        except:
            self.disconnect(websocket)

    async def broadcast(self, message: str):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                disconnected.append(connection)

        # 清理断开的连接
        for connection in disconnected:
            self.disconnect(connection)


manager = ConnectionManager()


# 请求模型
class TrainingControlRequest(BaseModel):
    action: str  # start, pause, resume, stop, cancel
    task_id: Optional[str] = None
    model_id: Optional[str] = None


class TrainingConfigRequest(BaseModel):
    model_type: str
    parameters: Dict[str, Any]
    training_data: Dict[str, Any]
    validation_split: float = 0.2
    epochs: int = 100
    batch_size: int = 32


@router.get("/tasks", response_model=StandardResponse, summary="获取训练任务列表")
async def get_training_tasks(
    status: Optional[str] = Query(None, description="任务状态过滤"),
    model_type: Optional[str] = Query(None, description="模型类型过滤"),
    limit: int = Query(50, description="返回数量限制"),
):
    """获取训练任务列表"""
    try:
        # 获取所有训练相关的任务
        tasks = task_manager.get_tasks_by_type("model_training", limit=limit)

        # 过滤条件
        if status:
            tasks = [task for task in tasks if task.status == status]

        if model_type:
            tasks = [
                task for task in tasks if task.metadata.get("model_type") == model_type
            ]

        # 转换为字典格式
        task_list = []
        for task in tasks:
            task_dict = task.to_dict()

            # 添加训练特定信息
            if hasattr(task, "progress"):
                task_dict["training_progress"] = {
                    "current_epoch": task.progress.get("current_epoch", 0),
                    "total_epochs": task.progress.get("total_epochs", 0),
                    "current_loss": task.progress.get("current_loss"),
                    "best_loss": task.progress.get("best_loss"),
                    "learning_rate": task.progress.get("learning_rate"),
                    "elapsed_time": task.progress.get("elapsed_time"),
                    "estimated_remaining": task.progress.get("estimated_remaining"),
                }

            task_list.append(task_dict)

        return StandardResponse(
            success=True,
            message=f"成功获取训练任务列表: {len(task_list)} 个任务",
            data={
                "tasks": task_list,
                "total_count": len(task_list),
                "filters": {"status": status, "model_type": model_type, "limit": limit},
            },
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取训练任务失败: {str(e)}")


@router.get("/tasks/{task_id}", response_model=StandardResponse, summary="获取训练任务详情")
async def get_training_task(task_id: str):
    """获取训练任务详情"""
    try:
        task = task_manager.get_task(task_id)

        if not task:
            raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")

        task_dict = task.to_dict()

        # 添加详细的训练信息
        if hasattr(task, "training_metrics"):
            task_dict["training_metrics"] = task.training_metrics

        if hasattr(task, "validation_metrics"):
            task_dict["validation_metrics"] = task.validation_metrics

        if hasattr(task, "training_history"):
            task_dict["training_history"] = task.training_history

        # 获取模型生命周期信息
        model_id = task.metadata.get("model_id")
        if model_id:
            lifecycle_info = model_lifecycle_manager.get_model_lifecycle(model_id)
            if lifecycle_info:
                task_dict["model_lifecycle"] = lifecycle_info.to_dict()

        return StandardResponse(success=True, message="成功获取训练任务详情", data=task_dict)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取任务详情失败: {str(e)}")


@router.get(
    "/tasks/{task_id}/progress", response_model=StandardResponse, summary="获取训练进度"
)
async def get_training_progress(task_id: str):
    """获取训练进度"""
    try:
        task = task_manager.get_task(task_id)

        if not task:
            raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")

        # 获取进度信息
        progress_info = {
            "task_id": task_id,
            "status": task.status,
            "progress_percentage": task.progress_percentage,
            "created_at": task.created_at.isoformat(),
            "updated_at": task.updated_at.isoformat(),
            "elapsed_time": (datetime.now() - task.created_at).total_seconds(),
        }

        # 添加训练特定进度
        if hasattr(task, "progress"):
            progress_info.update(
                {
                    "current_epoch": task.progress.get("current_epoch", 0),
                    "total_epochs": task.progress.get("total_epochs", 0),
                    "current_batch": task.progress.get("current_batch", 0),
                    "total_batches": task.progress.get("total_batches", 0),
                    "current_loss": task.progress.get("current_loss"),
                    "best_loss": task.progress.get("best_loss"),
                    "current_accuracy": task.progress.get("current_accuracy"),
                    "best_accuracy": task.progress.get("best_accuracy"),
                    "learning_rate": task.progress.get("learning_rate"),
                    "estimated_remaining": task.progress.get("estimated_remaining"),
                }
            )

        # 添加最近的训练日志
        if hasattr(task, "logs"):
            recent_logs = task.logs[-10:] if len(task.logs) > 10 else task.logs
            progress_info["recent_logs"] = recent_logs

        return StandardResponse(success=True, message="成功获取训练进度", data=progress_info)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取训练进度失败: {str(e)}")


@router.get(
    "/tasks/{task_id}/metrics", response_model=StandardResponse, summary="获取训练指标"
)
async def get_training_metrics(task_id: str):
    """获取训练指标历史"""
    try:
        task = task_manager.get_task(task_id)

        if not task:
            raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")

        metrics_data = {
            "task_id": task_id,
            "training_metrics": getattr(task, "training_metrics", {}),
            "validation_metrics": getattr(task, "validation_metrics", {}),
            "training_history": getattr(task, "training_history", []),
            "metric_summary": {},
        }

        # 计算指标摘要
        if hasattr(task, "training_history") and task.training_history:
            history = task.training_history

            # 提取损失和准确率历史
            loss_history = [
                epoch.get("loss", 0) for epoch in history if "loss" in epoch
            ]
            val_loss_history = [
                epoch.get("val_loss", 0) for epoch in history if "val_loss" in epoch
            ]
            acc_history = [
                epoch.get("accuracy", 0) for epoch in history if "accuracy" in epoch
            ]
            val_acc_history = [
                epoch.get("val_accuracy", 0)
                for epoch in history
                if "val_accuracy" in epoch
            ]

            metrics_data["metric_summary"] = {
                "total_epochs": len(history),
                "best_loss": min(loss_history) if loss_history else None,
                "best_val_loss": min(val_loss_history) if val_loss_history else None,
                "best_accuracy": max(acc_history) if acc_history else None,
                "best_val_accuracy": max(val_acc_history) if val_acc_history else None,
                "final_loss": loss_history[-1] if loss_history else None,
                "final_val_loss": val_loss_history[-1] if val_loss_history else None,
                "final_accuracy": acc_history[-1] if acc_history else None,
                "final_val_accuracy": val_acc_history[-1] if val_acc_history else None,
            }

        return StandardResponse(success=True, message="成功获取训练指标", data=metrics_data)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取训练指标失败: {str(e)}")


@router.post(
    "/tasks/{task_id}/control", response_model=StandardResponse, summary="控制训练任务"
)
async def control_training_task(task_id: str, request: TrainingControlRequest):
    """控制训练任务（暂停、恢复、停止等）"""
    try:
        task = task_manager.get_task(task_id)

        if not task:
            raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")

        action = request.action.lower()
        result = {"action": action, "task_id": task_id, "success": False}

        if action == "pause":
            success = task_manager.pause_task(task_id)
            result["success"] = success
            result["message"] = "任务已暂停" if success else "暂停任务失败"

        elif action == "resume":
            success = task_manager.resume_task(task_id)
            result["success"] = success
            result["message"] = "任务已恢复" if success else "恢复任务失败"

        elif action == "stop":
            success = task_manager.stop_task(task_id)
            result["success"] = success
            result["message"] = "任务已停止" if success else "停止任务失败"

        elif action == "cancel":
            success = task_manager.cancel_task(task_id)
            result["success"] = success
            result["message"] = "任务已取消" if success else "取消任务失败"

        else:
            raise HTTPException(status_code=400, detail=f"不支持的操作: {action}")

        # 广播状态更新
        if result["success"]:
            await manager.broadcast(
                json.dumps(
                    {
                        "type": "task_control",
                        "task_id": task_id,
                        "action": action,
                        "timestamp": datetime.now().isoformat(),
                    }
                )
            )

        return StandardResponse(
            success=result["success"], message=result["message"], data=result
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"控制训练任务失败: {str(e)}")


@router.get(
    "/models/{model_id}/training-history",
    response_model=StandardResponse,
    summary="获取模型训练历史",
)
async def get_model_training_history(
    model_id: str, limit: int = Query(10, description="返回数量限制")
):
    """获取模型的训练历史"""
    try:
        # 获取模型相关的训练任务
        tasks = task_manager.get_tasks_by_metadata("model_id", model_id, limit=limit)

        training_history = []
        for task in tasks:
            if task.task_type == "model_training":
                history_item = {
                    "task_id": task.task_id,
                    "status": task.status,
                    "created_at": task.created_at.isoformat(),
                    "completed_at": task.completed_at.isoformat()
                    if task.completed_at
                    else None,
                    "progress_percentage": task.progress_percentage,
                    "final_metrics": getattr(task, "final_metrics", {}),
                    "training_config": task.metadata.get("training_config", {}),
                }
                training_history.append(history_item)

        # 按创建时间排序
        training_history.sort(key=lambda x: x["created_at"], reverse=True)

        return StandardResponse(
            success=True,
            message=f"成功获取模型训练历史: {len(training_history)} 条记录",
            data={
                "model_id": model_id,
                "training_history": training_history,
                "total_count": len(training_history),
            },
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取模型训练历史失败: {str(e)}")


@router.get("/stats", response_model=StandardResponse, summary="获取训练统计")
async def get_training_stats():
    """获取训练统计信息"""
    try:
        # 获取所有训练任务
        all_tasks = task_manager.get_tasks_by_type("model_training")

        # 统计信息
        stats = {
            "total_tasks": len(all_tasks),
            "status_distribution": {},
            "model_type_distribution": {},
            "recent_tasks": [],
            "average_duration": 0,
            "success_rate": 0,
        }

        # 按状态统计
        for task in all_tasks:
            status = task.status
            stats["status_distribution"][status] = (
                stats["status_distribution"].get(status, 0) + 1
            )

            # 按模型类型统计
            model_type = task.metadata.get("model_type", "unknown")
            stats["model_type_distribution"][model_type] = (
                stats["model_type_distribution"].get(model_type, 0) + 1
            )

        # 最近的任务
        recent_tasks = sorted(all_tasks, key=lambda x: x.created_at, reverse=True)[:5]
        stats["recent_tasks"] = [
            {
                "task_id": task.task_id,
                "status": task.status,
                "model_type": task.metadata.get("model_type"),
                "created_at": task.created_at.isoformat(),
                "progress_percentage": task.progress_percentage,
            }
            for task in recent_tasks
        ]

        # 计算平均持续时间和成功率
        completed_tasks = [task for task in all_tasks if task.completed_at]
        if completed_tasks:
            durations = [
                (task.completed_at - task.created_at).total_seconds()
                for task in completed_tasks
            ]
            stats["average_duration"] = sum(durations) / len(durations)

            successful_tasks = [
                task for task in completed_tasks if task.status == "completed"
            ]
            stats["success_rate"] = len(successful_tasks) / len(completed_tasks) * 100

        return StandardResponse(success=True, message="成功获取训练统计信息", data=stats)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取训练统计失败: {str(e)}")


@router.post("/start", response_model=StandardResponse, summary="启动训练任务")
async def start_training(request: TrainingConfigRequest):
    """启动新的训练任务"""
    try:
        # 创建训练任务
        task_config = {
            "model_type": request.model_type,
            "parameters": request.parameters,
            "training_data": request.training_data,
            "validation_split": request.validation_split,
            "epochs": request.epochs,
            "batch_size": request.batch_size,
        }

        task_id = task_manager.create_task(
            task_type="model_training",
            task_config=task_config,
            metadata={"model_type": request.model_type, "training_config": task_config},
        )

        # 启动任务
        success = task_manager.start_task(task_id)

        if not success:
            raise HTTPException(status_code=500, detail="启动训练任务失败")

        # 广播任务启动事件
        await manager.broadcast(
            json.dumps(
                {
                    "type": "training_started",
                    "task_id": task_id,
                    "model_type": request.model_type,
                    "timestamp": datetime.now().isoformat(),
                }
            )
        )

        return StandardResponse(
            success=True,
            message=f"成功启动训练任务: {task_id}",
            data={
                "task_id": task_id,
                "model_type": request.model_type,
                "config": task_config,
                "started_at": datetime.now().isoformat(),
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"启动训练任务失败: {str(e)}")


@router.websocket("/ws/{task_id}")
async def websocket_training_progress(websocket: WebSocket, task_id: str):
    """WebSocket实时训练进度推送"""
    await manager.connect(websocket)

    try:
        # 发送初始状态
        task = task_manager.get_task(task_id)
        if task:
            initial_data = {
                "type": "initial_status",
                "task_id": task_id,
                "status": task.status,
                "progress_percentage": task.progress_percentage,
                "timestamp": datetime.now().isoformat(),
            }
            await manager.send_personal_message(json.dumps(initial_data), websocket)

        # 保持连接并定期发送更新
        while True:
            await asyncio.sleep(5)  # 每5秒更新一次

            task = task_manager.get_task(task_id)
            if task:
                progress_data = {
                    "type": "progress_update",
                    "task_id": task_id,
                    "status": task.status,
                    "progress_percentage": task.progress_percentage,
                    "timestamp": datetime.now().isoformat(),
                }

                # 添加训练特定信息
                if hasattr(task, "progress"):
                    progress_data.update(
                        {
                            "current_epoch": task.progress.get("current_epoch", 0),
                            "total_epochs": task.progress.get("total_epochs", 0),
                            "current_loss": task.progress.get("current_loss"),
                            "current_accuracy": task.progress.get("current_accuracy"),
                            "learning_rate": task.progress.get("learning_rate"),
                        }
                    )

                await manager.send_personal_message(
                    json.dumps(progress_data), websocket
                )

                # 如果任务完成，发送完成消息并断开连接
                if task.status in ["completed", "failed", "cancelled"]:
                    final_data = {
                        "type": "training_finished",
                        "task_id": task_id,
                        "status": task.status,
                        "timestamp": datetime.now().isoformat(),
                    }
                    await manager.send_personal_message(
                        json.dumps(final_data), websocket
                    )
                    break
            else:
                # 任务不存在
                error_data = {
                    "type": "error",
                    "message": f"任务不存在: {task_id}",
                    "timestamp": datetime.now().isoformat(),
                }
                await manager.send_personal_message(json.dumps(error_data), websocket)
                break

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        error_data = {
            "type": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat(),
        }
        try:
            await manager.send_personal_message(json.dumps(error_data), websocket)
        except:
            pass
        manager.disconnect(websocket)


@router.websocket("/ws/global")
async def websocket_global_training_updates(websocket: WebSocket):
    """WebSocket全局训练更新推送"""
    await manager.connect(websocket)

    try:
        # 发送欢迎消息
        welcome_data = {
            "type": "connected",
            "message": "已连接到全局训练更新",
            "timestamp": datetime.now().isoformat(),
        }
        await manager.send_personal_message(json.dumps(welcome_data), websocket)

        # 保持连接
        while True:
            await asyncio.sleep(30)  # 每30秒发送心跳

            heartbeat_data = {
                "type": "heartbeat",
                "timestamp": datetime.now().isoformat(),
            }
            await manager.send_personal_message(json.dumps(heartbeat_data), websocket)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        manager.disconnect(websocket)
