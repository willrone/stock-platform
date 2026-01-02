"""
模型生命周期管理器

提供模型状态跟踪和历史记录功能，包括：
- 模型状态转换管理
- 生命周期事件记录
- 状态历史查询
- 自动状态更新
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum
import json
import logging

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, and_
from sqlalchemy.orm import selectinload

from ...core.database import get_async_session as get_db
from ...models.task_models import ModelInfo, ModelLifecycleEvent
from ...core.config import settings

logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """模型状态枚举"""
    CREATING = "creating"
    TRAINING = "training"
    READY = "ready"
    ACTIVE = "active"
    DEPLOYED = "deployed"
    DEPRECATED = "deprecated"
    FAILED = "failed"
    ARCHIVED = "archived"


class ModelLifecycleManager:
    """模型生命周期管理器"""
    
    def __init__(self):
        self.status_transitions = {
            # 允许的状态转换映射
            ModelStatus.CREATING: [ModelStatus.TRAINING, ModelStatus.FAILED],
            ModelStatus.TRAINING: [ModelStatus.READY, ModelStatus.FAILED],
            ModelStatus.READY: [ModelStatus.ACTIVE, ModelStatus.DEPRECATED, ModelStatus.FAILED],
            ModelStatus.ACTIVE: [ModelStatus.DEPLOYED, ModelStatus.DEPRECATED, ModelStatus.FAILED],
            ModelStatus.DEPLOYED: [ModelStatus.ACTIVE, ModelStatus.DEPRECATED, ModelStatus.FAILED],
            ModelStatus.DEPRECATED: [ModelStatus.ARCHIVED],
            ModelStatus.FAILED: [ModelStatus.TRAINING, ModelStatus.ARCHIVED],
            ModelStatus.ARCHIVED: []  # 归档状态不能转换到其他状态
        }
    
    async def transition_status(
        self,
        model_id: str,
        new_status: str,
        reason: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        db: Optional[AsyncSession] = None
    ) -> bool:
        """
        转换模型状态
        
        Args:
            model_id: 模型ID
            new_status: 新状态
            reason: 状态转换原因
            metadata: 附加元数据
            db: 数据库会话
            
        Returns:
            bool: 转换是否成功
        """
        if db is None:
            async for session in get_db():
                return await self._transition_status_impl(
                    session, model_id, new_status, reason, metadata
                )
        else:
            return await self._transition_status_impl(
                db, model_id, new_status, reason, metadata
            )
    
    async def _transition_status_impl(
        self,
        db: AsyncSession,
        model_id: str,
        new_status: str,
        reason: Optional[str],
        metadata: Optional[Dict[str, Any]]
    ) -> bool:
        """状态转换实现"""
        try:
            # 获取当前模型信息
            result = await db.execute(
                select(ModelInfo).where(ModelInfo.model_id == model_id)
            )
            model = result.scalar_one_or_none()
            
            if not model:
                logger.error(f"模型不存在: {model_id}")
                return False
            
            current_status = ModelStatus(model.status)
            new_status_enum = ModelStatus(new_status)
            
            # 检查状态转换是否允许
            if not self._is_valid_transition(current_status, new_status_enum):
                logger.error(
                    f"无效的状态转换: {current_status.value} -> {new_status_enum.value}"
                )
                return False
            
            # 更新模型状态
            await db.execute(
                update(ModelInfo)
                .where(ModelInfo.model_id == model_id)
                .values(
                    status=new_status,
                    updated_at=datetime.utcnow()
                )
            )
            
            # 记录生命周期事件
            await self._record_lifecycle_event(
                db, model_id, current_status.value, new_status, reason, metadata
            )
            
            await db.commit()
            
            logger.info(
                f"模型状态转换成功: {model_id} {current_status.value} -> {new_status}"
            )
            return True
            
        except Exception as e:
            logger.error(f"状态转换失败: {e}")
            await db.rollback()
            return False
    
    def _is_valid_transition(
        self, 
        current_status: ModelStatus, 
        new_status: ModelStatus
    ) -> bool:
        """检查状态转换是否有效"""
        allowed_transitions = self.status_transitions.get(current_status, [])
        return new_status in allowed_transitions
    
    async def _record_lifecycle_event(
        self,
        db: AsyncSession,
        model_id: str,
        from_status: str,
        to_status: str,
        reason: Optional[str],
        metadata: Optional[Dict[str, Any]]
    ):
        """记录生命周期事件"""
        event = ModelLifecycleEvent(
            model_id=model_id,
            from_status=from_status,
            to_status=to_status,
            reason=reason or f"状态转换: {from_status} -> {to_status}",
            event_metadata=json.dumps(metadata) if metadata else None,
            created_at=datetime.utcnow()
        )
        
        db.add(event)
    
    async def get_lifecycle_history(
        self,
        model_id: str,
        limit: int = 50,
        db: Optional[AsyncSession] = None
    ) -> List[Dict[str, Any]]:
        """
        获取模型生命周期历史
        
        Args:
            model_id: 模型ID
            limit: 返回记录数限制
            db: 数据库会话
            
        Returns:
            List[Dict]: 生命周期事件列表
        """
        if db is None:
            async for session in get_db():
                return await self._get_lifecycle_history_impl(session, model_id, limit)
        else:
            return await self._get_lifecycle_history_impl(db, model_id, limit)
    
    async def _get_lifecycle_history_impl(
        self,
        db: AsyncSession,
        model_id: str,
        limit: int
    ) -> List[Dict[str, Any]]:
        """获取生命周期历史实现"""
        try:
            result = await db.execute(
                select(ModelLifecycleEvent)
                .where(ModelLifecycleEvent.model_id == model_id)
                .order_by(ModelLifecycleEvent.created_at.desc())
                .limit(limit)
            )
            
            events = result.scalars().all()
            
            return [
                {
                    "event_id": event.event_id,
                    "from_status": event.from_status,
                    "to_status": event.to_status,
                    "reason": event.reason,
                    "metadata": json.loads(event.event_metadata) if event.event_metadata else None,
                    "created_at": event.created_at.isoformat()
                }
                for event in events
            ]
            
        except Exception as e:
            logger.error(f"获取生命周期历史失败: {e}")
            return []
    
    async def get_models_by_status(
        self,
        status: str,
        limit: int = 100,
        db: Optional[AsyncSession] = None
    ) -> List[Dict[str, Any]]:
        """
        根据状态获取模型列表
        
        Args:
            status: 模型状态
            limit: 返回记录数限制
            db: 数据库会话
            
        Returns:
            List[Dict]: 模型列表
        """
        if db is None:
            async for session in get_db():
                return await self._get_models_by_status_impl(session, status, limit)
        else:
            return await self._get_models_by_status_impl(db, status, limit)
    
    async def _get_models_by_status_impl(
        self,
        db: AsyncSession,
        status: str,
        limit: int
    ) -> List[Dict[str, Any]]:
        """根据状态获取模型列表实现"""
        try:
            result = await db.execute(
                select(ModelInfo)
                .where(ModelInfo.status == status)
                .order_by(ModelInfo.updated_at.desc())
                .limit(limit)
            )
            
            models = result.scalars().all()
            
            return [
                {
                    "model_id": model.model_id,
                    "model_name": model.model_name,
                    "model_type": model.model_type,
                    "status": model.status,
                    "accuracy": model.accuracy,
                    "created_at": model.created_at.isoformat(),
                    "updated_at": model.updated_at.isoformat()
                }
                for model in models
            ]
            
        except Exception as e:
            logger.error(f"根据状态获取模型列表失败: {e}")
            return []
    
    async def auto_cleanup_old_models(
        self,
        days_threshold: int = 30,
        db: Optional[AsyncSession] = None
    ) -> int:
        """
        自动清理旧模型
        
        Args:
            days_threshold: 天数阈值
            db: 数据库会话
            
        Returns:
            int: 清理的模型数量
        """
        if db is None:
            async for session in get_db():
                return await self._auto_cleanup_old_models_impl(session, days_threshold)
        else:
            return await self._auto_cleanup_old_models_impl(db, days_threshold)
    
    async def _auto_cleanup_old_models_impl(
        self,
        db: AsyncSession,
        days_threshold: int
    ) -> int:
        """自动清理旧模型实现"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_threshold)
            
            # 查找需要清理的模型（失败状态且超过阈值时间）
            result = await db.execute(
                select(ModelInfo)
                .where(
                    and_(
                        ModelInfo.status == ModelStatus.FAILED.value,
                        ModelInfo.updated_at < cutoff_date
                    )
                )
            )
            
            models_to_cleanup = result.scalars().all()
            cleanup_count = 0
            
            for model in models_to_cleanup:
                # 转换到归档状态
                success = await self.transition_status(
                    model.model_id,
                    ModelStatus.ARCHIVED.value,
                    f"自动清理：超过{days_threshold}天的失败模型",
                    {"auto_cleanup": True, "days_threshold": days_threshold},
                    db
                )
                
                if success:
                    cleanup_count += 1
            
            logger.info(f"自动清理完成，清理了 {cleanup_count} 个模型")
            return cleanup_count
            
        except Exception as e:
            logger.error(f"自动清理失败: {e}")
            return 0
    
    async def get_status_statistics(
        self,
        db: Optional[AsyncSession] = None
    ) -> Dict[str, int]:
        """
        获取状态统计信息
        
        Args:
            db: 数据库会话
            
        Returns:
            Dict[str, int]: 状态统计
        """
        if db is None:
            async for session in get_db():
                return await self._get_status_statistics_impl(session)
        else:
            return await self._get_status_statistics_impl(db)
    
    async def _get_status_statistics_impl(
        self,
        db: AsyncSession
    ) -> Dict[str, int]:
        """获取状态统计信息实现"""
        try:
            from sqlalchemy import func
            
            result = await db.execute(
                select(ModelInfo.status, func.count(ModelInfo.model_id))
                .group_by(ModelInfo.status)
            )
            
            statistics = {}
            for status, count in result.fetchall():
                statistics[status] = count
            
            return statistics
            
        except Exception as e:
            logger.error(f"获取状态统计失败: {e}")
            return {}


# 全局实例
model_lifecycle_manager = ModelLifecycleManager()