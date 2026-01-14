"""
策略配置管理API
"""

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from loguru import logger

from app.core.database import get_async_session
from app.api.v1.schemas import StandardResponse
from app.models.strategy_config_models import StrategyConfig

router = APIRouter(prefix="/strategy-configs", tags=["策略配置"])


class StrategyConfigCreate(BaseModel):
    """创建策略配置请求"""
    config_name: str = Field(..., description="配置名称")
    strategy_name: str = Field(..., description="策略名称")
    parameters: Dict[str, Any] = Field(..., description="策略参数")
    description: Optional[str] = Field(None, description="配置描述")
    user_id: Optional[str] = Field(None, description="用户ID")


class StrategyConfigUpdate(BaseModel):
    """更新策略配置请求"""
    config_name: Optional[str] = Field(None, description="配置名称")
    parameters: Optional[Dict[str, Any]] = Field(None, description="策略参数")
    description: Optional[str] = Field(None, description="配置描述")


@router.get("", response_model=StandardResponse)
async def get_strategy_configs(
    strategy_name: Optional[str] = Query(None, description="策略名称筛选"),
    user_id: Optional[str] = Query(None, description="用户ID筛选")
):
    """获取策略配置列表"""
    try:
        async with get_async_session() as session:
            query = select(StrategyConfig)
            conditions = []
            
            if strategy_name:
                conditions.append(StrategyConfig.strategy_name == strategy_name)
            if user_id:
                conditions.append(StrategyConfig.user_id == user_id)
            
            if conditions:
                query = query.where(and_(*conditions))
            
            query = query.order_by(StrategyConfig.created_at.desc())
            result = await session.execute(query)
            configs = result.scalars().all()
            
            return StandardResponse(
                success=True,
                message="获取策略配置列表成功",
                data={
                    "configs": [config.to_dict() for config in configs],
                    "total_count": len(configs)
                }
            )
    except Exception as e:
        logger.error(f"获取策略配置列表失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取策略配置列表失败: {str(e)}")


@router.get("/{config_id}", response_model=StandardResponse)
async def get_strategy_config(config_id: str):
    """获取特定配置详情"""
    try:
        async with get_async_session() as session:
            result = await session.execute(
                select(StrategyConfig).where(StrategyConfig.config_id == config_id)
            )
            config = result.scalar_one_or_none()
            
            if not config:
                raise HTTPException(status_code=404, detail="配置不存在")
            
            return StandardResponse(
                success=True,
                message="获取配置详情成功",
                data=config.to_dict()
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取配置详情失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取配置详情失败: {str(e)}")


@router.post("", response_model=StandardResponse)
async def create_strategy_config(request: StrategyConfigCreate):
    """保存新配置"""
    try:
        async with get_async_session() as session:
            # 检查同一策略下是否已有同名配置
            existing = await session.execute(
                select(StrategyConfig).where(
                    and_(
                        StrategyConfig.strategy_name == request.strategy_name,
                        StrategyConfig.config_name == request.config_name
                    )
                )
            )
            if existing.scalar_one_or_none():
                raise HTTPException(
                    status_code=400,
                    detail=f"策略 {request.strategy_name} 下已存在名为 {request.config_name} 的配置"
                )
            
            # 创建新配置
            config = StrategyConfig(
                config_name=request.config_name,
                strategy_name=request.strategy_name,
                parameters=request.parameters,
                description=request.description,
                user_id=request.user_id
            )
            
            session.add(config)
            await session.commit()
            await session.refresh(config)
            
            return StandardResponse(
                success=True,
                message="保存配置成功",
                data=config.to_dict()
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"保存配置失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"保存配置失败: {str(e)}")


@router.put("/{config_id}", response_model=StandardResponse)
async def update_strategy_config(
    config_id: str,
    request: StrategyConfigUpdate
):
    """更新配置"""
    try:
        async with get_async_session() as session:
            result = await session.execute(
                select(StrategyConfig).where(StrategyConfig.config_id == config_id)
            )
            config = result.scalar_one_or_none()
            
            if not config:
                raise HTTPException(status_code=404, detail="配置不存在")
            
            # 更新字段
            if request.config_name is not None:
                config.config_name = request.config_name
            if request.parameters is not None:
                config.parameters = request.parameters
            if request.description is not None:
                config.description = request.description
            
            await session.commit()
            await session.refresh(config)
            
            return StandardResponse(
                success=True,
                message="更新配置成功",
                data=config.to_dict()
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新配置失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"更新配置失败: {str(e)}")


@router.delete("/{config_id}", response_model=StandardResponse)
async def delete_strategy_config(config_id: str):
    """删除配置"""
    try:
        async with get_async_session() as session:
            result = await session.execute(
                select(StrategyConfig).where(StrategyConfig.config_id == config_id)
            )
            config = result.scalar_one_or_none()
            
            if not config:
                raise HTTPException(status_code=404, detail="配置不存在")
            
            await session.delete(config)
            await session.commit()
            
            return StandardResponse(
                success=True,
                message="删除配置成功"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除配置失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"删除配置失败: {str(e)}")

