"""
策略配置管理API
"""

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from loguru import logger
import json
import uuid

from app.core.database import AsyncSessionLocal
from app.api.v1.schemas import StandardResponse
from app.models.strategy_config_models import StrategyConfig

router = APIRouter(prefix="/strategy-configs", tags=["策略配置"])


def clean_parameters(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    清理参数，确保所有值都是JSON可序列化的
    将numpy类型、datetime等转换为Python原生类型
    """
    cleaned = {}
    for key, value in parameters.items():
        # 处理numpy类型
        if hasattr(value, 'item'):  # numpy标量类型
            cleaned[key] = value.item()
        # 处理numpy数组
        elif hasattr(value, 'tolist'):  # numpy数组
            cleaned[key] = value.tolist()
        # 处理列表和字典（递归清理）
        elif isinstance(value, dict):
            cleaned[key] = clean_parameters(value)
        elif isinstance(value, list):
            cleaned[key] = [
                item.item() if hasattr(item, 'item') 
                else item.tolist() if hasattr(item, 'tolist')
                else clean_parameters(item) if isinstance(item, dict)
                else item
                for item in value
            ]
        # 其他类型直接使用
        else:
            cleaned[key] = value
    
    return cleaned


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
        async with AsyncSessionLocal() as session:
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
        async with AsyncSessionLocal() as session:
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
        logger.info(f"收到保存配置请求: strategy_name={request.strategy_name}, config_name={request.config_name}")
        logger.debug(f"原始参数: {request.parameters}, 参数类型: {type(request.parameters)}")
        
        # 清理参数，确保JSON可序列化
        cleaned_parameters = clean_parameters(request.parameters)
        logger.debug(f"清理后的参数: {cleaned_parameters}, 类型: {type(cleaned_parameters)}")
        
        # 验证参数可以JSON序列化
        try:
            json_str = json.dumps(cleaned_parameters, ensure_ascii=False)
            logger.debug(f"参数JSON序列化成功，长度: {len(json_str)}")
        except (TypeError, ValueError) as e:
            logger.error(f"参数无法JSON序列化: {str(e)}, 参数类型: {type(cleaned_parameters)}, 参数: {cleaned_parameters}")
            raise HTTPException(
                status_code=400,
                detail=f"参数包含无法序列化的数据类型: {str(e)}"
            )
        
        async with AsyncSessionLocal() as session:
            try:
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
                
                # 生成config_id
                config_id = str(uuid.uuid4())
                logger.debug(f"生成的config_id: {config_id}")
                
                # 创建新配置
                config = StrategyConfig(
                    config_id=config_id,
                    config_name=request.config_name,
                    strategy_name=request.strategy_name,
                    parameters=cleaned_parameters,
                    description=request.description,
                    user_id=request.user_id
                )
                
                logger.debug(f"准备保存配置对象: config_id={config_id}, parameters类型={type(cleaned_parameters)}")
                session.add(config)
                
                # 刷新以触发任何默认值
                await session.flush()
                logger.debug(f"配置对象已刷新，config_id={config.config_id}")
                
                await session.commit()
                logger.info(f"配置保存成功: config_id={config_id}")
                
                await session.refresh(config)
                
                return StandardResponse(
                    success=True,
                    message="保存配置成功",
                    data=config.to_dict()
                )
            except HTTPException:
                await session.rollback()
                raise
            except Exception as db_error:
                await session.rollback()
                logger.error(f"数据库操作失败: {str(db_error)}, 错误类型: {type(db_error).__name__}", exc_info=True)
                # 检查是否是表不存在错误
                error_str = str(db_error).lower()
                if "no such table" in error_str or "table" in error_str and "does not exist" in error_str:
                    logger.error("策略配置表不存在，请运行数据库迁移")
                    raise HTTPException(
                        status_code=500,
                        detail="数据库表不存在，请联系管理员运行数据库迁移"
                    )
                raise
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"保存配置失败: {str(e)}, 错误类型: {type(e).__name__}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"保存配置失败: {str(e)}")


@router.put("/{config_id}", response_model=StandardResponse)
async def update_strategy_config(
    config_id: str,
    request: StrategyConfigUpdate
):
    """更新配置"""
    try:
        async with AsyncSessionLocal() as session:
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
                # 清理参数，确保JSON可序列化
                cleaned_parameters = clean_parameters(request.parameters)
                # 验证参数可以JSON序列化
                try:
                    json.dumps(cleaned_parameters)
                except (TypeError, ValueError) as e:
                    logger.error(f"参数无法JSON序列化: {str(e)}, 参数: {cleaned_parameters}")
                    raise HTTPException(
                        status_code=400,
                        detail=f"参数包含无法序列化的数据类型: {str(e)}"
                    )
                config.parameters = cleaned_parameters
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
        async with AsyncSessionLocal() as session:
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

