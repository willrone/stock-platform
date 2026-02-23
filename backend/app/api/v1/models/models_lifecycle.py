"""
模型生命周期管理路由
负责模型生命周期、血缘关系和依赖管理
"""

from typing import Optional

from fastapi import APIRouter, HTTPException
from loguru import logger

from app.api.v1.schemas import StandardResponse
from app.services.models.lineage_tracker import lineage_tracker
from app.services.models.model_lifecycle_manager import model_lifecycle_manager

router = APIRouter()

@router.get("/{model_id}/lifecycle", response_model=StandardResponse)
async def get_model_lifecycle(model_id: str):
    """获取模型生命周期信息"""
    try:
        lifecycle_info = model_lifecycle_manager.get_model_lifecycle(model_id)

        if not lifecycle_info:
            raise HTTPException(status_code=404, detail=f"模型生命周期信息不存在: {model_id}")

        return StandardResponse(
            success=True, message="模型生命周期信息获取成功", data=lifecycle_info.to_dict()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取模型生命周期信息失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取模型生命周期信息失败: {str(e)}")



@router.get("/{model_id}/lineage", response_model=StandardResponse)
async def get_model_lineage(model_id: str):
    """获取模型血缘信息"""
    try:
        lineage_info = lineage_tracker.get_model_lineage(model_id)

        if not lineage_info:
            raise HTTPException(status_code=404, detail=f"模型血缘信息不存在: {model_id}")

        return StandardResponse(
            success=True, message="模型血缘信息获取成功", data=lineage_info.to_dict()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取模型血缘信息失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取模型血缘信息失败: {str(e)}")



@router.get("/{model_id}/dependencies", response_model=StandardResponse)
async def get_model_dependencies(model_id: str):
    """获取模型依赖关系"""
    try:
        # 获取血缘信息
        lineage_info = lineage_tracker.get_model_lineage(model_id)

        if not lineage_info:
            raise HTTPException(status_code=404, detail=f"模型依赖信息不存在: {model_id}")

        # 提取依赖关系
        dependencies = {
            "data_dependencies": lineage_info.data_dependencies,
            "feature_dependencies": lineage_info.feature_dependencies,
            "model_dependencies": lineage_info.model_dependencies,
            "config_dependencies": lineage_info.config_dependencies,
        }

        return StandardResponse(success=True, message="模型依赖关系获取成功", data=dependencies)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取模型依赖关系失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取模型依赖关系失败: {str(e)}")



@router.post("/{model_id}/lifecycle/transition", response_model=StandardResponse)
async def transition_model_lifecycle(
    model_id: str, new_stage: str, notes: Optional[str] = None
):
    """转换模型生命周期阶段"""
    try:
        # 验证阶段
        valid_stages = [
            "development",
            "testing",
            "staging",
            "production",
            "deprecated",
            "archived",
            "failed",
        ]

        if new_stage not in valid_stages:
            raise HTTPException(
                status_code=400,
                detail=f"无效的生命周期阶段: {new_stage}。有效阶段: {', '.join(valid_stages)}",
            )

        # 执行阶段转换
        success = model_lifecycle_manager.transition_stage(
            model_id=model_id, new_stage=new_stage, notes=notes
        )

        if not success:
            raise HTTPException(status_code=400, detail="生命周期阶段转换失败")

        # 获取更新后的生命周期信息
        updated_lifecycle = model_lifecycle_manager.get_model_lifecycle(model_id)

        return StandardResponse(
            success=True,
            message=f"模型生命周期已转换到: {new_stage}",
            data=updated_lifecycle.to_dict() if updated_lifecycle else {},
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"转换模型生命周期失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"转换模型生命周期失败: {str(e)}")

