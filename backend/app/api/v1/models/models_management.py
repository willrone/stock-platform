"""
模型管理路由
负责模型删除和标签管理
"""

from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException
from loguru import logger

from app.api.v1.schemas import StandardResponse
from app.core.database import SessionLocal
from app.models.task_models import ModelInfo
from app.repositories.task_repository import ModelInfoRepository

router = APIRouter()


@router.delete("/{model_id}", response_model=StandardResponse)
async def delete_model(model_id: str):
    """删除模型"""
    session = SessionLocal()
    try:
        model_repository = ModelInfoRepository(session)
        model = model_repository.get_model_info(model_id)

        if not model:
            raise HTTPException(status_code=404, detail=f"模型不存在: {model_id}")

        # 删除模型文件（如果存在）
        if model.file_path:
            try:
                model_file = Path(model.file_path)
                if model_file.exists():
                    model_file.unlink()
                    logger.info(f"已删除模型文件: {model.file_path}")
            except Exception as e:
                logger.warning(f"删除模型文件失败: {e}，继续删除数据库记录")

        # 删除数据库记录
        session.delete(model)
        session.commit()

        logger.info(f"模型删除成功: {model_id}")

        return StandardResponse(
            success=True, message="模型删除成功", data={"model_id": model_id}
        )

    except HTTPException:
        raise
    except Exception as e:
        session.rollback()
        logger.error(f"删除模型失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"删除模型失败: {str(e)}")
    finally:
        session.close()


@router.post("/{model_id}/tags", response_model=StandardResponse)
async def add_model_tags(model_id: str, tags: List[str]):
    """为模型添加标签"""
    session = SessionLocal()
    try:
        model = session.query(ModelInfo).filter(ModelInfo.model_id == model_id).first()
        if not model:
            raise HTTPException(status_code=404, detail=f"模型不存在: {model_id}")

        # 获取现有标签
        existing_tags = (
            model.hyperparameters.get("tags", []) if model.hyperparameters else []
        )

        # 合并标签（去重）
        all_tags = list(set(existing_tags + tags))

        # 更新模型标签
        if not model.hyperparameters:
            model.hyperparameters = {}
        model.hyperparameters["tags"] = all_tags

        session.commit()

        return StandardResponse(
            success=True,
            message=f"成功为模型添加标签: {', '.join(tags)}",
            data={"model_id": model_id, "tags": all_tags, "added_tags": tags},
        )

    except HTTPException:
        raise
    except Exception as e:
        session.rollback()
        logger.error(f"添加模型标签失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"添加模型标签失败: {str(e)}")
    finally:
        session.close()


@router.delete("/{model_id}/tags", response_model=StandardResponse)
async def remove_model_tags(model_id: str, tags: List[str]):
    """移除模型标签"""
    session = SessionLocal()
    try:
        model = session.query(ModelInfo).filter(ModelInfo.model_id == model_id).first()
        if not model:
            raise HTTPException(status_code=404, detail=f"模型不存在: {model_id}")

        # 获取现有标签
        existing_tags = (
            model.hyperparameters.get("tags", []) if model.hyperparameters else []
        )

        # 移除指定标签
        remaining_tags = [tag for tag in existing_tags if tag not in tags]

        # 更新模型标签
        if not model.hyperparameters:
            model.hyperparameters = {}
        model.hyperparameters["tags"] = remaining_tags

        session.commit()

        return StandardResponse(
            success=True,
            message=f"成功移除模型标签: {', '.join(tags)}",
            data={"model_id": model_id, "tags": remaining_tags, "removed_tags": tags},
        )

    except HTTPException:
        raise
    except Exception as e:
        session.rollback()
        logger.error(f"移除模型标签失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"移除模型标签失败: {str(e)}")
    finally:
        session.close()
