"""
模型查询路由
负责模型列表、详情、搜索等查询功能
"""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException
from loguru import logger
from sqlalchemy import or_

from app.api.v1.schemas import StandardResponse
from app.core.database import SessionLocal
from app.models.task_models import ModelInfo
from app.repositories.task_repository import ModelInfoRepository

router = APIRouter()


@router.get("", response_model=StandardResponse)
async def list_models():
    """获取模型列表"""
    session = SessionLocal()
    try:
        model_repository = ModelInfoRepository(session)

        # 获取所有模型（包括training、failed等状态）
        models = session.query(ModelInfo).order_by(ModelInfo.created_at.desc()).all()

        # 转换为前端期望的格式
        model_list = []
        for model in models:
            # 安全地获取performance_metrics
            performance_metrics = model.performance_metrics
            if isinstance(performance_metrics, str):
                try:
                    import json

                    performance_metrics = json.loads(performance_metrics)
                except:
                    performance_metrics = {}
            if not isinstance(performance_metrics, dict):
                performance_metrics = {}

            accuracy = performance_metrics.get("accuracy", 0.0)
            if isinstance(accuracy, dict):
                accuracy = (
                    accuracy.get("value", 0.0) if isinstance(accuracy, dict) else 0.0
                )

            model_data = {
                "model_id": model.model_id,
                "model_name": model.model_name,
                "model_type": model.model_type,
                "version": model.version,
                "accuracy": float(accuracy) if accuracy else 0.0,
                "created_at": model.created_at.isoformat()
                if model.created_at
                else datetime.now().isoformat(),
                "status": model.status,
                "training_progress": model.training_progress or 0.0,
                "training_stage": model.training_stage,
            }
            model_list.append(model_data)

        return StandardResponse(
            success=True, message="模型列表获取成功", data={"models": model_list}
        )

    except Exception as e:
        logger.error(f"获取模型列表失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取模型列表失败: {str(e)}")
    finally:
        session.close()


@router.get("/{model_id}", response_model=StandardResponse)
async def get_model_detail(model_id: str):
    """获取模型详情"""
    session = SessionLocal()
    try:
        model_repository = ModelInfoRepository(session)
        model = model_repository.get_model_info(model_id)

        if not model:
            raise HTTPException(status_code=404, detail=f"模型不存在: {model_id}")

        # 转换为前端期望的格式
        performance_metrics = model.performance_metrics or {}
        if isinstance(performance_metrics, str):
            try:
                import json

                performance_metrics = json.loads(performance_metrics)
            except:
                performance_metrics = {}
        if not isinstance(performance_metrics, dict):
            performance_metrics = {}

        # 提取准确率（从performance_metrics或计算）
        accuracy = performance_metrics.get("accuracy", 0.0)
        if isinstance(accuracy, dict):
            accuracy = accuracy.get("value", 0.0) if isinstance(accuracy, dict) else 0.0

        training_data_period = {}
        if model.training_data_start and model.training_data_end:
            training_data_period = {
                "start": model.training_data_start.isoformat(),
                "end": model.training_data_end.isoformat(),
            }

        # 从evaluation_report中提取stock_codes
        stock_codes = []
        if model.evaluation_report and isinstance(model.evaluation_report, dict):
            training_data_info = model.evaluation_report.get("training_data_info", {})
            if isinstance(training_data_info, dict):
                stock_codes = training_data_info.get("stock_codes", [])

        model_detail = {
            "model_id": model.model_id,
            "model_name": model.model_name,
            "model_type": model.model_type,
            "version": model.version,
            "accuracy": float(accuracy) if accuracy else 0.0,
            "description": f"{model.model_type}模型 - {model.model_name}",
            "performance_metrics": performance_metrics,
            "training_info": {
                "training_data_period": training_data_period,
                "hyperparameters": model.hyperparameters or {},
                "stock_codes": stock_codes,
            },
            "created_at": model.created_at.isoformat()
            if model.created_at
            else datetime.now().isoformat(),
            "status": model.status,
        }

        return StandardResponse(success=True, message="模型详情获取成功", data=model_detail)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取模型详情失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取模型详情失败: {str(e)}")
    finally:
        session.close()


@router.get("/{model_id}/versions", response_model=StandardResponse)
async def get_model_versions(model_id: str):
    """获取模型的所有版本"""
    session = SessionLocal()
    try:
        # 获取主模型
        model = session.query(ModelInfo).filter(ModelInfo.model_id == model_id).first()
        if not model:
            raise HTTPException(status_code=404, detail=f"模型不存在: {model_id}")

        # 获取所有版本（包括主模型本身）
        parent_id = model.parent_model_id or model_id
        versions = (
            session.query(ModelInfo)
            .filter(
                (ModelInfo.model_id == parent_id)
                | (ModelInfo.parent_model_id == parent_id)
            )
            .order_by(ModelInfo.created_at.desc())
            .all()
        )

        version_list = []
        for v in versions:
            version_list.append(
                {
                    "model_id": v.model_id,
                    "model_name": v.model_name,
                    "version": v.version,
                    "status": v.status,
                    "accuracy": (v.performance_metrics or {}).get("accuracy", 0.0),
                    "created_at": v.created_at.isoformat() if v.created_at else None,
                    "is_current": v.model_id == model_id,
                }
            )

        return StandardResponse(
            success=True, message="模型版本列表获取成功", data={"versions": version_list}
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取模型版本失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取模型版本失败: {str(e)}")
    finally:
        session.close()


@router.get("/search", response_model=StandardResponse)
async def search_models(
    query: Optional[str] = None,
    model_type: Optional[str] = None,
    status: Optional[str] = None,
    min_accuracy: Optional[float] = None,
    tags: Optional[str] = None,
    limit: int = 50,
):
    """搜索模型"""
    session = SessionLocal()
    try:
        # 构建查询
        query_filter = session.query(ModelInfo)

        # 文本搜索
        if query:
            query_filter = query_filter.filter(
                or_(
                    ModelInfo.model_name.contains(query),
                    ModelInfo.model_type.contains(query),
                )
            )

        # 模型类型过滤
        if model_type:
            query_filter = query_filter.filter(ModelInfo.model_type == model_type)

        # 状态过滤
        if status:
            query_filter = query_filter.filter(ModelInfo.status == status)

        # 准确率过滤
        if min_accuracy is not None:
            # 这里需要处理JSON字段的查询，简化处理
            models = query_filter.all()
            filtered_models = []
            for model in models:
                performance_metrics = model.performance_metrics or {}
                if isinstance(performance_metrics, str):
                    try:
                        import json

                        performance_metrics = json.loads(performance_metrics)
                    except:
                        performance_metrics = {}

                accuracy = performance_metrics.get("accuracy", 0.0)
                if isinstance(accuracy, dict):
                    accuracy = accuracy.get("value", 0.0)

                if float(accuracy) >= min_accuracy:
                    filtered_models.append(model)

            models = filtered_models
        else:
            models = query_filter.limit(limit).all()

        # 转换为返回格式
        model_list = []
        for model in models[:limit]:
            performance_metrics = model.performance_metrics or {}
            if isinstance(performance_metrics, str):
                try:
                    import json

                    performance_metrics = json.loads(performance_metrics)
                except:
                    performance_metrics = {}

            accuracy = performance_metrics.get("accuracy", 0.0)
            if isinstance(accuracy, dict):
                accuracy = accuracy.get("value", 0.0)

            model_data = {
                "model_id": model.model_id,
                "model_name": model.model_name,
                "model_type": model.model_type,
                "version": model.version,
                "accuracy": float(accuracy) if accuracy else 0.0,
                "status": model.status,
                "created_at": model.created_at.isoformat()
                if model.created_at
                else None,
                "performance_metrics": performance_metrics,
            }
            model_list.append(model_data)

        return StandardResponse(
            success=True,
            message=f"模型搜索完成，找到 {len(model_list)} 个结果",
            data={
                "models": model_list,
                "total_count": len(model_list),
                "search_params": {
                    "query": query,
                    "model_type": model_type,
                    "status": status,
                    "min_accuracy": min_accuracy,
                    "tags": tags,
                    "limit": limit,
                },
            },
        )

    except Exception as e:
        logger.error(f"模型搜索失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"模型搜索失败: {str(e)}")
    finally:
        session.close()
