"""
Qlib集成API接口

提供Qlib数据处理、因子计算和模型训练的API接口
"""

from datetime import datetime
from typing import List, Dict, Any, Optional
from loguru import logger

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.api.v1.schemas import StandardResponse
from app.services.qlib.enhanced_qlib_provider import EnhancedQlibDataProvider
from app.services.qlib.unified_qlib_training_engine import UnifiedQlibTrainingEngine

router = APIRouter(prefix="/qlib", tags=["Qlib集成"])

# 全局Qlib数据提供器和训练引擎实例
_qlib_provider: Optional[EnhancedQlibDataProvider] = None
_training_engine: Optional[UnifiedQlibTrainingEngine] = None


def get_qlib_provider() -> EnhancedQlibDataProvider:
    """获取Qlib数据提供器实例"""
    global _qlib_provider
    if _qlib_provider is None:
        _qlib_provider = EnhancedQlibDataProvider()
    return _qlib_provider


def get_training_engine() -> UnifiedQlibTrainingEngine:
    """获取统一训练引擎实例"""
    global _training_engine
    if _training_engine is None:
        _training_engine = UnifiedQlibTrainingEngine()
    return _training_engine


class QlibDatasetRequest(BaseModel):
    """Qlib数据集准备请求"""
    stock_codes: List[str]
    start_date: str
    end_date: str
    include_alpha_factors: bool = True
    use_cache: bool = True


class AlphaFactorsRequest(BaseModel):
    """Alpha因子计算请求"""
    stock_codes: List[str]
    start_date: str
    end_date: str
    use_cache: bool = True


class QlibModelConfigRequest(BaseModel):
    """Qlib模型配置请求"""
    model_type: str
    hyperparameters: Dict[str, Any] = {}


class ModelRecommendationRequest(BaseModel):
    """模型推荐请求"""
    sample_count: int
    feature_count: int
    task_type: str = "regression"


@router.post("/dataset/prepare", response_model=StandardResponse)
async def prepare_qlib_dataset(request: QlibDatasetRequest):
    """准备Qlib标准格式的数据集"""
    try:
        provider = get_qlib_provider()
        
        # 解析日期
        start_date = datetime.fromisoformat(request.start_date)
        end_date = datetime.fromisoformat(request.end_date)
        
        # 准备数据集
        dataset = await provider.prepare_qlib_dataset(
            stock_codes=request.stock_codes,
            start_date=start_date,
            end_date=end_date,
            include_alpha_factors=request.include_alpha_factors,
            use_cache=request.use_cache
        )
        
        # 验证数据格式
        is_valid = await provider.validate_qlib_data_format(dataset)
        
        return StandardResponse(
            success=True,
            message="Qlib数据集准备完成",
            data={
                "dataset_shape": dataset.shape,
                "columns": list(dataset.columns),
                "index_levels": len(dataset.index.names) if hasattr(dataset.index, 'names') else 1,
                "is_valid_format": is_valid,
                "sample_data": dataset.head().to_dict() if not dataset.empty else {}
            }
        )
        
    except Exception as e:
        logger.error(f"准备Qlib数据集失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"准备Qlib数据集失败: {str(e)}")


@router.post("/factors/alpha158", response_model=StandardResponse)
async def calculate_alpha158_factors(request: AlphaFactorsRequest):
    """计算Alpha158因子"""
    try:
        provider = get_qlib_provider()
        
        # 解析日期
        start_date = datetime.fromisoformat(request.start_date)
        end_date = datetime.fromisoformat(request.end_date)
        
        # 首先准备基础数据
        base_dataset = await provider.prepare_qlib_dataset(
            stock_codes=request.stock_codes,
            start_date=start_date,
            end_date=end_date,
            include_alpha_factors=False,  # 不包含Alpha因子
            use_cache=request.use_cache
        )
        
        if base_dataset.empty:
            raise HTTPException(status_code=404, detail="无法获取基础数据")
        
        # 计算Alpha158因子
        alpha_factors = await provider.alpha_calculator.calculate_alpha_factors(
            qlib_data=base_dataset,
            stock_codes=request.stock_codes,
            date_range=(start_date, end_date),
            use_cache=request.use_cache
        )
        
        return StandardResponse(
            success=True,
            message="Alpha158因子计算完成",
            data={
                "factors_shape": alpha_factors.shape,
                "factor_names": list(alpha_factors.columns),
                "sample_factors": alpha_factors.head().to_dict() if not alpha_factors.empty else {},
                "cache_used": request.use_cache
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"计算Alpha158因子失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"计算Alpha158因子失败: {str(e)}")


@router.post("/model/config", response_model=StandardResponse)
async def create_qlib_model_config(request: QlibModelConfigRequest):
    """创建Qlib模型配置"""
    try:
        provider = get_qlib_provider()
        
        # 创建模型配置
        config = await provider.create_qlib_model_config(
            model_type=request.model_type,
            hyperparameters=request.hyperparameters
        )
        
        return StandardResponse(
            success=True,
            message="Qlib模型配置创建成功",
            data={
                "model_config": config,
                "model_type": request.model_type,
                "hyperparameters": request.hyperparameters
            }
        )
        
    except Exception as e:
        logger.error(f"创建Qlib模型配置失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"创建Qlib模型配置失败: {str(e)}")


@router.get("/cache/stats", response_model=StandardResponse)
async def get_cache_stats():
    """获取缓存统计信息"""
    try:
        provider = get_qlib_provider()
        stats = await provider.get_cache_stats()
        
        return StandardResponse(
            success=True,
            message="缓存统计信息获取成功",
            data=stats
        )
        
    except Exception as e:
        logger.error(f"获取缓存统计失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取缓存统计失败: {str(e)}")


@router.delete("/cache/clear", response_model=StandardResponse)
async def clear_cache():
    """清空缓存"""
    try:
        provider = get_qlib_provider()
        await provider.clear_cache()
        
        return StandardResponse(
            success=True,
            message="缓存清空成功",
            data={}
        )
        
    except Exception as e:
        logger.error(f"清空缓存失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"清空缓存失败: {str(e)}")


@router.get("/status", response_model=StandardResponse)
async def get_qlib_status():
    """获取Qlib集成状态"""
    try:
        provider = get_qlib_provider()
        
        # 尝试初始化Qlib
        try:
            await provider.initialize_qlib()
            qlib_initialized = True
            qlib_error = None
        except Exception as e:
            qlib_initialized = False
            qlib_error = str(e)
        
        # 获取缓存统计
        cache_stats = await provider.get_cache_stats()
        
        return StandardResponse(
            success=True,
            message="Qlib状态获取成功",
            data={
                "qlib_available": cache_stats.get("qlib_available", False),
                "qlib_initialized": qlib_initialized,
                "qlib_error": qlib_error,
                "cache_stats": cache_stats,
                "supported_models": ["lightgbm", "xgboost", "mlp", "linear"],
                "alpha_factors_count": len(provider.alpha_calculator.alpha_expressions)
            }
        )
        
    except Exception as e:
        logger.error(f"获取Qlib状态失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取Qlib状态失败: {str(e)}")


@router.get("/factors/list", response_model=StandardResponse)
async def list_alpha_factors():
    """获取支持的Alpha因子列表"""
    try:
        provider = get_qlib_provider()
        
        factors = provider.alpha_calculator.alpha_expressions
        factor_info = []
        
        for factor_name, expression in factors.items():
            # 解析因子类型
            if any(keyword in factor_name for keyword in ['RESI', 'MA']):
                factor_type = "价格相关"
            elif any(keyword in factor_name for keyword in ['STD', 'VSTD']):
                factor_type = "波动率"
            elif 'CORR' in factor_name:
                factor_type = "相关性"
            elif any(keyword in factor_name for keyword in ['MAX', 'MIN']):
                factor_type = "极值"
            elif 'QTLU' in factor_name:
                factor_type = "分位数"
            else:
                factor_type = "其他"
            
            factor_info.append({
                "name": factor_name,
                "expression": expression,
                "type": factor_type,
                "description": f"{factor_type}因子 - {factor_name}"
            })
        
        return StandardResponse(
            success=True,
            message="Alpha因子列表获取成功",
            data={
                "total_factors": len(factor_info),
                "factors": factor_info,
                "factor_types": list(set(f["type"] for f in factor_info))
            }
        )
        
    except Exception as e:
        logger.error(f"获取Alpha因子列表失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取Alpha因子列表失败: {str(e)}")


@router.get("/models/supported", response_model=StandardResponse)
async def get_supported_models():
    """获取支持的模型列表"""
    try:
        engine = get_training_engine()
        
        supported_models = engine.get_supported_model_types()
        models_info = []
        
        for model_name in supported_models:
            metadata = engine.model_manager.get_model_metadata(model_name)
            if metadata:
                models_info.append({
                    "name": model_name,
                    "display_name": metadata.display_name,
                    "category": metadata.category.value,
                    "complexity": metadata.complexity.value,
                    "description": metadata.description,
                    "supported_tasks": metadata.supported_tasks,
                    "min_samples": metadata.min_samples,
                    "recommended_features": metadata.recommended_features,
                    "training_time_estimate": metadata.training_time_estimate,
                    "memory_requirement": metadata.memory_requirement
                })
        
        return StandardResponse(
            success=True,
            message="支持的模型列表获取成功",
            data={
                "total_models": len(models_info),
                "models": models_info,
                "categories": list(set(m["category"] for m in models_info)),
                "complexities": list(set(m["complexity"] for m in models_info))
            }
        )
        
    except Exception as e:
        logger.error(f"获取支持的模型列表失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取支持的模型列表失败: {str(e)}")


@router.get("/models/{model_name}/config", response_model=StandardResponse)
async def get_model_config_template(model_name: str):
    """获取模型配置模板"""
    try:
        engine = get_training_engine()
        
        template = engine.get_model_config_template(model_name)
        if not template:
            raise HTTPException(status_code=404, detail=f"不支持的模型类型: {model_name}")
        
        return StandardResponse(
            success=True,
            message="模型配置模板获取成功",
            data=template
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取模型配置模板失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取模型配置模板失败: {str(e)}")


@router.get("/models/{model_name}/hyperparameters", response_model=StandardResponse)
async def get_model_hyperparameters(model_name: str):
    """获取模型超参数规格"""
    try:
        engine = get_training_engine()
        
        hyperparameter_specs = engine.model_manager.get_hyperparameter_specs(model_name)
        if not hyperparameter_specs:
            raise HTTPException(status_code=404, detail=f"不支持的模型类型: {model_name}")
        
        specs_data = []
        for spec in hyperparameter_specs:
            specs_data.append({
                "name": spec.name,
                "type": spec.param_type,
                "default_value": spec.default_value,
                "min_value": spec.min_value,
                "max_value": spec.max_value,
                "choices": spec.choices,
                "description": spec.description
            })
        
        return StandardResponse(
            success=True,
            message="模型超参数规格获取成功",
            data={
                "model_name": model_name,
                "hyperparameters": specs_data
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取模型超参数规格失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取模型超参数规格失败: {str(e)}")


@router.post("/models/recommend", response_model=StandardResponse)
async def recommend_models(request: ModelRecommendationRequest):
    """推荐适合的模型"""
    try:
        engine = get_training_engine()
        
        recommendations = engine.recommend_models(
            sample_count=request.sample_count,
            feature_count=request.feature_count,
            task_type=request.task_type
        )
        
        # 获取推荐模型的详细信息
        recommended_models = []
        for model_name in recommendations:
            metadata = engine.model_manager.get_model_metadata(model_name)
            training_recommendations = engine.get_training_recommendations(model_name)
            
            if metadata:
                recommended_models.append({
                    "name": model_name,
                    "display_name": metadata.display_name,
                    "category": metadata.category.value,
                    "complexity": metadata.complexity.value,
                    "description": metadata.description,
                    "training_recommendations": training_recommendations,
                    "suitability_score": len(recommendations) - recommendations.index(model_name)  # 简单的适合度评分
                })
        
        return StandardResponse(
            success=True,
            message="模型推荐完成",
            data={
                "request_info": {
                    "sample_count": request.sample_count,
                    "feature_count": request.feature_count,
                    "task_type": request.task_type
                },
                "recommended_models": recommended_models,
                "total_recommendations": len(recommended_models)
            }
        )
        
    except Exception as e:
        logger.error(f"模型推荐失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"模型推荐失败: {str(e)}")


@router.get("/models/{model_name}/training-tips", response_model=StandardResponse)
async def get_training_tips(model_name: str):
    """获取模型训练建议"""
    try:
        engine = get_training_engine()
        
        recommendations = engine.get_training_recommendations(model_name)
        if not recommendations:
            raise HTTPException(status_code=404, detail=f"不支持的模型类型: {model_name}")
        
        return StandardResponse(
            success=True,
            message="训练建议获取成功",
            data={
                "model_name": model_name,
                "recommendations": recommendations
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取训练建议失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取训练建议失败: {str(e)}")