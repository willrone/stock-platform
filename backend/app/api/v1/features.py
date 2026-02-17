"""
特征管理API路由
添加特征计算和查询接口，支持技术指标配置管理
"""
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.api.v1.schemas import StandardResponse
from app.services.explainability.technical_analyzer import (
    TechnicalIndicator,
    technical_analyzer,
)
from app.services.features.feature_pipeline import feature_pipeline
from app.services.features.feature_store import feature_store

router = APIRouter(prefix="/features", tags=["特征管理"])


# 请求模型
class FeatureComputeRequest(BaseModel):
    stock_codes: List[str]
    start_date: str
    end_date: str
    feature_types: List[str] = []
    technical_indicators: List[str] = []


class TechnicalIndicatorConfig(BaseModel):
    indicator_type: str
    parameters: Dict[str, Any] = {}
    enabled: bool = True


class FeatureStoreRequest(BaseModel):
    feature_name: str
    feature_data: Dict[str, Any]
    metadata: Dict[str, Any] = {}
    tags: List[str] = []


class FeatureQueryRequest(BaseModel):
    stock_codes: Optional[List[str]] = None
    feature_names: Optional[List[str]] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    tags: Optional[List[str]] = None


@router.get(
    "/indicators/available", response_model=StandardResponse, summary="获取可用技术指标"
)
async def get_available_indicators():
    """获取所有可用的技术指标类型"""
    try:
        indicators = [
            {
                "name": indicator.value,
                "display_name": indicator.name,
                "category": "technical",
                "description": f"{indicator.name} 技术指标",
            }
            for indicator in TechnicalIndicator
        ]

        return StandardResponse(
            success=True,
            message=f"成功获取可用技术指标: {len(indicators)} 个",
            data={"indicators": indicators, "total_count": len(indicators)},
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取可用指标失败: {str(e)}")


@router.post("/compute", response_model=StandardResponse, summary="计算特征")
async def compute_features(request: FeatureComputeRequest):
    """计算股票特征"""
    try:
        # 解析日期
        start_date = datetime.strptime(request.start_date, "%Y-%m-%d").date()
        end_date = datetime.strptime(request.end_date, "%Y-%m-%d").date()

        # 验证日期范围
        if start_date >= end_date:
            raise HTTPException(status_code=400, detail="开始日期必须早于结束日期")

        # 计算特征
        results = {}

        for stock_code in request.stock_codes:
            try:
                # 触发特征计算
                feature_data = feature_pipeline.compute_features_for_stock(
                    stock_code=stock_code,
                    start_date=start_date,
                    end_date=end_date,
                    feature_types=request.feature_types,
                    technical_indicators=request.technical_indicators,
                )

                results[stock_code] = {
                    "success": True,
                    "feature_count": len(feature_data) if feature_data else 0,
                    "computed_features": list(feature_data.keys())
                    if feature_data
                    else [],
                }

            except Exception as e:
                results[stock_code] = {
                    "success": False,
                    "error": str(e),
                    "feature_count": 0,
                }

        # 统计结果
        successful_stocks = [k for k, v in results.items() if v["success"]]
        failed_stocks = [k for k, v in results.items() if not v["success"]]

        return StandardResponse(
            success=len(successful_stocks) > 0,
            message=f"特征计算完成: 成功 {len(successful_stocks)} 只股票，失败 {len(failed_stocks)} 只股票",
            data={
                "results": results,
                "summary": {
                    "total_stocks": len(request.stock_codes),
                    "successful_stocks": len(successful_stocks),
                    "failed_stocks": len(failed_stocks),
                    "success_rate": len(successful_stocks)
                    / len(request.stock_codes)
                    * 100,
                },
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"计算特征失败: {str(e)}")


@router.get(
    "/technical-indicators/{stock_code}",
    response_model=StandardResponse,
    summary="计算技术指标",
)
async def compute_technical_indicators(
    stock_code: str,
    start_date: str = Query(..., description="开始日期 (YYYY-MM-DD)"),
    end_date: str = Query(..., description="结束日期 (YYYY-MM-DD)"),
    indicators: str = Query(None, description="指标列表，逗号分隔"),
):
    """计算股票的技术指标"""
    try:
        # 解析日期
        start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()

        # 解析指标列表
        indicator_list = []
        if indicators:
            indicator_list = [ind.strip() for ind in indicators.split(",")]

        # 获取价格数据（这里需要从数据服务获取）
        # 暂时使用模拟数据
        import numpy as np
        import pandas as pd

        date_range = pd.date_range(start=start_dt, end=end_dt, freq="D")
        price_data = pd.DataFrame(
            {
                "open": np.random.randn(len(date_range)).cumsum() + 100,
                "high": np.random.randn(len(date_range)).cumsum() + 105,
                "low": np.random.randn(len(date_range)).cumsum() + 95,
                "close": np.random.randn(len(date_range)).cumsum() + 100,
            },
            index=date_range,
        )

        # 计算技术指标
        indicators_data = technical_analyzer.calculate_all_indicators(price_data)

        # 过滤指定的指标
        if indicator_list:
            available_indicators = [
                col
                for col in indicators_data.columns
                if any(ind in col for ind in indicator_list)
            ]
            indicators_data = indicators_data[available_indicators]

        # 转换为字典格式
        result_data = {}
        for column in indicators_data.columns:
            result_data[column] = {
                "values": indicators_data[column].dropna().to_dict(),
                "latest_value": float(indicators_data[column].dropna().iloc[-1])
                if not indicators_data[column].dropna().empty
                else None,
                "data_points": len(indicators_data[column].dropna()),
            }

        return StandardResponse(
            success=True,
            message=f"成功计算技术指标: {len(result_data)} 个指标",
            data={
                "stock_code": stock_code,
                "period": f"{start_date} to {end_date}",
                "indicators": result_data,
                "total_indicators": len(result_data),
            },
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"计算技术指标失败: {str(e)}")


@router.post("/store", response_model=StandardResponse, summary="存储特征")
async def store_feature(request: FeatureStoreRequest):
    """存储特征到特征存储"""
    try:
        # 存储特征
        feature_id = feature_store.store_feature(
            feature_name=request.feature_name,
            feature_data=request.feature_data,
            metadata=request.metadata,
            tags=request.tags,
        )

        return StandardResponse(
            success=True,
            message=f"成功存储特征: {request.feature_name}",
            data={
                "feature_id": feature_id,
                "feature_name": request.feature_name,
                "data_size": len(request.feature_data),
                "tags": request.tags,
            },
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"存储特征失败: {str(e)}")


@router.post("/query", response_model=StandardResponse, summary="查询特征")
async def query_features(request: FeatureQueryRequest):
    """查询特征数据"""
    try:
        # 解析日期
        start_date = None
        end_date = None

        if request.start_date:
            start_date = datetime.strptime(request.start_date, "%Y-%m-%d").date()

        if request.end_date:
            end_date = datetime.strptime(request.end_date, "%Y-%m-%d").date()

        # 查询特征
        features = feature_store.query_features(
            stock_codes=request.stock_codes,
            feature_names=request.feature_names,
            start_date=start_date,
            end_date=end_date,
            tags=request.tags,
        )

        return StandardResponse(
            success=True,
            message=f"成功查询特征: {len(features)} 个特征",
            data={
                "features": features,
                "total_count": len(features),
                "query_params": {
                    "stock_codes": request.stock_codes,
                    "feature_names": request.feature_names,
                    "start_date": request.start_date,
                    "end_date": request.end_date,
                    "tags": request.tags,
                },
            },
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询特征失败: {str(e)}")


@router.get("/list", response_model=StandardResponse, summary="获取特征列表")
async def list_features(
    stock_code: Optional[str] = Query(None, description="股票代码过滤"),
    feature_type: Optional[str] = Query(None, description="特征类型过滤"),
    tags: Optional[str] = Query(None, description="标签过滤，逗号分隔"),
    limit: int = Query(100, description="返回数量限制"),
):
    """获取特征列表"""
    try:
        # 解析标签
        tag_list = None
        if tags:
            tag_list = [tag.strip() for tag in tags.split(",")]

        # 获取特征列表
        features = feature_store.list_features(
            stock_code=stock_code, feature_type=feature_type, tags=tag_list, limit=limit
        )

        return StandardResponse(
            success=True,
            message=f"成功获取特征列表: {len(features)} 个特征",
            data={
                "features": features,
                "total_count": len(features),
                "filters": {
                    "stock_code": stock_code,
                    "feature_type": feature_type,
                    "tags": tags,
                    "limit": limit,
                },
            },
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取特征列表失败: {str(e)}")


@router.get(
    "/metadata/{feature_name}", response_model=StandardResponse, summary="获取特征元数据"
)
async def get_feature_metadata(feature_name: str):
    """获取特征元数据"""
    try:
        metadata = feature_store.get_feature_metadata(feature_name)

        if not metadata:
            raise HTTPException(status_code=404, detail=f"特征不存在: {feature_name}")

        return StandardResponse(
            success=True,
            message="成功获取特征元数据",
            data={"feature_name": feature_name, "metadata": metadata},
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取特征元数据失败: {str(e)}")


@router.get("/stats", response_model=StandardResponse, summary="获取特征统计")
async def get_feature_stats():
    """获取特征存储统计信息"""
    try:
        stats = feature_store.get_stats()

        return StandardResponse(success=True, message="成功获取特征统计信息", data=stats)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取特征统计失败: {str(e)}")


@router.post("/indicators/config", response_model=StandardResponse, summary="配置技术指标")
async def configure_technical_indicator(config: TechnicalIndicatorConfig):
    """配置技术指标参数"""
    try:
        # 验证指标类型
        try:
            TechnicalIndicator(config.indicator_type)
        except ValueError:
            raise HTTPException(
                status_code=400, detail=f"无效的技术指标类型: {config.indicator_type}"
            )

        # 保存配置（这里应该保存到配置存储中）
        config_id = f"indicator_config_{int(datetime.now().timestamp())}"

        # 模拟保存配置
        saved_config = {
            "config_id": config_id,
            "indicator_type": config.indicator_type,
            "parameters": config.parameters,
            "enabled": config.enabled,
            "created_at": datetime.now().isoformat(),
        }

        return StandardResponse(
            success=True,
            message=f"成功配置技术指标: {config.indicator_type}",
            data=saved_config,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"配置技术指标失败: {str(e)}")


@router.get("/indicators/config", response_model=StandardResponse, summary="获取技术指标配置")
async def get_technical_indicator_configs(
    indicator_type: Optional[str] = Query(None, description="指标类型过滤"),
    enabled_only: bool = Query(True, description="只返回启用的配置"),
):
    """获取技术指标配置列表"""
    try:
        # 模拟配置数据
        configs = []

        for indicator in TechnicalIndicator:
            if indicator_type and indicator.value != indicator_type:
                continue

            config = {
                "config_id": f"config_{indicator.value}",
                "indicator_type": indicator.value,
                "parameters": {
                    "period": 14 if "rsi" in indicator.value else 20,
                    "enabled": True,
                },
                "enabled": True,
                "created_at": datetime.now().isoformat(),
            }

            if not enabled_only or config["enabled"]:
                configs.append(config)

        return StandardResponse(
            success=True,
            message=f"成功获取技术指标配置: {len(configs)} 个配置",
            data={
                "configs": configs,
                "total_count": len(configs),
                "filters": {
                    "indicator_type": indicator_type,
                    "enabled_only": enabled_only,
                },
            },
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取技术指标配置失败: {str(e)}")


@router.delete(
    "/features/{feature_name}", response_model=StandardResponse, summary="删除特征"
)
async def delete_feature(feature_name: str):
    """删除特征"""
    try:
        success = feature_store.delete_feature(feature_name)

        if not success:
            raise HTTPException(status_code=404, detail=f"特征不存在: {feature_name}")

        return StandardResponse(
            success=True,
            message=f"成功删除特征: {feature_name}",
            data={
                "feature_name": feature_name,
                "deleted_at": datetime.now().isoformat(),
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除特征失败: {str(e)}")


@router.post("/batch-compute", response_model=StandardResponse, summary="批量计算特征")
async def batch_compute_features(
    stock_codes: List[str],
    feature_types: List[str] = Query([], description="特征类型列表"),
    start_date: str = Query(..., description="开始日期 (YYYY-MM-DD)"),
    end_date: str = Query(..., description="结束日期 (YYYY-MM-DD)"),
    parallel: bool = Query(True, description="是否并行计算"),
):
    """批量计算多只股票的特征"""
    try:
        # 解析日期
        start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()

        # 批量计算特征
        batch_results = feature_pipeline.batch_compute_features(
            stock_codes=stock_codes,
            feature_types=feature_types,
            start_date=start_dt,
            end_date=end_dt,
            parallel=parallel,
        )

        # 统计结果
        successful_count = sum(
            1 for result in batch_results.values() if result.get("success", False)
        )
        failed_count = len(batch_results) - successful_count

        return StandardResponse(
            success=successful_count > 0,
            message=f"批量特征计算完成: 成功 {successful_count} 只，失败 {failed_count} 只",
            data={
                "results": batch_results,
                "summary": {
                    "total_stocks": len(stock_codes),
                    "successful_count": successful_count,
                    "failed_count": failed_count,
                    "success_rate": successful_count / len(stock_codes) * 100
                    if stock_codes
                    else 0,
                    "parallel_execution": parallel,
                },
            },
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"批量计算特征失败: {str(e)}")


@router.get("/pipeline/status", response_model=StandardResponse, summary="获取特征管道状态")
async def get_pipeline_status():
    """获取特征计算管道状态"""
    try:
        status = feature_pipeline.get_status()

        return StandardResponse(success=True, message="成功获取特征管道状态", data=status)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取管道状态失败: {str(e)}")


@router.post("/pipeline/trigger", response_model=StandardResponse, summary="触发特征计算")
async def trigger_feature_computation(
    stock_code: str,
    feature_types: List[str] = Query([], description="特征类型列表"),
    force_recompute: bool = Query(False, description="是否强制重新计算"),
):
    """手动触发特征计算"""
    try:
        # 触发特征计算
        task_id = feature_pipeline.trigger_computation(
            stock_code=stock_code,
            feature_types=feature_types,
            force_recompute=force_recompute,
        )

        return StandardResponse(
            success=True,
            message=f"成功触发特征计算: {stock_code}",
            data={
                "task_id": task_id,
                "stock_code": stock_code,
                "feature_types": feature_types,
                "force_recompute": force_recompute,
                "triggered_at": datetime.now().isoformat(),
            },
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"触发特征计算失败: {str(e)}")
