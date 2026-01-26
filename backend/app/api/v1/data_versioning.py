"""
数据版本控制API路由
提供数据版本管理和血缘追踪的API接口
"""
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.api.v1.schemas import StandardResponse
from app.services.data_versioning.integration_service import data_versioning_integration
from app.services.data_versioning.lineage_tracker import NodeType, TransformationType
from app.services.data_versioning.version_manager import DataType, VersionStatus

router = APIRouter(prefix="/data-versioning", tags=["数据版本控制"])


# 请求模型
class CreateVersionRequest(BaseModel):
    file_path: str
    version_name: str
    data_type: str
    description: str = ""
    tags: List[str] = []
    copy_file: bool = False


class CreateFeatureVersionRequest(BaseModel):
    source_data_ids: List[str]
    feature_file_path: str
    feature_config: Dict[str, Any]


class CreateSnapshotRequest(BaseModel):
    snapshot_name: str
    version_ids: List[str]
    description: str = ""
    tags: List[str] = []


class TrackModelTrainingRequest(BaseModel):
    training_data_version_ids: List[str]
    feature_version_ids: List[str]
    model_id: str
    model_name: str
    training_config: Dict[str, Any]


@router.get("/versions", response_model=StandardResponse, summary="获取数据版本列表")
async def list_versions(
    data_type: Optional[str] = Query(None, description="数据类型过滤"),
    status: Optional[str] = Query(None, description="状态过滤"),
    tags: Optional[str] = Query(None, description="标签过滤，逗号分隔"),
    created_by: Optional[str] = Query(None, description="创建者过滤"),
    limit: int = Query(100, description="返回数量限制"),
):
    """获取数据版本列表"""
    try:
        # 转换参数
        data_type_enum = None
        if data_type:
            try:
                data_type_enum = DataType(data_type)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"无效的数据类型: {data_type}")

        status_enum = None
        if status:
            try:
                status_enum = VersionStatus(status)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"无效的状态: {status}")

        tags_list = None
        if tags:
            tags_list = [tag.strip() for tag in tags.split(",")]

        # 获取版本列表
        versions = data_versioning_integration.version_manager.list_versions(
            data_type=data_type_enum,
            status=status_enum,
            tags=tags_list,
            created_by=created_by,
        )

        # 限制返回数量
        versions = versions[:limit]

        return StandardResponse(
            success=True,
            message=f"成功获取版本列表: {len(versions)} 个版本",
            data={
                "versions": [version.to_dict() for version in versions],
                "total_count": len(versions),
                "filters": {
                    "data_type": data_type,
                    "status": status,
                    "tags": tags,
                    "created_by": created_by,
                    "limit": limit,
                },
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取版本列表失败: {str(e)}")


@router.get("/versions/{version_id}", response_model=StandardResponse, summary="获取版本详情")
async def get_version(version_id: str):
    """获取版本详情"""
    try:
        version = data_versioning_integration.version_manager.get_version(version_id)

        if not version:
            raise HTTPException(status_code=404, detail=f"版本不存在: {version_id}")

        # 获取血缘信息
        lineages = data_versioning_integration.version_manager.get_lineage_chain(
            version_id
        )

        return StandardResponse(
            success=True,
            message="成功获取版本详情",
            data={
                "version": version.to_dict(),
                "lineages": [lineage.to_dict() for lineage in lineages],
                "lineage_count": len(lineages),
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取版本详情失败: {str(e)}")


@router.post("/versions", response_model=StandardResponse, summary="创建数据版本")
async def create_version(request: CreateVersionRequest):
    """创建数据版本"""
    try:
        # 转换数据类型
        try:
            data_type_enum = DataType(request.data_type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"无效的数据类型: {request.data_type}")

        version_id = data_versioning_integration.version_manager.create_version(
            file_path=request.file_path,
            version_name=request.version_name,
            data_type=data_type_enum,
            created_by="api_user",
            description=request.description,
            tags=request.tags,
            copy_file=request.copy_file,
        )

        version = data_versioning_integration.version_manager.get_version(version_id)

        return StandardResponse(
            success=True,
            message=f"成功创建版本: {version_id}",
            data={
                "version_id": version_id,
                "version": version.to_dict() if version else None,
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"创建版本失败: {str(e)}")


@router.post("/versions/feature", response_model=StandardResponse, summary="创建特征版本")
async def create_feature_version(request: CreateFeatureVersionRequest):
    """创建特征版本"""
    try:
        version_id = data_versioning_integration.create_feature_version(
            source_data_ids=request.source_data_ids,
            feature_file_path=request.feature_file_path,
            feature_config=request.feature_config,
            created_by="api_user",
        )

        version = data_versioning_integration.version_manager.get_version(version_id)

        return StandardResponse(
            success=True,
            message=f"成功创建特征版本: {version_id}",
            data={
                "version_id": version_id,
                "version": version.to_dict() if version else None,
            },
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"创建特征版本失败: {str(e)}")


@router.get(
    "/versions/compare/{version_id1}/{version_id2}",
    response_model=StandardResponse,
    summary="比较版本",
)
async def compare_versions(version_id1: str, version_id2: str):
    """比较两个版本"""
    try:
        comparison = data_versioning_integration.get_version_comparison(
            version_id1, version_id2
        )

        return StandardResponse(success=True, message="成功比较版本", data=comparison)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"比较版本失败: {str(e)}")


@router.get(
    "/stocks/{stock_code}/versions", response_model=StandardResponse, summary="获取股票数据版本"
)
async def get_stock_versions(stock_code: str):
    """获取股票的所有数据版本"""
    try:
        versions = data_versioning_integration.get_stock_data_versions(stock_code)

        return StandardResponse(
            success=True,
            message=f"成功获取股票 {stock_code} 的版本信息",
            data={
                "stock_code": stock_code,
                "versions": versions,
                "total_versions": len(versions),
            },
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取股票版本失败: {str(e)}")


@router.get(
    "/stocks/{stock_code}/lineage", response_model=StandardResponse, summary="获取股票特征血缘"
)
async def get_stock_feature_lineage(stock_code: str):
    """获取股票的特征血缘"""
    try:
        lineage_info = data_versioning_integration.get_feature_lineage_for_stock(
            stock_code
        )

        return StandardResponse(
            success=True, message=f"成功获取股票 {stock_code} 的特征血缘", data=lineage_info
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取股票特征血缘失败: {str(e)}")


@router.get("/lineage/nodes", response_model=StandardResponse, summary="搜索血缘节点")
async def search_lineage_nodes(
    node_type: Optional[str] = Query(None, description="节点类型"),
    name_pattern: Optional[str] = Query(None, description="名称模式"),
    tags: Optional[str] = Query(None, description="标签，逗号分隔"),
    created_by: Optional[str] = Query(None, description="创建者"),
    limit: int = Query(100, description="返回数量限制"),
):
    """搜索血缘节点"""
    try:
        # 转换参数
        node_type_enum = None
        if node_type:
            try:
                node_type_enum = NodeType(node_type)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"无效的节点类型: {node_type}")

        tags_list = None
        if tags:
            tags_list = [tag.strip() for tag in tags.split(",")]

        # 搜索节点
        nodes = data_versioning_integration.lineage_tracker.search_nodes(
            node_type=node_type_enum,
            name_pattern=name_pattern,
            tags=tags_list,
            created_by=created_by,
        )

        # 限制返回数量
        nodes = nodes[:limit]

        return StandardResponse(
            success=True,
            message=f"成功搜索血缘节点: {len(nodes)} 个节点",
            data={
                "nodes": [node.to_dict() for node in nodes],
                "total_count": len(nodes),
                "filters": {
                    "node_type": node_type,
                    "name_pattern": name_pattern,
                    "tags": tags,
                    "created_by": created_by,
                    "limit": limit,
                },
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"搜索血缘节点失败: {str(e)}")


@router.get(
    "/lineage/features/{feature_id}", response_model=StandardResponse, summary="获取特征血缘"
)
async def get_feature_lineage(feature_id: str):
    """获取特征血缘"""
    try:
        lineage = data_versioning_integration.lineage_tracker.get_feature_lineage(
            feature_id
        )

        if not lineage:
            raise HTTPException(status_code=404, detail=f"特征不存在: {feature_id}")

        return StandardResponse(success=True, message="成功获取特征血缘", data=lineage)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取特征血缘失败: {str(e)}")


@router.get(
    "/lineage/models/{model_id}", response_model=StandardResponse, summary="获取模型血缘"
)
async def get_model_lineage(model_id: str):
    """获取模型血缘"""
    try:
        lineage = data_versioning_integration.get_model_training_lineage(model_id)

        return StandardResponse(success=True, message="成功获取模型血缘", data=lineage)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取模型血缘失败: {str(e)}")


@router.post(
    "/lineage/model-training", response_model=StandardResponse, summary="追踪模型训练"
)
async def track_model_training(request: TrackModelTrainingRequest):
    """追踪模型训练血缘"""
    try:
        lineage_ids = data_versioning_integration.track_model_training_with_versions(
            training_data_version_ids=request.training_data_version_ids,
            feature_version_ids=request.feature_version_ids,
            model_id=request.model_id,
            model_name=request.model_name,
            training_config=request.training_config,
            created_by="api_user",
        )

        return StandardResponse(
            success=True,
            message=f"成功追踪模型训练: {len(lineage_ids)} 个血缘关系",
            data={
                "model_id": request.model_id,
                "lineage_ids": lineage_ids,
                "lineage_count": len(lineage_ids),
            },
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"追踪模型训练失败: {str(e)}")


@router.post("/snapshots", response_model=StandardResponse, summary="创建数据快照")
async def create_snapshot(request: CreateSnapshotRequest):
    """创建数据快照"""
    try:
        snapshot_id = data_versioning_integration.version_manager.create_snapshot(
            snapshot_name=request.snapshot_name,
            version_ids=request.version_ids,
            created_by="api_user",
            description=request.description,
            tags=request.tags,
        )

        snapshot = data_versioning_integration.version_manager.get_snapshot(snapshot_id)

        return StandardResponse(
            success=True,
            message=f"成功创建数据快照: {snapshot_id}",
            data={
                "snapshot_id": snapshot_id,
                "snapshot": snapshot.to_dict() if snapshot else None,
            },
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"创建数据快照失败: {str(e)}")


@router.get("/snapshots", response_model=StandardResponse, summary="获取快照列表")
async def list_snapshots(
    created_by: Optional[str] = Query(None, description="创建者过滤"),
    limit: int = Query(100, description="返回数量限制"),
):
    """获取快照列表"""
    try:
        snapshots = data_versioning_integration.version_manager.list_snapshots(
            created_by=created_by
        )

        # 限制返回数量
        snapshots = snapshots[:limit]

        return StandardResponse(
            success=True,
            message=f"成功获取快照列表: {len(snapshots)} 个快照",
            data={
                "snapshots": [snapshot.to_dict() for snapshot in snapshots],
                "total_count": len(snapshots),
                "filters": {"created_by": created_by, "limit": limit},
            },
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取快照列表失败: {str(e)}")


@router.get("/stats", response_model=StandardResponse, summary="获取版本控制统计")
async def get_versioning_stats():
    """获取数据版本控制统计信息"""
    try:
        stats = data_versioning_integration.get_data_versioning_stats()

        return StandardResponse(success=True, message="成功获取版本控制统计信息", data=stats)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取统计信息失败: {str(e)}")


@router.get("/lineage/summary", response_model=StandardResponse, summary="获取血缘摘要")
async def get_lineage_summary():
    """获取血缘摘要"""
    try:
        summary = data_versioning_integration.lineage_tracker.get_lineage_summary()

        return StandardResponse(success=True, message="成功获取血缘摘要", data=summary)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取血缘摘要失败: {str(e)}")


@router.delete(
    "/versions/{version_id}", response_model=StandardResponse, summary="删除版本"
)
async def delete_version(
    version_id: str, remove_file: bool = Query(False, description="是否删除文件")
):
    """删除版本"""
    try:
        success = data_versioning_integration.version_manager.delete_version(
            version_id, remove_file=remove_file
        )

        if not success:
            raise HTTPException(status_code=404, detail=f"版本不存在: {version_id}")

        return StandardResponse(
            success=True,
            message=f"成功删除版本: {version_id}",
            data={"version_id": version_id, "file_removed": remove_file},
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除版本失败: {str(e)}")


@router.post("/cleanup", response_model=StandardResponse, summary="清理旧版本")
async def cleanup_old_versions(
    days_to_keep: int = Query(30, description="保留天数"),
    keep_tagged_versions: bool = Query(True, description="是否保留标记版本"),
):
    """清理旧版本"""
    try:
        result = data_versioning_integration.cleanup_old_versions(
            days_to_keep=days_to_keep, keep_tagged_versions=keep_tagged_versions
        )

        return StandardResponse(
            success=True,
            message=f"清理完成: 删除了 {result['deleted_versions']} 个版本",
            data=result,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"清理版本失败: {str(e)}")
