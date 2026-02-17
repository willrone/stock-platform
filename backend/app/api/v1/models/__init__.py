"""
模型管理路由 - 主入口
重构后的模块化结构，保持向后兼容
"""

from fastapi import APIRouter

from .models_evaluation import (
    get_model_evaluation_report,
    get_model_performance_history,
)
from .models_lifecycle import (
    get_model_dependencies,
    get_model_lifecycle,
    get_model_lineage,
    transition_model_lifecycle,
)
from .models_management import add_model_tags, delete_model, remove_model_tags

# 导入所有路由函数
from .models_query import (
    get_model_detail,
    get_model_versions,
    list_models,
    search_models,
)
from .models_training import create_training_task, get_available_features

# 创建主路由器
router = APIRouter(prefix="/models", tags=["模型管理"])

# 注册查询路由
router.add_api_route("", list_models, methods=["GET"])
router.add_api_route("/{model_id}", get_model_detail, methods=["GET"])
router.add_api_route("/{model_id}/versions", get_model_versions, methods=["GET"])
router.add_api_route("/search", search_models, methods=["GET"])

# 注册训练路由
router.add_api_route("/train", create_training_task, methods=["POST"])
router.add_api_route("/available-features", get_available_features, methods=["GET"])

# 注册评估路由
router.add_api_route(
    "/{model_id}/evaluation-report", get_model_evaluation_report, methods=["GET"]
)
router.add_api_route(
    "/{model_id}/performance-history", get_model_performance_history, methods=["GET"]
)

# 注册生命周期路由
router.add_api_route("/{model_id}/lifecycle", get_model_lifecycle, methods=["GET"])
router.add_api_route("/{model_id}/lineage", get_model_lineage, methods=["GET"])
router.add_api_route(
    "/{model_id}/dependencies", get_model_dependencies, methods=["GET"]
)
router.add_api_route(
    "/{model_id}/lifecycle/transition", transition_model_lifecycle, methods=["POST"]
)

# 注册管理路由
router.add_api_route("/{model_id}", delete_model, methods=["DELETE"])
router.add_api_route("/{model_id}/tags", add_model_tags, methods=["POST"])
router.add_api_route("/{model_id}/tags", remove_model_tags, methods=["DELETE"])

__all__ = ["router"]
