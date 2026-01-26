"""
健康检查路由
"""

from fastapi import APIRouter

from app.api.v1.schemas import StandardResponse

router = APIRouter(prefix="/health", tags=["健康检查"])


@router.get(
    "", response_model=StandardResponse, summary="健康检查", description="检查API服务运行状态"
)
async def health_check():
    """
    健康检查端点

    返回API服务的运行状态和版本信息。
    用于监控系统和负载均衡器检查服务可用性。

    Returns:
        StandardResponse: 包含服务状态信息
    """
    return StandardResponse(
        success=True,
        message="API服务运行正常",
        data={"status": "healthy", "version": "1.0.0"},
    )
