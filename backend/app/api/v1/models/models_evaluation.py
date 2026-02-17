"""
模型评估路由
负责模型评估报告和性能历史查询
"""

from datetime import datetime

from fastapi import APIRouter, HTTPException
from loguru import logger

from app.api.v1.schemas import StandardResponse
from app.core.database import SessionLocal
from app.models.task_models import ModelInfo

router = APIRouter()


@router.get("/{model_id}/evaluation-report", response_model=StandardResponse)
async def get_model_evaluation_report(model_id: str):
    """获取模型评估报告"""
    session = SessionLocal()
    try:
        model = session.query(ModelInfo).filter(ModelInfo.model_id == model_id).first()
        if not model:
            logger.warning(f"模型不存在: {model_id}")
            raise HTTPException(status_code=404, detail=f"模型不存在: {model_id}")

        logger.info(
            f"查询模型 {model_id} 的评估报告，状态: {model.status}, evaluation_report类型: {type(model.evaluation_report)}"
        )

        # 检查评估报告是否存在
        if model.evaluation_report is None:
            logger.warning(f"模型 {model_id} 的评估报告为 None，状态: {model.status}")
            raise HTTPException(status_code=404, detail="该模型尚未生成评估报告，请等待训练完成")

        # 检查评估报告是否为空字典
        if (
            isinstance(model.evaluation_report, dict)
            and len(model.evaluation_report) == 0
        ):
            logger.warning(f"模型 {model_id} 的评估报告为空字典")
            raise HTTPException(status_code=404, detail="该模型尚未生成评估报告，请等待训练完成")

        # 检查评估报告是否为空字符串
        if (
            isinstance(model.evaluation_report, str)
            and len(model.evaluation_report.strip()) == 0
        ):
            logger.warning(f"模型 {model_id} 的评估报告为空字符串")
            raise HTTPException(status_code=404, detail="该模型尚未生成评估报告，请等待训练完成")

        # 如果评估报告是字符串，尝试解析为JSON
        if isinstance(model.evaluation_report, str):
            try:
                import json

                evaluation_report = json.loads(model.evaluation_report)
                logger.info(f"成功解析模型 {model_id} 的评估报告（从字符串）")
                return StandardResponse(
                    success=True, message="评估报告获取成功", data=evaluation_report
                )
            except json.JSONDecodeError as e:
                logger.error(f"解析评估报告JSON失败: {e}")
                raise HTTPException(status_code=500, detail="评估报告格式错误")

        logger.info(f"成功获取模型 {model_id} 的评估报告")
        return StandardResponse(
            success=True, message="评估报告获取成功", data=model.evaluation_report
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取评估报告失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取评估报告失败: {str(e)}")
    finally:
        session.close()


@router.get("/{model_id}/performance-history", response_model=StandardResponse)
async def get_model_performance_history(model_id: str, time_range: str = "30d"):
    """获取模型性能历史"""
    try:
        # 解析时间范围
        from datetime import timedelta

        time_ranges = {
            "7d": timedelta(days=7),
            "30d": timedelta(days=30),
            "90d": timedelta(days=90),
            "1y": timedelta(days=365),
        }

        if time_range not in time_ranges:
            raise HTTPException(status_code=400, detail=f"不支持的时间范围: {time_range}")

        end_time = datetime.now()
        start_time = end_time - time_ranges[time_range]

        # 获取性能历史（这里需要从监控系统获取）
        try:
            from app.services.monitoring.performance_monitor import performance_monitor

            performance_history = performance_monitor.get_model_performance_history(
                model_id=model_id, start_time=start_time, end_time=end_time
            )
        except ImportError:
            # 如果监控服务不可用，返回空历史
            performance_history = []

        return StandardResponse(
            success=True,
            message=f"模型性能历史获取成功: {time_range}",
            data={
                "model_id": model_id,
                "time_range": time_range,
                "performance_history": performance_history,
                "summary": {
                    "total_records": len(performance_history),
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                },
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取模型性能历史失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取模型性能历史失败: {str(e)}")
