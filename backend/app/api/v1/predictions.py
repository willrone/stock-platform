"""
预测服务路由
"""

from datetime import datetime

from fastapi import APIRouter, HTTPException
from loguru import logger

from app.api.v1.schemas import PredictionRequest, StandardResponse
from app.core.config import settings
from app.services.prediction.prediction_engine import PredictionConfig, PredictionEngine

router = APIRouter(prefix="/predictions", tags=["预测服务"])


@router.post("", response_model=StandardResponse)
async def create_prediction(request: PredictionRequest):
    """创建预测任务"""
    try:
        prediction_engine = PredictionEngine(
            model_dir=str(settings.MODEL_STORAGE_PATH),
            data_dir=str(settings.DATA_ROOT_PATH),
        )
        prediction_config = PredictionConfig(
            model_id=request.model_id,
            horizon=request.horizon,
            confidence_level=request.confidence_level,
            risk_assessment=True,
        )

        results = prediction_engine.predict_multiple_stocks(
            request.stock_codes, prediction_config
        )

        from app.services.data.stock_data_loader import StockDataLoader

        loader = StockDataLoader(data_root=str(settings.DATA_ROOT_PATH))

        predictions = []
        for result in results:
            current_price = None
            try:
                historical = loader.load_stock_data(
                    result.stock_code, end_date=datetime.utcnow()
                )
                if not historical.empty and "close" in historical.columns:
                    current_price = float(historical["close"].iloc[-1])
            except Exception:
                current_price = None

            predicted_return = 0.0
            if current_price:
                predicted_return = (
                    result.predicted_price - current_price
                ) / current_price

            predictions.append(
                {
                    "stock_code": result.stock_code,
                    "predicted_direction": result.predicted_direction,
                    "predicted_return": predicted_return,
                    "predicted_price": result.predicted_price,
                    "confidence_score": result.confidence_score,
                    "confidence_interval": {
                        "lower": result.confidence_interval[0],
                        "upper": result.confidence_interval[1],
                    },
                    "risk_assessment": result.risk_metrics.to_dict(),
                }
            )

        return StandardResponse(
            success=True,
            message=f"成功预测 {len(request.stock_codes)} 只股票",
            data={
                "predictions": predictions,
                "model_id": request.model_id,
                "horizon": request.horizon,
            },
        )

    except Exception as e:
        logger.error(f"创建预测失败: {e}")
        raise HTTPException(status_code=500, detail=f"创建预测失败: {str(e)}")


@router.get("/{prediction_id}", response_model=StandardResponse)
async def get_prediction_result(prediction_id: str):
    """获取预测结果"""
    try:
        from app.core.database import SessionLocal
        from app.repositories.task_repository import PredictionResultRepository

        session = SessionLocal()
        try:
            prediction_repo = PredictionResultRepository(session)
            results = prediction_repo.get_prediction_results_by_task(prediction_id)
        finally:
            session.close()

        if not results:
            raise HTTPException(status_code=404, detail="预测结果不存在")

        response_data = {
            "prediction_id": prediction_id,
            "status": "completed",
            "created_at": datetime.now().isoformat(),
            "results": [
                {
                    "stock_code": result.stock_code,
                    "predicted_direction": result.predicted_direction,
                    "confidence_score": result.confidence_score,
                    "predicted_price": result.predicted_price,
                    "confidence_interval": {
                        "lower": result.confidence_interval_lower,
                        "upper": result.confidence_interval_upper,
                    },
                    "risk_assessment": result.risk_metrics or {},
                }
                for result in results
            ],
        }

        return StandardResponse(success=True, message="预测结果获取成功", data=response_data)

    except Exception as e:
        logger.error(f"获取预测结果失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取预测结果失败: {str(e)}")
