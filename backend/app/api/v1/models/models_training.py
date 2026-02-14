"""
模型训练路由
负责模型训练任务创建、执行和特征管理
"""

import asyncio
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from loguru import logger

from app.api.v1.schemas import ModelTrainingRequest, StandardResponse
from app.core.database import SessionLocal
from app.models.task_models import ModelInfo
from app.repositories.task_repository import ModelInfoRepository
from app.services.models.evaluation_report import EvaluationReportGenerator
from app.services.models.hyperparameter_tuning import (
    HyperparameterSpace,
    SearchStrategy,
)
from app.websocket import (
    notify_model_training_completed,
    notify_model_training_failed,
    notify_model_training_progress,
)

from .models_utils import (
    TRAINING_AVAILABLE,
    get_train_executor,
    _format_feature_importance_for_report,
    _normalize_performance_metrics_for_report,
)

router = APIRouter()


def _run_train_model_task_sync(
    model_id: str,
    model_name: str,
    model_type: str,
    stock_codes: list,
    start_date: datetime,
    end_date: datetime,
    hyperparameters: dict,
    enable_hyperparameter_tuning: bool = False,
    hyperparameter_search_strategy: str = "random_search",
    hyperparameter_search_trials: int = 10,
    selected_features: Optional[List[str]] = None,
    main_loop: Optional[asyncio.AbstractEventLoop] = None,
    # 统一训练体系新增参数
    feature_set: str = "alpha158",
    label_type: str = "regression",
    binary_threshold: float = 0.003,
    split_method: str = "ratio",
    train_end_date: Optional[str] = None,
    val_end_date: Optional[str] = None,
    # 滚动训练参数（P2）
    enable_rolling: bool = False,
    rolling_window_type: str = "sliding",
    rolling_step: int = 60,
    rolling_train_window: int = 480,
    rolling_valid_window: int = 60,
    enable_sample_decay: bool = True,
    sample_decay_rate: float = 0.999,
    # CSRankNorm 标签变换
    enable_cs_rank_norm: bool = False,
):
    """
    同步包装函数，用于在线程池中执行异步训练任务
    这样训练任务中的同步阻塞操作不会阻塞主事件循环
    """
    try:
        # 创建新的事件循环来运行异步函数
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(
                train_model_task(
                    model_id=model_id,
                    model_name=model_name,
                    model_type=model_type,
                    stock_codes=stock_codes,
                    start_date=start_date,
                    end_date=end_date,
                    hyperparameters=hyperparameters,
                    enable_hyperparameter_tuning=enable_hyperparameter_tuning,
                    hyperparameter_search_strategy=hyperparameter_search_strategy,
                    hyperparameter_search_trials=hyperparameter_search_trials,
                    selected_features=selected_features,
                    main_loop=main_loop,  # 传递主事件循环
                    feature_set=feature_set,
                    label_type=label_type,
                    binary_threshold=binary_threshold,
                    split_method=split_method,
                    train_end_date=train_end_date,
                    val_end_date=val_end_date,
                    enable_rolling=enable_rolling,
                    rolling_window_type=rolling_window_type,
                    rolling_step=rolling_step,
                    rolling_train_window=rolling_train_window,
                    rolling_valid_window=rolling_valid_window,
                    enable_sample_decay=enable_sample_decay,
                    sample_decay_rate=sample_decay_rate,
                    enable_cs_rank_norm=enable_cs_rank_norm,
                )
            )
        finally:
            loop.close()
    except Exception as e:
        logger.error(f"训练任务执行失败: {e}", exc_info=True)


# 创建线程池执行器（单例）
_train_executor = None


def get_train_executor():
    """获取训练任务线程池执行器"""
    global _train_executor
    if _train_executor is None:
        _train_executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="train_task"
        )
    return _train_executor


async def train_model_task(
    model_id: str,
    model_name: str,
    model_type: str,
    stock_codes: list,
    start_date: datetime,
    end_date: datetime,
    hyperparameters: dict,
    enable_hyperparameter_tuning: bool = False,
    hyperparameter_search_strategy: str = "random_search",
    hyperparameter_search_trials: int = 10,
    selected_features: Optional[List[str]] = None,
    main_loop: Optional[asyncio.AbstractEventLoop] = None,
    # 统一训练体系新增参数
    feature_set: str = "alpha158",
    label_type: str = "regression",
    binary_threshold: float = 0.003,
    split_method: str = "ratio",
    train_end_date: Optional[str] = None,
    val_end_date: Optional[str] = None,
    # 滚动训练参数（P2）
    enable_rolling: bool = False,
    rolling_window_type: str = "sliding",
    rolling_step: int = 60,
    rolling_train_window: int = 480,
    rolling_valid_window: int = 60,
    enable_sample_decay: bool = True,
    sample_decay_rate: float = 0.999,
    # CSRankNorm 标签变换
    enable_cs_rank_norm: bool = False,
):
    """后台训练任务 - 使用统一Qlib训练引擎"""
    session = SessionLocal()
    report_generator = EvaluationReportGenerator()

    try:
        model_repository = ModelInfoRepository(session)
        model_info = model_repository.get_model_info(model_id)

        if not model_info:
            logger.error(f"模型不存在: {model_id}")
            # 如果提供了主事件循环，在主循环中发送通知
            if main_loop:
                asyncio.run_coroutine_threadsafe(
                    notify_model_training_failed(model_id, "模型不存在"), main_loop
                )
            else:
                await notify_model_training_failed(model_id, "模型不存在")
            return

        try:
            # 导入统一Qlib训练引擎
            from app.services.qlib.unified_qlib_training_engine import (
                QlibModelType,
                QlibTrainingConfig,
                UnifiedQlibTrainingEngine,
            )

            # 创建训练引擎
            training_engine = UnifiedQlibTrainingEngine()

            # 定义进度回调函数
            async def progress_callback(
                model_id: str,
                progress: float,
                stage: str,
                message: str,
                metrics: dict = None,
            ):
                # 如果提供了主事件循环，在主循环中发送 WebSocket 通知
                if main_loop:
                    asyncio.run_coroutine_threadsafe(
                        notify_model_training_progress(
                            model_id, progress, stage, message, metrics
                        ),
                        main_loop,
                    )
                else:
                    await notify_model_training_progress(
                        model_id, progress, stage, message, metrics
                    )

                # 更新数据库（在当前事件循环中执行）
                model_info.training_stage = stage
                model_info.training_progress = progress
                session.commit()

            # 发送训练开始通知
            await progress_callback(model_id, 0.0, "initializing", "开始初始化训练")

            # 映射模型类型到Qlib模型类型 - 支持所有模型类型
            model_type_mapping = {
                # 传统机器学习模型
                "lightgbm": QlibModelType.LIGHTGBM,
                "xgboost": QlibModelType.XGBOOST,
                "random_forest": QlibModelType.LIGHTGBM,  # 使用LightGBM替代随机森林
                "linear_regression": QlibModelType.LINEAR,
                # 深度学习模型
                "mlp": QlibModelType.MLP,
                "lstm": QlibModelType.MLP,  # 暂时使用MLP替代，后续可扩展
                "transformer": QlibModelType.TRANSFORMER,
                "informer": QlibModelType.INFORMER,
                "timesnet": QlibModelType.TIMESNET,
                "patchtst": QlibModelType.PATCHTST,
            }

            qlib_model_type = model_type_mapping.get(model_type, QlibModelType.LIGHTGBM)

            # 超参数调优（如果启用）
            # 合并超参数，确保num_iterations被正确传递
            final_hyperparameters = hyperparameters.copy()

            # 如果超参数中有num_iterations或epochs，确保传递到模型配置
            if (
                "num_iterations" not in final_hyperparameters
                and "epochs" in final_hyperparameters
            ):
                final_hyperparameters["num_iterations"] = final_hyperparameters[
                    "epochs"
                ]

            tuning_summary = None
            if enable_hyperparameter_tuning and hyperparameter_search_trials > 0:
                await progress_callback(
                    model_id, 5.0, "hyperparameter_tuning", "开始超参数搜索"
                )

                # 定义超参数搜索空间
                param_space = {
                    "learning_rate": HyperparameterSpace(
                        name="learning_rate",
                        param_type="float",
                        min_value=0.01,
                        max_value=0.3,
                        step=0.01,
                    ),
                    "max_depth": HyperparameterSpace(
                        name="max_depth",
                        param_type="int",
                        min_value=3,
                        max_value=15,
                        step=1,
                    ),
                    "num_leaves": HyperparameterSpace(
                        name="num_leaves",
                        param_type="int",
                        min_value=31,
                        max_value=300,
                        step=10,
                    ),
                }

                # 定义训练函数
                async def train_with_params(params):
                    config = QlibTrainingConfig(
                        model_type=qlib_model_type,
                        hyperparameters={**hyperparameters, **params},
                        validation_split=0.2,
                        use_alpha_factors=True,
                        selected_features=selected_features,
                        feature_set=feature_set,
                        label_type=label_type,
                        binary_threshold=binary_threshold,
                        split_method=split_method,
                        train_end_date=train_end_date,
                        val_end_date=val_end_date,
                        enable_cs_rank_norm=enable_cs_rank_norm,
                    )

                    try:
                        result = await training_engine.train_model(
                            model_id=f"{model_id}_trial",
                            model_name=f"{model_name}_trial",
                            stock_codes=stock_codes,
                            start_date=start_date,
                            end_date=end_date,
                            config=config,
                        )
                        return {
                            "score": result.validation_metrics.get("accuracy", 0.0),
                            "accuracy": result.validation_metrics.get("accuracy", 0.0),
                            "r2": result.validation_metrics.get("r2", 0.0),
                        }
                    except Exception as e:
                        logger.warning(f"超参数试验失败: {e}")
                        return {"score": 0.0, "accuracy": 0.0, "r2": 0.0}

                # 执行超参数搜索（需要处理异步函数）
                best_trial = None
                best_score = float("-inf")

                # 手动执行搜索，因为需要支持异步函数
                import random
                from itertools import product

                strategy = (hyperparameter_search_strategy or "random_search").lower()
                total_trials = max(int(hyperparameter_search_trials), 1)

                def _generate_grid_combinations(space: dict) -> List[dict]:
                    values = {}
                    for name, spec in space.items():
                        if spec.param_type == "int":
                            values[name] = list(
                                range(
                                    int(spec.min_value),
                                    int(spec.max_value) + 1,
                                    int(spec.step or 1),
                                )
                            )
                        elif spec.param_type == "float":
                            step = float(spec.step or 0.01)
                            start = float(spec.min_value)
                            end = float(spec.max_value)
                            vals = []
                            current = start
                            while current <= end + 1e-9:
                                vals.append(round(current, 4))
                                current += step
                            values[name] = vals
                        else:
                            values[name] = list(spec.choices or [])
                    combos = [
                        dict(zip(values.keys(), combo))
                        for combo in product(*values.values())
                    ]
                    return combos

                if strategy == SearchStrategy.GRID_SEARCH.value:
                    combinations = _generate_grid_combinations(param_space)
                    if not combinations:
                        logger.warning("超参数网格为空，改用随机搜索")
                        combinations = None
                    if combinations and len(combinations) > total_trials:
                        combinations = random.sample(combinations, total_trials)
                    trial_params = combinations
                elif strategy == SearchStrategy.BAYESIAN_OPTIMIZATION.value:
                    logger.warning("暂不支持贝叶斯优化，改用随机搜索")
                    trial_params = None
                else:
                    trial_params = None

                if trial_params is None:
                    trial_params = []
                    for _ in range(total_trials):
                        params = {}
                        for param_name, space in param_space.items():
                            if space.param_type == "float":
                                params[param_name] = round(
                                    random.uniform(space.min_value, space.max_value), 4
                                )
                            elif space.param_type == "int":
                                params[param_name] = random.randint(
                                    space.min_value, space.max_value
                                )
                        trial_params.append(params)

                for trial_id, params in enumerate(trial_params):
                    try:
                        logger.info(
                            f"超参数试验 {trial_id + 1}/{len(trial_params)}: {params}"
                        )
                        metrics = await train_with_params(params)
                        score = metrics.get("score", metrics.get("accuracy", 0.0))

                        if score > best_score:
                            best_score = score
                            best_trial = type(
                                "HyperparameterTrial",
                                (),
                                {
                                    "trial_id": trial_id,
                                    "hyperparameters": params,
                                    "score": score,
                                    "metrics": metrics,
                                },
                            )()
                            logger.info(f"发现更好的超参数组合，得分: {score:.4f}")
                    except Exception as e:
                        logger.error(f"超参数试验 {trial_id} 失败: {e}", exc_info=True)

                if best_trial and best_trial.score > 0:
                    final_hyperparameters.update(best_trial.hyperparameters)
                    logger.info(
                        f"超参数调优完成，最佳超参数: {best_trial.hyperparameters}, 得分: {best_trial.score:.4f}"
                    )
                    tuning_summary = {
                        "strategy": strategy,
                        "trials": len(trial_params),
                        "best_score": best_trial.score,
                        "best_hyperparameters": best_trial.hyperparameters,
                    }
                    await progress_callback(
                        model_id,
                        10.0,
                        "hyperparameter_tuning",
                        f"超参数搜索完成，最佳得分: {best_trial.score:.4f}",
                        {
                            "best_score": best_trial.score,
                            "best_params": best_trial.hyperparameters,
                        },
                    )
                else:
                    logger.warning("超参数调优未找到有效结果，使用默认参数")
                    tuning_summary = {
                        "strategy": strategy,
                        "trials": len(trial_params),
                        "best_score": None,
                        "best_hyperparameters": None,
                    }
                    await progress_callback(
                        model_id, 10.0, "hyperparameter_tuning", "超参数搜索完成，使用默认参数"
                    )

            # 创建Qlib训练配置
            # 从超参数中获取num_iterations，Qlib官方基准默认1000
            num_iterations = (
                final_hyperparameters.get("num_iterations")
                or final_hyperparameters.get("n_estimators")
                or final_hyperparameters.get("epochs")
                or 1000
            )
            early_stopping_patience = max(num_iterations, 50)  # Qlib官方基准=50

            config = QlibTrainingConfig(
                model_type=qlib_model_type,
                hyperparameters=final_hyperparameters,
                validation_split=hyperparameters.get("validation_split", 0.2),
                early_stopping_patience=early_stopping_patience,  # 使用实际的迭代次数
                use_alpha_factors=True,
                cache_features=True,
                selected_features=selected_features,  # 传递用户选择的特征
                feature_set=feature_set,
                label_type=label_type,
                binary_threshold=binary_threshold,
                split_method=split_method,
                train_end_date=train_end_date,
                val_end_date=val_end_date,
                # 滚动训练配置（P2）
                enable_rolling=enable_rolling,
                rolling_window_type=rolling_window_type,
                rolling_step=rolling_step,
                rolling_train_window=rolling_train_window,
                rolling_valid_window=rolling_valid_window,
                enable_sample_decay=enable_sample_decay,
                sample_decay_rate=sample_decay_rate,
                enable_cs_rank_norm=enable_cs_rank_norm,
            )

            # 使用统一Qlib训练引擎训练模型
            result = await training_engine.train_model(
                model_id=model_id,
                model_name=model_name,
                stock_codes=stock_codes,
                start_date=start_date,
                end_date=end_date,
                config=config,
                progress_callback=progress_callback,
            )

            # 生成评估报告
            await progress_callback(model_id, 95.0, "generating_report", "生成评估报告")

            # 从训练结果中获取样本数
            train_samples = getattr(result, "train_samples", 0)
            validation_samples = getattr(result, "validation_samples", 0)
            test_samples = getattr(result, "test_samples", 0)
            total_samples = train_samples + validation_samples + test_samples

            # 确保超参数不为空
            if not final_hyperparameters or len(final_hyperparameters) == 0:
                logger.warning(f"模型 {model_id} 的超参数为空，使用默认值")
                final_hyperparameters = (
                    hyperparameters.copy() if hyperparameters else {}
                )

            logger.info(f"生成评估报告 - 模型 {model_id}, 超参数: {final_hyperparameters}")

            report = report_generator.generate_report(
                model_id=model_id,
                model_name=model_name,
                model_type=model_type,
                version=model_info.version,
                training_summary={
                    "duration": result.training_duration,
                    "total_samples": total_samples,
                    "train_samples": train_samples,
                    "validation_samples": validation_samples,
                    "test_samples": test_samples,
                    "epochs": len(result.training_history)
                    if result.training_history
                    else 0,
                    "batch_size": final_hyperparameters.get("batch_size", 32),
                    "learning_rate": final_hyperparameters.get("learning_rate", 0.0),
                },
                performance_metrics=_normalize_performance_metrics_for_report(
                    result.validation_metrics
                ),
                feature_importance=_format_feature_importance_for_report(
                    result.feature_importance
                ),
                training_history=result.training_history,
                hyperparameters=final_hyperparameters,
                training_data_info={
                    "stock_codes": stock_codes,
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                },
                feature_correlation=result.feature_correlation,
                hyperparameter_tuning=tuning_summary,
            )

            # 更新模型信息
            model_info.status = "ready"
            model_info.file_path = result.model_path
            model_info.training_progress = 100.0
            model_info.training_stage = "completed"

            # 规范化准确率
            accuracy = result.validation_metrics.get("accuracy", 0.0)
            if accuracy < 0:
                accuracy = max(0.0, result.validation_metrics.get("r2", 0.0))

            model_info.performance_metrics = {
                "accuracy": float(accuracy),
                "mse": result.validation_metrics.get("mse", 0.0),
                "mae": result.validation_metrics.get("mae", 0.0),
                "r2": result.validation_metrics.get("r2", 0.0),
            }
            model_info.evaluation_report = report_generator.to_dict(report)
            model_info.hyperparameters = final_hyperparameters
            session.commit()

            # 发送完成通知
            if main_loop:
                asyncio.run_coroutine_threadsafe(
                    notify_model_training_completed(
                        model_id, model_info.performance_metrics
                    ),
                    main_loop,
                )
            else:
                await notify_model_training_completed(
                    model_id, model_info.performance_metrics
                )
            logger.info(f"统一Qlib模型训练完成: {model_id}")

        except Exception as e:
            logger.error(f"统一Qlib模型训练失败: {model_id}, 错误: {e}", exc_info=True)
            model_info.status = "failed"
            model_info.training_stage = "failed"
            model_info.performance_metrics = {"error": str(e), "status": "failed"}
            session.commit()
            if main_loop:
                asyncio.run_coroutine_threadsafe(
                    notify_model_training_failed(model_id, str(e)), main_loop
                )
            else:
                await notify_model_training_failed(model_id, str(e))

    except Exception as e:
        logger.error(f"训练任务执行失败: {e}", exc_info=True)
        session.rollback()
        if main_loop:
            asyncio.run_coroutine_threadsafe(
                notify_model_training_failed(model_id, str(e)), main_loop
            )
        else:
            await notify_model_training_failed(model_id, str(e))
    finally:
        session.close()




@router.post("/train", response_model=StandardResponse)
async def create_training_task(request: ModelTrainingRequest):
    """创建模型训练任务"""
    if not TRAINING_AVAILABLE:
        raise HTTPException(status_code=503, detail="模型训练服务不可用，请检查依赖安装")

    session = SessionLocal()
    try:
        # 验证模型类型 - 支持所有Qlib模型类型
        valid_model_types = [
            # 传统机器学习模型
            "lightgbm",
            "xgboost",
            "linear_regression",
            "random_forest",
            # 深度学习模型
            "mlp",
            "lstm",
            "transformer",
            "informer",
            "timesnet",
            "patchtst",
        ]
        if request.model_type not in valid_model_types:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的模型类型: {request.model_type}。支持的类型: {', '.join(valid_model_types)}",
            )

        # 生成模型ID
        model_id = str(uuid.uuid4())

        # 创建模型目录
        models_dir = Path("backend/data/models")
        models_dir.mkdir(parents=True, exist_ok=True)
        model_file_path = models_dir / f"{model_id}.pkl"

        # 解析日期
        try:
            start_date = datetime.fromisoformat(request.start_date)
            end_date = datetime.fromisoformat(request.end_date)
        except ValueError:
            raise HTTPException(status_code=400, detail="日期格式错误，请使用 YYYY-MM-DD 格式")

        # 创建模型记录
        model_info = ModelInfo(
            model_id=model_id,
            model_name=request.model_name,
            model_type=request.model_type,
            version="1.0.0",
            file_path=str(model_file_path),
            training_data_start=start_date,
            training_data_end=end_date,
            hyperparameters=request.hyperparameters or {},
            status="training",
            parent_model_id=request.parent_model_id,
            created_at=datetime.utcnow(),
        )

        session.add(model_info)
        session.commit()

        logger.info(
            f"创建模型训练任务: {model_id}, 模型名称: {request.model_name}, 类型: {request.model_type}"
        )

        # 获取当前事件循环（主事件循环），用于发送 WebSocket 通知
        main_loop = asyncio.get_event_loop()

        # 使用线程池执行器在后台执行训练任务
        # 这样训练任务中的同步阻塞操作不会阻塞主事件循环，前端可以立即得到响应
        executor = get_train_executor()
        executor.submit(
            _run_train_model_task_sync,
            model_id=model_id,
            model_name=request.model_name,
            model_type=request.model_type,
            stock_codes=request.stock_codes,
            start_date=start_date,
            end_date=end_date,
            hyperparameters=request.hyperparameters or {},
            enable_hyperparameter_tuning=request.enable_hyperparameter_tuning,
            hyperparameter_search_strategy=request.hyperparameter_search_strategy,
            hyperparameter_search_trials=request.hyperparameter_search_trials,
            selected_features=request.selected_features,
            main_loop=main_loop,  # 传递主事件循环
            feature_set=request.feature_set,
            label_type=request.label_type,
            binary_threshold=request.binary_threshold,
            split_method=request.split_method,
            train_end_date=request.train_end_date,
            val_end_date=request.val_end_date,
            # 滚动训练参数（P2）
            enable_rolling=request.enable_rolling,
            rolling_window_type=request.rolling_window_type,
            rolling_step=request.rolling_step,
            rolling_train_window=request.rolling_train_window,
            rolling_valid_window=request.rolling_valid_window,
            enable_sample_decay=request.enable_sample_decay,
            sample_decay_rate=request.sample_decay_rate,
            enable_cs_rank_norm=request.enable_cs_rank_norm,
        )

        return StandardResponse(
            success=True,
            message="模型训练任务已创建，正在后台训练中",
            data={
                "model_id": model_id,
                "model_name": request.model_name,
                "model_type": request.model_type,
                "status": "training",
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        session.rollback()
        logger.error(f"创建模型训练任务失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"创建模型训练任务失败: {str(e)}")
    finally:
        session.close()



@router.get("/available-features", response_model=StandardResponse, summary="获取可用特征列表")
async def get_available_features(
    stock_code: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
):
    """
    获取可用于模型训练的特征列表

    如果提供了stock_code和日期范围，将基于实际数据返回可用特征。
    否则返回所有可能支持的特征列表。
    """
    try:
        from datetime import datetime, timedelta

        from app.services.prediction.technical_indicators import (
            TechnicalIndicatorCalculator,
        )
        from app.services.qlib.enhanced_qlib_provider import EnhancedQlibDataProvider

        # 基础价格特征
        base_features = ["open", "high", "low", "close", "volume"]

        # 技术指标特征
        indicator_calculator = TechnicalIndicatorCalculator()
        supported_indicators = indicator_calculator.get_supported_indicators_info()

        # 从技术指标信息中提取特征名称
        indicator_features = []
        indicator_mapping = {
            "MA5": "ma_5",
            "MA10": "ma_10",
            "MA20": "ma_20",
            "MA60": "ma_60",
            "SMA": "sma",
            "EMA": "ema",
            "WMA": "wma",
            "RSI": "rsi",
            "STOCH": "stoch",
            "WILLIAMS_R": "williams_r",
            "CCI": "cci",
            "MOMENTUM": "momentum",
            "ROC": "roc",
            "MACD": "macd",
            "MACD_SIGNAL": "macd_signal",
            "MACD_HISTOGRAM": "macd_histogram",
            "BOLLINGER": [
                "bb_upper",
                "bb_middle",
                "bb_lower",
                "bb_width",
                "bb_position",
            ],
            "SAR": "sar",
            "ADX": "adx",
            "VWAP": "vwap",
            "OBV": "obv",
            "AD_LINE": "ad_line",
            "VOLUME_RSI": "volume_rsi",
            "ATR": "atr",
            "VOLATILITY": "volatility",
            "HISTORICAL_VOLATILITY": "historical_volatility",
            "KDJ": ["kdj_k", "kdj_d", "kdj_j"],
        }

        for indicator_name in supported_indicators.keys():
            if indicator_name in indicator_mapping:
                mapping = indicator_mapping[indicator_name]
                if isinstance(mapping, list):
                    indicator_features.extend(mapping)
                else:
                    indicator_features.append(mapping)

        # 基本面特征
        fundamental_features = [
            "price_change",
            "price_change_5d",
            "price_change_20d",
            "volume_change",
            "volume_ma_ratio",
            "volatility_5d",
            "volatility_20d",
            "price_position",
        ]

        # Alpha因子特征（如果启用）
        alpha_features = []
        # Alpha158因子通常有158个特征，这里列出一些常见的
        alpha_features = [f"alpha_{i:03d}" for i in range(1, 159)]

        # 如果提供了股票代码和日期，尝试获取实际可用的特征
        if stock_code and start_date and end_date:
            try:
                provider = EnhancedQlibDataProvider()
                await provider.initialize_qlib()

                start_dt = datetime.fromisoformat(start_date)
                end_dt = datetime.fromisoformat(end_date)

                # 准备数据集以获取实际可用的特征
                dataset = await provider.prepare_qlib_dataset(
                    stock_codes=[stock_code],
                    start_date=start_dt,
                    end_date=end_dt,
                    include_alpha_factors=True,
                    use_cache=False,
                )

                if not dataset.empty:
                    # 获取实际存在的特征（排除label和元数据列）
                    actual_features = [
                        col
                        for col in dataset.columns
                        if col
                        not in [
                            "label",
                            "stock_code",
                            "date",
                            "instrument",
                            "datetime",
                            "ts_code",
                        ]
                    ]

                    # 分类实际特征
                    actual_base = [f for f in actual_features if f.startswith("$")]
                    actual_indicators = [
                        f
                        for f in actual_features
                        if f not in actual_base
                        and not f.startswith("alpha_")
                        and f not in ["ts_code"]
                    ]
                    actual_fundamental = [
                        f
                        for f in actual_features
                        if f
                        in [
                            "RET1",
                            "RET5",
                            "RET20",
                            "VOLUME_RET1",
                            "VOLUME_MA_RATIO",
                            "VOLATILITY5",
                            "VOLATILITY20",
                            "PRICE_POSITION",
                        ]
                    ]
                    actual_alpha = [
                        f for f in actual_features if f.startswith("alpha_")
                    ]

                    return StandardResponse(
                        success=True,
                        message="成功获取可用特征列表",
                        data={
                            "features": actual_features,  # 返回实际特征名称，用于训练
                            "feature_count": len(actual_features),
                            "feature_categories": {
                                "base_features": actual_base,
                                "indicator_features": actual_indicators,
                                "fundamental_features": actual_fundamental,
                                "alpha_features": actual_alpha,
                            },
                            "source": "actual_data",
                        },
                    )
            except Exception as e:
                logger.warning(f"获取实际特征失败，返回理论特征列表: {e}")

        # 返回理论特征列表（使用实际训练时的特征名称格式）
        # 将理论特征名称转换为实际训练时使用的格式
        actual_base_features = ["$open", "$high", "$low", "$close", "$volume"]
        actual_indicator_features = []
        # 映射技术指标到实际名称
        indicator_to_actual = {
            "ma_5": "MA5",
            "ma_10": "MA10",
            "ma_20": "MA20",
            "ma_60": "MA60",
            "sma": "SMA",
            "ema": "EMA20",
            "rsi": "RSI14",
            "macd": "MACD",
            "macd_signal": "MACD_SIGNAL",
            "macd_histogram": "MACD_HIST",
            "bb_upper": "BOLL_UPPER",
            "bb_middle": "BOLL_MIDDLE",
            "bb_lower": "BOLL_LOWER",
            "atr": "ATR14",
            "vwap": "VWAP",
            "obv": "OBV",
            "stoch": "STOCH_K",
            "kdj_k": "KDJ_K",
            "kdj_d": "KDJ_D",
            "kdj_j": "KDJ_J",
            "williams_r": "WILLIAMS_R",
            "cci": "CCI20",
            "momentum": "MOMENTUM",
            "roc": "ROC",
            "sar": "SAR",
            "adx": "ADX",
            "volume_rsi": "VOLUME_RSI",
        }
        for ind in indicator_features:
            actual_indicator_features.append(indicator_to_actual.get(ind, ind.upper()))

        actual_fundamental_features = [
            "RET1",
            "RET5",
            "RET20",
            "VOLUME_RET1",
            "VOLUME_MA_RATIO",
            "VOLATILITY5",
            "VOLATILITY20",
            "PRICE_POSITION",
        ]

        all_features = (
            actual_base_features
            + actual_indicator_features
            + actual_fundamental_features
            + alpha_features
        )

        return StandardResponse(
            success=True,
            message="成功获取可用特征列表",
            data={
                "features": all_features,
                "feature_count": len(all_features),
                "feature_categories": {
                    "base_features": actual_base_features,
                    "indicator_features": actual_indicator_features,
                    "fundamental_features": actual_fundamental_features,
                    "alpha_features": alpha_features,
                },
                "source": "theoretical",
            },
        )

    except Exception as e:
        logger.error(f"获取可用特征列表失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取可用特征列表失败: {str(e)}")

