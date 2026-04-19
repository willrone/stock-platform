"""Unified Qlib training orchestration layer."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, Dict, Optional, Tuple

from loguru import logger

from .pipeline import TrainingRequest


class QlibTrainingOrchestrator:
    """负责编排训练步骤、阶段监控与进度通知。"""

    def __init__(
        self,
        *,
        engine: Any,
        pipeline: Any,
        result_assembler: Any,
        qlib_available_getter: Callable[[], bool],
    ):
        self.engine = engine
        self.pipeline = pipeline
        self.result_assembler = result_assembler
        self.qlib_available_getter = qlib_available_getter

    async def execute(self, request: TrainingRequest) -> Any:
        """执行完整训练流程。"""
        logger.info(
            f"开始Qlib统一训练流程: {request.model_id}, 模型类型: {request.config.model_type.value}"
        )
        start_time = datetime.now()
        self.engine.performance_monitor.start_stage("total_training")

        try:
            self.pipeline.ensure_qlib_available(self.qlib_available_getter())
            await self._run_initialize_stage(request)
            dataset = await self._run_dataset_stage(request)
            model_config, feature_correlation = await self._run_model_config_stage(
                request,
                dataset,
            )
            train_dataset, val_dataset = await self._run_preprocessing_stage(
                request,
                dataset,
            )
            model, training_history, early_stopping_info = await self._run_training_stage(
                request,
                model_config,
                train_dataset,
                val_dataset,
            )
            training_metrics, validation_metrics, signal_quality = await self._run_evaluation_stage(
                request,
                model,
                train_dataset,
                val_dataset,
                training_history,
            )
            feature_importance = await self._run_feature_importance_stage(request, model)
            model_path = await self._run_save_stage(request, model, model_config)
            training_duration = await self._run_completion_stage(
                request,
                start_time,
                model_path,
                validation_metrics,
                early_stopping_info,
            )

            self.engine.performance_monitor.end_stage("total_training")
            self.engine.performance_monitor.print_summary()
            result = self.result_assembler.assemble(
                model_path=model_path,
                model_config=model_config,
                training_metrics=training_metrics,
                validation_metrics=validation_metrics,
                feature_importance=feature_importance,
                training_history=training_history,
                training_duration=training_duration,
                train_samples=len(train_dataset),
                validation_samples=len(val_dataset),
                feature_correlation=feature_correlation,
                early_stopping_info=early_stopping_info,
                signal_quality=signal_quality,
            )
            self._log_training_success(
                model_id=request.model_id,
                training_duration=training_duration,
                validation_metrics=validation_metrics,
                early_stopping_info=early_stopping_info,
            )
            return result
        except Exception as exc:
            await self._handle_failure(request, exc)
            raise

    async def _run_initialize_stage(self, request: TrainingRequest) -> None:
        await self._notify_progress(request, 5.0, "initializing", "初始化Qlib环境")
        self.engine.performance_monitor.start_stage("initialize_qlib")
        await self.pipeline.initialize_engine()
        self.engine.performance_monitor.end_stage("initialize_qlib")

    async def _run_dataset_stage(self, request: TrainingRequest) -> Any:
        details = {
            "stock_count": len(request.stock_codes),
            "date_range": f"{request.start_date.strftime('%Y-%m-%d')} 至 {request.end_date.strftime('%Y-%m-%d')}",
        }
        await self._notify_progress(request, 15.0, "preparing", "准备Qlib数据集", details)
        self.engine.performance_monitor.start_stage("prepare_dataset")
        dataset = await self.pipeline.prepare_dataset(request)
        self.engine.performance_monitor.end_stage("prepare_dataset")
        if dataset.empty:
            raise ValueError("无法获取训练数据")
        self.pipeline.log_dataset_overview(dataset)
        return dataset

    async def _run_model_config_stage(
        self,
        request: TrainingRequest,
        dataset: Any,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        details = {
            "dataset_shape": list(dataset.shape),
            "features_count": dataset.shape[1] if len(dataset.shape) > 1 else 0,
            "sample_count": dataset.shape[0],
        }
        await self._notify_progress(request, 25.0, "configuring", "配置Qlib模型", details)
        self.engine.performance_monitor.start_stage("create_model_config")
        model_config = await self.pipeline.create_model_config(request.config)
        self.engine.performance_monitor.end_stage("create_model_config")
        self.engine.performance_monitor.start_stage("analyze_features")
        feature_correlation = self.pipeline.analyze_feature_correlations(dataset)
        self.engine.performance_monitor.end_stage("analyze_features")
        return model_config, feature_correlation

    async def _run_preprocessing_stage(
        self,
        request: TrainingRequest,
        dataset: Any,
    ) -> Tuple[Any, Any]:
        await self._notify_progress(
            request,
            35.0,
            "preprocessing",
            "数据预处理",
            {"validation_split": request.config.validation_split},
        )
        self.engine.performance_monitor.start_stage("prepare_training_datasets")
        train_dataset, val_dataset = await self.pipeline.prepare_training_datasets(
            dataset,
            request.config.validation_split,
            request.config,
        )
        self.engine.performance_monitor.end_stage("prepare_training_datasets")
        logger.info(
            f"数据集分割完成: 训练集样本数={len(train_dataset)}, 验证集样本数={len(val_dataset)}"
        )
        if hasattr(train_dataset, "data"):
            logger.info(
                f"训练集数据形状: {train_dataset.data.shape}, 特征数={len(train_dataset.data.columns)}"
            )
        if hasattr(val_dataset, "data"):
            logger.info(
                f"验证集数据形状: {val_dataset.data.shape}, 特征数={len(val_dataset.data.columns)}"
            )
        return train_dataset, val_dataset

    async def _run_training_stage(
        self,
        request: TrainingRequest,
        model_config: Dict[str, Any],
        train_dataset: Any,
        val_dataset: Any,
    ) -> Tuple[Any, Any, Dict[str, Any]]:
        details = {
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "model_type": request.config.model_type.value,
            "early_stopping_enabled": request.config.enable_early_stopping,
        }
        await self._notify_progress(request, 45.0, "training", "开始Qlib模型训练", details)
        self.engine.performance_monitor.start_stage("train_model")
        training_output = await self.pipeline.train(
            model_config,
            train_dataset,
            val_dataset,
            request,
        )
        self.engine.performance_monitor.end_stage("train_model")
        return self.result_assembler.normalize_training_output(training_output)

    async def _run_evaluation_stage(
        self,
        request: TrainingRequest,
        model: Any,
        train_dataset: Any,
        val_dataset: Any,
        training_history: Any,
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, Any]]:
        await self._notify_progress(request, 85.0, "evaluating", "评估模型性能")
        self.engine.performance_monitor.start_stage("evaluate_model")
        training_metrics, validation_metrics, signal_quality = await self.pipeline.evaluate(
            model,
            train_dataset,
            val_dataset,
            request.model_id,
        )
        self.engine.performance_monitor.end_stage("evaluate_model")
        self.result_assembler.fill_accuracy_into_history(
            training_history,
            training_metrics.get("accuracy", 0.0),
            validation_metrics.get("accuracy", 0.0),
        )
        await self._notify_progress(
            request,
            90.0,
            "evaluating",
            "模型评估完成",
            {
                "validation_metrics": validation_metrics,
                "training_metrics": training_metrics,
                "signal_quality": signal_quality,
            },
        )
        return training_metrics, validation_metrics, signal_quality

    async def _run_feature_importance_stage(self, request: TrainingRequest, model: Any) -> Any:
        self.engine.performance_monitor.start_stage("extract_feature_importance")
        feature_importance = await self.pipeline.extract_feature_importance(
            model,
            request.config.model_type,
        )
        self.engine.performance_monitor.end_stage("extract_feature_importance")
        return feature_importance

    async def _run_save_stage(
        self,
        request: TrainingRequest,
        model: Any,
        model_config: Dict[str, Any],
    ) -> str:
        await self._notify_progress(request, 95.0, "saving", "保存模型")
        self.engine.performance_monitor.start_stage("save_model")
        model_path = await self.pipeline.save_model(model, request.model_id, model_config)
        self.engine.performance_monitor.end_stage("save_model")
        return model_path

    async def _run_completion_stage(
        self,
        request: TrainingRequest,
        start_time: datetime,
        model_path: str,
        validation_metrics: Dict[str, float],
        early_stopping_info: Dict[str, Any],
    ) -> float:
        training_duration = (datetime.now() - start_time).total_seconds()
        details = {
            "training_duration": training_duration,
            "final_accuracy": validation_metrics.get("accuracy", 0.0),
            "model_path": model_path,
            "early_stopped": early_stopping_info["early_stopped"],
            "early_stopping_reason": early_stopping_info["early_stopping_reason"],
        }
        await self._notify_progress(request, 100.0, "completed", "训练完成", details)
        return training_duration

    async def _notify_progress(
        self,
        request: TrainingRequest,
        progress: float,
        status: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        if request.progress_callback is None:
            return

        if details is None:
            await request.progress_callback(request.model_id, progress, status, message)
            return

        await request.progress_callback(
            request.model_id,
            progress,
            status,
            message,
            details,
        )

    async def _handle_failure(self, request: TrainingRequest, exc: Exception) -> None:
        logger.error(f"Qlib模型训练失败: {request.model_id}, 错误: {exc}", exc_info=True)
        if request.progress_callback is not None:
            await request.progress_callback(request.model_id, 0.0, "failed", f"训练失败: {str(exc)}")
        self.engine.performance_monitor.end_stage("total_training")
        self.engine.performance_monitor.print_summary()

    @staticmethod
    def _log_training_success(
        *,
        model_id: str,
        training_duration: float,
        validation_metrics: Dict[str, float],
        early_stopping_info: Dict[str, Any],
    ) -> None:
        logger.info(f"Qlib模型训练完成: {model_id}, 耗时: {training_duration:.2f}秒")
        if early_stopping_info["early_stopped"]:
            logger.info(
                "训练提前停止: "
                f"{early_stopping_info['early_stopping_reason']}, "
                f"停止轮次: {early_stopping_info['stopped_epoch']}"
            )
        logger.info(f"验证指标: {validation_metrics}")
