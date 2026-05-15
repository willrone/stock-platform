"""Training result assembly helpers."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


class QlibTrainingResultAssembler:
    """负责组装训练结果与历史回填。"""

    def __init__(self, result_cls: type):
        self.result_cls = result_cls

    @staticmethod
    def normalize_training_output(
        training_output: Any,
    ) -> Tuple[Any, List[Dict[str, Any]], Dict[str, Any]]:
        """统一训练输出结构，兼容旧返回格式。"""
        if len(training_output) == 3:
            model, training_history, early_stopping_info = training_output
            return model, training_history, early_stopping_info

        model, training_history = training_output
        return (
            model,
            training_history,
            {
                "early_stopped": False,
                "stopped_epoch": 0,
                "best_epoch": 0,
                "early_stopping_reason": None,
            },
        )

    @staticmethod
    def fill_accuracy_into_history(
        training_history: List[Dict[str, Any]],
        train_accuracy: float,
        val_accuracy: float,
    ) -> None:
        """将评估阶段产出的准确率回填到历史记录。"""
        if not training_history:
            return

        rounded_train_accuracy = round(train_accuracy, 4)
        rounded_val_accuracy = round(val_accuracy, 4)
        for history_item in training_history:
            if history_item.get("train_accuracy", 0.0) == 0.0:
                history_item["train_accuracy"] = rounded_train_accuracy
            if history_item.get("val_accuracy", 0.0) == 0.0:
                history_item["val_accuracy"] = rounded_val_accuracy

    def assemble(
        self,
        *,
        model_path: str,
        model_config: Dict[str, Any],
        training_metrics: Dict[str, Any],
        validation_metrics: Dict[str, Any],
        feature_importance: Dict[str, float],
        training_history: List[Dict[str, Any]],
        training_duration: float,
        train_samples: int,
        validation_samples: int,
        test_samples: int = 0,
        feature_correlation: Dict[str, Any],
        early_stopping_info: Dict[str, Any],
        signal_quality: Dict[str, Any],
        segment_evaluation: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """构建 QlibTrainingResult。"""
        return self.result_cls(
            model_path=model_path,
            model_config=model_config,
            training_metrics=training_metrics,
            validation_metrics=validation_metrics,
            feature_importance=feature_importance,
            training_history=training_history,
            training_duration=training_duration,
            train_samples=train_samples,
            validation_samples=validation_samples,
            test_samples=test_samples,
            early_stopped=early_stopping_info["early_stopped"],
            stopped_epoch=early_stopping_info["stopped_epoch"],
            best_epoch=early_stopping_info["best_epoch"],
            early_stopping_reason=early_stopping_info["early_stopping_reason"],
            feature_correlation=feature_correlation,
            signal_quality=signal_quality,
            segment_evaluation=segment_evaluation,
        )
