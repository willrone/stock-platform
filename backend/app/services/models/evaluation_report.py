"""
模型评估报告生成服务

生成详细的训练和评估报告，包括：
- 训练过程分析
- 性能指标可视化数据
- 特征重要性分析
- 预测结果分析
- 模型对比
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger

DEFAULT_EARLY_STOPPING_INFO = {
    "early_stopped": False,
    "stopped_epoch": 0,
    "best_epoch": 0,
    "early_stopping_reason": None,
}

DEFAULT_SIGNAL_QUALITY = {
    "ic": None,
    "icir": None,
    "rank_ic": None,
    "rank_icir": None,
    "long_short_ann_return": None,
    "long_short_ann_sharpe": None,
    "long_avg_ann_return": None,
    "long_avg_ann_sharpe": None,
    "sample_count": 0,
    "analysis_scope": None,
}

SEGMENT_NAMES = ("train", "validation", "test")


def _default_portfolio_bridge_summary(model_id: Optional[str] = None) -> Dict[str, Any]:
    return {
        "model_id": None,
        "task_count": 0,
        "tasks": [],
        "best_by_total_return": None,
        "best_by_sharpe": None,
        "smallest_drawdown": None,
    }


def _normalize_signal_quality(
    signal_quality: Optional[Dict[str, Any]], analysis_scope: Optional[str] = None
) -> Dict[str, Any]:
    normalized = dict(signal_quality) if isinstance(signal_quality, dict) else {}
    merged = {
        **DEFAULT_SIGNAL_QUALITY,
        **normalized,
    }
    if merged.get("analysis_scope") is None and analysis_scope is not None:
        merged["analysis_scope"] = analysis_scope
    return merged


def _normalize_segment_evaluation(
    segment_evaluation: Optional[Dict[str, Any]], training_summary: Dict[str, Any]
) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    source = segment_evaluation if isinstance(segment_evaluation, dict) else {}

    for segment_name in SEGMENT_NAMES:
        segment_payload = source.get(segment_name)
        segment_payload = segment_payload if isinstance(segment_payload, dict) else {}
        dataset_samples = int(
            segment_payload.get(
                "dataset_samples", training_summary.get(f"{segment_name}_samples", 0) or 0
            )
            or 0
        )
        evaluated_samples = int(
            segment_payload.get("evaluated_samples", segment_payload.get("dataset_samples", 0))
            or 0
        )
        performance_metrics = segment_payload.get("performance_metrics")
        performance_metrics = (
            dict(performance_metrics) if isinstance(performance_metrics, dict) else {}
        )

        normalized[segment_name] = {
            "dataset_samples": dataset_samples,
            "evaluated_samples": evaluated_samples,
            "performance_metrics": performance_metrics,
            "signal_quality": _normalize_signal_quality(
                segment_payload.get("signal_quality"), analysis_scope=None
            ),
        }

    return normalized


def build_official_record_summary(report: Dict[str, Any]) -> Dict[str, Any]:
    segment_evaluation = report.get("segment_evaluation")
    segment_evaluation = segment_evaluation if isinstance(segment_evaluation, dict) else {}

    signal_record: Dict[str, Dict[str, Any]] = {}
    sig_ana_record: Dict[str, Dict[str, Any]] = {}

    for segment_name in SEGMENT_NAMES:
        segment_payload = segment_evaluation.get(segment_name)
        segment_payload = segment_payload if isinstance(segment_payload, dict) else {}
        raw_signal_quality = segment_payload.get("signal_quality")
        signal_quality = (
            _normalize_signal_quality(raw_signal_quality, analysis_scope=segment_name)
            if isinstance(raw_signal_quality, dict)
            else _normalize_signal_quality(None, analysis_scope=segment_name)
        )
        has_signal_quality = isinstance(raw_signal_quality, dict) and any(
            value is not None for key, value in signal_quality.items() if key != "analysis_scope"
        )

        signal_record_item = {
            "dataset_samples": int(segment_payload.get("dataset_samples", 0) or 0),
            "evaluated_samples": int(segment_payload.get("evaluated_samples", 0) or 0),
            "has_signal_quality": has_signal_quality,
        }
        if has_signal_quality and signal_quality.get("analysis_scope") is not None:
            signal_record_item["analysis_scope"] = signal_quality["analysis_scope"]

        signal_record[segment_name] = signal_record_item
        sig_ana_record[segment_name] = signal_quality

    portfolio_bridge_summary = report.get("portfolio_bridge_summary")
    portfolio_bridge_summary = (
        dict(portfolio_bridge_summary)
        if isinstance(portfolio_bridge_summary, dict)
        else _default_portfolio_bridge_summary(report.get("model_id"))
    )

    return {
        "signal_record": signal_record,
        "sig_ana_record": sig_ana_record,
        "port_ana_record": {
            "task_count": int(portfolio_bridge_summary.get("task_count", 0) or 0),
            "best_by_total_return": portfolio_bridge_summary.get("best_by_total_return"),
            "best_by_sharpe": portfolio_bridge_summary.get("best_by_sharpe"),
            "smallest_drawdown": portfolio_bridge_summary.get("smallest_drawdown"),
            "tasks": list(portfolio_bridge_summary.get("tasks") or []),
        },
    }


def normalize_report_payload(report: Dict[str, Any]) -> Dict[str, Any]:
    """兼容并补齐评估报告字段，保证前端/导出结构稳定。"""
    if not isinstance(report, dict):
        return report

    normalized = dict(report)
    training_summary = normalized.get("training_summary")
    training_summary = training_summary if isinstance(training_summary, dict) else {}

    training_data_info = normalized.get("training_data_info")
    training_data_info = dict(training_data_info) if isinstance(training_data_info, dict) else {}
    for field in ("total_samples", "train_samples", "validation_samples", "test_samples"):
        if training_data_info.get(field) is None and training_summary.get(field) is not None:
            training_data_info[field] = training_summary.get(field)
    normalized["training_data_info"] = training_data_info

    early_stopping_info = normalized.get("early_stopping_info")
    if not isinstance(early_stopping_info, dict):
        early_stopping_info = {}
    normalized["early_stopping_info"] = {
        **DEFAULT_EARLY_STOPPING_INFO,
        **early_stopping_info,
    }

    signal_quality = normalized.get("signal_quality")
    if not isinstance(signal_quality, dict):
        signal_quality = {}
    normalized["signal_quality"] = _normalize_signal_quality(signal_quality)

    source_segment_evaluation = normalized.get("segment_evaluation")
    normalized["segment_evaluation"] = _normalize_segment_evaluation(
        source_segment_evaluation, training_summary
    )

    portfolio_bridge_summary = normalized.get("portfolio_bridge_summary")
    if not isinstance(portfolio_bridge_summary, dict):
        portfolio_bridge_summary = _default_portfolio_bridge_summary(
            normalized.get("model_id")
        )
    normalized["portfolio_bridge_summary"] = portfolio_bridge_summary

    official_record_summary = normalized.get("official_record_summary")
    if not isinstance(official_record_summary, dict):
        official_record_summary = build_official_record_summary(
            {
                "model_id": normalized.get("model_id"),
                "portfolio_bridge_summary": portfolio_bridge_summary,
            }
        )
    normalized["official_record_summary"] = official_record_summary

    return normalized


@dataclass
class TrainingSummary:
    """训练摘要"""

    model_id: str
    model_name: str
    model_type: str
    training_duration: float  # 秒
    total_samples: int
    train_samples: int
    validation_samples: int
    test_samples: int
    epochs: int
    batch_size: int
    learning_rate: float


@dataclass
class PerformanceMetrics:
    """性能指标"""

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    rmse: float
    mae: float
    sharpe_ratio: Optional[float] = None
    total_return: Optional[float] = None
    max_drawdown: Optional[float] = None
    win_rate: Optional[float] = None


@dataclass
class FeatureImportance:
    """特征重要性"""

    feature_name: str
    importance: float
    rank: int


@dataclass
class TrainingHistory:
    """训练历史"""

    epoch: int
    train_loss: float
    val_loss: float
    train_accuracy: float
    val_accuracy: float
    timestamp: str


@dataclass
class ModelEvaluationReport:
    """模型评估报告"""

    model_id: str
    model_name: str
    model_type: str
    version: str
    created_at: str

    # 训练信息
    training_summary: TrainingSummary

    # 性能指标
    performance_metrics: PerformanceMetrics

    # 特征重要性
    feature_importance: List[FeatureImportance]

    # 训练历史
    training_history: List[TrainingHistory]

    # 超参数
    hyperparameters: Dict[str, Any]

    # 训练数据信息
    training_data_info: Dict[str, Any]

    # 预测结果分析
    prediction_analysis: Optional[Dict[str, Any]] = None

    # 模型对比
    model_comparison: Optional[Dict[str, Any]] = None

    # 建议和改进
    recommendations: Optional[List[str]] = None

    # 特征相关性
    feature_correlation: Optional[Dict[str, Any]] = None

    # 超参数调优摘要
    hyperparameter_tuning: Optional[Dict[str, Any]] = None

    # 早停信息
    early_stopping_info: Optional[Dict[str, Any]] = None

    # 官方风格信号质量评估
    signal_quality: Optional[Dict[str, Any]] = None

    # 分段评估信息
    segment_evaluation: Optional[Dict[str, Any]] = None

    # 与正式任务桥接后的摘要
    portfolio_bridge_summary: Optional[Dict[str, Any]] = None


class EvaluationReportGenerator:
    """评估报告生成器"""

    def __init__(self):
        self.reports: Dict[str, ModelEvaluationReport] = {}

    def generate_report(
        self,
        model_id: str,
        model_name: str,
        model_type: str,
        version: str,
        training_summary: Dict[str, Any],
        performance_metrics: Dict[str, Any],
        feature_importance: List[Dict[str, Any]],
        training_history: List[Dict[str, Any]],
        hyperparameters: Dict[str, Any],
        training_data_info: Dict[str, Any],
        prediction_analysis: Optional[Dict[str, Any]] = None,
        feature_correlation: Optional[Dict[str, Any]] = None,
        hyperparameter_tuning: Optional[Dict[str, Any]] = None,
        early_stopping_info: Optional[Dict[str, Any]] = None,
        signal_quality: Optional[Dict[str, Any]] = None,
        segment_evaluation: Optional[Dict[str, Any]] = None,
        portfolio_bridge_summary: Optional[Dict[str, Any]] = None,
    ) -> ModelEvaluationReport:
        """生成评估报告"""

        # 构建训练摘要
        summary = TrainingSummary(
            model_id=model_id,
            model_name=model_name,
            model_type=model_type,
            training_duration=training_summary.get("duration", 0.0),
            total_samples=training_summary.get("total_samples", 0),
            train_samples=training_summary.get("train_samples", 0),
            validation_samples=training_summary.get("validation_samples", 0),
            test_samples=training_summary.get("test_samples", 0),
            epochs=training_summary.get("epochs", 0),
            batch_size=training_summary.get("batch_size", 32),
            learning_rate=training_summary.get("learning_rate", 0.001),
        )

        # 构建性能指标
        metrics = PerformanceMetrics(
            accuracy=performance_metrics.get("accuracy", 0.0),
            precision=performance_metrics.get("precision", 0.0),
            recall=performance_metrics.get("recall", 0.0),
            f1_score=performance_metrics.get("f1_score", 0.0),
            rmse=performance_metrics.get("rmse", 0.0),
            mae=performance_metrics.get("mae", 0.0),
            sharpe_ratio=performance_metrics.get("sharpe_ratio"),
            total_return=performance_metrics.get("total_return"),
            max_drawdown=performance_metrics.get("max_drawdown"),
            win_rate=performance_metrics.get("win_rate"),
        )

        # 构建特征重要性
        features = []
        # 处理不同的特征重要性格式
        if isinstance(feature_importance, dict):
            # 如果是字典格式 {feature_name: importance}
            sorted_features = sorted(
                feature_importance.items(), key=lambda x: x[1], reverse=True
            )
            for i, (feat_name, importance) in enumerate(sorted_features):
                features.append(
                    FeatureImportance(
                        feature_name=feat_name, importance=float(importance), rank=i + 1
                    )
                )
        elif isinstance(feature_importance, list):
            # 如果是列表格式
            for i, feat in enumerate(feature_importance):
                if isinstance(feat, dict):
                    features.append(
                        FeatureImportance(
                            feature_name=feat.get("name", f"feature_{i}"),
                            importance=float(feat.get("importance", 0.0)),
                            rank=i + 1,
                        )
                    )
                elif isinstance(feat, str):
                    # 如果是字符串列表，使用默认重要性
                    features.append(
                        FeatureImportance(feature_name=feat, importance=0.0, rank=i + 1)
                    )

        # 构建训练历史
        history = []
        for hist in training_history:
            history.append(
                TrainingHistory(
                    epoch=int(hist.get("epoch", 0) or 0),
                    train_loss=(None if hist.get("train_loss") is None else float(hist.get("train_loss", 0.0))),
                    val_loss=(None if hist.get("val_loss") is None else float(hist.get("val_loss", 0.0))),
                    train_accuracy=float(hist.get("train_accuracy", 0.0) or 0.0),
                    val_accuracy=float(hist.get("val_accuracy", 0.0) or 0.0),
                    timestamp=hist.get("timestamp", datetime.now().isoformat()),
                )
            )

        # 生成建议
        recommendations = self._generate_recommendations(metrics, features)

        # 创建报告
        report = ModelEvaluationReport(
            model_id=model_id,
            model_name=model_name,
            model_type=model_type,
            version=version,
            created_at=datetime.now().isoformat(),
            training_summary=summary,
            performance_metrics=metrics,
            feature_importance=features,
            feature_correlation=feature_correlation,
            training_history=history,
            hyperparameters=hyperparameters,
            hyperparameter_tuning=hyperparameter_tuning,
            early_stopping_info=early_stopping_info,
            signal_quality=signal_quality,
            segment_evaluation=segment_evaluation,
            portfolio_bridge_summary=portfolio_bridge_summary,
            training_data_info=training_data_info,
            prediction_analysis=prediction_analysis,
            recommendations=recommendations,
        )

        self.reports[model_id] = report
        return report

    def _generate_recommendations(
        self, metrics: PerformanceMetrics, features: List[FeatureImportance]
    ) -> List[str]:
        """生成改进建议"""
        recommendations = []

        # 基于准确率的建议
        if metrics.accuracy < 0.6:
            recommendations.append("模型准确率较低，建议：增加训练数据、调整模型架构或进行特征工程")
        elif metrics.accuracy < 0.75:
            recommendations.append("模型准确率中等，可以通过超参数调优或集成学习提升性能")

        # 基于过拟合的建议
        if metrics.precision > 0.9 and metrics.recall < 0.5:
            recommendations.append("模型可能存在过拟合，建议增加正则化或使用更多训练数据")

        # 基于特征重要性的建议
        if features:
            top_features = [f for f in features[:5] if f.importance > 0.1]
            if len(top_features) < 3:
                recommendations.append("重要特征较少，建议进行特征工程或特征选择")

        # 基于夏普比率的建议
        if metrics.sharpe_ratio and metrics.sharpe_ratio < 1.0:
            recommendations.append("夏普比率较低，建议优化风险控制策略或调整预测阈值")

        if not recommendations:
            recommendations.append("模型性能良好，可以尝试进一步优化超参数或使用集成方法")

        return recommendations

    def to_dict(self, report: ModelEvaluationReport) -> Dict[str, Any]:
        """转换为字典"""
        return normalize_report_payload(
            {
                "model_id": report.model_id,
                "model_name": report.model_name,
                "model_type": report.model_type,
                "version": report.version,
                "created_at": report.created_at,
                "training_summary": asdict(report.training_summary),
                "performance_metrics": asdict(report.performance_metrics),
                "feature_importance": [asdict(f) for f in report.feature_importance],
                "feature_correlation": report.feature_correlation,
                "training_history": [asdict(h) for h in report.training_history],
                "hyperparameters": report.hyperparameters,
                "hyperparameter_tuning": report.hyperparameter_tuning,
                "early_stopping_info": report.early_stopping_info,
                "signal_quality": report.signal_quality,
                "segment_evaluation": report.segment_evaluation,
                "portfolio_bridge_summary": report.portfolio_bridge_summary,
                "training_data_info": report.training_data_info,
                "prediction_analysis": report.prediction_analysis,
                "model_comparison": report.model_comparison,
                "recommendations": report.recommendations,
            }
        )

    def to_json(self, report: ModelEvaluationReport) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(report), ensure_ascii=False, indent=2)

    def get_report(self, model_id: str) -> Optional[ModelEvaluationReport]:
        """获取报告"""
        return self.reports.get(model_id)
