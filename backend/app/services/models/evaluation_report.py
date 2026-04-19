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

DEFAULT_PORTFOLIO_BRIDGE_SUMMARY = {
    "model_id": None,
    "task_count": 0,
    "tasks": [],
    "best_by_total_return": None,
    "best_by_sharpe": None,
    "smallest_drawdown": None,
}

DEFAULT_OFFICIAL_RECORD_SUMMARY = {
    "signal_record": {
        "train": {"dataset_samples": 0, "evaluated_samples": 0, "has_signal_quality": False},
        "validation": {"dataset_samples": 0, "evaluated_samples": 0, "has_signal_quality": False},
        "test": {"dataset_samples": 0, "evaluated_samples": 0, "has_signal_quality": False},
    },
    "sig_ana_record": {
        "train": {**DEFAULT_SIGNAL_QUALITY, "analysis_scope": "train"},
        "validation": {**DEFAULT_SIGNAL_QUALITY, "analysis_scope": "validation"},
        "test": {**DEFAULT_SIGNAL_QUALITY, "analysis_scope": "test"},
    },
    "port_ana_record": {
        "task_count": 0,
        "best_by_total_return": None,
        "best_by_sharpe": None,
        "smallest_drawdown": None,
        "tasks": [],
    },
}

DEFAULT_RANKING_OVERLAP_SUMMARY = {
    "available": False,
    "windows": [],
}

DEFAULT_EVENT_REPLAY_SUMMARY = {
    "available": False,
    "events": [],
}

DEFAULT_PER_STOCK_RANKING_PREFERENCE = {
    "stocks": [],
    "best_overall": None,
    "worst_overall": None,
}

DEFAULT_COST_VS_GROSS_GAP_SUMMARY = {
    "task_count": 0,
    "tasks": [],
    "largest_cost_gap": None,
    "best_gross_return": None,
    "best_net_return": None,
}


def _normalize_segment_entry(segment_entry: Any, fallback_samples: int = 0) -> Dict[str, Any]:
    segment_entry = dict(segment_entry) if isinstance(segment_entry, dict) else {}
    performance_metrics = segment_entry.get("performance_metrics")
    performance_metrics = dict(performance_metrics) if isinstance(performance_metrics, dict) else {}
    signal_quality = segment_entry.get("signal_quality")
    signal_quality = {
        **DEFAULT_SIGNAL_QUALITY,
        **(signal_quality if isinstance(signal_quality, dict) else {}),
    }
    return {
        "dataset_samples": int(segment_entry.get("dataset_samples") or fallback_samples or 0),
        "evaluated_samples": int(segment_entry.get("evaluated_samples") or signal_quality.get("sample_count") or 0),
        "performance_metrics": performance_metrics,
        "signal_quality": signal_quality,
    }


def _build_signal_record_entry(segment_name: str, segment_entry: Dict[str, Any]) -> Dict[str, Any]:
    signal_quality = segment_entry.get("signal_quality")
    signal_quality = signal_quality if isinstance(signal_quality, dict) else {}
    return {
        "dataset_samples": int(segment_entry.get("dataset_samples") or 0),
        "evaluated_samples": int(segment_entry.get("evaluated_samples") or signal_quality.get("sample_count") or 0),
        "has_signal_quality": bool(signal_quality.get("sample_count") or signal_quality.get("rank_ic") is not None or signal_quality.get("ic") is not None),
        "analysis_scope": signal_quality.get("analysis_scope") or segment_name,
    }



def build_official_record_summary(report: Dict[str, Any]) -> Dict[str, Any]:
    """Build a Qlib-record-style summary from normalized report data."""
    normalized = normalize_report_payload(report)
    if not isinstance(normalized, dict):
        return DEFAULT_OFFICIAL_RECORD_SUMMARY

    segment_evaluation = normalized.get("segment_evaluation") or {}
    portfolio_bridge_summary = normalized.get("portfolio_bridge_summary") or {}
    validation_signal_quality = normalized.get("signal_quality") or {}

    signal_record = {
        "train": _build_signal_record_entry("train", segment_evaluation.get("train") or {}),
        "validation": _build_signal_record_entry("validation", segment_evaluation.get("validation") or {}),
        "test": _build_signal_record_entry("test", segment_evaluation.get("test") or {}),
    }

    sig_ana_record = {
        "train": (segment_evaluation.get("train") or {}).get("signal_quality") or {**DEFAULT_SIGNAL_QUALITY, "analysis_scope": "train"},
        "validation": (segment_evaluation.get("validation") or {}).get("signal_quality") or validation_signal_quality or {**DEFAULT_SIGNAL_QUALITY, "analysis_scope": "validation"},
        "test": (segment_evaluation.get("test") or {}).get("signal_quality") or {**DEFAULT_SIGNAL_QUALITY, "analysis_scope": "test"},
    }

    port_ana_record = {
        "task_count": int(portfolio_bridge_summary.get("task_count") or 0),
        "best_by_total_return": portfolio_bridge_summary.get("best_by_total_return"),
        "best_by_sharpe": portfolio_bridge_summary.get("best_by_sharpe"),
        "smallest_drawdown": portfolio_bridge_summary.get("smallest_drawdown"),
        "tasks": portfolio_bridge_summary.get("tasks", []),
    }

    return {
        "signal_record": signal_record,
        "sig_ana_record": sig_ana_record,
        "port_ana_record": port_ana_record,
    }



def build_bridge_extension_summaries(report: Dict[str, Any]) -> Dict[str, Any]:
    normalized = normalize_report_payload(report)
    if not isinstance(normalized, dict):
        return {
            "ranking_overlap_summary": DEFAULT_RANKING_OVERLAP_SUMMARY,
            "event_replay_summary": DEFAULT_EVENT_REPLAY_SUMMARY,
            "per_stock_ranking_preference": DEFAULT_PER_STOCK_RANKING_PREFERENCE,
            "cost_vs_gross_gap_summary": DEFAULT_COST_VS_GROSS_GAP_SUMMARY,
        }

    portfolio_bridge_summary = normalized.get("portfolio_bridge_summary") or {}
    return {
        "ranking_overlap_summary": normalized.get("ranking_overlap_summary")
        if isinstance(normalized.get("ranking_overlap_summary"), dict)
        else DEFAULT_RANKING_OVERLAP_SUMMARY,
        "event_replay_summary": normalized.get("event_replay_summary")
        if isinstance(normalized.get("event_replay_summary"), dict)
        else DEFAULT_EVENT_REPLAY_SUMMARY,
        "per_stock_ranking_preference": portfolio_bridge_summary.get("per_stock_contribution_rollup")
        if isinstance(portfolio_bridge_summary.get("per_stock_contribution_rollup"), dict)
        else DEFAULT_PER_STOCK_RANKING_PREFERENCE,
        "cost_vs_gross_gap_summary": portfolio_bridge_summary.get("cost_vs_gross_gap_rollup")
        if isinstance(portfolio_bridge_summary.get("cost_vs_gross_gap_rollup"), dict)
        else DEFAULT_COST_VS_GROSS_GAP_SUMMARY,
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
    normalized["signal_quality"] = {
        **DEFAULT_SIGNAL_QUALITY,
        **signal_quality,
    }

    segment_evaluation = normalized.get("segment_evaluation")
    if not isinstance(segment_evaluation, dict):
        segment_evaluation = {}
    normalized["segment_evaluation"] = {
        "train": _normalize_segment_entry(
            segment_evaluation.get("train"),
            fallback_samples=training_summary.get("train_samples", 0),
        ),
        "validation": _normalize_segment_entry(
            segment_evaluation.get("validation")
            or {
                "performance_metrics": normalized.get("performance_metrics", {}),
                "signal_quality": normalized["signal_quality"],
            },
            fallback_samples=training_summary.get("validation_samples", 0),
        ),
        "test": _normalize_segment_entry(
            segment_evaluation.get("test"),
            fallback_samples=training_summary.get("test_samples", 0),
        ),
    }

    portfolio_bridge_summary = normalized.get("portfolio_bridge_summary")
    if not isinstance(portfolio_bridge_summary, dict):
        portfolio_bridge_summary = {}
    normalized["portfolio_bridge_summary"] = {
        **DEFAULT_PORTFOLIO_BRIDGE_SUMMARY,
        **portfolio_bridge_summary,
    }

    official_record_summary = normalized.get("official_record_summary")
    if not isinstance(official_record_summary, dict):
        official_record_summary = {}
    normalized["official_record_summary"] = {
        **DEFAULT_OFFICIAL_RECORD_SUMMARY,
        **official_record_summary,
    }

    ranking_overlap_summary = normalized.get("ranking_overlap_summary")
    if not isinstance(ranking_overlap_summary, dict):
        ranking_overlap_summary = {}
    normalized["ranking_overlap_summary"] = {
        **DEFAULT_RANKING_OVERLAP_SUMMARY,
        **ranking_overlap_summary,
    }

    event_replay_summary = normalized.get("event_replay_summary")
    if not isinstance(event_replay_summary, dict):
        event_replay_summary = {}
    normalized["event_replay_summary"] = {
        **DEFAULT_EVENT_REPLAY_SUMMARY,
        **event_replay_summary,
    }

    per_stock_ranking_preference = normalized.get("per_stock_ranking_preference")
    if not isinstance(per_stock_ranking_preference, dict):
        per_stock_ranking_preference = {}
    normalized["per_stock_ranking_preference"] = {
        **DEFAULT_PER_STOCK_RANKING_PREFERENCE,
        **per_stock_ranking_preference,
    }

    cost_vs_gross_gap_summary = normalized.get("cost_vs_gross_gap_summary")
    if not isinstance(cost_vs_gross_gap_summary, dict):
        cost_vs_gross_gap_summary = {}
    normalized["cost_vs_gross_gap_summary"] = {
        **DEFAULT_COST_VS_GROSS_GAP_SUMMARY,
        **cost_vs_gross_gap_summary,
    }

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

    # 分段评估（train / validation / test）
    segment_evaluation: Optional[Dict[str, Any]] = None


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
