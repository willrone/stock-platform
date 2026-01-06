"""
模型评估报告生成服务

生成详细的训练和评估报告，包括：
- 训练过程分析
- 性能指标可视化数据
- 特征重要性分析
- 预测结果分析
- 模型对比
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
import json
from loguru import logger


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
        hyperparameter_tuning: Optional[Dict[str, Any]] = None
    ) -> ModelEvaluationReport:
        """生成评估报告"""
        
        # 构建训练摘要
        summary = TrainingSummary(
            model_id=model_id,
            model_name=model_name,
            model_type=model_type,
            training_duration=training_summary.get('duration', 0.0),
            total_samples=training_summary.get('total_samples', 0),
            train_samples=training_summary.get('train_samples', 0),
            validation_samples=training_summary.get('validation_samples', 0),
            test_samples=training_summary.get('test_samples', 0),
            epochs=training_summary.get('epochs', 0),
            batch_size=training_summary.get('batch_size', 32),
            learning_rate=training_summary.get('learning_rate', 0.001)
        )
        
        # 构建性能指标
        metrics = PerformanceMetrics(
            accuracy=performance_metrics.get('accuracy', 0.0),
            precision=performance_metrics.get('precision', 0.0),
            recall=performance_metrics.get('recall', 0.0),
            f1_score=performance_metrics.get('f1_score', 0.0),
            rmse=performance_metrics.get('rmse', 0.0),
            mae=performance_metrics.get('mae', 0.0),
            sharpe_ratio=performance_metrics.get('sharpe_ratio'),
            total_return=performance_metrics.get('total_return'),
            max_drawdown=performance_metrics.get('max_drawdown'),
            win_rate=performance_metrics.get('win_rate')
        )
        
        # 构建特征重要性
        features = []
        # 处理不同的特征重要性格式
        if isinstance(feature_importance, dict):
            # 如果是字典格式 {feature_name: importance}
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            for i, (feat_name, importance) in enumerate(sorted_features):
                features.append(FeatureImportance(
                    feature_name=feat_name,
                    importance=float(importance),
                    rank=i + 1
                ))
        elif isinstance(feature_importance, list):
            # 如果是列表格式
            for i, feat in enumerate(feature_importance):
                if isinstance(feat, dict):
                    features.append(FeatureImportance(
                        feature_name=feat.get('name', f'feature_{i}'),
                        importance=float(feat.get('importance', 0.0)),
                        rank=i + 1
                    ))
                elif isinstance(feat, str):
                    # 如果是字符串列表，使用默认重要性
                    features.append(FeatureImportance(
                        feature_name=feat,
                        importance=0.0,
                        rank=i + 1
                    ))
        
        # 构建训练历史
        history = []
        for hist in training_history:
            history.append(TrainingHistory(
                epoch=hist.get('epoch', 0),
                train_loss=hist.get('train_loss', 0.0),
                val_loss=hist.get('val_loss', 0.0),
                train_accuracy=hist.get('train_accuracy', 0.0),
                val_accuracy=hist.get('val_accuracy', 0.0),
                timestamp=hist.get('timestamp', datetime.now().isoformat())
            ))
        
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
            training_data_info=training_data_info,
            prediction_analysis=prediction_analysis,
            recommendations=recommendations
        )
        
        self.reports[model_id] = report
        return report
    
    def _generate_recommendations(
        self,
        metrics: PerformanceMetrics,
        features: List[FeatureImportance]
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
        return {
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
            "training_data_info": report.training_data_info,
            "prediction_analysis": report.prediction_analysis,
            "model_comparison": report.model_comparison,
            "recommendations": report.recommendations
        }
    
    def to_json(self, report: ModelEvaluationReport) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(report), ensure_ascii=False, indent=2)
    
    def get_report(self, model_id: str) -> Optional[ModelEvaluationReport]:
        """获取报告"""
        return self.reports.get(model_id)
