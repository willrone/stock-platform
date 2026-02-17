"""
数据漂移检测器
检测输入数据分布变化，量化漂移程度并生成报告
"""
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger
from scipy import stats
from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class DriftType(Enum):
    """漂移类型"""

    FEATURE_DRIFT = "feature_drift"
    PREDICTION_DRIFT = "prediction_drift"
    CONCEPT_DRIFT = "concept_drift"


class DriftSeverity(Enum):
    """漂移严重程度"""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DriftMethod(Enum):
    """漂移检测方法"""

    KS_TEST = "ks_test"  # Kolmogorov-Smirnov检验
    WASSERSTEIN = "wasserstein"  # Wasserstein距离
    PSI = "psi"  # Population Stability Index
    JENSEN_SHANNON = "jensen_shannon"  # Jensen-Shannon散度
    PCA_RECONSTRUCTION = "pca_reconstruction"  # PCA重构误差


@dataclass
class DriftResult:
    """漂移检测结果"""

    feature_name: str
    drift_type: DriftType
    method: DriftMethod
    drift_score: float
    p_value: Optional[float]
    severity: DriftSeverity
    threshold: float
    timestamp: datetime
    reference_period: str
    detection_period: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_name": self.feature_name,
            "drift_type": self.drift_type.value,
            "method": self.method.value,
            "drift_score": self.drift_score,
            "p_value": self.p_value,
            "severity": self.severity.value,
            "threshold": self.threshold,
            "timestamp": self.timestamp.isoformat(),
            "reference_period": self.reference_period,
            "detection_period": self.detection_period,
            "details": self.details,
        }


@dataclass
class DriftReport:
    """漂移报告"""

    report_id: str
    timestamp: datetime
    model_id: str
    model_version: str
    overall_drift_score: float
    overall_severity: DriftSeverity
    feature_results: List[DriftResult]
    summary: Dict[str, Any]
    recommendations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "timestamp": self.timestamp.isoformat(),
            "model_id": self.model_id,
            "model_version": self.model_version,
            "overall_drift_score": self.overall_drift_score,
            "overall_severity": self.overall_severity.value,
            "feature_results": [r.to_dict() for r in self.feature_results],
            "summary": self.summary,
            "recommendations": self.recommendations,
        }


class StatisticalDriftDetector:
    """统计漂移检测器"""

    def __init__(self):
        self.methods = {
            DriftMethod.KS_TEST: self._ks_test,
            DriftMethod.WASSERSTEIN: self._wasserstein_distance,
            DriftMethod.PSI: self._psi_score,
            DriftMethod.JENSEN_SHANNON: self._jensen_shannon_divergence,
        }

    def detect_drift(
        self,
        reference_data: np.ndarray,
        current_data: np.ndarray,
        method: DriftMethod = DriftMethod.KS_TEST,
        threshold: float = 0.05,
    ) -> Tuple[float, Optional[float], DriftSeverity]:
        """
        检测数据漂移

        Args:
            reference_data: 参考数据
            current_data: 当前数据
            method: 检测方法
            threshold: 阈值

        Returns:
            (漂移分数, p值, 严重程度)
        """
        if method not in self.methods:
            raise ValueError(f"不支持的检测方法: {method}")

        detector_func = self.methods[method]
        drift_score, p_value = detector_func(reference_data, current_data)

        # 确定严重程度
        severity = self._determine_severity(drift_score, p_value, method, threshold)

        return drift_score, p_value, severity

    def _ks_test(
        self, ref_data: np.ndarray, cur_data: np.ndarray
    ) -> Tuple[float, float]:
        """Kolmogorov-Smirnov检验"""
        statistic, p_value = stats.ks_2samp(ref_data, cur_data)
        return float(statistic), float(p_value)

    def _wasserstein_distance(
        self, ref_data: np.ndarray, cur_data: np.ndarray
    ) -> Tuple[float, None]:
        """Wasserstein距离"""
        distance = wasserstein_distance(ref_data, cur_data)
        return float(distance), None

    def _psi_score(
        self, ref_data: np.ndarray, cur_data: np.ndarray, bins: int = 10
    ) -> Tuple[float, None]:
        """Population Stability Index"""
        # 创建分箱
        _, bin_edges = np.histogram(ref_data, bins=bins)

        # 计算参考和当前数据的分布
        ref_counts, _ = np.histogram(ref_data, bins=bin_edges)
        cur_counts, _ = np.histogram(cur_data, bins=bin_edges)

        # 转换为概率
        ref_probs = ref_counts / len(ref_data)
        cur_probs = cur_counts / len(cur_data)

        # 避免零概率
        ref_probs = np.where(ref_probs == 0, 1e-6, ref_probs)
        cur_probs = np.where(cur_probs == 0, 1e-6, cur_probs)

        # 计算PSI
        psi = np.sum((cur_probs - ref_probs) * np.log(cur_probs / ref_probs))

        return float(psi), None

    def _jensen_shannon_divergence(
        self, ref_data: np.ndarray, cur_data: np.ndarray, bins: int = 10
    ) -> Tuple[float, None]:
        """Jensen-Shannon散度"""
        # 创建分箱
        combined_data = np.concatenate([ref_data, cur_data])
        _, bin_edges = np.histogram(combined_data, bins=bins)

        # 计算概率分布
        ref_counts, _ = np.histogram(ref_data, bins=bin_edges)
        cur_counts, _ = np.histogram(cur_data, bins=bin_edges)

        ref_probs = ref_counts / len(ref_data)
        cur_probs = cur_counts / len(cur_data)

        # 避免零概率
        ref_probs = np.where(ref_probs == 0, 1e-6, ref_probs)
        cur_probs = np.where(cur_probs == 0, 1e-6, cur_probs)

        # 计算JS散度
        m = 0.5 * (ref_probs + cur_probs)
        js_div = 0.5 * stats.entropy(ref_probs, m) + 0.5 * stats.entropy(cur_probs, m)

        return float(js_div), None

    def _determine_severity(
        self,
        drift_score: float,
        p_value: Optional[float],
        method: DriftMethod,
        threshold: float,
    ) -> DriftSeverity:
        """确定漂移严重程度"""
        if method == DriftMethod.KS_TEST:
            if p_value is None:
                return DriftSeverity.NONE

            if p_value > 0.1:
                return DriftSeverity.NONE
            elif p_value > 0.05:
                return DriftSeverity.LOW
            elif p_value > 0.01:
                return DriftSeverity.MEDIUM
            elif p_value > 0.001:
                return DriftSeverity.HIGH
            else:
                return DriftSeverity.CRITICAL

        elif method == DriftMethod.PSI:
            if drift_score < 0.1:
                return DriftSeverity.NONE
            elif drift_score < 0.2:
                return DriftSeverity.LOW
            elif drift_score < 0.3:
                return DriftSeverity.MEDIUM
            elif drift_score < 0.5:
                return DriftSeverity.HIGH
            else:
                return DriftSeverity.CRITICAL

        elif method == DriftMethod.WASSERSTEIN:
            # 基于数据范围的相对距离
            if drift_score < threshold * 0.5:
                return DriftSeverity.NONE
            elif drift_score < threshold:
                return DriftSeverity.LOW
            elif drift_score < threshold * 2:
                return DriftSeverity.MEDIUM
            elif drift_score < threshold * 5:
                return DriftSeverity.HIGH
            else:
                return DriftSeverity.CRITICAL

        elif method == DriftMethod.JENSEN_SHANNON:
            if drift_score < 0.1:
                return DriftSeverity.NONE
            elif drift_score < 0.2:
                return DriftSeverity.LOW
            elif drift_score < 0.4:
                return DriftSeverity.MEDIUM
            elif drift_score < 0.6:
                return DriftSeverity.HIGH
            else:
                return DriftSeverity.CRITICAL

        return DriftSeverity.NONE


class MultivariateDriftDetector:
    """多变量漂移检测器"""

    def __init__(self, n_components: int = 5):
        """
        初始化多变量漂移检测器

        Args:
            n_components: PCA主成分数量
        """
        self.n_components = n_components
        self.pca = None
        self.scaler = None
        self.reference_reconstructed = None

    def fit_reference(self, reference_data: np.ndarray):
        """拟合参考数据"""
        # 标准化
        self.scaler = StandardScaler()
        scaled_data = self.scaler.fit_transform(reference_data)

        # PCA降维
        self.pca = PCA(n_components=min(self.n_components, scaled_data.shape[1]))
        self.pca.fit(scaled_data)

        # 计算重构数据
        transformed = self.pca.transform(scaled_data)
        self.reference_reconstructed = self.pca.inverse_transform(transformed)

        logger.info(
            f"多变量漂移检测器已拟合，解释方差比: {self.pca.explained_variance_ratio_.sum():.3f}"
        )

    def detect_drift(self, current_data: np.ndarray) -> Tuple[float, DriftSeverity]:
        """
        检测多变量漂移

        Args:
            current_data: 当前数据

        Returns:
            (漂移分数, 严重程度)
        """
        if self.pca is None or self.scaler is None:
            raise ValueError("需要先调用fit_reference方法")

        # 标准化当前数据
        scaled_current = self.scaler.transform(current_data)

        # PCA变换和重构
        transformed_current = self.pca.transform(scaled_current)
        reconstructed_current = self.pca.inverse_transform(transformed_current)

        # 计算重构误差
        ref_error = np.mean(
            np.sum(
                (
                    self.reference_reconstructed
                    - self.scaler.transform(
                        self.scaler.inverse_transform(self.reference_reconstructed)
                    )
                )
                ** 2,
                axis=1,
            )
        )

        cur_error = np.mean(
            np.sum((reconstructed_current - scaled_current) ** 2, axis=1)
        )

        # 计算漂移分数（相对重构误差）
        drift_score = cur_error / (ref_error + 1e-8)

        # 确定严重程度
        if drift_score < 1.2:
            severity = DriftSeverity.NONE
        elif drift_score < 1.5:
            severity = DriftSeverity.LOW
        elif drift_score < 2.0:
            severity = DriftSeverity.MEDIUM
        elif drift_score < 3.0:
            severity = DriftSeverity.HIGH
        else:
            severity = DriftSeverity.CRITICAL

        return float(drift_score), severity


class DriftDetector:
    """数据漂移检测器主类"""

    def __init__(self, window_size: int = 1000, reference_window_days: int = 7):
        """
        初始化漂移检测器

        Args:
            window_size: 滑动窗口大小
            reference_window_days: 参考窗口天数
        """
        self.window_size = window_size
        self.reference_window_days = reference_window_days

        # 数据存储
        self.feature_data: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=window_size)
        )
        self.prediction_data: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=window_size)
        )

        # 检测器
        self.statistical_detector = StatisticalDriftDetector()
        self.multivariate_detector = MultivariateDriftDetector()

        # 参考数据
        self.reference_data: Dict[str, np.ndarray] = {}
        self.reference_fitted = False

        # 检测历史
        self.drift_reports: List[DriftReport] = []
        self.max_reports = 1000

        # 线程锁
        self.lock = threading.Lock()

        logger.info("数据漂移检测器初始化完成")

    def add_sample(
        self,
        model_id: str,
        model_version: str,
        features: Dict[str, float],
        prediction: Optional[float] = None,
        timestamp: Optional[datetime] = None,
    ):
        """添加样本数据"""
        if timestamp is None:
            timestamp = datetime.now()

        key = f"{model_id}_{model_version}"

        with self.lock:
            # 存储特征数据
            sample_data = {
                "timestamp": timestamp,
                "features": features,
                "prediction": prediction,
            }
            self.feature_data[key].append(sample_data)

            if prediction is not None:
                self.prediction_data[key].append(
                    {"timestamp": timestamp, "prediction": prediction}
                )

    def fit_reference_data(self, model_id: str, model_version: str):
        """拟合参考数据"""
        key = f"{model_id}_{model_version}"

        with self.lock:
            if key not in self.feature_data or len(self.feature_data[key]) < 100:
                logger.warning(f"模型 {key} 的数据不足，无法拟合参考数据")
                return

            # 获取参考期间的数据
            cutoff_time = datetime.now() - timedelta(days=self.reference_window_days)
            reference_samples = [
                sample
                for sample in self.feature_data[key]
                if sample["timestamp"] >= cutoff_time
            ]

            if len(reference_samples) < 50:
                logger.warning(f"模型 {key} 的参考数据不足")
                return

            # 提取特征矩阵
            feature_names = list(reference_samples[0]["features"].keys())
            feature_matrix = np.array(
                [
                    [sample["features"][name] for name in feature_names]
                    for sample in reference_samples
                ]
            )

            # 存储参考数据
            self.reference_data[key] = {
                "features": feature_matrix,
                "feature_names": feature_names,
                "timestamp": datetime.now(),
            }

            # 拟合多变量检测器
            try:
                self.multivariate_detector.fit_reference(feature_matrix)
                self.reference_fitted = True
                logger.info(f"已拟合模型 {key} 的参考数据")
            except Exception as e:
                logger.error(f"拟合参考数据失败: {e}")

    def detect_drift(
        self,
        model_id: str,
        model_version: str,
        methods: List[DriftMethod] = None,
        detection_window_hours: int = 1,
    ) -> Optional[DriftReport]:
        """
        检测数据漂移

        Args:
            model_id: 模型ID
            model_version: 模型版本
            methods: 检测方法列表
            detection_window_hours: 检测窗口小时数

        Returns:
            漂移报告
        """
        if methods is None:
            methods = [DriftMethod.KS_TEST, DriftMethod.PSI]

        key = f"{model_id}_{model_version}"

        with self.lock:
            # 检查是否有参考数据
            if key not in self.reference_data:
                logger.warning(f"模型 {key} 没有参考数据，先拟合参考数据")
                self.fit_reference_data(model_id, model_version)
                if key not in self.reference_data:
                    return None

            # 获取检测期间的数据
            cutoff_time = datetime.now() - timedelta(hours=detection_window_hours)
            current_samples = [
                sample
                for sample in self.feature_data[key]
                if sample["timestamp"] >= cutoff_time
            ]

            if len(current_samples) < 20:
                logger.warning(f"模型 {key} 的当前数据不足，无法进行漂移检测")
                return None

            # 提取当前特征矩阵
            reference_info = self.reference_data[key]
            feature_names = reference_info["feature_names"]

            current_matrix = np.array(
                [
                    [sample["features"].get(name, 0.0) for name in feature_names]
                    for sample in current_samples
                ]
            )

            # 执行漂移检测
            feature_results = []

            # 单变量检测
            for i, feature_name in enumerate(feature_names):
                ref_feature = reference_info["features"][:, i]
                cur_feature = current_matrix[:, i]

                for method in methods:
                    try:
                        (
                            drift_score,
                            p_value,
                            severity,
                        ) = self.statistical_detector.detect_drift(
                            ref_feature, cur_feature, method
                        )

                        result = DriftResult(
                            feature_name=feature_name,
                            drift_type=DriftType.FEATURE_DRIFT,
                            method=method,
                            drift_score=drift_score,
                            p_value=p_value,
                            severity=severity,
                            threshold=0.05 if method == DriftMethod.KS_TEST else 0.2,
                            timestamp=datetime.now(),
                            reference_period=f"{self.reference_window_days}天前至今",
                            detection_period=f"{detection_window_hours}小时内",
                            details={
                                "reference_samples": len(ref_feature),
                                "current_samples": len(cur_feature),
                                "reference_mean": float(np.mean(ref_feature)),
                                "current_mean": float(np.mean(cur_feature)),
                                "reference_std": float(np.std(ref_feature)),
                                "current_std": float(np.std(cur_feature)),
                            },
                        )

                        feature_results.append(result)

                    except Exception as e:
                        logger.error(f"特征 {feature_name} 的 {method.value} 检测失败: {e}")

            # 多变量检测
            if self.reference_fitted:
                try:
                    (
                        mv_drift_score,
                        mv_severity,
                    ) = self.multivariate_detector.detect_drift(current_matrix)

                    mv_result = DriftResult(
                        feature_name="multivariate",
                        drift_type=DriftType.FEATURE_DRIFT,
                        method=DriftMethod.PCA_RECONSTRUCTION,
                        drift_score=mv_drift_score,
                        p_value=None,
                        severity=mv_severity,
                        threshold=1.5,
                        timestamp=datetime.now(),
                        reference_period=f"{self.reference_window_days}天前至今",
                        detection_period=f"{detection_window_hours}小时内",
                        details={
                            "n_features": len(feature_names),
                            "pca_components": self.multivariate_detector.n_components,
                            "explained_variance": float(
                                self.multivariate_detector.pca.explained_variance_ratio_.sum()
                            ),
                        },
                    )

                    feature_results.append(mv_result)

                except Exception as e:
                    logger.error(f"多变量漂移检测失败: {e}")

            # 计算总体漂移分数和严重程度
            if feature_results:
                drift_scores = [
                    r.drift_score for r in feature_results if r.drift_score is not None
                ]
                overall_drift_score = np.mean(drift_scores) if drift_scores else 0.0

                severities = [r.severity for r in feature_results]
                severity_levels = {
                    DriftSeverity.NONE: 0,
                    DriftSeverity.LOW: 1,
                    DriftSeverity.MEDIUM: 2,
                    DriftSeverity.HIGH: 3,
                    DriftSeverity.CRITICAL: 4,
                }

                max_severity_level = max(severity_levels[s] for s in severities)
                overall_severity = [
                    k for k, v in severity_levels.items() if v == max_severity_level
                ][0]
            else:
                overall_drift_score = 0.0
                overall_severity = DriftSeverity.NONE

            # 生成摘要和建议
            summary = self._generate_summary(feature_results)
            recommendations = self._generate_recommendations(
                feature_results, overall_severity
            )

            # 创建报告
            report = DriftReport(
                report_id=f"drift_{int(datetime.now().timestamp())}_{model_id}",
                timestamp=datetime.now(),
                model_id=model_id,
                model_version=model_version,
                overall_drift_score=overall_drift_score,
                overall_severity=overall_severity,
                feature_results=feature_results,
                summary=summary,
                recommendations=recommendations,
            )

            # 存储报告
            self.drift_reports.append(report)
            if len(self.drift_reports) > self.max_reports:
                self.drift_reports = self.drift_reports[-self.max_reports :]

            logger.info(f"完成漂移检测: {model_id}, 总体严重程度: {overall_severity.value}")
            return report

    def _generate_summary(self, results: List[DriftResult]) -> Dict[str, Any]:
        """生成摘要统计"""
        if not results:
            return {}

        severity_counts = defaultdict(int)
        method_counts = defaultdict(int)
        drift_type_counts = defaultdict(int)

        for result in results:
            severity_counts[result.severity.value] += 1
            method_counts[result.method.value] += 1
            drift_type_counts[result.drift_type.value] += 1

        return {
            "total_features_checked": len(results),
            "severity_distribution": dict(severity_counts),
            "method_distribution": dict(method_counts),
            "drift_type_distribution": dict(drift_type_counts),
            "features_with_drift": len(
                [r for r in results if r.severity != DriftSeverity.NONE]
            ),
        }

    def _generate_recommendations(
        self, results: List[DriftResult], overall_severity: DriftSeverity
    ) -> List[str]:
        """生成建议"""
        recommendations = []

        if overall_severity == DriftSeverity.NONE:
            recommendations.append("数据分布稳定，无需采取行动")
        elif overall_severity == DriftSeverity.LOW:
            recommendations.append("检测到轻微数据漂移，建议继续监控")
            recommendations.append("可以考虑增加数据收集频率")
        elif overall_severity == DriftSeverity.MEDIUM:
            recommendations.append("检测到中等程度数据漂移，建议调查原因")
            recommendations.append("考虑重新训练模型或调整特征工程")
        elif overall_severity == DriftSeverity.HIGH:
            recommendations.append("检测到严重数据漂移，需要立即关注")
            recommendations.append("强烈建议重新训练模型")
            recommendations.append("检查数据源是否发生变化")
        elif overall_severity == DriftSeverity.CRITICAL:
            recommendations.append("检测到严重数据漂移，模型可能不再可靠")
            recommendations.append("建议暂停使用当前模型")
            recommendations.append("立即重新训练模型或回滚到稳定版本")

        # 针对特定特征的建议
        high_drift_features = [
            r.feature_name
            for r in results
            if r.severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL]
        ]
        if high_drift_features:
            recommendations.append(f"重点关注漂移严重的特征: {', '.join(high_drift_features)}")

        return recommendations

    def get_drift_history(
        self,
        model_id: Optional[str] = None,
        model_version: Optional[str] = None,
        limit: int = 100,
    ) -> List[DriftReport]:
        """获取漂移检测历史"""
        with self.lock:
            reports = self.drift_reports

            if model_id:
                reports = [r for r in reports if r.model_id == model_id]

            if model_version:
                reports = [r for r in reports if r.model_version == model_version]

            return reports[-limit:]

    def get_latest_report(
        self, model_id: str, model_version: str
    ) -> Optional[DriftReport]:
        """获取最新的漂移报告"""
        reports = self.get_drift_history(model_id, model_version, limit=1)
        return reports[0] if reports else None

    def clear_reference_data(self, model_id: str, model_version: str):
        """清除参考数据"""
        key = f"{model_id}_{model_version}"
        with self.lock:
            if key in self.reference_data:
                del self.reference_data[key]
                logger.info(f"已清除模型 {key} 的参考数据")

    def get_drift_metrics(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        model_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """获取漂移检测指标（API 兼容方法）"""
        with self.lock:
            reports = self.drift_reports.copy()

        # 时间过滤
        if start_time:
            reports = [r for r in reports if r.timestamp >= start_time]
        if end_time:
            reports = [r for r in reports if r.timestamp <= end_time]

        # 模型过滤
        if model_id:
            reports = [r for r in reports if r.model_id == model_id]

        return [r.to_dict() for r in reports[-limit:]]

    def get_model_drift_status(
        self,
        model_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """获取模型漂移状态（API 兼容方法）"""
        with self.lock:
            reports = [r for r in self.drift_reports if r.model_id == model_id]

        # 时间过滤
        if start_time:
            reports = [r for r in reports if r.timestamp >= start_time]
        if end_time:
            reports = [r for r in reports if r.timestamp <= end_time]

        if not reports:
            return {
                "model_id": model_id,
                "drift_detected": False,
                "drift_score": 0,
                "last_drift_time": None,
            }

        latest_report = reports[-1]
        drift_detected = latest_report.overall_severity not in [
            DriftSeverity.NONE,
            DriftSeverity.LOW,
        ]

        return {
            "model_id": model_id,
            "drift_detected": drift_detected,
            "drift_score": latest_report.overall_drift_score,
            "severity": latest_report.overall_severity.value,
            "last_drift_time": latest_report.timestamp.isoformat()
            if drift_detected
            else None,
            "total_reports": len(reports),
        }

    def get_overall_drift_status(self) -> Dict[str, Any]:
        """获取整体漂移状态（API 兼容方法）"""
        with self.lock:
            if not self.drift_reports:
                return {
                    "total_models_monitored": 0,
                    "models_with_drift": 0,
                    "overall_health": "healthy",
                    "last_check": None,
                }

            # 获取最近的报告（每个模型最新的一个）
            latest_by_model: Dict[str, DriftReport] = {}
            for report in self.drift_reports:
                key = f"{report.model_id}_{report.model_version}"
                latest_by_model[key] = report

            # 统计
            models_with_drift = sum(
                1
                for r in latest_by_model.values()
                if r.overall_severity not in [DriftSeverity.NONE, DriftSeverity.LOW]
            )

            # 确定整体健康状态
            severities = [r.overall_severity for r in latest_by_model.values()]
            if any(s == DriftSeverity.CRITICAL for s in severities):
                overall_health = "critical"
            elif any(s == DriftSeverity.HIGH for s in severities):
                overall_health = "warning"
            elif any(s == DriftSeverity.MEDIUM for s in severities):
                overall_health = "attention"
            else:
                overall_health = "healthy"

            latest_report = max(self.drift_reports, key=lambda r: r.timestamp)

            return {
                "total_models_monitored": len(latest_by_model),
                "models_with_drift": models_with_drift,
                "overall_health": overall_health,
                "last_check": latest_report.timestamp.isoformat(),
                "severity_distribution": {
                    s.value: sum(
                        1 for r in latest_by_model.values() if r.overall_severity == s
                    )
                    for s in DriftSeverity
                },
            }


# 全局漂移检测器实例
drift_detector = DriftDetector()
