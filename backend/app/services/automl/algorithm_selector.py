"""
智能算法选择器

基于数据特征自动推荐最适合的算法，提供：
- 数据特征分析
- 算法适用性评估
- 初始超参数建议
- 性能预估
"""

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler


class DataCharacteristic(Enum):
    """数据特征枚举"""

    SMALL_DATASET = "small_dataset"  # < 1000 samples
    MEDIUM_DATASET = "medium_dataset"  # 1000-10000 samples
    LARGE_DATASET = "large_dataset"  # > 10000 samples
    HIGH_DIMENSIONAL = "high_dimensional"  # > 100 features
    LOW_DIMENSIONAL = "low_dimensional"  # < 20 features
    TIME_SERIES = "time_series"
    NOISY_DATA = "noisy_data"
    SPARSE_DATA = "sparse_data"
    NON_LINEAR = "non_linear"
    SEASONAL = "seasonal"
    TRENDING = "trending"


@dataclass
class AlgorithmRecommendation:
    """算法推荐结果"""

    algorithm: str
    confidence: float
    reasoning: List[str]
    initial_params: Dict[str, Any]
    expected_performance: Optional[float] = None
    training_time_estimate: Optional[float] = None
    resource_requirements: Optional[Dict[str, Any]] = None


class AlgorithmSelector:
    """智能算法选择器"""

    def __init__(self):
        self.algorithm_profiles = self._initialize_algorithm_profiles()
        self.performance_history = {}

    def _initialize_algorithm_profiles(self) -> Dict[str, Dict[str, Any]]:
        """初始化算法特征档案"""
        return {
            "lightgbm": {
                "strengths": [
                    DataCharacteristic.MEDIUM_DATASET,
                    DataCharacteristic.LARGE_DATASET,
                    DataCharacteristic.HIGH_DIMENSIONAL,
                    DataCharacteristic.NON_LINEAR,
                ],
                "weaknesses": [
                    DataCharacteristic.SMALL_DATASET,
                    DataCharacteristic.SPARSE_DATA,
                ],
                "training_speed": "fast",
                "memory_usage": "low",
                "interpretability": "medium",
                "default_params": {
                    "n_estimators": 100,
                    "max_depth": 6,
                    "learning_rate": 0.1,
                    "num_leaves": 31,
                    "min_child_samples": 20,
                },
                "param_sensitivity": {
                    "learning_rate": "high",
                    "n_estimators": "medium",
                    "max_depth": "medium",
                },
            },
            "xgboost": {
                "strengths": [
                    DataCharacteristic.MEDIUM_DATASET,
                    DataCharacteristic.LARGE_DATASET,
                    DataCharacteristic.NON_LINEAR,
                    DataCharacteristic.NOISY_DATA,
                ],
                "weaknesses": [
                    DataCharacteristic.SMALL_DATASET,
                    DataCharacteristic.SPARSE_DATA,
                ],
                "training_speed": "medium",
                "memory_usage": "medium",
                "interpretability": "medium",
                "default_params": {
                    "n_estimators": 100,
                    "max_depth": 6,
                    "learning_rate": 0.1,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                },
                "param_sensitivity": {
                    "learning_rate": "high",
                    "n_estimators": "medium",
                    "max_depth": "high",
                },
            },
            "linear_regression": {
                "strengths": [
                    DataCharacteristic.SMALL_DATASET,
                    DataCharacteristic.LOW_DIMENSIONAL,
                    DataCharacteristic.SPARSE_DATA,
                ],
                "weaknesses": [
                    DataCharacteristic.NON_LINEAR,
                    DataCharacteristic.HIGH_DIMENSIONAL,
                ],
                "training_speed": "very_fast",
                "memory_usage": "very_low",
                "interpretability": "very_high",
                "default_params": {"fit_intercept": True, "normalize": False},
                "param_sensitivity": {"regularization": "low"},
            },
            "transformer": {
                "strengths": [
                    DataCharacteristic.LARGE_DATASET,
                    DataCharacteristic.TIME_SERIES,
                    DataCharacteristic.SEASONAL,
                    DataCharacteristic.NON_LINEAR,
                ],
                "weaknesses": [
                    DataCharacteristic.SMALL_DATASET,
                    DataCharacteristic.LOW_DIMENSIONAL,
                ],
                "training_speed": "slow",
                "memory_usage": "high",
                "interpretability": "low",
                "default_params": {
                    "d_model": 128,
                    "nhead": 8,
                    "num_layers": 4,
                    "dropout": 0.1,
                    "learning_rate": 1e-4,
                },
                "param_sensitivity": {
                    "learning_rate": "very_high",
                    "d_model": "high",
                    "num_layers": "medium",
                },
            },
            "informer": {
                "strengths": [
                    DataCharacteristic.LARGE_DATASET,
                    DataCharacteristic.TIME_SERIES,
                    DataCharacteristic.SEASONAL,
                    DataCharacteristic.TRENDING,
                ],
                "weaknesses": [
                    DataCharacteristic.SMALL_DATASET,
                    DataCharacteristic.LOW_DIMENSIONAL,
                ],
                "training_speed": "slow",
                "memory_usage": "high",
                "interpretability": "low",
                "default_params": {
                    "d_model": 128,
                    "n_heads": 8,
                    "e_layers": 2,
                    "d_layers": 1,
                    "dropout": 0.05,
                    "learning_rate": 1e-4,
                },
                "param_sensitivity": {
                    "learning_rate": "very_high",
                    "d_model": "high",
                    "e_layers": "medium",
                },
            },
            "timesnet": {
                "strengths": [
                    DataCharacteristic.LARGE_DATASET,
                    DataCharacteristic.TIME_SERIES,
                    DataCharacteristic.SEASONAL,
                    DataCharacteristic.NON_LINEAR,
                ],
                "weaknesses": [
                    DataCharacteristic.SMALL_DATASET,
                    DataCharacteristic.SPARSE_DATA,
                ],
                "training_speed": "slow",
                "memory_usage": "high",
                "interpretability": "low",
                "default_params": {
                    "d_model": 128,
                    "d_ff": 256,
                    "num_kernels": 6,
                    "dropout": 0.1,
                    "learning_rate": 1e-4,
                },
                "param_sensitivity": {
                    "learning_rate": "very_high",
                    "d_model": "high",
                    "num_kernels": "medium",
                },
            },
        }

    async def analyze_data_characteristics(
        self, data: pd.DataFrame, target_column: str, time_column: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        分析数据特征

        Args:
            data: 数据集
            target_column: 目标列名
            time_column: 时间列名

        Returns:
            Dict: 数据特征分析结果
        """
        try:
            characteristics = []
            analysis_results = {}

            # 基本统计信息
            n_samples, n_features = data.shape
            analysis_results["n_samples"] = n_samples
            analysis_results["n_features"] = n_features

            # 数据集大小特征
            if n_samples < 1000:
                characteristics.append(DataCharacteristic.SMALL_DATASET)
            elif n_samples < 10000:
                characteristics.append(DataCharacteristic.MEDIUM_DATASET)
            else:
                characteristics.append(DataCharacteristic.LARGE_DATASET)

            # 特征维度特征
            if n_features < 20:
                characteristics.append(DataCharacteristic.LOW_DIMENSIONAL)
            elif n_features > 100:
                characteristics.append(DataCharacteristic.HIGH_DIMENSIONAL)

            # 时间序列特征
            if time_column and time_column in data.columns:
                characteristics.append(DataCharacteristic.TIME_SERIES)

                # 季节性分析
                seasonality_score = await self._analyze_seasonality(
                    data, target_column, time_column
                )
                analysis_results["seasonality_score"] = seasonality_score
                if seasonality_score > 0.3:
                    characteristics.append(DataCharacteristic.SEASONAL)

                # 趋势分析
                trend_score = await self._analyze_trend(
                    data, target_column, time_column
                )
                analysis_results["trend_score"] = trend_score
                if abs(trend_score) > 0.2:
                    characteristics.append(DataCharacteristic.TRENDING)

            # 数据质量特征
            missing_ratio = data.isnull().sum().sum() / (n_samples * n_features)
            analysis_results["missing_ratio"] = missing_ratio

            # 稀疏性分析
            if missing_ratio > 0.3:
                characteristics.append(DataCharacteristic.SPARSE_DATA)

            # 噪声分析
            noise_score = await self._analyze_noise(data, target_column)
            analysis_results["noise_score"] = noise_score
            if noise_score > 0.5:
                characteristics.append(DataCharacteristic.NOISY_DATA)

            # 非线性分析
            nonlinearity_score = await self._analyze_nonlinearity(data, target_column)
            analysis_results["nonlinearity_score"] = nonlinearity_score
            if nonlinearity_score > 0.4:
                characteristics.append(DataCharacteristic.NON_LINEAR)

            # 特征相关性分析
            feature_correlations = await self._analyze_feature_correlations(
                data, target_column
            )
            analysis_results["feature_correlations"] = feature_correlations

            analysis_results["characteristics"] = [c.value for c in characteristics]
            analysis_results["analysis_timestamp"] = datetime.utcnow().isoformat()

            return analysis_results

        except Exception as e:
            logger.error(f"数据特征分析失败: {e}")
            return {
                "error": str(e),
                "characteristics": [],
                "analysis_timestamp": datetime.utcnow().isoformat(),
            }

    async def _analyze_seasonality(
        self, data: pd.DataFrame, target_column: str, time_column: str
    ) -> float:
        """分析季节性"""
        try:
            # 简单的季节性检测：计算不同周期的自相关
            target_series = data[target_column].dropna()

            if len(target_series) < 50:
                return 0.0

            # 检查不同周期的自相关
            periods = [7, 30, 90, 252]  # 周、月、季、年
            max_autocorr = 0.0

            for period in periods:
                if len(target_series) > period:
                    autocorr = target_series.autocorr(lag=period)
                    if not np.isnan(autocorr):
                        max_autocorr = max(max_autocorr, abs(autocorr))

            return max_autocorr

        except Exception as e:
            logger.warning(f"季节性分析失败: {e}")
            return 0.0

    async def _analyze_trend(
        self, data: pd.DataFrame, target_column: str, time_column: str
    ) -> float:
        """分析趋势"""
        try:
            target_series = data[target_column].dropna()

            if len(target_series) < 10:
                return 0.0

            # 使用线性回归检测趋势
            x = np.arange(len(target_series))
            y = target_series.values

            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

            # 返回标准化的斜率作为趋势强度
            trend_strength = slope / (np.std(y) + 1e-8)

            return trend_strength

        except Exception as e:
            logger.warning(f"趋势分析失败: {e}")
            return 0.0

    async def _analyze_noise(self, data: pd.DataFrame, target_column: str) -> float:
        """分析噪声水平"""
        try:
            target_series = data[target_column].dropna()

            if len(target_series) < 10:
                return 0.0

            # 计算信噪比的倒数作为噪声分数
            signal_power = np.var(target_series)

            # 使用移动平均来估计信号
            window_size = min(10, len(target_series) // 4)
            if window_size < 2:
                return 0.0

            smoothed = target_series.rolling(window=window_size, center=True).mean()
            noise = target_series - smoothed
            noise_power = np.var(noise.dropna())

            if signal_power == 0:
                return 1.0

            snr = signal_power / (noise_power + 1e-8)
            noise_score = 1.0 / (1.0 + snr)  # 转换为0-1的噪声分数

            return noise_score

        except Exception as e:
            logger.warning(f"噪声分析失败: {e}")
            return 0.0

    async def _analyze_nonlinearity(
        self, data: pd.DataFrame, target_column: str
    ) -> float:
        """分析非线性程度"""
        try:
            # 选择数值特征
            numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()
            if target_column in numeric_features:
                numeric_features.remove(target_column)

            if len(numeric_features) == 0:
                return 0.0

            # 计算互信息（非线性关系）和线性相关性的比值
            X = data[numeric_features].fillna(0)
            y = data[target_column].fillna(0)

            if len(X) < 10:
                return 0.0

            # 互信息
            mi_scores = mutual_info_regression(X, y, random_state=42)
            avg_mi = np.mean(mi_scores)

            # 线性相关性
            correlations = []
            for feature in numeric_features:
                corr = np.corrcoef(X[feature], y)[0, 1]
                if not np.isnan(corr):
                    correlations.append(abs(corr))

            avg_corr = np.mean(correlations) if correlations else 0.0

            # 非线性分数：互信息相对于线性相关性的比值
            if avg_corr == 0:
                return min(avg_mi, 1.0)

            nonlinearity_score = avg_mi / (avg_corr + 1e-8)
            return min(nonlinearity_score, 1.0)

        except Exception as e:
            logger.warning(f"非线性分析失败: {e}")
            return 0.0

    async def _analyze_feature_correlations(
        self, data: pd.DataFrame, target_column: str
    ) -> Dict[str, Any]:
        """分析特征相关性"""
        try:
            numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()
            if target_column in numeric_features:
                numeric_features.remove(target_column)

            if len(numeric_features) == 0:
                return {"error": "没有数值特征"}

            # 计算与目标变量的相关性
            target_correlations = {}
            for feature in numeric_features:
                corr = data[feature].corr(data[target_column])
                if not np.isnan(corr):
                    target_correlations[feature] = abs(corr)

            # 特征间相关性
            feature_corr_matrix = data[numeric_features].corr()
            high_corr_pairs = []

            for i in range(len(numeric_features)):
                for j in range(i + 1, len(numeric_features)):
                    corr = feature_corr_matrix.iloc[i, j]
                    if not np.isnan(corr) and abs(corr) > 0.8:
                        high_corr_pairs.append(
                            {
                                "feature1": numeric_features[i],
                                "feature2": numeric_features[j],
                                "correlation": corr,
                            }
                        )

            return {
                "target_correlations": target_correlations,
                "high_correlation_pairs": high_corr_pairs,
                "avg_target_correlation": np.mean(list(target_correlations.values()))
                if target_correlations
                else 0.0,
                "max_target_correlation": max(target_correlations.values())
                if target_correlations
                else 0.0,
            }

        except Exception as e:
            logger.warning(f"特征相关性分析失败: {e}")
            return {"error": str(e)}

    async def recommend_algorithms(
        self,
        data_characteristics: Dict[str, Any],
        constraints: Optional[Dict[str, Any]] = None,
        top_k: int = 3,
    ) -> List[AlgorithmRecommendation]:
        """
        推荐算法

        Args:
            data_characteristics: 数据特征分析结果
            constraints: 约束条件（如训练时间、内存等）
            top_k: 返回推荐数量

        Returns:
            List[AlgorithmRecommendation]: 算法推荐列表
        """
        try:
            characteristics = [
                DataCharacteristic(c)
                for c in data_characteristics.get("characteristics", [])
            ]

            algorithm_scores = {}

            for algorithm, profile in self.algorithm_profiles.items():
                score = await self._calculate_algorithm_score(
                    algorithm,
                    profile,
                    characteristics,
                    data_characteristics,
                    constraints,
                )
                algorithm_scores[algorithm] = score

            # 排序并选择top_k
            sorted_algorithms = sorted(
                algorithm_scores.items(),
                key=lambda x: x[1]["total_score"],
                reverse=True,
            )

            recommendations = []
            for algorithm, score_info in sorted_algorithms[:top_k]:
                recommendation = AlgorithmRecommendation(
                    algorithm=algorithm,
                    confidence=score_info["total_score"],
                    reasoning=score_info["reasoning"],
                    initial_params=self._get_initial_params(
                        algorithm, data_characteristics
                    ),
                    expected_performance=score_info.get("expected_performance"),
                    training_time_estimate=score_info.get("training_time_estimate"),
                    resource_requirements=score_info.get("resource_requirements"),
                )
                recommendations.append(recommendation)

            return recommendations

        except Exception as e:
            logger.error(f"算法推荐失败: {e}")
            return []

    async def _calculate_algorithm_score(
        self,
        algorithm: str,
        profile: Dict[str, Any],
        characteristics: List[DataCharacteristic],
        data_characteristics: Dict[str, Any],
        constraints: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """计算算法适用性分数"""

        score = 0.0
        reasoning = []

        # 基于优势特征加分
        strengths = profile.get("strengths", [])
        for strength in strengths:
            if strength in characteristics:
                score += 1.0
                reasoning.append(f"适合{strength.value}的数据特征")

        # 基于劣势特征减分
        weaknesses = profile.get("weaknesses", [])
        for weakness in weaknesses:
            if weakness in characteristics:
                score -= 0.5
                reasoning.append(f"不太适合{weakness.value}的数据特征")

        # 数据大小适配性
        n_samples = data_characteristics.get("n_samples", 0)
        if algorithm in ["transformer", "informer", "timesnet"] and n_samples < 5000:
            score -= 1.0
            reasoning.append("深度学习模型需要更多数据")
        elif algorithm == "linear_regression" and n_samples > 50000:
            score += 0.5
            reasoning.append("线性模型适合大数据集的快速训练")

        # 特征维度适配性
        n_features = data_characteristics.get("n_features", 0)
        if algorithm == "linear_regression" and n_features > 100:
            score -= 0.5
            reasoning.append("高维数据可能导致线性模型过拟合")

        # 约束条件检查
        if constraints:
            # 训练时间约束
            max_training_time = constraints.get("max_training_time")
            if max_training_time:
                training_speed = profile.get("training_speed", "medium")
                if training_speed == "slow" and max_training_time < 3600:  # 1小时
                    score -= 1.0
                    reasoning.append("训练时间约束不满足")
                elif training_speed in ["fast", "very_fast"]:
                    score += 0.5
                    reasoning.append("满足训练时间要求")

            # 内存约束
            max_memory = constraints.get("max_memory_gb")
            if max_memory:
                memory_usage = profile.get("memory_usage", "medium")
                if memory_usage == "high" and max_memory < 8:
                    score -= 1.0
                    reasoning.append("内存需求超出限制")
                elif memory_usage in ["low", "very_low"]:
                    score += 0.5
                    reasoning.append("内存使用效率高")

        # 历史性能加权
        if algorithm in self.performance_history:
            historical_performance = np.mean(self.performance_history[algorithm])
            score += historical_performance * 0.3
            reasoning.append(f"历史平均性能: {historical_performance:.3f}")

        # 标准化分数到0-1范围
        normalized_score = max(0.0, min(1.0, (score + 2.0) / 4.0))

        # 估算性能和资源需求
        expected_performance = self._estimate_performance(
            algorithm, data_characteristics
        )
        training_time_estimate = self._estimate_training_time(
            algorithm, data_characteristics
        )
        resource_requirements = self._estimate_resource_requirements(
            algorithm, data_characteristics
        )

        return {
            "total_score": normalized_score,
            "reasoning": reasoning,
            "expected_performance": expected_performance,
            "training_time_estimate": training_time_estimate,
            "resource_requirements": resource_requirements,
        }

    def _get_initial_params(
        self, algorithm: str, data_characteristics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """获取初始参数建议"""
        profile = self.algorithm_profiles.get(algorithm, {})
        default_params = profile.get("default_params", {}).copy()

        # 根据数据特征调整参数
        n_samples = data_characteristics.get("n_samples", 1000)
        n_features = data_characteristics.get("n_features", 10)

        if algorithm in ["lightgbm", "xgboost"]:
            # 根据数据大小调整树的数量
            if n_samples < 1000:
                default_params["n_estimators"] = 50
            elif n_samples > 10000:
                default_params["n_estimators"] = 200

            # 根据特征数量调整深度
            if n_features < 10:
                default_params["max_depth"] = 4
            elif n_features > 50:
                default_params["max_depth"] = 8

        elif algorithm in ["transformer", "informer", "timesnet"]:
            # 根据数据大小调整模型大小
            if n_samples < 5000:
                default_params["d_model"] = 64
                default_params["num_layers"] = 2
            elif n_samples > 50000:
                default_params["d_model"] = 256
                default_params["num_layers"] = 6

        return default_params

    def _estimate_performance(
        self, algorithm: str, data_characteristics: Dict[str, Any]
    ) -> float:
        """估算预期性能"""
        # 基于算法类型和数据特征的简单性能估算
        base_performance = {
            "lightgbm": 0.85,
            "xgboost": 0.83,
            "linear_regression": 0.70,
            "transformer": 0.88,
            "informer": 0.86,
            "timesnet": 0.87,
        }

        performance = base_performance.get(algorithm, 0.75)

        # 根据数据特征调整
        n_samples = data_characteristics.get("n_samples", 1000)
        noise_score = data_characteristics.get("noise_score", 0.3)

        # 数据量影响
        if n_samples < 500:
            performance *= 0.9
        elif n_samples > 10000:
            performance *= 1.05

        # 噪声影响
        performance *= 1.0 - noise_score * 0.2

        return min(0.95, max(0.5, performance))

    def _estimate_training_time(
        self, algorithm: str, data_characteristics: Dict[str, Any]
    ) -> float:
        """估算训练时间（分钟）"""
        n_samples = data_characteristics.get("n_samples", 1000)
        n_features = data_characteristics.get("n_features", 10)

        # 基础训练时间（分钟）
        base_times = {
            "linear_regression": 0.1,
            "lightgbm": 2.0,
            "xgboost": 3.0,
            "transformer": 30.0,
            "informer": 25.0,
            "timesnet": 35.0,
        }

        base_time = base_times.get(algorithm, 5.0)

        # 根据数据大小缩放
        scale_factor = (n_samples / 1000) * (n_features / 10)
        estimated_time = base_time * scale_factor

        return max(0.1, estimated_time)

    def _estimate_resource_requirements(
        self, algorithm: str, data_characteristics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """估算资源需求"""
        n_samples = data_characteristics.get("n_samples", 1000)
        n_features = data_characteristics.get("n_features", 10)

        # 基础资源需求
        base_requirements = {
            "linear_regression": {
                "cpu_cores": 1,
                "memory_gb": 0.5,
                "gpu_required": False,
            },
            "lightgbm": {"cpu_cores": 4, "memory_gb": 2.0, "gpu_required": False},
            "xgboost": {"cpu_cores": 4, "memory_gb": 3.0, "gpu_required": False},
            "transformer": {"cpu_cores": 8, "memory_gb": 8.0, "gpu_required": True},
            "informer": {"cpu_cores": 8, "memory_gb": 6.0, "gpu_required": True},
            "timesnet": {"cpu_cores": 8, "memory_gb": 10.0, "gpu_required": True},
        }

        requirements = base_requirements.get(
            algorithm, {"cpu_cores": 2, "memory_gb": 2.0, "gpu_required": False}
        ).copy()

        # 根据数据大小调整
        scale_factor = max(1.0, (n_samples * n_features) / 10000)
        requirements["memory_gb"] *= scale_factor

        return requirements

    def update_performance_history(self, algorithm: str, performance: float):
        """更新算法性能历史"""
        if algorithm not in self.performance_history:
            self.performance_history[algorithm] = []

        self.performance_history[algorithm].append(performance)

        # 保持最近50次记录
        if len(self.performance_history[algorithm]) > 50:
            self.performance_history[algorithm] = self.performance_history[algorithm][
                -50:
            ]


# 全局实例
algorithm_selector = AlgorithmSelector()
