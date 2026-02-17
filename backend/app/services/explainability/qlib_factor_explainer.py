"""
量化因子解释功能
支持Qlib因子的解释性分析，实现因子贡献度可视化
"""
import json
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class FactorCategory(Enum):
    """因子类别"""

    PRICE = "price"  # 价格因子
    VOLUME = "volume"  # 成交量因子
    VOLATILITY = "volatility"  # 波动率因子
    MOMENTUM = "momentum"  # 动量因子
    REVERSAL = "reversal"  # 反转因子
    SIZE = "size"  # 规模因子
    VALUE = "value"  # 价值因子
    QUALITY = "quality"  # 质量因子
    GROWTH = "growth"  # 成长因子
    TECHNICAL = "technical"  # 技术因子


class ExplanationLevel(Enum):
    """解释层级"""

    FACTOR = "factor"  # 因子级别
    CATEGORY = "category"  # 类别级别
    STOCK = "stock"  # 个股级别
    PORTFOLIO = "portfolio"  # 组合级别


@dataclass
class FactorContribution:
    """因子贡献度"""

    factor_name: str
    factor_category: FactorCategory
    contribution_score: float
    contribution_percentage: float
    statistical_significance: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    factor_value: Optional[float] = None
    normalized_value: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "factor_name": self.factor_name,
            "factor_category": self.factor_category.value,
            "contribution_score": self.contribution_score,
            "contribution_percentage": self.contribution_percentage,
            "statistical_significance": self.statistical_significance,
            "confidence_interval": list(self.confidence_interval)
            if self.confidence_interval
            else None,
            "factor_value": self.factor_value,
            "normalized_value": self.normalized_value,
        }


@dataclass
class FactorExplanation:
    """因子解释结果"""

    explanation_id: str
    model_id: str
    model_version: str
    stock_code: Optional[str]
    prediction_date: datetime
    predicted_return: float
    actual_return: Optional[float]
    factor_contributions: List[FactorContribution]
    category_contributions: Dict[str, float]
    explanation_level: ExplanationLevel
    base_return: float
    total_attribution: float
    unexplained_variance: float
    created_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "explanation_id": self.explanation_id,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "stock_code": self.stock_code,
            "prediction_date": self.prediction_date.isoformat(),
            "predicted_return": self.predicted_return,
            "actual_return": self.actual_return,
            "factor_contributions": [fc.to_dict() for fc in self.factor_contributions],
            "category_contributions": self.category_contributions,
            "explanation_level": self.explanation_level.value,
            "base_return": self.base_return,
            "total_attribution": self.total_attribution,
            "unexplained_variance": self.unexplained_variance,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }


class QlibFactorMapper:
    """Qlib因子映射器"""

    # Alpha158因子到类别的映射
    FACTOR_CATEGORY_MAPPING = {
        # 价格因子
        "OPEN": FactorCategory.PRICE,
        "HIGH": FactorCategory.PRICE,
        "LOW": FactorCategory.PRICE,
        "CLOSE": FactorCategory.PRICE,
        "VWAP": FactorCategory.PRICE,
        # 成交量因子
        "VOLUME": FactorCategory.VOLUME,
        "TURN": FactorCategory.VOLUME,
        # 动量因子
        "ROC": FactorCategory.MOMENTUM,
        "MA": FactorCategory.MOMENTUM,
        "EMA": FactorCategory.MOMENTUM,
        "SLOPE": FactorCategory.MOMENTUM,
        "CORR": FactorCategory.MOMENTUM,
        # 反转因子
        "RESI": FactorCategory.REVERSAL,
        "WVMA": FactorCategory.REVERSAL,
        # 波动率因子
        "STD": FactorCategory.VOLATILITY,
        "BETA": FactorCategory.VOLATILITY,
        "RSQR": FactorCategory.VOLATILITY,
        # 技术因子
        "RSI": FactorCategory.TECHNICAL,
        "PSY": FactorCategory.TECHNICAL,
        "BIAS": FactorCategory.TECHNICAL,
        "WILLR": FactorCategory.TECHNICAL,
        "CCI": FactorCategory.TECHNICAL,
        "TR": FactorCategory.TECHNICAL,
        "ATR": FactorCategory.TECHNICAL,
        "DMA": FactorCategory.TECHNICAL,
        "DMI": FactorCategory.TECHNICAL,
        "ADTM": FactorCategory.TECHNICAL,
        "MACD": FactorCategory.TECHNICAL,
        "ADXR": FactorCategory.TECHNICAL,
        "TRIX": FactorCategory.TECHNICAL,
        "VR": FactorCategory.TECHNICAL,
        "EMV": FactorCategory.TECHNICAL,
        "CR": FactorCategory.TECHNICAL,
        "WR": FactorCategory.TECHNICAL,
    }

    @classmethod
    def get_factor_category(cls, factor_name: str) -> FactorCategory:
        """获取因子类别"""
        # 提取因子的基础名称（去除数字和特殊字符）
        base_name = "".join([c for c in factor_name if c.isalpha()]).upper()

        # 查找匹配的类别
        for pattern, category in cls.FACTOR_CATEGORY_MAPPING.items():
            if pattern in base_name:
                return category

        # 默认返回技术因子
        return FactorCategory.TECHNICAL

    @classmethod
    def parse_alpha158_factors(
        cls, factor_names: List[str]
    ) -> Dict[str, FactorCategory]:
        """解析Alpha158因子名称"""
        factor_categories = {}

        for factor_name in factor_names:
            category = cls.get_factor_category(factor_name)
            factor_categories[factor_name] = category

        return factor_categories


class FactorAttributionAnalyzer:
    """因子归因分析器"""

    def __init__(self):
        self.factor_mapper = QlibFactorMapper()
        self.scaler = StandardScaler()

    def analyze_factor_attribution(
        self,
        factor_values: pd.DataFrame,
        model_predictions: pd.Series,
        factor_weights: Optional[np.ndarray] = None,
        base_return: float = 0.0,
    ) -> List[FactorContribution]:
        """
        分析因子归因

        Args:
            factor_values: 因子值DataFrame
            model_predictions: 模型预测值
            factor_weights: 因子权重（如果模型提供）
            base_return: 基准收益率

        Returns:
            因子贡献度列表
        """
        contributions = []

        try:
            # 如果没有提供权重，使用线性回归估计
            if factor_weights is None:
                factor_weights = self._estimate_factor_weights(
                    factor_values, model_predictions
                )

            # 标准化因子值
            normalized_factors = self.scaler.fit_transform(factor_values)

            # 计算每个因子的贡献
            for i, factor_name in enumerate(factor_values.columns):
                # 因子值
                factor_value = factor_values.iloc[-1, i]  # 使用最新的因子值
                normalized_value = normalized_factors[-1, i]

                # 因子权重
                weight = factor_weights[i] if len(factor_weights) > i else 0.0

                # 贡献度计算
                contribution_score = weight * normalized_value

                # 获取因子类别
                factor_category = self.factor_mapper.get_factor_category(factor_name)

                # 计算统计显著性
                significance = self._calculate_factor_significance(
                    factor_values.iloc[:, i], model_predictions
                )

                contribution = FactorContribution(
                    factor_name=factor_name,
                    factor_category=factor_category,
                    contribution_score=contribution_score,
                    contribution_percentage=0.0,  # 稍后计算
                    statistical_significance=significance,
                    factor_value=factor_value,
                    normalized_value=normalized_value,
                )

                contributions.append(contribution)

            # 计算贡献度百分比
            total_abs_contribution = sum(
                abs(c.contribution_score) for c in contributions
            )

            if total_abs_contribution > 0:
                for contribution in contributions:
                    contribution.contribution_percentage = (
                        abs(contribution.contribution_score)
                        / total_abs_contribution
                        * 100
                    )

            # 按贡献度排序
            contributions.sort(key=lambda x: abs(x.contribution_score), reverse=True)

        except Exception as e:
            logger.error(f"因子归因分析失败: {e}")
            raise

        return contributions

    def _estimate_factor_weights(
        self, factor_values: pd.DataFrame, predictions: pd.Series
    ) -> np.ndarray:
        """估计因子权重"""
        try:
            from sklearn.linear_model import LinearRegression

            # 对齐数据
            aligned_factors = factor_values.loc[predictions.index]

            # 移除缺失值
            valid_mask = ~(aligned_factors.isnull().any(axis=1) | predictions.isnull())
            clean_factors = aligned_factors[valid_mask]
            clean_predictions = predictions[valid_mask]

            if len(clean_factors) < 10:  # 数据太少
                return np.zeros(len(factor_values.columns))

            # 线性回归
            model = LinearRegression()
            model.fit(clean_factors, clean_predictions)

            return model.coef_

        except Exception as e:
            logger.warning(f"估计因子权重失败: {e}")
            return np.zeros(len(factor_values.columns))

    def _calculate_factor_significance(
        self, factor_series: pd.Series, predictions: pd.Series
    ) -> Optional[float]:
        """计算因子统计显著性"""
        try:
            # 对齐数据
            aligned_data = pd.concat([factor_series, predictions], axis=1).dropna()

            if len(aligned_data) < 10:
                return None

            # 计算相关性的p值
            _, p_value = stats.pearsonr(
                aligned_data.iloc[:, 0], aligned_data.iloc[:, 1]
            )

            return p_value

        except Exception as e:
            logger.warning(f"计算因子显著性失败: {e}")
            return None

    def aggregate_category_contributions(
        self, factor_contributions: List[FactorContribution]
    ) -> Dict[str, float]:
        """聚合类别贡献度"""
        category_contributions = {}

        for contribution in factor_contributions:
            category = contribution.factor_category.value

            if category not in category_contributions:
                category_contributions[category] = 0.0

            category_contributions[category] += contribution.contribution_score

        return category_contributions


class FactorClusterAnalyzer:
    """因子聚类分析器"""

    def __init__(self, n_clusters: int = 5):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.pca = PCA(n_components=2)

    def analyze_factor_clusters(
        self,
        factor_values: pd.DataFrame,
        factor_contributions: List[FactorContribution],
    ) -> Dict[str, Any]:
        """
        分析因子聚类

        Args:
            factor_values: 因子值
            factor_contributions: 因子贡献度

        Returns:
            聚类分析结果
        """
        try:
            # 准备数据
            contribution_scores = np.array(
                [fc.contribution_score for fc in factor_contributions]
            )
            factor_names = [fc.factor_name for fc in factor_contributions]

            # 获取因子值的统计特征
            factor_stats = factor_values.describe().T

            # 组合特征：贡献度 + 统计特征
            features = np.column_stack(
                [
                    contribution_scores.reshape(-1, 1),
                    factor_stats[["mean", "std", "min", "max"]].values,
                ]
            )

            # 标准化
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)

            # 聚类
            cluster_labels = self.kmeans.fit_predict(features_scaled)

            # PCA降维用于可视化
            pca_features = self.pca.fit_transform(features_scaled)

            # 分析每个聚类
            cluster_analysis = {}
            for cluster_id in range(self.n_clusters):
                cluster_mask = cluster_labels == cluster_id
                cluster_factors = [
                    factor_names[i] for i in range(len(factor_names)) if cluster_mask[i]
                ]
                cluster_contributions = [
                    contribution_scores[i]
                    for i in range(len(contribution_scores))
                    if cluster_mask[i]
                ]

                if cluster_factors:
                    cluster_analysis[f"cluster_{cluster_id}"] = {
                        "factors": cluster_factors,
                        "avg_contribution": np.mean(cluster_contributions),
                        "total_contribution": np.sum(cluster_contributions),
                        "factor_count": len(cluster_factors),
                    }

            return {
                "cluster_labels": cluster_labels.tolist(),
                "cluster_centers": self.kmeans.cluster_centers_.tolist(),
                "pca_features": pca_features.tolist(),
                "cluster_analysis": cluster_analysis,
                "explained_variance_ratio": self.pca.explained_variance_ratio_.tolist(),
            }

        except Exception as e:
            logger.error(f"因子聚类分析失败: {e}")
            return {}


class QlibFactorExplainer:
    """Qlib因子解释器主类"""

    def __init__(self, storage_path: str = "data/factor_explanations"):
        """
        初始化因子解释器

        Args:
            storage_path: 存储路径
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.attribution_analyzer = FactorAttributionAnalyzer()
        self.cluster_analyzer = FactorClusterAnalyzer()

        # 解释结果存储
        self.explanations: Dict[str, FactorExplanation] = {}
        self.max_explanations = 10000

        # 线程锁
        self.lock = threading.Lock()

        logger.info(f"Qlib因子解释器初始化完成，存储路径: {self.storage_path}")

    def explain_prediction(
        self,
        model_id: str,
        model_version: str,
        factor_values: pd.DataFrame,
        prediction: float,
        stock_code: Optional[str] = None,
        prediction_date: Optional[datetime] = None,
        actual_return: Optional[float] = None,
        factor_weights: Optional[np.ndarray] = None,
        base_return: float = 0.0,
    ) -> str:
        """
        解释预测结果

        Args:
            model_id: 模型ID
            model_version: 模型版本
            factor_values: 因子值DataFrame
            prediction: 预测值
            stock_code: 股票代码
            prediction_date: 预测日期
            actual_return: 实际收益率
            factor_weights: 因子权重
            base_return: 基准收益率

        Returns:
            解释ID
        """
        if prediction_date is None:
            prediction_date = datetime.now()

        explanation_id = f"factor_exp_{int(datetime.now().timestamp())}_{model_id}"

        try:
            # 分析因子归因
            factor_contributions = self.attribution_analyzer.analyze_factor_attribution(
                factor_values=factor_values,
                model_predictions=pd.Series([prediction], index=[prediction_date]),
                factor_weights=factor_weights,
                base_return=base_return,
            )

            # 聚合类别贡献度
            category_contributions = (
                self.attribution_analyzer.aggregate_category_contributions(
                    factor_contributions
                )
            )

            # 计算总归因和未解释方差
            total_attribution = sum(
                fc.contribution_score for fc in factor_contributions
            )
            unexplained_variance = prediction - base_return - total_attribution

            # 创建解释结果
            explanation = FactorExplanation(
                explanation_id=explanation_id,
                model_id=model_id,
                model_version=model_version,
                stock_code=stock_code,
                prediction_date=prediction_date,
                predicted_return=prediction,
                actual_return=actual_return,
                factor_contributions=factor_contributions,
                category_contributions=category_contributions,
                explanation_level=ExplanationLevel.STOCK
                if stock_code
                else ExplanationLevel.PORTFOLIO,
                base_return=base_return,
                total_attribution=total_attribution,
                unexplained_variance=unexplained_variance,
                created_at=datetime.now(),
                metadata={
                    "num_factors": len(factor_contributions),
                    "top_factor": factor_contributions[0].factor_name
                    if factor_contributions
                    else None,
                    "attribution_coverage": abs(
                        total_attribution / (prediction - base_return)
                    )
                    if prediction != base_return
                    else 0,
                },
            )

            # 存储解释结果
            with self.lock:
                self.explanations[explanation_id] = explanation

                # 限制存储数量
                if len(self.explanations) > self.max_explanations:
                    oldest_id = min(
                        self.explanations.keys(),
                        key=lambda x: self.explanations[x].created_at,
                    )
                    del self.explanations[oldest_id]

            # 保存解释结果
            self._save_explanation(explanation)

            logger.info(f"生成因子解释: {explanation_id}")
            return explanation_id

        except Exception as e:
            logger.error(f"因子解释失败: {e}")
            raise

    def explain_portfolio_prediction(
        self,
        model_id: str,
        model_version: str,
        portfolio_factor_values: Dict[str, pd.DataFrame],
        portfolio_predictions: Dict[str, float],
        portfolio_weights: Dict[str, float],
        prediction_date: Optional[datetime] = None,
    ) -> str:
        """
        解释组合预测

        Args:
            model_id: 模型ID
            model_version: 模型版本
            portfolio_factor_values: 组合因子值字典 {stock_code: factor_df}
            portfolio_predictions: 组合预测字典 {stock_code: prediction}
            portfolio_weights: 组合权重字典 {stock_code: weight}
            prediction_date: 预测日期

        Returns:
            解释ID
        """
        if prediction_date is None:
            prediction_date = datetime.now()

        explanation_id = f"portfolio_exp_{int(datetime.now().timestamp())}_{model_id}"

        try:
            # 聚合组合级别的因子贡献
            portfolio_factor_contributions = {}
            total_portfolio_prediction = 0.0

            for stock_code, weight in portfolio_weights.items():
                if (
                    stock_code in portfolio_factor_values
                    and stock_code in portfolio_predictions
                ):
                    # 获取个股的因子解释
                    stock_explanation_id = self.explain_prediction(
                        model_id=model_id,
                        model_version=model_version,
                        factor_values=portfolio_factor_values[stock_code],
                        prediction=portfolio_predictions[stock_code],
                        stock_code=stock_code,
                        prediction_date=prediction_date,
                    )

                    stock_explanation = self.explanations[stock_explanation_id]

                    # 按权重聚合因子贡献
                    for fc in stock_explanation.factor_contributions:
                        if fc.factor_name not in portfolio_factor_contributions:
                            portfolio_factor_contributions[fc.factor_name] = {
                                "contribution_score": 0.0,
                                "factor_category": fc.factor_category,
                                "weighted_count": 0.0,
                            }

                        portfolio_factor_contributions[fc.factor_name][
                            "contribution_score"
                        ] += (fc.contribution_score * weight)
                        portfolio_factor_contributions[fc.factor_name][
                            "weighted_count"
                        ] += weight

                    total_portfolio_prediction += (
                        portfolio_predictions[stock_code] * weight
                    )

            # 转换为FactorContribution对象
            factor_contributions = []
            for factor_name, data in portfolio_factor_contributions.items():
                contribution = FactorContribution(
                    factor_name=factor_name,
                    factor_category=data["factor_category"],
                    contribution_score=data["contribution_score"],
                    contribution_percentage=0.0,  # 稍后计算
                )
                factor_contributions.append(contribution)

            # 计算贡献度百分比
            total_abs_contribution = sum(
                abs(fc.contribution_score) for fc in factor_contributions
            )
            if total_abs_contribution > 0:
                for fc in factor_contributions:
                    fc.contribution_percentage = (
                        abs(fc.contribution_score) / total_abs_contribution * 100
                    )

            # 排序
            factor_contributions.sort(
                key=lambda x: abs(x.contribution_score), reverse=True
            )

            # 聚合类别贡献度
            category_contributions = (
                self.attribution_analyzer.aggregate_category_contributions(
                    factor_contributions
                )
            )

            # 创建组合解释结果
            explanation = FactorExplanation(
                explanation_id=explanation_id,
                model_id=model_id,
                model_version=model_version,
                stock_code=None,  # 组合级别
                prediction_date=prediction_date,
                predicted_return=total_portfolio_prediction,
                actual_return=None,
                factor_contributions=factor_contributions,
                category_contributions=category_contributions,
                explanation_level=ExplanationLevel.PORTFOLIO,
                base_return=0.0,
                total_attribution=sum(
                    fc.contribution_score for fc in factor_contributions
                ),
                unexplained_variance=0.0,  # 组合级别暂不计算
                created_at=datetime.now(),
                metadata={
                    "portfolio_size": len(portfolio_weights),
                    "num_factors": len(factor_contributions),
                    "portfolio_weights": portfolio_weights,
                },
            )

            # 存储解释结果
            with self.lock:
                self.explanations[explanation_id] = explanation

            # 保存解释结果
            self._save_explanation(explanation)

            logger.info(f"生成组合因子解释: {explanation_id}")
            return explanation_id

        except Exception as e:
            logger.error(f"组合因子解释失败: {e}")
            raise

    def get_explanation(self, explanation_id: str) -> Optional[FactorExplanation]:
        """获取解释结果"""
        return self.explanations.get(explanation_id)

    def get_model_explanations(
        self,
        model_id: str,
        model_version: Optional[str] = None,
        stock_code: Optional[str] = None,
        limit: int = 100,
    ) -> List[FactorExplanation]:
        """获取模型的解释结果"""
        explanations = []

        for explanation in self.explanations.values():
            if explanation.model_id != model_id:
                continue

            if model_version and explanation.model_version != model_version:
                continue

            if stock_code and explanation.stock_code != stock_code:
                continue

            explanations.append(explanation)

        # 按创建时间排序
        explanations.sort(key=lambda x: x.created_at, reverse=True)

        return explanations[:limit]

    def get_factor_importance_ranking(
        self, model_id: str, model_version: Optional[str] = None, top_k: int = 20
    ) -> List[Tuple[str, float, str]]:
        """
        获取因子重要性排名

        Returns:
            (因子名称, 平均贡献度, 因子类别) 的列表
        """
        explanations = self.get_model_explanations(model_id, model_version)

        if not explanations:
            return []

        # 聚合因子贡献度
        factor_contributions = {}
        factor_categories = {}
        factor_counts = {}

        for explanation in explanations:
            for fc in explanation.factor_contributions:
                name = fc.factor_name

                if name not in factor_contributions:
                    factor_contributions[name] = 0.0
                    factor_categories[name] = fc.factor_category.value
                    factor_counts[name] = 0

                factor_contributions[name] += abs(fc.contribution_score)
                factor_counts[name] += 1

        # 计算平均贡献度
        avg_contributions = {}
        for name in factor_contributions:
            avg_contributions[name] = factor_contributions[name] / factor_counts[name]

        # 排序并返回前K个
        sorted_factors = sorted(
            [
                (name, score, factor_categories[name])
                for name, score in avg_contributions.items()
            ],
            key=lambda x: x[1],
            reverse=True,
        )

        return sorted_factors[:top_k]

    def get_category_importance_ranking(
        self, model_id: str, model_version: Optional[str] = None
    ) -> List[Tuple[str, float]]:
        """获取因子类别重要性排名"""
        explanations = self.get_model_explanations(model_id, model_version)

        if not explanations:
            return []

        # 聚合类别贡献度
        category_contributions = {}
        category_counts = {}

        for explanation in explanations:
            for category, contribution in explanation.category_contributions.items():
                if category not in category_contributions:
                    category_contributions[category] = 0.0
                    category_counts[category] = 0

                category_contributions[category] += abs(contribution)
                category_counts[category] += 1

        # 计算平均贡献度
        avg_contributions = {}
        for category in category_contributions:
            avg_contributions[category] = (
                category_contributions[category] / category_counts[category]
            )

        # 排序
        sorted_categories = sorted(
            avg_contributions.items(), key=lambda x: x[1], reverse=True
        )

        return sorted_categories

    def generate_factor_report(
        self, model_id: str, model_version: Optional[str] = None
    ) -> Dict[str, Any]:
        """生成因子分析报告"""
        explanations = self.get_model_explanations(model_id, model_version)

        if not explanations:
            return {"error": "没有找到解释结果"}

        # 基本统计
        total_explanations = len(explanations)

        # 因子重要性排名
        factor_ranking = self.get_factor_importance_ranking(
            model_id, model_version, top_k=20
        )

        # 类别重要性排名
        category_ranking = self.get_category_importance_ranking(model_id, model_version)

        # 预测准确性分析（如果有实际收益率）
        accuracy_stats = {}
        predictions_with_actual = [
            e for e in explanations if e.actual_return is not None
        ]

        if predictions_with_actual:
            predicted_returns = [e.predicted_return for e in predictions_with_actual]
            actual_returns = [e.actual_return for e in predictions_with_actual]

            correlation = np.corrcoef(predicted_returns, actual_returns)[0, 1]
            mse = np.mean(
                [(p - a) ** 2 for p, a in zip(predicted_returns, actual_returns)]
            )
            mae = np.mean(
                [abs(p - a) for p, a in zip(predicted_returns, actual_returns)]
            )

            accuracy_stats = {
                "correlation": correlation,
                "mse": mse,
                "mae": mae,
                "num_samples": len(predictions_with_actual),
            }

        # 归因覆盖率统计
        attribution_coverages = []
        for explanation in explanations:
            if explanation.predicted_return != explanation.base_return:
                coverage = abs(
                    explanation.total_attribution
                    / (explanation.predicted_return - explanation.base_return)
                )
                attribution_coverages.append(coverage)

        avg_attribution_coverage = (
            np.mean(attribution_coverages) if attribution_coverages else 0
        )

        return {
            "model_id": model_id,
            "model_version": model_version,
            "total_explanations": total_explanations,
            "top_factors": factor_ranking,
            "category_importance": category_ranking,
            "accuracy_stats": accuracy_stats,
            "avg_attribution_coverage": avg_attribution_coverage,
            "generated_at": datetime.now().isoformat(),
        }

    def analyze_factor_clusters(
        self, model_id: str, model_version: Optional[str] = None
    ) -> Dict[str, Any]:
        """分析因子聚类"""
        explanations = self.get_model_explanations(model_id, model_version)

        if not explanations:
            return {"error": "没有找到解释结果"}

        # 聚合所有因子贡献度
        all_factor_contributions = []
        factor_names = set()

        for explanation in explanations:
            all_factor_contributions.extend(explanation.factor_contributions)
            factor_names.update(
                fc.factor_name for fc in explanation.factor_contributions
            )

        if not all_factor_contributions:
            return {"error": "没有有效的因子贡献度数据"}

        # 创建因子值DataFrame（模拟数据，实际应该从数据源获取）
        factor_values = pd.DataFrame(
            np.random.randn(100, len(factor_names)), columns=list(factor_names)
        )

        # 执行聚类分析
        cluster_result = self.cluster_analyzer.analyze_factor_clusters(
            factor_values, all_factor_contributions
        )

        return cluster_result

    def _save_explanation(self, explanation: FactorExplanation):
        """保存解释结果"""
        try:
            explanation_file = self.storage_path / f"{explanation.explanation_id}.json"

            with open(explanation_file, "w", encoding="utf-8") as f:
                json.dump(explanation.to_dict(), f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(f"保存因子解释结果失败: {e}")


# 全局Qlib因子解释器实例
qlib_factor_explainer = QlibFactorExplainer()
