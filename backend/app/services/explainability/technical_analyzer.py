"""
技术指标影响分析器
分析RSI、MACD等指标对预测的影响，提供指标重要性排序
"""
import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.inspection import permutation_importance


class TechnicalIndicator(Enum):
    """技术指标类型"""

    RSI = "rsi"  # 相对强弱指数
    MACD = "macd"  # 移动平均收敛散度
    BOLLINGER_BANDS = "bollinger_bands"  # 布林带
    MOVING_AVERAGE = "moving_average"  # 移动平均线
    STOCHASTIC = "stochastic"  # 随机指标
    WILLIAMS_R = "williams_r"  # 威廉指标
    CCI = "cci"  # 商品通道指数
    ATR = "atr"  # 平均真实范围
    VOLUME = "volume"  # 成交量指标
    MOMENTUM = "momentum"  # 动量指标


class AnalysisMethod(Enum):
    """分析方法"""

    CORRELATION = "correlation"  # 相关性分析
    MUTUAL_INFO = "mutual_info"  # 互信息
    FEATURE_IMPORTANCE = "feature_importance"  # 特征重要性
    PERMUTATION = "permutation"  # 排列重要性
    STATISTICAL_TEST = "statistical_test"  # 统计检验
    SHAP_VALUES = "shap_values"  # SHAP值


@dataclass
class IndicatorImportance:
    """指标重要性"""

    indicator_name: str
    indicator_type: TechnicalIndicator
    importance_score: float
    analysis_method: AnalysisMethod
    confidence_interval: Optional[Tuple[float, float]] = None
    p_value: Optional[float] = None
    additional_metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "indicator_name": self.indicator_name,
            "indicator_type": self.indicator_type.value,
            "importance_score": self.importance_score,
            "analysis_method": self.analysis_method.value,
            "confidence_interval": list(self.confidence_interval)
            if self.confidence_interval
            else None,
            "p_value": self.p_value,
            "additional_metrics": self.additional_metrics,
        }


@dataclass
class IndicatorAnalysisResult:
    """指标分析结果"""

    analysis_id: str
    model_id: str
    model_version: str
    target_variable: str
    analysis_period: str
    indicator_importance: List[IndicatorImportance]
    summary_stats: Dict[str, Any]
    recommendations: List[str]
    created_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "analysis_id": self.analysis_id,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "target_variable": self.target_variable,
            "analysis_period": self.analysis_period,
            "indicator_importance": [
                imp.to_dict() for imp in self.indicator_importance
            ],
            "summary_stats": self.summary_stats,
            "recommendations": self.recommendations,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }


class TechnicalIndicatorCalculator:
    """技术指标计算器"""

    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI指标"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def calculate_macd(
        prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> Dict[str, pd.Series]:
        """计算MACD指标"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line

        return {"macd": macd_line, "signal": signal_line, "histogram": histogram}

    @staticmethod
    def calculate_bollinger_bands(
        prices: pd.Series, period: int = 20, std_dev: float = 2
    ) -> Dict[str, pd.Series]:
        """计算布林带"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()

        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)

        return {
            "upper": upper_band,
            "middle": sma,
            "lower": lower_band,
            "width": upper_band - lower_band,
            "position": (prices - lower_band) / (upper_band - lower_band),
        }

    @staticmethod
    def calculate_moving_average(
        prices: pd.Series, period: int = 20, ma_type: str = "sma"
    ) -> pd.Series:
        """计算移动平均线"""
        if ma_type == "sma":
            return prices.rolling(window=period).mean()
        elif ma_type == "ema":
            return prices.ewm(span=period).mean()
        else:
            raise ValueError(f"不支持的移动平均类型: {ma_type}")

    @staticmethod
    def calculate_stochastic(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k_period: int = 14,
        d_period: int = 3,
    ) -> Dict[str, pd.Series]:
        """计算随机指标"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()

        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()

        return {"k": k_percent, "d": d_percent}

    @staticmethod
    def calculate_williams_r(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> pd.Series:
        """计算威廉指标"""
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()

        williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
        return williams_r

    @staticmethod
    def calculate_cci(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20
    ) -> pd.Series:
        """计算商品通道指数"""
        typical_price = (high + low + close) / 3
        sma = typical_price.rolling(window=period).mean()
        mean_deviation = typical_price.rolling(window=period).apply(
            lambda x: np.mean(np.abs(x - x.mean()))
        )

        cci = (typical_price - sma) / (0.015 * mean_deviation)
        return cci

    @staticmethod
    def calculate_atr(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> pd.Series:
        """计算平均真实范围"""
        tr1 = high - low
        tr2 = np.abs(high - close.shift(1))
        tr3 = np.abs(low - close.shift(1))

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr


class IndicatorImpactAnalyzer:
    """指标影响分析器"""

    def __init__(self):
        self.calculator = TechnicalIndicatorCalculator()

    def analyze_correlation(
        self, indicators_data: pd.DataFrame, target: pd.Series, method: str = "pearson"
    ) -> List[IndicatorImportance]:
        """相关性分析"""
        results = []

        for column in indicators_data.columns:
            try:
                if method == "pearson":
                    corr, p_value = stats.pearsonr(
                        indicators_data[column].dropna(),
                        target[indicators_data[column].dropna().index],
                    )
                elif method == "spearman":
                    corr, p_value = stats.spearmanr(
                        indicators_data[column].dropna(),
                        target[indicators_data[column].dropna().index],
                    )
                else:
                    raise ValueError(f"不支持的相关性方法: {method}")

                # 确定指标类型
                indicator_type = self._infer_indicator_type(column)

                importance = IndicatorImportance(
                    indicator_name=column,
                    indicator_type=indicator_type,
                    importance_score=abs(corr),
                    analysis_method=AnalysisMethod.CORRELATION,
                    p_value=p_value,
                    additional_metrics={
                        "correlation": corr,
                        "correlation_method": method,
                    },
                )

                results.append(importance)

            except Exception as e:
                logger.warning(f"计算 {column} 的相关性失败: {e}")

        return sorted(results, key=lambda x: x.importance_score, reverse=True)

    def analyze_mutual_information(
        self,
        indicators_data: pd.DataFrame,
        target: pd.Series,
        discrete_features: bool = False,
    ) -> List[IndicatorImportance]:
        """互信息分析"""
        results = []

        # 处理缺失值
        clean_data = indicators_data.dropna()
        clean_target = target[clean_data.index]

        try:
            mi_scores = mutual_info_regression(
                clean_data,
                clean_target,
                discrete_features=discrete_features,
                random_state=42,
            )

            for i, column in enumerate(clean_data.columns):
                indicator_type = self._infer_indicator_type(column)

                importance = IndicatorImportance(
                    indicator_name=column,
                    indicator_type=indicator_type,
                    importance_score=mi_scores[i],
                    analysis_method=AnalysisMethod.MUTUAL_INFO,
                    additional_metrics={"mutual_info_score": mi_scores[i]},
                )

                results.append(importance)

        except Exception as e:
            logger.error(f"互信息分析失败: {e}")

        return sorted(results, key=lambda x: x.importance_score, reverse=True)

    def analyze_feature_importance(
        self,
        indicators_data: pd.DataFrame,
        target: pd.Series,
        model_type: str = "random_forest",
    ) -> List[IndicatorImportance]:
        """特征重要性分析"""
        results = []

        # 处理缺失值
        clean_data = indicators_data.dropna()
        clean_target = target[clean_data.index]

        try:
            if model_type == "random_forest":
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(clean_data, clean_target)
                importances = model.feature_importances_
            else:
                raise ValueError(f"不支持的模型类型: {model_type}")

            for i, column in enumerate(clean_data.columns):
                indicator_type = self._infer_indicator_type(column)

                importance = IndicatorImportance(
                    indicator_name=column,
                    indicator_type=indicator_type,
                    importance_score=importances[i],
                    analysis_method=AnalysisMethod.FEATURE_IMPORTANCE,
                    additional_metrics={
                        "feature_importance": importances[i],
                        "model_type": model_type,
                    },
                )

                results.append(importance)

        except Exception as e:
            logger.error(f"特征重要性分析失败: {e}")

        return sorted(results, key=lambda x: x.importance_score, reverse=True)

    def analyze_permutation_importance(
        self,
        indicators_data: pd.DataFrame,
        target: pd.Series,
        model_type: str = "random_forest",
        n_repeats: int = 10,
    ) -> List[IndicatorImportance]:
        """排列重要性分析"""
        results = []

        # 处理缺失值
        clean_data = indicators_data.dropna()
        clean_target = target[clean_data.index]

        try:
            if model_type == "random_forest":
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            else:
                raise ValueError(f"不支持的模型类型: {model_type}")

            model.fit(clean_data, clean_target)

            # 计算排列重要性
            perm_importance = permutation_importance(
                model, clean_data, clean_target, n_repeats=n_repeats, random_state=42
            )

            for i, column in enumerate(clean_data.columns):
                indicator_type = self._infer_indicator_type(column)

                # 计算置信区间
                mean_importance = perm_importance.importances_mean[i]
                std_importance = perm_importance.importances_std[i]
                ci_lower = mean_importance - 1.96 * std_importance
                ci_upper = mean_importance + 1.96 * std_importance

                importance = IndicatorImportance(
                    indicator_name=column,
                    indicator_type=indicator_type,
                    importance_score=mean_importance,
                    analysis_method=AnalysisMethod.PERMUTATION,
                    confidence_interval=(ci_lower, ci_upper),
                    additional_metrics={
                        "permutation_importance_mean": mean_importance,
                        "permutation_importance_std": std_importance,
                        "n_repeats": n_repeats,
                    },
                )

                results.append(importance)

        except Exception as e:
            logger.error(f"排列重要性分析失败: {e}")

        return sorted(results, key=lambda x: x.importance_score, reverse=True)

    def analyze_statistical_significance(
        self,
        indicators_data: pd.DataFrame,
        target: pd.Series,
        test_type: str = "f_test",
    ) -> List[IndicatorImportance]:
        """统计显著性分析"""
        results = []

        # 处理缺失值
        clean_data = indicators_data.dropna()
        clean_target = target[clean_data.index]

        try:
            if test_type == "f_test":
                f_scores, p_values = f_regression(clean_data, clean_target)

                for i, column in enumerate(clean_data.columns):
                    indicator_type = self._infer_indicator_type(column)

                    importance = IndicatorImportance(
                        indicator_name=column,
                        indicator_type=indicator_type,
                        importance_score=f_scores[i],
                        analysis_method=AnalysisMethod.STATISTICAL_TEST,
                        p_value=p_values[i],
                        additional_metrics={
                            "f_score": f_scores[i],
                            "test_type": test_type,
                        },
                    )

                    results.append(importance)
            else:
                raise ValueError(f"不支持的统计检验类型: {test_type}")

        except Exception as e:
            logger.error(f"统计显著性分析失败: {e}")

        return sorted(results, key=lambda x: x.importance_score, reverse=True)

    def _infer_indicator_type(self, column_name: str) -> TechnicalIndicator:
        """推断指标类型"""
        column_lower = column_name.lower()

        if "rsi" in column_lower:
            return TechnicalIndicator.RSI
        elif "macd" in column_lower:
            return TechnicalIndicator.MACD
        elif "bollinger" in column_lower or "bb" in column_lower:
            return TechnicalIndicator.BOLLINGER_BANDS
        elif "ma" in column_lower or "sma" in column_lower or "ema" in column_lower:
            return TechnicalIndicator.MOVING_AVERAGE
        elif "stoch" in column_lower or "k" in column_lower or "d" in column_lower:
            return TechnicalIndicator.STOCHASTIC
        elif "williams" in column_lower or "wr" in column_lower:
            return TechnicalIndicator.WILLIAMS_R
        elif "cci" in column_lower:
            return TechnicalIndicator.CCI
        elif "atr" in column_lower:
            return TechnicalIndicator.ATR
        elif "volume" in column_lower or "vol" in column_lower:
            return TechnicalIndicator.VOLUME
        elif "momentum" in column_lower or "mom" in column_lower:
            return TechnicalIndicator.MOMENTUM
        else:
            return TechnicalIndicator.MOMENTUM  # 默认类型


class TechnicalAnalyzer:
    """技术指标分析器主类"""

    def __init__(self, storage_path: str = "data/technical_analysis"):
        """
        初始化技术分析器

        Args:
            storage_path: 存储路径
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.calculator = TechnicalIndicatorCalculator()
        self.impact_analyzer = IndicatorImpactAnalyzer()

        # 分析结果存储
        self.analysis_results: Dict[str, IndicatorAnalysisResult] = {}
        self.max_results = 1000

        # 线程锁
        self.lock = threading.Lock()

        logger.info(f"技术指标分析器初始化完成，存储路径: {self.storage_path}")

    def calculate_all_indicators(
        self, price_data: pd.DataFrame, volume_data: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        计算所有技术指标

        Args:
            price_data: 价格数据（包含open, high, low, close列）
            volume_data: 成交量数据

        Returns:
            包含所有指标的DataFrame
        """
        indicators = pd.DataFrame(index=price_data.index)

        try:
            # 基础价格数据
            close = price_data["close"]
            high = price_data["high"]
            low = price_data["low"]

            # RSI指标
            indicators["rsi_14"] = self.calculator.calculate_rsi(close, 14)
            indicators["rsi_21"] = self.calculator.calculate_rsi(close, 21)

            # MACD指标
            macd_data = self.calculator.calculate_macd(close)
            indicators["macd"] = macd_data["macd"]
            indicators["macd_signal"] = macd_data["signal"]
            indicators["macd_histogram"] = macd_data["histogram"]

            # 布林带
            bb_data = self.calculator.calculate_bollinger_bands(close)
            indicators["bb_upper"] = bb_data["upper"]
            indicators["bb_middle"] = bb_data["middle"]
            indicators["bb_lower"] = bb_data["lower"]
            indicators["bb_width"] = bb_data["width"]
            indicators["bb_position"] = bb_data["position"]

            # 移动平均线
            indicators["sma_5"] = self.calculator.calculate_moving_average(
                close, 5, "sma"
            )
            indicators["sma_10"] = self.calculator.calculate_moving_average(
                close, 10, "sma"
            )
            indicators["sma_20"] = self.calculator.calculate_moving_average(
                close, 20, "sma"
            )
            indicators["sma_50"] = self.calculator.calculate_moving_average(
                close, 50, "sma"
            )
            indicators["ema_12"] = self.calculator.calculate_moving_average(
                close, 12, "ema"
            )
            indicators["ema_26"] = self.calculator.calculate_moving_average(
                close, 26, "ema"
            )

            # 随机指标
            stoch_data = self.calculator.calculate_stochastic(high, low, close)
            indicators["stoch_k"] = stoch_data["k"]
            indicators["stoch_d"] = stoch_data["d"]

            # 威廉指标
            indicators["williams_r"] = self.calculator.calculate_williams_r(
                high, low, close
            )

            # CCI指标
            indicators["cci"] = self.calculator.calculate_cci(high, low, close)

            # ATR指标
            indicators["atr"] = self.calculator.calculate_atr(high, low, close)

            # 价格变化率
            indicators["price_change_1d"] = close.pct_change(1)
            indicators["price_change_5d"] = close.pct_change(5)
            indicators["price_change_10d"] = close.pct_change(10)

            # 动量指标
            indicators["momentum_10"] = close / close.shift(10) - 1
            indicators["momentum_20"] = close / close.shift(20) - 1

            # 成交量指标（如果提供）
            if volume_data is not None:
                indicators["volume"] = volume_data
                indicators["volume_sma_20"] = volume_data.rolling(20).mean()
                indicators["volume_ratio"] = volume_data / indicators["volume_sma_20"]

            logger.info(f"计算了 {len(indicators.columns)} 个技术指标")

        except Exception as e:
            logger.error(f"计算技术指标失败: {e}")
            raise

        return indicators

    def analyze_indicator_impact(
        self,
        model_id: str,
        model_version: str,
        indicators_data: pd.DataFrame,
        target: pd.Series,
        analysis_methods: Optional[List[AnalysisMethod]] = None,
        target_variable: str = "return",
    ) -> str:
        """
        分析指标影响

        Args:
            model_id: 模型ID
            model_version: 模型版本
            indicators_data: 指标数据
            target: 目标变量
            analysis_methods: 分析方法列表
            target_variable: 目标变量名称

        Returns:
            分析ID
        """
        if analysis_methods is None:
            analysis_methods = [
                AnalysisMethod.CORRELATION,
                AnalysisMethod.FEATURE_IMPORTANCE,
                AnalysisMethod.MUTUAL_INFO,
            ]

        analysis_id = f"tech_analysis_{int(datetime.now().timestamp())}_{model_id}"

        try:
            all_importance_results = []

            # 执行各种分析方法
            for method in analysis_methods:
                if method == AnalysisMethod.CORRELATION:
                    results = self.impact_analyzer.analyze_correlation(
                        indicators_data, target
                    )
                elif method == AnalysisMethod.MUTUAL_INFO:
                    results = self.impact_analyzer.analyze_mutual_information(
                        indicators_data, target
                    )
                elif method == AnalysisMethod.FEATURE_IMPORTANCE:
                    results = self.impact_analyzer.analyze_feature_importance(
                        indicators_data, target
                    )
                elif method == AnalysisMethod.PERMUTATION:
                    results = self.impact_analyzer.analyze_permutation_importance(
                        indicators_data, target
                    )
                elif method == AnalysisMethod.STATISTICAL_TEST:
                    results = self.impact_analyzer.analyze_statistical_significance(
                        indicators_data, target
                    )
                else:
                    logger.warning(f"不支持的分析方法: {method}")
                    continue

                all_importance_results.extend(results)

            # 聚合结果
            aggregated_importance = self._aggregate_importance_results(
                all_importance_results
            )

            # 生成摘要统计
            summary_stats = self._generate_summary_stats(
                aggregated_importance, indicators_data, target
            )

            # 生成建议
            recommendations = self._generate_recommendations(aggregated_importance)

            # 创建分析结果
            analysis_result = IndicatorAnalysisResult(
                analysis_id=analysis_id,
                model_id=model_id,
                model_version=model_version,
                target_variable=target_variable,
                analysis_period=f"{indicators_data.index.min()} to {indicators_data.index.max()}",
                indicator_importance=aggregated_importance,
                summary_stats=summary_stats,
                recommendations=recommendations,
                created_at=datetime.now(),
                metadata={
                    "num_indicators": len(indicators_data.columns),
                    "num_samples": len(indicators_data),
                    "analysis_methods": [m.value for m in analysis_methods],
                },
            )

            # 存储结果
            with self.lock:
                self.analysis_results[analysis_id] = analysis_result

                # 限制存储数量
                if len(self.analysis_results) > self.max_results:
                    oldest_id = min(
                        self.analysis_results.keys(),
                        key=lambda x: self.analysis_results[x].created_at,
                    )
                    del self.analysis_results[oldest_id]

            # 保存结果
            self._save_analysis_result(analysis_result)

            logger.info(f"完成技术指标影响分析: {analysis_id}")
            return analysis_id

        except Exception as e:
            logger.error(f"技术指标影响分析失败: {e}")
            raise

    def _aggregate_importance_results(
        self, importance_results: List[IndicatorImportance]
    ) -> List[IndicatorImportance]:
        """聚合重要性结果"""
        # 按指标名称分组
        indicator_groups = {}
        for importance in importance_results:
            name = importance.indicator_name
            if name not in indicator_groups:
                indicator_groups[name] = []
            indicator_groups[name].append(importance)

        # 计算每个指标的平均重要性
        aggregated_results = []
        for name, group in indicator_groups.items():
            if len(group) == 1:
                aggregated_results.append(group[0])
            else:
                # 计算平均重要性分数
                avg_score = np.mean([imp.importance_score for imp in group])

                # 使用第一个结果作为模板
                template = group[0]

                aggregated_importance = IndicatorImportance(
                    indicator_name=name,
                    indicator_type=template.indicator_type,
                    importance_score=avg_score,
                    analysis_method=AnalysisMethod.FEATURE_IMPORTANCE,  # 聚合方法
                    additional_metrics={
                        "aggregated_from": [imp.analysis_method.value for imp in group],
                        "individual_scores": [imp.importance_score for imp in group],
                        "score_std": np.std([imp.importance_score for imp in group]),
                    },
                )

                aggregated_results.append(aggregated_importance)

        return sorted(
            aggregated_results, key=lambda x: x.importance_score, reverse=True
        )

    def _generate_summary_stats(
        self,
        importance_results: List[IndicatorImportance],
        indicators_data: pd.DataFrame,
        target: pd.Series,
    ) -> Dict[str, Any]:
        """生成摘要统计"""
        scores = [imp.importance_score for imp in importance_results]

        # 按指标类型分组统计
        type_stats = {}
        for imp in importance_results:
            indicator_type = imp.indicator_type.value
            if indicator_type not in type_stats:
                type_stats[indicator_type] = []
            type_stats[indicator_type].append(imp.importance_score)

        type_avg_scores = {k: np.mean(v) for k, v in type_stats.items()}

        return {
            "total_indicators": len(importance_results),
            "mean_importance": np.mean(scores),
            "std_importance": np.std(scores),
            "max_importance": np.max(scores),
            "min_importance": np.min(scores),
            "top_5_indicators": [imp.indicator_name for imp in importance_results[:5]],
            "indicator_type_avg_scores": type_avg_scores,
            "data_period": {
                "start": str(indicators_data.index.min()),
                "end": str(indicators_data.index.max()),
                "num_samples": len(indicators_data),
            },
        }

    def _generate_recommendations(
        self, importance_results: List[IndicatorImportance]
    ) -> List[str]:
        """生成建议"""
        recommendations = []

        if not importance_results:
            return ["无法生成建议：没有有效的指标分析结果"]

        # 获取前5个最重要的指标
        top_indicators = importance_results[:5]
        top_names = [imp.indicator_name for imp in top_indicators]

        recommendations.append(f"重点关注以下技术指标: {', '.join(top_names)}")

        # 按指标类型分析
        type_counts = {}
        for imp in top_indicators:
            indicator_type = imp.indicator_type.value
            type_counts[indicator_type] = type_counts.get(indicator_type, 0) + 1

        most_important_type = max(type_counts, key=type_counts.get)
        recommendations.append(f"最具影响力的指标类型是: {most_important_type}")

        # 根据重要性分数给出建议
        max_score = importance_results[0].importance_score
        if max_score > 0.5:
            recommendations.append("存在高影响力指标，建议在模型中重点使用")
        elif max_score > 0.3:
            recommendations.append("指标影响力中等，建议结合多个指标使用")
        else:
            recommendations.append("指标影响力较低，建议考虑其他特征或指标组合")

        # 低重要性指标建议
        low_importance_indicators = [
            imp.indicator_name
            for imp in importance_results[-5:]
            if imp.importance_score < 0.1
        ]
        if low_importance_indicators:
            recommendations.append(
                f"以下指标影响力较低，可考虑移除: {', '.join(low_importance_indicators)}"
            )

        return recommendations

    def get_analysis_result(
        self, analysis_id: str
    ) -> Optional[IndicatorAnalysisResult]:
        """获取分析结果"""
        return self.analysis_results.get(analysis_id)

    def get_model_analyses(
        self, model_id: str, model_version: Optional[str] = None, limit: int = 10
    ) -> List[IndicatorAnalysisResult]:
        """获取模型的分析结果"""
        results = []

        for analysis in self.analysis_results.values():
            if analysis.model_id != model_id:
                continue

            if model_version and analysis.model_version != model_version:
                continue

            results.append(analysis)

        # 按创建时间排序
        results.sort(key=lambda x: x.created_at, reverse=True)

        return results[:limit]

    def get_indicator_ranking(
        self, model_id: str, model_version: Optional[str] = None, top_k: int = 20
    ) -> List[Tuple[str, float]]:
        """获取指标重要性排名"""
        analyses = self.get_model_analyses(model_id, model_version)

        if not analyses:
            return []

        # 聚合所有分析结果
        indicator_scores = {}
        indicator_counts = {}

        for analysis in analyses:
            for importance in analysis.indicator_importance:
                name = importance.indicator_name
                score = importance.importance_score

                if name not in indicator_scores:
                    indicator_scores[name] = 0
                    indicator_counts[name] = 0

                indicator_scores[name] += score
                indicator_counts[name] += 1

        # 计算平均分数
        avg_scores = {}
        for name in indicator_scores:
            avg_scores[name] = indicator_scores[name] / indicator_counts[name]

        # 排序并返回前K个
        sorted_indicators = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_indicators[:top_k]

    def _save_analysis_result(self, analysis_result: IndicatorAnalysisResult):
        """保存分析结果"""
        try:
            result_file = self.storage_path / f"{analysis_result.analysis_id}.json"

            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(analysis_result.to_dict(), f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(f"保存分析结果失败: {e}")


# 全局技术指标分析器实例
technical_analyzer = TechnicalAnalyzer()
