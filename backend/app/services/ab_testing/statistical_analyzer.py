"""
统计显著性分析器
实现A/B测试结果分析，提供统计显著性检验
"""
import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu, ttest_ind
import math

from app.services.ab_testing.metrics_collector import MetricValue, MetricType, BusinessMetricsCollector
from app.services.ab_testing.traffic_manager import ABExperiment, TrafficVariant, VariantType

logger = logging.getLogger(__name__)

class TestType(Enum):
    """统计检验类型"""
    T_TEST = "t_test"  # t检验
    WELCH_T_TEST = "welch_t_test"  # Welch t检验
    MANN_WHITNEY = "mann_whitney"  # Mann-Whitney U检验
    CHI_SQUARE = "chi_square"  # 卡方检验
    PROPORTION_Z_TEST = "proportion_z_test"  # 比例z检验
    FISHER_EXACT = "fisher_exact"  # Fisher精确检验

class EffectSize(Enum):
    """效应量类型"""
    COHENS_D = "cohens_d"  # Cohen's d
    GLASS_DELTA = "glass_delta"  # Glass's Δ
    HEDGES_G = "hedges_g"  # Hedges' g
    CRAMERS_V = "cramers_v"  # Cramér's V
    ODDS_RATIO = "odds_ratio"  # 比值比

@dataclass
class StatisticalTestResult:
    """统计检验结果"""
    test_type: TestType
    statistic: float
    p_value: float
    degrees_of_freedom: Optional[int]
    confidence_level: float
    confidence_interval: Optional[Tuple[float, float]]
    effect_size: Optional[float]
    effect_size_type: Optional[EffectSize]
    power: Optional[float]
    sample_size_control: int
    sample_size_treatment: int
    is_significant: bool
    interpretation: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'test_type': self.test_type.value,
            'statistic': self.statistic,
            'p_value': self.p_value,
            'degrees_of_freedom': self.degrees_of_freedom,
            'confidence_level': self.confidence_level,
            'confidence_interval': list(self.confidence_interval) if self.confidence_interval else None,
            'effect_size': self.effect_size,
            'effect_size_type': self.effect_size_type.value if self.effect_size_type else None,
            'power': self.power,
            'sample_size_control': self.sample_size_control,
            'sample_size_treatment': self.sample_size_treatment,
            'is_significant': self.is_significant,
            'interpretation': self.interpretation
        }

@dataclass
class ExperimentAnalysisResult:
    """实验分析结果"""
    experiment_id: str
    experiment_name: str
    analysis_timestamp: datetime
    # 基本信息
    total_users: int
    experiment_duration_days: float
    # 变体对比结果
    variant_comparisons: List[Dict[str, Any]]
    # 总体结论
    overall_winner: Optional[str]
    overall_confidence: float
    recommendation: str
    # 详细分析
    metric_analyses: Dict[str, Dict[str, Any]]
    # 统计功效分析
    power_analysis: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'experiment_id': self.experiment_id,
            'experiment_name': self.experiment_name,
            'analysis_timestamp': self.analysis_timestamp.isoformat(),
            'total_users': self.total_users,
            'experiment_duration_days': self.experiment_duration_days,
            'variant_comparisons': self.variant_comparisons,
            'overall_winner': self.overall_winner,
            'overall_confidence': self.overall_confidence,
            'recommendation': self.recommendation,
            'metric_analyses': self.metric_analyses,
            'power_analysis': self.power_analysis
        }

class StatisticalTestEngine:
    """统计检验引擎"""
    
    def __init__(self):
        self.alpha = 0.05  # 显著性水平
        self.power_threshold = 0.8  # 统计功效阈值
    
    def t_test(
        self, 
        control_data: np.ndarray, 
        treatment_data: np.ndarray,
        equal_var: bool = True,
        confidence_level: float = 0.95
    ) -> StatisticalTestResult:
        """t检验"""
        if equal_var:
            statistic, p_value = ttest_ind(control_data, treatment_data, equal_var=True)
            test_type = TestType.T_TEST
            df = len(control_data) + len(treatment_data) - 2
        else:
            statistic, p_value = ttest_ind(control_data, treatment_data, equal_var=False)
            test_type = TestType.WELCH_T_TEST
            # Welch-Satterthwaite方程计算自由度
            s1_sq = np.var(control_data, ddof=1)
            s2_sq = np.var(treatment_data, ddof=1)
            n1, n2 = len(control_data), len(treatment_data)
            df = (s1_sq/n1 + s2_sq/n2)**2 / ((s1_sq/n1)**2/(n1-1) + (s2_sq/n2)**2/(n2-1))
        
        # 计算置信区间
        alpha = 1 - confidence_level
        t_critical = stats.t.ppf(1 - alpha/2, df)
        
        mean_diff = np.mean(treatment_data) - np.mean(control_data)
        pooled_se = np.sqrt(np.var(control_data, ddof=1)/len(control_data) + 
                           np.var(treatment_data, ddof=1)/len(treatment_data))
        
        ci_lower = mean_diff - t_critical * pooled_se
        ci_upper = mean_diff + t_critical * pooled_se
        
        # 计算Cohen's d
        pooled_std = np.sqrt(((len(control_data)-1)*np.var(control_data, ddof=1) + 
                             (len(treatment_data)-1)*np.var(treatment_data, ddof=1)) / 
                            (len(control_data) + len(treatment_data) - 2))
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
        
        # 计算统计功效
        power = self._calculate_power_t_test(len(control_data), len(treatment_data), cohens_d, self.alpha)
        
        is_significant = p_value < self.alpha
        interpretation = self._interpret_t_test_result(statistic, p_value, cohens_d, is_significant)
        
        return StatisticalTestResult(
            test_type=test_type,
            statistic=float(statistic),
            p_value=float(p_value),
            degrees_of_freedom=int(df),
            confidence_level=confidence_level,
            confidence_interval=(float(ci_lower), float(ci_upper)),
            effect_size=float(cohens_d),
            effect_size_type=EffectSize.COHENS_D,
            power=float(power),
            sample_size_control=len(control_data),
            sample_size_treatment=len(treatment_data),
            is_significant=is_significant,
            interpretation=interpretation
        )
    
    def mann_whitney_test(
        self, 
        control_data: np.ndarray, 
        treatment_data: np.ndarray,
        confidence_level: float = 0.95
    ) -> StatisticalTestResult:
        """Mann-Whitney U检验（非参数检验）"""
        statistic, p_value = mannwhitneyu(control_data, treatment_data, alternative='two-sided')
        
        # 计算效应量（rank-biserial correlation）
        n1, n2 = len(control_data), len(treatment_data)
        u1 = statistic
        u2 = n1 * n2 - u1
        effect_size = 1 - (2 * min(u1, u2)) / (n1 * n2)
        
        is_significant = p_value < self.alpha
        interpretation = self._interpret_mann_whitney_result(statistic, p_value, effect_size, is_significant)
        
        return StatisticalTestResult(
            test_type=TestType.MANN_WHITNEY,
            statistic=float(statistic),
            p_value=float(p_value),
            degrees_of_freedom=None,
            confidence_level=confidence_level,
            confidence_interval=None,
            effect_size=float(effect_size),
            effect_size_type=None,
            power=None,
            sample_size_control=len(control_data),
            sample_size_treatment=len(treatment_data),
            is_significant=is_significant,
            interpretation=interpretation
        )
    
    def proportion_z_test(
        self,
        control_successes: int,
        control_total: int,
        treatment_successes: int,
        treatment_total: int,
        confidence_level: float = 0.95
    ) -> StatisticalTestResult:
        """比例z检验"""
        p1 = control_successes / control_total
        p2 = treatment_successes / treatment_total
        
        # 合并比例
        p_pooled = (control_successes + treatment_successes) / (control_total + treatment_total)
        
        # 标准误差
        se = np.sqrt(p_pooled * (1 - p_pooled) * (1/control_total + 1/treatment_total))
        
        # z统计量
        z_statistic = (p2 - p1) / se if se > 0 else 0
        
        # p值（双尾检验）
        p_value = 2 * (1 - stats.norm.cdf(abs(z_statistic)))
        
        # 置信区间
        alpha = 1 - confidence_level
        z_critical = stats.norm.ppf(1 - alpha/2)
        
        diff = p2 - p1
        se_diff = np.sqrt(p1*(1-p1)/control_total + p2*(1-p2)/treatment_total)
        
        ci_lower = diff - z_critical * se_diff
        ci_upper = diff + z_critical * se_diff
        
        # 效应量（Cohen's h）
        cohens_h = 2 * (np.arcsin(np.sqrt(p2)) - np.arcsin(np.sqrt(p1)))
        
        # 统计功效
        power = self._calculate_power_proportion_test(control_total, treatment_total, p1, p2, self.alpha)
        
        is_significant = p_value < self.alpha
        interpretation = self._interpret_proportion_test_result(z_statistic, p_value, cohens_h, is_significant)
        
        return StatisticalTestResult(
            test_type=TestType.PROPORTION_Z_TEST,
            statistic=float(z_statistic),
            p_value=float(p_value),
            degrees_of_freedom=None,
            confidence_level=confidence_level,
            confidence_interval=(float(ci_lower), float(ci_upper)),
            effect_size=float(cohens_h),
            effect_size_type=EffectSize.COHENS_D,  # 使用Cohen's h，但归类为Cohen's d
            power=float(power),
            sample_size_control=control_total,
            sample_size_treatment=treatment_total,
            is_significant=is_significant,
            interpretation=interpretation
        )
    
    def chi_square_test(
        self,
        contingency_table: np.ndarray,
        confidence_level: float = 0.95
    ) -> StatisticalTestResult:
        """卡方检验"""
        chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
        
        # 计算Cramér's V
        n = np.sum(contingency_table)
        cramers_v = np.sqrt(chi2_stat / (n * (min(contingency_table.shape) - 1)))
        
        is_significant = p_value < self.alpha
        interpretation = self._interpret_chi_square_result(chi2_stat, p_value, cramers_v, is_significant)
        
        return StatisticalTestResult(
            test_type=TestType.CHI_SQUARE,
            statistic=float(chi2_stat),
            p_value=float(p_value),
            degrees_of_freedom=int(dof),
            confidence_level=confidence_level,
            confidence_interval=None,
            effect_size=float(cramers_v),
            effect_size_type=EffectSize.CRAMERS_V,
            power=None,
            sample_size_control=int(np.sum(contingency_table[0, :])),
            sample_size_treatment=int(np.sum(contingency_table[1, :])),
            is_significant=is_significant,
            interpretation=interpretation
        )
    
    def _calculate_power_t_test(self, n1: int, n2: int, effect_size: float, alpha: float) -> float:
        """计算t检验的统计功效"""
        try:
            from scipy.stats import nct
            
            # 非中心参数
            delta = effect_size * np.sqrt(n1 * n2 / (n1 + n2))
            df = n1 + n2 - 2
            
            # 临界值
            t_critical = stats.t.ppf(1 - alpha/2, df)
            
            # 功效计算
            power = 1 - nct.cdf(t_critical, df, delta) + nct.cdf(-t_critical, df, delta)
            return min(max(power, 0), 1)
        except:
            return 0.5  # 默认值
    
    def _calculate_power_proportion_test(self, n1: int, n2: int, p1: float, p2: float, alpha: float) -> float:
        """计算比例检验的统计功效"""
        try:
            # 合并比例
            p_pooled = (n1 * p1 + n2 * p2) / (n1 + n2)
            
            # 标准误差
            se_null = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
            se_alt = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
            
            # 临界值
            z_critical = stats.norm.ppf(1 - alpha/2)
            
            # 功效计算
            z_beta = (abs(p2 - p1) - z_critical * se_null) / se_alt
            power = stats.norm.cdf(z_beta)
            
            return min(max(power, 0), 1)
        except:
            return 0.5  # 默认值
    
    def _interpret_t_test_result(self, statistic: float, p_value: float, effect_size: float, is_significant: bool) -> str:
        """解释t检验结果"""
        if is_significant:
            if abs(effect_size) < 0.2:
                effect_desc = "小"
            elif abs(effect_size) < 0.5:
                effect_desc = "中等"
            elif abs(effect_size) < 0.8:
                effect_desc = "大"
            else:
                effect_desc = "非常大"
            
            direction = "显著高于" if statistic > 0 else "显著低于"
            return f"实验组{direction}对照组，效应量为{effect_desc}（Cohen's d = {effect_size:.3f}）"
        else:
            return f"实验组与对照组无显著差异（p = {p_value:.3f}）"
    
    def _interpret_mann_whitney_result(self, statistic: float, p_value: float, effect_size: float, is_significant: bool) -> str:
        """解释Mann-Whitney检验结果"""
        if is_significant:
            return f"实验组与对照组存在显著差异，效应量 = {effect_size:.3f}"
        else:
            return f"实验组与对照组无显著差异（p = {p_value:.3f}）"
    
    def _interpret_proportion_test_result(self, statistic: float, p_value: float, effect_size: float, is_significant: bool) -> str:
        """解释比例检验结果"""
        if is_significant:
            direction = "显著高于" if statistic > 0 else "显著低于"
            return f"实验组转化率{direction}对照组（Cohen's h = {effect_size:.3f}）"
        else:
            return f"实验组与对照组转化率无显著差异（p = {p_value:.3f}）"
    
    def _interpret_chi_square_result(self, statistic: float, p_value: float, effect_size: float, is_significant: bool) -> str:
        """解释卡方检验结果"""
        if is_significant:
            return f"变量间存在显著关联，关联强度为{effect_size:.3f}（Cramér's V）"
        else:
            return f"变量间无显著关联（p = {p_value:.3f}）"

class ExperimentAnalyzer:
    """实验分析器"""
    
    def __init__(self, metrics_collector: BusinessMetricsCollector):
        self.metrics_collector = metrics_collector
        self.test_engine = StatisticalTestEngine()
        
        logger.info("实验分析器初始化完成")
    
    def analyze_experiment(
        self, 
        experiment: ABExperiment,
        metric_ids: Optional[List[str]] = None,
        confidence_level: float = 0.95
    ) -> ExperimentAnalysisResult:
        """分析A/B测试实验"""
        logger.info(f"开始分析实验: {experiment.name} ({experiment.experiment_id})")
        
        # 获取实验持续时间
        duration_days = 0.0
        if experiment.start_time and experiment.end_time:
            duration_days = (experiment.end_time - experiment.start_time).total_seconds() / 86400
        elif experiment.start_time:
            duration_days = (datetime.now() - experiment.start_time).total_seconds() / 86400
        
        # 确定要分析的指标
        if metric_ids is None:
            metric_ids = [m.metric_id for m in self.metrics_collector.list_metric_definitions()]
        
        # 获取对照组和实验组
        control_variants = [v for v in experiment.variants if v.variant_type == VariantType.CONTROL]
        treatment_variants = [v for v in experiment.variants if v.variant_type == VariantType.TREATMENT]
        
        if not control_variants:
            raise ValueError("实验必须包含至少一个对照组")
        
        control_variant = control_variants[0]  # 使用第一个对照组
        
        # 分析结果
        variant_comparisons = []
        metric_analyses = {}
        overall_winners = []
        confidence_scores = []
        
        # 对每个实验组进行分析
        for treatment_variant in treatment_variants:
            comparison_result = self._compare_variants(
                experiment, control_variant, treatment_variant, metric_ids, confidence_level
            )
            variant_comparisons.append(comparison_result)
            
            # 收集获胜者信息
            for metric_id, analysis in comparison_result['metric_results'].items():
                if analysis['is_significant'] and analysis['winner'] == treatment_variant.variant_id:
                    overall_winners.append(treatment_variant.variant_id)
                    confidence_scores.append(1 - analysis['p_value'])
        
        # 分析每个指标
        for metric_id in metric_ids:
            metric_analyses[metric_id] = self._analyze_metric_across_variants(
                experiment, metric_id, confidence_level
            )
        
        # 确定总体获胜者
        overall_winner = None
        overall_confidence = 0.0
        
        if overall_winners:
            # 选择获胜次数最多的变体
            from collections import Counter
            winner_counts = Counter(overall_winners)
            overall_winner = winner_counts.most_common(1)[0][0]
            overall_confidence = np.mean(confidence_scores)
        
        # 生成建议
        recommendation = self._generate_recommendation(
            experiment, variant_comparisons, overall_winner, overall_confidence
        )
        
        # 统计功效分析
        power_analysis = self._perform_power_analysis(experiment, metric_analyses)
        
        return ExperimentAnalysisResult(
            experiment_id=experiment.experiment_id,
            experiment_name=experiment.name,
            analysis_timestamp=datetime.now(),
            total_users=self._get_total_users(experiment),
            experiment_duration_days=duration_days,
            variant_comparisons=variant_comparisons,
            overall_winner=overall_winner,
            overall_confidence=overall_confidence,
            recommendation=recommendation,
            metric_analyses=metric_analyses,
            power_analysis=power_analysis
        )
    
    def _compare_variants(
        self,
        experiment: ABExperiment,
        control_variant: TrafficVariant,
        treatment_variant: TrafficVariant,
        metric_ids: List[str],
        confidence_level: float
    ) -> Dict[str, Any]:
        """比较两个变体"""
        comparison = {
            'control_variant': control_variant.to_dict(),
            'treatment_variant': treatment_variant.to_dict(),
            'metric_results': {}
        }
        
        for metric_id in metric_ids:
            # 获取指标值
            control_metric = self.metrics_collector.calculate_metric(
                metric_id, experiment.experiment_id, control_variant.variant_id
            )
            treatment_metric = self.metrics_collector.calculate_metric(
                metric_id, experiment.experiment_id, treatment_variant.variant_id
            )
            
            if not control_metric or not treatment_metric:
                continue
            
            # 执行统计检验
            test_result = self._perform_statistical_test(
                control_metric, treatment_metric, confidence_level
            )
            
            # 确定获胜者
            winner = None
            if test_result.is_significant:
                if test_result.statistic > 0:
                    winner = treatment_variant.variant_id
                else:
                    winner = control_variant.variant_id
            
            comparison['metric_results'][metric_id] = {
                'control_value': control_metric.value,
                'treatment_value': treatment_metric.value,
                'relative_change': ((treatment_metric.value - control_metric.value) / control_metric.value * 100) if control_metric.value != 0 else 0,
                'test_result': test_result.to_dict(),
                'winner': winner,
                'is_significant': test_result.is_significant,
                'p_value': test_result.p_value,
                'effect_size': test_result.effect_size
            }
        
        return comparison
    
    def _perform_statistical_test(
        self,
        control_metric: MetricValue,
        treatment_metric: MetricValue,
        confidence_level: float
    ) -> StatisticalTestResult:
        """执行统计检验"""
        # 根据指标类型选择合适的检验方法
        metric_def = self.metrics_collector.get_metric_definition(control_metric.metric_id)
        
        if not metric_def:
            # 默认使用t检验
            return self._mock_t_test_result(control_metric, treatment_metric, confidence_level)
        
        if metric_def.metric_type == MetricType.CONVERSION:
            # 转化率使用比例检验
            control_successes = int(control_metric.value * control_metric.sample_size)
            treatment_successes = int(treatment_metric.value * treatment_metric.sample_size)
            
            return self.test_engine.proportion_z_test(
                control_successes, control_metric.sample_size,
                treatment_successes, treatment_metric.sample_size,
                confidence_level
            )
        
        elif metric_def.metric_type in [MetricType.HISTOGRAM, MetricType.DURATION, MetricType.REVENUE]:
            # 连续变量使用t检验（这里简化处理，实际应该获取原始数据）
            return self._mock_t_test_result(control_metric, treatment_metric, confidence_level)
        
        else:
            # 其他类型使用t检验
            return self._mock_t_test_result(control_metric, treatment_metric, confidence_level)
    
    def _mock_t_test_result(
        self,
        control_metric: MetricValue,
        treatment_metric: MetricValue,
        confidence_level: float
    ) -> StatisticalTestResult:
        """模拟t检验结果（当无法获取原始数据时）"""
        # 基于指标值和样本量模拟统计检验
        mean_diff = treatment_metric.value - control_metric.value
        
        # 估算标准误差
        pooled_se = abs(mean_diff) / max(np.sqrt(control_metric.sample_size + treatment_metric.sample_size), 1)
        
        # 计算t统计量
        t_statistic = mean_diff / pooled_se if pooled_se > 0 else 0
        
        # 自由度
        df = control_metric.sample_size + treatment_metric.sample_size - 2
        
        # p值
        p_value = 2 * (1 - stats.t.cdf(abs(t_statistic), df))
        
        # 效应量
        pooled_std = pooled_se * np.sqrt(control_metric.sample_size + treatment_metric.sample_size)
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
        
        # 置信区间
        alpha = 1 - confidence_level
        t_critical = stats.t.ppf(1 - alpha/2, df)
        ci_lower = mean_diff - t_critical * pooled_se
        ci_upper = mean_diff + t_critical * pooled_se
        
        is_significant = p_value < 0.05
        interpretation = f"{'显著' if is_significant else '不显著'}差异 (p = {p_value:.3f})"
        
        return StatisticalTestResult(
            test_type=TestType.T_TEST,
            statistic=float(t_statistic),
            p_value=float(p_value),
            degrees_of_freedom=int(df),
            confidence_level=confidence_level,
            confidence_interval=(float(ci_lower), float(ci_upper)),
            effect_size=float(cohens_d),
            effect_size_type=EffectSize.COHENS_D,
            power=0.8,  # 假设值
            sample_size_control=control_metric.sample_size,
            sample_size_treatment=treatment_metric.sample_size,
            is_significant=is_significant,
            interpretation=interpretation
        )
    
    def _analyze_metric_across_variants(
        self,
        experiment: ABExperiment,
        metric_id: str,
        confidence_level: float
    ) -> Dict[str, Any]:
        """分析指标在所有变体中的表现"""
        variant_values = {}
        
        for variant in experiment.variants:
            metric_value = self.metrics_collector.calculate_metric(
                metric_id, experiment.experiment_id, variant.variant_id
            )
            if metric_value:
                variant_values[variant.variant_id] = {
                    'variant_name': variant.name,
                    'value': metric_value.value,
                    'sample_size': metric_value.sample_size,
                    'confidence_interval': metric_value.confidence_interval
                }
        
        # 找出最佳表现的变体
        best_variant = None
        best_value = None
        
        for variant_id, data in variant_values.items():
            if best_value is None or data['value'] > best_value:
                best_variant = variant_id
                best_value = data['value']
        
        return {
            'metric_id': metric_id,
            'variant_values': variant_values,
            'best_variant': best_variant,
            'best_value': best_value
        }
    
    def _generate_recommendation(
        self,
        experiment: ABExperiment,
        variant_comparisons: List[Dict[str, Any]],
        overall_winner: Optional[str],
        overall_confidence: float
    ) -> str:
        """生成实验建议"""
        if not overall_winner:
            return "实验结果无显著差异，建议继续收集数据或重新设计实验"
        
        if overall_confidence < 0.8:
            return f"实验组 {overall_winner} 表现较好，但置信度较低（{overall_confidence:.2f}），建议延长实验时间"
        
        elif overall_confidence < 0.95:
            return f"实验组 {overall_winner} 表现显著优于对照组，建议谨慎推广"
        
        else:
            return f"实验组 {overall_winner} 表现显著优于对照组，建议全量推广"
    
    def _perform_power_analysis(
        self,
        experiment: ABExperiment,
        metric_analyses: Dict[str, Any]
    ) -> Dict[str, Any]:
        """执行统计功效分析"""
        total_sample_size = self._get_total_users(experiment)
        
        # 计算每个变体的样本量
        variant_sample_sizes = {}
        for variant in experiment.variants:
            expected_size = int(total_sample_size * variant.traffic_percentage / 100)
            variant_sample_sizes[variant.variant_id] = expected_size
        
        return {
            'total_sample_size': total_sample_size,
            'variant_sample_sizes': variant_sample_sizes,
            'minimum_detectable_effect': 0.05,  # 5%的最小可检测效应
            'statistical_power': 0.8,  # 期望的统计功效
            'significance_level': 0.05,  # 显著性水平
            'recommendation': "样本量充足" if total_sample_size > 1000 else "建议增加样本量"
        }
    
    def _get_total_users(self, experiment: ABExperiment) -> int:
        """获取实验总用户数"""
        # 这里简化处理，实际应该从用户分配记录中统计
        return 1000  # 模拟值

# 全局统计分析器实例
def create_statistical_analyzer(metrics_collector: BusinessMetricsCollector) -> ExperimentAnalyzer:
    """创建统计分析器实例"""
    return ExperimentAnalyzer(metrics_collector)