"""
风险评估服务 - 专门的风险分析和评估功能
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from scipy import stats
from loguru import logger

from app.models.task_models import RiskMetrics
from app.core.error_handler import PredictionError, ErrorSeverity, ErrorContext


@dataclass
class RiskAssessmentConfig:
    """风险评估配置"""
    confidence_levels: List[float] = None
    time_horizons: List[int] = None  # 天数
    risk_metrics: List[str] = None
    monte_carlo_simulations: int = 10000
    
    def __post_init__(self):
        if self.confidence_levels is None:
            self.confidence_levels = [0.90, 0.95, 0.99]
        if self.time_horizons is None:
            self.time_horizons = [1, 5, 10, 20]
        if self.risk_metrics is None:
            self.risk_metrics = ['var', 'es', 'volatility', 'max_drawdown', 'sharpe_ratio']


@dataclass
class ConfidenceInterval:
    """置信区间"""
    lower_bound: float
    upper_bound: float
    confidence_level: float
    method: str  # parametric, bootstrap, monte_carlo


@dataclass
class RiskAssessmentResult:
    """风险评估结果"""
    stock_code: str
    assessment_date: datetime
    current_price: float
    predicted_price: float
    confidence_intervals: Dict[float, ConfidenceInterval]
    risk_metrics: Dict[str, float]
    scenario_analysis: Dict[str, float]
    stress_test_results: Dict[str, float]
    risk_rating: str  # low, medium, high, extreme


class ConfidenceIntervalCalculator:
    """置信区间计算器"""
    
    @staticmethod
    def parametric_interval(predicted_price: float, volatility: float, 
                          confidence_level: float, time_horizon: int = 1) -> ConfidenceInterval:
        """基于参数方法计算置信区间"""
        # 调整时间期限的波动率
        adjusted_volatility = volatility * np.sqrt(time_horizon)
        
        # 计算Z分数
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        
        # 计算置信区间
        margin = predicted_price * adjusted_volatility * z_score
        lower_bound = predicted_price - margin
        upper_bound = predicted_price + margin
        
        return ConfidenceInterval(
            lower_bound=max(0, lower_bound),  # 价格不能为负
            upper_bound=upper_bound,
            confidence_level=confidence_level,
            method="parametric"
        )
    
    @staticmethod
    def bootstrap_interval(historical_returns: pd.Series, current_price: float,
                          confidence_level: float, n_bootstrap: int = 1000) -> ConfidenceInterval:
        """基于自助法计算置信区间"""
        if len(historical_returns) < 30:
            logger.warning("历史数据不足，使用参数方法")
            volatility = historical_returns.std()
            return ConfidenceIntervalCalculator.parametric_interval(
                current_price, volatility, confidence_level
            )
        
        # 自助抽样
        bootstrap_returns = []
        for _ in range(n_bootstrap):
            sample_returns = np.random.choice(historical_returns.dropna(), size=len(historical_returns), replace=True)
            bootstrap_returns.append(np.mean(sample_returns))
        
        bootstrap_returns = np.array(bootstrap_returns)
        
        # 计算置信区间
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_return = np.percentile(bootstrap_returns, lower_percentile)
        upper_return = np.percentile(bootstrap_returns, upper_percentile)
        
        lower_bound = current_price * (1 + lower_return)
        upper_bound = current_price * (1 + upper_return)
        
        return ConfidenceInterval(
            lower_bound=max(0, lower_bound),
            upper_bound=upper_bound,
            confidence_level=confidence_level,
            method="bootstrap"
        )
    
    @staticmethod
    def monte_carlo_interval(current_price: float, expected_return: float, volatility: float,
                           confidence_level: float, time_horizon: int = 1, 
                           n_simulations: int = 10000) -> ConfidenceInterval:
        """基于蒙特卡洛模拟计算置信区间"""
        # 几何布朗运动模拟
        dt = 1 / 252  # 日时间步长
        n_steps = time_horizon
        
        # 生成随机路径
        random_shocks = np.random.normal(0, 1, (n_simulations, n_steps))
        
        # 计算价格路径
        price_paths = np.zeros((n_simulations, n_steps + 1))
        price_paths[:, 0] = current_price
        
        for t in range(1, n_steps + 1):
            price_paths[:, t] = price_paths[:, t-1] * np.exp(
                (expected_return - 0.5 * volatility**2) * dt + 
                volatility * np.sqrt(dt) * random_shocks[:, t-1]
            )
        
        # 获取最终价格分布
        final_prices = price_paths[:, -1]
        
        # 计算置信区间
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bound = np.percentile(final_prices, lower_percentile)
        upper_bound = np.percentile(final_prices, upper_percentile)
        
        return ConfidenceInterval(
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            confidence_level=confidence_level,
            method="monte_carlo"
        )


class RiskMetricsCalculator:
    """风险指标计算器"""
    
    @staticmethod
    def calculate_var(returns: pd.Series, confidence_levels: List[float]) -> Dict[float, float]:
        """计算不同置信水平的VaR"""
        var_results = {}
        clean_returns = returns.dropna()
        
        if len(clean_returns) == 0:
            return {level: 0.0 for level in confidence_levels}
        
        for level in confidence_levels:
            var_results[level] = np.percentile(clean_returns, (1 - level) * 100)
        
        return var_results
    
    @staticmethod
    def calculate_expected_shortfall(returns: pd.Series, confidence_levels: List[float]) -> Dict[float, float]:
        """计算期望损失(ES/CVaR)"""
        es_results = {}
        clean_returns = returns.dropna()
        
        if len(clean_returns) == 0:
            return {level: 0.0 for level in confidence_levels}
        
        for level in confidence_levels:
            var = np.percentile(clean_returns, (1 - level) * 100)
            tail_returns = clean_returns[clean_returns <= var]
            es_results[level] = tail_returns.mean() if len(tail_returns) > 0 else var
        
        return es_results
    
    @staticmethod
    def calculate_volatility_metrics(returns: pd.Series, time_horizons: List[int]) -> Dict[str, float]:
        """计算波动率指标"""
        clean_returns = returns.dropna()
        if len(clean_returns) == 0:
            return {}
        
        metrics = {}
        
        # 历史波动率
        daily_vol = clean_returns.std()
        metrics['daily_volatility'] = daily_vol
        metrics['annualized_volatility'] = daily_vol * np.sqrt(252)
        
        # 不同时间期限的波动率
        for horizon in time_horizons:
            if len(clean_returns) >= horizon:
                rolling_vol = clean_returns.rolling(window=horizon).std().iloc[-1]
                metrics[f'volatility_{horizon}d'] = rolling_vol * np.sqrt(252)
        
        # GARCH波动率（简化版）
        try:
            # 简单的EWMA波动率
            lambda_param = 0.94
            ewma_var = clean_returns.ewm(alpha=1-lambda_param).var().iloc[-1]
            metrics['garch_volatility'] = np.sqrt(ewma_var * 252)
        except:
            metrics['garch_volatility'] = metrics['annualized_volatility']
        
        return metrics
    
    @staticmethod
    def calculate_drawdown_metrics(prices: pd.Series) -> Dict[str, float]:
        """计算回撤指标"""
        if len(prices) == 0:
            return {}
        
        # 计算累积收益
        returns = prices.pct_change().fillna(0)
        cumulative = (1 + returns).cumprod()
        
        # 计算回撤
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        metrics = {
            'max_drawdown': drawdown.min(),
            'current_drawdown': drawdown.iloc[-1],
            'avg_drawdown': drawdown[drawdown < 0].mean() if (drawdown < 0).any() else 0,
            'drawdown_duration': len(drawdown[drawdown < 0]) if (drawdown < 0).any() else 0
        }
        
        return metrics
    
    @staticmethod
    def calculate_performance_metrics(returns: pd.Series, risk_free_rate: float = 0.02) -> Dict[str, float]:
        """计算绩效指标"""
        clean_returns = returns.dropna()
        if len(clean_returns) == 0:
            return {}
        
        metrics = {}
        
        # 基础统计
        metrics['mean_return'] = clean_returns.mean() * 252
        metrics['std_return'] = clean_returns.std() * np.sqrt(252)
        
        # 夏普比率
        excess_returns = clean_returns.mean() - risk_free_rate / 252
        if clean_returns.std() > 0:
            metrics['sharpe_ratio'] = excess_returns / clean_returns.std() * np.sqrt(252)
        else:
            metrics['sharpe_ratio'] = 0
        
        # 索提诺比率
        downside_returns = clean_returns[clean_returns < 0]
        if len(downside_returns) > 0:
            downside_deviation = downside_returns.std() * np.sqrt(252)
            metrics['sortino_ratio'] = metrics['mean_return'] / downside_deviation
        else:
            metrics['sortino_ratio'] = float('inf') if metrics['mean_return'] > 0 else 0
        
        # 偏度和峰度
        metrics['skewness'] = clean_returns.skew()
        metrics['kurtosis'] = clean_returns.kurtosis()
        
        return metrics


class ScenarioAnalysis:
    """情景分析"""
    
    @staticmethod
    def stress_test(current_price: float, volatility: float, 
                   scenarios: Dict[str, float]) -> Dict[str, float]:
        """压力测试"""
        results = {}
        
        default_scenarios = {
            'market_crash': -0.20,  # 市场崩盘 -20%
            'severe_correction': -0.15,  # 严重调整 -15%
            'moderate_correction': -0.10,  # 温和调整 -10%
            'normal_volatility': -0.05,  # 正常波动 -5%
            'bull_market': 0.15,  # 牛市 +15%
            'extreme_bull': 0.25   # 极端牛市 +25%
        }
        
        test_scenarios = scenarios if scenarios else default_scenarios
        
        for scenario_name, shock in test_scenarios.items():
            stressed_price = current_price * (1 + shock)
            results[scenario_name] = stressed_price
        
        return results
    
    @staticmethod
    def sensitivity_analysis(current_price: float, base_volatility: float,
                           volatility_shocks: List[float] = None) -> Dict[str, Dict[str, float]]:
        """敏感性分析"""
        if volatility_shocks is None:
            volatility_shocks = [-0.5, -0.25, 0, 0.25, 0.5, 1.0]
        
        results = {}
        
        for shock in volatility_shocks:
            shocked_vol = base_volatility * (1 + shock)
            
            # 计算不同置信水平下的VaR
            confidence_levels = [0.90, 0.95, 0.99]
            var_results = {}
            
            for level in confidence_levels:
                z_score = stats.norm.ppf(1 - level)
                var = current_price * shocked_vol * z_score
                var_results[f'var_{level}'] = var
            
            results[f'vol_shock_{shock:+.0%}'] = var_results
        
        return results


class RiskAssessmentService:
    """风险评估服务主类"""
    
    def __init__(self):
        self.confidence_calculator = ConfidenceIntervalCalculator()
        self.risk_calculator = RiskMetricsCalculator()
        self.scenario_analysis = ScenarioAnalysis()
    
    def assess_prediction_risk(self, stock_code: str, current_price: float, 
                             predicted_price: float, historical_data: pd.DataFrame,
                             config: Optional[RiskAssessmentConfig] = None) -> RiskAssessmentResult:
        """全面的预测风险评估"""
        if config is None:
            config = RiskAssessmentConfig()
        
        try:
            # 计算历史收益率
            returns = historical_data['close'].pct_change().dropna()
            
            # 计算置信区间
            confidence_intervals = self._calculate_confidence_intervals(
                current_price, predicted_price, returns, config
            )
            
            # 计算风险指标
            risk_metrics = self._calculate_risk_metrics(returns, historical_data['close'], config)
            
            # 情景分析
            scenario_results = self.scenario_analysis.stress_test(
                predicted_price, returns.std()
            )
            
            # 敏感性分析
            sensitivity_results = self.scenario_analysis.sensitivity_analysis(
                predicted_price, returns.std()
            )
            
            # 风险评级
            risk_rating = self._calculate_risk_rating(risk_metrics, returns)
            
            result = RiskAssessmentResult(
                stock_code=stock_code,
                assessment_date=datetime.now(),
                current_price=current_price,
                predicted_price=predicted_price,
                confidence_intervals=confidence_intervals,
                risk_metrics=risk_metrics,
                scenario_analysis=scenario_results,
                stress_test_results=sensitivity_results,
                risk_rating=risk_rating
            )
            
            logger.info(f"风险评估完成: {stock_code}, 风险等级: {risk_rating}")
            return result
            
        except Exception as e:
            raise PredictionError(
                message=f"风险评估失败: {str(e)}",
                severity=ErrorSeverity.HIGH,
                context=ErrorContext(stock_code=stock_code),
                original_exception=e
            )
    
    def _calculate_confidence_intervals(self, current_price: float, predicted_price: float,
                                      returns: pd.Series, config: RiskAssessmentConfig) -> Dict[float, ConfidenceInterval]:
        """计算置信区间"""
        intervals = {}
        volatility = returns.std()
        expected_return = returns.mean()
        
        for confidence_level in config.confidence_levels:
            # 使用多种方法计算置信区间
            if len(returns) >= 100:
                # 数据充足时使用蒙特卡洛方法
                interval = self.confidence_calculator.monte_carlo_interval(
                    predicted_price, expected_return, volatility, confidence_level,
                    n_simulations=config.monte_carlo_simulations
                )
            elif len(returns) >= 30:
                # 中等数据量使用自助法
                interval = self.confidence_calculator.bootstrap_interval(
                    returns, predicted_price, confidence_level
                )
            else:
                # 数据不足时使用参数方法
                interval = self.confidence_calculator.parametric_interval(
                    predicted_price, volatility, confidence_level
                )
            
            intervals[confidence_level] = interval
        
        return intervals
    
    def _calculate_risk_metrics(self, returns: pd.Series, prices: pd.Series,
                              config: RiskAssessmentConfig) -> Dict[str, float]:
        """计算风险指标"""
        all_metrics = {}
        
        if 'var' in config.risk_metrics:
            var_metrics = self.risk_calculator.calculate_var(returns, config.confidence_levels)
            all_metrics.update({f'var_{level}': value for level, value in var_metrics.items()})
        
        if 'es' in config.risk_metrics:
            es_metrics = self.risk_calculator.calculate_expected_shortfall(returns, config.confidence_levels)
            all_metrics.update({f'es_{level}': value for level, value in es_metrics.items()})
        
        if 'volatility' in config.risk_metrics:
            vol_metrics = self.risk_calculator.calculate_volatility_metrics(returns, config.time_horizons)
            all_metrics.update(vol_metrics)
        
        if 'max_drawdown' in config.risk_metrics:
            dd_metrics = self.risk_calculator.calculate_drawdown_metrics(prices)
            all_metrics.update(dd_metrics)
        
        if 'sharpe_ratio' in config.risk_metrics:
            perf_metrics = self.risk_calculator.calculate_performance_metrics(returns)
            all_metrics.update(perf_metrics)
        
        return all_metrics
    
    def _calculate_risk_rating(self, risk_metrics: Dict[str, float], returns: pd.Series) -> str:
        """计算风险评级"""
        try:
            # 基于多个指标计算风险评分
            risk_score = 0
            
            # 波动率评分 (0-25分)
            volatility = risk_metrics.get('annualized_volatility', 0)
            if volatility > 0.4:
                risk_score += 25
            elif volatility > 0.3:
                risk_score += 20
            elif volatility > 0.2:
                risk_score += 15
            elif volatility > 0.1:
                risk_score += 10
            else:
                risk_score += 5
            
            # VaR评分 (0-25分)
            var_95 = abs(risk_metrics.get('var_0.95', 0))
            if var_95 > 0.1:
                risk_score += 25
            elif var_95 > 0.05:
                risk_score += 20
            elif var_95 > 0.03:
                risk_score += 15
            elif var_95 > 0.02:
                risk_score += 10
            else:
                risk_score += 5
            
            # 最大回撤评分 (0-25分)
            max_dd = abs(risk_metrics.get('max_drawdown', 0))
            if max_dd > 0.3:
                risk_score += 25
            elif max_dd > 0.2:
                risk_score += 20
            elif max_dd > 0.15:
                risk_score += 15
            elif max_dd > 0.1:
                risk_score += 10
            else:
                risk_score += 5
            
            # 夏普比率评分 (0-25分，分数越低风险越高)
            sharpe = risk_metrics.get('sharpe_ratio', 0)
            if sharpe < -1:
                risk_score += 25
            elif sharpe < 0:
                risk_score += 20
            elif sharpe < 0.5:
                risk_score += 15
            elif sharpe < 1:
                risk_score += 10
            else:
                risk_score += 5
            
            # 根据总分确定风险等级
            if risk_score >= 80:
                return "extreme"
            elif risk_score >= 60:
                return "high"
            elif risk_score >= 40:
                return "medium"
            else:
                return "low"
                
        except Exception as e:
            logger.warning(f"风险评级计算失败: {e}")
            return "medium"  # 默认中等风险
    
    def calculate_portfolio_risk(self, positions: Dict[str, float], 
                               individual_risks: Dict[str, RiskAssessmentResult]) -> Dict[str, float]:
        """计算投资组合风险"""
        try:
            if not positions or not individual_risks:
                return {}
            
            # 计算权重
            total_value = sum(positions.values())
            weights = {stock: value / total_value for stock, value in positions.items()}
            
            # 计算组合VaR (简化版，假设独立)
            portfolio_var_95 = 0
            for stock, weight in weights.items():
                if stock in individual_risks:
                    stock_var = individual_risks[stock].risk_metrics.get('var_0.95', 0)
                    portfolio_var_95 += (weight * stock_var) ** 2
            
            portfolio_var_95 = np.sqrt(portfolio_var_95)
            
            # 计算组合波动率
            portfolio_vol = 0
            for stock, weight in weights.items():
                if stock in individual_risks:
                    stock_vol = individual_risks[stock].risk_metrics.get('annualized_volatility', 0)
                    portfolio_vol += (weight * stock_vol) ** 2
            
            portfolio_vol = np.sqrt(portfolio_vol)
            
            return {
                'portfolio_var_95': portfolio_var_95,
                'portfolio_volatility': portfolio_vol,
                'diversification_ratio': portfolio_vol / np.mean([
                    individual_risks[stock].risk_metrics.get('annualized_volatility', 0)
                    for stock in weights.keys() if stock in individual_risks
                ])
            }
            
        except Exception as e:
            logger.error(f"组合风险计算失败: {e}")
            return {}