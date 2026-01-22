"""
分析相关的数据模型
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional


@dataclass
class ExtendedRiskMetrics:
    """扩展的风险指标（基于现有数据计算）"""
    # 现有指标
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    
    # 新增指标
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown_duration: int = 0
    var_95: float = 0.0  # 95% VaR
    downside_deviation: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """转换为字典"""
        return asdict(self)


@dataclass
class MonthlyReturnsAnalysis:
    """月度收益分析"""
    year: int
    month: int
    date: str
    monthly_return: float
    cumulative_return: float
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


@dataclass
class PositionAnalysis:
    """持仓分析（基于交易记录计算）"""
    stock_code: str
    stock_name: str = ""
    total_return: float = 0.0
    holding_days: int = 0
    trade_count: int = 0
    win_rate: float = 0.0
    avg_holding_period: int = 0
    max_position_value: float = 0.0
    winning_trades: int = 0
    losing_trades: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


@dataclass
class EnhancedPositionAnalysis:
    """增强的持仓分析（包含完整分析结果）"""
    # 股票表现数据（兼容原有格式）
    stock_performance: List[Dict[str, Any]]
    # 持仓权重分析
    position_weights: Optional[Dict[str, Any]] = None
    # 交易模式分析
    trading_patterns: Optional[Dict[str, Any]] = None
    # 持仓时间分析
    holding_periods: Optional[Dict[str, Any]] = None
    # 风险集中度分析
    concentration_risk: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {
            'stock_performance': self.stock_performance,
        }
        if self.position_weights:
            result['position_weights'] = self.position_weights
        if self.trading_patterns:
            result['trading_patterns'] = self.trading_patterns
        if self.holding_periods:
            result['holding_periods'] = self.holding_periods
        if self.concentration_risk:
            result['concentration_risk'] = self.concentration_risk
        return result


@dataclass
class DrawdownAnalysis:
    """回撤详细分析"""
    max_drawdown: float
    max_drawdown_date: Optional[str]
    max_drawdown_start: Optional[str]
    max_drawdown_end: Optional[str]
    max_drawdown_duration: int
    drawdown_curve: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


@dataclass
class EnhancedBacktestResult:
    """增强的回测结果数据结构（基于现有格式扩展）"""
    
    # === 现有字段（保持兼容） ===
    strategy_name: str
    stock_codes: List[str]
    start_date: str
    end_date: str
    initial_cash: float
    final_value: float
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    total_trades: int
    win_rate: float
    profit_factor: float
    winning_trades: int
    losing_trades: int
    backtest_config: Dict[str, Any]
    trade_history: List[Dict[str, Any]]
    portfolio_history: List[Dict[str, Any]]
    
    # === 新增字段（用于增强可视化） ===
    # 扩展的风险指标
    extended_risk_metrics: Optional[ExtendedRiskMetrics] = None
    # 月度收益分析
    monthly_returns: Optional[List[MonthlyReturnsAnalysis]] = None
    # 持仓分析（支持增强格式）
    position_analysis: Optional[Any] = None  # 可以是List[PositionAnalysis]或EnhancedPositionAnalysis
    # 基准对比数据
    benchmark_data: Optional[Dict[str, Any]] = None
    # 回撤详细分析
    drawdown_analysis: Optional[DrawdownAnalysis] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = asdict(self)
        
        # 处理嵌套对象
        if self.extended_risk_metrics:
            result['extended_risk_metrics'] = self.extended_risk_metrics.to_dict()
        
        if self.monthly_returns:
            result['monthly_returns'] = [mr.to_dict() for mr in self.monthly_returns]
        
        if self.position_analysis:
            # 支持两种格式：List[PositionAnalysis] 或 EnhancedPositionAnalysis
            if isinstance(self.position_analysis, EnhancedPositionAnalysis):
                result['position_analysis'] = self.position_analysis.to_dict()
            elif isinstance(self.position_analysis, list):
                result['position_analysis'] = [pa.to_dict() for pa in self.position_analysis]
            else:
                result['position_analysis'] = self.position_analysis
        
        if self.drawdown_analysis:
            result['drawdown_analysis'] = self.drawdown_analysis.to_dict()
        
        return result
