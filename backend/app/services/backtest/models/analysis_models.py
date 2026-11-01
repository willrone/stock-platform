"""
分析相关的数据模型
"""

import math
from dataclasses import asdict, dataclass
from numbers import Real
from typing import Any, Dict, List, Optional


def sanitize_non_finite(obj: Any) -> Any:
    """把 NaN / ±inf 递归换成 None —— **JSON 规范不允许 NaN / Infinity 字面量**。

    Python 的 ``json.dumps`` 默认 ``allow_nan=True``，会吐出裸的 ``NaN`` / ``Infinity``；
    FastAPI 的 JSONResponse 走的就是这条路径 → 那不是合法 JSON，浏览器的
    ``JSON.parse`` 直接抛异常 → 整个响应页挂掉（不是某个字段降级，是全挂）。

    真实触发路径：``portfolio_history`` 里混进一个 NaN（例如某天组合估值缺失）→
    回撤序列该点为 NaN → ``drawdown_curve`` 里出现 ``{"drawdown": NaN}``。
    实测（backend/tests/unit/backtest/test_backtest_data_adapter_properties.py
    的回归用例）：修复前 ``json.dumps(payload)`` 产出 ``NaN`` 裸字面量，
    ``allow_nan=False`` 直接 ValueError；修复后合法。

    只在**响应边界**（to_dict）做，不改动中间计算口径 —— 数值本身仍然是 NaN，
    只是不再以非法 JSON 的形式出网。
    """

    if isinstance(obj, bool):
        return obj
    if isinstance(obj, int):
        return obj  # 整数必然有限，且要保住 int 类型（不能变成 3.0）
    if isinstance(obj, Real):  # float / np.float64 / np.float32 ...
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, dict):
        return {key: sanitize_non_finite(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_non_finite(item) for item in obj]
    return obj


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
        result: Dict[str, Any] = {
            "stock_performance": self.stock_performance,
        }
        if self.position_weights:
            result["position_weights"] = self.position_weights
        if self.trading_patterns:
            result["trading_patterns"] = self.trading_patterns
        if self.holding_periods:
            result["holding_periods"] = self.holding_periods
        if self.concentration_risk:
            result["concentration_risk"] = self.concentration_risk
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
        # 非有限值（NaN/±inf）→ None：这里的曲线会直接落库/出网，
        # 而 NaN 不是合法 JSON（见模块级 sanitize_non_finite 的说明）
        return sanitize_non_finite(asdict(self))


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
    position_analysis: Optional[Any] = (
        None  # 可以是List[PositionAnalysis]或EnhancedPositionAnalysis
    )
    # 基准对比数据
    benchmark_data: Optional[Dict[str, Any]] = None
    # 回撤详细分析
    drawdown_analysis: Optional[DrawdownAnalysis] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = asdict(self)

        # 处理嵌套对象
        if self.extended_risk_metrics:
            result["extended_risk_metrics"] = self.extended_risk_metrics.to_dict()

        if self.monthly_returns:
            result["monthly_returns"] = [mr.to_dict() for mr in self.monthly_returns]

        if self.position_analysis:
            # 支持两种格式：List[PositionAnalysis] 或 EnhancedPositionAnalysis
            if isinstance(self.position_analysis, EnhancedPositionAnalysis):
                result["position_analysis"] = self.position_analysis.to_dict()
            elif isinstance(self.position_analysis, list):
                result["position_analysis"] = [
                    pa.to_dict() for pa in self.position_analysis
                ]
            else:
                result["position_analysis"] = self.position_analysis

        if self.drawdown_analysis:
            result["drawdown_analysis"] = self.drawdown_analysis.to_dict()

        # **响应边界**：非有限值（NaN / ±inf）一律换成 None。
        # 这里就是最终 JSON 的来源（调用方 data=enhanced_result.to_dict()），
        # 不做这一步 → 裸的 NaN 字面量出网 → 前端 JSON.parse 直接抛。
        return sanitize_non_finite(result)
