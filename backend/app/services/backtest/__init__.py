"""
回测引擎模块

该模块包含所有与策略回测和执行相关的服务，提供完整的量化交易策略验证解决方案。

模块结构：
- core: 核心引擎和基础类
- strategies: 交易策略实现
- execution: 回测执行器和进度监控
- analysis: 绩效分析和持仓分析
- reporting: 报告生成和图表数据
- optimization: 策略参数优化
- utils: 工具类和适配器
"""

# 从 models 导入数据模型
from .models import (
    SignalType,
    OrderType,
    TradingSignal,
    Trade,
    Position,
    BacktestConfig
)

# 从 core 导入策略和管理器
from .core import (
    BaseStrategy,
    MovingAverageStrategy,
    RSIStrategy,
    MACDStrategy,
    StrategyFactory,
    PortfolioManager
)

# 执行模块
from .execution import (
    BacktestExecutor,
    DataLoader,
    backtest_progress_monitor,
    BacktestProgressData,
    BacktestProgressStage
)

# 策略模块
from .strategies import (
    # 技术分析策略
    BollingerBandStrategy,
    StochasticStrategy,
    CCIStrategy,
    
    # 统计套利策略
    StatisticalArbitrageStrategy,
    PairsTradingStrategy,
    MeanReversionStrategy,
    CointegrationStrategy,
    
    # 因子投资策略
    FactorStrategy,
    ValueFactorStrategy,
    MomentumFactorStrategy,
    LowVolatilityStrategy,
    MultiFactorStrategy,
    
    # 高级策略工厂
    AdvancedStrategyFactory
)

# 分析模块（可选导入，避免循环依赖）
try:
    from .analysis import (
        EnhancedMetricsCalculator,
        PositionAnalyzer,
        MonthlyAnalyzer,
        BacktestComparisonAnalyzer
    )
    _ANALYSIS_AVAILABLE = True
except ImportError:
    _ANALYSIS_AVAILABLE = False

# 报告模块（可选导入）
try:
    from .reporting import (
        BacktestReportGenerator,
        ChartDataGenerator
    )
    _REPORTING_AVAILABLE = True
except ImportError:
    _REPORTING_AVAILABLE = False

# 工具模块（可选导入）
try:
    from .utils import (
        BacktestDataAdapter,
        ExtendedRiskMetrics,
        MonthlyReturnsAnalysis
    )
    _UTILS_AVAILABLE = True
except ImportError:
    _UTILS_AVAILABLE = False

__all__ = [
    # 枚举类型
    'SignalType',
    'OrderType',
    
    # 数据类
    'TradingSignal',
    'Trade',
    'Position',
    'BacktestConfig',
    
    # 策略基类
    'BaseStrategy',
    'MovingAverageStrategy',
    'RSIStrategy',
    'MACDStrategy',
    'StrategyFactory',
    
    # 管理器
    'PortfolioManager',
    
    # 执行器
    'BacktestExecutor',
    'DataLoader',
    'backtest_progress_monitor',
    'BacktestProgressData',
    'BacktestProgressStage',
    
    # 技术分析策略
    'BollingerBandStrategy',
    'StochasticStrategy',
    'CCIStrategy',
    
    # 统计套利策略
    'StatisticalArbitrageStrategy',
    'PairsTradingStrategy',
    'MeanReversionStrategy',
    'CointegrationStrategy',
    
    # 因子投资策略
    'FactorStrategy',
    'ValueFactorStrategy',
    'MomentumFactorStrategy',
    'LowVolatilityStrategy',
    'MultiFactorStrategy',
    
    # 高级策略工厂
    'AdvancedStrategyFactory'
]

# 向后兼容：导出分析、报告和工具模块的类（如果可用）
if _ANALYSIS_AVAILABLE:
    __all__.extend([
        'EnhancedMetricsCalculator',
        'PositionAnalyzer',
        'MonthlyAnalyzer',
        'BacktestComparisonAnalyzer'
    ])

if _REPORTING_AVAILABLE:
    __all__.extend([
        'BacktestReportGenerator',
        'ChartDataGenerator'
    ])

if _UTILS_AVAILABLE:
    __all__.extend([
        'BacktestDataAdapter',
        'ExtendedRiskMetrics',
        'MonthlyReturnsAnalysis'
    ])
