"""
回测详细结果数据模型
用于存储可视化所需的扩展数据
"""

from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
)

from app.core.database import Base


class BacktestDetailedResult(Base):
    """回测详细数据表（存储可视化所需的详细数据）"""

    __tablename__ = "backtest_detailed_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(String(50), nullable=False, index=True)
    backtest_id = Column(String(50), nullable=False, index=True)

    # 扩展风险指标
    sortino_ratio = Column(Float, default=0.0, comment="索提诺比率")
    calmar_ratio = Column(Float, default=0.0, comment="卡玛比率")
    max_drawdown_duration = Column(Integer, default=0, comment="最大回撤持续天数")
    var_95 = Column(Float, default=0.0, comment="95% VaR")
    downside_deviation = Column(Float, default=0.0, comment="下行偏差")

    # 回撤分析数据（JSON格式存储详细回撤曲线）
    drawdown_analysis = Column(JSON, nullable=True, comment="回撤详细分析数据")

    # 月度收益数据（JSON格式存储月度收益矩阵）
    monthly_returns = Column(JSON, nullable=True, comment="月度收益分析数据")

    # 持仓分析数据（JSON格式存储各股票表现）
    position_analysis = Column(JSON, nullable=True, comment="持仓分析数据")

    # 基准对比数据（JSON格式存储基准对比指标）
    benchmark_comparison = Column(JSON, nullable=True, comment="基准对比数据")

    # 滚动指标数据（JSON格式存储滚动夏普比率等）
    rolling_metrics = Column(JSON, nullable=True, comment="滚动指标数据")

    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(
        DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # 创建索引
    __table_args__ = (
        Index("idx_backtest_detailed_task_id", "task_id"),
        Index("idx_backtest_detailed_backtest_id", "backtest_id"),
    )

    def to_dict(self):
        return {
            "id": self.id,
            "task_id": self.task_id,
            "backtest_id": self.backtest_id,
            "extended_risk_metrics": {
                "sortino_ratio": self.sortino_ratio,
                "calmar_ratio": self.calmar_ratio,
                "max_drawdown_duration": self.max_drawdown_duration,
                "var_95": self.var_95,
                "downside_deviation": self.downside_deviation,
            },
            "drawdown_analysis": self.drawdown_analysis,
            "monthly_returns": self.monthly_returns,
            "position_analysis": self.position_analysis,
            "benchmark_comparison": self.benchmark_comparison,
            "rolling_metrics": self.rolling_metrics,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class BacktestChartCache(Base):
    """回测图表数据缓存表（用于提高图表加载性能）"""

    __tablename__ = "backtest_chart_cache"

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(String(50), nullable=False)
    chart_type = Column(
        String(50),
        nullable=False,
        comment="图表类型：equity_curve, drawdown_curve, monthly_heatmap等",
    )
    chart_data = Column(JSON, nullable=False, comment="图表数据JSON")
    data_hash = Column(String(64), nullable=True, comment="数据哈希值，用于检测数据变化")
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True, comment="缓存过期时间")

    # 创建唯一索引和其他索引
    __table_args__ = (
        Index("uk_task_chart", "task_id", "chart_type", unique=True),
        Index("idx_chart_cache_expires", "expires_at"),
        Index("idx_chart_cache_task_id", "task_id"),
    )

    def to_dict(self):
        return {
            "id": self.id,
            "task_id": self.task_id,
            "chart_type": self.chart_type,
            "chart_data": self.chart_data,
            "data_hash": self.data_hash,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }

    def is_expired(self) -> bool:
        """检查缓存是否过期"""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at


class PortfolioSnapshot(Base):
    """组合快照表（存储组合历史数据，用于绘制收益曲线）"""

    __tablename__ = "portfolio_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(String(50), nullable=False, index=True)
    backtest_id = Column(String(50), nullable=False, index=True)
    snapshot_date = Column(DateTime, nullable=False, comment="快照日期")
    portfolio_value = Column(Float, nullable=False, comment="组合总价值")
    cash = Column(Float, nullable=False, comment="现金余额")
    positions_count = Column(Integer, nullable=False, default=0, comment="持仓股票数量")
    total_return = Column(Float, nullable=False, default=0.0, comment="累计收益率")
    drawdown = Column(Float, nullable=False, default=0.0, comment="回撤幅度")
    positions = Column(JSON, nullable=True, comment="持仓详情JSON")

    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # 创建索引
    __table_args__ = (
        Index("idx_portfolio_task_date", "task_id", "snapshot_date"),
        Index("idx_portfolio_backtest_date", "backtest_id", "snapshot_date"),
    )

    def to_dict(self):
        return {
            "id": self.id,
            "task_id": self.task_id,
            "backtest_id": self.backtest_id,
            "snapshot_date": self.snapshot_date.isoformat()
            if self.snapshot_date
            else None,
            "portfolio_value": self.portfolio_value,
            "cash": self.cash,
            "positions_count": self.positions_count,
            "total_return": self.total_return,
            "drawdown": self.drawdown,
            "positions": self.positions,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class TradeRecord(Base):
    """交易记录表（存储详细的交易记录）"""

    __tablename__ = "trade_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(String(50), nullable=False, index=True)
    backtest_id = Column(String(50), nullable=False, index=True)
    trade_id = Column(String(50), nullable=False, comment="交易ID")
    stock_code = Column(String(20), nullable=False, comment="股票代码")
    stock_name = Column(String(100), nullable=True, comment="股票名称")
    action = Column(String(10), nullable=False, comment="交易动作：BUY/SELL")
    quantity = Column(Integer, nullable=False, comment="交易数量")
    price = Column(Float, nullable=False, comment="交易价格")
    timestamp = Column(DateTime, nullable=False, comment="交易时间")
    commission = Column(Float, nullable=False, default=0.0, comment="手续费")
    pnl = Column(Float, nullable=True, comment="盈亏金额（仅卖出时有值）")
    holding_days = Column(Integer, nullable=True, comment="持仓天数（仅卖出时有值）")

    # 技术指标（交易时的技术指标快照）
    technical_indicators = Column(JSON, nullable=True, comment="交易时的技术指标")

    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # 创建索引
    __table_args__ = (
        Index("idx_trade_task_stock", "task_id", "stock_code"),
        Index("idx_trade_backtest_time", "backtest_id", "timestamp"),
        Index("idx_trade_stock_time", "stock_code", "timestamp"),
    )

    def to_dict(self):
        return {
            "id": self.id,
            "task_id": self.task_id,
            "backtest_id": self.backtest_id,
            "trade_id": self.trade_id,
            "stock_code": self.stock_code,
            "stock_name": self.stock_name,
            "action": self.action,
            "quantity": self.quantity,
            "price": self.price,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "commission": self.commission,
            "pnl": self.pnl,
            "holding_days": self.holding_days,
            "technical_indicators": self.technical_indicators,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class SignalRecord(Base):
    """信号记录表（存储回测过程中生成的交易信号）"""

    __tablename__ = "signal_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(String(50), nullable=False, index=True)
    backtest_id = Column(String(50), nullable=False, index=True)
    signal_id = Column(String(50), nullable=False, comment="信号ID")
    stock_code = Column(String(20), nullable=False, comment="股票代码")
    stock_name = Column(String(100), nullable=True, comment="股票名称")
    signal_type = Column(String(10), nullable=False, comment="信号类型：BUY/SELL")
    timestamp = Column(DateTime, nullable=False, comment="信号时间")
    price = Column(Float, nullable=False, comment="信号价格")
    strength = Column(Float, nullable=False, default=0.0, comment="信号强度")
    reason = Column(Text, nullable=True, comment="信号原因")
    signal_metadata = Column(JSON, nullable=True, comment="元数据（JSON格式）")
    executed = Column(Boolean, nullable=False, default=False, comment="是否被执行")
    execution_reason = Column(Text, nullable=True, comment="执行原因：已执行时为空，未执行时记录未执行原因")
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # 创建索引
    __table_args__ = (
        Index("idx_signal_task_id", "task_id"),  # 单独的task_id索引，用于快速查询
        Index("idx_signal_task_stock", "task_id", "stock_code"),
        Index("idx_signal_backtest_time", "backtest_id", "timestamp"),
        Index("idx_signal_stock_time", "stock_code", "timestamp"),
        Index("idx_signal_type", "signal_type"),
        Index("idx_signal_executed", "executed"),
    )

    def to_dict(self):
        # 安全地获取 execution_reason，兼容字段不存在的情况
        execution_reason = None
        try:
            execution_reason = self.execution_reason
        except (AttributeError, KeyError):
            # 如果字段不存在，返回 None
            pass

        return {
            "id": self.id,
            "task_id": self.task_id,
            "backtest_id": self.backtest_id,
            "signal_id": self.signal_id,
            "stock_code": self.stock_code,
            "stock_name": self.stock_name,
            "signal_type": self.signal_type,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "price": self.price,
            "strength": self.strength,
            "reason": self.reason,
            "metadata": self.signal_metadata,  # 对外接口仍使用metadata名称
            "executed": self.executed,
            "execution_reason": execution_reason,  # 确保总是返回这个字段，即使值为 None
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class BacktestBenchmark(Base):
    """回测基准数据表（存储基准指数数据用于对比）"""

    __tablename__ = "backtest_benchmarks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(String(50), nullable=False, index=True)
    backtest_id = Column(String(50), nullable=False, index=True)
    benchmark_symbol = Column(String(20), nullable=False, comment="基准代码，如000300.SH")
    benchmark_name = Column(String(100), nullable=False, comment="基准名称，如沪深300")
    benchmark_data = Column(JSON, nullable=False, comment="基准历史数据")

    # 基准对比指标
    correlation = Column(Float, nullable=True, comment="相关系数")
    beta = Column(Float, nullable=True, comment="贝塔系数")
    alpha = Column(Float, nullable=True, comment="阿尔法系数")
    tracking_error = Column(Float, nullable=True, comment="跟踪误差")
    information_ratio = Column(Float, nullable=True, comment="信息比率")
    excess_return = Column(Float, nullable=True, comment="超额收益")

    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(
        DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # 创建索引
    __table_args__ = (
        Index("idx_benchmark_task_symbol", "task_id", "benchmark_symbol"),
        Index("idx_benchmark_backtest_id", "backtest_id"),
    )

    def to_dict(self):
        return {
            "id": self.id,
            "task_id": self.task_id,
            "backtest_id": self.backtest_id,
            "benchmark_symbol": self.benchmark_symbol,
            "benchmark_name": self.benchmark_name,
            "benchmark_data": self.benchmark_data,
            "comparison_metrics": {
                "correlation": self.correlation,
                "beta": self.beta,
                "alpha": self.alpha,
                "tracking_error": self.tracking_error,
                "information_ratio": self.information_ratio,
                "excess_return": self.excess_return,
            },
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class BacktestStatistics(Base):
    """回测统计信息表（预计算统计，加速页面加载）"""

    __tablename__ = "backtest_statistics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(
        String(50), nullable=False, unique=True, index=True, comment="任务ID"
    )
    backtest_id = Column(String(50), nullable=False, index=True, comment="回测ID")

    # ========== 信号统计 ==========
    total_signals = Column(Integer, default=0, comment="总信号数")
    buy_signals = Column(Integer, default=0, comment="买入信号数")
    sell_signals = Column(Integer, default=0, comment="卖出信号数")
    executed_signals = Column(Integer, default=0, comment="已执行信号数")
    unexecuted_signals = Column(Integer, default=0, comment="未执行信号数")
    execution_rate = Column(Float, default=0.0, comment="执行率")
    avg_signal_strength = Column(Float, default=0.0, comment="平均信号强度")

    # ========== 交易统计 ==========
    total_trades = Column(Integer, default=0, comment="总交易数")
    buy_trades = Column(Integer, default=0, comment="买入交易数")
    sell_trades = Column(Integer, default=0, comment="卖出交易数")
    winning_trades = Column(Integer, default=0, comment="盈利交易数")
    losing_trades = Column(Integer, default=0, comment="亏损交易数")
    win_rate = Column(Float, default=0.0, comment="胜率")
    avg_profit = Column(Float, default=0.0, comment="平均盈利")
    avg_loss = Column(Float, default=0.0, comment="平均亏损")
    profit_factor = Column(Float, default=0.0, comment="盈亏比")
    total_commission = Column(Float, default=0.0, comment="总手续费")
    total_pnl = Column(Float, default=0.0, comment="总盈亏")
    avg_holding_days = Column(Float, default=0.0, comment="平均持仓天数")

    # ========== 持仓统计 ==========
    total_stocks = Column(Integer, default=0, comment="总股票数")
    profitable_stocks = Column(Integer, default=0, comment="盈利股票数")
    avg_stock_return = Column(Float, default=0.0, comment="平均股票收益率")
    max_stock_return = Column(Float, nullable=True, comment="最大股票收益率")
    min_stock_return = Column(Float, nullable=True, comment="最小股票收益率")

    # ========== 时间范围统计 ==========
    first_signal_date = Column(DateTime, nullable=True, comment="第一个信号日期")
    last_signal_date = Column(DateTime, nullable=True, comment="最后一个信号日期")
    first_trade_date = Column(DateTime, nullable=True, comment="第一笔交易日期")
    last_trade_date = Column(DateTime, nullable=True, comment="最后一笔交易日期")
    trading_days = Column(Integer, default=0, comment="交易天数")

    # ========== 股票分布统计 ==========
    unique_stocks_signaled = Column(Integer, default=0, comment="产生信号的股票数")
    unique_stocks_traded = Column(Integer, default=0, comment="实际交易的股票数")
    most_signaled_stock = Column(String(20), nullable=True, comment="信号最多的股票代码")
    most_traded_stock = Column(String(20), nullable=True, comment="交易最多的股票代码")

    # ========== 性能指标统计 ==========
    max_single_profit = Column(Float, nullable=True, comment="单笔最大盈利")
    max_single_loss = Column(Float, nullable=True, comment="单笔最大亏损")
    max_consecutive_wins = Column(Integer, default=0, comment="最大连续盈利次数")
    max_consecutive_losses = Column(Integer, default=0, comment="最大连续亏损次数")
    largest_position_size = Column(Float, nullable=True, comment="最大持仓金额")

    # 元数据
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(
        DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # 创建索引
    __table_args__ = (
        Index("idx_statistics_task_id", "task_id", unique=True),
        Index("idx_statistics_backtest_id", "backtest_id"),
    )

    def to_dict(self):
        """转换为字典格式"""
        return {
            "id": self.id,
            "task_id": self.task_id,
            "backtest_id": self.backtest_id,
            "signal_statistics": {
                "total_signals": self.total_signals,
                "buy_signals": self.buy_signals,
                "sell_signals": self.sell_signals,
                "executed_signals": self.executed_signals,
                "unexecuted_signals": self.unexecuted_signals,
                "execution_rate": self.execution_rate,
                "avg_signal_strength": self.avg_signal_strength,
            },
            "trade_statistics": {
                "total_trades": self.total_trades,
                "buy_trades": self.buy_trades,
                "sell_trades": self.sell_trades,
                "winning_trades": self.winning_trades,
                "losing_trades": self.losing_trades,
                "win_rate": self.win_rate,
                "avg_profit": self.avg_profit,
                "avg_loss": self.avg_loss,
                "profit_factor": self.profit_factor,
                "total_commission": self.total_commission,
                "total_pnl": self.total_pnl,
                "avg_holding_days": self.avg_holding_days,
            },
            "position_statistics": {
                "total_stocks": self.total_stocks,
                "profitable_stocks": self.profitable_stocks,
                "avg_stock_return": self.avg_stock_return,
                "max_stock_return": self.max_stock_return,
                "min_stock_return": self.min_stock_return,
            },
            "time_range": {
                "first_signal_date": self.first_signal_date.isoformat()
                if self.first_signal_date
                else None,
                "last_signal_date": self.last_signal_date.isoformat()
                if self.last_signal_date
                else None,
                "first_trade_date": self.first_trade_date.isoformat()
                if self.first_trade_date
                else None,
                "last_trade_date": self.last_trade_date.isoformat()
                if self.last_trade_date
                else None,
                "trading_days": self.trading_days,
            },
            "stock_distribution": {
                "unique_stocks_signaled": self.unique_stocks_signaled,
                "unique_stocks_traded": self.unique_stocks_traded,
                "most_signaled_stock": self.most_signaled_stock,
                "most_traded_stock": self.most_traded_stock,
            },
            "performance_metrics": {
                "max_single_profit": self.max_single_profit,
                "max_single_loss": self.max_single_loss,
                "max_consecutive_wins": self.max_consecutive_wins,
                "max_consecutive_losses": self.max_consecutive_losses,
                "largest_position_size": self.largest_position_size,
            },
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
