"""
回测详细结果数据模型（PostgreSQL）
子表存储可视化所需的扩展数据；BacktestDetailedResult 已合并到 BacktestResult。
"""

import enum

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.sql import func

from app.core.database import Base
from app.models.task_models import BacktestResult


# ── 向后兼容别名：引用方 import BacktestDetailedResult 不会报错 ──
BacktestDetailedResult = BacktestResult


# ──────────────────────────── 枚举 ────────────────────────────


class TradeAction(enum.Enum):
    """交易动作"""

    BUY = "BUY"
    SELL = "SELL"


class SignalType(enum.Enum):
    """信号类型"""

    BUY = "BUY"
    SELL = "SELL"


# ──────────────────────────── 子表 ────────────────────────────


class BacktestChartCache(Base):
    """回测图表数据缓存表"""

    __tablename__ = "backtest_chart_cache"

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    backtest_id = Column(
        UUID(as_uuid=True),
        ForeignKey("backtest_results.backtest_id", ondelete="CASCADE"),
        nullable=False,
    )
    chart_type = Column(
        String(50),
        nullable=False,
        comment="图表类型：equity_curve, drawdown_curve, monthly_heatmap 等",
    )
    chart_data = Column(JSONB, nullable=False, comment="图表数据")
    data_hash = Column(String(64), nullable=True, comment="数据哈希值，用于检测数据变化")
    created_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    expires_at = Column(DateTime(timezone=True), nullable=True, comment="缓存过期时间")

    __table_args__ = (
        Index(
            "uq_chart_cache_backtest_type",
            "backtest_id",
            "chart_type",
            unique=True,
        ),
        Index("ix_chart_cache_expires", "expires_at"),
    )

    def to_dict(self):
        return {
            "id": str(self.id),
            "backtest_id": str(self.backtest_id),
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
        from datetime import datetime, timezone

        return datetime.now(timezone.utc) > self.expires_at


class PortfolioSnapshot(Base):
    """组合快照表（存储组合历史数据，用于绘制收益曲线）"""

    __tablename__ = "portfolio_snapshots"

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    backtest_id = Column(
        UUID(as_uuid=True),
        ForeignKey("backtest_results.backtest_id", ondelete="CASCADE"),
        nullable=False,
    )
    snapshot_date = Column(DateTime(timezone=True), nullable=False, comment="快照日期")
    portfolio_value = Column(Numeric(15, 2), nullable=False, comment="组合总价值")
    cash = Column(Numeric(15, 2), nullable=False, comment="现金余额")
    positions_count = Column(Integer, nullable=False, server_default=text("0"), comment="持仓股票数量")
    total_return = Column(Float, nullable=False, server_default=text("0"), comment="累计收益率")
    drawdown = Column(Float, nullable=False, server_default=text("0"), comment="回撤幅度")
    positions = Column(JSONB, nullable=True, comment="持仓详情")

    created_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        Index("ix_portfolio_backtest_date", "backtest_id", "snapshot_date"),
    )

    def to_dict(self):
        return {
            "id": str(self.id),
            "backtest_id": str(self.backtest_id),
            "snapshot_date": self.snapshot_date.isoformat()
            if self.snapshot_date
            else None,
            "portfolio_value": float(self.portfolio_value)
            if self.portfolio_value
            else None,
            "cash": float(self.cash) if self.cash else None,
            "positions_count": self.positions_count,
            "total_return": self.total_return,
            "drawdown": self.drawdown,
            "positions": self.positions,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class TradeRecord(Base):
    """交易记录表"""

    __tablename__ = "trade_records"

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    backtest_id = Column(
        UUID(as_uuid=True),
        ForeignKey("backtest_results.backtest_id", ondelete="CASCADE"),
        nullable=False,
    )
    trade_id = Column(String(50), nullable=False, comment="交易ID")
    stock_code = Column(String(20), nullable=False, comment="股票代码")
    stock_name = Column(String(100), nullable=True, comment="股票名称")
    action = Column(
        Enum(TradeAction, name="trade_action", create_constraint=True),
        nullable=False,
        comment="交易动作",
    )
    quantity = Column(Integer, nullable=False, comment="交易数量")
    price = Column(Numeric(15, 2), nullable=False, comment="交易价格")
    timestamp = Column(DateTime(timezone=True), nullable=False, comment="交易时间")
    commission = Column(
        Numeric(15, 2), nullable=False, server_default=text("0"), comment="手续费"
    )
    pnl = Column(Numeric(15, 2), nullable=True, comment="盈亏金额（仅卖出时有值）")
    holding_days = Column(Integer, nullable=True, comment="持仓天数（仅卖出时有值）")
    technical_indicators = Column(JSONB, nullable=True, comment="交易时的技术指标")

    created_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        Index("ix_trade_backtest_time", "backtest_id", "timestamp"),
        Index("ix_trade_stock_time", "stock_code", "timestamp"),
    )

    def to_dict(self):
        return {
            "id": str(self.id),
            "backtest_id": str(self.backtest_id),
            "trade_id": self.trade_id,
            "stock_code": self.stock_code,
            "stock_name": self.stock_name,
            "action": self.action.value if self.action else None,
            "quantity": self.quantity,
            "price": float(self.price) if self.price else None,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "commission": float(self.commission) if self.commission else None,
            "pnl": float(self.pnl) if self.pnl is not None else None,
            "holding_days": self.holding_days,
            "technical_indicators": self.technical_indicators,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class SignalRecord(Base):
    """信号记录表"""

    __tablename__ = "signal_records"

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    backtest_id = Column(
        UUID(as_uuid=True),
        ForeignKey("backtest_results.backtest_id", ondelete="CASCADE"),
        nullable=False,
    )
    signal_id = Column(String(50), nullable=False, comment="信号ID")
    stock_code = Column(String(20), nullable=False, comment="股票代码")
    stock_name = Column(String(100), nullable=True, comment="股票名称")
    signal_type = Column(
        Enum(SignalType, name="signal_type", create_constraint=True),
        nullable=False,
        comment="信号类型",
    )
    timestamp = Column(DateTime(timezone=True), nullable=False, comment="信号时间")
    price = Column(Numeric(15, 2), nullable=False, comment="信号价格")
    strength = Column(Float, nullable=False, server_default=text("0"), comment="信号强度")
    reason = Column(Text, nullable=True, comment="信号原因")
    signal_metadata = Column(JSONB, nullable=True, comment="元数据")
    executed = Column(Boolean, nullable=False, server_default=text("false"), comment="是否被执行")
    execution_reason = Column(
        Text, nullable=True, comment="未执行原因"
    )
    created_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        Index("ix_signal_backtest_time", "backtest_id", "timestamp"),
        Index("ix_signal_stock_time", "stock_code", "timestamp"),
    )

    def to_dict(self):
        execution_reason = None
        try:
            execution_reason = self.execution_reason
        except (AttributeError, KeyError):
            pass

        return {
            "id": str(self.id),
            "backtest_id": str(self.backtest_id),
            "signal_id": self.signal_id,
            "stock_code": self.stock_code,
            "stock_name": self.stock_name,
            "signal_type": self.signal_type.value if self.signal_type else None,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "price": float(self.price) if self.price else None,
            "strength": self.strength,
            "reason": self.reason,
            "metadata": self.signal_metadata,
            "executed": self.executed,
            "execution_reason": execution_reason,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class BacktestBenchmark(Base):
    """回测基准数据表"""

    __tablename__ = "backtest_benchmarks"

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    backtest_id = Column(
        UUID(as_uuid=True),
        ForeignKey("backtest_results.backtest_id", ondelete="CASCADE"),
        nullable=False,
    )
    benchmark_symbol = Column(String(20), nullable=False, comment="基准代码，如 000300.SH")
    benchmark_name = Column(String(100), nullable=False, comment="基准名称，如沪深300")
    benchmark_data = Column(JSONB, nullable=False, comment="基准历史数据")

    # 基准对比指标
    correlation = Column(Float, nullable=True, comment="相关系数")
    beta = Column(Float, nullable=True, comment="贝塔系数")
    alpha = Column(Float, nullable=True, comment="阿尔法系数")
    tracking_error = Column(Float, nullable=True, comment="跟踪误差")
    information_ratio = Column(Float, nullable=True, comment="信息比率")
    excess_return = Column(Float, nullable=True, comment="超额收益")

    created_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    __table_args__ = (
        Index("ix_benchmark_backtest_symbol", "backtest_id", "benchmark_symbol"),
    )

    def to_dict(self):
        return {
            "id": str(self.id),
            "backtest_id": str(self.backtest_id),
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

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    backtest_id = Column(
        UUID(as_uuid=True),
        ForeignKey("backtest_results.backtest_id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        comment="回测ID",
    )

    # ========== 信号统计 ==========
    total_signals = Column(Integer, server_default=text("0"), comment="总信号数")
    buy_signals = Column(Integer, server_default=text("0"), comment="买入信号数")
    sell_signals = Column(Integer, server_default=text("0"), comment="卖出信号数")
    executed_signals = Column(Integer, server_default=text("0"), comment="已执行信号数")
    unexecuted_signals = Column(Integer, server_default=text("0"), comment="未执行信号数")
    execution_rate = Column(Float, server_default=text("0"), comment="执行率")
    avg_signal_strength = Column(Float, server_default=text("0"), comment="平均信号强度")

    # ========== 交易统计 ==========
    total_trades = Column(Integer, server_default=text("0"), comment="总交易数")
    buy_trades = Column(Integer, server_default=text("0"), comment="买入交易数")
    sell_trades = Column(Integer, server_default=text("0"), comment="卖出交易数")
    winning_trades = Column(Integer, server_default=text("0"), comment="盈利交易数")
    losing_trades = Column(Integer, server_default=text("0"), comment="亏损交易数")
    win_rate = Column(Float, server_default=text("0"), comment="胜率")
    avg_profit = Column(Numeric(15, 2), server_default=text("0"), comment="平均盈利")
    avg_loss = Column(Numeric(15, 2), server_default=text("0"), comment="平均亏损")
    profit_factor = Column(Float, server_default=text("0"), comment="盈亏比")
    total_commission = Column(Numeric(15, 2), server_default=text("0"), comment="总手续费")
    total_pnl = Column(Numeric(15, 2), server_default=text("0"), comment="总盈亏")
    avg_holding_days = Column(Float, server_default=text("0"), comment="平均持仓天数")

    # ========== 持仓统计 ==========
    total_stocks = Column(Integer, server_default=text("0"), comment="总股票数")
    profitable_stocks = Column(Integer, server_default=text("0"), comment="盈利股票数")
    avg_stock_return = Column(Float, server_default=text("0"), comment="平均股票���益率")
    max_stock_return = Column(Float, nullable=True, comment="最大股票收益率")
    min_stock_return = Column(Float, nullable=True, comment="最小股票收益率")

    # ========== 时间范围统计 ==========
    first_signal_date = Column(DateTime(timezone=True), nullable=True, comment="第一个信号日期")
    last_signal_date = Column(DateTime(timezone=True), nullable=True, comment="最后一个信号日期")
    first_trade_date = Column(DateTime(timezone=True), nullable=True, comment="第一笔交易日期")
    last_trade_date = Column(DateTime(timezone=True), nullable=True, comment="最后一笔交易日期")
    trading_days = Column(Integer, server_default=text("0"), comment="交易天数")

    # ========== 股票分布统计 ==========
    unique_stocks_signaled = Column(Integer, server_default=text("0"), comment="产生信号的股票数")
    unique_stocks_traded = Column(Integer, server_default=text("0"), comment="实际交易的股票数")
    most_signaled_stock = Column(String(20), nullable=True, comment="信号最多的股票代码")
    most_traded_stock = Column(String(20), nullable=True, comment="交易最多的股票代码")

    # ========== 性能指标统计 ==========
    max_single_profit = Column(Numeric(15, 2), nullable=True, comment="单笔最大盈利")
    max_single_loss = Column(Numeric(15, 2), nullable=True, comment="单笔最大亏损")
    max_consecutive_wins = Column(Integer, server_default=text("0"), comment="最大连续盈利次数")
    max_consecutive_losses = Column(Integer, server_default=text("0"), comment="最大连续亏损次数")
    largest_position_size = Column(Numeric(15, 2), nullable=True, comment="最大持仓金额")

    # 元数据
    created_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    __table_args__ = (
        Index("uq_statistics_backtest_id", "backtest_id", unique=True),
    )

    def to_dict(self):
        """转换为字典格式"""
        return {
            "id": str(self.id),
            "backtest_id": str(self.backtest_id),
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
                "avg_profit": float(self.avg_profit) if self.avg_profit else None,
                "avg_loss": float(self.avg_loss) if self.avg_loss else None,
                "profit_factor": self.profit_factor,
                "total_commission": float(self.total_commission)
                if self.total_commission
                else None,
                "total_pnl": float(self.total_pnl) if self.total_pnl else None,
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
                "max_single_profit": float(self.max_single_profit)
                if self.max_single_profit
                else None,
                "max_single_loss": float(self.max_single_loss)
                if self.max_single_loss
                else None,
                "max_consecutive_wins": self.max_consecutive_wins,
                "max_consecutive_losses": self.max_consecutive_losses,
                "largest_position_size": float(self.largest_position_size)
                if self.largest_position_size
                else None,
            },
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
