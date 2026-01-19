"""
回测详细结果数据模型
用于存储可视化所需的扩展数据
"""

from datetime import datetime
from typing import Optional, Dict, Any
from sqlalchemy import Column, Integer, String, DateTime, Float, Text, JSON, Boolean, ForeignKey, Index
from sqlalchemy.orm import relationship
import uuid

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
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # 创建索引
    __table_args__ = (
        Index('idx_backtest_detailed_task_id', 'task_id'),
        Index('idx_backtest_detailed_backtest_id', 'backtest_id'),
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
                "downside_deviation": self.downside_deviation
            },
            "drawdown_analysis": self.drawdown_analysis,
            "monthly_returns": self.monthly_returns,
            "position_analysis": self.position_analysis,
            "benchmark_comparison": self.benchmark_comparison,
            "rolling_metrics": self.rolling_metrics,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }


class BacktestChartCache(Base):
    """回测图表数据缓存表（用于提高图表加载性能）"""
    __tablename__ = "backtest_chart_cache"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(String(50), nullable=False)
    chart_type = Column(String(50), nullable=False, comment="图表类型：equity_curve, drawdown_curve, monthly_heatmap等")
    chart_data = Column(JSON, nullable=False, comment="图表数据JSON")
    data_hash = Column(String(64), nullable=True, comment="数据哈希值，用于检测数据变化")
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True, comment="缓存过期时间")
    
    # 创建唯一索引和其他索引
    __table_args__ = (
        Index('uk_task_chart', 'task_id', 'chart_type', unique=True),
        Index('idx_chart_cache_expires', 'expires_at'),
        Index('idx_chart_cache_task_id', 'task_id'),
    )
    
    def to_dict(self):
        return {
            "id": self.id,
            "task_id": self.task_id,
            "chart_type": self.chart_type,
            "chart_data": self.chart_data,
            "data_hash": self.data_hash,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None
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
        Index('idx_portfolio_task_date', 'task_id', 'snapshot_date'),
        Index('idx_portfolio_backtest_date', 'backtest_id', 'snapshot_date'),
    )
    
    def to_dict(self):
        return {
            "id": self.id,
            "task_id": self.task_id,
            "backtest_id": self.backtest_id,
            "snapshot_date": self.snapshot_date.isoformat() if self.snapshot_date else None,
            "portfolio_value": self.portfolio_value,
            "cash": self.cash,
            "positions_count": self.positions_count,
            "total_return": self.total_return,
            "drawdown": self.drawdown,
            "positions": self.positions,
            "created_at": self.created_at.isoformat() if self.created_at else None
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
        Index('idx_trade_task_stock', 'task_id', 'stock_code'),
        Index('idx_trade_backtest_time', 'backtest_id', 'timestamp'),
        Index('idx_trade_stock_time', 'stock_code', 'timestamp'),
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
            "created_at": self.created_at.isoformat() if self.created_at else None
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
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    # 创建索引
    __table_args__ = (
        Index('idx_signal_task_stock', 'task_id', 'stock_code'),
        Index('idx_signal_backtest_time', 'backtest_id', 'timestamp'),
        Index('idx_signal_stock_time', 'stock_code', 'timestamp'),
        Index('idx_signal_type', 'signal_type'),
        Index('idx_signal_executed', 'executed'),
    )
    
    def to_dict(self):
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
            "created_at": self.created_at.isoformat() if self.created_at else None
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
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # 创建索引
    __table_args__ = (
        Index('idx_benchmark_task_symbol', 'task_id', 'benchmark_symbol'),
        Index('idx_benchmark_backtest_id', 'backtest_id'),
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
                "excess_return": self.excess_return
            },
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }