"""
回测引擎服务测试

测试策略回测和性能指标计算功能。
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app', 'services'))

from backtest_engine import (
    BacktestService,
    BacktestConfig,
    BacktestResult,
    Trade,
    OrderType,
    SimpleBacktestEngine
)


class TestBacktestConfig:
    """回测配置测试"""
    
    def test_backtest_config_creation(self):
        """测试回测配置创建"""
        config = BacktestConfig(
            initial_cash=50000.0,
            commission=0.002,
            max_position_size=0.3
        )
        
        assert config.initial_cash == 50000.0
        assert config.commission == 0.002
        assert config.max_position_size == 0.3
    
    def test_backtest_config_defaults(self):
        """测试回测配置默认值"""
        config = BacktestConfig()
        
        assert config.initial_cash == 100000.0
        assert config.commission == 0.001
        assert config.max_position_size == 0.2


class TestTrade:
    """交易记录测试"""
    
    def test_trade_creation(self):
        """测试交易记录创建"""
        trade = Trade(
            stock_code="000001.SZ",
            entry_date=datetime(2023, 1, 1),
            exit_date=datetime(2023, 1, 5),
            entry_price=10.0,
            exit_price=11.0,
            quantity=1000,
            order_type=OrderType.SELL,
            pnl=1000.0,
            pnl_pct=0.1,
            commission=10.0,
            duration_days=4
        )
        
        assert trade.stock_code == "000001.SZ"
        assert trade.pnl == 1000.0
        assert trade.pnl_pct == 0.1
        assert trade.duration_days == 4


class TestBacktestResult:
    """回测结果测试"""
    
    def test_backtest_result_to_dict(self):
        """测试回测结果转换为字典"""
        result = BacktestResult(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
            initial_cash=100000.0,
            final_value=120000.0,
            total_return=0.2,
            annualized_return=0.2,
            max_drawdown=-0.1,
            sharpe_ratio=1.5,
            calmar_ratio=2.0,
            total_trades=10,
            winning_trades=6,
            losing_trades=4,
            win_rate=0.6,
            avg_win=2000.0,
            avg_loss=-1000.0,
            profit_factor=2.0,
            volatility=0.15,
            var_95=-0.02,
            max_consecutive_losses=2,
            trades=[],
            daily_returns=pd.Series([0.01, 0.02, -0.01]),
            cumulative_returns=pd.Series([0.01, 0.03, 0.02])
        )
        
        result_dict = result.to_dict()
        
        assert result_dict["portfolio"]["total_return"] == 0.2
        assert result_dict["risk_metrics"]["sharpe_ratio"] == 1.5
        assert result_dict["trading_stats"]["win_rate"] == 0.6
        assert "period" in result_dict
        assert "portfolio" in result_dict
        assert "risk_metrics" in result_dict
        assert "trading_stats" in result_dict


class TestSimpleBacktestEngine:
    """简单回测引擎测试"""
    
    def test_backtest_engine_creation(self):
        """测试回测引擎创建"""
        config = BacktestConfig()
        engine = SimpleBacktestEngine(config)
        
        assert engine.config == config
        assert engine.cash == config.initial_cash
        assert engine.positions == {}
        assert engine.trades == []
    
    def test_run_backtest_basic(self):
        """测试基本回测运行"""
        config = BacktestConfig(initial_cash=10000.0)
        engine = SimpleBacktestEngine(config)
        
        # 创建测试数据
        dates = [datetime(2023, 1, 1), datetime(2023, 1, 2), datetime(2023, 1, 3)]
        price_data = pd.DataFrame({
            'date': dates * 2,
            'stock_code': ['000001.SZ'] * 3 + ['000002.SZ'] * 3,
            'open': [10.0, 10.1, 10.2, 20.0, 20.1, 20.2],
            'high': [10.2, 10.3, 10.4, 20.2, 20.3, 20.4],
            'low': [9.8, 9.9, 10.0, 19.8, 19.9, 20.0],
            'close': [10.0, 10.1, 10.2, 20.0, 20.1, 20.2],
            'volume': [1000000] * 6
        })
        
        signals = pd.DataFrame({
            'date': dates * 2,
            'stock_code': ['000001.SZ'] * 3 + ['000002.SZ'] * 3,
            'signal': [1, 0, -1, 1, 0, -1]  # 买入, 持有, 卖出
        })
        
        result = engine.run_backtest(price_data, signals)
        
        assert isinstance(result, BacktestResult)
        assert result.initial_cash == 10000.0
        assert result.start_date == datetime(2023, 1, 1)
        assert result.end_date == datetime(2023, 1, 3)
    
    def test_buy_stock(self):
        """测试买入股票"""
        config = BacktestConfig(initial_cash=10000.0, max_position_size=0.5)
        engine = SimpleBacktestEngine(config)
        
        # 执行买入
        engine._buy_stock("000001.SZ", 10.0, datetime(2023, 1, 1))
        
        # 检查结果
        assert "000001.SZ" in engine.positions
        assert engine.positions["000001.SZ"] > 0
        assert engine.cash < 10000.0  # 现金应该减少
    
    def test_sell_stock(self):
        """测试卖出股票"""
        config = BacktestConfig(initial_cash=10000.0)
        engine = SimpleBacktestEngine(config)
        
        # 先设置持仓
        engine.positions["000001.SZ"] = 100
        initial_cash = engine.cash
        
        # 执行卖出
        engine._sell_stock("000001.SZ", 12.0, datetime(2023, 1, 2))
        
        # 检查结果
        assert engine.positions["000001.SZ"] == 0
        assert engine.cash > initial_cash  # 现金应该增加
        assert len(engine.trades) == 1  # 应该有一笔交易记录


class TestBacktestService:
    """回测服务测试"""
    
    def test_backtest_service_creation(self):
        """测试回测服务创建"""
        service = BacktestService()
        
        assert service.default_config is not None
        assert isinstance(service.default_config, BacktestConfig)
    
    @pytest.mark.asyncio
    async def test_run_strategy_backtest(self):
        """测试运行策略回测"""
        service = BacktestService()
        
        result = await service.run_strategy_backtest(
            strategy_name="test_strategy",
            stock_codes=["000001.SZ", "000002.SZ"],
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 10),
            config=BacktestConfig(initial_cash=50000.0)
        )
        
        assert isinstance(result, BacktestResult)
        assert result.initial_cash == 50000.0
        assert result.start_date == datetime(2023, 1, 1)
        assert result.end_date == datetime(2023, 1, 10)
    
    def test_generate_mock_price_data(self):
        """测试生成模拟价格数据"""
        service = BacktestService()
        
        price_data = service._generate_mock_price_data(
            stock_codes=["000001.SZ", "000002.SZ"],
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 3)
        )
        
        assert len(price_data) == 6  # 2股票 * 3天
        assert list(price_data.columns) == ['date', 'stock_code', 'open', 'high', 'low', 'close', 'volume']
        assert price_data['stock_code'].nunique() == 2
    
    def test_generate_mock_signals(self):
        """测试生成模拟交易信号"""
        service = BacktestService()
        
        signals = service._generate_mock_signals(
            stock_codes=["000001.SZ"],
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 3)
        )
        
        assert len(signals) == 3  # 1股票 * 3天
        assert list(signals.columns) == ['date', 'stock_code', 'signal']
        assert all(signals['signal'].isin([0, 1, -1]))


# 运行测试的示例
if __name__ == "__main__":
    import asyncio
    
    async def run_basic_tests():
        """运行基本测试"""
        print("开始回测引擎功能测试...")
        
        # 测试回测配置
        config = BacktestConfig()
        print(f"✓ 回测配置创建成功: 初始资金={config.initial_cash}")
        
        # 测试回测引擎
        engine = SimpleBacktestEngine(config)
        print("✓ 回测引擎创建成功")
        
        # 测试回测服务
        service = BacktestService()
        print("✓ 回测服务创建成功")
        
        # 运行简单回测
        result = await service.run_strategy_backtest(
            strategy_name="test_strategy",
            stock_codes=["000001.SZ"],
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 5)
        )
        print(f"✓ 策略回测完成: 总收益={result.total_return:.4f}")
        
        print("所有基本测试通过！")
    
    asyncio.run(run_basic_tests())