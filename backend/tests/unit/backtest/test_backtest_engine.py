"""
回测引擎服务测试

测试策略回测和性能指标计算功能。
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from app.services.backtest.models.data_models import BacktestConfig, Trade
from app.services.backtest.models.enums import OrderType, SignalType
from app.services.backtest.execution.backtest_executor import BacktestExecutor
from app.services.backtest.execution.validators import validate_backtest_parameters


class TestBacktestConfig:
    """回测配置测试"""

    def test_backtest_config_creation(self):
        """测试回测配置创建"""
        config = BacktestConfig(
            initial_cash=50000.0,
            commission_rate=0.002,
            max_position_size=0.3
        )
        assert config.initial_cash == 50000.0
        assert config.commission_rate == 0.002
        assert config.max_position_size == 0.3

    def test_backtest_config_defaults(self):
        """测试回测配置默认值"""
        config = BacktestConfig()
        assert config.initial_cash == 100000.0
        assert config.commission_rate == 0.001
        assert config.max_position_size == 0.2


class TestTrade:
    """交易记录测试"""

    def test_trade_creation(self):
        """测试交易记录创建"""
        trade = Trade(
            trade_id="t001",
            stock_code="000001.SZ",
            action="SELL",
            quantity=1000,
            price=11.0,
            timestamp=datetime(2023, 1, 5),
            commission=10.0,
            slippage_cost=5.0,
            pnl=1000.0,
            cumulative_pnl=1000.0,
        )
        assert trade.stock_code == "000001.SZ"
        assert trade.pnl == 1000.0
        assert trade.action == "SELL"
        assert trade.commission == 10.0


class TestOrderType:
    """订单类型测试"""

    def test_order_types(self):
        assert OrderType.MARKET.value == "market"
        assert OrderType.LIMIT.value == "limit"
        assert OrderType.STOP.value == "stop"


class TestBacktestExecutor:
    """回测执行器测试"""

    def test_backtest_executor_creation(self):
        """测试回测执行器创建"""
        executor = BacktestExecutor()
        assert executor.enable_parallel is True
        assert executor.execution_stats["total_backtests"] == 0

    def test_validate_backtest_parameters(self):
        """测试回测参数验证"""
        result = validate_backtest_parameters(
            strategy_name="moving_average",
            stock_codes=["000001.SZ"],
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
            strategy_config={"short_window": 5, "long_window": 20},
        )
        assert result is True

    def test_get_execution_statistics(self):
        """测试获取执行统计"""
        executor = BacktestExecutor()
        stats = executor.get_execution_statistics()
        assert "total_backtests" in stats
        assert "successful_backtests" in stats
        assert "failed_backtests" in stats

    @pytest.mark.asyncio
    async def test_run_backtest_returns_dict(self):
        """测试 run_backtest 返回字典"""
        executor = BacktestExecutor()

        # Create minimal mock data with preloaded_stock_data to skip file I/O
        dates = pd.date_range('2023-01-01', periods=60, freq='B')
        mock_data = {}
        for code in ['000001.SZ']:
            np.random.seed(42)
            prices = 10 + np.cumsum(np.random.normal(0, 0.1, len(dates)))
            prices = np.abs(prices) + 5  # ensure positive
            df = pd.DataFrame({
                'open': prices * 0.99,
                'high': prices * 1.02,
                'low': prices * 0.98,
                'close': prices,
                'volume': np.random.randint(100000, 1000000, len(dates)),
            }, index=dates)
            mock_data[code] = df

        result = await executor.run_backtest(
            strategy_name="moving_average",
            stock_codes=["000001.SZ"],
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 3, 31),
            strategy_config={"short_window": 5, "long_window": 20},
            backtest_config=BacktestConfig(initial_cash=50000.0),
            preloaded_stock_data=mock_data,
        )
        assert isinstance(result, dict)
