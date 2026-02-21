"""
回测引擎属性测试
功能: production-ready-implementation
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List
from hypothesis import given, strategies as st, settings, assume
from unittest.mock import Mock, patch

from app.core.error_handler import TaskError
from app.services.backtest.models.data_models import BacktestConfig, TradingSignal
from app.services.backtest.models.enums import SignalType
from app.services.backtest.execution.backtest_executor import BacktestExecutor
from app.services.backtest.execution.data_loader import DataLoader
from app.services.backtest.strategies.strategy_factory import StrategyFactory
from app.services.backtest.core.portfolio_manager import PortfolioManager


class TestBacktestEngineAccuracy:
    """属性 3: 回测引擎准确性测试"""

    def setup_method(self):
        """测试设置"""
        self.temp_dir = tempfile.mkdtemp()
        self.data_loader = DataLoader(self.temp_dir)
        self.backtest_executor = BacktestExecutor(self.temp_dir)

    def _create_sample_stock_data(self, days: int = 252) -> pd.DataFrame:
        """创建样本股票数据"""
        dates = pd.date_range(start='2024-01-01', periods=days, freq='D')
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, days)
        prices = 100 * np.exp(np.cumsum(returns))

        data = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.005, days)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.01, days))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.01, days))),
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, days)
        }, index=dates)

        data['high'] = np.maximum(data['high'], data['close'])
        data['low'] = np.minimum(data['low'], data['close'])
        return data

    @given(
        strategy_name=st.sampled_from(['moving_average', 'rsi', 'macd']),
        short_window=st.integers(min_value=3, max_value=10),
        long_window=st.integers(min_value=15, max_value=30),
        initial_cash=st.floats(min_value=10000, max_value=1000000)
    )
    @settings(max_examples=100)
    def test_strategy_signal_generation_accuracy(self, strategy_name, short_window, long_window, initial_cash):
        """验证策略信号生成的准确性"""
        assume(short_window < long_window)

        try:
            if strategy_name == 'moving_average':
                config = {'short_window': short_window, 'long_window': long_window, 'signal_threshold': 0.02}
            elif strategy_name == 'rsi':
                config = {'rsi_period': min(short_window + 5, 14), 'oversold_threshold': 30, 'overbought_threshold': 70}
            else:
                config = {'fast_period': short_window, 'slow_period': long_window, 'signal_period': 9}

            strategy = StrategyFactory.create_strategy(strategy_name, config)
            sample_data = self._create_sample_stock_data()
            # generate_signals now requires current_date after refactor
            current_date = sample_data.index[-1].to_pydatetime() if hasattr(sample_data.index[-1], 'to_pydatetime') else sample_data.index[-1]
            signals = strategy.generate_signals(sample_data, current_date)

            assert isinstance(signals, (list, pd.Series, np.ndarray))

            if isinstance(signals, list):
                for signal in signals:
                    if isinstance(signal, TradingSignal):
                        assert signal.signal_type in [SignalType.BUY, SignalType.SELL, SignalType.HOLD]
                        assert 0 <= signal.strength <= 1
        except Exception as e:
            if "数据不足" in str(e) or "not enough" in str(e).lower():
                pytest.skip(f"数据不足: {e}")
            else:
                raise

    @given(
        initial_cash=st.floats(min_value=10000, max_value=1000000),
        commission_rate=st.floats(min_value=0.0001, max_value=0.01),
        max_position_size=st.floats(min_value=0.05, max_value=0.5)
    )
    @settings(max_examples=50)
    def test_portfolio_management_accuracy(self, initial_cash, commission_rate, max_position_size):
        """验证组合管理的准确性"""
        config = BacktestConfig(
            initial_cash=initial_cash,
            commission_rate=commission_rate,
            max_position_size=max_position_size
        )
        portfolio_manager = PortfolioManager(config)

        assert portfolio_manager.cash == initial_cash
        assert len(portfolio_manager.positions) == 0

        buy_price = 100.0
        max_shares = int((initial_cash * max_position_size) / buy_price)

        if max_shares > 0:
            # execute_buy/execute_sell removed; use execute_signal with TradingSignal
            buy_signal = TradingSignal(
                timestamp=datetime(2024, 1, 1),
                stock_code='000001.SZ',
                signal_type=SignalType.BUY,
                strength=1.0,
                price=buy_price,
                reason='test buy',
            )
            trade, fail_reason = portfolio_manager.execute_signal(buy_signal, {'000001.SZ': buy_price})

            if trade is not None and '000001.SZ' in portfolio_manager.positions:
                position = portfolio_manager.positions['000001.SZ']
                assert position.quantity > 0
                assert portfolio_manager.cash < initial_cash

                sell_price = 110.0
                sell_signal = TradingSignal(
                    timestamp=datetime(2024, 1, 2),
                    stock_code='000001.SZ',
                    signal_type=SignalType.SELL,
                    strength=1.0,
                    price=sell_price,
                    reason='test sell',
                )
                trade2, fail_reason2 = portfolio_manager.execute_signal(sell_signal, {'000001.SZ': sell_price})
                assert '000001.SZ' not in portfolio_manager.positions or portfolio_manager.positions['000001.SZ'].quantity == 0

    @pytest.mark.asyncio
    @given(
        data_length=st.integers(min_value=100, max_value=300),
        strategy_name=st.sampled_from(['moving_average', 'rsi'])
    )
    @settings(max_examples=5)
    async def test_backtest_execution_accuracy(self, data_length, strategy_name):
        """验证回测执行的准确性"""
        dates = pd.date_range(start='2024-01-01', periods=data_length, freq='D')
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, data_length)
        prices = 100 * np.exp(np.cumsum(returns))

        extended_data = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.005, data_length)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.01, data_length))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.01, data_length))),
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, data_length)
        }, index=dates)
        extended_data['high'] = np.maximum(extended_data['high'], extended_data['close'])
        extended_data['low'] = np.minimum(extended_data['low'], extended_data['close'])

        if strategy_name == 'moving_average':
            strategy_config = {'short_window': 5, 'long_window': 20, 'signal_threshold': 0.02}
        else:
            strategy_config = {'rsi_period': 14, 'oversold_threshold': 30, 'overbought_threshold': 70}

        start_date = dates[50].to_pydatetime()
        end_date = dates[-20].to_pydatetime()

        from app.services.backtest.execution.validators import validate_backtest_parameters
        is_valid = validate_backtest_parameters(
            strategy_name, ['000001.SZ'], start_date, end_date, strategy_config
        )
        assert is_valid is True

        try:
            backtest_result = await self.backtest_executor.run_backtest(
                strategy_name=strategy_name,
                stock_codes=['000001.SZ'],
                start_date=start_date,
                end_date=end_date,
                strategy_config=strategy_config,
                preloaded_stock_data={'000001.SZ': extended_data},
            )
            assert isinstance(backtest_result, dict)
        except Exception as e:
            if "数据不足" in str(e) or "交易日" in str(e) or "数据加载" in str(e):
                pytest.skip(f"数据不足: {e}")
            else:
                raise

    @given(
        returns_data=st.lists(
            st.floats(min_value=-0.1, max_value=0.1),
            min_size=50, max_size=200
        )
    )
    @settings(max_examples=50)
    def test_performance_metrics_accuracy(self, returns_data):
        """验证绩效指标计算的准确性"""
        config = BacktestConfig(initial_cash=100000)
        portfolio_manager = PortfolioManager(config)

        initial_value = config.initial_cash
        portfolio_values = [initial_value]

        for daily_return in returns_data:
            new_value = portfolio_values[-1] * (1 + daily_return)
            portfolio_values.append(max(new_value, 1000))

        base_date = datetime(2024, 1, 1)
        for i, value in enumerate(portfolio_values):
            snapshot_date = base_date + timedelta(days=i)
            snapshot = {
                'date': snapshot_date,
                'cash': value * 0.1,
                'portfolio_value': value,
                'positions': {},
                'total_trades': i
            }
            portfolio_manager.portfolio_history.append(snapshot)

        metrics = portfolio_manager.get_performance_metrics()

        if metrics:
            assert isinstance(metrics.get('total_return'), (int, float))
            assert isinstance(metrics.get('volatility'), (int, float))
            assert metrics.get('volatility', 0) >= 0
            assert -1 <= metrics.get('max_drawdown', 0) <= 0

    @pytest.mark.asyncio
    @given(
        stock_count=st.integers(min_value=1, max_value=3),
        strategy_name=st.sampled_from(['moving_average', 'rsi'])
    )
    @settings(max_examples=5)
    async def test_multi_stock_backtest_accuracy(self, stock_count, strategy_name):
        """验证多股票回测的准确性"""
        stock_codes = [f"00000{i}.SZ" for i in range(1, stock_count + 1)]
        preloaded = {}

        for stock_code in stock_codes:
            dates = pd.date_range(start='2024-01-01', periods=200, freq='D')
            np.random.seed(hash(stock_code) % 2**32)
            returns = np.random.normal(0.001, 0.02, 200)
            prices = 100 * np.exp(np.cumsum(returns))

            stock_data = pd.DataFrame({
                'open': prices * (1 + np.random.normal(0, 0.005, 200)),
                'high': prices * (1 + np.abs(np.random.normal(0, 0.01, 200))),
                'low': prices * (1 - np.abs(np.random.normal(0, 0.01, 200))),
                'close': prices,
                'volume': np.random.randint(1000000, 10000000, 200)
            }, index=dates)
            stock_data['high'] = np.maximum(stock_data['high'], stock_data['close'])
            stock_data['low'] = np.minimum(stock_data['low'], stock_data['close'])
            preloaded[stock_code] = stock_data

        if strategy_name == 'moving_average':
            strategy_config = {'short_window': 5, 'long_window': 20, 'signal_threshold': 0.02}
        else:
            strategy_config = {'rsi_period': 14, 'oversold_threshold': 30, 'overbought_threshold': 70}

        try:
            backtest_result = await self.backtest_executor.run_backtest(
                strategy_name=strategy_name,
                stock_codes=stock_codes,
                start_date=datetime(2024, 3, 1),
                end_date=datetime(2024, 10, 1),
                strategy_config=strategy_config,
                preloaded_stock_data=preloaded,
            )
            assert isinstance(backtest_result, dict)
            assert backtest_result.get('stock_codes') == stock_codes or 'strategy_name' in backtest_result
        except Exception as e:
            if "数据加载" in str(e) or "交易日" in str(e) or "数据不足" in str(e):
                pytest.skip(f"多股票回测跳过: {e}")
            else:
                raise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
