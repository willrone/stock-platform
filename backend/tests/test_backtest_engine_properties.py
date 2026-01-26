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
from app.services.backtest import (
    BacktestConfig,
    BacktestExecutor,
    DataLoader,
    MACDStrategy,
    MovingAverageStrategy,
    PortfolioManager,
    RSIStrategy,
    SignalType,
    StrategyFactory,
    TradingSignal,
)


class TestBacktestEngineAccuracy:
    """属性 3: 回测引擎准确性测试"""
    
    def setup_method(self):
        """测试设置"""
        # 创建临时目录用于测试数据
        self.temp_dir = tempfile.mkdtemp()
        self.data_loader = DataLoader(self.temp_dir)
        self.backtest_executor = BacktestExecutor(self.temp_dir)
        
        # 创建测试数据目录结构
        os.makedirs(os.path.join(self.temp_dir, "daily", "000001.SZ"), exist_ok=True)
        
        # 创建模拟股票数据
        self.sample_data = self._create_sample_stock_data()
        
        # 保存测试数据
        data_path = os.path.join(self.temp_dir, "daily", "000001.SZ", "2024.parquet")
        self.sample_data.to_parquet(data_path)
    
    def _create_sample_stock_data(self, days: int = 252) -> pd.DataFrame:
        """创建样本股票数据"""
        dates = pd.date_range(start='2024-01-01', periods=days, freq='D')
        
        # 生成模拟价格数据
        np.random.seed(42)  # 确保可重现
        returns = np.random.normal(0.001, 0.02, days)
        prices = 100 * np.exp(np.cumsum(returns))
        
        data = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.005, days)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.01, days))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.01, days))),
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, days)
        }, index=dates)
        
        # 确保high >= close >= low
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
        """
        功能: production-ready-implementation, 属性 3: 回测引擎准确性
        验证策略信号生成的准确性 - 策略应该基于真实技术指标生成合理的交易信号
        """
        assume(short_window < long_window)  # 确保短期窗口小于长期窗口
        
        try:
            # 创建策略配置
            if strategy_name == 'moving_average':
                config = {
                    'short_window': short_window,
                    'long_window': long_window,
                    'signal_threshold': 0.02
                }
            elif strategy_name == 'rsi':
                config = {
                    'rsi_period': min(short_window + 5, 14),
                    'oversold_threshold': 30,
                    'overbought_threshold': 70
                }
            else:  # macd
                config = {
                    'fast_period': short_window,
                    'slow_period': long_window,
                    'signal_period': 9
                }
            
            # 创建策略
            strategy = StrategyFactory.create_strategy(strategy_name, config)
            
            # 验证策略属性
            assert strategy.name is not None
            assert strategy.config == config
            
            # 准备测试数据
            test_data = self.sample_data.copy()
            test_data.attrs['stock_code'] = '000001.SZ'
            
            # 生成信号
            current_date = test_data.index[-50]  # 使用倒数第50个交易日
            signals = strategy.generate_signals(test_data, current_date)
            
            # 验证信号格式
            assert isinstance(signals, list)
            
            for signal in signals:
                assert isinstance(signal, TradingSignal)
                assert signal.stock_code == '000001.SZ'
                assert signal.signal_type in [SignalType.BUY, SignalType.SELL, SignalType.HOLD]
                assert 0 <= signal.strength <= 1
                assert signal.price > 0
                assert isinstance(signal.reason, str)
                assert len(signal.reason) > 0
                assert signal.timestamp == current_date
            
            # 验证指标计算
            indicators = strategy.calculate_indicators(test_data)
            assert isinstance(indicators, dict)
            assert len(indicators) > 0
            
            # 验证指标数据类型和长度
            for name, indicator in indicators.items():
                assert isinstance(indicator, pd.Series)
                assert len(indicator) == len(test_data)
                # 验证没有全为NaN的指标
                assert not indicator.isnull().all()
                
        except Exception as e:
            # 如果是参数验证错误，这是可接受的
            if "不支持的策略" in str(e) or "参数" in str(e):
                pytest.skip(f"策略参数错误: {e}")
            else:
                raise
    
    @given(
        initial_cash=st.floats(min_value=50000, max_value=500000),
        commission_rate=st.floats(min_value=0.0001, max_value=0.01),
        max_position_size=st.floats(min_value=0.1, max_value=0.5)
    )
    @settings(max_examples=50)
    def test_portfolio_management_accuracy(self, initial_cash, commission_rate, max_position_size):
        """
        功能: production-ready-implementation, 属性 3: 回测引擎准确性
        验证组合管理的准确性 - 组合管理应该正确执行交易并计算盈亏
        """
        # 创建回测配置
        config = BacktestConfig(
            initial_cash=initial_cash,
            commission_rate=commission_rate,
            max_position_size=max_position_size
        )
        
        # 创建组合管理器
        portfolio_manager = PortfolioManager(config)
        
        # 验证初始状态
        assert portfolio_manager.cash == initial_cash
        assert len(portfolio_manager.positions) == 0
        assert len(portfolio_manager.trades) == 0
        
        # 创建测试信号
        current_date = datetime(2024, 6, 15)
        buy_signal = TradingSignal(
            timestamp=current_date,
            stock_code='000001.SZ',
            signal_type=SignalType.BUY,
            strength=0.8,
            price=100.0,
            reason="测试买入信号"
        )
        
        # 执行买入信号
        current_prices = {'000001.SZ': 100.0}
        trade, _ = portfolio_manager.execute_signal(buy_signal, current_prices)
        
        # 验证买入交易
        if trade:  # 如果有足够资金执行交易
            assert trade.action == "BUY"
            assert trade.stock_code == '000001.SZ'
            assert trade.quantity > 0
            assert trade.price > 0
            assert trade.commission >= 0
            
            # 验证持仓更新
            assert '000001.SZ' in portfolio_manager.positions
            position = portfolio_manager.positions['000001.SZ']
            assert position.quantity == trade.quantity
            assert position.avg_cost == trade.price
            
            # 验证现金减少
            expected_cash = initial_cash - (trade.quantity * trade.price + trade.commission)
            assert abs(portfolio_manager.cash - expected_cash) < 0.01
            
            # 验证组合价值计算
            portfolio_value = portfolio_manager.get_portfolio_value(current_prices)
            expected_value = portfolio_manager.cash + position.quantity * current_prices['000001.SZ']
            assert abs(portfolio_value - expected_value) < 0.01
            
            # 创建卖出信号
            sell_signal = TradingSignal(
                timestamp=current_date + timedelta(days=10),
                stock_code='000001.SZ',
                signal_type=SignalType.SELL,
                strength=0.7,
                price=105.0,
                reason="测试卖出信号"
            )
            
            # 执行卖出信号
            sell_prices = {'000001.SZ': 105.0}
            sell_trade, _ = portfolio_manager.execute_signal(sell_signal, sell_prices)
            
            if sell_trade:
                assert sell_trade.action == "SELL"
                assert sell_trade.quantity == position.quantity
                assert sell_trade.pnl != 0  # 应该有盈亏
                
                # 验证持仓清空
                assert '000001.SZ' not in portfolio_manager.positions
    
    @given(
        data_length=st.integers(min_value=100, max_value=300),
        strategy_name=st.sampled_from(['moving_average', 'rsi'])
    )
    @settings(max_examples=30)
    def test_backtest_execution_accuracy(self, data_length, strategy_name):
        """
        功能: production-ready-implementation, 属性 3: 回测引擎准确性
        验证回测执行的准确性 - 回测应该使用真实历史数据并产生合理结果
        """
        # 创建更长的测试数据
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
        
        # 保存扩展数据
        extended_data_path = os.path.join(self.temp_dir, "daily", "000001.SZ", "extended.parquet")
        extended_data.to_parquet(extended_data_path)
        
        try:
            # 配置策略参数
            if strategy_name == 'moving_average':
                strategy_config = {
                    'short_window': 5,
                    'long_window': 20,
                    'signal_threshold': 0.02
                }
            else:  # rsi
                strategy_config = {
                    'rsi_period': 14,
                    'oversold_threshold': 30,
                    'overbought_threshold': 70
                }
            
            # 验证回测参数
            start_date = dates[50]  # 留出足够的历史数据
            end_date = dates[-20]   # 留出一些缓冲
            
            is_valid = self.backtest_executor.validate_backtest_parameters(
                strategy_name, ['000001.SZ'], start_date, end_date, strategy_config
            )
            assert is_valid is True
            
            # 执行回测
            backtest_result = self.backtest_executor.run_backtest(
                strategy_name=strategy_name,
                stock_codes=['000001.SZ'],
                start_date=start_date,
                end_date=end_date,
                strategy_config=strategy_config
            )
            
            # 验证回测结果结构
            assert isinstance(backtest_result, dict)
            
            required_fields = [
                'strategy_name', 'stock_codes', 'start_date', 'end_date',
                'initial_cash', 'final_value', 'total_return', 'annualized_return',
                'volatility', 'sharpe_ratio', 'max_drawdown', 'total_trades',
                'win_rate', 'profit_factor', 'trade_history'
            ]
            
            for field in required_fields:
                assert field in backtest_result, f"缺少字段: {field}"
            
            # 验证数值合理性
            assert backtest_result['strategy_name'] == strategy_name
            assert backtest_result['stock_codes'] == ['000001.SZ']
            assert backtest_result['initial_cash'] > 0
            assert backtest_result['final_value'] > 0
            assert isinstance(backtest_result['total_return'], (int, float))
            assert isinstance(backtest_result['total_trades'], int)
            assert backtest_result['total_trades'] >= 0
            assert 0 <= backtest_result['win_rate'] <= 1
            
            # 验证交易历史
            trade_history = backtest_result['trade_history']
            assert isinstance(trade_history, list)
            
            for trade in trade_history:
                assert isinstance(trade, dict)
                assert 'trade_id' in trade
                assert 'stock_code' in trade
                assert 'action' in trade
                assert trade['action'] in ['BUY', 'SELL']
                assert 'quantity' in trade
                assert 'price' in trade
                assert trade['quantity'] > 0
                assert trade['price'] > 0
            
            # 验证组合历史
            portfolio_history = backtest_result.get('portfolio_history', [])
            assert isinstance(portfolio_history, list)
            
            for snapshot in portfolio_history:
                assert 'date' in snapshot
                assert 'portfolio_value' in snapshot
                assert 'cash' in snapshot
                assert snapshot['portfolio_value'] > 0
                assert snapshot['cash'] >= 0
            
        except Exception as e:
            # 如果是数据不足或参数问题，可以跳过
            if "数据不足" in str(e) or "交易日数量不足" in str(e):
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
        """
        功能: production-ready-implementation, 属性 3: 回测引擎准确性
        验证绩效指标计算的准确性 - 绩效指标应该基于真实交易数据正确计算
        """
        # 创建组合管理器
        config = BacktestConfig(initial_cash=100000)
        portfolio_manager = PortfolioManager(config)
        
        # 模拟组合价值历史
        initial_value = config.initial_cash
        portfolio_values = [initial_value]
        
        for daily_return in returns_data:
            new_value = portfolio_values[-1] * (1 + daily_return)
            portfolio_values.append(max(new_value, 1000))  # 确保价值不会变成负数
        
        # 创建组合历史快照
        base_date = datetime(2024, 1, 1)
        for i, value in enumerate(portfolio_values):
            snapshot_date = base_date + timedelta(days=i)
            snapshot = {
                'date': snapshot_date,
                'cash': value * 0.1,  # 假设10%现金
                'portfolio_value': value,
                'positions': {},
                'total_trades': i
            }
            portfolio_manager.portfolio_history.append(snapshot)
        
        # 计算绩效指标
        metrics = portfolio_manager.get_performance_metrics()
        
        if metrics:  # 如果有足够数据计算指标
            # 验证指标类型和合理性
            assert isinstance(metrics.get('total_return'), (int, float))
            assert isinstance(metrics.get('annualized_return'), (int, float))
            assert isinstance(metrics.get('volatility'), (int, float))
            assert isinstance(metrics.get('sharpe_ratio'), (int, float))
            assert isinstance(metrics.get('max_drawdown'), (int, float))
            
            # 验证指标范围
            assert metrics.get('volatility', 0) >= 0
            assert -1 <= metrics.get('max_drawdown', 0) <= 0  # 回撤应该是负数或0
            assert 0 <= metrics.get('win_rate', 0) <= 1
            assert metrics.get('total_trades', 0) >= 0
            
            # 验证总收益计算
            expected_total_return = (portfolio_values[-1] - initial_value) / initial_value
            actual_total_return = metrics.get('total_return', 0)
            assert abs(actual_total_return - expected_total_return) < 0.01  # 允许小误差
    
    @given(
        stock_count=st.integers(min_value=1, max_value=5),
        strategy_name=st.sampled_from(['moving_average', 'rsi'])
    )
    @settings(max_examples=20)
    def test_multi_stock_backtest_accuracy(self, stock_count, strategy_name):
        """
        功能: production-ready-implementation, 属性 3: 回测引擎准确性
        验证多股票回测的准确性 - 多股票回测应该正确处理多个股票的数据和信号
        """
        # 创建多个股票的测试数据
        stock_codes = [f"00000{i}.SZ" for i in range(1, stock_count + 1)]
        
        for stock_code in stock_codes:
            # 为每只股票创建目录和数据
            stock_dir = os.path.join(self.temp_dir, "daily", stock_code)
            os.makedirs(stock_dir, exist_ok=True)
            
            # 创建略有不同的价格数据
            dates = pd.date_range(start='2024-01-01', periods=200, freq='D')
            np.random.seed(hash(stock_code) % 2**32)  # 基于股票代码的种子
            
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
            
            # 保存数据
            data_path = os.path.join(stock_dir, "2024.parquet")
            stock_data.to_parquet(data_path)
        
        try:
            # 配置策略
            if strategy_name == 'moving_average':
                strategy_config = {
                    'short_window': 5,
                    'long_window': 20,
                    'signal_threshold': 0.02
                }
            else:  # rsi
                strategy_config = {
                    'rsi_period': 14,
                    'oversold_threshold': 30,
                    'overbought_threshold': 70
                }
            
            # 执行多股票回测
            start_date = datetime(2024, 2, 1)
            end_date = datetime(2024, 11, 1)
            
            backtest_result = self.backtest_executor.run_backtest(
                strategy_name=strategy_name,
                stock_codes=stock_codes,
                start_date=start_date,
                end_date=end_date,
                strategy_config=strategy_config
            )
            
            # 验证多股票回测结果
            assert backtest_result['stock_codes'] == stock_codes
            assert len(backtest_result['stock_codes']) == stock_count
            
            # 验证交易历史包含多只股票
            trade_history = backtest_result['trade_history']
            if trade_history:
                traded_stocks = set(trade['stock_code'] for trade in trade_history)
                # 至少应该有一只股票被交易（可能不是所有股票都会产生信号）
                assert len(traded_stocks) >= 1
                assert all(stock in stock_codes for stock in traded_stocks)
            
            # 验证组合历史
            portfolio_history = backtest_result.get('portfolio_history', [])
            if portfolio_history:
                # 验证组合价值的连续性
                values = [snapshot['portfolio_value'] for snapshot in portfolio_history]
                assert all(v > 0 for v in values)
                
        except Exception as e:
            if "数据加载失败" in str(e) or "交易日数量不足" in str(e):
                pytest.skip(f"多股票回测跳过: {e}")
            else:
                raise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])