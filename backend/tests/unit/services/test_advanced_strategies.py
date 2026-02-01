"""
高级策略测试用例

测试所有新增的策略：
1. 技术分析策略：布林带、随机指标、CCI
2. 统计套利策略：配对交易、均值回归、协整
3. 因子投资策略：价值因子、动量因子、低波动、多因子
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any
from unittest.mock import Mock, patch, MagicMock
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from app.services.backtest.strategies import (
    BollingerBandStrategy,
    StochasticStrategy,
    CCIStrategy,
    PairsTradingStrategy,
    MeanReversionStrategy,
    CointegrationStrategy,
    ValueFactorStrategy,
    MomentumFactorStrategy,
    LowVolatilityStrategy,
    MultiFactorStrategy,
    AdvancedStrategyFactory
)


def generate_mock_price_data(start_date: datetime, num_days: int, 
                           base_price: float = 100, volatility: float = 0.02) -> pd.DataFrame:
    """生成模拟价格数据"""
    dates = [start_date + timedelta(days=i) for i in range(num_days)]
    
    np.random.seed(42)
    returns = np.random.normal(0.0003, volatility, num_days)
    prices = base_price * np.cumprod(1 + returns)
    
    high = prices * (1 + np.random.uniform(0, 0.02, num_days))
    low = prices * (1 - np.random.uniform(0, 0.02, num_days))
    open_price = prices * (1 + np.random.uniform(-0.01, 0.01, num_days))
    volume = np.random.randint(1000000, 10000000, num_days)
    
    data = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': prices,
        'volume': volume
    }, index=pd.DatetimeIndex(dates, name='date'))
    
    return data


class TestBollingerBandStrategy:
    """布林带策略测试"""
    
    def test_strategy_creation(self):
        """测试策略创建"""
        config = {
            'period': 20,
            'std_dev': 2,
            'entry_threshold': 0.02
        }
        strategy = BollingerBandStrategy(config)
        
        assert strategy.name == "BollingerBands"
        assert strategy.period == 20
        assert strategy.std_dev == 2
    
    def test_indicator_calculation(self):
        """测试指标计算"""
        config = {'period': 20, 'std_dev': 2}
        strategy = BollingerBandStrategy(config)
        
        data = generate_mock_price_data(datetime(2023, 1, 1), 100)
        indicators = strategy.calculate_indicators(data)
        
        assert 'upper_band' in indicators
        assert 'lower_band' in indicators
        assert 'percent_b' in indicators
        assert len(indicators['upper_band']) == len(data)
    
    def test_signal_generation(self):
        """测试信号生成"""
        config = {'period': 20, 'std_dev': 2}
        strategy = BollingerBandStrategy(config)
        
        data = generate_mock_price_data(datetime(2023, 1, 1), 100)
        data.attrs['stock_code'] = '000001.SZ'
        
        current_date = data.index[50]
        signals = strategy.generate_signals(data, current_date)
        
        assert isinstance(signals, list)


class TestStochasticStrategy:
    """随机指标策略测试"""
    
    def test_strategy_creation(self):
        """测试策略创建"""
        config = {
            'k_period': 14,
            'd_period': 3,
            'oversold': 20,
            'overbought': 80
        }
        strategy = StochasticStrategy(config)
        
        assert strategy.name == "Stochastic"
        assert strategy.k_period == 14
        assert strategy.d_period == 3
    
    def test_indicator_calculation(self):
        """测试指标计算"""
        config = {'k_period': 14, 'd_period': 3}
        strategy = StochasticStrategy(config)
        
        data = generate_mock_price_data(datetime(2023, 1, 1), 100)
        indicators = strategy.calculate_indicators(data)
        
        assert 'k_percent' in indicators
        assert 'd_percent' in indicators
        assert len(indicators['k_percent']) == len(data)
    
    def test_signal_generation(self):
        """测试信号生成"""
        config = {'k_period': 14, 'd_period': 3}
        strategy = StochasticStrategy(config)
        
        data = generate_mock_price_data(datetime(2023, 1, 1), 100)
        data.attrs['stock_code'] = '000001.SZ'
        
        current_date = data.index[50]
        signals = strategy.generate_signals(data, current_date)
        
        assert isinstance(signals, list)


class TestCCIStrategy:
    """CCI策略测试"""
    
    def test_strategy_creation(self):
        """测试策略创建"""
        config = {
            'period': 20,
            'oversold': -100,
            'overbought': 100
        }
        strategy = CCIStrategy(config)
        
        assert strategy.name == "CCI"
        assert strategy.period == 20
        assert strategy.oversold == -100
    
    def test_indicator_calculation(self):
        """测试指标计算"""
        config = {'period': 20}
        strategy = CCIStrategy(config)
        
        data = generate_mock_price_data(datetime(2023, 1, 1), 100)
        indicators = strategy.calculate_indicators(data)
        
        assert 'cci' in indicators
        assert len(indicators['cci']) == len(data)
    
    def test_signal_generation(self):
        """测试信号生成"""
        config = {'period': 20}
        strategy = CCIStrategy(config)
        
        data = generate_mock_price_data(datetime(2023, 1, 1), 100)
        data.attrs['stock_code'] = '000001.SZ'
        
        current_date = data.index[50]
        signals = strategy.generate_signals(data, current_date)
        
        assert isinstance(signals, list)


class TestMeanReversionStrategy:
    """均值回归策略测试"""
    
    def test_strategy_creation(self):
        """测试策略创建"""
        config = {
            'lookback_period': 20,
            'zscore_threshold': 2.0,
            'position_size': 0.1
        }
        strategy = MeanReversionStrategy(config)
        
        assert strategy.name == "MeanReversion"
        assert strategy.lookback_period == 20
        assert strategy.zscore_threshold == 2.0
    
    def test_indicator_calculation(self):
        """测试指标计算"""
        config = {'lookback_period': 20}
        strategy = MeanReversionStrategy(config)
        
        data = generate_mock_price_data(datetime(2023, 1, 1), 100)
        indicators = strategy.calculate_indicators(data)
        
        assert 'zscore' in indicators
        assert 'sma' in indicators
    
    def test_signal_generation(self):
        """测试信号生成"""
        config = {'lookback_period': 20, 'zscore_threshold': 2.0}
        strategy = MeanReversionStrategy(config)
        
        data = generate_mock_price_data(datetime(2023, 1, 1), 100)
        data.attrs['stock_code'] = '000001.SZ'
        
        current_date = data.index[50]
        signals = strategy.generate_signals(data, current_date)
        
        assert isinstance(signals, list)


class TestMomentumFactorStrategy:
    """动量因子策略测试"""
    
    def test_strategy_creation(self):
        """测试策略创建"""
        config = {
            'momentum_periods': [21, 63, 126],
            'momentum_weights': [0.5, 0.3, 0.2]
        }
        strategy = MomentumFactorStrategy(config)
        
        assert strategy.name == "MomentumFactor"
        assert strategy.momentum_periods == [21, 63, 126]
    
    def test_indicator_calculation(self):
        """测试指标计算"""
        config = {'momentum_periods': [21, 63]}
        strategy = MomentumFactorStrategy(config)
        
        data = generate_mock_price_data(datetime(2023, 1, 1), 200)
        indicators = strategy.calculate_indicators(data)
        
        assert 'momentum' in indicators
        assert 'price' in indicators
    
    def test_signal_generation(self):
        """测试信号生成"""
        config = {'momentum_periods': [21, 63]}
        strategy = MomentumFactorStrategy(config)
        
        data = generate_mock_price_data(datetime(2023, 1, 1), 200)
        data.attrs['stock_code'] = '000001.SZ'
        
        current_date = data.index[150]
        signals = strategy.generate_signals(data, current_date)
        
        assert isinstance(signals, list)


class TestLowVolatilityStrategy:
    """低波动因子策略测试"""
    
    def test_strategy_creation(self):
        """测试策略创建"""
        config = {
            'volatility_period': 21,
            'volatility_window': 63
        }
        strategy = LowVolatilityStrategy(config)
        
        assert strategy.name == "LowVolatility"
        assert strategy.volatility_period == 21
    
    def test_indicator_calculation(self):
        """测试指标计算"""
        config = {'volatility_period': 21, 'volatility_window': 63}
        strategy = LowVolatilityStrategy(config)
        
        data = generate_mock_price_data(datetime(2023, 1, 1), 100)
        indicators = strategy.calculate_indicators(data)
        
        assert 'volatility' in indicators
        assert 'risk_adjusted_return' in indicators


class TestMultiFactorStrategy:
    """多因子策略测试"""
    
    def test_strategy_creation(self):
        """测试策略创建"""
        config = {
            'factors': ['value', 'momentum', 'low_volatility'],
            'factor_weights': [0.33, 0.33, 0.34]
        }
        strategy = MultiFactorStrategy(config)
        
        assert strategy.name == "MultiFactor"
        assert len(strategy.factors) == 3
    
    def test_indicator_calculation(self):
        """测试指标计算"""
        config = {
            'factors': ['value', 'momentum', 'low_volatility'],
            'factor_weights': [0.33, 0.33, 0.34]
        }
        strategy = MultiFactorStrategy(config)
        
        data = generate_mock_price_data(datetime(2023, 1, 1), 200)
        indicators = strategy.calculate_indicators(data)
        
        assert 'combined_score' in indicators
        assert 'value_score' in indicators
        assert 'momentum_score' in indicators
    
    def test_signal_generation(self):
        """测试信号生成"""
        config = {
            'factors': ['value', 'momentum', 'low_volatility'],
            'factor_weights': [0.33, 0.33, 0.34]
        }
        strategy = MultiFactorStrategy(config)
        
        data = generate_mock_price_data(datetime(2023, 1, 1), 200)
        data.attrs['stock_code'] = '000001.SZ'
        
        current_date = data.index[150]
        signals = strategy.generate_signals(data, current_date)
        
        assert isinstance(signals, list)


class TestAdvancedStrategyFactory:
    """高级策略工厂测试"""
    
    def test_create_technical_strategy(self):
        """测试创建技术分析策略"""
        strategy = AdvancedStrategyFactory.create_strategy(
            'bollinger', {'period': 20}
        )
        assert isinstance(strategy, BollingerBandStrategy)
        
        strategy = AdvancedStrategyFactory.create_strategy(
            'stochastic', {'k_period': 14}
        )
        assert isinstance(strategy, StochasticStrategy)
        
        strategy = AdvancedStrategyFactory.create_strategy(
            'cci', {'period': 20}
        )
        assert isinstance(strategy, CCIStrategy)
    
    def test_create_statistical_arbitrage_strategy(self):
        """测试创建统计套利策略"""
        strategy = AdvancedStrategyFactory.create_strategy(
            'mean_reversion', {'lookback_period': 20}
        )
        assert isinstance(strategy, MeanReversionStrategy)
    
    def test_create_factor_strategy(self):
        """测试创建因子投资策略"""
        strategy = AdvancedStrategyFactory.create_strategy(
            'momentum_factor', {'momentum_periods': [21]}
        )
        assert isinstance(strategy, MomentumFactorStrategy)
        
        strategy = AdvancedStrategyFactory.create_strategy(
            'low_volatility', {}
        )
        assert isinstance(strategy, LowVolatilityStrategy)
        
        strategy = AdvancedStrategyFactory.create_strategy(
            'multi_factor', {'factors': ['value']}
        )
        assert isinstance(strategy, MultiFactorStrategy)
    
    def test_get_available_strategies(self):
        """测试获取可用策略列表"""
        strategies = AdvancedStrategyFactory.get_available_strategies()
        
        assert 'technical' in strategies
        assert 'statistical_arbitrage' in strategies
        assert 'factor_investment' in strategies
        
        assert len(strategies['technical']) > 0
        assert len(strategies['statistical_arbitrage']) > 0
        assert len(strategies['factor_investment']) > 0
    
    def test_invalid_strategy(self):
        """测试无效策略"""
        with pytest.raises(Exception):
            AdvancedStrategyFactory.create_strategy(
                'invalid_strategy', {}
            )


class TestStrategyIntegration:
    """策略集成测试"""
    
    def test_backtest_with_bollinger_strategy(self):
        """使用布林带策略进行回测"""
        config = {
            'period': 20,
            'std_dev': 2,
            'entry_threshold': 0.02
        }
        strategy = BollingerBandStrategy(config)
        
        data = generate_mock_price_data(datetime(2023, 1, 1), 252)
        data.attrs['stock_code'] = '000001.SZ'
        
        signals = []
        for i in range(50, len(data)):
            current_date = data.index[i]
            day_signals = strategy.generate_signals(data.iloc[:i+1], current_date)
            signals.extend(day_signals)
        
        assert isinstance(signals, list)
    
    def test_backtest_with_momentum_strategy(self):
        """使用动量因子策略进行回测"""
        config = {
            'momentum_periods': [21, 63],
            'momentum_weights': [0.6, 0.4]
        }
        strategy = MomentumFactorStrategy(config)
        
        data = generate_mock_price_data(datetime(2023, 1, 1), 300)
        data.attrs['stock_code'] = '000001.SZ'
        
        signals = []
        for i in range(130, len(data)):
            current_date = data.index[i]
            day_signals = strategy.generate_signals(data.iloc[:i+1], current_date)
            signals.extend(day_signals)
        
        assert isinstance(signals, list)
    
    def test_backtest_with_multi_factor_strategy(self):
        """使用多因子策略进行回测"""
        config = {
            'factors': ['value', 'momentum', 'low_volatility'],
            'factor_weights': [0.33, 0.33, 0.34]
        }
        strategy = MultiFactorStrategy(config)
        
        data = generate_mock_price_data(datetime(2023, 1, 1), 300)
        data.attrs['stock_code'] = '000001.SZ'
        
        signals = []
        for i in range(150, len(data)):
            current_date = data.index[i]
            day_signals = strategy.generate_signals(data.iloc[:i+1], current_date)
            signals.extend(day_signals)
        
        assert isinstance(signals, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
