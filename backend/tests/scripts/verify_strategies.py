#!/usr/bin/env python3
"""
高级策略验证脚本

验证所有新增策略的实现是否正确
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.backtest.strategies import (
    BollingerBandStrategy,
    StochasticStrategy,
    CCIStrategy,
    MeanReversionStrategy,
    MomentumFactorStrategy,
    LowVolatilityStrategy,
    MultiFactorStrategy,
    ValueFactorStrategy,
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


def test_bollinger_strategy():
    """测试布林带策略"""
    print("测试布林带策略...")
    
    config = {'period': 20, 'std_dev': 2}
    strategy = BollingerBandStrategy(config)
    
    data = generate_mock_price_data(datetime(2023, 1, 1), 100)
    data.attrs['stock_code'] = '000001.SZ'
    
    indicators = strategy.calculate_indicators(data)
    assert 'upper_band' in indicators
    assert 'lower_band' in indicators
    assert 'percent_b' in indicators
    
    current_date = data.index[50]
    signals = strategy.generate_signals(data, current_date)
    assert isinstance(signals, list)
    
    print("✓ 布林带策略测试通过")
    return True


def test_stochastic_strategy():
    """测试随机指标策略"""
    print("测试随机指标策略...")
    
    config = {'k_period': 14, 'd_period': 3}
    strategy = StochasticStrategy(config)
    
    data = generate_mock_price_data(datetime(2023, 1, 1), 100)
    data.attrs['stock_code'] = '000001.SZ'
    
    indicators = strategy.calculate_indicators(data)
    assert 'k_percent' in indicators
    assert 'd_percent' in indicators
    
    current_date = data.index[50]
    signals = strategy.generate_signals(data, current_date)
    assert isinstance(signals, list)
    
    print("✓ 随机指标策略测试通过")
    return True


def test_cci_strategy():
    """测试CCI策略"""
    print("测试CCI策略...")
    
    config = {'period': 20}
    strategy = CCIStrategy(config)
    
    data = generate_mock_price_data(datetime(2023, 1, 1), 100)
    data.attrs['stock_code'] = '000001.SZ'
    
    indicators = strategy.calculate_indicators(data)
    assert 'cci' in indicators
    
    current_date = data.index[50]
    signals = strategy.generate_signals(data, current_date)
    assert isinstance(signals, list)
    
    print("✓ CCI策略测试通过")
    return True


def test_mean_reversion_strategy():
    """测试均值回归策略"""
    print("测试均值回归策略...")
    
    config = {'lookback_period': 20, 'zscore_threshold': 2.0}
    strategy = MeanReversionStrategy(config)
    
    data = generate_mock_price_data(datetime(2023, 1, 1), 100)
    data.attrs['stock_code'] = '000001.SZ'
    
    indicators = strategy.calculate_indicators(data)
    assert 'zscore' in indicators
    assert 'sma' in indicators
    
    current_date = data.index[50]
    signals = strategy.generate_signals(data, current_date)
    assert isinstance(signals, list)
    
    print("✓ 均值回归策略测试通过")
    return True


def test_momentum_factor_strategy():
    """测试动量因子策略"""
    print("测试动量因子策略...")
    
    config = {'momentum_periods': [21, 63], 'momentum_weights': [0.6, 0.4]}
    strategy = MomentumFactorStrategy(config)
    
    data = generate_mock_price_data(datetime(2023, 1, 1), 200)
    data.attrs['stock_code'] = '000001.SZ'
    
    indicators = strategy.calculate_indicators(data)
    assert 'momentum' in indicators
    
    current_date = data.index[150]
    signals = strategy.generate_signals(data, current_date)
    assert isinstance(signals, list)
    
    print("✓ 动量因子策略测试通过")
    return True


def test_value_factor_strategy():
    """测试价值因子策略"""
    print("测试价值因子策略...")
    
    config = {
        'pe_weight': 0.25,
        'pb_weight': 0.25,
        'ps_weight': 0.25,
        'ev_ebitda_weight': 0.25
    }
    strategy = ValueFactorStrategy(config)
    
    data = generate_mock_price_data(datetime(2023, 1, 1), 300)
    data.attrs['stock_code'] = '000001.SZ'
    
    indicators = strategy.calculate_indicators(data)
    assert 'value_score' in indicators
    assert 'pe_ratio' in indicators
    assert 'pb_ratio' in indicators
    
    current_date = data.index[280]
    signals = strategy.generate_signals(data, current_date)
    assert isinstance(signals, list)
    
    print("✓ 价值因子策略测试通过")
    return True


def test_low_volatility_strategy():
    """测试低波动因子策略"""
    print("测试低波动因子策略...")
    
    config = {'volatility_period': 21, 'volatility_window': 63}
    strategy = LowVolatilityStrategy(config)
    
    data = generate_mock_price_data(datetime(2023, 1, 1), 100)
    data.attrs['stock_code'] = '000001.SZ'
    
    indicators = strategy.calculate_indicators(data)
    assert 'volatility' in indicators
    assert 'risk_adjusted_return' in indicators
    
    print("✓ 低波动因子策略测试通过")
    return True


def test_multi_factor_strategy():
    """测试多因子策略"""
    print("测试多因子策略...")
    
    config = {
        'factors': ['value', 'momentum', 'low_volatility'],
        'factor_weights': [0.33, 0.33, 0.34]
    }
    strategy = MultiFactorStrategy(config)
    
    data = generate_mock_price_data(datetime(2023, 1, 1), 300)
    data.attrs['stock_code'] = '000001.SZ'
    
    indicators = strategy.calculate_indicators(data)
    assert 'combined_score' in indicators
    assert 'value_score' in indicators
    assert 'momentum_score' in indicators
    
    current_date = data.index[200]
    signals = strategy.generate_signals(data, current_date)
    assert isinstance(signals, list)
    
    print("✓ 多因子策略测试通过")
    return True


def test_strategy_factory():
    """测试高级策略工厂"""
    print("测试高级策略工厂...")
    
    strategies = AdvancedStrategyFactory.get_available_strategies()
    
    assert 'technical' in strategies
    assert 'statistical_arbitrage' in strategies
    assert 'factor_investment' in strategies
    
    assert len(strategies['technical']) == 3
    assert len(strategies['statistical_arbitrage']) == 3
    assert len(strategies['factor_investment']) == 4
    
    strategy = AdvancedStrategyFactory.create_strategy('bollinger', {'period': 20})
    assert isinstance(strategy, BollingerBandStrategy)
    
    strategy = AdvancedStrategyFactory.create_strategy('mean_reversion', {})
    assert isinstance(strategy, MeanReversionStrategy)
    
    strategy = AdvancedStrategyFactory.create_strategy('momentum_factor', {})
    assert isinstance(strategy, MomentumFactorStrategy)
    
    print("✓ 高级策略工厂测试通过")
    return True


def test_backtest_simulation():
    """回测模拟测试"""
    print("执行回测模拟测试...")
    
    strategies_to_test = [
        ('bollinger', {'period': 20, 'std_dev': 2}),
        ('momentum_factor', {'momentum_periods': [21, 63], 'momentum_weights': [0.5, 0.5]}),
        ('mean_reversion', {'lookback_period': 20, 'zscore_threshold': 2.0}),
        ('multi_factor', {'factors': ['value', 'momentum'], 'factor_weights': [0.5, 0.5]})
    ]
    
    for strategy_name, config in strategies_to_test:
        try:
            strategy = AdvancedStrategyFactory.create_strategy(strategy_name, config)
            
            data = generate_mock_price_data(datetime(2023, 1, 1), 300)
            data.attrs['stock_code'] = '000001.SZ'
            
            all_signals = []
            min_idx = 50 if strategy_name in ['bollinger', 'mean_reversion'] else 130
            
            for i in range(min_idx, len(data)):
                current_date = data.index[i]
                day_signals = strategy.generate_signals(data.iloc[:i+1], current_date)
                all_signals.extend(day_signals)
            
            print(f"  ✓ {strategy_name}: 产生 {len(all_signals)} 个信号")
            
        except Exception as e:
            print(f"  ✗ {strategy_name}: 测试失败 - {e}")
            return False
    
    print("✓ 回测模拟测试通过")
    return True


def main():
    """主测试函数"""
    print("=" * 60)
    print("高级策略验证测试")
    print("=" * 60)
    
    tests = [
        test_bollinger_strategy,
        test_stochastic_strategy,
        test_cci_strategy,
        test_mean_reversion_strategy,
        test_momentum_factor_strategy,
        test_value_factor_strategy,
        test_low_volatility_strategy,
        test_multi_factor_strategy,
        test_strategy_factory,
        test_backtest_simulation
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ {test.__name__} 测试失败: {e}")
            failed += 1
    
    print("=" * 60)
    print(f"测试结果: {passed} 通过, {failed} 失败")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
