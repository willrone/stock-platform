#!/usr/bin/env python3
"""
策略功能验证脚本

验证所有策略是否能正常生成交易信号
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
    PairsTradingStrategy,
    CointegrationStrategy,
    ValueFactorStrategy,
    MomentumFactorStrategy,
    LowVolatilityStrategy,
    MultiFactorStrategy,
    AdvancedStrategyFactory,
    StrategyFactory
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


def test_basic_strategies():
    """测试基础策略"""
    print("=" * 60)
    print("测试基础策略")
    print("=" * 60)
    
    basic_strategies = StrategyFactory.get_available_strategies()
    print(f"基础策略列表: {basic_strategies}")
    
    all_passed = True
    
    for strategy_name in basic_strategies:
        try:
            strategy = StrategyFactory.create_strategy(strategy_name, {})
            
            data = generate_mock_price_data(datetime(2023, 1, 1), 300)
            data.attrs['stock_code'] = '000001.SZ'
            
            signals = []
            for i in range(50, len(data)):
                current_date = data.index[i]
                day_signals = strategy.generate_signals(data.iloc[:i+1], current_date)
                signals.extend(day_signals)
            
            buy_signals = [s for s in signals if s.signal_type.value == 1]
            sell_signals = [s for s in signals if s.signal_type.value == -1]
            
            print(f"✓ {strategy_name}: 买入信号 {len(buy_signals)}, 卖出信号 {len(sell_signals)}")
            
        except Exception as e:
            print(f"✗ {strategy_name}: 测试失败 - {e}")
            all_passed = False
    
    return all_passed


def test_advanced_strategies():
    """测试高级策略"""
    print("\n" + "=" * 60)
    print("测试高级策略")
    print("=" * 60)
    
    strategies_to_test = [
        ('布林带', BollingerBandStrategy, {'period': 20, 'std_dev': 2}),
        ('随机指标', StochasticStrategy, {'k_period': 14, 'd_period': 3}),
        ('CCI', CCIStrategy, {'period': 20}),
        ('配对交易', PairsTradingStrategy, {'lookback_period': 20, 'entry_threshold': 2.0}),
        ('均值回归', MeanReversionStrategy, {'lookback_period': 20, 'zscore_threshold': 2.0}),
        ('协整', CointegrationStrategy, {'lookback_period': 60, 'entry_threshold': 2.0}),
        ('价值因子', ValueFactorStrategy, {}),
        ('动量因子', MomentumFactorStrategy, {'momentum_periods': [21, 63]}),
        ('低波动', LowVolatilityStrategy, {'volatility_period': 21, 'volatility_window': 63}),
        ('多因子', MultiFactorStrategy, {'factors': ['value', 'momentum'], 'factor_weights': [0.5, 0.5]}),
    ]
    
    all_passed = True
    
    for name, StrategyClass, config in strategies_to_test:
        try:
            strategy = StrategyClass(config)
            
            min_data = 130 if name in ['协整', '价值因子', '多因子'] else (50 if name in ['布林带', '均值回归', '配对交易'] else 70)
            data = generate_mock_price_data(datetime(2023, 1, 1), 300)
            data.attrs['stock_code'] = '000001.SZ'
            
            signals = []
            for i in range(min_data, len(data)):
                current_date = data.index[i]
                day_signals = strategy.generate_signals(data.iloc[:i+1], current_date)
                signals.extend(day_signals)
            
            buy_signals = [s for s in signals if s.signal_type.value == 1]
            sell_signals = [s for s in signals if s.signal_type.value == -1]
            
            if len(buy_signals) == 0 and len(sell_signals) == 0:
                print(f"⚠ {name}: 无信号（可能需要更长回测期间）")
            else:
                print(f"✓ {name}: 买入信号 {len(buy_signals)}, 卖出信号 {len(sell_signals)}")
            
        except Exception as e:
            print(f"✗ {name}: 测试失败 - {e}")
            all_passed = False
    
    return all_passed


def test_strategy_factory():
    """测试高级策略工厂"""
    print("\n" + "=" * 60)
    print("测试高级策略工厂")
    print("=" * 60)
    
    try:
        categories = AdvancedStrategyFactory.get_available_strategies()
        
        print("策略分类:")
        for category, strategies in categories.items():
            print(f"  {category}: {strategies}")
        
        all_strategies = []
        for strategies in categories.values():
            all_strategies.extend(strategies)
        
        print(f"\n总共 {len(all_strategies)} 个策略")
        
        for strategy_name in all_strategies:
            strategy = AdvancedStrategyFactory.create_strategy(strategy_name, {})
            print(f"✓ {strategy_name}: {strategy.name}")
        
        return True
        
    except Exception as e:
        print(f"✗ 策略工厂测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("量化交易策略功能验证")
    print("=" * 60 + "\n")
    
    results = []
    
    results.append(("基础策略", test_basic_strategies()))
    results.append(("高级策略", test_advanced_strategies()))
    results.append(("策略工厂", test_strategy_factory()))
    
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"{test_name}: {status}")
    
    all_passed = all(passed for _, passed in results)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ 所有策略测试通过！")
    else:
        print("✗ 部分策略测试失败，请检查日志")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
