"""
批量信号生成器单元测试
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from app.services.backtest.execution.batch_signal_generator import BatchSignalGenerator
from app.services.backtest.strategies.strategy_factory import StrategyFactory
from app.services.backtest.models import SignalType


def test_batch_signal_generator():
    """测试批量信号生成器"""
    
    print("=" * 80)
    print("批量信号生成器测试")
    print("=" * 80)
    
    # 1. 创建测试数据
    print("\n1. 创建测试数据...")
    stock_count = 10
    days = 100
    
    all_stocks_data = {}
    start_date = datetime(2024, 1, 1)
    
    for i in range(stock_count):
        stock_code = f"{600000 + i:06d}.SH"
        
        # 生成模拟价格数据
        dates = pd.date_range(start_date, periods=days, freq='D')
        np.random.seed(i)
        
        # 生成趋势 + 随机波动
        trend = np.linspace(100, 120, days)
        noise = np.random.randn(days) * 2
        close_prices = trend + noise
        
        df = pd.DataFrame({
            'open': close_prices * 0.99,
            'high': close_prices * 1.02,
            'low': close_prices * 0.98,
            'close': close_prices,
            'volume': np.random.randint(1000000, 10000000, days)
        }, index=dates)
        
        df.attrs['stock_code'] = stock_code
        all_stocks_data[stock_code] = df
    
    print(f"   ✓ 创建了 {stock_count} 只股票，每只 {days} 天数据")
    
    # 2. 创建策略
    print("\n2. 创建 MACD 策略...")
    strategy = StrategyFactory.create_strategy("MACD", {
        "fast_period": 12,
        "slow_period": 26,
        "signal_period": 9
    })
    print(f"   ✓ 策略创建成功: {strategy.name}")
    
    # 3. 创建批量信号生成器
    print("\n3. 创建批量信号生成器...")
    batch_generator = BatchSignalGenerator(strategy)
    print("   ✓ 批量生成器创建成功")
    
    # 4. 预计算所有信号
    print("\n4. 预计算所有信号...")
    import time
    start_time = time.time()
    
    success = batch_generator.precompute_all_signals(all_stocks_data)
    
    elapsed = time.time() - start_time
    
    if success:
        print(f"   ✓ 预计算成功，耗时: {elapsed:.4f}s")
        
        # 获取统计信息
        stats = batch_generator.get_stats()
        print(f"\n   统计信息:")
        print(f"   - 总信号数: {stats['total_signals']}")
        print(f"   - 买入信号: {stats['buy_signals']}")
        print(f"   - 卖出信号: {stats['sell_signals']}")
        print(f"   - 股票数: {stats['stocks_count']}")
    else:
        print("   ✗ 预计算失败")
        return False
    
    # 5. 测试信号查询
    print("\n5. 测试信号查询...")
    test_stock = list(all_stocks_data.keys())[0]
    test_date = start_date + timedelta(days=30)
    
    signals = batch_generator.get_signals(test_stock, test_date)
    print(f"   查询 {test_stock} @ {test_date.date()}")
    print(f"   结果: {len(signals)} 个信号")
    
    if signals:
        for sig in signals:
            print(f"   - {sig.signal_type.name}: 强度={sig.strength:.2f}, 价格={sig.price:.2f}")
    
    # 6. 性能对比测试
    print("\n6. 性能对比测试...")
    
    # 传统方式：逐日生成信号
    print("   传统方式（逐日生成）...")
    traditional_start = time.time()
    traditional_signals = 0
    
    for stock_code, df in all_stocks_data.items():
        for date in df.index[30:]:  # 跳过前30天（预热期）
            signals = strategy.generate_signals(df, date)
            traditional_signals += len(signals)
    
    traditional_elapsed = time.time() - traditional_start
    print(f"   耗时: {traditional_elapsed:.4f}s, 信号数: {traditional_signals}")
    
    # 批量方式：预计算 + 查询
    print("   批量方式（预计算 + 查询）...")
    batch_start = time.time()
    batch_signals = 0
    
    for stock_code, df in all_stocks_data.items():
        for date in df.index[30:]:
            signals = batch_generator.get_signals(stock_code, date)
            batch_signals += len(signals)
    
    batch_elapsed = time.time() - batch_start
    print(f"   耗时: {batch_elapsed:.4f}s, 信号数: {batch_signals}")
    
    # 计算加速比
    if batch_elapsed > 0:
        speedup = traditional_elapsed / batch_elapsed
        print(f"\n   ⚡ 加速比: {speedup:.2f}x")
        
        if speedup > 2:
            print(f"   ✅ 性能提升显著！")
        else:
            print(f"   ⚠️  加速效果不明显")
    
    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)
    
    return True


if __name__ == "__main__":
    test_batch_signal_generator()
