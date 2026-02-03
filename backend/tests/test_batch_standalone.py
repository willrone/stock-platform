"""
批量信号生成性能测试（完全独立，不依赖 app 模块）
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict
import time


class SignalType(Enum):
    """信号类型"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass
class TradingSignal:
    """交易信号"""
    stock_code: str
    signal_type: SignalType
    strength: float
    price: float
    timestamp: datetime


class SimpleStrategy:
    """简单的动量策略"""
    
    def __init__(self):
        self.name = "SimpleStrategy"
    
    def generate_signals_traditional(self, df: pd.DataFrame, current_date: datetime) -> List[TradingSignal]:
        """传统方式：单日生成信号"""
        signals = []
        
        if current_date not in df.index:
            return signals
        
        idx = df.index.get_loc(current_date)
        if idx == 0:
            return signals
        
        prev_close = df.iloc[idx - 1]['close']
        curr_close = df.loc[current_date, 'close']
        returns = (curr_close - prev_close) / prev_close
        
        stock_code = df.attrs.get('stock_code', 'UNKNOWN')
        
        if returns > 0.01:
            signal = TradingSignal(
                stock_code=stock_code,
                signal_type=SignalType.BUY,
                strength=min(returns * 10, 1.0),
                price=curr_close,
                timestamp=current_date
            )
            signals.append(signal)
        elif returns < -0.01:
            signal = TradingSignal(
                stock_code=stock_code,
                signal_type=SignalType.SELL,
                strength=min(abs(returns) * 10, 1.0),
                price=curr_close,
                timestamp=current_date
            )
            signals.append(signal)
        
        return signals
    
    def generate_signals_batch(self, stock_data: Dict[str, pd.DataFrame]) -> List[TradingSignal]:
        """批量方式：一次性生成所有信号"""
        all_signals = []
        
        for stock_code, df in stock_data.items():
            # 向量化计算收益率
            df = df.copy()
            df['returns'] = df['close'].pct_change()
            
            # 向量化生成信号
            buy_mask = df['returns'] > 0.01
            sell_mask = df['returns'] < -0.01
            
            # 批量创建买入信号
            for date, row in df[buy_mask].iterrows():
                signal = TradingSignal(
                    stock_code=stock_code,
                    signal_type=SignalType.BUY,
                    strength=min(row['returns'] * 10, 1.0),
                    price=row['close'],
                    timestamp=date
                )
                all_signals.append(signal)
            
            # 批量创建卖出信号
            for date, row in df[sell_mask].iterrows():
                signal = TradingSignal(
                    stock_code=stock_code,
                    signal_type=SignalType.SELL,
                    strength=min(abs(row['returns']) * 10, 1.0),
                    price=row['close'],
                    timestamp=date
                )
                all_signals.append(signal)
        
        return all_signals


def create_test_data(stock_count: int, days: int) -> Dict[str, pd.DataFrame]:
    """创建测试数据"""
    all_stocks_data = {}
    start_date = datetime(2021, 1, 1)
    
    for i in range(stock_count):
        stock_code = f"{600000 + i:06d}.SH"
        
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
    
    return all_stocks_data


def test_performance():
    """性能测试"""
    
    print("=" * 80)
    print("批量信号生成性能测试")
    print("=" * 80)
    
    # 测试配置
    stock_count = 50
    days = 750  # 3年交易日
    
    # 1. 创建测试数据
    print(f"\n1. 创建测试数据...")
    all_stocks_data = create_test_data(stock_count, days)
    print(f"   ✓ {stock_count} 只股票 × {days} 天 = {stock_count * days:,} 数据点")
    
    # 2. 创建策略
    print(f"\n2. 创建策略...")
    strategy = SimpleStrategy()
    print(f"   ✓ {strategy.name}")
    
    # 3. 测试批量模式
    print(f"\n3. 批量模式测试...")
    
    batch_start = time.time()
    all_signals = strategy.generate_signals_batch(all_stocks_data)
    batch_precompute_time = time.time() - batch_start
    
    print(f"   预计算耗时: {batch_precompute_time:.4f}s")
    print(f"   生成信号数: {len(all_signals):,}")
    
    # 构建查询索引
    signal_cache = {}
    for sig in all_signals:
        key = (sig.stock_code, sig.timestamp.date())
        if key not in signal_cache:
            signal_cache[key] = []
        signal_cache[key].append(sig)
    
    # 测试查询性能
    query_start = time.time()
    query_count = 0
    for stock_code, df in all_stocks_data.items():
        for date in df.index:
            key = (stock_code, date.date())
            signals = signal_cache.get(key, [])
            query_count += 1
    query_time = time.time() - query_start
    
    print(f"   查询耗时: {query_time:.4f}s ({query_count:,} 次查询)")
    batch_total_time = batch_precompute_time + query_time
    print(f"   总耗时: {batch_total_time:.4f}s")
    
    # 4. 测试传统模式
    print(f"\n4. 传统模式测试...")
    
    traditional_start = time.time()
    traditional_signals = 0
    
    for stock_code, df in all_stocks_data.items():
        for date in df.index:
            signals = strategy.generate_signals_traditional(df, date)
            traditional_signals += len(signals)
    
    traditional_time = time.time() - traditional_start
    print(f"   耗时: {traditional_time:.4f}s")
    print(f"   生成信号数: {traditional_signals:,}")
    
    # 5. 性能对比
    print(f"\n" + "=" * 80)
    print("性能对比")
    print("=" * 80)
    
    print(f"\n批量模式:")
    print(f"  预计算: {batch_precompute_time:.4f}s")
    print(f"  查询: {query_time:.4f}s")
    print(f"  总计: {batch_total_time:.4f}s")
    
    print(f"\n传统模式:")
    print(f"  总计: {traditional_time:.4f}s")
    
    if batch_total_time > 0:
        speedup = traditional_time / batch_total_time
        print(f"\n⚡ 加速比: {speedup:.2f}x")
        
        if speedup > 3:
            print(f"✅ 性能提升显著！")
        elif speedup > 1.5:
            print(f"✅ 有明显提升")
        elif speedup > 1:
            print(f"⚠️  有一定提升，但不够明显")
        else:
            print(f"❌ 批量模式反而更慢")
    
    # 6. 推算 500 只股票性能
    print(f"\n" + "=" * 80)
    print("500 只股票性能推算")
    print("=" * 80)
    
    scale_factor = 500 / stock_count
    estimated_batch = batch_total_time * scale_factor
    estimated_traditional = traditional_time * scale_factor
    
    print(f"\n当前测试: {stock_count} 只股票 × {days} 天")
    print(f"\n推算 500 只股票:")
    print(f"  批量模式: {estimated_batch:.2f}s = {estimated_batch/60:.2f} 分钟")
    print(f"  传统模式: {estimated_traditional:.2f}s = {estimated_traditional/60:.2f} 分钟")
    
    target_time = 180  # 3 分钟
    if estimated_batch < target_time:
        print(f"\n✅ 批量模式预计可在 3 分钟内完成！")
        print(f"   剩余时间预算: {target_time - estimated_batch:.2f}s")
    else:
        needed_speedup = estimated_batch / target_time
        print(f"\n⚠️  批量模式预计需要 {estimated_batch/60:.2f} 分钟")
        print(f"   还需要 {needed_speedup:.2f}x 加速才能达到 3 分钟目标")
    
    # 7. 详细性能分析
    print(f"\n" + "=" * 80)
    print("性能分析")
    print("=" * 80)
    
    total_operations = stock_count * days
    
    print(f"\n批量模式:")
    print(f"  每次操作耗时: {batch_total_time / total_operations * 1000:.4f} ms")
    print(f"  吞吐量: {total_operations / batch_total_time:.0f} ops/s")
    
    print(f"\n传统模式:")
    print(f"  每次操作耗时: {traditional_time / total_operations * 1000:.4f} ms")
    print(f"  吞吐量: {total_operations / traditional_time:.0f} ops/s")
    
    print(f"\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)


if __name__ == "__main__":
    test_performance()
