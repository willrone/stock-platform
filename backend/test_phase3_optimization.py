"""
Phase 3 优化测试脚本

测试向量化优化的性能提升
"""

import sys
import time
import numpy as np
from datetime import datetime, timedelta

# 添加项目路径
sys.path.insert(0, '/Users/ronghui/Projects/willrone/backend')

from app.services.backtest.execution.vectorized_loop import (
    vectorized_price_lookup_core,
    extract_signals_vectorized,
    update_portfolio_value_vectorized,
    NUMBA_AVAILABLE,
)

print(f"🔧 Numba 可用: {NUMBA_AVAILABLE}")
print()

# 模拟数据
N_STOCKS = 414  # 股票数量
N_DAYS = 730    # 交易日数量

print(f"📊 测试配置: {N_STOCKS} 只股票 × {N_DAYS} 个交易日")
print()

# 生成测试数据
np.random.seed(42)
close_mat = np.random.uniform(10, 100, (N_STOCKS, N_DAYS))
valid_mat = np.random.rand(N_STOCKS, N_DAYS) > 0.05  # 95% 有效
signal_mat = np.zeros((N_STOCKS, N_DAYS), dtype=np.int8)

# 随机生成一些信号（约 5% 的数据点有信号）
signal_indices = np.random.choice(N_STOCKS * N_DAYS, size=int(N_STOCKS * N_DAYS * 0.05), replace=False)
for idx in signal_indices:
    i = idx // N_DAYS
    j = idx % N_DAYS
    signal_mat[i, j] = np.random.choice([1, -1])  # BUY or SELL

positions = np.random.uniform(0, 1000, N_STOCKS)
positions[positions < 500] = 0  # 约一半股票无持仓

print("=" * 60)
print("测试 1: 价格查找性能")
print("=" * 60)

# 测试价格查找
stock_indices = np.arange(N_STOCKS, dtype=np.int32)
date_idx = N_DAYS // 2

# 预热（触发 JIT 编译）
if NUMBA_AVAILABLE:
    _ = vectorized_price_lookup_core(stock_indices[:10], date_idx, close_mat, valid_mat)
    print("✅ JIT 编译完成（预热）")

# 基准测试：传统方法
def traditional_price_lookup(stock_indices, date_idx, close_mat, valid_mat):
    prices = []
    valid_flags = []
    for stock_idx in stock_indices:
        if valid_mat[stock_idx, date_idx]:
            prices.append(close_mat[stock_idx, date_idx])
            valid_flags.append(True)
        else:
            prices.append(np.nan)
            valid_flags.append(False)
    return np.array(prices), np.array(valid_flags)

# 测试传统方法
t0 = time.perf_counter()
for _ in range(100):
    prices_trad, valid_trad = traditional_price_lookup(stock_indices, date_idx, close_mat, valid_mat)
time_trad = time.perf_counter() - t0

# 测试向量化方法
t0 = time.perf_counter()
for _ in range(100):
    prices_vec, valid_vec = vectorized_price_lookup_core(stock_indices, date_idx, close_mat, valid_mat)
time_vec = time.perf_counter() - t0

print(f"传统方法: {time_trad*1000:.2f} ms (100次)")
print(f"向量化方法: {time_vec*1000:.2f} ms (100次)")
print(f"⚡ 提速: {time_trad/time_vec:.2f}x")
print()

print("=" * 60)
print("测试 2: 信号提取性能")
print("=" * 60)

# 预热
if NUMBA_AVAILABLE:
    _ = extract_signals_vectorized(signal_mat, date_idx, valid_mat)
    print("✅ JIT 编译完成（预热）")

# 基准测试：传统方法
def traditional_signal_extraction(signal_mat, date_idx, valid_mat):
    stock_indices = []
    signal_types = []
    for i in range(signal_mat.shape[0]):
        if valid_mat[i, date_idx] and signal_mat[i, date_idx] != 0:
            stock_indices.append(i)
            signal_types.append(signal_mat[i, date_idx])
    return np.array(stock_indices, dtype=np.int32), np.array(signal_types, dtype=np.int8)

# 测试传统方法
t0 = time.perf_counter()
for _ in range(100):
    indices_trad, types_trad = traditional_signal_extraction(signal_mat, date_idx, valid_mat)
time_trad = time.perf_counter() - t0

# 测试向量化方法
t0 = time.perf_counter()
for _ in range(100):
    indices_vec, types_vec = extract_signals_vectorized(signal_mat, date_idx, valid_mat)
time_vec = time.perf_counter() - t0

print(f"传统方法: {time_trad*1000:.2f} ms (100次)")
print(f"向量化方法: {time_vec*1000:.2f} ms (100次)")
print(f"⚡ 提速: {time_trad/time_vec:.2f}x")
print(f"提取信号数: {len(indices_vec)}")
print()

print("=" * 60)
print("测试 3: 组合价值计算性能")
print("=" * 60)

# 预热
if NUMBA_AVAILABLE:
    _ = update_portfolio_value_vectorized(positions, close_mat[:, date_idx], valid_mat[:, date_idx], 100000.0)
    print("✅ JIT 编译完成（预热）")

# 基准测试：传统方法
def traditional_portfolio_value(positions, prices, valid, cash):
    total_value = cash
    for i in range(len(positions)):
        if valid[i] and positions[i] > 0:
            total_value += positions[i] * prices[i]
    return total_value

prices_day = close_mat[:, date_idx]
valid_day = valid_mat[:, date_idx]
cash = 100000.0

# 测试传统方法
t0 = time.perf_counter()
for _ in range(1000):
    value_trad = traditional_portfolio_value(positions, prices_day, valid_day, cash)
time_trad = time.perf_counter() - t0

# 测试向量化方法
t0 = time.perf_counter()
for _ in range(1000):
    value_vec = update_portfolio_value_vectorized(positions, prices_day, valid_day, cash)
time_vec = time.perf_counter() - t0

print(f"传统方法: {time_trad*1000:.2f} ms (1000次)")
print(f"向量化方法: {time_vec*1000:.2f} ms (1000次)")
print(f"⚡ 提速: {time_trad/time_vec:.2f}x")
print(f"组合价值: ${value_vec:,.2f}")
print()

print("=" * 60)
print("测试 4: 完整回测循环模拟")
print("=" * 60)

# 模拟完整的回测循环
def simulate_backtest_loop_traditional(N_DAYS, close_mat, valid_mat, signal_mat, positions, cash):
    """传统方法：逐日循环"""
    total_signals = 0
    
    for day_idx in range(N_DAYS):
        # 1. 价格查找
        prices = []
        for i in range(close_mat.shape[0]):
            if valid_mat[i, day_idx]:
                prices.append(close_mat[i, day_idx])
            else:
                prices.append(np.nan)
        
        # 2. 信号提取
        signals = []
        for i in range(signal_mat.shape[0]):
            if valid_mat[i, day_idx] and signal_mat[i, day_idx] != 0:
                signals.append((i, signal_mat[i, day_idx]))
        
        total_signals += len(signals)
        
        # 3. 组合价值更新
        total_value = cash
        for i in range(len(positions)):
            if not np.isnan(prices[i]) and positions[i] > 0:
                total_value += positions[i] * prices[i]
    
    return total_signals

def simulate_backtest_loop_vectorized(N_DAYS, close_mat, valid_mat, signal_mat, positions, cash):
    """向量化方法：使用 Numba 加速"""
    total_signals = 0
    stock_indices = np.arange(close_mat.shape[0], dtype=np.int32)
    
    for day_idx in range(N_DAYS):
        # 1. 价格查找（向量化）
        prices, valid_flags = vectorized_price_lookup_core(stock_indices, day_idx, close_mat, valid_mat)
        
        # 2. 信号提取（向量化）
        sig_indices, sig_types = extract_signals_vectorized(signal_mat, day_idx, valid_mat)
        total_signals += len(sig_indices)
        
        # 3. 组合价值更新（向量化）
        total_value = update_portfolio_value_vectorized(positions, prices, valid_flags, cash)
    
    return total_signals

# 预热
if NUMBA_AVAILABLE:
    _ = simulate_backtest_loop_vectorized(10, close_mat, valid_mat, signal_mat, positions, cash)
    print("✅ JIT 编译完成（预热）")

# 测试传统方法（只测试部分天数，避免太慢）
print("测试传统方法（100天）...")
t0 = time.perf_counter()
signals_trad = simulate_backtest_loop_traditional(100, close_mat, valid_mat, signal_mat, positions, cash)
time_trad = time.perf_counter() - t0

# 测试向量化方法（完整天数）
print("测试向量化方法（730天）...")
t0 = time.perf_counter()
signals_vec = simulate_backtest_loop_vectorized(N_DAYS, close_mat, valid_mat, signal_mat, positions, cash)
time_vec = time.perf_counter() - t0

# 估算传统方法的完整时间
time_trad_full = time_trad * (N_DAYS / 100)

print(f"传统方法（估算730天）: {time_trad_full:.3f} 秒")
print(f"向量化方法（730天）: {time_vec:.3f} 秒")
print(f"⚡ 提速: {time_trad_full/time_vec:.2f}x")
print(f"总信号数: {signals_vec}")
print()

print("=" * 60)
print("📊 性能提升总结")
print("=" * 60)
print(f"✅ 价格查找: ~{time_trad/time_vec:.1f}x 提速")
print(f"✅ 信号提取: ~{time_trad/time_vec:.1f}x 提速")
print(f"✅ 组合价值计算: ~{time_trad/time_vec:.1f}x 提速")
print(f"✅ 完整循环: ~{time_trad_full/time_vec:.1f}x 提速")
print()

# 估算实际回测的性能提升
baseline_time = 357.7  # 基线耗时（秒）
estimated_improvement = time_trad_full / time_vec
estimated_new_time = baseline_time / estimated_improvement

print("=" * 60)
print("🎯 预期回测性能提升")
print("=" * 60)
print(f"基线耗时: {baseline_time:.1f} 秒")
print(f"预期提速: {estimated_improvement:.2f}x")
print(f"预期新耗时: {estimated_new_time:.1f} 秒")
print(f"目标耗时: 180 秒")
print(f"距离目标: {estimated_new_time - 180:.1f} 秒 ({'✅ 已达标' if estimated_new_time < 180 else '❌ 需继续优化'})")
print()
