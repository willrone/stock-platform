# Opt2: 批量交易执行优化方案

## 当前瓶颈分析
根据基线任务 dcba5363 的性能分析：
- **execute_trades_batch**: 72.0秒（29.1%）⚠️ 最大单点瓶颈
- 主要问题：逐个信号执行（for signal in all_signals）

## 优化方案

### 方案 A: 批量验证信号（推荐）
**位置**: `backtest_executor.py` 第 1598 行附近

**当前代码**:
```python
for signal in all_signals:
    # 验证信号
    is_valid, validation_reason = strategy.validate_signal(
        signal,
        portfolio_manager.get_portfolio_value(current_prices),
        portfolio_manager.positions,
    )
    if not is_valid:
        unexecuted_signals.append(...)
        continue
    
    # 执行信号
    trade, failure_reason = portfolio_manager.execute_signal(
        signal, current_prices
    )
```

**优化后代码**:
```python
# 1. 批量验证所有信号
portfolio_value = portfolio_manager.get_portfolio_value(current_prices)
valid_signals = []
for signal in all_signals:
    is_valid, validation_reason = strategy.validate_signal(
        signal, portfolio_value, portfolio_manager.positions
    )
    if is_valid:
        valid_signals.append(signal)
    else:
        unexecuted_signals.append({
            "stock_code": signal.stock_code,
            "timestamp": signal.timestamp,
            "signal_type": signal.signal_type.name,
            "execution_reason": validation_reason or "信号验证失败",
        })

# 2. 批量执行有效信号
for signal in valid_signals:
    trade_exec_start = time.perf_counter() if self.enable_performance_profiling else None
    trade, failure_reason = portfolio_manager.execute_signal(signal, current_prices)
    
    if self.enable_performance_profiling and trade_exec_start:
        trade_exec_duration = time.perf_counter() - trade_exec_start
        trade_execution_times.append(trade_exec_duration)
        self.performance_profiler.record_function_call("execute_signal", trade_exec_duration)
    
    if trade:
        executed_trades += 1
        trades_this_day += 1
        executed_trade_signals.append({
            "stock_code": signal.stock_code,
            "timestamp": signal.timestamp,
            "signal_type": signal.signal_type.name,
        })
    else:
        unexecuted_signals.append({
            "stock_code": signal.stock_code,
            "timestamp": signal.timestamp,
            "signal_type": signal.signal_type.name,
            "execution_reason": failure_reason or "执行失败（未知原因）",
        })
```

**优化收益**:
- 减少重复调用 `get_portfolio_value()`（每天只调用1次，而非每个信号1次）
- 预期提升：10-15%（节省 34-51秒）

### 方案 B: 向量化信号验证（高级）
**前提**: 需要修改 `strategy.validate_signal()` 支持批量验证

**优化后代码**:
```python
# 批量验证（需要策略支持）
portfolio_value = portfolio_manager.get_portfolio_value(current_prices)
validation_results = strategy.validate_signals_batch(
    all_signals, portfolio_value, portfolio_manager.positions
)

valid_signals = []
for signal, (is_valid, reason) in zip(all_signals, validation_results):
    if is_valid:
        valid_signals.append(signal)
    else:
        unexecuted_signals.append({...})

# 批量执行
for signal in valid_signals:
    trade, failure_reason = portfolio_manager.execute_signal(signal, current_prices)
    ...
```

**优化收益**:
- 进一步减少函数调用开销
- 预期提升：20-30%（节省 68-102秒）
- **风险**: 需要修改所有策略类，工作量大

## 实施计划

### 阶段 1: 方案 A（低风险，快速实施）
1. 修改 `backtest_executor.py` 第 1598 行附近的代码
2. 提取 `portfolio_value` 到循环外
3. 分离验证和执行逻辑
4. 创建验证任务（500股×3年×RSI）
5. 对比性能数据

**预期结果**:
- 从 289-306秒（Opt1后）降至 238-255秒
- 总提升：25-30%（相比基线 340秒）

### 阶段 2: 方案 B（可选，如果阶段1效果不佳）
1. 在 `BaseStrategy` 添加 `validate_signals_batch()` 方法
2. 为 RSI 策略实现批量验证
3. 修改 `backtest_executor.py` 调用批量验证
4. 创建验证任务
5. 对比性能数据

**预期结果**:
- 从 340秒降至 238-272秒
- 总提升：20-30%

## 验证标准
1. ✅ 任务成功完成
2. ✅ 性能提升 10-15%（方案A）或 20-30%（方案B）
3. ✅ Sharpe 比率保持在 0.42 左右（误差 ±5%）
4. ✅ 交易数保持在 7,591 左右（误差 ±2%）
5. ✅ 所有单元测试通过

## 风险评估
- **方案 A**: 低风险，只是重构代码结构，不改变逻辑
- **方案 B**: 中风险，需要修改策略接口，可能影响其他策略

## 时间估算
- **方案 A**: 30-45分钟（编码15分钟 + 测试30分钟）
- **方案 B**: 2-3小时（设计1小时 + 编码1小时 + 测试1小时）
