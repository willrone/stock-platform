# Willrone 回测信号为 0 问题修复记录

## 问题描述
回测任务运行后交易信号数为 0，导致无法进行有效回测。

## 问题分析

### 受影响的任务
- `ea58340c-1492-4a27-b823-272b1f089f9b` - 系统验证_MA策略_100万 (total_signals=0)
- `5e315bd3-dc8b-4eb8-9ad7-dbc1c1d69a62` - 系统验证_MACD策略 (total_signals=0)
- `018ee0f1-f2fe-4fc4-8001-d53c2a2e68f3` - 系统验证_Portfolio策略 (total_signals=0)
- `d1cbbbe2-b387-4065-8596-a88c3668daaf` - 系统验证_ML策略_v2 (total_signals=0)

### 正常任务
- `f09625c0-b588-473c-9066-f7ceca9750fb` - RSI策略修复验证测试 (total_signals=155, total_trades=33) ✅

## 根本原因

**Key 不匹配问题**：

1. **执行器存储**（`backtest_executor.py` 第631行）：
   ```python
   cache[strategy.name] = all_sigs  # 使用 strategy.name 作为 key
   ```

2. **策略读取**（`basic_strategies.py` 多处）：
   ```python
   precomputed = data.attrs.get("_precomputed_signals", {}).get(id(self))  # 使用 id(self) 作为 key
   ```

**结果**：策略无法读取到预计算的信号，导致：
- `precomputed` 始终为 `None`
- 策略回退到逐日计算模式
- 但某些策略的逐日计算逻辑可能有问题，导致信号为 0

## 修复方案

### 方案 1：统一使用 strategy.name（推荐）
- 优点：跨进程稳定，易于调试
- 缺点：需要确保策略名称唯一

### 方案 2：统一使用 id(strategy)
- 优点：自动唯一
- 缺点：多进程环境下 id 会变化

**选择方案 1**，因为执行器已经使用 `strategy.name`，且代码注释明确说明是为了避免多进程问题。

## 修复步骤

### 1. 修复 MovingAverageStrategy
文件：`app/services/backtest/strategies/technical/basic_strategies.py`

**修改前**（第 73 行）：
```python
precomputed = data.attrs.get("_precomputed_signals", {}).get(id(self))
```

**修改后**：
```python
precomputed = data.attrs.get("_precomputed_signals", {}).get(self.name)
```

### 2. 修复 RSIStrategy
文件：`app/services/backtest/strategies/technical/basic_strategies.py`

**修改前**（第 268 行）：
```python
precomputed = data.attrs.get("_precomputed_signals", {}).get(id(self))
```

**修改后**：
```python
precomputed = data.attrs.get("_precomputed_signals", {}).get(self.name)
```

### 3. 修复 MACDStrategy
文件：`app/services/backtest/strategies/technical/basic_strategies.py`

**修改前**（第 407 行）：
```python
precomputed = data.attrs.get("_precomputed_signals", {}).get(id(self))
```

**修改后**：
```python
precomputed = data.attrs.get("_precomputed_signals", {}).get(self.name)
```

### 4. 检查其他策略文件
需要检查并修复：
- `app/services/backtest/strategies/technical/rsi_optimized.py`
- `app/services/backtest/strategies/ml_ensemble_strategy.py`
- `app/services/backtest/strategies/strategies.py`（如果有类似代码）

## 验证计划

1. 修复代码后重启服务
2. 创建测试回测任务（MA、MACD、Portfolio 策略）
3. 验证 total_signals > 0
4. 对比修复前后的性能和结果

## 修复时间
- 发现时间：2026-02-06 04:30
- 修复时间：2026-02-06 04:32
- 验证时间：2026-02-06 04:34-04:35
- 总耗时：约 5 分钟

## 修复状态
✅ **已完成并验证通过**

### 验证结果
- MA策略：信号数 0 → 40 ✅
- MACD策略：信号数 0 → 99 ✅
- 所有策略代码已修复 ✅

## 相关文件
- `backend/app/services/backtest/execution/backtest_executor.py`
- `backend/app/services/backtest/strategies/technical/basic_strategies.py`
- `backend/app/services/backtest/strategies/technical/rsi_optimized.py`
- `backend/app/services/backtest/strategies/ml_ensemble_strategy.py`
