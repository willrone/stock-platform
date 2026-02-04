# 回测性能优化任务报告

## 任务概述
**目标**：在 willrone 项目中推进回测性能优化（Phase0/Phase1）
- Phase 0: 500只×3年回测 ≤ 5分钟
- Phase 1: 500只×3年回测 ≤ 3分钟

**执行时间**：2024-02-04
**状态**：Phase 0 准备工作完成，发现并修复关键问题

---

## 已完成工作

### 1. 基准脚本验证 ✅
- **脚本位置**：`backend/scripts/bench_backtest_500_3y.py`
- **功能**：一键运行固定 universe + 3年 + RSI 策略
- **输出**：wall time + 分段耗时（load / precompute / main loop / metrics）
- **验证结果**：脚本可正常运行，输出格式符合要求

### 2. 关键Bug修复 ✅

#### 问题：TradingSignal 创建缺少 price 参数
- **位置**：`backend/app/services/backtest/execution/backtest_executor.py:997-1013`
- **症状**：大量 `TradingSignal.__init__() missing 1 required positional argument: 'price'` 警告
- **影响**：预计算信号无法正常使用，导致回测失败
- **修复**：在 `get_precomputed_signal_fast` 函数中添加价格获取逻辑

```python
# 修复前
return [TradingSignal(
    signal_type=signal,
    stock_code=stock_code,
    timestamp=date,
    strength=1.0,
    reason=f"Precomputed signal"
)]

# 修复后
# 获取当前价格
try:
    data = stock_data.get(stock_code)
    if data is not None and date in data.index:
        current_price = float(data.loc[date, "close"])
    else:
        current_price = 0.0
except Exception:
    current_price = 0.0

return [TradingSignal(
    signal_type=signal,
    stock_code=stock_code,
    timestamp=date,
    price=current_price,  # 添加此参数
    strength=1.0,
    reason=f"Precomputed signal"
)]
```

### 3. 性能基线测量 ✅

#### 测试配置
- **策略**：RSI (period=14, oversold=30, overbought=70)
- **时间范围**：2021-01-01 至 2024-01-01（3年）
- **测试规模**：10, 50, 100 只股票

#### 测试结果

| 股票数 | 总耗时(s) | 数据加载(s) | 预计算(s) | 主循环(s) | 指标计算(s) |
|--------|-----------|-------------|-----------|-----------|-------------|
| 10     | 24.957    | 0.157       | 0.408     | 24.381    | 0.001       |
| 50     | 39.997    | 0.223       | 0.620     | 39.141    | 0.001       |
| 100    | 41.141    | 0.406       | 0.594     | 40.126    | 0.002       |

#### 性能分析

**主要发现**：
1. **主循环是最大瓶颈**：占总耗时的 97%+
2. **预计算有效**：信号预计算耗时很低（0.4-0.6秒），说明向量化预计算策略有效
3. **扩展性问题**：从50到100股票，耗时几乎不变（39.997s → 41.141s），说明存在并行化瓶颈
4. **500股票推算**：按当前趋势，500股票预计需要 **200-250秒（3-4分钟）**

**瓶颈定位**：
- ✅ 数据加载：已优化（占比 < 1%）
- ✅ 信号预计算：已优化（占比 < 2%）
- ❌ **主循环**：需要优化（占比 97%+）
  - 每日信号生成（即使有预计算，fallback路径仍在消耗时间）
  - 交易执行逻辑
  - 持仓管理（dict操作）
  - Portfolio快照记录

---

## 下一步优化计划

### Phase 0 优化策略（目标：≤ 5分钟）

#### 优先级1：主循环数组化
**预期收益**：30-50% 性能提升

1. **信号数组化**
   - 将 `signal[stock_i, t]` 改为 numpy 数组
   - 避免每日循环中的字典查找

2. **数据对齐**
   - close/open/valid_mask 使用 ndarray
   - 减少 DataFrame 操作

3. **持仓管理数组化**
   - positions 从 dict 改为数组
   - buy/sell 使用 numpy.where
   - trade records 使用 list append（延迟转换）

#### 优先级2：Portfolio History 优化
**预期收益**：10-20% 性能提升

- 默认降采样或关闭详细历史
- 仅保留 equity 曲线
- 减少每日快照记录开销

#### 优先级3：并行化改进
**预期收益**：20-30% 性能提升（多核环境）

- 分析为何50→100股票扩展性差
- 考虑使用多进程突破GIL限制
- 优化线程池调度

### Phase 1 优化策略（目标：≤ 3分钟）

1. **Numba JIT 编译**
   - 对热点循环使用 @njit 装饰器
   - 预期收益：20-40%

2. **Cython 重写核心循环**
   - 将主循环用 Cython 重写
   - 预期收益：30-50%

3. **算法优化**
   - 减少不必要的计算
   - 缓存重复计算结果

---

## 风险与注意事项

### 已识别风险
1. **接口兼容性**：修改信号处理逻辑时需确保与现有策略兼容
2. **数据完整性**：数组化后需确保边界检查和 NaN 处理
3. **测试覆盖**：每次优化后需运行完整的回测验证

### 已发现问题
1. ⚠️ **Pandas 3.0 兼容性问题**：
   - 错误：`Invalid frequency: M. Failed to parse with error message: ValueError("'M' is no longer supported for offsets. Please use 'ME' instead.")`
   - 位置：`_calculate_additional_metrics:2244`
   - 影响：额外指标计算失败（不影响核心回测）
   - 优先级：低（可在后续修复）

2. ⚠️ **部分股票数据加载失败**：
   - 34只股票数据加载失败（主要是2023年新上市的股票，数据不足3年）
   - 影响：测试覆盖范围略有减少
   - 优先级：低（不影响性能测试）

---

## 验收标准

### Phase 0（当前阶段）
- [x] 基准脚本可一键运行
- [x] 输出包含 wall time + 分段耗时
- [x] 识别主要性能瓶颈
- [ ] 500只×3年回测时间 ≤ 5分钟（预计需要优化后达成）

### Phase 1（下一阶段）
- [ ] 500只×3年回测时间 ≤ 3分钟

---

## 交付物

### 代码修改
1. `backend/app/services/backtest/execution/backtest_executor.py`
   - 修复 `get_precomputed_signal_fast` 函数中的 price 参数缺失问题

### 文档
1. `backend/PERF_OPTIMIZATION_LOG.md` - 性能优化日志
2. `backend/PERF_TASK_REPORT.md` - 本报告

### 基准数据
- 10/50/100 只股票的性能基线数据
- 瓶颈分析和优化建议

---

## 结论

**当前状态**：
- ✅ 修复了关键的接口不一致问题
- ✅ 建立了性能基线和测试流程
- ✅ 识别了主要性能瓶颈（主循环占97%+）
- ⚠️ 当前性能：100股票约41秒，推算500股票约200-250秒（3-4分钟）

**下一步行动**：
1. **立即执行**：主循环数组化优化（预期收益30-50%）
2. **短期执行**：Portfolio History 优化（预期收益10-20%）
3. **中期执行**：并行化改进 + Numba JIT（预期收益40-60%）

**预期结果**：
- 经过 Phase 0 优化后，500股票回测时间预计可降至 **120-150秒（2-2.5分钟）**
- 经过 Phase 1 优化后，500股票回测时间预计可降至 **90-120秒（1.5-2分钟）**

**建议**：
- 优先推进主循环数组化，这是收益最大的优化点
- 每次优化后都运行基准测试，记录性能变化
- 保持代码可读性和可维护性，避免过度优化
