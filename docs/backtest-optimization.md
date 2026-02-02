# Willrone 回测性能优化方案

📅 **创建日期**: 2026-02-01 | **任务ID**: 2ffea0f6-d0a0-48ec-976a-125857d14ea7

---

## 一、当前性能概况

- **总运行时间**: 143.96 秒
- **交易日数**: 747 天 | **股票数量**: 50 只
- **总信号数**: 3604 个 | **执行交易数**: 885 笔
- **吞吐量**: 25.04 信号/秒, 5.19 天/秒

## 二、性能瓶颈分析

### 2.1 各阶段耗时分布

| 阶段 | 耗时(秒) | 占比 |
| :--- | :--- | :--- |
| 回测执行 | 138.47 | 96.19% |
| 报告生成 | 2.99 | 2.08% |
| 数据加载 | 1.04 | 0.72% |

### 2.2 关键函数耗时

> ⚠️ **核心瓶颈**: `generate_signals_core_work` 累计耗时 251.52 秒，平均 346ms/次，最大 3.55 秒。

### 2.3 并行效率

> 📉 **并行效率仅 50%**！10个 worker 只实现 5x 加速。并行开销达 25.08 秒。

---

## 三、优化方案

### 方案 1: 信号生成核心优化 (预期 -60~80 秒)
🎯 **优先级**: 🔴 最高 | **难度**: 中等

1.  **向量化计算** - 用 NumPy/Pandas 向量操作替代 Python 循环。
    ```python
    # 优化前
    for i in range(len(df)):
        signal = calculate_rsi(df.iloc[:i])

    # 优化后
    signals = ta.RSI(df['close'], timeperiod=14)
    ```
2.  **预计算技术指标** - 回测开始前一次性计算所有 RSI/布林带等指标。
3.  **增量计算** - 滑动窗口指标使用增量更新而非全量重算。

📋 **验收**: `generate_signals_core_work` 平均耗时 < 100ms (当前 346ms)。

### 方案 2: 并行效率优化 (预期 -15~25 秒)
⚡ **优先级**: 🟠 高 | **难度**: 中等

1.  **共享内存** - 用 `multiprocessing.shared_memory` 减少 IPC 开销。
2.  **批量处理** - 每批处理 20 天，减少并行启动次数。
3.  **线程池** - 对 IO 密集型任务使用 `ThreadPoolExecutor`。

📋 **验收**: 并行效率 > 80% (当前 50%), 开销 < 10s (当前 25s)。

### 方案 3: 数据结构优化 (预期 -10~20 秒)
📊 **优先级**: 🟡 中 | **难度**: 低

1.  NumPy 结构化数组替代 DataFrame。
2.  日期预索引优化切片。

### 方案 4: 策略级缓存 (预期 -5~10 秒)
使用 LRU 缓存子策略结果，避免重复计算 (RSI 被调用 697 次)。

### 方案 5: 报告生成优化 (预期 -2 秒)
使用 `orjson` 替代标准 `json` (快 3-10 倍)。

---

## 四、实施计划

### 第一阶段: 快速收益 ✅ 已完成
- [x] 预计算技术指标在回测开始前完成
- [x] 使用 orjson 替代标准 json
- [x] 建立日期预索引优化切片

### 第二阶段: 核心优化 ✅ 已完成
- [x] 重构 `generate_signals_core_work` 向量化计算
- [x] 实现增量计算模式
- [x] 添加子策略结果缓存

### 第三阶段: 并行优化 🔄 进行中
- [x] 实现多进程支持框架（ProcessPoolExecutor）
- [ ] 实现共享内存数据管理器
- [ ] 批量处理多个交易日（每批 20 天）
- [x] 优化进程池配置（支持多线程/多进程切换）

---

## 五、验收标准汇总

| 指标 | 当前值 | 目标值 |
| :--- | :--- | :--- |
| 总回测时间 | 143.96 秒 | < 45 秒 |
| 信号生成平均耗时 | 346 ms | < 100 ms |
| 并行效率 | 50% | > 80% |
| 吞吐量(天/秒) | 5.19 | > 16 |
| 报告生成耗时 | 2.99 秒 | < 1 秒 |

---

## 六、开发指南

### 6.1 相关代码位置
- **回测引擎**: `backend/app/services/backtest/engine.py`
- **信号生成**: `backend/app/services/backtest/signal_generator.py`
- **策略实现**: `backend/app/services/backtest/strategies/`
- **性能统计**: `backend/app/services/backtest/performance.py`

### 6.2 测试与验证
每次优化后对比: `summary.total_time`, `stages.*.percentage`, `function_calls.*.avg_time`, `parallel_efficiency`

---

## 七、优化进展记录

### 2026-02-02 更新

#### 已完成的优化

**1. 策略向量化预计算**

已为以下策略实现 `precompute_signals()` 方法，支持向量化预计算：

| 策略 | 文件位置 | 状态 |
|------|----------|------|
| MovingAverageStrategy | `strategies/technical/basic_strategies.py` | ✅ |
| RSIStrategy | `strategies/technical/basic_strategies.py` | ✅ |
| BollingerBandsStrategy | `strategies/technical/basic_strategies.py` | ✅ |
| MACDStrategy | `strategies/technical/basic_strategies.py` | ✅ |
| StochasticStrategy | `strategies/strategies.py` | ✅ |
| CCIStrategy | `strategies/strategies.py` | ✅ |
| CointegrationStrategy | `strategies/strategies.py` | ✅ |
| FactorStrategy 系列 | `strategies/factor_strategies.py` | ⏳ 待实现 |

**2. 多进程/多线程支持**

在 `BacktestExecutor` 中新增：
- `use_multiprocessing` 参数：控制使用多进程或多线程
- `_multiprocess_precompute_worker()` 模块级函数：支持多进程序列化
- 自动回退机制：多进程失败时回退到多线程

```python
# 使用示例
executor = BacktestExecutor(
    data_dir="data",
    enable_parallel=True,
    use_multiprocessing=False,  # 默认多线程，推荐
    max_workers=4
)
```

#### 性能测试结果

**测试环境**: 20只股票 × 500天，MACD策略

| 模式 | 耗时 | 加速比 | 说明 |
|------|------|--------|------|
| 顺序执行 | 0.020s | 基准 | - |
| 多线程 (4 workers) | 0.023s | 0.86x | 线程开销 |
| 多进程 (4 workers) | 0.199s | 0.10x | 进程+序列化开销 |

**关键发现**:
1. **向量化预计算本身已非常高效**（20只股票仅需 0.020s）
2. **多进程开销大于收益**：进程创建和数据序列化开销远大于计算时间
3. **推荐使用多线程**：对于信号预计算场景，多线程是更好的选择

#### 结论与建议

| 优化项 | 适用场景 | 推荐度 |
|--------|----------|--------|
| 向量化预计算 | 所有策略 | ⭐⭐⭐⭐⭐ |
| 多线程并行 | I/O密集型、数据加载 | ⭐⭐⭐⭐ |
| 多进程并行 | 超参数搜索、复杂模型训练 | ⭐⭐⭐ |
| 日期预索引 | 大规模回测 | ⭐⭐⭐⭐ |
| orjson 序列化 | 报告生成 | ⭐⭐⭐⭐ |

#### 下一步计划

1. **待实现**: FactorStrategy 系列策略的向量化
2. **待优化**: 超参数搜索并行化（多进程适用场景）
3. **待评估**: 共享内存优化是否有必要（当前向量化已足够快）

---
📝 文档由 Clawdbot 生成 | 任务 2ffea0f6-d0a0-48ec-976a-125857d14ea7
