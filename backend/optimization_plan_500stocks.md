# 500只股票3年回测性能优化方案

**目标**: 将 500 只股票 × 3 年（约 750 个交易日）的回测时间从当前压缩到 **3 分钟以内**

**当前基准**: 50 只股票 × 747 天 = 143.96 秒
**推算**: 500 只股票 × 750 天 ≈ **1440 秒（24 分钟）**
**目标**: **180 秒（3 分钟）** → 需要 **8x 加速**

---

## 一、性能瓶颈分析

### 1.1 当前瓶颈（50只股票测试）

| 阶段 | 耗时 | 占比 | 瓶颈点 |
|------|------|------|--------|
| 回测执行 | 138.47s | 96.19% | **核心瓶颈** |
| 报告生成 | 2.99s | 2.08% | 次要 |
| 数据加载 | 1.04s | 0.72% | 可忽略 |

**关键问题**:
- `generate_signals_core_work` 平均 346ms/次，累计 251.52s
- 并行效率仅 50%（10 workers 只有 5x 加速）
- 向量化预计算已实现，但未充分利用

### 1.2 500只股票推算

```
当前: 50 stocks × 750 days = 143.96s
推算: 500 stocks × 750 days = 1439.6s (24分钟)
目标: 500 stocks × 750 days = 180s (3分钟)
需要加速: 1439.6 / 180 = 8x
```

---

## 二、优化策略（8x 加速路径）

### 🎯 策略 1: 批量向量化处理 (3x 加速)

**当前问题**: 逐日生成信号，每天都要调用策略
**优化方案**: 一次性对所有股票×所有日期进行向量化计算

```python
# 当前方式（慢）
for date in dates:
    for stock in stocks:
        signal = strategy.generate_signal(stock, date)  # 500×750 = 375,000 次调用

# 优化方式（快）
all_signals = strategy.precompute_all_signals_batch(all_stocks_data)  # 1 次调用
```

**实现要点**:
1. 将所有股票数据合并为 MultiIndex DataFrame
2. 使用 `groupby` + 向量化操作一次性计算所有信号
3. 预先构建信号查询索引（stock_code, date）

**预期收益**: 375,000 次函数调用 → 1 次批量计算 = **3x 加速**

---

### ⚡ 策略 2: 真正的多进程并行 (2x 加速)

**当前问题**: 
- 多进程开销大于收益（因为单次计算太快）
- 数据序列化成为瓶颈

**优化方案**: 
1. **按股票分组并行**（而非按日期）
   - 每个进程处理 50-100 只股票的完整 3 年数据
   - 减少进程间通信次数
   
2. **共享内存优化**
   ```python
   from multiprocessing import shared_memory
   
   # 将价格数据放入共享内存
   shm = shared_memory.SharedMemory(create=True, size=data.nbytes)
   shared_array = np.ndarray(data.shape, dtype=data.dtype, buffer=shm.buf)
   shared_array[:] = data[:]
   ```

3. **进程池预热**
   - 启动时创建进程池，避免每次回测都重新创建

**预期收益**: 并行效率从 50% 提升到 80% = **2x 加速**

---

### 🚀 策略 3: 数据结构优化 (1.5x 加速)

**当前问题**: DataFrame 操作开销大

**优化方案**:
1. **NumPy 结构化数组**
   ```python
   # 替代 DataFrame
   dtype = [('date', 'datetime64[D]'), ('open', 'f4'), ('high', 'f4'), 
            ('low', 'f4'), ('close', 'f4'), ('volume', 'f8')]
   data = np.array(records, dtype=dtype)
   ```

2. **预计算索引映射**
   ```python
   # 日期 → 索引的快速查找
   date_to_idx = {date: idx for idx, date in enumerate(dates)}
   stock_to_idx = {stock: idx for idx, stock in enumerate(stocks)}
   ```

3. **信号存储优化**
   - 使用 int8 存储信号（-1/0/1）而非对象
   - 使用位图存储布尔标志

**预期收益**: **1.5x 加速**

---

### 🔧 策略 4: 策略计算优化 (1.3x 加速)

**优化点**:
1. **技术指标缓存**
   ```python
   @lru_cache(maxsize=1000)
   def get_indicator(stock_code, indicator_name):
       return compute_indicator(stock_code, indicator_name)
   ```

2. **增量计算**
   - RSI/MACD 等指标支持滑动窗口增量更新
   - 避免每次都重新计算整个历史

3. **JIT 编译**
   ```python
   from numba import jit
   
   @jit(nopython=True)
   def calculate_signals_fast(prices, params):
       # 纯 NumPy 计算，编译为机器码
       pass
   ```

**预期收益**: **1.3x 加速**

---

## 三、实施计划

### Phase 1: 批量向量化（最高优先级）⏱️ 2-3天

**任务清单**:
- [ ] 实现 `BatchSignalGenerator` 类
  - [ ] 支持 MultiIndex DataFrame 输入
  - [ ] 实现 `precompute_all_signals_batch()` 方法
  - [ ] 构建 (stock, date) → signal 的快速查询索引
- [ ] 重构 `BacktestExecutor.run()`
  - [ ] 预先批量生成所有信号
  - [ ] 回测循环只做信号查询和交易执行
- [ ] 性能测试
  - [ ] 50 只股票基准测试
  - [ ] 500 只股票压力测试

**验收标准**: 50 只股票回测时间 < 50 秒（当前 144 秒）

---

### Phase 2: 多进程并行优化 ⏱️ 2-3天

**任务清单**:
- [ ] 实现共享内存数据管理器
  - [ ] `SharedMemoryDataManager` 类
  - [ ] 价格数据共享内存映射
- [ ] 按股票分组并行
  - [ ] 每个进程处理 50-100 只股票
  - [ ] 进程池预热机制
- [ ] 性能测试
  - [ ] 对比多线程 vs 多进程
  - [ ] 测量并行效率

**验收标准**: 并行效率 > 80%，500 只股票 < 120 秒

---

### Phase 3: 数据结构优化 ⏱️ 1-2天

**任务清单**:
- [ ] NumPy 结构化数组替代 DataFrame
- [ ] 预计算索引映射
- [ ] 信号存储优化（int8 + 位图）

**验收标准**: 500 只股票 < 90 秒

---

### Phase 4: 策略计算优化 ⏱️ 1-2天

**任务清单**:
- [ ] 技术指标 LRU 缓存
- [ ] 增量计算实现
- [ ] Numba JIT 编译关键函数

**验收标准**: 500 只股票 < 60 秒

---

## 四、最终目标验收

| 指标 | 当前值 | 目标值 | 状态 |
|------|--------|--------|------|
| 50 只股票 × 750 天 | 143.96s | < 20s | ⏳ |
| 500 只股票 × 750 天 | ~1440s | < 180s | ⏳ |
| 并行效率 | 50% | > 80% | ⏳ |
| 信号生成平均耗时 | 346ms | < 50ms | ⏳ |
| 内存占用 | - | < 4GB | ⏳ |

---

## 五、风险与备选方案

### 风险点
1. **内存不足**: 500 只股票 × 750 天 × 多个指标可能超过内存
   - **缓解**: 分批处理（每批 100 只股票）
   
2. **精度损失**: NumPy float32 可能影响计算精度
   - **缓解**: 关键计算保持 float64

3. **代码复杂度**: 过度优化可能影响可维护性
   - **缓解**: 保留原有接口，优化实现层

### 备选方案
- 如果 Python 优化不够，考虑 Rust/C++ 扩展
- 使用 GPU 加速（CuPy）
- 分布式计算（Ray/Dask）

---

## 六、开发指南

### 6.1 关键文件

```
backend/app/services/backtest/
├── execution/
│   ├── backtest_executor.py          # 主执行器（需重构）
│   ├── batch_signal_generator.py     # 新增：批量信号生成器
│   └── shared_memory_manager.py      # 新增：共享内存管理
├── strategies/
│   └── base_strategy.py              # 需添加批量接口
└── utils/
    └── performance_profiler.py       # 性能监控
```

### 6.2 性能测试脚本

```bash
# 基准测试
python backend/scripts/benchmark_backtest.py --stocks 50 --days 750

# 压力测试
python backend/scripts/benchmark_backtest.py --stocks 500 --days 750

# 性能分析
python -m cProfile -o profile.stats backend/scripts/benchmark_backtest.py
python -m pstats profile.stats
```

---

📝 **文档创建**: 2026-02-03
🎯 **预计完成**: 2026-02-10（7 天）
👤 **负责人**: Clawdbot + 荣辉
