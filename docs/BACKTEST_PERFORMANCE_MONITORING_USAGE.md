# 回测性能监控使用指南

## 概述

回测执行器已集成性能监控功能，可以自动收集和分析回测任务的性能数据，帮助识别性能瓶颈。

## 启用性能监控

### 方法1：在创建执行器时启用

```python
from app.services.backtest.backtest_executor import BacktestExecutor

# 创建执行器并启用性能监控
executor = BacktestExecutor(
    data_dir="backend/data",
    enable_parallel=True,
    enable_performance_profiling=True  # 启用性能分析
)

# 执行回测（会自动收集性能数据）
result = await executor.run_backtest(
    strategy_name="ma_crossover",
    stock_codes=["000001.SZ", "000002.SZ"],
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2023, 12, 31),
    strategy_config={},
    task_id="test_task"
)

# 性能报告会自动包含在result中
performance_data = result.get('performance_analysis')
```

### 方法2：通过环境变量启用（推荐用于生产环境）

```python
import os

# 设置环境变量
os.environ['ENABLE_BACKTEST_PERFORMANCE_PROFILING'] = 'true'

# 创建执行器时会自动读取环境变量
executor = BacktestExecutor(data_dir="backend/data")
```

## 性能报告内容

性能监控会自动收集以下数据：

### 1. 阶段性能统计

- **initialization**: 初始化阶段
- **strategy_setup**: 策略设置阶段
- **data_loading**: 数据加载阶段
- **backtest_execution**: 回测执行阶段（核心）
- **metrics_calculation**: 指标计算阶段
- **report_generation**: 报告生成阶段

每个阶段包含：
- 耗时（duration）
- 内存使用（memory_before, memory_after, memory_peak）
- CPU使用率（cpu_avg）
- 详细信息（details）

### 2. 函数调用统计

记录关键函数的调用次数和耗时：
- `generate_signals`: 信号生成函数
- `execute_signal`: 交易执行函数
- `execute_trades_batch`: 批量交易执行

### 3. 并行化效率分析

分析并行执行的效率：
- 顺序执行时间 vs 并行执行时间
- 加速比（speedup）
- 并行效率（efficiency）

### 4. 内存快照

在关键时间点记录内存使用：
- backtest_start
- after_data_loading
- after_backtest_execution
- backtest_end

## 查看性能报告

### 1. 从回测结果中获取

```python
result = await executor.run_backtest(...)

# 获取性能分析报告
performance_report = result.get('performance_analysis')

if performance_report:
    print(f"总执行时间: {performance_report['summary']['total_time']:.2f}秒")
    print(f"信号生成速度: {performance_report['summary']['signals_per_second']:.2f} 信号/秒")
    
    # 查看各阶段耗时
    for stage_name, stage_data in performance_report['stages'].items():
        print(f"{stage_name}: {stage_data['duration']:.2f}秒 ({stage_data['percentage']:.1f}%)")
```

### 2. 从保存的文件中读取

性能报告会自动保存到：
```
backend/data/performance_reports/backtest_{task_id}_performance.json
```

```python
import json

# 读取性能报告
with open('backend/data/performance_reports/backtest_test_task_performance.json', 'r') as f:
    performance_report = json.load(f)
    
# 分析性能数据
print(json.dumps(performance_report, indent=2, ensure_ascii=False))
```

### 3. 查看控制台输出

启用性能监控后，会在控制台自动打印性能摘要：

```
================================================================================
回测性能分析摘要
================================================================================
总执行时间: 125.34秒
总信号数: 1523
总交易数: 456
总交易日数: 252
信号生成速度: 12.15 信号/秒
交易执行速度: 3.64 交易/秒
处理速度: 2.01 天/秒

各阶段性能:
  strategy_setup:
    耗时: 0.12秒 (0.1%)
    内存: 245.32MB (峰值: 245.32MB)
    CPU: 15.2%
  data_loading:
    耗时: 2.45秒 (2.0%)
    内存: 512.67MB (峰值: 512.67MB)
    CPU: 25.3%
  backtest_execution:
    耗时: 118.23秒 (94.3%)
    内存: 523.45MB (峰值: 623.12MB)
    CPU: 85.6%
  ...

最耗时的函数调用 (Top 10):
  1. generate_signals:
     调用次数: 2520
     总耗时: 95.23秒
     平均耗时: 0.0378秒
  ...
```

## 性能优化建议

基于性能报告，可以采取以下优化措施：

### 1. 如果数据加载是瓶颈

```python
# 优化建议：
# - 使用Parquet格式替代CSV
# - 启用数据缓存
# - 预加载常用数据
```

### 2. 如果信号生成是瓶颈

```python
# 优化建议：
# - 优化策略计算逻辑
# - 使用向量化操作
# - 考虑使用多进程替代多线程
```

### 3. 如果并行化效率低

```python
# 优化建议：
# - 调整worker数量
# - 检查GIL限制
# - 考虑使用多进程
```

## 性能监控开销

性能监控会带来一定的开销：

- **内存开销**: 约增加 10-50MB（取决于监控的详细程度）
- **CPU开销**: 约增加 1-5%（主要是时间统计和内存快照）
- **磁盘开销**: 每个回测任务生成一个JSON报告文件（通常 < 100KB）

**建议**：
- 开发调试时：启用性能监控
- 生产环境：根据需要选择性启用
- 性能测试：必须启用

## 禁用性能监控

如果不需要性能监控，可以：

```python
# 创建执行器时不启用
executor = BacktestExecutor(
    data_dir="backend/data",
    enable_performance_profiling=False  # 默认就是False
)
```

## 高级用法

### 自定义性能监控

如果需要更详细的监控，可以直接使用性能分析器：

```python
from app.services.backtest.performance_profiler import BacktestPerformanceProfiler

profiler = BacktestPerformanceProfiler(enable_memory_tracking=True)
profiler.start_backtest()

# 自定义阶段监控
profiler.start_stage("custom_stage")
# ... 执行操作 ...
profiler.end_stage("custom_stage")

# 记录函数调用
import time
start = time.perf_counter()
# ... 执行函数 ...
profiler.record_function_call("my_function", time.perf_counter() - start)

# 生成报告
report = profiler.generate_report()
profiler.print_summary()
```

## 故障排查

### 问题1：性能监控未启用

**症状**：回测结果中没有 `performance_analysis` 字段

**解决**：
```python
# 检查是否启用了性能监控
executor = BacktestExecutor(enable_performance_profiling=True)
```

### 问题2：性能报告文件未生成

**症状**：控制台显示保存失败

**解决**：
```python
# 确保目录存在
import os
os.makedirs("backend/data/performance_reports", exist_ok=True)
```

### 问题3：内存使用异常

**症状**：内存使用量异常高

**解决**：
```python
# 禁用内存跟踪（减少开销）
executor = BacktestExecutor(
    enable_performance_profiling=True,
    # 但性能分析器内部可以禁用内存跟踪
)
# 注意：需要在性能分析器初始化时设置 enable_memory_tracking=False
```

## 示例：完整的性能分析流程

```python
import asyncio
from datetime import datetime
from app.services.backtest.backtest_executor import BacktestExecutor
from app.services.backtest.backtest_engine import BacktestConfig

async def analyze_backtest_performance():
    # 创建执行器并启用性能监控
    executor = BacktestExecutor(
        data_dir="backend/data",
        enable_parallel=True,
        enable_performance_profiling=True
    )
    
    # 执行回测
    result = await executor.run_backtest(
        strategy_name="ma_crossover",
        stock_codes=["000001.SZ", "000002.SZ", "000003.SZ"],
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 12, 31),
        strategy_config={
            "short_window": 5,
            "long_window": 20
        },
        backtest_config=BacktestConfig(
            initial_cash=100000.0,
            commission_rate=0.0003,
            slippage_rate=0.0001
        ),
        task_id="performance_test"
    )
    
    # 分析性能报告
    perf_report = result.get('performance_analysis')
    if perf_report:
        print("\n=== 性能分析结果 ===")
        print(f"总耗时: {perf_report['summary']['total_time']:.2f}秒")
        print(f"最耗时的阶段:")
        
        # 按耗时排序
        stages = sorted(
            perf_report['stages'].items(),
            key=lambda x: x[1]['duration'],
            reverse=True
        )
        
        for stage_name, stage_data in stages[:3]:
            print(f"  {stage_name}: {stage_data['duration']:.2f}秒")
        
        # 分析并行化效率
        if perf_report.get('parallel_efficiency'):
            print("\n=== 并行化效率 ===")
            for op_name, metrics in perf_report['parallel_efficiency'].items():
                print(f"{op_name}:")
                print(f"  加速比: {metrics['speedup']:.2f}x")
                print(f"  效率: {metrics['efficiency_percent']:.1f}%")
    
    return result

# 运行
if __name__ == "__main__":
    asyncio.run(analyze_backtest_performance())
```

## 总结

性能监控功能已完全集成到回测执行器中，只需在创建执行器时启用即可。性能数据会自动收集、分析和保存，帮助您快速定位性能瓶颈并进行优化。
