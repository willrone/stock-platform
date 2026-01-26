"""
回测性能分析器使用示例

展示如何在回测执行器中集成性能分析功能
"""

import asyncio
from datetime import datetime, timedelta

from ..core.backtest_engine import BacktestConfig
from ..execution.backtest_executor import BacktestExecutor
from .performance_profiler import (
    BacktestPerformanceProfiler,
    PerformanceContext,
    profile_function,
)


async def example_basic_usage():
    """基础使用示例"""
    # 创建性能分析器
    profiler = BacktestPerformanceProfiler(enable_memory_tracking=True)

    # 开始回测分析
    profiler.start_backtest()

    try:
        # 监控数据加载阶段
        profiler.start_stage("data_loading")
        # ... 执行数据加载 ...
        profiler.end_stage("data_loading")

        # 监控信号生成阶段
        profiler.start_stage("signal_generation")
        # ... 执行信号生成 ...
        profiler.end_stage("signal_generation")

        # 监控交易执行阶段
        profiler.start_stage("trade_execution")
        # ... 执行交易 ...
        profiler.end_stage("trade_execution")

    finally:
        profiler.end_backtest()

    # 生成并打印报告
    profiler.print_summary()

    # 保存报告
    profiler.save_report("backtest_performance.json")


async def example_with_context_manager():
    """使用上下文管理器示例"""
    profiler = BacktestPerformanceProfiler()
    profiler.start_backtest()

    try:
        # 使用上下文管理器自动管理阶段
        with PerformanceContext(profiler, "data_loading"):
            # ... 执行数据加载 ...
            pass

        with PerformanceContext(profiler, "signal_generation"):
            # ... 执行信号生成 ...
            pass

    finally:
        profiler.end_backtest()

    profiler.print_summary()


async def example_with_decorator():
    """使用装饰器示例"""
    profiler = BacktestPerformanceProfiler()

    # 使用装饰器监控函数调用
    @profile_function(profiler)
    def expensive_operation():
        # ... 耗时操作 ...
        pass

    # 调用函数，自动记录性能
    expensive_operation()


async def example_integrated_with_executor():
    """与回测执行器集成的示例"""
    profiler = BacktestPerformanceProfiler(enable_memory_tracking=True)
    executor = BacktestExecutor(data_dir="backend/data")

    profiler.start_backtest()

    try:
        # 执行回测
        result = await executor.run_backtest(
            strategy_name="ma_crossover",
            stock_codes=["000001.SZ", "000002.SZ"],
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
            strategy_config={},
            backtest_config=BacktestConfig(),
            task_id="test_task",
        )

        # 更新统计信息
        profiler.update_backtest_stats(
            signals=result.get("total_signals", 0),
            trades=result.get("executed_trades", 0),
            days=result.get("trading_days", 0),
        )

    finally:
        profiler.end_backtest()

    # 生成报告
    report = profiler.generate_report()
    profiler.print_summary()
    profiler.save_report("backtest_performance.json")

    return result, report


async def example_parallel_efficiency_analysis():
    """并行化效率分析示例"""
    profiler = BacktestPerformanceProfiler()

    import time
    from concurrent.futures import ThreadPoolExecutor

    def process_stock(stock_code: str):
        """处理单只股票"""
        time.sleep(0.1)  # 模拟处理时间
        return f"Processed {stock_code}"

    stock_codes = [f"00000{i}.SZ" for i in range(1, 11)]

    # 顺序执行
    start = time.perf_counter()
    results_sequential = [process_stock(code) for code in stock_codes]
    sequential_time = time.perf_counter() - start

    # 并行执行
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as executor:
        results_parallel = list(executor.map(process_stock, stock_codes))
    parallel_time = time.perf_counter() - start

    # 记录并行化效率
    profiler.record_parallel_efficiency(
        operation_name="stock_processing",
        sequential_time=sequential_time,
        parallel_time=parallel_time,
        worker_count=4,
    )

    # 打印效率报告
    report = profiler.generate_report()
    print(f"顺序执行时间: {sequential_time:.2f}秒")
    print(f"并行执行时间: {parallel_time:.2f}秒")
    print(f"加速比: {report['parallel_efficiency']['stock_processing']['speedup']:.2f}x")
    print(
        f"效率: {report['parallel_efficiency']['stock_processing']['efficiency_percent']:.1f}%"
    )


async def example_memory_analysis():
    """内存分析示例"""
    profiler = BacktestPerformanceProfiler(enable_memory_tracking=True)

    profiler.start_backtest()

    # 在不同阶段记录内存快照
    profiler.take_memory_snapshot("after_init")

    # ... 执行一些操作 ...

    profiler.take_memory_snapshot("after_data_loading")

    # ... 执行更多操作 ...

    profiler.take_memory_snapshot("after_signal_generation")

    profiler.end_backtest()

    # 查看内存快照
    report = profiler.generate_report()
    print("内存快照:")
    for snapshot in report["memory_snapshots"]:
        print(
            f"  {snapshot['label']}: {snapshot['current_mb']:.2f}MB (峰值: {snapshot['peak_mb']:.2f}MB)"
        )


if __name__ == "__main__":
    # 运行示例
    asyncio.run(example_basic_usage())
    # asyncio.run(example_integrated_with_executor())
    # asyncio.run(example_parallel_efficiency_analysis())
    # asyncio.run(example_memory_analysis())
