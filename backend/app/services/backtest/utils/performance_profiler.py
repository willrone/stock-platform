"""
回测性能分析器 - 详细的性能统计和瓶颈分析工具

提供细粒度的性能监控，包括：
- 各阶段耗时统计
- 函数调用统计
- 内存使用分析
- 并行化效率分析
- 数据库操作统计
"""

import json
import time
import tracemalloc
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import psutil
from loguru import logger


@dataclass
class StageMetrics:
    """阶段性能指标"""

    name: str
    start_time: float
    end_time: Optional[float] = None
    duration: float = 0.0
    memory_before: float = 0.0
    memory_after: float = 0.0
    memory_peak: float = 0.0
    cpu_before: float = 0.0
    cpu_after: float = 0.0
    cpu_avg: float = 0.0
    call_count: int = 0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FunctionCallStats:
    """函数调用统计"""

    name: str
    call_count: int = 0
    total_time: float = 0.0
    avg_time: float = 0.0
    min_time: float = float("inf")
    max_time: float = 0.0


@dataclass
class ParallelEfficiencyMetrics:
    """并行化效率指标"""

    sequential_time: float = 0.0
    parallel_time: float = 0.0
    speedup: float = 0.0
    efficiency: float = 0.0
    worker_count: int = 0


class BacktestPerformanceProfiler:
    """回测性能分析器"""

    def __init__(self, enable_memory_tracking: bool = True):
        """
        初始化性能分析器

        Args:
            enable_memory_tracking: 是否启用内存跟踪（会增加开销）
        """
        self.enable_memory_tracking = enable_memory_tracking
        self.process = psutil.Process()

        # 阶段统计
        self.stages: Dict[str, StageMetrics] = {}
        self.current_stage: Optional[str] = None

        # 函数调用统计
        self.function_calls: Dict[str, FunctionCallStats] = {}

        # 并行化效率统计
        self.parallel_metrics: Dict[str, ParallelEfficiencyMetrics] = {}

        # 数据库操作统计
        self.db_operations: List[Dict[str, Any]] = []

        # 内存跟踪
        self.memory_snapshots: List[Dict[str, Any]] = []
        if enable_memory_tracking:
            tracemalloc.start()

        # 总体统计
        self.start_time = time.perf_counter()
        self.total_signals = 0
        self.total_trades = 0
        self.total_trading_days = 0

        logger.info("回测性能分析器初始化完成")

    def start_backtest(self):
        """开始回测性能分析"""
        self.start_time = time.perf_counter()
        self.start_stage("total_backtest")
        logger.info("开始回测性能分析")

    def end_backtest(self):
        """结束回测性能分析"""
        if "total_backtest" in self.stages:
            self.end_stage("total_backtest")

        if self.enable_memory_tracking:
            tracemalloc.stop()

        logger.info("回测性能分析完成")

    def start_stage(self, stage_name: str, details: Optional[Dict[str, Any]] = None):
        """
        开始监控一个阶段

        Args:
            stage_name: 阶段名称
            details: 阶段详细信息
        """
        if stage_name in self.stages:
            logger.warning(f"阶段 {stage_name} 已存在，将覆盖")

        memory_before = self._get_memory_usage()
        cpu_before = self._get_cpu_usage()

        self.stages[stage_name] = StageMetrics(
            name=stage_name,
            start_time=time.perf_counter(),
            memory_before=memory_before,
            cpu_before=cpu_before,
            details=details or {},
        )
        self.current_stage = stage_name

        logger.debug(f"开始监控阶段: {stage_name}")

    def end_stage(
        self, stage_name: str, details: Optional[Dict[str, Any]] = None
    ) -> StageMetrics:
        """
        结束监控一个阶段

        Args:
            stage_name: 阶段名称
            details: 阶段详细信息

        Returns:
            阶段性能指标
        """
        if stage_name not in self.stages:
            logger.warning(f"阶段 {stage_name} 不存在")
            return StageMetrics(name=stage_name, start_time=time.perf_counter())

        stage = self.stages[stage_name]
        end_time = time.perf_counter()

        stage.end_time = end_time
        stage.duration = end_time - stage.start_time
        stage.memory_after = self._get_memory_usage()
        stage.cpu_after = self._get_cpu_usage()
        stage.cpu_avg = (stage.cpu_before + stage.cpu_after) / 2

        # 更新详细信息
        if details:
            stage.details.update(details)

        # 获取内存峰值
        if self.enable_memory_tracking:
            current, peak = tracemalloc.get_traced_memory()
            stage.memory_peak = peak / 1024 / 1024  # 转换为MB

        logger.info(
            f"阶段 {stage_name} 完成: "
            f"耗时={stage.duration:.2f}秒, "
            f"内存={stage.memory_after:.2f}MB, "
            f"CPU={stage.cpu_avg:.1f}%"
        )

        if self.current_stage == stage_name:
            self.current_stage = None

        return stage

    def record_function_call(self, func_name: str, duration: float):
        """
        记录函数调用

        Args:
            func_name: 函数名称
            duration: 调用耗时（秒）
        """
        if func_name not in self.function_calls:
            self.function_calls[func_name] = FunctionCallStats(name=func_name)

        stats = self.function_calls[func_name]
        stats.call_count += 1
        stats.total_time += duration
        stats.avg_time = stats.total_time / stats.call_count
        stats.min_time = min(stats.min_time, duration)
        stats.max_time = max(stats.max_time, duration)

    def record_parallel_efficiency(
        self,
        operation_name: str,
        sequential_time: float,
        parallel_time: float,
        worker_count: int,
    ):
        """
        记录并行化效率

        Args:
            operation_name: 操作名称
            sequential_time: 顺序执行时间
            parallel_time: 并行执行时间
            worker_count: 工作线程/进程数
        """
        speedup = sequential_time / parallel_time if parallel_time > 0 else 0
        efficiency = (speedup / worker_count * 100) if worker_count > 0 else 0

        self.parallel_metrics[operation_name] = ParallelEfficiencyMetrics(
            sequential_time=sequential_time,
            parallel_time=parallel_time,
            speedup=speedup,
            efficiency=efficiency,
            worker_count=worker_count,
        )

        logger.info(
            f"并行化效率 {operation_name}: " f"加速比={speedup:.2f}x, " f"效率={efficiency:.1f}%"
        )

    def record_db_operation(
        self, operation: str, duration: float, details: Optional[Dict] = None
    ):
        """
        记录数据库操作

        Args:
            operation: 操作类型（SELECT, INSERT, UPDATE等）
            duration: 操作耗时（秒）
            details: 详细信息
        """
        self.db_operations.append(
            {
                "operation": operation,
                "duration": duration,
                "timestamp": time.perf_counter(),
                "details": details or {},
            }
        )

    def take_memory_snapshot(self, label: str):
        """
        记录内存快照

        Args:
            label: 快照标签
        """
        if not self.enable_memory_tracking:
            return

        current, peak = tracemalloc.get_traced_memory()
        self.memory_snapshots.append(
            {
                "label": label,
                "timestamp": time.perf_counter(),
                "current_mb": current / 1024 / 1024,
                "peak_mb": peak / 1024 / 1024,
                "rss_mb": self._get_memory_usage(),
            }
        )

    def update_backtest_stats(self, signals: int = 0, trades: int = 0, days: int = 0):
        """
        更新回测统计信息

        Args:
            signals: 生成的信号数
            trades: 执行的交易数
            days: 处理的交易日数
        """
        self.total_signals += signals
        self.total_trades += trades
        self.total_trading_days = max(self.total_trading_days, days)

    def _get_memory_usage(self) -> float:
        """获取当前内存使用（MB）"""
        try:
            return self.process.memory_info().rss / 1024 / 1024
        except Exception:
            return 0.0

    def _get_cpu_usage(self) -> float:
        """获取当前CPU使用率（%）"""
        try:
            return self.process.cpu_percent(interval=0.1)
        except Exception:
            return 0.0

    def generate_report(self) -> Dict[str, Any]:
        """
        生成性能分析报告

        Returns:
            详细的性能报告
        """
        total_time = time.perf_counter() - self.start_time

        # 计算各阶段占比
        stage_percentages = {}
        for stage_name, stage in self.stages.items():
            if stage.duration > 0:
                stage_percentages[stage_name] = stage.duration / total_time * 100

        # 数据库操作统计
        db_stats = self._analyze_db_operations()

        # 函数调用统计（按总耗时排序）
        top_functions = sorted(
            self.function_calls.values(), key=lambda x: x.total_time, reverse=True
        )[:20]

        report = {
            "summary": {
                "total_time": total_time,
                "total_signals": self.total_signals,
                "total_trades": self.total_trades,
                "total_trading_days": self.total_trading_days,
                "signals_per_second": self.total_signals / total_time
                if total_time > 0
                else 0,
                "trades_per_second": self.total_trades / total_time
                if total_time > 0
                else 0,
                "days_per_second": self.total_trading_days / total_time
                if total_time > 0
                else 0,
            },
            "stages": {
                name: {
                    "duration": stage.duration,
                    "percentage": stage_percentages.get(name, 0),
                    "memory_before_mb": stage.memory_before,
                    "memory_after_mb": stage.memory_after,
                    "memory_peak_mb": stage.memory_peak,
                    "cpu_avg_percent": stage.cpu_avg,
                    "details": stage.details,
                }
                for name, stage in self.stages.items()
            },
            "function_calls": {
                func.name: {
                    "call_count": func.call_count,
                    "total_time": func.total_time,
                    "avg_time": func.avg_time,
                    "min_time": func.min_time,
                    "max_time": func.max_time,
                }
                for func in top_functions
            },
            "parallel_efficiency": {
                name: {
                    "sequential_time": metric.sequential_time,
                    "parallel_time": metric.parallel_time,
                    "speedup": metric.speedup,
                    "efficiency_percent": metric.efficiency,
                    "worker_count": metric.worker_count,
                }
                for name, metric in self.parallel_metrics.items()
            },
            "database_operations": db_stats,
            "memory_snapshots": self.memory_snapshots,
            "timestamp": datetime.utcnow().isoformat(),
        }

        return report

    def _analyze_db_operations(self) -> Dict[str, Any]:
        """分析数据库操作统计"""
        if not self.db_operations:
            return {}

        operations_by_type = defaultdict(list)
        for op in self.db_operations:
            operations_by_type[op["operation"]].append(op["duration"])

        stats = {}
        for op_type, durations in operations_by_type.items():
            stats[op_type] = {
                "count": len(durations),
                "total_time": sum(durations),
                "avg_time": sum(durations) / len(durations),
                "min_time": min(durations),
                "max_time": max(durations),
            }

        return stats

    def print_summary(self):
        """打印性能摘要"""
        report = self.generate_report()
        summary = report["summary"]

        print("=" * 80)
        print("回测性能分析摘要")
        print("=" * 80)
        print(f"总执行时间: {summary['total_time']:.2f}秒")
        print(f"总信号数: {summary['total_signals']}")
        print(f"总交易数: {summary['total_trades']}")
        print(f"总交易日数: {summary['total_trading_days']}")
        print(f"信号生成速度: {summary['signals_per_second']:.2f} 信号/秒")
        print(f"交易执行速度: {summary['trades_per_second']:.2f} 交易/秒")
        print(f"处理速度: {summary['days_per_second']:.2f} 天/秒")

        print("\n各阶段性能:")
        for stage_name, stage_data in report["stages"].items():
            print(f"  {stage_name}:")
            print(
                f"    耗时: {stage_data['duration']:.2f}秒 ({stage_data['percentage']:.1f}%)"
            )
            print(
                f"    内存: {stage_data['memory_after_mb']:.2f}MB (峰值: {stage_data['memory_peak_mb']:.2f}MB)"
            )
            print(f"    CPU: {stage_data['cpu_avg_percent']:.1f}%")

        if report["function_calls"]:
            print("\n最耗时的函数调用 (Top 10):")
            for i, (func_name, func_data) in enumerate(
                list(report["function_calls"].items())[:10], 1
            ):
                print(f"  {i}. {func_name}:")
                print(f"     调用次数: {func_data['call_count']}")
                print(f"     总耗时: {func_data['total_time']:.4f}秒")
                print(f"     平均耗时: {func_data['avg_time']:.4f}秒")

        if report["parallel_efficiency"]:
            print("\n并行化效率:")
            for op_name, metrics in report["parallel_efficiency"].items():
                print(f"  {op_name}:")
                print(f"    加速比: {metrics['speedup']:.2f}x")
                print(f"    效率: {metrics['efficiency_percent']:.1f}%")

        if report["database_operations"]:
            print("\n数据库操作统计:")
            for op_type, stats in report["database_operations"].items():
                print(f"  {op_type}:")
                print(f"    操作次数: {stats['count']}")
                print(f"    总耗时: {stats['total_time']:.4f}秒")
                print(f"    平均耗时: {stats['avg_time']*1000:.2f}ms")

        print("=" * 80)

    def save_report(self, filepath: str):
        """
        保存性能报告到文件

        Args:
            filepath: 文件路径（JSON格式）
        """
        report = self.generate_report()

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)
            logger.info(f"性能报告已保存到: {filepath}")
        except Exception as e:
            logger.error(f"保存性能报告失败: {e}")


# 性能分析装饰器
def profile_function(profiler: Optional[BacktestPerformanceProfiler] = None):
    """
    函数性能分析装饰器

    Usage:
        @profile_function(profiler)
        def my_function():
            pass
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            if profiler is None:
                return func(*args, **kwargs)

            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.perf_counter() - start
                profiler.record_function_call(func.__name__, duration)

        return wrapper

    return decorator


# 上下文管理器
class PerformanceContext:
    """性能分析上下文管理器"""

    def __init__(
        self,
        profiler: BacktestPerformanceProfiler,
        stage_name: str,
        details: Optional[Dict] = None,
    ):
        self.profiler = profiler
        self.stage_name = stage_name
        self.details = details

    def __enter__(self):
        self.profiler.start_stage(self.stage_name, self.details)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.profiler.end_stage(self.stage_name)
        return False
