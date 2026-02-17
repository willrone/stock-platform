"""
性能追踪模块
负责性能分析、计时、内存快照等
"""

from typing import Any, Dict, Optional

# 性能监控（可选导入）
try:
    from ..utils.performance_profiler import (
        BacktestPerformanceProfiler,
        PerformanceContext,
    )

    PERFORMANCE_PROFILING_AVAILABLE = True
except ImportError:
    PERFORMANCE_PROFILING_AVAILABLE = False
    BacktestPerformanceProfiler = None
    PerformanceContext = None


class PerformanceTracker:
    """性能追踪器"""

    def __init__(self, enable_profiling: bool = False):
        """
        初始化性能追踪器

        Args:
            enable_profiling: 是否启用性能分析
        """
        self.enable_profiling = enable_profiling and PERFORMANCE_PROFILING_AVAILABLE
        self.profiler: Optional[BacktestPerformanceProfiler] = None
        self.perf_breakdown: Dict[str, float] = {}

    def start_backtest(self):
        """开始回测性能追踪"""
        if self.enable_profiling:
            self.profiler = BacktestPerformanceProfiler(enable_memory_tracking=True)
            self.profiler.start_backtest()
            self.profiler.take_memory_snapshot("backtest_start")

    def end_backtest(self):
        """结束回测性能追踪"""
        if self.enable_profiling and self.profiler:
            self.profiler.end_backtest()
            self.profiler.take_memory_snapshot("backtest_end")

    def start_stage(self, stage_name: str, metadata: Dict[str, Any] = None):
        """开始一个阶段"""
        if self.enable_profiling and self.profiler:
            self.profiler.start_stage(stage_name, metadata)

    def end_stage(self, stage_name: str, metadata: Dict[str, Any] = None):
        """结束一个阶段"""
        if self.enable_profiling and self.profiler:
            self.profiler.end_stage(stage_name, metadata)

    def take_memory_snapshot(self, label: str):
        """拍摄内存快照"""
        if self.enable_profiling and self.profiler:
            self.profiler.take_memory_snapshot(label)

    def generate_report(self) -> Dict[str, Any]:
        """生成性能报告"""
        if self.enable_profiling and self.profiler:
            return self.profiler.generate_report()
        return {}

    def print_summary(self):
        """打印性能摘要"""
        if self.enable_profiling and self.profiler:
            self.profiler.print_summary()

    def save_report(self, filepath: str):
        """保存性能报告到文件"""
        if self.enable_profiling and self.profiler:
            self.profiler.save_report(filepath)
