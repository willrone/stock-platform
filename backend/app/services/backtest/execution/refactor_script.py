#!/usr/bin/env python3
"""
BacktestExecutor 重构脚本
自动将 backtest_executor.py 拆分为多个模块
"""

import re
from pathlib import Path
from typing import List, Tuple

def read_file(filepath: str) -> List[str]:
    """读取文件内容"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.readlines()

def write_file(filepath: str, lines: List[str]):
    """写入文件"""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(lines)

def extract_lines(lines: List[str], start: int, end: int) -> List[str]:
    """提取指定行范围的内容（行号从1开始）"""
    return lines[start-1:end]

def extract_imports(lines: List[str]) -> List[str]:
    """提取导入语句"""
    imports = []
    in_import_block = True
    
    for line in lines:
        stripped = line.strip()
        
        # 跳过文档字符串和注释
        if stripped.startswith('"""') or stripped.startswith("'''"):
            continue
        if stripped.startswith('#'):
            continue
        if not stripped:
            continue
            
        # 导入语句
        if stripped.startswith('import ') or stripped.startswith('from '):
            imports.append(line)
        elif in_import_block and stripped and not stripped.startswith('import') and not stripped.startswith('from'):
            # 遇到非导入语句，结束导入块
            break
    
    return imports

def create_data_preprocessor(lines: List[str]) -> str:
    """创建 data_preprocessor.py"""
    content = []
    
    # 文件头
    content.append('"""\n')
    content.append('数据预处理模块\n')
    content.append('负责回测数据的预处理、索引构建、信号预计算等\n')
    content.append('"""\n\n')
    
    # 导入语句
    imports = extract_imports(lines)
    content.extend(imports)
    content.append('\n')
    
    # 添加模块级函数 _multiprocess_precompute_worker (Line 42-106)
    content.append('# 多进程预计算 worker 函数\n')
    worker_lines = extract_lines(lines, 42, 106)
    content.extend(worker_lines)
    content.append('\n\n')
    
    # 创建 DataPreprocessor 类
    content.append('class DataPreprocessor:\n')
    content.append('    """数据预处理器"""\n\n')
    content.append('    def __init__(self, enable_parallel: bool = True, max_workers: int = 8, use_multiprocessing: bool = True):\n')
    content.append('        """\n')
    content.append('        初始化数据预处理器\n\n')
    content.append('        Args:\n')
    content.append('            enable_parallel: 是否启用并行化\n')
    content.append('            max_workers: 最大工作线程/进程数\n')
    content.append('            use_multiprocessing: 是否使用多进程\n')
    content.append('        """\n')
    content.append('        self.enable_parallel = enable_parallel\n')
    content.append('        self.max_workers = max_workers\n')
    content.append('        self.use_multiprocessing = use_multiprocessing\n\n')
    
    # 提取方法
    methods = [
        (549, 565, '_get_trading_calendar'),
        (566, 576, '_build_date_index'),
        (577, 597, '_warm_indicator_cache'),
        (598, 684, '_precompute_strategy_signals'),
        (685, 797, '_extract_precomputed_signals_to_dict'),
        (798, 933, '_build_aligned_arrays'),
        (934, 989, '_precompute_signals_multiprocess'),
    ]
    
    for start, end, method_name in methods:
        method_lines = extract_lines(lines, start, end)
        content.extend(method_lines)
        content.append('\n')
    
    return ''.join(content)

def create_backtest_loop_executor(lines: List[str]) -> str:
    """创建 backtest_loop_executor.py"""
    content = []
    
    # 文件头
    content.append('"""\n')
    content.append('回测循环执行模块\n')
    content.append('负责核心回测循环的执行\n')
    content.append('"""\n\n')
    
    # 导入语句
    imports = extract_imports(lines)
    content.extend(imports)
    content.append('\n')
    
    # 创建 BacktestLoopExecutor 类
    content.append('class BacktestLoopExecutor:\n')
    content.append('    """回测循环执行器"""\n\n')
    content.append('    def __init__(self):\n')
    content.append('        """初始化回测循环执行器"""\n')
    content.append('        pass\n\n')
    
    # 提取 _execute_backtest_loop 方法 (Line 990-2045)
    loop_lines = extract_lines(lines, 990, 2045)
    content.extend(loop_lines)
    content.append('\n')
    
    return ''.join(content)

def create_report_generator(lines: List[str]) -> str:
    """创建 report_generator.py"""
    content = []
    
    # 文件头
    content.append('"""\n')
    content.append('回测报告生成模块\n')
    content.append('负责生成回测报告、计算指标等\n')
    content.append('"""\n\n')
    
    # 导入语句
    imports = extract_imports(lines)
    content.extend(imports)
    content.append('\n')
    
    # 创建 BacktestReportGenerator 类
    content.append('class BacktestReportGenerator:\n')
    content.append('    """回测报告生成器"""\n\n')
    content.append('    def __init__(self):\n')
    content.append('        """初始化报告生成器"""\n')
    content.append('        pass\n\n')
    
    # 提取方法
    methods = [
        (2046, 2200, '_generate_backtest_report'),
        (2201, 2414, '_rebalance_topk_buffer'),
        (2415, 2541, '_calculate_additional_metrics'),
    ]
    
    for start, end, method_name in methods:
        method_lines = extract_lines(lines, start, end)
        content.extend(method_lines)
        content.append('\n')
    
    return ''.join(content)

def create_validators(lines: List[str]) -> str:
    """创建 validators.py"""
    content = []
    
    # 文件头
    content.append('"""\n')
    content.append('参数验证和统计模块\n')
    content.append('"""\n\n')
    
    # 导入语句
    imports = extract_imports(lines)
    content.extend(imports)
    content.append('\n')
    
    # 提取验证方法（作为独立函数）
    methods = [
        (2542, 2601, 'validate_backtest_parameters'),
        (2613, 2704, '_get_execution_failure_reason'),
    ]
    
    for start, end, method_name in methods:
        method_lines = extract_lines(lines, start, end)
        # 移除缩进（从类方法转为模块函数）
        adjusted_lines = []
        for line in method_lines:
            if line.startswith('    '):
                adjusted_lines.append(line[4:])  # 移除4个空格
            else:
                adjusted_lines.append(line)
        content.extend(adjusted_lines)
        content.append('\n')
    
    # 添加 get_execution_statistics 函数
    content.append('def get_execution_statistics(execution_stats: dict) -> dict:\n')
    content.append('    """获取执行统计信息"""\n')
    content.append('    return {\n')
    content.append('        "total_backtests": execution_stats.get("total_backtests", 0),\n')
    content.append('        "successful_backtests": execution_stats.get("successful_backtests", 0),\n')
    content.append('        "failed_backtests": execution_stats.get("failed_backtests", 0),\n')
    content.append('    }\n')
    
    return ''.join(content)

def create_performance_tracker(lines: List[str]) -> str:
    """创建 performance_tracker.py"""
    content = []
    
    # 文件头
    content.append('"""\n')
    content.append('性能追踪模块\n')
    content.append('负责性能分析、计时、内存快照等\n')
    content.append('"""\n\n')
    
    content.append('import time\n')
    content.append('from typing import Optional, Dict, Any\n')
    content.append('from loguru import logger\n\n')
    
    content.append('# 性能监控（可选导入）\n')
    content.append('try:\n')
    content.append('    from ..utils.performance_profiler import (\n')
    content.append('        BacktestPerformanceProfiler,\n')
    content.append('        PerformanceContext,\n')
    content.append('    )\n')
    content.append('    PERFORMANCE_PROFILING_AVAILABLE = True\n')
    content.append('except ImportError:\n')
    content.append('    PERFORMANCE_PROFILING_AVAILABLE = False\n')
    content.append('    BacktestPerformanceProfiler = None\n')
    content.append('    PerformanceContext = None\n\n')
    
    content.append('class PerformanceTracker:\n')
    content.append('    """性能追踪器"""\n\n')
    content.append('    def __init__(self, enable_profiling: bool = False):\n')
    content.append('        """\n')
    content.append('        初始化性能追踪器\n\n')
    content.append('        Args:\n')
    content.append('            enable_profiling: 是否启用性能分析\n')
    content.append('        """\n')
    content.append('        self.enable_profiling = enable_profiling and PERFORMANCE_PROFILING_AVAILABLE\n')
    content.append('        self.profiler: Optional[BacktestPerformanceProfiler] = None\n')
    content.append('        self.perf_breakdown: Dict[str, float] = {}\n\n')
    content.append('    def start_backtest(self):\n')
    content.append('        """开始回测性能追踪"""\n')
    content.append('        if self.enable_profiling:\n')
    content.append('            self.profiler = BacktestPerformanceProfiler(enable_memory_tracking=True)\n')
    content.append('            self.profiler.start_backtest()\n')
    content.append('            self.profiler.take_memory_snapshot("backtest_start")\n\n')
    content.append('    def end_backtest(self):\n')
    content.append('        """结束回测性能追踪"""\n')
    content.append('        if self.enable_profiling and self.profiler:\n')
    content.append('            self.profiler.end_backtest()\n')
    content.append('            self.profiler.take_memory_snapshot("backtest_end")\n\n')
    content.append('    def start_stage(self, stage_name: str, metadata: Dict[str, Any] = None):\n')
    content.append('        """开始一个阶段"""\n')
    content.append('        if self.enable_profiling and self.profiler:\n')
    content.append('            self.profiler.start_stage(stage_name, metadata)\n\n')
    content.append('    def end_stage(self, stage_name: str, metadata: Dict[str, Any] = None):\n')
    content.append('        """结束一个阶段"""\n')
    content.append('        if self.enable_profiling and self.profiler:\n')
    content.append('            self.profiler.end_stage(stage_name, metadata)\n\n')
    content.append('    def take_memory_snapshot(self, label: str):\n')
    content.append('        """拍摄内存快照"""\n')
    content.append('        if self.enable_profiling and self.profiler:\n')
    content.append('            self.profiler.take_memory_snapshot(label)\n\n')
    content.append('    def generate_report(self) -> Dict[str, Any]:\n')
    content.append('        """生成性能报告"""\n')
    content.append('        if self.enable_profiling and self.profiler:\n')
    content.append('            return self.profiler.generate_report()\n')
    content.append('        return {}\n\n')
    content.append('    def print_summary(self):\n')
    content.append('        """打印性能摘要"""\n')
    content.append('        if self.enable_profiling and self.profiler:\n')
    content.append('            self.profiler.print_summary()\n\n')
    content.append('    def save_report(self, filepath: str):\n')
    content.append('        """保存性能报告到文件"""\n')
    content.append('        if self.enable_profiling and self.profiler:\n')
    content.append('            self.profiler.save_report(filepath)\n')
    
    return ''.join(content)

def main():
    """主函数"""
    print("=== BacktestExecutor 重构脚本 ===\n")
    
    # 读取原始文件
    original_file = 'backtest_executor.py'
    print(f"1. 读取原始文件: {original_file}")
    lines = read_file(original_file)
    print(f"   总行数: {len(lines)}")
    
    # 创建新模块
    print("\n2. 创建新模块文件...")
    
    print("   - 创建 data_preprocessor.py")
    data_preprocessor_content = create_data_preprocessor(lines)
    write_file('data_preprocessor.py', data_preprocessor_content)
    print(f"     ✓ 已创建 ({len(data_preprocessor_content.split(chr(10)))} 行)")
    
    print("   - 创建 backtest_loop_executor.py")
    loop_executor_content = create_backtest_loop_executor(lines)
    write_file('backtest_loop_executor.py', loop_executor_content)
    print(f"     ✓ 已创建 ({len(loop_executor_content.split(chr(10)))} 行)")
    
    print("   - 创建 report_generator.py")
    report_generator_content = create_report_generator(lines)
    write_file('report_generator.py', report_generator_content)
    print(f"     ✓ 已创建 ({len(report_generator_content.split(chr(10)))} 行)")
    
    print("   - 创建 validators.py")
    validators_content = create_validators(lines)
    write_file('validators.py', validators_content)
    print(f"     ✓ 已创建 ({len(validators_content.split(chr(10)))} 行)")
    
    print("   - 创建 performance_tracker.py")
    performance_tracker_content = create_performance_tracker(lines)
    write_file('performance_tracker.py', performance_tracker_content)
    print(f"     ✓ 已创建 ({len(performance_tracker_content.split(chr(10)))} 行)")
    
    print("\n3. 重构完成！")
    print("\n下一步:")
    print("   1. 检查生成的文件")
    print("   2. 重构 backtest_executor.py 主文件")
    print("   3. 验证语法和导入")
    print("   4. 运行测试")

if __name__ == '__main__':
    main()
