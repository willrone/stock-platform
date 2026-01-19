"""
Qlib 性能监控模块

用于监控回测流程的性能，包括执行时间、内存使用和并行计算效率
"""

import time
import psutil
import datetime
import pandas as pd
from typing import Dict, List, Optional, Any
from loguru import logger


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.start_time = time.time()
        self.stages = {}
        self.metrics = {
            'execution_time': 0.0,
            'memory_usage': 0.0,
            'cpu_usage': 0.0,
            'stages': {}
        }
        self.process = psutil.Process()
        logger.info("性能监控器初始化完成")
    
    def start_stage(self, stage_name: str):
        """开始监控一个阶段"""
        self.stages[stage_name] = {
            'start_time': time.time(),
            'start_memory': self._get_memory_usage(),
            'start_cpu': self._get_cpu_usage()
        }
        logger.debug(f"开始监控阶段: {stage_name}")
    
    def end_stage(self, stage_name: str) -> Dict[str, Any]:
        """结束监控一个阶段并返回性能指标"""
        if stage_name not in self.stages:
            logger.warning(f"阶段 {stage_name} 未开始监控")
            return {}
        
        stage_data = self.stages[stage_name]
        end_time = time.time()
        end_memory = self._get_memory_usage()
        end_cpu = self._get_cpu_usage()
        
        # 计算阶段指标
        duration = end_time - stage_data['start_time']
        memory_diff = end_memory - stage_data['start_memory']
        cpu_diff = end_cpu - stage_data['start_cpu']
        
        stage_metrics = {
            'duration': duration,
            'memory_usage_mb': end_memory,
            'memory_diff_mb': memory_diff,
            'cpu_usage_percent': end_cpu,
            'cpu_diff_percent': cpu_diff
        }
        
        # 保存阶段指标
        self.metrics['stages'][stage_name] = stage_metrics
        
        logger.info(f"阶段 {stage_name} 完成: 耗时={duration:.2f}秒, 内存={end_memory:.2f}MB, CPU={end_cpu:.1f}%")
        return stage_metrics
    
    def _get_memory_usage(self) -> float:
        """获取当前内存使用情况（MB）"""
        try:
            return self.process.memory_info().rss / 1024 / 1024
        except Exception as e:
            logger.warning(f"获取内存使用情况失败: {e}")
            return 0.0
    
    def _get_cpu_usage(self) -> float:
        """获取当前CPU使用情况（%）"""
        try:
            return self.process.cpu_percent(interval=0.1)
        except Exception as e:
            logger.warning(f"获取CPU使用情况失败: {e}")
            return 0.0
    
    def get_overall_metrics(self) -> Dict[str, Any]:
        """获取整体性能指标"""
        self.metrics['execution_time'] = time.time() - self.start_time
        self.metrics['memory_usage'] = self._get_memory_usage()
        self.metrics['cpu_usage'] = self._get_cpu_usage()
        
        return self.metrics
    
    def print_summary(self):
        """打印性能摘要"""
        overall = self.get_overall_metrics()
        
        logger.info("=" * 60)
        logger.info("性能监控摘要")
        logger.info("=" * 60)
        logger.info(f"总执行时间: {overall['execution_time']:.2f}秒")
        logger.info(f"峰值内存使用: {overall['memory_usage']:.2f}MB")
        logger.info(f"CPU使用率: {overall['cpu_usage']:.1f}%")
        logger.info("\n各阶段性能:")
        
        for stage_name, metrics in overall['stages'].items():
            logger.info(f"  {stage_name}:")
            logger.info(f"    耗时: {metrics['duration']:.2f}秒")
            logger.info(f"    内存: {metrics['memory_usage_mb']:.2f}MB")
            logger.info(f"    CPU: {metrics['cpu_usage_percent']:.1f}%")
        
        logger.info("=" * 60)


class BenchmarkRunner:
    """基准测试运行器"""
    
    def __init__(self):
        self.benchmark_results = []
        logger.info("基准测试运行器初始化完成")
    
    async def run_benchmark(
        self,
        test_name: str,
        func,
        *args,
        **kwargs
    ) -> Dict[str, Any]:
        """运行基准测试"""
        monitor = PerformanceMonitor()
        
        logger.info(f"开始基准测试: {test_name}")
        monitor.start_stage(test_name)
        
        # 执行测试函数
        result = None
        error = None
        
        try:
            if hasattr(func, '__await__'):
                # 异步函数
                result = await func(*args, **kwargs)
            else:
                # 同步函数
                result = func(*args, **kwargs)
        except Exception as e:
            error = str(e)
            logger.error(f"基准测试执行失败: {error}")
        
        monitor.end_stage(test_name)
        metrics = monitor.get_overall_metrics()
        
        # 保存测试结果
        test_result = {
            'test_name': test_name,
            'timestamp': datetime.datetime.now(),
            'metrics': metrics,
            'error': error
        }
        
        self.benchmark_results.append(test_result)
        
        # 打印测试结果
        logger.info(f"基准测试 {test_name} 完成")
        monitor.print_summary()
        
        if error:
            logger.error(f"测试失败原因: {error}")
        
        return test_result
    
    def compare_results(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        """比较多个基准测试结果"""
        if not results:
            return pd.DataFrame()
        
        # 构建比较表格
        data = []
        for result in results:
            test_name = result['test_name']
            timestamp = result['timestamp']
            metrics = result['metrics']
            error = result['error']
            
            row = {
                'test_name': test_name,
                'timestamp': timestamp,
                'execution_time': metrics.get('execution_time', 0),
                'memory_usage': metrics.get('memory_usage', 0),
                'cpu_usage': metrics.get('cpu_usage', 0),
                'error': error
            }
            
            # 添加各阶段的耗时
            for stage_name, stage_metrics in metrics.get('stages', {}).items():
                row[f'{stage_name}_duration'] = stage_metrics.get('duration', 0)
            
            data.append(row)
        
        df = pd.DataFrame(data)
        return df
    
    def save_results(self, filename: str):
        """保存基准测试结果到文件"""
        if not self.benchmark_results:
            logger.warning("没有基准测试结果可保存")
            return
        
        try:
            df = self.compare_results(self.benchmark_results)
            df.to_csv(filename, index=False)
            logger.info(f"基准测试结果已保存到: {filename}")
        except Exception as e:
            logger.error(f"保存基准测试结果失败: {e}")


# 全局性能监控器实例
_global_performance_monitor = None
_global_benchmark_runner = None

def get_performance_monitor() -> PerformanceMonitor:
    """获取全局性能监控器实例"""
    global _global_performance_monitor
    if _global_performance_monitor is None:
        _global_performance_monitor = PerformanceMonitor()
    return _global_performance_monitor

def get_benchmark_runner() -> BenchmarkRunner:
    """获取全局基准测试运行器实例"""
    global _global_benchmark_runner
    if _global_benchmark_runner is None:
        _global_benchmark_runner = BenchmarkRunner()
    return _global_benchmark_runner
