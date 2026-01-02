"""
健康检查和性能测试服务
提供部署后的自动验证和性能基准测试
"""
import asyncio
import time
import statistics
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import numpy as np
from pathlib import Path
from loguru import logger

class HealthStatus(Enum):
    """健康状态"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

class TestType(Enum):
    """测试类型"""
    CONNECTIVITY = "connectivity"
    FUNCTIONALITY = "functionality"
    PERFORMANCE = "performance"
    LOAD = "load"
    STRESS = "stress"

@dataclass
class HealthCheckResult:
    """健康检查结果"""
    check_name: str
    status: HealthStatus
    message: str
    duration_ms: float
    timestamp: datetime
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'check_name': self.check_name,
            'status': self.status.value,
            'message': self.message,
            'duration_ms': self.duration_ms,
            'timestamp': self.timestamp.isoformat(),
            'details': self.details
        }

@dataclass
class PerformanceMetrics:
    """性能指标"""
    response_time_ms: float
    throughput_rps: float
    error_rate: float
    cpu_usage: float
    memory_usage_mb: float
    success_count: int
    error_count: int
    total_requests: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'response_time_ms': self.response_time_ms,
            'throughput_rps': self.throughput_rps,
            'error_rate': self.error_rate,
            'cpu_usage': self.cpu_usage,
            'memory_usage_mb': self.memory_usage_mb,
            'success_count': self.success_count,
            'error_count': self.error_count,
            'total_requests': self.total_requests
        }

@dataclass
class PerformanceTestResult:
    """性能测试结果"""
    test_name: str
    test_type: TestType
    duration_seconds: float
    metrics: PerformanceMetrics
    baseline_metrics: Optional[PerformanceMetrics]
    passed: bool
    message: str
    timestamp: datetime
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'test_name': self.test_name,
            'test_type': self.test_type.value,
            'duration_seconds': self.duration_seconds,
            'metrics': self.metrics.to_dict(),
            'baseline_metrics': self.baseline_metrics.to_dict() if self.baseline_metrics else None,
            'passed': self.passed,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'details': self.details
        }

class ModelHealthChecker:
    """模型健康检查器"""
    
    def __init__(self):
        self.checks: Dict[str, Callable] = {}
        self.check_history: List[HealthCheckResult] = []
        
    def register_check(self, name: str, check_func: Callable, description: str = ""):
        """注册健康检查函数"""
        self.checks[name] = {
            'func': check_func,
            'description': description
        }
        logger.info(f"注册健康检查: {name}")
    
    async def run_check(self, check_name: str, **kwargs) -> HealthCheckResult:
        """运行单个健康检查"""
        if check_name not in self.checks:
            return HealthCheckResult(
                check_name=check_name,
                status=HealthStatus.UNKNOWN,
                message=f"未找到检查: {check_name}",
                duration_ms=0.0,
                timestamp=datetime.now()
            )
        
        start_time = time.time()
        
        try:
            check_info = self.checks[check_name]
            check_func = check_info['func']
            
            if asyncio.iscoroutinefunction(check_func):
                result = await check_func(**kwargs)
            else:
                result = check_func(**kwargs)
            
            duration_ms = (time.time() - start_time) * 1000
            
            # 解析结果
            if isinstance(result, dict):
                status = HealthStatus(result.get('status', 'unknown'))
                message = result.get('message', '检查完成')
                details = result.get('details', {})
            elif isinstance(result, bool):
                status = HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY
                message = '检查通过' if result else '检查失败'
                details = {}
            else:
                status = HealthStatus.UNKNOWN
                message = f'未知结果类型: {type(result)}'
                details = {'result': str(result)}
            
            health_result = HealthCheckResult(
                check_name=check_name,
                status=status,
                message=message,
                duration_ms=duration_ms,
                timestamp=datetime.now(),
                details=details
            )
            
            self.check_history.append(health_result)
            return health_result
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            
            health_result = HealthCheckResult(
                check_name=check_name,
                status=HealthStatus.UNHEALTHY,
                message=f"检查异常: {str(e)}",
                duration_ms=duration_ms,
                timestamp=datetime.now(),
                details={'error': str(e), 'error_type': type(e).__name__}
            )
            
            self.check_history.append(health_result)
            logger.error(f"健康检查失败 {check_name}: {e}")
            return health_result
    
    async def run_all_checks(self, **kwargs) -> Dict[str, HealthCheckResult]:
        """运行所有健康检查"""
        results = {}
        
        for check_name in self.checks:
            result = await self.run_check(check_name, **kwargs)
            results[check_name] = result
        
        return results
    
    def get_overall_health(self, results: Dict[str, HealthCheckResult]) -> HealthStatus:
        """获取总体健康状态"""
        if not results:
            return HealthStatus.UNKNOWN
        
        statuses = [result.status for result in results.values()]
        
        if HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        elif HealthStatus.HEALTHY in statuses:
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.UNKNOWN
    
    def get_check_history(self, check_name: Optional[str] = None, limit: int = 100) -> List[HealthCheckResult]:
        """获取检查历史"""
        history = self.check_history
        
        if check_name:
            history = [r for r in history if r.check_name == check_name]
        
        return history[-limit:]

class PerformanceTester:
    """性能测试器"""
    
    def __init__(self):
        self.test_functions: Dict[str, Callable] = {}
        self.baseline_metrics: Dict[str, PerformanceMetrics] = {}
        self.test_history: List[PerformanceTestResult] = []
    
    def register_test(self, name: str, test_func: Callable, description: str = ""):
        """注册性能测试函数"""
        self.test_functions[name] = {
            'func': test_func,
            'description': description
        }
        logger.info(f"注册性能测试: {name}")
    
    def set_baseline(self, test_name: str, metrics: PerformanceMetrics):
        """设置基准指标"""
        self.baseline_metrics[test_name] = metrics
        logger.info(f"设置基准指标: {test_name}")
    
    async def run_performance_test(
        self, 
        test_name: str, 
        duration_seconds: int = 60,
        concurrent_users: int = 1,
        **kwargs
    ) -> PerformanceTestResult:
        """运行性能测试"""
        if test_name not in self.test_functions:
            return PerformanceTestResult(
                test_name=test_name,
                test_type=TestType.PERFORMANCE,
                duration_seconds=0.0,
                metrics=PerformanceMetrics(0, 0, 1.0, 0, 0, 0, 1, 1),
                baseline_metrics=None,
                passed=False,
                message=f"未找到测试: {test_name}",
                timestamp=datetime.now()
            )
        
        start_time = time.time()
        
        try:
            test_info = self.test_functions[test_name]
            test_func = test_info['func']
            
            # 运行性能测试
            if asyncio.iscoroutinefunction(test_func):
                metrics = await test_func(
                    duration_seconds=duration_seconds,
                    concurrent_users=concurrent_users,
                    **kwargs
                )
            else:
                metrics = test_func(
                    duration_seconds=duration_seconds,
                    concurrent_users=concurrent_users,
                    **kwargs
                )
            
            actual_duration = time.time() - start_time
            
            # 获取基准指标
            baseline_metrics = self.baseline_metrics.get(test_name)
            
            # 评估测试结果
            passed, message = self._evaluate_performance(metrics, baseline_metrics)
            
            result = PerformanceTestResult(
                test_name=test_name,
                test_type=TestType.PERFORMANCE,
                duration_seconds=actual_duration,
                metrics=metrics,
                baseline_metrics=baseline_metrics,
                passed=passed,
                message=message,
                timestamp=datetime.now(),
                details={
                    'requested_duration': duration_seconds,
                    'concurrent_users': concurrent_users
                }
            )
            
            self.test_history.append(result)
            return result
            
        except Exception as e:
            actual_duration = time.time() - start_time
            
            result = PerformanceTestResult(
                test_name=test_name,
                test_type=TestType.PERFORMANCE,
                duration_seconds=actual_duration,
                metrics=PerformanceMetrics(0, 0, 1.0, 0, 0, 0, 1, 1),
                baseline_metrics=None,
                passed=False,
                message=f"测试异常: {str(e)}",
                timestamp=datetime.now(),
                details={'error': str(e), 'error_type': type(e).__name__}
            )
            
            self.test_history.append(result)
            logger.error(f"性能测试失败 {test_name}: {e}")
            return result
    
    def _evaluate_performance(
        self, 
        metrics: PerformanceMetrics, 
        baseline: Optional[PerformanceMetrics]
    ) -> Tuple[bool, str]:
        """评估性能测试结果"""
        if not baseline:
            # 没有基准，只检查基本指标
            if metrics.error_rate > 0.05:  # 错误率超过5%
                return False, f"错误率过高: {metrics.error_rate:.2%}"
            if metrics.response_time_ms > 5000:  # 响应时间超过5秒
                return False, f"响应时间过长: {metrics.response_time_ms:.0f}ms"
            return True, "性能测试通过（无基准对比）"
        
        # 与基准对比
        issues = []
        
        # 响应时间不应超过基准的150%
        if metrics.response_time_ms > baseline.response_time_ms * 1.5:
            issues.append(f"响应时间退化: {metrics.response_time_ms:.0f}ms vs {baseline.response_time_ms:.0f}ms")
        
        # 吞吐量不应低于基准的80%
        if metrics.throughput_rps < baseline.throughput_rps * 0.8:
            issues.append(f"吞吐量下降: {metrics.throughput_rps:.1f} vs {baseline.throughput_rps:.1f} RPS")
        
        # 错误率不应超过基准的2倍
        if metrics.error_rate > baseline.error_rate * 2:
            issues.append(f"错误率上升: {metrics.error_rate:.2%} vs {baseline.error_rate:.2%}")
        
        if issues:
            return False, "; ".join(issues)
        else:
            return True, "性能测试通过（符合基准要求）"
    
    async def run_load_test(
        self, 
        test_name: str, 
        max_users: int = 100,
        ramp_up_seconds: int = 60,
        **kwargs
    ) -> PerformanceTestResult:
        """运行负载测试"""
        logger.info(f"开始负载测试: {test_name}, 最大用户数: {max_users}")
        
        start_time = time.time()
        
        try:
            # 模拟负载测试
            response_times = []
            error_count = 0
            success_count = 0
            
            # 逐步增加负载
            for current_users in range(1, max_users + 1, max(1, max_users // 10)):
                logger.info(f"负载测试 - 当前用户数: {current_users}")
                
                # 运行短期测试
                test_result = await self.run_performance_test(
                    test_name, 
                    duration_seconds=10,
                    concurrent_users=current_users,
                    **kwargs
                )
                
                response_times.append(test_result.metrics.response_time_ms)
                success_count += test_result.metrics.success_count
                error_count += test_result.metrics.error_count
                
                # 如果错误率过高，停止测试
                if test_result.metrics.error_rate > 0.1:
                    logger.warning(f"负载测试提前停止 - 错误率过高: {test_result.metrics.error_rate:.2%}")
                    break
                
                await asyncio.sleep(ramp_up_seconds / 10)  # 等待间隔
            
            actual_duration = time.time() - start_time
            
            # 计算总体指标
            avg_response_time = statistics.mean(response_times) if response_times else 0
            total_requests = success_count + error_count
            error_rate = error_count / total_requests if total_requests > 0 else 0
            throughput = total_requests / actual_duration if actual_duration > 0 else 0
            
            metrics = PerformanceMetrics(
                response_time_ms=avg_response_time,
                throughput_rps=throughput,
                error_rate=error_rate,
                cpu_usage=0.0,  # 需要实际监控
                memory_usage_mb=0.0,  # 需要实际监控
                success_count=success_count,
                error_count=error_count,
                total_requests=total_requests
            )
            
            # 评估负载测试结果
            passed = error_rate < 0.05 and avg_response_time < 2000
            message = "负载测试通过" if passed else f"负载测试失败 - 错误率: {error_rate:.2%}, 响应时间: {avg_response_time:.0f}ms"
            
            result = PerformanceTestResult(
                test_name=test_name,
                test_type=TestType.LOAD,
                duration_seconds=actual_duration,
                metrics=metrics,
                baseline_metrics=self.baseline_metrics.get(test_name),
                passed=passed,
                message=message,
                timestamp=datetime.now(),
                details={
                    'max_users': max_users,
                    'ramp_up_seconds': ramp_up_seconds,
                    'response_time_trend': response_times
                }
            )
            
            self.test_history.append(result)
            return result
            
        except Exception as e:
            actual_duration = time.time() - start_time
            
            result = PerformanceTestResult(
                test_name=test_name,
                test_type=TestType.LOAD,
                duration_seconds=actual_duration,
                metrics=PerformanceMetrics(0, 0, 1.0, 0, 0, 0, 1, 1),
                baseline_metrics=None,
                passed=False,
                message=f"负载测试异常: {str(e)}",
                timestamp=datetime.now(),
                details={'error': str(e)}
            )
            
            self.test_history.append(result)
            logger.error(f"负载测试失败 {test_name}: {e}")
            return result
    
    def get_test_history(self, test_name: Optional[str] = None, limit: int = 100) -> List[PerformanceTestResult]:
        """获取测试历史"""
        history = self.test_history
        
        if test_name:
            history = [r for r in history if r.test_name == test_name]
        
        return history[-limit:]

class HealthMonitor:
    """健康监控服务"""
    
    def __init__(self):
        self.health_checker = ModelHealthChecker()
        self.performance_tester = PerformanceTester()
        
        # 注册默认检查和测试
        self._register_default_checks()
        self._register_default_tests()
        
        logger.info("健康监控服务初始化完成")
    
    def _register_default_checks(self):
        """注册默认健康检查"""
        
        # 模型文件检查
        async def model_file_check(model_path: str = None, **kwargs) -> Dict[str, Any]:
            if not model_path:
                return {
                    'status': 'unhealthy',
                    'message': '未提供模型路径',
                    'details': {}
                }
            
            model_path = Path(model_path)
            if not model_path.exists():
                return {
                    'status': 'unhealthy',
                    'message': f'模型文件不存在: {model_path}',
                    'details': {'model_path': str(model_path)}
                }
            
            file_size = model_path.stat().st_size
            return {
                'status': 'healthy',
                'message': '模型文件存在',
                'details': {
                    'model_path': str(model_path),
                    'file_size_bytes': file_size,
                    'file_size_mb': file_size / (1024 * 1024)
                }
            }
        
        self.health_checker.register_check('model_file', model_file_check, "检查模型文件是否存在")
        
        # 模型加载检查
        async def model_load_check(model_path: str = None, **kwargs) -> Dict[str, Any]:
            if not model_path:
                return {
                    'status': 'unhealthy',
                    'message': '未提供模型路径'
                }
            
            try:
                # 尝试加载模型（这里需要根据实际模型类型实现）
                import pickle
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                
                return {
                    'status': 'healthy',
                    'message': '模型加载成功',
                    'details': {
                        'model_type': type(model).__name__,
                        'has_predict': hasattr(model, 'predict')
                    }
                }
            except Exception as e:
                return {
                    'status': 'unhealthy',
                    'message': f'模型加载失败: {str(e)}',
                    'details': {'error': str(e)}
                }
        
        self.health_checker.register_check('model_load', model_load_check, "检查模型是否可以正常加载")
        
        # API连通性检查
        async def api_connectivity_check(**kwargs) -> Dict[str, Any]:
            try:
                # 模拟API连通性检查
                await asyncio.sleep(0.1)  # 模拟网络延迟
                
                return {
                    'status': 'healthy',
                    'message': 'API连通性正常',
                    'details': {'response_time_ms': 100}
                }
            except Exception as e:
                return {
                    'status': 'unhealthy',
                    'message': f'API连通性检查失败: {str(e)}',
                    'details': {'error': str(e)}
                }
        
        self.health_checker.register_check('api_connectivity', api_connectivity_check, "检查API连通性")
    
    def _register_default_tests(self):
        """注册默认性能测试"""
        
        # 模型预测性能测试
        async def model_prediction_test(
            duration_seconds: int = 60,
            concurrent_users: int = 1,
            model_path: str = None,
            **kwargs
        ) -> PerformanceMetrics:
            
            if not model_path:
                raise ValueError("未提供模型路径")
            
            # 模拟性能测试
            start_time = time.time()
            response_times = []
            success_count = 0
            error_count = 0
            
            # 生成测试数据
            test_data = np.random.randn(100, 10)  # 模拟特征数据
            
            while time.time() - start_time < duration_seconds:
                try:
                    # 模拟并发请求
                    tasks = []
                    for _ in range(concurrent_users):
                        tasks.append(self._simulate_prediction_request(model_path, test_data))
                    
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    for result in results:
                        if isinstance(result, Exception):
                            error_count += 1
                        else:
                            success_count += 1
                            response_times.append(result)
                    
                    await asyncio.sleep(0.1)  # 控制请求频率
                    
                except Exception as e:
                    error_count += 1
                    logger.error(f"性能测试请求失败: {e}")
            
            # 计算指标
            total_requests = success_count + error_count
            avg_response_time = statistics.mean(response_times) if response_times else 0
            error_rate = error_count / total_requests if total_requests > 0 else 0
            throughput = total_requests / duration_seconds
            
            return PerformanceMetrics(
                response_time_ms=avg_response_time,
                throughput_rps=throughput,
                error_rate=error_rate,
                cpu_usage=0.0,  # 需要实际监控
                memory_usage_mb=0.0,  # 需要实际监控
                success_count=success_count,
                error_count=error_count,
                total_requests=total_requests
            )
        
        self.performance_tester.register_test('model_prediction', model_prediction_test, "模型预测性能测试")
    
    async def _simulate_prediction_request(self, model_path: str, test_data: np.ndarray) -> float:
        """模拟预测请求"""
        start_time = time.time()
        
        try:
            # 模拟模型预测
            await asyncio.sleep(0.01 + np.random.exponential(0.05))  # 模拟预测时间
            
            # 随机选择测试数据
            sample_data = test_data[np.random.randint(0, len(test_data))]
            
            # 模拟预测结果
            prediction = np.random.random()
            
            return (time.time() - start_time) * 1000  # 返回响应时间（毫秒）
            
        except Exception as e:
            logger.error(f"模拟预测请求失败: {e}")
            raise
    
    async def run_health_checks(self, **kwargs) -> Dict[str, Any]:
        """运行健康检查"""
        results = await self.health_checker.run_all_checks(**kwargs)
        overall_status = self.health_checker.get_overall_health(results)
        
        return {
            'overall_status': overall_status.value,
            'checks': {name: result.to_dict() for name, result in results.items()},
            'timestamp': datetime.now().isoformat()
        }
    
    async def run_performance_test(self, test_name: str, **kwargs) -> Dict[str, Any]:
        """运行性能测试"""
        result = await self.performance_tester.run_performance_test(test_name, **kwargs)
        return result.to_dict()
    
    async def run_load_test(self, test_name: str, **kwargs) -> Dict[str, Any]:
        """运行负载测试"""
        result = await self.performance_tester.run_load_test(test_name, **kwargs)
        return result.to_dict()
    
    def set_performance_baseline(self, test_name: str, metrics: Dict[str, Any]):
        """设置性能基准"""
        baseline_metrics = PerformanceMetrics(**metrics)
        self.performance_tester.set_baseline(test_name, baseline_metrics)
    
    def get_health_history(self, check_name: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """获取健康检查历史"""
        history = self.health_checker.get_check_history(check_name, limit)
        return [result.to_dict() for result in history]
    
    def get_performance_history(self, test_name: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """获取性能测试历史"""
        history = self.performance_tester.get_test_history(test_name, limit)
        return [result.to_dict() for result in history]

# 全局健康监控实例
health_monitor = HealthMonitor()