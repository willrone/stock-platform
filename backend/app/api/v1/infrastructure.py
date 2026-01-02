"""
基础设施监控和调度API
"""
from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging

from app.services.infrastructure.resource_monitor import resource_monitor, ResourceThresholds
from app.services.infrastructure.task_scheduler import task_scheduler, TaskPriority, ResourceRequirement
from app.services.infrastructure.deployment_manager import (
    deployment_manager, DeploymentConfig, DeploymentStrategy
)
from app.services.infrastructure.compatibility_validator import compatibility_validator
from app.services.infrastructure.health_monitor import health_monitor

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/infrastructure", tags=["基础设施"])

@router.get("/resources/current", summary="获取当前资源使用情况")
async def get_current_resources():
    """获取当前系统资源使用情况"""
    try:
        usage = resource_monitor.get_current_usage()
        return {
            "success": True,
            "data": usage.to_dict()
        }
    except Exception as e:
        logger.error(f"获取资源使用情况失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取资源使用情况失败: {str(e)}")

@router.get("/resources/history", summary="获取资源使用历史")
async def get_resource_history(
    duration_minutes: int = Query(60, description="历史数据时间范围（分钟）")
):
    """获取指定时间范围内的资源使用历史"""
    try:
        start_time = datetime.now() - timedelta(minutes=duration_minutes)
        history = resource_monitor.get_usage_history(start_time=start_time)
        
        return {
            "success": True,
            "data": {
                "history": [usage.to_dict() for usage in history],
                "count": len(history),
                "duration_minutes": duration_minutes
            }
        }
    except Exception as e:
        logger.error(f"获取资源历史失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取资源历史失败: {str(e)}")

@router.get("/resources/average", summary="获取平均资源使用情况")
async def get_average_resources(
    duration_minutes: int = Query(60, description="统计时间范围（分钟）")
):
    """获取指定时间范围内的平均资源使用情况"""
    try:
        avg_usage = resource_monitor.get_average_usage(duration_minutes)
        
        if avg_usage is None:
            return {
                "success": True,
                "data": None,
                "message": "暂无历史数据"
            }
        
        return {
            "success": True,
            "data": avg_usage.to_dict()
        }
    except Exception as e:
        logger.error(f"获取平均资源使用情况失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取平均资源使用情况失败: {str(e)}")

@router.post("/resources/check-availability", summary="检查资源可用性")
async def check_resource_availability(
    memory_gb: float = 0,
    cpu_percent: float = 0,
    gpu_memory_gb: float = 0
):
    """检查是否有足够的资源可用"""
    try:
        availability = resource_monitor.is_resource_available(
            required_memory_gb=memory_gb,
            required_cpu_percent=cpu_percent,
            required_gpu_memory_gb=gpu_memory_gb
        )
        
        return {
            "success": True,
            "data": availability
        }
    except Exception as e:
        logger.error(f"检查资源可用性失败: {e}")
        raise HTTPException(status_code=500, detail=f"检查资源可用性失败: {str(e)}")

@router.get("/resources/thresholds", summary="获取资源阈值配置")
async def get_resource_thresholds():
    """获取当前资源阈值配置"""
    try:
        thresholds = resource_monitor.thresholds
        return {
            "success": True,
            "data": {
                "cpu_warning": thresholds.cpu_warning,
                "cpu_critical": thresholds.cpu_critical,
                "memory_warning": thresholds.memory_warning,
                "memory_critical": thresholds.memory_critical,
                "disk_warning": thresholds.disk_warning,
                "disk_critical": thresholds.disk_critical,
                "gpu_memory_warning": thresholds.gpu_memory_warning,
                "gpu_memory_critical": thresholds.gpu_memory_critical
            }
        }
    except Exception as e:
        logger.error(f"获取资源阈值失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取资源阈值失败: {str(e)}")

@router.put("/resources/thresholds", summary="更新资源阈值配置")
async def update_resource_thresholds(
    cpu_warning: Optional[float] = None,
    cpu_critical: Optional[float] = None,
    memory_warning: Optional[float] = None,
    memory_critical: Optional[float] = None,
    disk_warning: Optional[float] = None,
    disk_critical: Optional[float] = None,
    gpu_memory_warning: Optional[float] = None,
    gpu_memory_critical: Optional[float] = None
):
    """更新资源阈值配置"""
    try:
        # 更新阈值
        if cpu_warning is not None:
            resource_monitor.thresholds.cpu_warning = cpu_warning
        if cpu_critical is not None:
            resource_monitor.thresholds.cpu_critical = cpu_critical
        if memory_warning is not None:
            resource_monitor.thresholds.memory_warning = memory_warning
        if memory_critical is not None:
            resource_monitor.thresholds.memory_critical = memory_critical
        if disk_warning is not None:
            resource_monitor.thresholds.disk_warning = disk_warning
        if disk_critical is not None:
            resource_monitor.thresholds.disk_critical = disk_critical
        if gpu_memory_warning is not None:
            resource_monitor.thresholds.gpu_memory_warning = gpu_memory_warning
        if gpu_memory_critical is not None:
            resource_monitor.thresholds.gpu_memory_critical = gpu_memory_critical
        
        return {
            "success": True,
            "message": "资源阈值已更新"
        }
    except Exception as e:
        logger.error(f"更新资源阈值失败: {e}")
        raise HTTPException(status_code=500, detail=f"更新资源阈值失败: {str(e)}")

@router.get("/scheduler/stats", summary="获取调度器统计信息")
async def get_scheduler_stats():
    """获取任务调度器统计信息"""
    try:
        stats = task_scheduler.get_scheduler_stats()
        return {
            "success": True,
            "data": stats
        }
    except Exception as e:
        logger.error(f"获取调度器统计信息失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取调度器统计信息失败: {str(e)}")

@router.get("/scheduler/tasks", summary="获取所有任务状态")
async def get_all_tasks():
    """获取所有任务的状态信息"""
    try:
        tasks = task_scheduler.get_all_tasks()
        return {
            "success": True,
            "data": tasks
        }
    except Exception as e:
        logger.error(f"获取任务状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取任务状态失败: {str(e)}")

@router.get("/scheduler/tasks/{task_id}", summary="获取特定任务状态")
async def get_task_status(task_id: str):
    """获取特定任务的状态信息"""
    try:
        task_status = task_scheduler.get_task_status(task_id)
        
        if task_status is None:
            raise HTTPException(status_code=404, detail="任务不存在")
        
        return {
            "success": True,
            "data": task_status
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取任务状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取任务状态失败: {str(e)}")

@router.delete("/scheduler/tasks/{task_id}", summary="取消任务")
async def cancel_task(task_id: str):
    """取消指定的任务"""
    try:
        success = task_scheduler.cancel_task(task_id)
        
        if not success:
            raise HTTPException(status_code=400, detail="任务无法取消或不存在")
        
        return {
            "success": True,
            "message": "任务已取消"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"取消任务失败: {e}")
        raise HTTPException(status_code=500, detail=f"取消任务失败: {str(e)}")

@router.post("/scheduler/start", summary="启动调度器")
async def start_scheduler():
    """启动任务调度器"""
    try:
        await task_scheduler.start()
        return {
            "success": True,
            "message": "调度器已启动"
        }
    except Exception as e:
        logger.error(f"启动调度器失败: {e}")
        raise HTTPException(status_code=500, detail=f"启动调度器失败: {str(e)}")

@router.post("/scheduler/stop", summary="停止调度器")
async def stop_scheduler():
    """停止任务调度器"""
    try:
        await task_scheduler.stop()
        return {
            "success": True,
            "message": "调度器已停止"
        }
    except Exception as e:
        logger.error(f"停止调度器失败: {e}")
        raise HTTPException(status_code=500, detail=f"停止调度器失败: {str(e)}")

@router.post("/monitoring/start", summary="启动资源监控")
async def start_monitoring(
    interval: float = Query(30.0, description="监控间隔（秒）")
):
    """启动资源监控"""
    try:
        await resource_monitor.start_monitoring(interval)
        return {
            "success": True,
            "message": f"资源监控已启动，间隔 {interval} 秒"
        }
    except Exception as e:
        logger.error(f"启动资源监控失败: {e}")
        raise HTTPException(status_code=500, detail=f"启动资源监控失败: {str(e)}")

@router.post("/monitoring/stop", summary="停止资源监控")
async def stop_monitoring():
    """停止资源监控"""
    try:
        await resource_monitor.stop_monitoring()
        return {
            "success": True,
            "message": "资源监控已停止"
        }
    except Exception as e:
        logger.error(f"停止资源监控失败: {e}")
        raise HTTPException(status_code=500, detail=f"停止资源监控失败: {str(e)}")

@router.get("/health", summary="基础设施健康检查")
async def infrastructure_health():
    """检查基础设施组件的健康状态"""
    try:
        # 检查资源监控器状态
        current_usage = resource_monitor.get_current_usage()
        alerts = resource_monitor.check_thresholds(current_usage)
        
        # 检查调度器状态
        scheduler_stats = task_scheduler.get_scheduler_stats()
        
        # 计算健康分数
        health_score = 100
        if alerts:
            critical_alerts = [a for a in alerts if a['level'] == 'critical']
            warning_alerts = [a for a in alerts if a['level'] == 'warning']
            health_score -= len(critical_alerts) * 20 + len(warning_alerts) * 10
        
        health_status = "healthy"
        if health_score < 60:
            health_status = "critical"
        elif health_score < 80:
            health_status = "warning"
        
        return {
            "success": True,
            "data": {
                "status": health_status,
                "health_score": max(0, health_score),
                "resource_usage": current_usage.to_dict(),
                "alerts": alerts,
                "scheduler_stats": scheduler_stats,
                "timestamp": datetime.now().isoformat()
            }
        }
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        raise HTTPException(status_code=500, detail=f"健康检查失败: {str(e)}")

# 部署管理API

@router.post("/deployments", summary="创建部署")
async def create_deployment(
    strategy: str,
    model_path: str,
    model_id: str,
    version: str,
    # 蓝绿部署配置
    blue_green_switch_delay: int = 30,
    # 金丝雀部署配置
    canary_traffic_percentage: float = 10.0,
    canary_duration_minutes: int = 30,
    canary_success_threshold: float = 0.95,
    # 滚动部署配置
    rolling_batch_size: int = 1,
    rolling_batch_delay: int = 60,
    # 健康检查配置
    health_check_enabled: bool = True,
    health_check_timeout: int = 300,
    health_check_interval: int = 10,
    # 自动回滚配置
    auto_rollback_enabled: bool = True,
    rollback_threshold_error_rate: float = 0.05,
    rollback_threshold_latency_ms: float = 1000.0
):
    """创建新的部署"""
    try:
        # 验证部署策略
        try:
            deployment_strategy = DeploymentStrategy(strategy)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"不支持的部署策略: {strategy}")
        
        # 创建部署配置
        config = DeploymentConfig(
            strategy=deployment_strategy,
            model_path=model_path,
            model_id=model_id,
            version=version,
            blue_green_switch_delay=blue_green_switch_delay,
            canary_traffic_percentage=canary_traffic_percentage,
            canary_duration_minutes=canary_duration_minutes,
            canary_success_threshold=canary_success_threshold,
            rolling_batch_size=rolling_batch_size,
            rolling_batch_delay=rolling_batch_delay,
            health_check_enabled=health_check_enabled,
            health_check_timeout=health_check_timeout,
            health_check_interval=health_check_interval,
            auto_rollback_enabled=auto_rollback_enabled,
            rollback_threshold_error_rate=rollback_threshold_error_rate,
            rollback_threshold_latency_ms=rollback_threshold_latency_ms
        )
        
        # 执行部署
        deployment_id = await deployment_manager.deploy(config)
        
        return {
            "success": True,
            "data": {
                "deployment_id": deployment_id,
                "message": "部署已创建"
            }
        }
    except Exception as e:
        logger.error(f"创建部署失败: {e}")
        raise HTTPException(status_code=500, detail=f"创建部署失败: {str(e)}")

@router.get("/deployments", summary="获取所有部署")
async def get_all_deployments():
    """获取所有部署记录"""
    try:
        deployments = deployment_manager.get_all_deployments()
        return {
            "success": True,
            "data": {
                "deployments": deployments,
                "count": len(deployments)
            }
        }
    except Exception as e:
        logger.error(f"获取部署列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取部署列表失败: {str(e)}")

@router.get("/deployments/{deployment_id}", summary="获取部署状态")
async def get_deployment_status(deployment_id: str):
    """获取特定部署的状态"""
    try:
        deployment = deployment_manager.get_deployment_status(deployment_id)
        
        if deployment is None:
            raise HTTPException(status_code=404, detail="部署不存在")
        
        return {
            "success": True,
            "data": deployment
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取部署状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取部署状态失败: {str(e)}")

@router.get("/deployments/active", summary="获取活跃部署")
async def get_active_deployment():
    """获取当前活跃的部署"""
    try:
        active_deployment = deployment_manager.get_active_deployment()
        
        return {
            "success": True,
            "data": active_deployment
        }
    except Exception as e:
        logger.error(f"获取活跃部署失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取活跃部署失败: {str(e)}")

@router.post("/deployments/{deployment_id}/rollback", summary="回滚部署")
async def rollback_deployment(
    deployment_id: str,
    target_version: Optional[str] = None
):
    """回滚指定的部署"""
    try:
        rollback_id = await deployment_manager.rollback(deployment_id, target_version)
        
        return {
            "success": True,
            "data": {
                "rollback_deployment_id": rollback_id,
                "original_deployment_id": deployment_id,
                "target_version": target_version,
                "message": "回滚已完成"
            }
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"回滚部署失败: {e}")
        raise HTTPException(status_code=500, detail=f"回滚部署失败: {str(e)}")

@router.get("/deployments/strategies", summary="获取支持的部署策略")
async def get_deployment_strategies():
    """获取所有支持的部署策略"""
    try:
        strategies = [
            {
                "name": strategy.value,
                "display_name": {
                    "blue_green": "蓝绿部署",
                    "canary": "金丝雀发布",
                    "rolling": "滚动部署",
                    "immediate": "立即部署"
                }.get(strategy.value, strategy.value),
                "description": {
                    "blue_green": "创建两个相同的生产环境，通过切换流量实现零停机部署",
                    "canary": "先将少量流量导向新版本，验证无问题后再全量发布",
                    "rolling": "逐步替换旧版本实例，确保服务持续可用",
                    "immediate": "直接替换当前版本，适用于开发和测试环境"
                }.get(strategy.value, "")
            }
            for strategy in DeploymentStrategy
        ]
        
        return {
            "success": True,
            "data": {
                "strategies": strategies
            }
        }
    except Exception as e:
        logger.error(f"获取部署策略失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取部署策略失败: {str(e)}")

# 兼容性验证API

@router.post("/compatibility/validate", summary="验证模型兼容性")
async def validate_model_compatibility(
    model_path: str,
    target_model_path: Optional[str] = None
):
    """验证模型兼容性"""
    try:
        # 提取目标模型元数据（如果提供）
        target_metadata = None
        if target_model_path:
            target_metadata = compatibility_validator.extract_model_metadata(target_model_path)
        
        # 执行兼容性验证
        validation_result = compatibility_validator.validate_compatibility(
            model_path, target_metadata
        )
        
        return {
            "success": True,
            "data": validation_result
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"兼容性验证失败: {e}")
        raise HTTPException(status_code=500, detail=f"兼容性验证失败: {str(e)}")

@router.get("/compatibility/metadata/{model_path:path}", summary="提取模型元数据")
async def extract_model_metadata(model_path: str):
    """提取模型元数据"""
    try:
        metadata = compatibility_validator.extract_model_metadata(model_path)
        
        return {
            "success": True,
            "data": metadata.to_dict()
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"提取模型元数据失败: {e}")
        raise HTTPException(status_code=500, detail=f"提取模型元数据失败: {str(e)}")

@router.post("/compatibility/check-deployment", summary="检查部署兼容性")
async def check_deployment_compatibility(
    model_path: str,
    deployment_strategy: str = "immediate"
):
    """检查模型是否适合部署"""
    try:
        # 验证模型兼容性
        validation_result = compatibility_validator.validate_compatibility(model_path)
        
        # 检查是否适合部署
        can_deploy = validation_result['compatible']
        deployment_recommendations = []
        
        if not can_deploy:
            deployment_recommendations.append("解决所有兼容性问题后再部署")
        elif validation_result['compatibility_level'] == 'warning':
            deployment_recommendations.append("建议在测试环境中验证后再部署到生产环境")
            if deployment_strategy in ['blue_green', 'canary']:
                deployment_recommendations.append(f"推荐使用 {deployment_strategy} 策略进行安全部署")
        else:
            deployment_recommendations.append("模型兼容性良好，可以安全部署")
        
        return {
            "success": True,
            "data": {
                "can_deploy": can_deploy,
                "compatibility_summary": validation_result['summary'],
                "compatibility_level": validation_result['compatibility_level'],
                "deployment_recommendations": deployment_recommendations,
                "validation_details": validation_result['validation_results']
            }
        }
    except Exception as e:
        logger.error(f"部署兼容性检查失败: {e}")
        raise HTTPException(status_code=500, detail=f"部署兼容性检查失败: {str(e)}")

@router.get("/compatibility/system-info", summary="获取系统信息")
async def get_system_info():
    """获取当前系统信息"""
    try:
        import platform
        import pkg_resources
        
        # 获取已安装的包
        installed_packages = {}
        for dist in pkg_resources.working_set:
            installed_packages[dist.project_name.lower()] = dist.version
        
        system_info = {
            "platform": platform.platform(),
            "system": platform.system(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "installed_packages": installed_packages
        }
        
        return {
            "success": True,
            "data": system_info
        }
    except Exception as e:
        logger.error(f"获取系统信息失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取系统信息失败: {str(e)}")
# 健康检查和性能测试API

@router.post("/health/check", summary="运行健康检查")
async def run_health_checks(
    model_path: Optional[str] = None,
    check_names: Optional[List[str]] = None
):
    """运行健康检查"""
    try:
        kwargs = {}
        if model_path:
            kwargs['model_path'] = model_path
        
        if check_names:
            # 运行指定的检查
            results = {}
            for check_name in check_names:
                result = await health_monitor.health_checker.run_check(check_name, **kwargs)
                results[check_name] = result.to_dict()
            
            overall_status = health_monitor.health_checker.get_overall_health(
                {name: result for name, result in results.items()}
            )
            
            return {
                "success": True,
                "data": {
                    'overall_status': overall_status.value,
                    'checks': results,
                    'timestamp': datetime.now().isoformat()
                }
            }
        else:
            # 运行所有检查
            result = await health_monitor.run_health_checks(**kwargs)
            return {
                "success": True,
                "data": result
            }
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        raise HTTPException(status_code=500, detail=f"健康检查失败: {str(e)}")

@router.get("/health/checks", summary="获取可用的健康检查")
async def get_available_health_checks():
    """获取所有可用的健康检查"""
    try:
        checks = []
        for name, info in health_monitor.health_checker.checks.items():
            checks.append({
                'name': name,
                'description': info.get('description', ''),
            })
        
        return {
            "success": True,
            "data": {
                "checks": checks,
                "count": len(checks)
            }
        }
    except Exception as e:
        logger.error(f"获取健康检查列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取健康检查列表失败: {str(e)}")

@router.get("/health/history", summary="获取健康检查历史")
async def get_health_check_history(
    check_name: Optional[str] = None,
    limit: int = Query(100, description="返回记录数量限制")
):
    """获取健康检查历史"""
    try:
        history = health_monitor.get_health_history(check_name, limit)
        
        return {
            "success": True,
            "data": {
                "history": history,
                "count": len(history),
                "check_name": check_name
            }
        }
    except Exception as e:
        logger.error(f"获取健康检查历史失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取健康检查历史失败: {str(e)}")

@router.post("/performance/test", summary="运行性能测试")
async def run_performance_test(
    test_name: str,
    duration_seconds: int = 60,
    concurrent_users: int = 1,
    model_path: Optional[str] = None
):
    """运行性能测试"""
    try:
        kwargs = {
            'duration_seconds': duration_seconds,
            'concurrent_users': concurrent_users
        }
        if model_path:
            kwargs['model_path'] = model_path
        
        result = await health_monitor.run_performance_test(test_name, **kwargs)
        
        return {
            "success": True,
            "data": result
        }
    except Exception as e:
        logger.error(f"性能测试失败: {e}")
        raise HTTPException(status_code=500, detail=f"性能测试失败: {str(e)}")

@router.post("/performance/load-test", summary="运行负载测试")
async def run_load_test(
    test_name: str,
    max_users: int = 100,
    ramp_up_seconds: int = 60,
    model_path: Optional[str] = None
):
    """运行负载测试"""
    try:
        kwargs = {
            'max_users': max_users,
            'ramp_up_seconds': ramp_up_seconds
        }
        if model_path:
            kwargs['model_path'] = model_path
        
        result = await health_monitor.run_load_test(test_name, **kwargs)
        
        return {
            "success": True,
            "data": result
        }
    except Exception as e:
        logger.error(f"负载测试失败: {e}")
        raise HTTPException(status_code=500, detail=f"负载测试失败: {str(e)}")

@router.get("/performance/tests", summary="获取可用的性能测试")
async def get_available_performance_tests():
    """获取所有可用的性能测试"""
    try:
        tests = []
        for name, info in health_monitor.performance_tester.test_functions.items():
            tests.append({
                'name': name,
                'description': info.get('description', ''),
                'has_baseline': name in health_monitor.performance_tester.baseline_metrics
            })
        
        return {
            "success": True,
            "data": {
                "tests": tests,
                "count": len(tests)
            }
        }
    except Exception as e:
        logger.error(f"获取性能测试列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取性能测试列表失败: {str(e)}")

@router.post("/performance/baseline", summary="设置性能基准")
async def set_performance_baseline(
    test_name: str,
    response_time_ms: float,
    throughput_rps: float,
    error_rate: float,
    cpu_usage: float = 0.0,
    memory_usage_mb: float = 0.0,
    success_count: int = 100,
    error_count: int = 0,
    total_requests: int = 100
):
    """设置性能基准指标"""
    try:
        metrics = {
            'response_time_ms': response_time_ms,
            'throughput_rps': throughput_rps,
            'error_rate': error_rate,
            'cpu_usage': cpu_usage,
            'memory_usage_mb': memory_usage_mb,
            'success_count': success_count,
            'error_count': error_count,
            'total_requests': total_requests
        }
        
        health_monitor.set_performance_baseline(test_name, metrics)
        
        return {
            "success": True,
            "data": {
                "message": f"已设置 {test_name} 的性能基准",
                "baseline_metrics": metrics
            }
        }
    except Exception as e:
        logger.error(f"设置性能基准失败: {e}")
        raise HTTPException(status_code=500, detail=f"设置性能基准失败: {str(e)}")

@router.get("/performance/history", summary="获取性能测试历史")
async def get_performance_test_history(
    test_name: Optional[str] = None,
    limit: int = Query(100, description="返回记录数量限制")
):
    """获取性能测试历史"""
    try:
        history = health_monitor.get_performance_history(test_name, limit)
        
        return {
            "success": True,
            "data": {
                "history": history,
                "count": len(history),
                "test_name": test_name
            }
        }
    except Exception as e:
        logger.error(f"获取性能测试历史失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取性能测试历史失败: {str(e)}")

@router.post("/deployment/validate", summary="部署前验证")
async def validate_deployment_readiness(
    model_path: str,
    run_health_checks: bool = True,
    run_performance_test: bool = True,
    performance_test_duration: int = 30
):
    """部署前的综合验证"""
    try:
        validation_results = {
            'model_path': model_path,
            'timestamp': datetime.now().isoformat(),
            'overall_ready': True,
            'issues': [],
            'recommendations': []
        }
        
        # 1. 兼容性验证
        compatibility_result = compatibility_validator.validate_compatibility(model_path)
        validation_results['compatibility'] = compatibility_result
        
        if not compatibility_result['compatible']:
            validation_results['overall_ready'] = False
            validation_results['issues'].append("模型兼容性验证失败")
            validation_results['recommendations'].extend(compatibility_result['recommendations'])
        
        # 2. 健康检查
        if run_health_checks:
            health_result = await health_monitor.run_health_checks(model_path=model_path)
            validation_results['health_checks'] = health_result
            
            if health_result['overall_status'] != 'healthy':
                validation_results['overall_ready'] = False
                validation_results['issues'].append(f"健康检查状态: {health_result['overall_status']}")
                validation_results['recommendations'].append("解决健康检查问题后再部署")
        
        # 3. 性能测试
        if run_performance_test:
            try:
                performance_result = await health_monitor.run_performance_test(
                    'model_prediction',
                    duration_seconds=performance_test_duration,
                    model_path=model_path
                )
                validation_results['performance_test'] = performance_result
                
                if not performance_result['passed']:
                    validation_results['overall_ready'] = False
                    validation_results['issues'].append("性能测试未通过")
                    validation_results['recommendations'].append("优化模型性能后再部署")
            except Exception as e:
                validation_results['performance_test'] = {'error': str(e)}
                validation_results['recommendations'].append("性能测试失败，建议检查模型")
        
        # 4. 生成部署建议
        if validation_results['overall_ready']:
            validation_results['recommendations'].append("模型已准备就绪，可以安全部署")
            if compatibility_result.get('compatibility_level') == 'warning':
                validation_results['recommendations'].append("建议使用蓝绿部署或金丝雀发布策略")
        else:
            validation_results['recommendations'].append("请解决所有问题后再尝试部署")
        
        return {
            "success": True,
            "data": validation_results
        }
    except Exception as e:
        logger.error(f"部署验证失败: {e}")
        raise HTTPException(status_code=500, detail=f"部署验证失败: {str(e)}")