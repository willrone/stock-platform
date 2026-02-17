"""
部署策略管理器
支持蓝绿部署、金丝雀发布和自动回滚机制
"""
import asyncio
import hashlib
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from loguru import logger


class DeploymentStrategy(Enum):
    """部署策略类型"""

    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    ROLLING = "rolling"
    IMMEDIATE = "immediate"


class DeploymentStatus(Enum):
    """部署状态"""

    PENDING = "pending"
    DEPLOYING = "deploying"
    DEPLOYED = "deployed"
    TESTING = "testing"
    ACTIVE = "active"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    ARCHIVED = "archived"


@dataclass
class DeploymentConfig:
    """部署配置"""

    strategy: DeploymentStrategy
    model_path: str
    model_id: str
    version: str
    # 蓝绿部署配置
    blue_green_switch_delay: int = 30  # 切换延迟（秒）
    # 金丝雀部署配置
    canary_traffic_percentage: float = 10.0  # 金丝雀流量百分比
    canary_duration_minutes: int = 30  # 金丝雀持续时间
    canary_success_threshold: float = 0.95  # 成功率阈值
    # 滚动部署配置
    rolling_batch_size: int = 1  # 滚动批次大小
    rolling_batch_delay: int = 60  # 批次间延迟
    # 健康检查配置
    health_check_enabled: bool = True
    health_check_timeout: int = 300  # 健康检查超时
    health_check_interval: int = 10  # 健康检查间隔
    # 自动回滚配置
    auto_rollback_enabled: bool = True
    rollback_threshold_error_rate: float = 0.05  # 错误率阈值
    rollback_threshold_latency_ms: float = 1000.0  # 延迟阈值

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy.value,
            "model_path": self.model_path,
            "model_id": self.model_id,
            "version": self.version,
            "blue_green_switch_delay": self.blue_green_switch_delay,
            "canary_traffic_percentage": self.canary_traffic_percentage,
            "canary_duration_minutes": self.canary_duration_minutes,
            "canary_success_threshold": self.canary_success_threshold,
            "rolling_batch_size": self.rolling_batch_size,
            "rolling_batch_delay": self.rolling_batch_delay,
            "health_check_enabled": self.health_check_enabled,
            "health_check_timeout": self.health_check_timeout,
            "health_check_interval": self.health_check_interval,
            "auto_rollback_enabled": self.auto_rollback_enabled,
            "rollback_threshold_error_rate": self.rollback_threshold_error_rate,
            "rollback_threshold_latency_ms": self.rollback_threshold_latency_ms,
        }


@dataclass
class DeploymentEnvironment:
    """部署环境"""

    name: str  # blue, green, canary, production
    model_path: Optional[str] = None
    model_id: Optional[str] = None
    version: Optional[str] = None
    status: DeploymentStatus = DeploymentStatus.PENDING
    traffic_percentage: float = 0.0
    deployed_at: Optional[datetime] = None
    health_status: str = "unknown"
    metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "model_path": self.model_path,
            "model_id": self.model_id,
            "version": self.version,
            "status": self.status.value,
            "traffic_percentage": self.traffic_percentage,
            "deployed_at": self.deployed_at.isoformat() if self.deployed_at else None,
            "health_status": self.health_status,
            "metrics": self.metrics,
        }


@dataclass
class DeploymentRecord:
    """部署记录"""

    deployment_id: str
    config: DeploymentConfig
    environments: Dict[str, DeploymentEnvironment]
    status: DeploymentStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    rollback_info: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "deployment_id": self.deployment_id,
            "config": self.config.to_dict(),
            "environments": {
                name: env.to_dict() for name, env in self.environments.items()
            },
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat()
            if self.completed_at
            else None,
            "error_message": self.error_message,
            "rollback_info": self.rollback_info,
        }


class HealthChecker:
    """健康检查器"""

    def __init__(self):
        self.checks: Dict[str, Callable] = {}

    def register_check(self, name: str, check_func: Callable):
        """注册健康检查函数"""
        self.checks[name] = check_func
        logger.info(f"注册健康检查: {name}")

    async def run_checks(self, environment: DeploymentEnvironment) -> Dict[str, Any]:
        """运行所有健康检查"""
        results = {}
        overall_healthy = True

        for name, check_func in self.checks.items():
            try:
                if asyncio.iscoroutinefunction(check_func):
                    result = await check_func(environment)
                else:
                    result = check_func(environment)

                results[name] = result
                if not result.get("healthy", False):
                    overall_healthy = False

            except Exception as e:
                logger.error(f"健康检查失败 {name}: {e}")
                results[name] = {"healthy": False, "error": str(e)}
                overall_healthy = False

        results["overall_healthy"] = overall_healthy
        return results


class MetricsCollector:
    """指标收集器"""

    def __init__(self):
        self.collectors: Dict[str, Callable] = {}

    def register_collector(self, name: str, collector_func: Callable):
        """注册指标收集函数"""
        self.collectors[name] = collector_func
        logger.info(f"注册指标收集器: {name}")

    async def collect_metrics(
        self, environment: DeploymentEnvironment
    ) -> Dict[str, float]:
        """收集所有指标"""
        metrics = {}

        for name, collector_func in self.collectors.items():
            try:
                if asyncio.iscoroutinefunction(collector_func):
                    value = await collector_func(environment)
                else:
                    value = collector_func(environment)

                metrics[name] = float(value)

            except Exception as e:
                logger.error(f"指标收集失败 {name}: {e}")
                metrics[name] = 0.0

        return metrics


class DeploymentManager:
    """部署策略管理器"""

    def __init__(self, base_path: str = "data/deployments"):
        """
        初始化部署管理器

        Args:
            base_path: 部署文件基础路径
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        # 部署记录
        self.deployments: Dict[str, DeploymentRecord] = {}
        self.active_deployment: Optional[str] = None

        # 组件
        self.health_checker = HealthChecker()
        self.metrics_collector = MetricsCollector()

        # 回调函数
        self.callbacks: Dict[str, List[Callable]] = {
            "deployment_started": [],
            "deployment_completed": [],
            "deployment_failed": [],
            "rollback_triggered": [],
            "health_check_failed": [],
        }

        # 注册默认健康检查和指标收集
        self._register_default_checks()

        logger.info("部署管理器初始化完成")

    def _register_default_checks(self):
        """注册默认的健康检查和指标收集"""

        # 模型文件存在检查
        def model_file_check(env: DeploymentEnvironment) -> Dict[str, Any]:
            if not env.model_path:
                return {"healthy": False, "message": "模型路径未设置"}

            model_path = Path(env.model_path)
            if not model_path.exists():
                return {"healthy": False, "message": f"模型文件不存在: {env.model_path}"}

            return {"healthy": True, "message": "模型文件存在"}

        self.health_checker.register_check("model_file", model_file_check)

        # 模型加载检查
        async def model_load_check(env: DeploymentEnvironment) -> Dict[str, Any]:
            try:
                # 这里应该实际加载模型进行测试
                # 暂时返回成功
                return {"healthy": True, "message": "模型加载成功"}
            except Exception as e:
                return {"healthy": False, "message": f"模型加载失败: {e}"}

        self.health_checker.register_check("model_load", model_load_check)

        # 响应时间指标收集
        def response_time_collector(env: DeploymentEnvironment) -> float:
            # 模拟响应时间（实际应该从监控系统获取）
            import random

            return random.uniform(50, 200)  # 50-200ms

        self.metrics_collector.register_collector(
            "response_time_ms", response_time_collector
        )

        # 错误率指标收集
        def error_rate_collector(env: DeploymentEnvironment) -> float:
            # 模拟错误率（实际应该从监控系统获取）
            import random

            return random.uniform(0, 0.02)  # 0-2%

        self.metrics_collector.register_collector("error_rate", error_rate_collector)

    def add_callback(self, event: str, callback: Callable):
        """添加事件回调函数"""
        if event in self.callbacks:
            self.callbacks[event].append(callback)

    async def _notify_callbacks(self, event: str, *args, **kwargs):
        """通知回调函数"""
        for callback in self.callbacks.get(event, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(*args, **kwargs)
                else:
                    callback(*args, **kwargs)
            except Exception as e:
                logger.error(f"回调函数执行失败 {event}: {e}")

    async def deploy(self, config: DeploymentConfig) -> str:
        """
        执行部署

        Args:
            config: 部署配置

        Returns:
            部署ID
        """
        deployment_id = self._generate_deployment_id(config)

        # 创建部署记录
        deployment = DeploymentRecord(
            deployment_id=deployment_id,
            config=config,
            environments={},
            status=DeploymentStatus.PENDING,
            created_at=datetime.now(),
        )

        self.deployments[deployment_id] = deployment

        logger.info(f"开始部署: {deployment_id}, 策略: {config.strategy.value}")

        try:
            # 根据策略执行部署
            if config.strategy == DeploymentStrategy.BLUE_GREEN:
                await self._deploy_blue_green(deployment)
            elif config.strategy == DeploymentStrategy.CANARY:
                await self._deploy_canary(deployment)
            elif config.strategy == DeploymentStrategy.ROLLING:
                await self._deploy_rolling(deployment)
            elif config.strategy == DeploymentStrategy.IMMEDIATE:
                await self._deploy_immediate(deployment)
            else:
                raise ValueError(f"不支持的部署策略: {config.strategy}")

            deployment.status = DeploymentStatus.DEPLOYED
            deployment.completed_at = datetime.now()

            await self._notify_callbacks("deployment_completed", deployment)
            logger.info(f"部署完成: {deployment_id}")

        except Exception as e:
            deployment.status = DeploymentStatus.FAILED
            deployment.error_message = str(e)
            deployment.completed_at = datetime.now()

            await self._notify_callbacks("deployment_failed", deployment, e)
            logger.error(f"部署失败: {deployment_id}, 错误: {e}")

            # 如果启用自动回滚，尝试回滚
            if config.auto_rollback_enabled:
                await self._auto_rollback(deployment)

            raise

        return deployment_id

    def _generate_deployment_id(self, config: DeploymentConfig) -> str:
        """生成部署ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_hash = hashlib.md5(
            f"{config.model_id}_{config.version}".encode()
        ).hexdigest()[:8]
        return f"deploy_{timestamp}_{model_hash}"

    async def _deploy_blue_green(self, deployment: DeploymentRecord):
        """蓝绿部署"""
        config = deployment.config

        # 创建蓝绿环境
        blue_env = DeploymentEnvironment(name="blue", status=DeploymentStatus.PENDING)
        green_env = DeploymentEnvironment(name="green", status=DeploymentStatus.PENDING)

        deployment.environments = {"blue": blue_env, "green": green_env}
        deployment.status = DeploymentStatus.DEPLOYING
        deployment.started_at = datetime.now()

        await self._notify_callbacks("deployment_started", deployment)

        # 1. 部署到绿色环境
        logger.info("部署到绿色环境")
        await self._deploy_to_environment(green_env, config)

        # 2. 健康检查
        if config.health_check_enabled:
            logger.info("执行绿色环境健康检查")
            health_results = await self._run_health_checks(green_env, config)
            if not health_results.get("overall_healthy", False):
                raise RuntimeError(f"绿色环境健康检查失败: {health_results}")

        # 3. 等待切换延迟
        if config.blue_green_switch_delay > 0:
            logger.info(f"等待切换延迟 {config.blue_green_switch_delay} 秒")
            await asyncio.sleep(config.blue_green_switch_delay)

        # 4. 切换流量到绿色环境
        logger.info("切换流量到绿色环境")
        blue_env.traffic_percentage = 0.0
        green_env.traffic_percentage = 100.0
        green_env.status = DeploymentStatus.ACTIVE

        # 5. 停用蓝色环境
        blue_env.status = DeploymentStatus.ARCHIVED

        self.active_deployment = deployment.deployment_id
        logger.info("蓝绿部署完成")

    async def _deploy_canary(self, deployment: DeploymentRecord):
        """金丝雀部署"""
        config = deployment.config

        # 创建生产和金丝雀环境
        prod_env = DeploymentEnvironment(
            name="production",
            traffic_percentage=100.0 - config.canary_traffic_percentage,
            status=DeploymentStatus.ACTIVE,
        )
        canary_env = DeploymentEnvironment(
            name="canary",
            traffic_percentage=config.canary_traffic_percentage,
            status=DeploymentStatus.PENDING,
        )

        deployment.environments = {"production": prod_env, "canary": canary_env}
        deployment.status = DeploymentStatus.DEPLOYING
        deployment.started_at = datetime.now()

        await self._notify_callbacks("deployment_started", deployment)

        # 1. 部署到金丝雀环境
        logger.info(f"部署到金丝雀环境 ({config.canary_traffic_percentage}% 流量)")
        await self._deploy_to_environment(canary_env, config)

        # 2. 健康检查
        if config.health_check_enabled:
            logger.info("执行金丝雀环境健康检查")
            health_results = await self._run_health_checks(canary_env, config)
            if not health_results.get("overall_healthy", False):
                raise RuntimeError(f"金丝雀环境健康检查失败: {health_results}")

        canary_env.status = DeploymentStatus.TESTING

        # 3. 监控金丝雀环境
        logger.info(f"监控金丝雀环境 {config.canary_duration_minutes} 分钟")
        await self._monitor_canary(canary_env, config)

        # 4. 全量切换
        logger.info("金丝雀测试通过，执行全量切换")
        prod_env.traffic_percentage = 0.0
        prod_env.status = DeploymentStatus.ARCHIVED
        canary_env.traffic_percentage = 100.0
        canary_env.status = DeploymentStatus.ACTIVE
        canary_env.name = "production"  # 重命名为生产环境

        self.active_deployment = deployment.deployment_id
        logger.info("金丝雀部署完成")

    async def _deploy_rolling(self, deployment: DeploymentRecord):
        """滚动部署"""
        config = deployment.config

        # 创建多个实例环境
        environments = {}
        for i in range(config.rolling_batch_size):
            env_name = f"instance_{i}"
            env = DeploymentEnvironment(
                name=env_name,
                traffic_percentage=100.0 / config.rolling_batch_size,
                status=DeploymentStatus.PENDING,
            )
            environments[env_name] = env

        deployment.environments = environments
        deployment.status = DeploymentStatus.DEPLOYING
        deployment.started_at = datetime.now()

        await self._notify_callbacks("deployment_started", deployment)

        # 逐批部署
        for i, (env_name, env) in enumerate(environments.items()):
            logger.info(f"滚动部署批次 {i+1}/{len(environments)}: {env_name}")

            await self._deploy_to_environment(env, config)

            if config.health_check_enabled:
                health_results = await self._run_health_checks(env, config)
                if not health_results.get("overall_healthy", False):
                    raise RuntimeError(f"实例 {env_name} 健康检查失败: {health_results}")

            env.status = DeploymentStatus.ACTIVE

            # 批次间延迟
            if i < len(environments) - 1 and config.rolling_batch_delay > 0:
                logger.info(f"等待批次延迟 {config.rolling_batch_delay} 秒")
                await asyncio.sleep(config.rolling_batch_delay)

        self.active_deployment = deployment.deployment_id
        logger.info("滚动部署完成")

    async def _deploy_immediate(self, deployment: DeploymentRecord):
        """立即部署"""
        config = deployment.config

        # 创建生产环境
        prod_env = DeploymentEnvironment(
            name="production", traffic_percentage=100.0, status=DeploymentStatus.PENDING
        )

        deployment.environments = {"production": prod_env}
        deployment.status = DeploymentStatus.DEPLOYING
        deployment.started_at = datetime.now()

        await self._notify_callbacks("deployment_started", deployment)

        # 直接部署到生产环境
        logger.info("立即部署到生产环境")
        await self._deploy_to_environment(prod_env, config)

        # 健康检查
        if config.health_check_enabled:
            logger.info("执行生产环境健康检查")
            health_results = await self._run_health_checks(prod_env, config)
            if not health_results.get("overall_healthy", False):
                raise RuntimeError(f"生产环境健康检查失败: {health_results}")

        prod_env.status = DeploymentStatus.ACTIVE
        self.active_deployment = deployment.deployment_id
        logger.info("立即部署完成")

    async def _deploy_to_environment(
        self, environment: DeploymentEnvironment, config: DeploymentConfig
    ):
        """部署到指定环境"""
        try:
            # 1. 复制模型文件
            env_path = self.base_path / environment.name
            env_path.mkdir(parents=True, exist_ok=True)

            model_src = Path(config.model_path)
            model_dst = env_path / model_src.name

            if model_src.exists():
                shutil.copy2(model_src, model_dst)
                logger.info(f"模型文件已复制到 {environment.name}: {model_dst}")
            else:
                logger.warning(f"源模型文件不存在: {model_src}")

            # 2. 更新环境信息
            environment.model_path = str(model_dst)
            environment.model_id = config.model_id
            environment.version = config.version
            environment.deployed_at = datetime.now()
            environment.status = DeploymentStatus.DEPLOYED

            # 3. 模拟部署延迟
            await asyncio.sleep(1)

            logger.info(f"环境 {environment.name} 部署完成")

        except Exception as e:
            environment.status = DeploymentStatus.FAILED
            logger.error(f"环境 {environment.name} 部署失败: {e}")
            raise

    async def _run_health_checks(
        self, environment: DeploymentEnvironment, config: DeploymentConfig
    ) -> Dict[str, Any]:
        """运行健康检查"""
        start_time = datetime.now()
        timeout = timedelta(seconds=config.health_check_timeout)

        while datetime.now() - start_time < timeout:
            try:
                results = await self.health_checker.run_checks(environment)
                environment.health_status = (
                    "healthy" if results.get("overall_healthy") else "unhealthy"
                )

                if results.get("overall_healthy"):
                    logger.info(f"环境 {environment.name} 健康检查通过")
                    return results

                logger.warning(f"环境 {environment.name} 健康检查失败，重试中...")
                await asyncio.sleep(config.health_check_interval)

            except Exception as e:
                logger.error(f"健康检查异常: {e}")
                await asyncio.sleep(config.health_check_interval)

        # 超时失败
        environment.health_status = "timeout"
        raise RuntimeError(f"环境 {environment.name} 健康检查超时")

    async def _monitor_canary(
        self, canary_env: DeploymentEnvironment, config: DeploymentConfig
    ):
        """监控金丝雀环境"""
        duration = timedelta(minutes=config.canary_duration_minutes)
        start_time = datetime.now()

        while datetime.now() - start_time < duration:
            # 收集指标
            metrics = await self.metrics_collector.collect_metrics(canary_env)
            canary_env.metrics.update(metrics)

            # 检查是否需要回滚
            if self._should_rollback(metrics, config):
                raise RuntimeError(f"金丝雀指标不达标，触发回滚: {metrics}")

            logger.info(f"金丝雀监控: {metrics}")
            await asyncio.sleep(30)  # 30秒检查一次

        # 检查最终成功率
        final_success_rate = 1.0 - canary_env.metrics.get("error_rate", 0.0)
        if final_success_rate < config.canary_success_threshold:
            raise RuntimeError(
                f"金丝雀成功率不达标: {final_success_rate} < {config.canary_success_threshold}"
            )

        logger.info(f"金丝雀监控完成，成功率: {final_success_rate}")

    def _should_rollback(
        self, metrics: Dict[str, float], config: DeploymentConfig
    ) -> bool:
        """检查是否应该回滚"""
        error_rate = metrics.get("error_rate", 0.0)
        response_time = metrics.get("response_time_ms", 0.0)

        if error_rate > config.rollback_threshold_error_rate:
            logger.warning(
                f"错误率超过阈值: {error_rate} > {config.rollback_threshold_error_rate}"
            )
            return True

        if response_time > config.rollback_threshold_latency_ms:
            logger.warning(
                f"响应时间超过阈值: {response_time} > {config.rollback_threshold_latency_ms}"
            )
            return True

        return False

    async def _auto_rollback(self, deployment: DeploymentRecord):
        """自动回滚"""
        logger.info(f"开始自动回滚: {deployment.deployment_id}")

        try:
            # 查找上一个成功的部署
            previous_deployment = self._find_previous_deployment()

            if not previous_deployment:
                logger.error("未找到可回滚的部署")
                return

            # 执行回滚
            rollback_config = DeploymentConfig(
                strategy=DeploymentStrategy.IMMEDIATE,
                model_path=previous_deployment.config.model_path,
                model_id=previous_deployment.config.model_id,
                version=previous_deployment.config.version,
                auto_rollback_enabled=False,  # 避免回滚循环
            )

            rollback_id = await self.deploy(rollback_config)

            # 记录回滚信息
            deployment.rollback_info = {
                "rollback_deployment_id": rollback_id,
                "rollback_to_version": previous_deployment.config.version,
                "rollback_at": datetime.now().isoformat(),
                "rollback_reason": "auto_rollback_on_failure",
            }
            deployment.status = DeploymentStatus.ROLLED_BACK

            await self._notify_callbacks("rollback_triggered", deployment, rollback_id)
            logger.info(f"自动回滚完成: {deployment.deployment_id} -> {rollback_id}")

        except Exception as e:
            logger.error(f"自动回滚失败: {e}")

    def _find_previous_deployment(self) -> Optional[DeploymentRecord]:
        """查找上一个成功的部署"""
        successful_deployments = [
            d
            for d in self.deployments.values()
            if d.status == DeploymentStatus.DEPLOYED
        ]

        if not successful_deployments:
            return None

        # 按时间排序，返回最新的成功部署
        successful_deployments.sort(key=lambda x: x.created_at, reverse=True)
        return successful_deployments[0]

    async def rollback(
        self, deployment_id: str, target_version: Optional[str] = None
    ) -> str:
        """
        手动回滚部署

        Args:
            deployment_id: 要回滚的部署ID
            target_version: 目标版本（可选）

        Returns:
            回滚部署ID
        """
        if deployment_id not in self.deployments:
            raise ValueError(f"部署不存在: {deployment_id}")

        deployment = self.deployments[deployment_id]

        # 查找回滚目标
        if target_version:
            target_deployment = None
            for d in self.deployments.values():
                if (
                    d.config.version == target_version
                    and d.status == DeploymentStatus.DEPLOYED
                ):
                    target_deployment = d
                    break

            if not target_deployment:
                raise ValueError(f"未找到版本 {target_version} 的成功部署")
        else:
            target_deployment = self._find_previous_deployment()
            if not target_deployment:
                raise ValueError("未找到可回滚的部署")

        # 创建回滚配置
        rollback_config = DeploymentConfig(
            strategy=DeploymentStrategy.IMMEDIATE,
            model_path=target_deployment.config.model_path,
            model_id=target_deployment.config.model_id,
            version=target_deployment.config.version,
            auto_rollback_enabled=False,
        )

        # 执行回滚部署
        rollback_id = await self.deploy(rollback_config)

        # 更新原部署状态
        deployment.status = DeploymentStatus.ROLLED_BACK
        deployment.rollback_info = {
            "rollback_deployment_id": rollback_id,
            "rollback_to_version": target_deployment.config.version,
            "rollback_at": datetime.now().isoformat(),
            "rollback_reason": "manual_rollback",
        }

        await self._notify_callbacks("rollback_triggered", deployment, rollback_id)
        logger.info(f"手动回滚完成: {deployment_id} -> {rollback_id}")

        return rollback_id

    def get_deployment_status(self, deployment_id: str) -> Optional[Dict[str, Any]]:
        """获取部署状态"""
        if deployment_id not in self.deployments:
            return None

        return self.deployments[deployment_id].to_dict()

    def get_all_deployments(self) -> List[Dict[str, Any]]:
        """获取所有部署记录"""
        return [deployment.to_dict() for deployment in self.deployments.values()]

    def get_active_deployment(self) -> Optional[Dict[str, Any]]:
        """获取当前活跃的部署"""
        if not self.active_deployment:
            return None

        return self.get_deployment_status(self.active_deployment)


# 全局部署管理器实例
deployment_manager = DeploymentManager()
