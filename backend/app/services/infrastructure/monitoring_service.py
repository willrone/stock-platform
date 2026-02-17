"""
数据监控服务
提供系统监控、性能统计和健康检查功能
"""

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Dict, List, Optional

# 使用TYPE_CHECKING避免循环导入
if TYPE_CHECKING:
    from app.services.data import ParquetManager, SimpleDataService
    from app.services.prediction import TechnicalIndicatorCalculator


@dataclass
class ServiceHealthStatus:
    """服务健康状态"""

    service_name: str
    is_healthy: bool
    response_time_ms: float
    last_check: datetime
    error_message: Optional[str] = None


@dataclass
class PerformanceMetrics:
    """性能指标"""

    service_name: str
    avg_response_time: float
    max_response_time: float
    min_response_time: float
    request_count: int
    error_count: int
    success_rate: float
    timestamp: datetime


@dataclass
class ErrorStatistics:
    """错误统计"""

    error_type: str
    count: int
    last_occurrence: datetime
    sample_message: str


class DataMonitoringService:
    """数据监控服务"""

    def __init__(
        self,
        data_service: "SimpleDataService",
        indicators_service: "TechnicalIndicatorCalculator",
        parquet_manager: "ParquetManager",
    ):
        self.data_service = data_service
        self.indicators_service = indicators_service
        self.parquet_manager = parquet_manager
        from loguru import logger

        self.logger = logger

        # 性能监控数据
        self._response_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self._error_counts: Dict[str, int] = defaultdict(int)
        self._request_counts: Dict[str, int] = defaultdict(int)
        self._error_logs: List[Dict[str, Any]] = []

        # 健康检查缓存
        self._health_cache: Dict[str, ServiceHealthStatus] = {}
        self._cache_ttl = 60  # 缓存60秒

        # 监控任务
        self._monitoring_task: Optional[asyncio.Task] = None
        self._shutdown = False

    async def start_monitoring(self):
        """启动监控任务"""
        if self._monitoring_task is None or self._monitoring_task.done():
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())

    async def stop_monitoring(self):
        """停止监控任务"""
        self._shutdown = True
        if self._monitoring_task and not self._monitoring_task.done():
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass

    async def _monitoring_loop(self):
        """监控循环"""
        while not self._shutdown:
            try:
                # 执行健康检查
                await self._perform_health_checks()

                # 清理旧的错误日志
                self._cleanup_old_logs()

                # 等待下次检查
                await asyncio.sleep(30)  # 每30秒检查一次

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"监控循环出错: {e}")
                await asyncio.sleep(5)

    async def check_service_health(self, service_name: str) -> ServiceHealthStatus:
        """
        检查服务健康状态

        Args:
            service_name: 服务名称

        Returns:
            ServiceHealthStatus: 健康状态
        """
        # 检查缓存
        if service_name in self._health_cache:
            cached = self._health_cache[service_name]
            if (datetime.now() - cached.last_check).seconds < self._cache_ttl:
                return cached

        start_time = time.time()
        is_healthy = True
        error_message = None

        try:
            if service_name == "data_service":
                # 检查数据服务
                status = await self.data_service.check_remote_service_status()
                if not status.is_available:
                    is_healthy = False
                    error_message = f"远端服务连接失败: {status.error_message}"
            elif service_name == "indicators_service":
                # 检查技术指标服务（简单测试）
                from app.models.stock_simple import StockData

                test_data = [
                    StockData(
                        stock_code="TEST",
                        date=datetime.now(),
                        open=1.0,
                        high=2.0,
                        low=1.0,
                        close=1.5,
                        volume=1000,
                    ),
                    StockData(
                        stock_code="TEST",
                        date=datetime.now(),
                        open=1.5,
                        high=2.5,
                        low=1.5,
                        close=2.0,
                        volume=1000,
                    ),
                    StockData(
                        stock_code="TEST",
                        date=datetime.now(),
                        open=2.0,
                        high=3.0,
                        low=2.0,
                        close=2.5,
                        volume=1000,
                    ),
                ]
                result = self.indicators_service.calculate_moving_average(test_data, 3)
                if not result or len(result) == 0:
                    is_healthy = False
                    error_message = "技术指标计算失败"
            elif service_name == "parquet_manager":
                # 检查Parquet管理器
                stats = self.parquet_manager.get_storage_stats()
                if stats is None:
                    is_healthy = False
                    error_message = "存储统计获取失败"
            else:
                raise ValueError(f"未知服务: {service_name}")

        except Exception as e:
            is_healthy = False
            error_message = str(e)
            self.logger.warning(f"服务健康检查失败 {service_name}: {e}")

        response_time = (time.time() - start_time) * 1000  # 转换为毫秒

        # 记录性能数据
        self.record_request(service_name, response_time, is_healthy)

        status = ServiceHealthStatus(
            service_name=service_name,
            is_healthy=is_healthy,
            response_time_ms=response_time,
            last_check=datetime.now(),
            error_message=error_message,
        )

        # 更新缓存
        self._health_cache[service_name] = status

        return status

    async def _perform_health_checks(self):
        """执行所有服务的健康检查"""
        services = ["data_service", "indicators_service", "parquet_manager"]
        if self.sync_engine:
            services.append("sync_engine")

        for service_name in services:
            try:
                await self.check_service_health(service_name)
            except Exception as e:
                self.logger.error(f"健康检查失败 {service_name}: {e}")

    def record_request(
        self, service_name: str, response_time_ms: float, success: bool = True
    ):
        """
        记录请求性能数据

        Args:
            service_name: 服务名称
            response_time_ms: 响应时间（毫秒）
            success: 是否成功
        """
        self._response_times[service_name].append(response_time_ms)
        self._request_counts[service_name] += 1

        if not success:
            self._error_counts[service_name] += 1

    def record_error(self, service_name: str, error_type: str, error_message: str):
        """
        记录错误信息

        Args:
            service_name: 服务名称
            error_type: 错误类型
            error_message: 错误消息
        """
        error_log = {
            "service_name": service_name,
            "error_type": error_type,
            "error_message": error_message,
            "timestamp": datetime.now(),
        }

        self._error_logs.append(error_log)
        self._error_counts[service_name] += 1

        # 限制错误日志数量
        if len(self._error_logs) > 1000:
            self._error_logs = self._error_logs[-500:]

    def get_performance_metrics(
        self, service_name: str
    ) -> Optional[PerformanceMetrics]:
        """
        获取服务性能指标

        Args:
            service_name: 服务名称

        Returns:
            Optional[PerformanceMetrics]: 性能指标
        """
        if service_name not in self._response_times:
            return None

        response_times = list(self._response_times[service_name])
        if not response_times:
            return None

        request_count = self._request_counts[service_name]
        error_count = self._error_counts[service_name]
        success_rate = (
            (request_count - error_count) / request_count if request_count > 0 else 0.0
        )

        return PerformanceMetrics(
            service_name=service_name,
            avg_response_time=sum(response_times) / len(response_times),
            max_response_time=max(response_times),
            min_response_time=min(response_times),
            request_count=request_count,
            error_count=error_count,
            success_rate=success_rate,
            timestamp=datetime.now(),
        )

    def get_error_statistics(self, hours: int = 24) -> List[ErrorStatistics]:
        """
        获取错误统计

        Args:
            hours: 统计时间范围（小时）

        Returns:
            List[ErrorStatistics]: 错误统计列表
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)

        # 过滤时间范围内的错误
        recent_errors = [
            log for log in self._error_logs if log["timestamp"] >= cutoff_time
        ]

        # 按错误类型分组统计
        error_groups = defaultdict(list)
        for error in recent_errors:
            key = f"{error['service_name']}:{error['error_type']}"
            error_groups[key].append(error)

        statistics = []
        for key, errors in error_groups.items():
            service_name, error_type = key.split(":", 1)
            latest_error = max(errors, key=lambda x: x["timestamp"])

            stat = ErrorStatistics(
                error_type=f"{service_name} - {error_type}",
                count=len(errors),
                last_occurrence=latest_error["timestamp"],
                sample_message=latest_error["error_message"][:200],  # 截断长消息
            )
            statistics.append(stat)

        # 按错误数量排序
        statistics.sort(key=lambda x: x.count, reverse=True)
        return statistics

    def get_system_overview(self) -> Dict[str, Any]:
        """
        获取系统概览

        Returns:
            Dict[str, Any]: 系统概览数据
        """
        overview = {
            "timestamp": datetime.now().isoformat(),
            "services": {},
            "overall_health": True,
            "total_requests": sum(self._request_counts.values()),
            "total_errors": sum(self._error_counts.values()),
            "storage_stats": None,
            "data_quality": None,
        }

        # 收集各服务状态
        for service_name in [
            "data_service",
            "indicators_service",
            "parquet_manager",
            "sync_engine",
        ]:
            if service_name in self._health_cache:
                health = self._health_cache[service_name]
                metrics = self.get_performance_metrics(service_name)

                overview["services"][service_name] = {
                    "healthy": health.is_healthy,
                    "response_time": health.response_time_ms,
                    "last_check": health.last_check.isoformat(),
                    "error_message": health.error_message,
                    "metrics": metrics.to_dict() if metrics else None,
                }

                if not health.is_healthy:
                    overview["overall_health"] = False

        # 获取存储统计
        try:
            storage_stats = self.parquet_manager.get_storage_stats()
            overview["storage_stats"] = storage_stats
        except Exception as e:
            self.logger.warning(f"获取存储统计失败: {e}")

        # 获取数据质量检查结果
        try:
            data_quality = self.check_data_quality()
            overview["data_quality"] = data_quality
        except Exception as e:
            self.logger.warning(f"数据质量检查失败: {e}")

        return overview

    def check_data_quality(self) -> Dict[str, Any]:
        """
        检查数据质量

        Returns:
            Dict[str, Any]: 数据质量检查结果
        """
        quality_report = {
            "overall_score": 0.0,
            "checks": {},
            "issues": [],
            "recommendations": [],
        }

        try:
            # 获取存储统计
            storage_stats = self.parquet_manager.get_storage_stats()

            checks = {}
            issues = []
            recommendations = []

            # 检查1: 数据完整性
            if storage_stats.get("total_files", 0) > 0:
                checks["data_completeness"] = {
                    "score": 1.0,
                    "status": "pass",
                    "message": f"发现 {storage_stats['total_files']} 个数据文件",
                }
            else:
                checks["data_completeness"] = {
                    "score": 0.0,
                    "status": "fail",
                    "message": "未发现任何数据文件",
                }
                issues.append("系统中没有数据文件")
                recommendations.append("执行数据同步以获取股票数据")

            # 检查2: 数据新鲜度
            date_range = storage_stats.get("date_range")
            if date_range:
                end_date = (
                    datetime.fromisoformat(date_range["end"])
                    if isinstance(date_range["end"], str)
                    else date_range["end"]
                )
                days_old = (datetime.now() - end_date).days

                if days_old <= 1:
                    freshness_score = 1.0
                    freshness_status = "excellent"
                elif days_old <= 7:
                    freshness_score = 0.8
                    freshness_status = "good"
                elif days_old <= 30:
                    freshness_score = 0.6
                    freshness_status = "fair"
                else:
                    freshness_score = 0.3
                    freshness_status = "poor"

                checks["data_freshness"] = {
                    "score": freshness_score,
                    "status": freshness_status,
                    "message": f"数据最后更新于 {days_old} 天前",
                    "days_old": days_old,
                }

                if days_old > 7:
                    issues.append(f"数据已过期 {days_old} 天")
                    recommendations.append("执行数据同步以获取最新数据")
            else:
                checks["data_freshness"] = {
                    "score": 0.0,
                    "status": "unknown",
                    "message": "无法确定数据新鲜度",
                }

            # 检查3: 存储效率
            total_size = storage_stats.get("total_size", 0)
            total_records = storage_stats.get("total_records", 0)

            if total_records > 0 and total_size > 0:
                bytes_per_record = total_size / total_records

                if bytes_per_record < 100:  # 每条记录小于100字节，压缩效果好
                    efficiency_score = 1.0
                    efficiency_status = "excellent"
                elif bytes_per_record < 200:
                    efficiency_score = 0.8
                    efficiency_status = "good"
                elif bytes_per_record < 500:
                    efficiency_score = 0.6
                    efficiency_status = "fair"
                else:
                    efficiency_score = 0.4
                    efficiency_status = "poor"

                checks["storage_efficiency"] = {
                    "score": efficiency_score,
                    "status": efficiency_status,
                    "message": f"每条记录占用 {bytes_per_record:.1f} 字节",
                    "bytes_per_record": bytes_per_record,
                }

                if efficiency_score < 0.7:
                    issues.append("存储效率较低")
                    recommendations.append("考虑优化数据压缩或清理冗余数据")
            else:
                checks["storage_efficiency"] = {
                    "score": 0.0,
                    "status": "unknown",
                    "message": "无法计算存储效率",
                }

            # 检查4: 股票覆盖度
            stock_count = storage_stats.get("stock_count", 0)
            if stock_count >= 100:
                coverage_score = 1.0
                coverage_status = "excellent"
            elif stock_count >= 50:
                coverage_score = 0.8
                coverage_status = "good"
            elif stock_count >= 10:
                coverage_score = 0.6
                coverage_status = "fair"
            elif stock_count > 0:
                coverage_score = 0.4
                coverage_status = "poor"
            else:
                coverage_score = 0.0
                coverage_status = "fail"

            checks["stock_coverage"] = {
                "score": coverage_score,
                "status": coverage_status,
                "message": f"覆盖 {stock_count} 只股票",
                "stock_count": stock_count,
            }

            if stock_count < 50:
                issues.append("股票覆盖度较低")
                recommendations.append("增加更多股票的数据同步")

            # 计算总体评分
            scores = [check["score"] for check in checks.values()]
            overall_score = sum(scores) / len(scores) if scores else 0.0

            quality_report.update(
                {
                    "overall_score": round(overall_score, 2),
                    "checks": checks,
                    "issues": issues,
                    "recommendations": recommendations,
                }
            )

        except Exception as e:
            self.logger.error(f"数据质量检查失败: {e}")
            quality_report["checks"]["system_error"] = {
                "score": 0.0,
                "status": "error",
                "message": f"质量检查失败: {str(e)}",
            }

        return quality_report

    def detect_anomalies(self) -> List[Dict[str, Any]]:
        """
        检测异常数据

        Returns:
            List[Dict[str, Any]]: 异常检测结果
        """
        anomalies = []

        try:
            # 检查错误率异常
            for service_name, error_count in self._error_counts.items():
                request_count = self._request_counts.get(service_name, 0)
                if request_count > 10:  # 至少有10个请求才检查
                    error_rate = error_count / request_count
                    if error_rate > 0.1:  # 错误率超过10%
                        anomalies.append(
                            {
                                "type": "high_error_rate",
                                "service": service_name,
                                "severity": "high" if error_rate > 0.3 else "medium",
                                "value": error_rate,
                                "message": f"{service_name} 错误率异常: {error_rate:.1%}",
                                "detected_at": datetime.now().isoformat(),
                            }
                        )

            # 检查响应时间异常
            for service_name, response_times in self._response_times.items():
                if len(response_times) >= 5:  # 至少有5个样本
                    avg_time = sum(response_times) / len(response_times)
                    max_time = max(response_times)

                    if avg_time > 5000:  # 平均响应时间超过5秒
                        anomalies.append(
                            {
                                "type": "slow_response",
                                "service": service_name,
                                "severity": "high" if avg_time > 10000 else "medium",
                                "value": avg_time,
                                "message": f"{service_name} 响应时间异常: {avg_time:.0f}ms",
                                "detected_at": datetime.now().isoformat(),
                            }
                        )

                    if max_time > avg_time * 3:  # 最大响应时间是平均值的3倍以上
                        anomalies.append(
                            {
                                "type": "response_spike",
                                "service": service_name,
                                "severity": "medium",
                                "value": max_time,
                                "message": f"{service_name} 响应时间峰值: {max_time:.0f}ms",
                                "detected_at": datetime.now().isoformat(),
                            }
                        )

            # 检查服务健康状态异常
            for service_name, health_status in self._health_cache.items():
                if not health_status.is_healthy:
                    anomalies.append(
                        {
                            "type": "service_unhealthy",
                            "service": service_name,
                            "severity": "high",
                            "value": health_status.error_message,
                            "message": f"{service_name} 服务不健康: {health_status.error_message}",
                            "detected_at": datetime.now().isoformat(),
                        }
                    )

        except Exception as e:
            self.logger.error(f"异常检测失败: {e}")
            anomalies.append(
                {
                    "type": "detection_error",
                    "service": "monitoring",
                    "severity": "medium",
                    "value": str(e),
                    "message": f"异常检测失败: {str(e)}",
                    "detected_at": datetime.now().isoformat(),
                }
            )

        return anomalies

    def _cleanup_old_logs(self):
        """清理旧的错误日志"""
        cutoff_time = datetime.now() - timedelta(days=7)  # 保留7天
        self._error_logs = [
            log for log in self._error_logs if log["timestamp"] >= cutoff_time
        ]

    async def cleanup(self):
        """清理资源"""
        await self.stop_monitoring()
        self._health_cache.clear()
        self._response_times.clear()
        self._error_counts.clear()
        self._request_counts.clear()
        self._error_logs.clear()


# 扩展PerformanceMetrics类的方法
def _extend_performance_metrics():
    """扩展PerformanceMetrics类"""

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "service_name": self.service_name,
            "avg_response_time": self.avg_response_time,
            "max_response_time": self.max_response_time,
            "min_response_time": self.min_response_time,
            "request_count": self.request_count,
            "error_count": self.error_count,
            "success_rate": self.success_rate,
            "timestamp": self.timestamp.isoformat(),
        }

    PerformanceMetrics.to_dict = to_dict


# 执行扩展
_extend_performance_metrics()
