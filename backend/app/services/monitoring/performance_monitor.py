"""
模型性能监控器
收集预测请求和响应数据，监控准确率和延迟指标
"""
import asyncio
import json
import statistics
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from loguru import logger


class MetricType(Enum):
    """指标类型"""

    LATENCY = "latency"
    ACCURACY = "accuracy"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    PREDICTION_DISTRIBUTION = "prediction_distribution"
    FEATURE_DRIFT = "feature_drift"


class AlertLevel(Enum):
    """告警级别"""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class PredictionRecord:
    """预测记录"""

    request_id: str
    timestamp: datetime
    model_id: str
    model_version: str
    input_features: Dict[str, Any]
    prediction: Any
    confidence: Optional[float]
    latency_ms: float
    success: bool
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "timestamp": self.timestamp.isoformat(),
            "model_id": self.model_id,
            "model_version": self.model_version,
            "input_features": self.input_features,
            "prediction": self.prediction,
            "confidence": self.confidence,
            "latency_ms": self.latency_ms,
            "success": self.success,
            "error_message": self.error_message,
        }


@dataclass
class PerformanceMetrics:
    """性能指标"""

    timestamp: datetime
    model_id: str
    model_version: str
    # 延迟指标
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    # 吞吐量指标
    requests_per_second: float
    total_requests: int
    # 准确率指标
    success_rate: float
    error_rate: float
    # 预测分布
    prediction_stats: Dict[str, float]
    # 置信度统计
    avg_confidence: Optional[float]
    low_confidence_rate: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "model_id": self.model_id,
            "model_version": self.model_version,
            "avg_latency_ms": self.avg_latency_ms,
            "p50_latency_ms": self.p50_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "p99_latency_ms": self.p99_latency_ms,
            "requests_per_second": self.requests_per_second,
            "total_requests": self.total_requests,
            "success_rate": self.success_rate,
            "error_rate": self.error_rate,
            "prediction_stats": self.prediction_stats,
            "avg_confidence": self.avg_confidence,
            "low_confidence_rate": self.low_confidence_rate,
        }


@dataclass
class AlertRule:
    """告警规则"""

    name: str
    metric_type: MetricType
    condition: str  # "gt", "lt", "eq", "ne"
    threshold: float
    level: AlertLevel
    enabled: bool = True
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "metric_type": self.metric_type.value,
            "condition": self.condition,
            "threshold": self.threshold,
            "level": self.level.value,
            "enabled": self.enabled,
            "description": self.description,
        }


@dataclass
class Alert:
    """告警"""

    alert_id: str
    rule_name: str
    level: AlertLevel
    message: str
    metric_value: float
    threshold: float
    timestamp: datetime
    model_id: str
    model_version: str
    resolved: bool = False
    resolved_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "rule_name": self.rule_name,
            "level": self.level.value,
            "message": self.message,
            "metric_value": self.metric_value,
            "threshold": self.threshold,
            "timestamp": self.timestamp.isoformat(),
            "model_id": self.model_id,
            "model_version": self.model_version,
            "resolved": self.resolved,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
        }


class MetricsCalculator:
    """指标计算器"""

    def __init__(self, window_size: int = 1000):
        """
        初始化指标计算器

        Args:
            window_size: 滑动窗口大小
        """
        self.window_size = window_size
        self.records: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.lock = threading.Lock()

    def add_record(self, record: PredictionRecord):
        """添加预测记录"""
        with self.lock:
            key = f"{record.model_id}_{record.model_version}"
            self.records[key].append(record)

    def calculate_metrics(
        self, model_id: str, model_version: str
    ) -> Optional[PerformanceMetrics]:
        """计算性能指标"""
        with self.lock:
            key = f"{model_id}_{model_version}"
            records = list(self.records[key])

        if not records:
            return None

        # 过滤最近的记录（例如最近5分钟）
        cutoff_time = datetime.now() - timedelta(minutes=5)
        recent_records = [r for r in records if r.timestamp >= cutoff_time]

        if not recent_records:
            return None

        # 计算延迟指标
        latencies = [r.latency_ms for r in recent_records]
        avg_latency = statistics.mean(latencies)
        p50_latency = np.percentile(latencies, 50)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)

        # 计算吞吐量指标
        time_span = (
            recent_records[-1].timestamp - recent_records[0].timestamp
        ).total_seconds()
        requests_per_second = len(recent_records) / max(time_span, 1)

        # 计算成功率
        successful_requests = [r for r in recent_records if r.success]
        success_rate = len(successful_requests) / len(recent_records)
        error_rate = 1 - success_rate

        # 计算预测分布统计
        predictions = [
            r.prediction for r in successful_requests if r.prediction is not None
        ]
        prediction_stats = {}
        if predictions:
            if all(isinstance(p, (int, float)) for p in predictions):
                # 数值预测
                prediction_stats = {
                    "mean": float(np.mean(predictions)),
                    "std": float(np.std(predictions)),
                    "min": float(np.min(predictions)),
                    "max": float(np.max(predictions)),
                }
            else:
                # 分类预测
                from collections import Counter

                pred_counts = Counter(predictions)
                total = len(predictions)
                prediction_stats = {
                    str(k): v / total for k, v in pred_counts.most_common()
                }

        # 计算置信度统计
        confidences = [
            r.confidence for r in successful_requests if r.confidence is not None
        ]
        avg_confidence = statistics.mean(confidences) if confidences else None
        low_confidence_rate = (
            len([c for c in confidences if c < 0.7]) / len(confidences)
            if confidences
            else 0.0
        )

        return PerformanceMetrics(
            timestamp=datetime.now(),
            model_id=model_id,
            model_version=model_version,
            avg_latency_ms=avg_latency,
            p50_latency_ms=p50_latency,
            p95_latency_ms=p95_latency,
            p99_latency_ms=p99_latency,
            requests_per_second=requests_per_second,
            total_requests=len(recent_records),
            success_rate=success_rate,
            error_rate=error_rate,
            prediction_stats=prediction_stats,
            avg_confidence=avg_confidence,
            low_confidence_rate=low_confidence_rate,
        )

    def get_model_keys(self) -> List[str]:
        """获取所有模型键"""
        with self.lock:
            return list(self.records.keys())


class AlertManager:
    """告警管理器"""

    def __init__(self):
        self.rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.callbacks: List[Callable[[Alert], None]] = []
        self.lock = threading.Lock()

        # 注册默认告警规则
        self._register_default_rules()

    def _register_default_rules(self):
        """注册默认告警规则"""
        default_rules = [
            AlertRule(
                name="high_latency",
                metric_type=MetricType.LATENCY,
                condition="gt",
                threshold=1000.0,  # 1秒
                level=AlertLevel.WARNING,
                description="平均延迟过高",
            ),
            AlertRule(
                name="critical_latency",
                metric_type=MetricType.LATENCY,
                condition="gt",
                threshold=5000.0,  # 5秒
                level=AlertLevel.CRITICAL,
                description="平均延迟严重过高",
            ),
            AlertRule(
                name="high_error_rate",
                metric_type=MetricType.ERROR_RATE,
                condition="gt",
                threshold=0.05,  # 5%
                level=AlertLevel.WARNING,
                description="错误率过高",
            ),
            AlertRule(
                name="critical_error_rate",
                metric_type=MetricType.ERROR_RATE,
                condition="gt",
                threshold=0.20,  # 20%
                level=AlertLevel.CRITICAL,
                description="错误率严重过高",
            ),
            AlertRule(
                name="low_throughput",
                metric_type=MetricType.THROUGHPUT,
                condition="lt",
                threshold=1.0,  # 1 RPS
                level=AlertLevel.WARNING,
                description="吞吐量过低",
            ),
            AlertRule(
                name="low_confidence",
                metric_type=MetricType.ACCURACY,
                condition="gt",
                threshold=0.30,  # 30%的预测置信度低于0.7
                level=AlertLevel.WARNING,
                description="低置信度预测比例过高",
            ),
        ]

        for rule in default_rules:
            self.rules[rule.name] = rule

    def add_rule(self, rule: AlertRule):
        """添加告警规则"""
        with self.lock:
            self.rules[rule.name] = rule
        logger.info(f"添加告警规则: {rule.name}")

    def remove_rule(self, rule_name: str):
        """移除告警规则"""
        with self.lock:
            if rule_name in self.rules:
                del self.rules[rule_name]
                logger.info(f"移除告警规则: {rule_name}")

    def add_callback(self, callback: Callable[[Alert], None]):
        """添加告警回调函数"""
        self.callbacks.append(callback)

    def check_metrics(self, metrics: PerformanceMetrics):
        """检查指标并触发告警"""
        with self.lock:
            for rule_name, rule in self.rules.items():
                if not rule.enabled:
                    continue

                # 获取指标值
                metric_value = self._get_metric_value(metrics, rule.metric_type)
                if metric_value is None:
                    continue

                # 检查条件
                should_alert = self._check_condition(
                    metric_value, rule.condition, rule.threshold
                )
                alert_key = f"{rule_name}_{metrics.model_id}_{metrics.model_version}"

                if should_alert:
                    # 触发告警
                    if alert_key not in self.active_alerts:
                        alert = Alert(
                            alert_id=f"alert_{int(time.time())}_{rule_name}",
                            rule_name=rule_name,
                            level=rule.level,
                            message=f"{rule.description}: {metric_value:.2f} {rule.condition} {rule.threshold}",
                            metric_value=metric_value,
                            threshold=rule.threshold,
                            timestamp=datetime.now(),
                            model_id=metrics.model_id,
                            model_version=metrics.model_version,
                        )

                        self.active_alerts[alert_key] = alert
                        self.alert_history.append(alert)

                        # 通知回调函数
                        for callback in self.callbacks:
                            try:
                                callback(alert)
                            except Exception as e:
                                logger.error(f"告警回调执行失败: {e}")

                        logger.warning(f"触发告警: {alert.message}")
                else:
                    # 解除告警
                    if alert_key in self.active_alerts:
                        alert = self.active_alerts[alert_key]
                        alert.resolved = True
                        alert.resolved_at = datetime.now()
                        del self.active_alerts[alert_key]

                        logger.info(f"解除告警: {alert.message}")

    def _get_metric_value(
        self, metrics: PerformanceMetrics, metric_type: MetricType
    ) -> Optional[float]:
        """获取指标值"""
        if metric_type == MetricType.LATENCY:
            return metrics.avg_latency_ms
        elif metric_type == MetricType.ERROR_RATE:
            return metrics.error_rate
        elif metric_type == MetricType.THROUGHPUT:
            return metrics.requests_per_second
        elif metric_type == MetricType.ACCURACY:
            return metrics.low_confidence_rate
        else:
            return None

    def _check_condition(self, value: float, condition: str, threshold: float) -> bool:
        """检查条件"""
        if condition == "gt":
            return value > threshold
        elif condition == "lt":
            return value < threshold
        elif condition == "eq":
            return abs(value - threshold) < 1e-6
        elif condition == "ne":
            return abs(value - threshold) >= 1e-6
        else:
            return False

    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """获取活跃告警"""
        with self.lock:
            return [a.to_dict() for a in self.active_alerts.values()]

    def get_alert_history(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        alert_type: Optional[str] = None,
        severity: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """获取告警历史"""
        with self.lock:
            history = self.alert_history.copy()
        
        # 时间过滤
        if start_time:
            history = [a for a in history if a.timestamp >= start_time]
        if end_time:
            history = [a for a in history if a.timestamp <= end_time]
        
        # 类型过滤
        if alert_type:
            history = [a for a in history if a.rule_name.startswith(alert_type)]
        
        # 严重程度过滤
        if severity:
            history = [a for a in history if a.level.value == severity]
        
        return [a.to_dict() for a in history[-limit:]]

    def get_alert_configs(
        self,
        alert_type: Optional[str] = None,
        enabled: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """获取告警配置列表"""
        with self.lock:
            configs = list(self.rules.values())
        
        if alert_type:
            configs = [c for c in configs if c.metric_type.value == alert_type]
        if enabled is not None:
            configs = [c for c in configs if c.enabled == enabled]
        
        return [c.to_dict() for c in configs]

    def get_alert_config(self, alert_id: str) -> Optional[Dict[str, Any]]:
        """获取告警配置详情"""
        with self.lock:
            rule = self.rules.get(alert_id)
            return rule.to_dict() if rule else None

    def create_alert_config(self, config: Dict[str, Any]) -> str:
        """创建告警配置"""
        alert_id = f"alert_{config['alert_type']}_{config['metric_name']}_{int(time.time())}"
        
        rule = AlertRule(
            name=alert_id,
            metric_type=MetricType(config.get('alert_type', 'latency')),
            condition=config.get('comparison', 'gt'),
            threshold=config.get('threshold', 0),
            level=AlertLevel(config.get('severity', 'warning')),
            enabled=config.get('enabled', True),
            description=config.get('description', ''),
        )
        
        with self.lock:
            self.rules[alert_id] = rule
        
        return alert_id

    def update_alert_config(self, alert_id: str, update_data: Dict[str, Any]) -> bool:
        """更新告警配置"""
        with self.lock:
            if alert_id not in self.rules:
                return False
            
            rule = self.rules[alert_id]
            if 'threshold' in update_data:
                rule.threshold = update_data['threshold']
            if 'comparison' in update_data:
                rule.condition = update_data['comparison']
            if 'enabled' in update_data:
                rule.enabled = update_data['enabled']
            if 'description' in update_data:
                rule.description = update_data['description']
            
            return True

    def delete_alert_config(self, alert_id: str) -> bool:
        """删除告警配置"""
        with self.lock:
            if alert_id in self.rules:
                del self.rules[alert_id]
                return True
            return False

    def resolve_alert(self, alert_id: str, resolution_note: Optional[str] = None) -> bool:
        """解决告警"""
        with self.lock:
            for key, alert in self.active_alerts.items():
                if alert.alert_id == alert_id:
                    alert.resolved = True
                    alert.resolved_at = datetime.now()
                    del self.active_alerts[key]
                    return True
            return False

    def test_alert(
        self,
        alert_type: str,
        metric_name: str,
        test_value: float,
    ) -> Dict[str, Any]:
        """测试告警配置"""
        triggered_rules = []
        
        with self.lock:
            for rule_name, rule in self.rules.items():
                if not rule.enabled:
                    continue
                
                should_alert = self._check_condition(test_value, rule.condition, rule.threshold)
                if should_alert:
                    triggered_rules.append({
                        "rule_name": rule_name,
                        "threshold": rule.threshold,
                        "condition": rule.condition,
                        "level": rule.level.value,
                    })
        
        return {
            "test_value": test_value,
            "alert_type": alert_type,
            "metric_name": metric_name,
            "triggered_rules": triggered_rules,
            "would_alert": len(triggered_rules) > 0,
        }


class PerformanceMonitor:
    """模型性能监控器"""

    def __init__(self, window_size: int = 1000, metrics_interval: float = 60.0):
        """
        初始化性能监控器

        Args:
            window_size: 滑动窗口大小
            metrics_interval: 指标计算间隔（秒）
        """
        self.window_size = window_size
        self.metrics_interval = metrics_interval

        # 组件
        self.metrics_calculator = MetricsCalculator(window_size)
        self.alert_manager = AlertManager()

        # 存储
        self.metrics_history: List[PerformanceMetrics] = []
        self.max_history_size = 10000

        # 监控任务
        self.monitoring_task: Optional[asyncio.Task] = None
        self.running = False

        logger.info("性能监控器初始化完成")

    def record_prediction(
        self,
        request_id: str,
        model_id: str,
        model_version: str,
        input_features: Dict[str, Any],
        prediction: Any,
        confidence: Optional[float] = None,
        latency_ms: float = 0.0,
        success: bool = True,
        error_message: Optional[str] = None,
    ):
        """记录预测请求"""
        record = PredictionRecord(
            request_id=request_id,
            timestamp=datetime.now(),
            model_id=model_id,
            model_version=model_version,
            input_features=input_features,
            prediction=prediction,
            confidence=confidence,
            latency_ms=latency_ms,
            success=success,
            error_message=error_message,
        )

        self.metrics_calculator.add_record(record)
        logger.debug(f"记录预测: {request_id}, 延迟: {latency_ms}ms, 成功: {success}")

    async def start_monitoring(self):
        """启动监控"""
        if self.running:
            logger.warning("监控已在运行")
            return

        self.running = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("性能监控已启动")

    async def stop_monitoring(self):
        """停止监控"""
        if not self.running:
            return

        self.running = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass

        logger.info("性能监控已停止")

    async def _monitoring_loop(self):
        """监控循环"""
        while self.running:
            try:
                # 计算所有模型的指标
                model_keys = self.metrics_calculator.get_model_keys()

                for key in model_keys:
                    model_id, model_version = key.split("_", 1)
                    metrics = self.metrics_calculator.calculate_metrics(
                        model_id, model_version
                    )

                    if metrics:
                        # 存储指标历史
                        self.metrics_history.append(metrics)
                        if len(self.metrics_history) > self.max_history_size:
                            self.metrics_history = self.metrics_history[
                                -self.max_history_size :
                            ]

                        # 检查告警
                        self.alert_manager.check_metrics(metrics)

                        logger.debug(
                            f"计算指标: {model_id}_{model_version}, RPS: {metrics.requests_per_second:.2f}"
                        )

                await asyncio.sleep(self.metrics_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"监控循环异常: {e}")
                await asyncio.sleep(self.metrics_interval)

    def get_current_metrics(
        self, model_id: str, model_version: str
    ) -> Optional[PerformanceMetrics]:
        """获取当前指标"""
        return self.metrics_calculator.calculate_metrics(model_id, model_version)

    def get_metrics_history(
        self,
        model_id: Optional[str] = None,
        model_version: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000,
    ) -> List[PerformanceMetrics]:
        """获取指标历史"""
        history = self.metrics_history

        # 过滤条件
        if model_id:
            history = [m for m in history if m.model_id == model_id]

        if model_version:
            history = [m for m in history if m.model_version == model_version]

        if start_time:
            history = [m for m in history if m.timestamp >= start_time]

        if end_time:
            history = [m for m in history if m.timestamp <= end_time]

        return history[-limit:]

    def add_alert_rule(self, rule: AlertRule):
        """添加告警规则"""
        self.alert_manager.add_rule(rule)

    def remove_alert_rule(self, rule_name: str):
        """移除告警规则"""
        self.alert_manager.remove_rule(rule_name)

    def get_alert_rules(self) -> List[AlertRule]:
        """获取所有告警规则"""
        return list(self.alert_manager.rules.values())

    def get_active_alerts(self) -> List[Alert]:
        """获取活跃告警"""
        return self.alert_manager.get_active_alerts()

    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """获取告警历史"""
        return self.alert_manager.get_alert_history(limit)

    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """添加告警回调"""
        self.alert_manager.add_callback(callback)

    def get_summary_stats(self) -> Dict[str, Any]:
        """获取汇总统计"""
        if not self.metrics_history:
            return {}

        recent_metrics = self.metrics_history[-10:]  # 最近10个指标点

        # 计算平均值
        avg_latency = statistics.mean([m.avg_latency_ms for m in recent_metrics])
        avg_throughput = statistics.mean(
            [m.requests_per_second for m in recent_metrics]
        )
        avg_error_rate = statistics.mean([m.error_rate for m in recent_metrics])

        # 活跃告警统计
        active_alerts = self.get_active_alerts()
        alert_counts = {
            "critical": len(
                [a for a in active_alerts if a.level == AlertLevel.CRITICAL]
            ),
            "warning": len([a for a in active_alerts if a.level == AlertLevel.WARNING]),
            "info": len([a for a in active_alerts if a.level == AlertLevel.INFO]),
        }

        return {
            "avg_latency_ms": avg_latency,
            "avg_throughput_rps": avg_throughput,
            "avg_error_rate": avg_error_rate,
            "total_models_monitored": len(
                set(f"{m.model_id}_{m.model_version}" for m in recent_metrics)
            ),
            "active_alerts": alert_counts,
            "total_metrics_points": len(self.metrics_history),
            "monitoring_active": self.running,
        }

    def get_metrics(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        model_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """获取性能指标（API 兼容方法）"""
        history = self.get_metrics_history(
            model_id=model_id,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
        )
        return [m.to_dict() for m in history]

    def get_model_performance(
        self,
        model_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        include_predictions: bool = False,
    ) -> Dict[str, Any]:
        """获取模型性能指标（API 兼容方法）"""
        history = self.get_metrics_history(
            model_id=model_id,
            start_time=start_time,
            end_time=end_time,
        )
        
        if not history:
            return {
                "model_id": model_id,
                "total_predictions": 0,
                "average_accuracy": 0,
                "average_latency": 0,
                "error_rate": 0,
            }
        
        return {
            "model_id": model_id,
            "total_predictions": sum(m.total_requests for m in history),
            "average_accuracy": statistics.mean([m.success_rate for m in history]) if history else 0,
            "average_latency": statistics.mean([m.avg_latency_ms for m in history]) if history else 0,
            "error_rate": statistics.mean([m.error_rate for m in history]) if history else 0,
            "metrics_history": [m.to_dict() for m in history] if include_predictions else [],
        }

    def get_system_status(self) -> Dict[str, Any]:
        """获取系统整体状态（API 兼容方法）"""
        model_keys = self.metrics_calculator.get_model_keys()
        recent_metrics = self.metrics_history[-50:] if self.metrics_history else []
        
        # 计算活跃模型（最近5分钟有数据的模型）
        cutoff_time = datetime.now() - timedelta(minutes=5)
        active_models = set()
        for m in recent_metrics:
            if m.timestamp >= cutoff_time:
                active_models.add(f"{m.model_id}_{m.model_version}")
        
        # 计算平均指标
        avg_latency = 0
        error_rate = 0
        if recent_metrics:
            avg_latency = statistics.mean([m.avg_latency_ms for m in recent_metrics])
            error_rate = statistics.mean([m.error_rate for m in recent_metrics])
        
        return {
            "total_models": len(model_keys),
            "active_models": len(active_models),
            "average_latency": avg_latency,
            "error_rate": error_rate,
            "monitoring_active": self.running,
            "total_metrics_points": len(self.metrics_history),
            "last_update": self.metrics_history[-1].timestamp.isoformat() if self.metrics_history else None,
        }

    def get_recent_metrics(self, limit: int = 50) -> List[Dict[str, Any]]:
        """获取最近的性能指标（API 兼容方法）"""
        recent = self.metrics_history[-limit:] if self.metrics_history else []
        return [m.to_dict() for m in recent]


# 全局性能监控器实例
performance_monitor = PerformanceMonitor()
performance_monitor = PerformanceMonitor()
alert_manager = AlertManager()
