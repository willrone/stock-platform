"""
业务指标收集器
收集关键业务指标数据，支持实时指标计算
"""
import asyncio
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import statistics
import numpy as np
from collections import defaultdict, deque
import threading
import uuid
from loguru import logger

class MetricType(Enum):
    """指标类型"""
    COUNTER = "counter"  # 计数器
    GAUGE = "gauge"  # 仪表盘
    HISTOGRAM = "histogram"  # 直方图
    RATE = "rate"  # 比率
    CONVERSION = "conversion"  # 转化率
    REVENUE = "revenue"  # 收入
    DURATION = "duration"  # 持续时间

class AggregationType(Enum):
    """聚合类型"""
    SUM = "sum"
    MEAN = "mean"
    MEDIAN = "median"
    COUNT = "count"
    MIN = "min"
    MAX = "max"
    P95 = "p95"
    P99 = "p99"
    RATE = "rate"
    UNIQUE_COUNT = "unique_count"

@dataclass
class MetricDefinition:
    """指标定义"""
    metric_id: str
    name: str
    description: str
    metric_type: MetricType
    aggregation_type: AggregationType
    unit: str = ""
    tags: List[str] = field(default_factory=list)
    # 计算配置
    window_size_minutes: int = 60  # 计算窗口大小（分钟）
    calculation_interval_seconds: int = 60  # 计算间隔（秒）
    # 转化率配置
    numerator_event: Optional[str] = None  # 分子事件
    denominator_event: Optional[str] = None  # 分母事件
    # 收入配置
    revenue_field: Optional[str] = None  # 收入字段名
    # 过滤条件
    filters: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'metric_id': self.metric_id,
            'name': self.name,
            'description': self.description,
            'metric_type': self.metric_type.value,
            'aggregation_type': self.aggregation_type.value,
            'unit': self.unit,
            'tags': self.tags,
            'window_size_minutes': self.window_size_minutes,
            'calculation_interval_seconds': self.calculation_interval_seconds,
            'numerator_event': self.numerator_event,
            'denominator_event': self.denominator_event,
            'revenue_field': self.revenue_field,
            'filters': self.filters
        }

@dataclass
class MetricEvent:
    """指标事件"""
    event_id: str
    user_id: str
    experiment_id: Optional[str]
    variant_id: Optional[str]
    event_name: str
    timestamp: datetime
    properties: Dict[str, Any] = field(default_factory=dict)
    value: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'event_id': self.event_id,
            'user_id': self.user_id,
            'experiment_id': self.experiment_id,
            'variant_id': self.variant_id,
            'event_name': self.event_name,
            'timestamp': self.timestamp.isoformat(),
            'properties': self.properties,
            'value': self.value
        }

@dataclass
class MetricValue:
    """指标值"""
    metric_id: str
    experiment_id: Optional[str]
    variant_id: Optional[str]
    value: float
    timestamp: datetime
    sample_size: int = 0
    confidence_interval: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'metric_id': self.metric_id,
            'experiment_id': self.experiment_id,
            'variant_id': self.variant_id,
            'value': self.value,
            'timestamp': self.timestamp.isoformat(),
            'sample_size': self.sample_size,
            'confidence_interval': self.confidence_interval,
            'metadata': self.metadata
        }

class MetricCalculator:
    """指标计算器"""
    
    def __init__(self, definition: MetricDefinition):
        self.definition = definition
        self.events: deque = deque(maxlen=10000)
        self.lock = threading.Lock()
    
    def add_event(self, event: MetricEvent):
        """添加事件"""
        # 检查过滤条件
        if not self._matches_filters(event):
            return
        
        with self.lock:
            self.events.append(event)
    
    def _matches_filters(self, event: MetricEvent) -> bool:
        """检查事件是否匹配过滤条件"""
        for key, expected_value in self.definition.filters.items():
            if key == 'event_name':
                if event.event_name != expected_value:
                    return False
            elif key in event.properties:
                if event.properties[key] != expected_value:
                    return False
            else:
                return False
        return True
    
    def calculate_metric(
        self, 
        experiment_id: Optional[str] = None,
        variant_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Optional[MetricValue]:
        """计算指标值"""
        with self.lock:
            # 过滤事件
            filtered_events = self._filter_events(experiment_id, variant_id, start_time, end_time)
            
            if not filtered_events:
                return None
            
            # 根据指标类型计算值
            value = self._calculate_value(filtered_events)
            
            if value is None:
                return None
            
            # 计算置信区间
            confidence_interval = self._calculate_confidence_interval(filtered_events, value)
            
            return MetricValue(
                metric_id=self.definition.metric_id,
                experiment_id=experiment_id,
                variant_id=variant_id,
                value=value,
                timestamp=datetime.now(),
                sample_size=len(filtered_events),
                confidence_interval=confidence_interval,
                metadata={
                    'calculation_method': self.definition.aggregation_type.value,
                    'window_start': start_time.isoformat() if start_time else None,
                    'window_end': end_time.isoformat() if end_time else None
                }
            )
    
    def _filter_events(
        self, 
        experiment_id: Optional[str],
        variant_id: Optional[str],
        start_time: Optional[datetime],
        end_time: Optional[datetime]
    ) -> List[MetricEvent]:
        """过滤事件"""
        events = list(self.events)
        
        # 时间过滤
        if start_time:
            events = [e for e in events if e.timestamp >= start_time]
        if end_time:
            events = [e for e in events if e.timestamp <= end_time]
        
        # 实验过滤
        if experiment_id:
            events = [e for e in events if e.experiment_id == experiment_id]
        if variant_id:
            events = [e for e in events if e.variant_id == variant_id]
        
        return events
    
    def _calculate_value(self, events: List[MetricEvent]) -> Optional[float]:
        """计算指标值"""
        if not events:
            return None
        
        if self.definition.metric_type == MetricType.COUNTER:
            return float(len(events))
        
        elif self.definition.metric_type == MetricType.GAUGE:
            # 使用最新的值
            latest_event = max(events, key=lambda e: e.timestamp)
            return latest_event.value if latest_event.value is not None else 1.0
        
        elif self.definition.metric_type in [MetricType.HISTOGRAM, MetricType.DURATION]:
            values = [e.value for e in events if e.value is not None]
            if not values:
                return None
            
            if self.definition.aggregation_type == AggregationType.SUM:
                return float(sum(values))
            elif self.definition.aggregation_type == AggregationType.MEAN:
                return float(statistics.mean(values))
            elif self.definition.aggregation_type == AggregationType.MEDIAN:
                return float(statistics.median(values))
            elif self.definition.aggregation_type == AggregationType.MIN:
                return float(min(values))
            elif self.definition.aggregation_type == AggregationType.MAX:
                return float(max(values))
            elif self.definition.aggregation_type == AggregationType.P95:
                return float(np.percentile(values, 95))
            elif self.definition.aggregation_type == AggregationType.P99:
                return float(np.percentile(values, 99))
            else:
                return float(statistics.mean(values))
        
        elif self.definition.metric_type == MetricType.RATE:
            # 计算事件发生率（每分钟）
            if len(events) < 2:
                return 0.0
            
            time_span = (events[-1].timestamp - events[0].timestamp).total_seconds() / 60.0
            return len(events) / max(time_span, 1.0)
        
        elif self.definition.metric_type == MetricType.CONVERSION:
            return self._calculate_conversion_rate(events)
        
        elif self.definition.metric_type == MetricType.REVENUE:
            return self._calculate_revenue(events)
        
        else:
            return float(len(events))
    
    def _calculate_conversion_rate(self, events: List[MetricEvent]) -> Optional[float]:
        """计算转化率"""
        if not self.definition.numerator_event or not self.definition.denominator_event:
            return None
        
        # 按用户分组
        user_events = defaultdict(list)
        for event in events:
            user_events[event.user_id].append(event)
        
        numerator_users = set()
        denominator_users = set()
        
        for user_id, user_event_list in user_events.items():
            has_numerator = any(e.event_name == self.definition.numerator_event for e in user_event_list)
            has_denominator = any(e.event_name == self.definition.denominator_event for e in user_event_list)
            
            if has_numerator:
                numerator_users.add(user_id)
            if has_denominator:
                denominator_users.add(user_id)
        
        if len(denominator_users) == 0:
            return 0.0
        
        return len(numerator_users) / len(denominator_users)
    
    def _calculate_revenue(self, events: List[MetricEvent]) -> Optional[float]:
        """计算收入"""
        if not self.definition.revenue_field:
            return None
        
        total_revenue = 0.0
        for event in events:
            if self.definition.revenue_field in event.properties:
                try:
                    revenue = float(event.properties[self.definition.revenue_field])
                    total_revenue += revenue
                except (ValueError, TypeError):
                    continue
        
        return total_revenue
    
    def _calculate_confidence_interval(
        self, 
        events: List[MetricEvent], 
        value: float,
        confidence_level: float = 0.95
    ) -> Optional[Dict[str, float]]:
        """计算置信区间"""
        if len(events) < 30:  # 样本量太小
            return None
        
        try:
            if self.definition.metric_type == MetricType.CONVERSION:
                # 二项分布的置信区间
                n = len(set(e.user_id for e in events))  # 用户数
                p = value  # 转化率
                
                if n == 0 or p == 0 or p == 1:
                    return None
                
                # 使用正态近似
                z_score = 1.96 if confidence_level == 0.95 else 2.576  # 99%
                margin_of_error = z_score * np.sqrt(p * (1 - p) / n)
                
                return {
                    'lower': max(0, p - margin_of_error),
                    'upper': min(1, p + margin_of_error),
                    'confidence_level': confidence_level
                }
            
            elif self.definition.metric_type in [MetricType.HISTOGRAM, MetricType.DURATION, MetricType.REVENUE]:
                # 连续变量的置信区间
                values = [e.value for e in events if e.value is not None]
                if len(values) < 30:
                    return None
                
                mean_val = statistics.mean(values)
                std_val = statistics.stdev(values)
                n = len(values)
                
                z_score = 1.96 if confidence_level == 0.95 else 2.576
                margin_of_error = z_score * (std_val / np.sqrt(n))
                
                return {
                    'lower': mean_val - margin_of_error,
                    'upper': mean_val + margin_of_error,
                    'confidence_level': confidence_level
                }
            
        except Exception as e:
            logger.error(f"计算置信区间失败: {e}")
        
        return None

class BusinessMetricsCollector:
    """业务指标收集器"""
    
    def __init__(self):
        self.metric_definitions: Dict[str, MetricDefinition] = {}
        self.metric_calculators: Dict[str, MetricCalculator] = {}
        self.metric_values: List[MetricValue] = []
        self.max_values_history = 100000
        
        # 实时计算任务
        self.calculation_tasks: Dict[str, asyncio.Task] = {}
        self.running = False
        
        # 线程锁
        self.lock = threading.Lock()
        
        # 注册默认指标
        self._register_default_metrics()
        
        logger.info("业务指标收集器初始化完成")
    
    def _register_default_metrics(self):
        """注册默认业务指标"""
        default_metrics = [
            MetricDefinition(
                metric_id="prediction_requests",
                name="预测请求数",
                description="模型预测请求总数",
                metric_type=MetricType.COUNTER,
                aggregation_type=AggregationType.COUNT,
                unit="次",
                tags=["prediction", "usage"],
                filters={"event_name": "prediction_request"}
            ),
            MetricDefinition(
                metric_id="prediction_accuracy",
                name="预测准确率",
                description="模型预测准确率",
                metric_type=MetricType.CONVERSION,
                aggregation_type=AggregationType.RATE,
                unit="%",
                tags=["prediction", "quality"],
                numerator_event="prediction_correct",
                denominator_event="prediction_made"
            ),
            MetricDefinition(
                metric_id="response_time",
                name="响应时间",
                description="模型预测响应时间",
                metric_type=MetricType.HISTOGRAM,
                aggregation_type=AggregationType.P95,
                unit="ms",
                tags=["performance", "latency"],
                filters={"event_name": "prediction_response"}
            ),
            MetricDefinition(
                metric_id="user_engagement",
                name="用户参与度",
                description="用户与系统的交互次数",
                metric_type=MetricType.COUNTER,
                aggregation_type=AggregationType.COUNT,
                unit="次",
                tags=["user", "engagement"],
                filters={"event_name": "user_interaction"}
            ),
            MetricDefinition(
                metric_id="model_confidence",
                name="模型置信度",
                description="模型预测的平均置信度",
                metric_type=MetricType.HISTOGRAM,
                aggregation_type=AggregationType.MEAN,
                unit="",
                tags=["prediction", "confidence"],
                filters={"event_name": "prediction_made"}
            )
        ]
        
        for metric_def in default_metrics:
            self.register_metric(metric_def)
    
    def register_metric(self, metric_definition: MetricDefinition):
        """注册指标定义"""
        with self.lock:
            self.metric_definitions[metric_definition.metric_id] = metric_definition
            self.metric_calculators[metric_definition.metric_id] = MetricCalculator(metric_definition)
        
        logger.info(f"注册业务指标: {metric_definition.name} ({metric_definition.metric_id})")
    
    def unregister_metric(self, metric_id: str):
        """注销指标定义"""
        with self.lock:
            if metric_id in self.metric_definitions:
                del self.metric_definitions[metric_id]
            if metric_id in self.metric_calculators:
                del self.metric_calculators[metric_id]
            
            # 停止计算任务
            if metric_id in self.calculation_tasks:
                self.calculation_tasks[metric_id].cancel()
                del self.calculation_tasks[metric_id]
        
        logger.info(f"注销业务指标: {metric_id}")
    
    def record_event(
        self,
        event_name: str,
        user_id: str,
        experiment_id: Optional[str] = None,
        variant_id: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
        value: Optional[float] = None
    ):
        """记录业务事件"""
        event = MetricEvent(
            event_id=str(uuid.uuid4()),
            user_id=user_id,
            experiment_id=experiment_id,
            variant_id=variant_id,
            event_name=event_name,
            timestamp=datetime.now(),
            properties=properties or {},
            value=value
        )
        
        # 将事件分发给相关的指标计算器
        with self.lock:
            for calculator in self.metric_calculators.values():
                calculator.add_event(event)
        
        logger.debug(f"记录业务事件: {event_name} (用户: {user_id})")
    
    def calculate_metric(
        self,
        metric_id: str,
        experiment_id: Optional[str] = None,
        variant_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Optional[MetricValue]:
        """计算指标值"""
        if metric_id not in self.metric_calculators:
            logger.error(f"指标不存在: {metric_id}")
            return None
        
        calculator = self.metric_calculators[metric_id]
        metric_value = calculator.calculate_metric(experiment_id, variant_id, start_time, end_time)
        
        if metric_value:
            # 存储指标值
            with self.lock:
                self.metric_values.append(metric_value)
                if len(self.metric_values) > self.max_values_history:
                    self.metric_values = self.metric_values[-self.max_values_history:]
        
        return metric_value
    
    def calculate_all_metrics(
        self,
        experiment_id: Optional[str] = None,
        variant_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, MetricValue]:
        """计算所有指标"""
        results = {}
        
        for metric_id in self.metric_definitions:
            metric_value = self.calculate_metric(metric_id, experiment_id, variant_id, start_time, end_time)
            if metric_value:
                results[metric_id] = metric_value
        
        return results
    
    async def start_real_time_calculation(self):
        """启动实时指标计算"""
        if self.running:
            logger.warning("实时计算已在运行")
            return
        
        self.running = True
        
        # 为每个指标启动计算任务
        for metric_id, definition in self.metric_definitions.items():
            task = asyncio.create_task(
                self._metric_calculation_loop(metric_id, definition.calculation_interval_seconds)
            )
            self.calculation_tasks[metric_id] = task
        
        logger.info("启动实时指标计算")
    
    async def stop_real_time_calculation(self):
        """停止实时指标计算"""
        if not self.running:
            return
        
        self.running = False
        
        # 取消所有计算任务
        for task in self.calculation_tasks.values():
            task.cancel()
        
        # 等待任务完成
        if self.calculation_tasks:
            await asyncio.gather(*self.calculation_tasks.values(), return_exceptions=True)
        
        self.calculation_tasks.clear()
        logger.info("停止实时指标计算")
    
    async def _metric_calculation_loop(self, metric_id: str, interval_seconds: int):
        """指标计算循环"""
        while self.running:
            try:
                # 计算指标
                metric_value = self.calculate_metric(metric_id)
                
                if metric_value:
                    logger.debug(f"实时计算指标: {metric_id} = {metric_value.value}")
                
                await asyncio.sleep(interval_seconds)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"指标计算循环异常 {metric_id}: {e}")
                await asyncio.sleep(interval_seconds)
    
    def get_metric_definition(self, metric_id: str) -> Optional[MetricDefinition]:
        """获取指标定义"""
        return self.metric_definitions.get(metric_id)
    
    def list_metric_definitions(self) -> List[MetricDefinition]:
        """列出所有指标定义"""
        return list(self.metric_definitions.values())
    
    def get_metric_history(
        self,
        metric_id: Optional[str] = None,
        experiment_id: Optional[str] = None,
        variant_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[MetricValue]:
        """获取指标历史"""
        with self.lock:
            values = self.metric_values
            
            # 过滤条件
            if metric_id:
                values = [v for v in values if v.metric_id == metric_id]
            
            if experiment_id:
                values = [v for v in values if v.experiment_id == experiment_id]
            
            if variant_id:
                values = [v for v in values if v.variant_id == variant_id]
            
            if start_time:
                values = [v for v in values if v.timestamp >= start_time]
            
            if end_time:
                values = [v for v in values if v.timestamp <= end_time]
            
            return values[-limit:]
    
    def get_experiment_metrics_summary(self, experiment_id: str) -> Dict[str, Any]:
        """获取实验指标摘要"""
        # 计算实验的所有指标
        metrics = self.calculate_all_metrics(experiment_id=experiment_id)
        
        summary = {
            'experiment_id': experiment_id,
            'timestamp': datetime.now().isoformat(),
            'metrics': {},
            'total_metrics': len(metrics)
        }
        
        for metric_id, metric_value in metrics.items():
            definition = self.metric_definitions.get(metric_id)
            summary['metrics'][metric_id] = {
                'name': definition.name if definition else metric_id,
                'value': metric_value.value,
                'unit': definition.unit if definition else '',
                'sample_size': metric_value.sample_size,
                'confidence_interval': metric_value.confidence_interval
            }
        
        return summary

# 全局业务指标收集器实例
business_metrics_collector = BusinessMetricsCollector()