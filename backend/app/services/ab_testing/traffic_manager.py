"""
A/B测试流量分割管理器
支持按比例分割用户流量，实现用户分组和标识
"""
import logging
import hashlib
import random
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import threading
from collections import defaultdict, deque
import uuid

logger = logging.getLogger(__name__)

class TrafficSplitMethod(Enum):
    """流量分割方法"""
    HASH_BASED = "hash_based"  # 基于哈希的分割
    RANDOM = "random"  # 随机分割
    WEIGHTED_RANDOM = "weighted_random"  # 加权随机分割
    STICKY_SESSION = "sticky_session"  # 粘性会话

class ExperimentStatus(Enum):
    """实验状态"""
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

class VariantType(Enum):
    """变体类型"""
    CONTROL = "control"  # 对照组
    TREATMENT = "treatment"  # 实验组

@dataclass
class TrafficVariant:
    """流量变体"""
    variant_id: str
    name: str
    description: str
    variant_type: VariantType
    traffic_percentage: float
    model_id: Optional[str] = None
    model_version: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'variant_id': self.variant_id,
            'name': self.name,
            'description': self.description,
            'variant_type': self.variant_type.value,
            'traffic_percentage': self.traffic_percentage,
            'model_id': self.model_id,
            'model_version': self.model_version,
            'config': self.config
        }

@dataclass
class ABExperiment:
    """A/B测试实验"""
    experiment_id: str
    name: str
    description: str
    status: ExperimentStatus
    variants: List[TrafficVariant]
    split_method: TrafficSplitMethod
    # 实验配置
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    target_sample_size: Optional[int] = None
    confidence_level: float = 0.95
    # 流量配置
    traffic_allocation: float = 1.0  # 参与实验的流量比例
    user_id_field: str = "user_id"  # 用户标识字段
    # 分层配置
    layer_name: Optional[str] = None
    layer_priority: int = 0
    # 过滤条件
    inclusion_criteria: Dict[str, Any] = field(default_factory=dict)
    exclusion_criteria: Dict[str, Any] = field(default_factory=dict)
    # 元数据
    created_at: datetime = field(default_factory=datetime.now)
    created_by: str = ""
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'experiment_id': self.experiment_id,
            'name': self.name,
            'description': self.description,
            'status': self.status.value,
            'variants': [v.to_dict() for v in self.variants],
            'split_method': self.split_method.value,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'target_sample_size': self.target_sample_size,
            'confidence_level': self.confidence_level,
            'traffic_allocation': self.traffic_allocation,
            'user_id_field': self.user_id_field,
            'layer_name': self.layer_name,
            'layer_priority': self.layer_priority,
            'inclusion_criteria': self.inclusion_criteria,
            'exclusion_criteria': self.exclusion_criteria,
            'created_at': self.created_at.isoformat(),
            'created_by': self.created_by,
            'tags': self.tags
        }

@dataclass
class UserAssignment:
    """用户分组分配"""
    user_id: str
    experiment_id: str
    variant_id: str
    assigned_at: datetime
    session_id: Optional[str] = None
    user_attributes: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'user_id': self.user_id,
            'experiment_id': self.experiment_id,
            'variant_id': self.variant_id,
            'assigned_at': self.assigned_at.isoformat(),
            'session_id': self.session_id,
            'user_attributes': self.user_attributes
        }

@dataclass
class TrafficAllocation:
    """流量分配结果"""
    user_id: str
    experiment_id: str
    variant_id: str
    variant_name: str
    model_id: Optional[str]
    model_version: Optional[str]
    assignment_reason: str
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'user_id': self.user_id,
            'experiment_id': self.experiment_id,
            'variant_id': self.variant_id,
            'variant_name': self.variant_name,
            'model_id': self.model_id,
            'model_version': self.model_version,
            'assignment_reason': self.assignment_reason,
            'timestamp': self.timestamp.isoformat()
        }

class HashBasedSplitter:
    """基于哈希的流量分割器"""
    
    def __init__(self, salt: str = "ab_test_salt"):
        self.salt = salt
    
    def assign_variant(
        self, 
        user_id: str, 
        experiment: ABExperiment
    ) -> Optional[TrafficVariant]:
        """分配用户到变体"""
        # 创建哈希键
        hash_key = f"{user_id}_{experiment.experiment_id}_{self.salt}"
        
        # 计算哈希值
        hash_value = hashlib.md5(hash_key.encode()).hexdigest()
        hash_int = int(hash_value[:8], 16)
        
        # 转换为0-100的百分比
        percentage = (hash_int % 10000) / 100.0
        
        # 根据流量分配确定是否参与实验
        if percentage >= experiment.traffic_allocation * 100:
            return None  # 不参与实验
        
        # 在参与实验的用户中分配变体
        cumulative_percentage = 0.0
        for variant in experiment.variants:
            cumulative_percentage += variant.traffic_percentage
            if percentage < cumulative_percentage:
                return variant
        
        # 默认返回第一个变体
        return experiment.variants[0] if experiment.variants else None

class RandomSplitter:
    """随机流量分割器"""
    
    def __init__(self, seed: Optional[int] = None):
        self.random = random.Random(seed)
    
    def assign_variant(
        self, 
        user_id: str, 
        experiment: ABExperiment
    ) -> Optional[TrafficVariant]:
        """随机分配用户到变体"""
        # 检查是否参与实验
        if self.random.random() >= experiment.traffic_allocation:
            return None
        
        # 随机选择变体
        total_weight = sum(v.traffic_percentage for v in experiment.variants)
        if total_weight <= 0:
            return None
        
        rand_value = self.random.random() * total_weight
        cumulative_weight = 0.0
        
        for variant in experiment.variants:
            cumulative_weight += variant.traffic_percentage
            if rand_value <= cumulative_weight:
                return variant
        
        return experiment.variants[-1] if experiment.variants else None

class StickySessionSplitter:
    """粘性会话分割器"""
    
    def __init__(self):
        self.user_assignments: Dict[str, Dict[str, str]] = defaultdict(dict)
        self.lock = threading.Lock()
    
    def assign_variant(
        self, 
        user_id: str, 
        experiment: ABExperiment,
        session_id: Optional[str] = None
    ) -> Optional[TrafficVariant]:
        """基于会话的粘性分配"""
        with self.lock:
            # 检查是否已有分配
            if user_id in self.user_assignments and experiment.experiment_id in self.user_assignments[user_id]:
                variant_id = self.user_assignments[user_id][experiment.experiment_id]
                for variant in experiment.variants:
                    if variant.variant_id == variant_id:
                        return variant
            
            # 使用哈希分割器进行新分配
            hash_splitter = HashBasedSplitter()
            assigned_variant = hash_splitter.assign_variant(user_id, experiment)
            
            if assigned_variant:
                self.user_assignments[user_id][experiment.experiment_id] = assigned_variant.variant_id
            
            return assigned_variant

class TrafficManager:
    """流量分割管理器"""
    
    def __init__(self):
        self.experiments: Dict[str, ABExperiment] = {}
        self.user_assignments: Dict[str, List[UserAssignment]] = defaultdict(list)
        self.assignment_history: deque = deque(maxlen=10000)
        
        # 分割器
        self.splitters = {
            TrafficSplitMethod.HASH_BASED: HashBasedSplitter(),
            TrafficSplitMethod.RANDOM: RandomSplitter(),
            TrafficSplitMethod.WEIGHTED_RANDOM: RandomSplitter(),
            TrafficSplitMethod.STICKY_SESSION: StickySessionSplitter()
        }
        
        # 线程锁
        self.lock = threading.Lock()
        
        logger.info("流量分割管理器初始化完成")
    
    def create_experiment(self, experiment: ABExperiment) -> str:
        """创建A/B测试实验"""
        # 验证实验配置
        self._validate_experiment(experiment)
        
        with self.lock:
            self.experiments[experiment.experiment_id] = experiment
        
        logger.info(f"创建A/B测试实验: {experiment.name} ({experiment.experiment_id})")
        return experiment.experiment_id
    
    def _validate_experiment(self, experiment: ABExperiment):
        """验证实验配置"""
        if not experiment.variants:
            raise ValueError("实验必须至少包含一个变体")
        
        # 检查流量分配总和
        total_percentage = sum(v.traffic_percentage for v in experiment.variants)
        if abs(total_percentage - 100.0) > 0.01:
            raise ValueError(f"变体流量分配总和必须为100%，当前为{total_percentage}%")
        
        # 检查变体ID唯一性
        variant_ids = [v.variant_id for v in experiment.variants]
        if len(variant_ids) != len(set(variant_ids)):
            raise ValueError("变体ID必须唯一")
        
        # 检查至少有一个对照组
        control_variants = [v for v in experiment.variants if v.variant_type == VariantType.CONTROL]
        if not control_variants:
            raise ValueError("实验必须至少包含一个对照组变体")
    
    def update_experiment(self, experiment_id: str, updates: Dict[str, Any]) -> bool:
        """更新实验配置"""
        with self.lock:
            if experiment_id not in self.experiments:
                return False
            
            experiment = self.experiments[experiment_id]
            
            # 只允许更新特定字段
            allowed_fields = ['description', 'end_time', 'target_sample_size', 'tags']
            for field, value in updates.items():
                if field in allowed_fields:
                    setattr(experiment, field, value)
            
            logger.info(f"更新实验配置: {experiment_id}")
            return True
    
    def start_experiment(self, experiment_id: str) -> bool:
        """启动实验"""
        with self.lock:
            if experiment_id not in self.experiments:
                return False
            
            experiment = self.experiments[experiment_id]
            if experiment.status != ExperimentStatus.DRAFT:
                return False
            
            experiment.status = ExperimentStatus.ACTIVE
            experiment.start_time = datetime.now()
            
            logger.info(f"启动A/B测试实验: {experiment_id}")
            return True
    
    def pause_experiment(self, experiment_id: str) -> bool:
        """暂停实验"""
        with self.lock:
            if experiment_id not in self.experiments:
                return False
            
            experiment = self.experiments[experiment_id]
            if experiment.status != ExperimentStatus.ACTIVE:
                return False
            
            experiment.status = ExperimentStatus.PAUSED
            
            logger.info(f"暂停A/B测试实验: {experiment_id}")
            return True
    
    def stop_experiment(self, experiment_id: str) -> bool:
        """停止实验"""
        with self.lock:
            if experiment_id not in self.experiments:
                return False
            
            experiment = self.experiments[experiment_id]
            if experiment.status not in [ExperimentStatus.ACTIVE, ExperimentStatus.PAUSED]:
                return False
            
            experiment.status = ExperimentStatus.COMPLETED
            experiment.end_time = datetime.now()
            
            logger.info(f"停止A/B测试实验: {experiment_id}")
            return True
    
    def assign_user_to_experiment(
        self, 
        user_id: str, 
        experiment_id: str,
        user_attributes: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None
    ) -> Optional[TrafficAllocation]:
        """为用户分配实验变体"""
        with self.lock:
            if experiment_id not in self.experiments:
                return None
            
            experiment = self.experiments[experiment_id]
            
            # 检查实验状态
            if experiment.status != ExperimentStatus.ACTIVE:
                return None
            
            # 检查实验时间范围
            now = datetime.now()
            if experiment.start_time and now < experiment.start_time:
                return None
            if experiment.end_time and now > experiment.end_time:
                return None
            
            # 检查用户是否已分配
            existing_assignment = self._get_user_assignment(user_id, experiment_id)
            if existing_assignment:
                variant = self._get_variant_by_id(experiment, existing_assignment.variant_id)
                if variant:
                    return TrafficAllocation(
                        user_id=user_id,
                        experiment_id=experiment_id,
                        variant_id=variant.variant_id,
                        variant_name=variant.name,
                        model_id=variant.model_id,
                        model_version=variant.model_version,
                        assignment_reason="existing_assignment",
                        timestamp=now
                    )
            
            # 检查包含/排除条件
            if not self._check_user_criteria(user_attributes or {}, experiment):
                return None
            
            # 使用分割器分配变体
            splitter = self.splitters.get(experiment.split_method)
            if not splitter:
                logger.error(f"不支持的分割方法: {experiment.split_method}")
                return None
            
            assigned_variant = None
            if experiment.split_method == TrafficSplitMethod.STICKY_SESSION:
                assigned_variant = splitter.assign_variant(user_id, experiment, session_id)
            else:
                assigned_variant = splitter.assign_variant(user_id, experiment)
            
            if not assigned_variant:
                return None
            
            # 记录用户分配
            assignment = UserAssignment(
                user_id=user_id,
                experiment_id=experiment_id,
                variant_id=assigned_variant.variant_id,
                assigned_at=now,
                session_id=session_id,
                user_attributes=user_attributes or {}
            )
            
            self.user_assignments[user_id].append(assignment)
            self.assignment_history.append(assignment)
            
            # 创建流量分配结果
            allocation = TrafficAllocation(
                user_id=user_id,
                experiment_id=experiment_id,
                variant_id=assigned_variant.variant_id,
                variant_name=assigned_variant.name,
                model_id=assigned_variant.model_id,
                model_version=assigned_variant.model_version,
                assignment_reason="new_assignment",
                timestamp=now
            )
            
            logger.debug(f"用户分配: {user_id} -> {experiment_id}:{assigned_variant.variant_id}")
            return allocation
    
    def _get_user_assignment(self, user_id: str, experiment_id: str) -> Optional[UserAssignment]:
        """获取用户的实验分配"""
        assignments = self.user_assignments.get(user_id, [])
        for assignment in assignments:
            if assignment.experiment_id == experiment_id:
                return assignment
        return None
    
    def _get_variant_by_id(self, experiment: ABExperiment, variant_id: str) -> Optional[TrafficVariant]:
        """根据ID获取变体"""
        for variant in experiment.variants:
            if variant.variant_id == variant_id:
                return variant
        return None
    
    def _check_user_criteria(self, user_attributes: Dict[str, Any], experiment: ABExperiment) -> bool:
        """检查用户是否满足实验条件"""
        # 检查包含条件
        for key, expected_value in experiment.inclusion_criteria.items():
            if key not in user_attributes:
                return False
            if user_attributes[key] != expected_value:
                return False
        
        # 检查排除条件
        for key, excluded_value in experiment.exclusion_criteria.items():
            if key in user_attributes and user_attributes[key] == excluded_value:
                return False
        
        return True
    
    def get_experiment(self, experiment_id: str) -> Optional[ABExperiment]:
        """获取实验配置"""
        return self.experiments.get(experiment_id)
    
    def list_experiments(
        self, 
        status: Optional[ExperimentStatus] = None,
        layer_name: Optional[str] = None
    ) -> List[ABExperiment]:
        """列出实验"""
        experiments = list(self.experiments.values())
        
        if status:
            experiments = [e for e in experiments if e.status == status]
        
        if layer_name:
            experiments = [e for e in experiments if e.layer_name == layer_name]
        
        return experiments
    
    def get_user_assignments(self, user_id: str) -> List[UserAssignment]:
        """获取用户的所有实验分配"""
        return self.user_assignments.get(user_id, [])
    
    def get_experiment_assignments(
        self, 
        experiment_id: str,
        limit: int = 1000
    ) -> List[UserAssignment]:
        """获取实验的用户分配"""
        assignments = []
        for user_assignments in self.user_assignments.values():
            for assignment in user_assignments:
                if assignment.experiment_id == experiment_id:
                    assignments.append(assignment)
                    if len(assignments) >= limit:
                        break
            if len(assignments) >= limit:
                break
        
        return assignments
    
    def get_traffic_stats(self, experiment_id: str) -> Dict[str, Any]:
        """获取流量统计"""
        if experiment_id not in self.experiments:
            return {}
        
        experiment = self.experiments[experiment_id]
        assignments = self.get_experiment_assignments(experiment_id)
        
        # 按变体统计
        variant_stats = defaultdict(int)
        for assignment in assignments:
            variant_stats[assignment.variant_id] += 1
        
        total_assignments = len(assignments)
        
        # 计算实际流量分配
        actual_allocation = {}
        for variant in experiment.variants:
            count = variant_stats[variant.variant_id]
            percentage = (count / total_assignments * 100) if total_assignments > 0 else 0
            actual_allocation[variant.variant_id] = {
                'name': variant.name,
                'expected_percentage': variant.traffic_percentage,
                'actual_percentage': percentage,
                'user_count': count
            }
        
        return {
            'experiment_id': experiment_id,
            'experiment_name': experiment.name,
            'status': experiment.status.value,
            'total_users': total_assignments,
            'variant_allocation': actual_allocation,
            'start_time': experiment.start_time.isoformat() if experiment.start_time else None,
            'end_time': experiment.end_time.isoformat() if experiment.end_time else None
        }
    
    def delete_experiment(self, experiment_id: str) -> bool:
        """删除实验"""
        with self.lock:
            if experiment_id not in self.experiments:
                return False
            
            experiment = self.experiments[experiment_id]
            if experiment.status == ExperimentStatus.ACTIVE:
                return False  # 不能删除活跃的实验
            
            del self.experiments[experiment_id]
            
            # 清理用户分配记录
            for user_id in list(self.user_assignments.keys()):
                self.user_assignments[user_id] = [
                    a for a in self.user_assignments[user_id] 
                    if a.experiment_id != experiment_id
                ]
                if not self.user_assignments[user_id]:
                    del self.user_assignments[user_id]
            
            logger.info(f"删除A/B测试实验: {experiment_id}")
            return True

# 全局流量管理器实例
traffic_manager = TrafficManager()