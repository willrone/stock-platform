"""
资源监控服务
监控系统资源使用情况，包括CPU、内存、GPU等
"""
import asyncio
import psutil
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import json
from loguru import logger

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

@dataclass
class ResourceUsage:
    """资源使用情况数据类"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float
    disk_percent: float
    disk_used_gb: float
    disk_total_gb: float
    gpu_usage: Optional[List[Dict[str, Any]]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

@dataclass
class ResourceThresholds:
    """资源阈值配置"""
    cpu_warning: float = 80.0
    cpu_critical: float = 95.0
    memory_warning: float = 80.0
    memory_critical: float = 95.0
    disk_warning: float = 85.0
    disk_critical: float = 95.0
    gpu_memory_warning: float = 80.0
    gpu_memory_critical: float = 95.0

class ResourceMonitor:
    """资源监控器"""
    
    def __init__(self, 
                 thresholds: Optional[ResourceThresholds] = None,
                 history_size: int = 1000):
        """
        初始化资源监控器
        
        Args:
            thresholds: 资源阈值配置
            history_size: 历史数据保存数量
        """
        self.thresholds = thresholds or ResourceThresholds()
        self.history_size = history_size
        self.usage_history: List[ResourceUsage] = []
        self._monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None
        self.callbacks: List[callable] = []
        
    def add_callback(self, callback: callable):
        """添加资源使用回调函数"""
        self.callbacks.append(callback)
        
    def remove_callback(self, callback: callable):
        """移除资源使用回调函数"""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    def get_current_usage(self) -> ResourceUsage:
        """获取当前资源使用情况"""
        # CPU使用率
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # 内存使用情况
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used_gb = memory.used / (1024**3)
        memory_total_gb = memory.total / (1024**3)
        
        # 磁盘使用情况
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        disk_used_gb = disk.used / (1024**3)
        disk_total_gb = disk.total / (1024**3)
        
        # GPU使用情况
        gpu_usage = None
        if GPU_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                gpu_usage = []
                for gpu in gpus:
                    gpu_info = {
                        'id': gpu.id,
                        'name': gpu.name,
                        'load': gpu.load * 100,  # 转换为百分比
                        'memory_used': gpu.memoryUsed,
                        'memory_total': gpu.memoryTotal,
                        'memory_percent': (gpu.memoryUsed / gpu.memoryTotal) * 100,
                        'temperature': gpu.temperature
                    }
                    gpu_usage.append(gpu_info)
            except Exception as e:
                logger.warning(f"获取GPU信息失败: {e}")
                gpu_usage = None
        
        return ResourceUsage(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_used_gb=memory_used_gb,
            memory_total_gb=memory_total_gb,
            disk_percent=disk_percent,
            disk_used_gb=disk_used_gb,
            disk_total_gb=disk_total_gb,
            gpu_usage=gpu_usage
        )
    
    def check_thresholds(self, usage: ResourceUsage) -> List[Dict[str, Any]]:
        """检查资源使用是否超过阈值"""
        alerts = []
        
        # 检查CPU
        if usage.cpu_percent >= self.thresholds.cpu_critical:
            alerts.append({
                'type': 'cpu',
                'level': 'critical',
                'message': f'CPU使用率达到 {usage.cpu_percent:.1f}%',
                'value': usage.cpu_percent,
                'threshold': self.thresholds.cpu_critical
            })
        elif usage.cpu_percent >= self.thresholds.cpu_warning:
            alerts.append({
                'type': 'cpu',
                'level': 'warning',
                'message': f'CPU使用率达到 {usage.cpu_percent:.1f}%',
                'value': usage.cpu_percent,
                'threshold': self.thresholds.cpu_warning
            })
        
        # 检查内存
        if usage.memory_percent >= self.thresholds.memory_critical:
            alerts.append({
                'type': 'memory',
                'level': 'critical',
                'message': f'内存使用率达到 {usage.memory_percent:.1f}%',
                'value': usage.memory_percent,
                'threshold': self.thresholds.memory_critical
            })
        elif usage.memory_percent >= self.thresholds.memory_warning:
            alerts.append({
                'type': 'memory',
                'level': 'warning',
                'message': f'内存使用率达到 {usage.memory_percent:.1f}%',
                'value': usage.memory_percent,
                'threshold': self.thresholds.memory_warning
            })
        
        # 检查磁盘
        if usage.disk_percent >= self.thresholds.disk_critical:
            alerts.append({
                'type': 'disk',
                'level': 'critical',
                'message': f'磁盘使用率达到 {usage.disk_percent:.1f}%',
                'value': usage.disk_percent,
                'threshold': self.thresholds.disk_critical
            })
        elif usage.disk_percent >= self.thresholds.disk_warning:
            alerts.append({
                'type': 'disk',
                'level': 'warning',
                'message': f'磁盘使用率达到 {usage.disk_percent:.1f}%',
                'value': usage.disk_percent,
                'threshold': self.thresholds.disk_warning
            })
        
        # 检查GPU
        if usage.gpu_usage:
            for gpu in usage.gpu_usage:
                gpu_memory_percent = gpu['memory_percent']
                if gpu_memory_percent >= self.thresholds.gpu_memory_critical:
                    alerts.append({
                        'type': 'gpu_memory',
                        'level': 'critical',
                        'message': f'GPU {gpu["id"]} 内存使用率达到 {gpu_memory_percent:.1f}%',
                        'value': gpu_memory_percent,
                        'threshold': self.thresholds.gpu_memory_critical,
                        'gpu_id': gpu['id']
                    })
                elif gpu_memory_percent >= self.thresholds.gpu_memory_warning:
                    alerts.append({
                        'type': 'gpu_memory',
                        'level': 'warning',
                        'message': f'GPU {gpu["id"]} 内存使用率达到 {gpu_memory_percent:.1f}%',
                        'value': gpu_memory_percent,
                        'threshold': self.thresholds.gpu_memory_warning,
                        'gpu_id': gpu['id']
                    })
        
        return alerts
    
    async def _monitor_loop(self, interval: float = 30.0):
        """监控循环"""
        while self._monitoring:
            try:
                # 获取当前资源使用情况
                usage = self.get_current_usage()
                
                # 添加到历史记录
                self.usage_history.append(usage)
                
                # 保持历史记录大小
                if len(self.usage_history) > self.history_size:
                    self.usage_history.pop(0)
                
                # 检查阈值
                alerts = self.check_thresholds(usage)
                
                # 调用回调函数
                for callback in self.callbacks:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(usage, alerts)
                        else:
                            callback(usage, alerts)
                    except Exception as e:
                        logger.error(f"资源监控回调函数执行失败: {e}")
                
                # 记录告警
                if alerts:
                    for alert in alerts:
                        logger.warning(f"资源告警: {alert['message']}")
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"资源监控循环出错: {e}")
                await asyncio.sleep(interval)
    
    async def start_monitoring(self, interval: float = 30.0):
        """开始监控"""
        if self._monitoring:
            logger.warning("资源监控已经在运行")
            return
        
        self._monitoring = True
        self._monitor_task = asyncio.create_task(self._monitor_loop(interval))
        logger.info(f"开始资源监控，间隔 {interval} 秒")
    
    async def stop_monitoring(self):
        """停止监控"""
        if not self._monitoring:
            return
        
        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("资源监控已停止")
    
    def get_usage_history(self, 
                         start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None) -> List[ResourceUsage]:
        """获取历史资源使用数据"""
        if not start_time and not end_time:
            return self.usage_history.copy()
        
        filtered_history = []
        for usage in self.usage_history:
            if start_time and usage.timestamp < start_time:
                continue
            if end_time and usage.timestamp > end_time:
                continue
            filtered_history.append(usage)
        
        return filtered_history
    
    def get_average_usage(self, 
                         duration_minutes: int = 60) -> Optional[ResourceUsage]:
        """获取指定时间段内的平均资源使用情况"""
        if not self.usage_history:
            return None
        
        cutoff_time = datetime.now() - timedelta(minutes=duration_minutes)
        recent_usage = [
            usage for usage in self.usage_history 
            if usage.timestamp >= cutoff_time
        ]
        
        if not recent_usage:
            return None
        
        # 计算平均值
        avg_cpu = sum(u.cpu_percent for u in recent_usage) / len(recent_usage)
        avg_memory = sum(u.memory_percent for u in recent_usage) / len(recent_usage)
        avg_disk = sum(u.disk_percent for u in recent_usage) / len(recent_usage)
        
        # GPU平均值计算
        avg_gpu_usage = None
        if recent_usage[0].gpu_usage:
            gpu_count = len(recent_usage[0].gpu_usage)
            avg_gpu_usage = []
            
            for gpu_id in range(gpu_count):
                gpu_loads = [u.gpu_usage[gpu_id]['load'] for u in recent_usage if u.gpu_usage and len(u.gpu_usage) > gpu_id]
                gpu_memory_percents = [u.gpu_usage[gpu_id]['memory_percent'] for u in recent_usage if u.gpu_usage and len(u.gpu_usage) > gpu_id]
                
                if gpu_loads and gpu_memory_percents:
                    avg_gpu_info = recent_usage[0].gpu_usage[gpu_id].copy()
                    avg_gpu_info['load'] = sum(gpu_loads) / len(gpu_loads)
                    avg_gpu_info['memory_percent'] = sum(gpu_memory_percents) / len(gpu_memory_percents)
                    avg_gpu_usage.append(avg_gpu_info)
        
        return ResourceUsage(
            timestamp=datetime.now(),
            cpu_percent=avg_cpu,
            memory_percent=avg_memory,
            memory_used_gb=recent_usage[-1].memory_used_gb,  # 使用最新值
            memory_total_gb=recent_usage[-1].memory_total_gb,
            disk_percent=avg_disk,
            disk_used_gb=recent_usage[-1].disk_used_gb,  # 使用最新值
            disk_total_gb=recent_usage[-1].disk_total_gb,
            gpu_usage=avg_gpu_usage
        )
    
    def is_resource_available(self, 
                            required_memory_gb: float = 0,
                            required_cpu_percent: float = 0,
                            required_gpu_memory_gb: float = 0) -> Dict[str, Any]:
        """检查是否有足够的资源可用"""
        current_usage = self.get_current_usage()
        
        # 检查内存
        available_memory_gb = current_usage.memory_total_gb - current_usage.memory_used_gb
        memory_available = available_memory_gb >= required_memory_gb
        
        # 检查CPU（基于当前使用率预估）
        estimated_cpu_usage = current_usage.cpu_percent + required_cpu_percent
        cpu_available = estimated_cpu_usage <= self.thresholds.cpu_warning
        
        # 检查GPU内存
        gpu_available = True
        available_gpu_memory = 0
        if required_gpu_memory_gb > 0 and current_usage.gpu_usage:
            for gpu in current_usage.gpu_usage:
                gpu_available_memory = gpu['memory_total'] - gpu['memory_used']
                available_gpu_memory = max(available_gpu_memory, gpu_available_memory / 1024)  # 转换为GB
            
            gpu_available = available_gpu_memory >= required_gpu_memory_gb
        
        return {
            'available': memory_available and cpu_available and gpu_available,
            'memory_available': memory_available,
            'cpu_available': cpu_available,
            'gpu_available': gpu_available,
            'available_memory_gb': available_memory_gb,
            'available_gpu_memory_gb': available_gpu_memory,
            'current_usage': current_usage.to_dict()
        }

# 全局资源监控器实例
resource_monitor = ResourceMonitor()