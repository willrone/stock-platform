"""
增强的日志记录系统
实现结构化日志、日志轮转和清理功能
"""

import json
import logging
import logging.handlers
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import time
import gzip
import os


class LogLevel(Enum):
    """日志级别"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogCategory(Enum):
    """日志分类"""
    SYSTEM = "system"
    API = "api"
    DATA = "data"
    MONITORING = "monitoring"
    PERFORMANCE = "performance"
    SECURITY = "security"
    ERROR = "error"


@dataclass
class StructuredLogEntry:
    """结构化日志条目"""
    timestamp: str
    level: str
    category: str
    message: str
    service: str
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    duration_ms: Optional[float] = None
    status_code: Optional[int] = None
    error_code: Optional[str] = None
    stack_trace: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False, default=str)


class StructuredFormatter(logging.Formatter):
    """结构化日志格式化器"""
    
    def __init__(self, service_name: str = "stock-platform"):
        super().__init__()
        self.service_name = service_name
    
    def format(self, record: logging.LogRecord) -> str:
        """格式化日志记录"""
        # 获取额外的上下文信息
        extra_data = getattr(record, 'extra_data', {})
        
        log_entry = StructuredLogEntry(
            timestamp=datetime.fromtimestamp(record.created).isoformat(),
            level=record.levelname,
            category=extra_data.get('category', LogCategory.SYSTEM.value),
            message=record.getMessage(),
            service=self.service_name,
            request_id=extra_data.get('request_id'),
            user_id=extra_data.get('user_id'),
            session_id=extra_data.get('session_id'),
            duration_ms=extra_data.get('duration_ms'),
            status_code=extra_data.get('status_code'),
            error_code=extra_data.get('error_code'),
            stack_trace=extra_data.get('stack_trace'),
            metadata=extra_data.get('metadata')
        )
        
        return log_entry.to_json()


class LogRotationManager:
    """日志轮转管理器"""
    
    def __init__(
        self,
        log_dir: Path,
        max_file_size: int = 100 * 1024 * 1024,  # 100MB
        max_files: int = 10,
        compress_old_files: bool = True
    ):
        self.log_dir = Path(log_dir)
        self.max_file_size = max_file_size
        self.max_files = max_files
        self.compress_old_files = compress_old_files
        
        # 确保日志目录存在
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 启动清理线程
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()
    
    def _cleanup_loop(self):
        """清理循环"""
        while True:
            try:
                self.cleanup_old_logs()
                time.sleep(3600)  # 每小时检查一次
            except Exception as e:
                logging.error(f"日志清理失败: {e}")
                time.sleep(300)  # 出错后5分钟再试
    
    def cleanup_old_logs(self):
        """清理旧日志文件"""
        try:
            # 获取所有日志文件
            log_files = []
            for pattern in ['*.log', '*.log.*', '*.gz']:
                log_files.extend(self.log_dir.glob(pattern))
            
            # 按修改时间排序
            log_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            
            # 压缩旧文件
            if self.compress_old_files:
                self._compress_old_files(log_files)
            
            # 删除超出数量限制的文件
            if len(log_files) > self.max_files:
                for old_file in log_files[self.max_files:]:
                    try:
                        old_file.unlink()
                        logging.info(f"删除旧日志文件: {old_file}")
                    except Exception as e:
                        logging.error(f"删除日志文件失败 {old_file}: {e}")
            
            # 删除超过30天的日志文件
            cutoff_time = time.time() - (30 * 24 * 3600)  # 30天前
            for log_file in log_files:
                if log_file.stat().st_mtime < cutoff_time:
                    try:
                        log_file.unlink()
                        logging.info(f"删除过期日志文件: {log_file}")
                    except Exception as e:
                        logging.error(f"删除过期日志文件失败 {log_file}: {e}")
        
        except Exception as e:
            logging.error(f"日志清理过程失败: {e}")
    
    def _compress_old_files(self, log_files: List[Path]):
        """压缩旧日志文件"""
        for log_file in log_files:
            # 跳过已压缩的文件和当前活跃的日志文件
            if log_file.suffix == '.gz' or log_file.name.endswith('.log'):
                continue
            
            # 检查文件是否足够旧（1天以上）
            if time.time() - log_file.stat().st_mtime < 24 * 3600:
                continue
            
            try:
                compressed_path = log_file.with_suffix(log_file.suffix + '.gz')
                
                with open(log_file, 'rb') as f_in:
                    with gzip.open(compressed_path, 'wb') as f_out:
                        f_out.writelines(f_in)
                
                # 删除原文件
                log_file.unlink()
                logging.info(f"压缩日志文件: {log_file} -> {compressed_path}")
                
            except Exception as e:
                logging.error(f"压缩日志文件失败 {log_file}: {e}")


class EnhancedLogger:
    """增强的日志记录器"""
    
    def __init__(
        self,
        name: str,
        log_dir: str = "logs",
        service_name: str = "stock-platform",
        enable_console: bool = True,
        enable_file: bool = True,
        log_level: LogLevel = LogLevel.INFO
    ):
        self.name = name
        self.service_name = service_name
        self.log_dir = Path(log_dir)
        
        # 创建日志目录
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.value))
        
        # 清除现有的处理器
        self.logger.handlers.clear()
        
        # 创建格式化器
        self.formatter = StructuredFormatter(service_name)
        
        # 添加控制台处理器
        if enable_console:
            self._add_console_handler()
        
        # 添加文件处理器
        if enable_file:
            self._add_file_handler()
        
        # 创建日志轮转管理器
        self.rotation_manager = LogRotationManager(self.log_dir)
        
        # 性能统计
        self.stats = {
            'total_logs': 0,
            'logs_by_level': {level.value: 0 for level in LogLevel},
            'logs_by_category': {cat.value: 0 for cat in LogCategory},
            'start_time': datetime.now()
        }
    
    def _add_console_handler(self):
        """添加控制台处理器"""
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(self.formatter)
        self.logger.addHandler(console_handler)
    
    def _add_file_handler(self):
        """添加文件处理器"""
        # 主日志文件
        main_log_file = self.log_dir / f"{self.service_name}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            main_log_file,
            maxBytes=100 * 1024 * 1024,  # 100MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setFormatter(self.formatter)
        self.logger.addHandler(file_handler)
        
        # 错误日志文件
        error_log_file = self.log_dir / f"{self.service_name}-error.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=50 * 1024 * 1024,  # 50MB
            backupCount=3,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(self.formatter)
        self.logger.addHandler(error_handler)
    
    def _log_with_context(
        self,
        level: LogLevel,
        message: str,
        category: LogCategory = LogCategory.SYSTEM,
        **kwargs
    ):
        """带上下文的日志记录"""
        extra_data = {
            'category': category.value,
            **kwargs
        }
        
        # 更新统计信息
        self.stats['total_logs'] += 1
        self.stats['logs_by_level'][level.value] += 1
        self.stats['logs_by_category'][category.value] += 1
        
        # 记录日志
        log_level = getattr(logging, level.value)
        self.logger.log(log_level, message, extra={'extra_data': extra_data})
    
    def debug(self, message: str, category: LogCategory = LogCategory.SYSTEM, **kwargs):
        """调试日志"""
        self._log_with_context(LogLevel.DEBUG, message, category, **kwargs)
    
    def info(self, message: str, category: LogCategory = LogCategory.SYSTEM, **kwargs):
        """信息日志"""
        self._log_with_context(LogLevel.INFO, message, category, **kwargs)
    
    def warning(self, message: str, category: LogCategory = LogCategory.SYSTEM, **kwargs):
        """警告日志"""
        self._log_with_context(LogLevel.WARNING, message, category, **kwargs)
    
    def error(self, message: str, category: LogCategory = LogCategory.ERROR, **kwargs):
        """错误日志"""
        self._log_with_context(LogLevel.ERROR, message, category, **kwargs)
    
    def critical(self, message: str, category: LogCategory = LogCategory.ERROR, **kwargs):
        """严重错误日志"""
        self._log_with_context(LogLevel.CRITICAL, message, category, **kwargs)
    
    def log_api_request(
        self,
        method: str,
        path: str,
        status_code: int,
        duration_ms: float,
        request_id: Optional[str] = None,
        user_id: Optional[str] = None,
        **kwargs
    ):
        """记录API请求日志"""
        message = f"{method} {path} - {status_code} ({duration_ms:.2f}ms)"
        
        self._log_with_context(
            LogLevel.INFO,
            message,
            LogCategory.API,
            request_id=request_id,
            user_id=user_id,
            status_code=status_code,
            duration_ms=duration_ms,
            metadata={
                'method': method,
                'path': path,
                **kwargs
            }
        )
    
    def log_data_operation(
        self,
        operation: str,
        stock_code: str,
        records_count: int,
        duration_ms: float,
        success: bool = True,
        **kwargs
    ):
        """记录数据操作日志"""
        status = "成功" if success else "失败"
        message = f"数据操作{status}: {operation} {stock_code} ({records_count}条记录, {duration_ms:.2f}ms)"
        
        level = LogLevel.INFO if success else LogLevel.ERROR
        
        self._log_with_context(
            level,
            message,
            LogCategory.DATA,
            duration_ms=duration_ms,
            metadata={
                'operation': operation,
                'stock_code': stock_code,
                'records_count': records_count,
                'success': success,
                **kwargs
            }
        )
    
    def log_performance_metric(
        self,
        metric_name: str,
        value: float,
        unit: str = "",
        **kwargs
    ):
        """记录性能指标日志"""
        message = f"性能指标: {metric_name} = {value}{unit}"
        
        self._log_with_context(
            LogLevel.INFO,
            message,
            LogCategory.PERFORMANCE,
            metadata={
                'metric_name': metric_name,
                'value': value,
                'unit': unit,
                **kwargs
            }
        )
    
    def log_security_event(
        self,
        event_type: str,
        severity: str,
        description: str,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        **kwargs
    ):
        """记录安全事件日志"""
        message = f"安全事件: {event_type} - {description}"
        
        level = LogLevel.WARNING if severity == 'medium' else LogLevel.ERROR
        
        self._log_with_context(
            level,
            message,
            LogCategory.SECURITY,
            user_id=user_id,
            metadata={
                'event_type': event_type,
                'severity': severity,
                'ip_address': ip_address,
                **kwargs
            }
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """获取日志统计信息"""
        uptime = datetime.now() - self.stats['start_time']
        
        return {
            'total_logs': self.stats['total_logs'],
            'logs_by_level': self.stats['logs_by_level'].copy(),
            'logs_by_category': self.stats['logs_by_category'].copy(),
            'uptime_seconds': uptime.total_seconds(),
            'logs_per_minute': self.stats['total_logs'] / max(uptime.total_seconds() / 60, 1),
            'log_directory': str(self.log_dir),
            'service_name': self.service_name
        }
    
    def search_logs(
        self,
        query: str,
        level: Optional[LogLevel] = None,
        category: Optional[LogCategory] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """搜索日志"""
        # 这是一个简化的实现，实际应用中可能需要使用专门的日志搜索引擎
        results = []
        
        try:
            log_files = list(self.log_dir.glob("*.log"))
            
            for log_file in log_files:
                if len(results) >= limit:
                    break
                
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            if len(results) >= limit:
                                break
                            
                            try:
                                log_entry = json.loads(line.strip())
                                
                                # 应用过滤条件
                                if level and log_entry.get('level') != level.value:
                                    continue
                                
                                if category and log_entry.get('category') != category.value:
                                    continue
                                
                                if start_time:
                                    log_time = datetime.fromisoformat(log_entry.get('timestamp', ''))
                                    if log_time < start_time:
                                        continue
                                
                                if end_time:
                                    log_time = datetime.fromisoformat(log_entry.get('timestamp', ''))
                                    if log_time > end_time:
                                        continue
                                
                                # 搜索查询字符串
                                if query.lower() in log_entry.get('message', '').lower():
                                    results.append(log_entry)
                            
                            except json.JSONDecodeError:
                                continue
                
                except Exception as e:
                    self.error(f"搜索日志文件失败 {log_file}: {e}")
        
        except Exception as e:
            self.error(f"日志搜索失败: {e}")
        
        return results


# 全局日志实例
_loggers: Dict[str, EnhancedLogger] = {}


def get_logger(name: str, **kwargs) -> EnhancedLogger:
    """获取日志记录器实例"""
    if name not in _loggers:
        _loggers[name] = EnhancedLogger(name, **kwargs)
    return _loggers[name]


def get_api_logger() -> EnhancedLogger:
    """获取API日志记录器"""
    return get_logger("api", log_dir="logs/api")


def get_data_logger() -> EnhancedLogger:
    """获取数据日志记录器"""
    return get_logger("data", log_dir="logs/data")


def get_monitoring_logger() -> EnhancedLogger:
    """获取监控日志记录器"""
    return get_logger("monitoring", log_dir="logs/monitoring")