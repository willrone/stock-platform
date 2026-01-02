"""
数据生命周期管理服务
实现临时文件和日志的定期清理，以及历史任务记录的保留策略
"""

import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path
from dataclasses import dataclass
from loguru import logger

from app.models.database import DatabaseManager, TaskStatus
from .parquet_manager import ParquetManager


@dataclass
class CleanupResult:
    """清理结果"""
    deleted_files: int
    deleted_size: int  # 字节
    deleted_tasks: int
    deleted_logs: int
    errors: List[str]
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'deleted_files': self.deleted_files,
            'deleted_size': self.deleted_size,
            'deleted_tasks': self.deleted_tasks,
            'deleted_logs': self.deleted_logs,
            'errors': self.errors,
            'deleted_size_mb': round(self.deleted_size / (1024 * 1024), 2)
        }


@dataclass
class RetentionPolicy:
    """数据保留策略"""
    # Parquet数据文件保留天数
    parquet_retention_days: int = 365
    
    # 任务记录保留天数
    task_retention_days: int = 180
    
    # 日志文件保留天数
    log_retention_days: int = 30
    
    # 临时文件保留天数
    temp_file_retention_days: int = 7
    
    # 失败任务保留天数（比成功任务保留时间短）
    failed_task_retention_days: int = 90
    
    # 模型文件保留天数
    model_retention_days: int = 90


class DataLifecycleManager:
    """数据生命周期管理器"""
    
    def __init__(
        self, 
        db_manager: DatabaseManager,
        parquet_manager: ParquetManager,
        retention_policy: Optional[RetentionPolicy] = None,
        base_path: str = "data"
    ):
        self.db_manager = db_manager
        self.parquet_manager = parquet_manager
        self.retention_policy = retention_policy or RetentionPolicy()
        self.base_path = Path(base_path)
        from loguru import logger
        self.logger = logger
        
        # 确保基础目录存在
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # 定义各种数据目录
        self.temp_dir = self.base_path / "temp"
        self.log_dir = self.base_path / "logs"
        self.model_dir = self.base_path / "models"
        
        # 创建目录
        for directory in [self.temp_dir, self.log_dir, self.model_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def run_cleanup(self, dry_run: bool = False) -> CleanupResult:
        """
        运行完整的数据清理
        
        Args:
            dry_run: 是否为试运行（不实际删除文件）
        
        Returns:
            CleanupResult: 清理结果
        """
        self.logger.info(f"开始数据清理 (dry_run={dry_run})")
        
        result = CleanupResult(
            deleted_files=0,
            deleted_size=0,
            deleted_tasks=0,
            deleted_logs=0,
            errors=[]
        )
        
        try:
            # 1. 清理Parquet数据文件
            parquet_result = self._cleanup_parquet_files(dry_run)
            result.deleted_files += parquet_result['deleted_files']
            result.deleted_size += parquet_result['deleted_size']
            result.errors.extend(parquet_result['errors'])
            
            # 2. 清理数据库中的旧任务
            task_result = self._cleanup_old_tasks(dry_run)
            result.deleted_tasks += task_result['deleted_tasks']
            result.errors.extend(task_result['errors'])
            
            # 3. 清理日志文件
            log_result = self._cleanup_log_files(dry_run)
            result.deleted_logs += log_result['deleted_files']
            result.deleted_size += log_result['deleted_size']
            result.errors.extend(log_result['errors'])
            
            # 4. 清理临时文件
            temp_result = self._cleanup_temp_files(dry_run)
            result.deleted_files += temp_result['deleted_files']
            result.deleted_size += temp_result['deleted_size']
            result.errors.extend(temp_result['errors'])
            
            # 5. 清理旧模型文件
            model_result = self._cleanup_model_files(dry_run)
            result.deleted_files += model_result['deleted_files']
            result.deleted_size += model_result['deleted_size']
            result.errors.extend(model_result['errors'])
            
            self.logger.info(f"数据清理完成: {result.to_dict()}")
            
        except Exception as e:
            error_msg = f"数据清理过程中发生错误: {e}"
            self.logger.error(error_msg)
            result.errors.append(error_msg)
        
        return result
    
    def _cleanup_parquet_files(self, dry_run: bool) -> Dict:
        """清理Parquet数据文件"""
        self.logger.info("开始清理Parquet文件")
        
        result = {
            'deleted_files': 0,
            'deleted_size': 0,
            'errors': []
        }
        
        try:
            if dry_run:
                # 试运行：只统计不删除
                cutoff_date = datetime.now() - timedelta(days=self.retention_policy.parquet_retention_days)
                
                for file_path in self.parquet_manager.base_path.rglob("*.parquet"):
                    try:
                        stat = file_path.stat()
                        if datetime.fromtimestamp(stat.st_mtime) < cutoff_date:
                            result['deleted_files'] += 1
                            result['deleted_size'] += stat.st_size
                    except Exception as e:
                        result['errors'].append(f"检查文件失败 {file_path}: {e}")
            else:
                # 实际删除
                deleted_count = self.parquet_manager.cleanup_old_files(
                    days_to_keep=self.retention_policy.parquet_retention_days
                )
                result['deleted_files'] = deleted_count
                
        except Exception as e:
            result['errors'].append(f"清理Parquet文件失败: {e}")
        
        return result
    
    def _cleanup_old_tasks(self, dry_run: bool) -> Dict:
        """清理数据库中的旧任务"""
        self.logger.info("开始清理旧任务记录")
        
        result = {
            'deleted_tasks': 0,
            'errors': []
        }
        
        try:
            # 计算截止日期
            task_cutoff = datetime.now() - timedelta(days=self.retention_policy.task_retention_days)
            failed_task_cutoff = datetime.now() - timedelta(days=self.retention_policy.failed_task_retention_days)
            
            if dry_run:
                # 试运行：只统计
                # 统计普通任务
                normal_tasks = self.db_manager.fetch_all("""
                    SELECT COUNT(*) as count FROM tasks 
                    WHERE created_at < ? AND status NOT IN ('failed', 'cancelled')
                """, (task_cutoff,))
                
                # 统计失败任务
                failed_tasks = self.db_manager.fetch_all("""
                    SELECT COUNT(*) as count FROM tasks 
                    WHERE created_at < ? AND status IN ('failed', 'cancelled')
                """, (failed_task_cutoff,))
                
                result['deleted_tasks'] = (normal_tasks[0]['count'] if normal_tasks else 0) + \
                                        (failed_tasks[0]['count'] if failed_tasks else 0)
            else:
                # 实际删除
                with self.db_manager.get_connection() as conn:
                    # 删除旧的普通任务及其结果
                    cursor1 = conn.execute("""
                        DELETE FROM task_results 
                        WHERE task_id IN (
                            SELECT id FROM tasks 
                            WHERE created_at < ? AND status NOT IN ('failed', 'cancelled')
                        )
                    """, (task_cutoff,))
                    
                    cursor2 = conn.execute("""
                        DELETE FROM tasks 
                        WHERE created_at < ? AND status NOT IN ('failed', 'cancelled')
                    """, (task_cutoff,))
                    
                    # 删除旧的失败任务及其结果
                    cursor3 = conn.execute("""
                        DELETE FROM task_results 
                        WHERE task_id IN (
                            SELECT id FROM tasks 
                            WHERE created_at < ? AND status IN ('failed', 'cancelled')
                        )
                    """, (failed_task_cutoff,))
                    
                    cursor4 = conn.execute("""
                        DELETE FROM tasks 
                        WHERE created_at < ? AND status IN ('failed', 'cancelled')
                    """, (failed_task_cutoff,))
                    
                    result['deleted_tasks'] = cursor2.rowcount + cursor4.rowcount
                    
                    conn.commit()
                    
        except Exception as e:
            result['errors'].append(f"清理旧任务失败: {e}")
        
        return result
    
    def _cleanup_log_files(self, dry_run: bool) -> Dict:
        """清理日志文件"""
        self.logger.info("开始清理日志文件")
        
        result = {
            'deleted_files': 0,
            'deleted_size': 0,
            'errors': []
        }
        
        try:
            cutoff_date = datetime.now() - timedelta(days=self.retention_policy.log_retention_days)
            
            # 查找日志文件
            log_patterns = ['*.log', '*.log.*', '*.out', '*.err']
            
            for pattern in log_patterns:
                for file_path in self.log_dir.glob(pattern):
                    try:
                        stat = file_path.stat()
                        if datetime.fromtimestamp(stat.st_mtime) < cutoff_date:
                            if dry_run:
                                result['deleted_files'] += 1
                                result['deleted_size'] += stat.st_size
                            else:
                                result['deleted_files'] += 1
                                result['deleted_size'] += stat.st_size
                                file_path.unlink()
                                self.logger.info(f"删除日志文件: {file_path}")
                    except Exception as e:
                        result['errors'].append(f"处理日志文件失败 {file_path}: {e}")
                        
        except Exception as e:
            result['errors'].append(f"清理日志文件失败: {e}")
        
        return result
    
    def _cleanup_temp_files(self, dry_run: bool) -> Dict:
        """清理临时文件"""
        self.logger.info("开始清理临时文件")
        
        result = {
            'deleted_files': 0,
            'deleted_size': 0,
            'errors': []
        }
        
        try:
            cutoff_date = datetime.now() - timedelta(days=self.retention_policy.temp_file_retention_days)
            
            # 清理临时目录中的文件
            for file_path in self.temp_dir.rglob("*"):
                if file_path.is_file():
                    try:
                        stat = file_path.stat()
                        if datetime.fromtimestamp(stat.st_mtime) < cutoff_date:
                            if dry_run:
                                result['deleted_files'] += 1
                                result['deleted_size'] += stat.st_size
                            else:
                                result['deleted_files'] += 1
                                result['deleted_size'] += stat.st_size
                                file_path.unlink()
                                self.logger.info(f"删除临时文件: {file_path}")
                    except Exception as e:
                        result['errors'].append(f"处理临时文件失败 {file_path}: {e}")
            
            # 清理空目录
            if not dry_run:
                self._cleanup_empty_directories(self.temp_dir)
                        
        except Exception as e:
            result['errors'].append(f"清理临时文件失败: {e}")
        
        return result
    
    def _cleanup_model_files(self, dry_run: bool) -> Dict:
        """清理旧模型文件"""
        self.logger.info("开始清理模型文件")
        
        result = {
            'deleted_files': 0,
            'deleted_size': 0,
            'errors': []
        }
        
        try:
            cutoff_date = datetime.now() - timedelta(days=self.retention_policy.model_retention_days)
            
            # 获取数据库中活跃的模型文件路径
            active_models = self.db_manager.fetch_all("""
                SELECT file_path FROM model_metadata WHERE is_active = 1
            """)
            active_paths = {row['file_path'] for row in active_models}
            
            # 查找模型文件
            model_patterns = ['*.pkl', '*.joblib', '*.h5', '*.pt', '*.pth', '*.onnx']
            
            for pattern in model_patterns:
                for file_path in self.model_dir.glob(pattern):
                    try:
                        # 跳过活跃的模型文件
                        if str(file_path) in active_paths:
                            continue
                            
                        stat = file_path.stat()
                        if datetime.fromtimestamp(stat.st_mtime) < cutoff_date:
                            if dry_run:
                                result['deleted_files'] += 1
                                result['deleted_size'] += stat.st_size
                            else:
                                result['deleted_files'] += 1
                                result['deleted_size'] += stat.st_size
                                file_path.unlink()
                                self.logger.info(f"删除模型文件: {file_path}")
                    except Exception as e:
                        result['errors'].append(f"处理模型文件失败 {file_path}: {e}")
                        
        except Exception as e:
            result['errors'].append(f"清理模型文件失败: {e}")
        
        return result
    
    def _cleanup_empty_directories(self, root_path: Path):
        """清理空目录"""
        try:
            for dirpath, dirnames, filenames in os.walk(root_path, topdown=False):
                for dirname in dirnames:
                    dir_path = Path(dirpath) / dirname
                    try:
                        if not any(dir_path.iterdir()):
                            dir_path.rmdir()
                            self.logger.info(f"删除空目录: {dir_path}")
                    except OSError:
                        pass  # 目录不为空或其他错误
        except Exception as e:
            self.logger.warning(f"清理空目录失败: {e}")
    
    def get_storage_usage(self) -> Dict:
        """获取存储使用情况"""
        usage = {
            'parquet': {'size': 0, 'files': 0},
            'database': {'size': 0, 'files': 1},
            'logs': {'size': 0, 'files': 0},
            'temp': {'size': 0, 'files': 0},
            'models': {'size': 0, 'files': 0},
            'total': {'size': 0, 'files': 0}
        }
        
        try:
            # Parquet文件统计
            parquet_stats = self.parquet_manager.get_storage_stats()
            usage['parquet']['size'] = parquet_stats.get('total_size', 0)
            usage['parquet']['files'] = parquet_stats.get('total_files', 0)
            
            # 数据库文件统计
            if self.db_manager.db_path and os.path.exists(self.db_manager.db_path):
                usage['database']['size'] = os.path.getsize(self.db_manager.db_path)
            
            # 其他目录统计
            for category, directory in [
                ('logs', self.log_dir),
                ('temp', self.temp_dir),
                ('models', self.model_dir)
            ]:
                if directory.exists():
                    for file_path in directory.rglob("*"):
                        if file_path.is_file():
                            try:
                                usage[category]['size'] += file_path.stat().st_size
                                usage[category]['files'] += 1
                            except OSError:
                                pass
            
            # 计算总计
            usage['total']['size'] = sum(cat['size'] for cat in usage.values() if isinstance(cat, dict) and 'size' in cat)
            usage['total']['files'] = sum(cat['files'] for cat in usage.values() if isinstance(cat, dict) and 'files' in cat)
            
            # 转换为MB
            for category in usage:
                if isinstance(usage[category], dict) and 'size' in usage[category]:
                    usage[category]['size_mb'] = round(usage[category]['size'] / (1024 * 1024), 2)
            
        except Exception as e:
            self.logger.error(f"获取存储使用情况失败: {e}")
        
        return usage
    
    def schedule_cleanup(self, interval_hours: int = 24) -> bool:
        """
        安排定期清理任务
        
        Args:
            interval_hours: 清理间隔（小时）
        
        Returns:
            bool: 是否成功安排
        """
        try:
            # 这里可以集成任务调度器（如APScheduler）
            # 目前只是记录配置
            config_key = "cleanup_schedule"
            config_value = {
                'enabled': True,
                'interval_hours': interval_hours,
                'last_run': None,
                'next_run': (datetime.now() + timedelta(hours=interval_hours)).isoformat()
            }
            
            # 保存到系统配置
            with self.db_manager.get_connection() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO system_config (key, value, description, updated_at)
                    VALUES (?, ?, ?, ?)
                """, (
                    config_key,
                    str(config_value),
                    "数据清理调度配置",
                    datetime.now()
                ))
                conn.commit()
            
            self.logger.info(f"已安排定期清理任务，间隔 {interval_hours} 小时")
            return True
            
        except Exception as e:
            self.logger.error(f"安排定期清理任务失败: {e}")
            return False