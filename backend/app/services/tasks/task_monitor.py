"""
任务监控服务
用于监控和处理卡住的任务
"""

import logging
import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Any
from pathlib import Path

from app.core.database import SessionLocal
from app.repositories.task_repository import TaskRepository
from app.models.task_models import TaskStatus

logger = logging.getLogger(__name__)


class TaskMonitor:
    """任务监控器"""
    
    def __init__(self, db_path: str = "data/app.db"):
        self.db_path = db_path
        
    def get_stuck_tasks(self, timeout_minutes: int = 30) -> List[Dict[str, Any]]:
        """
        获取卡住的任务
        
        Args:
            timeout_minutes: 任务超时时间（分钟）
        
        Returns:
            卡住的任务列表
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 计算超时时间点
            timeout_time = datetime.now() - timedelta(minutes=timeout_minutes)
            timeout_str = timeout_time.isoformat()
            
            # 查询运行中但超时的任务
            cursor.execute('''
                SELECT task_id, task_name, task_type, status, created_at, started_at, progress
                FROM tasks 
                WHERE status IN ('running', 'queued') 
                AND (started_at IS NULL OR started_at < ?)
                ORDER BY created_at DESC
            ''', (timeout_str,))
            
            tasks = []
            for row in cursor.fetchall():
                task_id, task_name, task_type, status, created_at, started_at, progress = row
                tasks.append({
                    'task_id': task_id,
                    'task_name': task_name,
                    'task_type': task_type,
                    'status': status,
                    'created_at': created_at,
                    'started_at': started_at,
                    'progress': progress
                })
            
            return tasks
            
        except Exception as e:
            logger.error(f"获取卡住任务失败: {e}")
            return []
        finally:
            conn.close()
    
    def force_complete_task(self, task_id: str, status: str = 'cancelled') -> bool:
        """
        强制完成任务
        
        Args:
            task_id: 任务ID
            status: 新状态 (cancelled, failed, completed)
        
        Returns:
            是否成功
        """
        try:
            session = SessionLocal()
            task_repository = TaskRepository(session)
            
            # 更新任务状态
            task = task_repository.update_task_status(
                task_id=task_id,
                status=getattr(TaskStatus, status.upper()),
                progress=100.0,
                completed_at=datetime.now()
            )
            
            if task:
                logger.info(f"强制完成任务: {task_id} -> {status}")
                return True
            else:
                logger.warning(f"任务不存在: {task_id}")
                return False
                
        except Exception as e:
            logger.error(f"强制完成任务失败: {task_id}, 错误: {e}")
            return False
        finally:
            session.close()
    
    def cleanup_stuck_tasks(self, timeout_minutes: int = 30, auto_fix: bool = False) -> Dict[str, Any]:
        """
        清理卡住的任务
        
        Args:
            timeout_minutes: 任务超时时间（分钟）
            auto_fix: 是否自动修复
        
        Returns:
            清理结果
        """
        stuck_tasks = self.get_stuck_tasks(timeout_minutes)
        
        result = {
            'total_stuck': len(stuck_tasks),
            'fixed_tasks': [],
            'failed_tasks': []
        }
        
        if not stuck_tasks:
            logger.info("没有发现卡住的任务")
            return result
        
        logger.info(f"发现 {len(stuck_tasks)} 个卡住的任务")
        
        for task in stuck_tasks:
            task_id = task['task_id']
            task_name = task['task_name']
            
            if auto_fix:
                # 自动修复：将任务标记为已取消
                if self.force_complete_task(task_id, 'cancelled'):
                    result['fixed_tasks'].append({
                        'task_id': task_id,
                        'task_name': task_name,
                        'action': 'cancelled'
                    })
                    logger.info(f"自动修复任务: {task_id} ({task_name})")
                else:
                    result['failed_tasks'].append({
                        'task_id': task_id,
                        'task_name': task_name,
                        'error': '修复失败'
                    })
            else:
                logger.info(f"发现卡住任务: {task_id} ({task_name}) - {task['status']} - {task['progress']}%")
        
        return result
    
    def get_task_statistics(self) -> Dict[str, Any]:
        """获取任务统计信息"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 统计各状态任务数量
            cursor.execute('''
                SELECT status, COUNT(*) as count
                FROM tasks 
                GROUP BY status
            ''')
            
            status_counts = {}
            for status, count in cursor.fetchall():
                status_counts[status] = count
            
            # 统计最近24小时的任务
            yesterday = (datetime.now() - timedelta(days=1)).isoformat()
            cursor.execute('''
                SELECT COUNT(*) as count
                FROM tasks 
                WHERE created_at > ?
            ''', (yesterday,))
            
            recent_count = cursor.fetchone()[0]
            
            return {
                'status_counts': status_counts,
                'recent_24h': recent_count,
                'total_tasks': sum(status_counts.values())
            }
            
        except Exception as e:
            logger.error(f"获取任务统计失败: {e}")
            return {}
        finally:
            conn.close()


# 全局任务监控器实例
task_monitor = TaskMonitor()