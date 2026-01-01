"""
任务管理服务
实现任务创建、状态管理、进度跟踪和结果保存功能
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import logging

from app.models.database import DatabaseManager, Task, TaskResult, TaskStatus
from app.models.stock_simple import StockData


@dataclass
class TaskCreateRequest:
    """任务创建请求"""
    name: str
    description: str
    stock_codes: List[str]
    indicators: List[str]
    models: List[str]
    parameters: Dict[str, Any]
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None


@dataclass
class TaskUpdateRequest:
    """任务更新请求"""
    task_id: int
    status: Optional[TaskStatus] = None
    progress: Optional[float] = None
    error_message: Optional[str] = None


@dataclass
class TaskQuery:
    """任务查询条件"""
    status: Optional[TaskStatus] = None
    stock_code: Optional[str] = None
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
    limit: int = 100
    offset: int = 0


@dataclass
class TaskSummary:
    """任务摘要"""
    id: int
    name: str
    status: TaskStatus
    progress: float
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    stock_count: int
    result_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'id': self.id,
            'name': self.name,
            'status': self.status.value,
            'progress': self.progress,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'stock_count': self.stock_count,
            'result_count': self.result_count
        }


class TaskManager:
    """任务管理器"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)
        
        # 任务状态变更回调
        self.status_change_callbacks = []
        
        # 进度更新回调
        self.progress_callbacks = []
    
    def create_task(self, request: TaskCreateRequest) -> int:
        """
        创建新任务
        
        Args:
            request: 任务创建请求
        
        Returns:
            int: 任务ID
        """
        try:
            # 验证输入
            if not request.name.strip():
                raise ValueError("任务名称不能为空")
            
            if not request.stock_codes:
                raise ValueError("股票代码列表不能为空")
            
            if not request.indicators:
                raise ValueError("指标列表不能为空")
            
            if not request.models:
                raise ValueError("模型列表不能为空")
            
            # 创建任务对象
            task = Task(
                name=request.name.strip(),
                description=request.description.strip(),
                stock_codes=json.dumps(request.stock_codes),
                indicators=json.dumps(request.indicators),
                models=json.dumps(request.models),
                parameters=json.dumps(request.parameters),
                status=TaskStatus.PENDING,
                progress=0.0,
                created_at=datetime.now()
            )
            
            # 保存到数据库
            with self.db_manager.get_connection() as conn:
                cursor = conn.execute("""
                    INSERT INTO tasks (name, description, stock_codes, indicators, models, 
                                     parameters, status, progress, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    task.name, task.description, task.stock_codes, task.indicators,
                    task.models, task.parameters, task.status.value, task.progress, task.created_at
                ))
                task_id = cursor.lastrowid
                conn.commit()
            
            self.logger.info(f"创建任务成功: ID={task_id}, 名称={task.name}")
            
            # 触发状态变更回调
            self._trigger_status_change_callbacks(task_id, TaskStatus.PENDING)
            
            return task_id
            
        except Exception as e:
            self.logger.error(f"创建任务失败: {e}")
            raise
    
    def get_task(self, task_id: int) -> Optional[Task]:
        """
        获取任务详情
        
        Args:
            task_id: 任务ID
        
        Returns:
            Optional[Task]: 任务对象，如果不存在则返回None
        """
        try:
            row = self.db_manager.fetch_one("SELECT * FROM tasks WHERE id = ?", (task_id,))
            
            if not row:
                return None
            
            return Task(
                id=row['id'],
                name=row['name'],
                description=row['description'],
                stock_codes=row['stock_codes'],
                indicators=row['indicators'],
                models=row['models'],
                parameters=row['parameters'],
                status=TaskStatus(row['status']),
                progress=row['progress'],
                created_at=row['created_at'],
                started_at=row['started_at'],
                completed_at=row['completed_at'],
                error_message=row['error_message']
            )
            
        except Exception as e:
            self.logger.error(f"获取任务失败 {task_id}: {e}")
            return None
    
    def update_task(self, request: TaskUpdateRequest) -> bool:
        """
        更新任务状态
        
        Args:
            request: 任务更新请求
        
        Returns:
            bool: 更新是否成功
        """
        try:
            # 构建更新语句
            update_fields = []
            update_values = []
            
            if request.status is not None:
                update_fields.append("status = ?")
                update_values.append(request.status.value)
                
                # 如果状态变为运行中，设置开始时间
                if request.status == TaskStatus.RUNNING:
                    update_fields.append("started_at = ?")
                    update_values.append(datetime.now())
                
                # 如果状态变为完成或失败，设置完成时间
                elif request.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                    update_fields.append("completed_at = ?")
                    update_values.append(datetime.now())
            
            if request.progress is not None:
                update_fields.append("progress = ?")
                update_values.append(max(0.0, min(100.0, request.progress)))
            
            if request.error_message is not None:
                update_fields.append("error_message = ?")
                update_values.append(request.error_message)
            
            if not update_fields:
                return True  # 没有需要更新的字段
            
            # 执行更新
            update_values.append(request.task_id)
            sql = f"UPDATE tasks SET {', '.join(update_fields)} WHERE id = ?"
            
            with self.db_manager.get_connection() as conn:
                cursor = conn.execute(sql, update_values)
                updated_rows = cursor.rowcount
                conn.commit()
            
            if updated_rows > 0:
                self.logger.info(f"更新任务成功: ID={request.task_id}")
                
                # 触发回调
                if request.status is not None:
                    self._trigger_status_change_callbacks(request.task_id, request.status)
                
                if request.progress is not None:
                    self._trigger_progress_callbacks(request.task_id, request.progress)
                
                return True
            else:
                self.logger.warning(f"任务不存在: ID={request.task_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"更新任务失败 {request.task_id}: {e}")
            return False
    
    def query_tasks(self, query: TaskQuery) -> List[TaskSummary]:
        """
        查询任务列表
        
        Args:
            query: 查询条件
        
        Returns:
            List[TaskSummary]: 任务摘要列表
        """
        try:
            # 构建查询条件
            where_conditions = []
            query_params = []
            
            if query.status is not None:
                where_conditions.append("t.status = ?")
                query_params.append(query.status.value)
            
            if query.stock_code is not None:
                where_conditions.append("t.stock_codes LIKE ?")
                query_params.append(f'%"{query.stock_code}"%')
            
            if query.created_after is not None:
                where_conditions.append("t.created_at >= ?")
                query_params.append(query.created_after)
            
            if query.created_before is not None:
                where_conditions.append("t.created_at <= ?")
                query_params.append(query.created_before)
            
            # 构建完整查询
            where_clause = ""
            if where_conditions:
                where_clause = "WHERE " + " AND ".join(where_conditions)
            
            sql = f"""
                SELECT 
                    t.*,
                    COUNT(tr.id) as result_count
                FROM tasks t
                LEFT JOIN task_results tr ON t.id = tr.task_id
                {where_clause}
                GROUP BY t.id
                ORDER BY t.created_at DESC
                LIMIT ? OFFSET ?
            """
            
            query_params.extend([query.limit, query.offset])
            
            rows = self.db_manager.fetch_all(sql, tuple(query_params))
            
            # 转换为TaskSummary对象
            summaries = []
            for row in rows:
                try:
                    stock_codes = json.loads(row['stock_codes'])
                    stock_count = len(stock_codes)
                except (json.JSONDecodeError, TypeError):
                    stock_count = 0
                
                summary = TaskSummary(
                    id=row['id'],
                    name=row['name'],
                    status=TaskStatus(row['status']),
                    progress=row['progress'],
                    created_at=row['created_at'] if isinstance(row['created_at'], datetime) else datetime.fromisoformat(str(row['created_at'])),
                    started_at=row['started_at'] if row['started_at'] is None or isinstance(row['started_at'], datetime) else datetime.fromisoformat(str(row['started_at'])),
                    completed_at=row['completed_at'] if row['completed_at'] is None or isinstance(row['completed_at'], datetime) else datetime.fromisoformat(str(row['completed_at'])),
                    stock_count=stock_count,
                    result_count=row['result_count']
                )
                summaries.append(summary)
            
            return summaries
            
        except Exception as e:
            self.logger.error(f"查询任务失败: {e}")
            return []
    
    def delete_task(self, task_id: int) -> bool:
        """
        删除任务及其相关结果
        
        Args:
            task_id: 任务ID
        
        Returns:
            bool: 删除是否成功
        """
        try:
            with self.db_manager.get_connection() as conn:
                # 先删除任务结果
                conn.execute("DELETE FROM task_results WHERE task_id = ?", (task_id,))
                
                # 再删除任务
                cursor = conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
                deleted_rows = cursor.rowcount
                
                conn.commit()
            
            if deleted_rows > 0:
                self.logger.info(f"删除任务成功: ID={task_id}")
                return True
            else:
                self.logger.warning(f"任务不存在: ID={task_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"删除任务失败 {task_id}: {e}")
            return False
    
    def save_task_result(self, task_id: int, result: TaskResult) -> bool:
        """
        保存任务结果
        
        Args:
            task_id: 任务ID
            result: 任务结果
        
        Returns:
            bool: 保存是否成功
        """
        try:
            # 设置任务ID
            result.task_id = task_id
            result.created_at = datetime.now()
            
            # 保存到数据库
            with self.db_manager.get_connection() as conn:
                cursor = conn.execute("""
                    INSERT INTO task_results (task_id, stock_code, prediction_date, prediction_value,
                                            confidence, model_name, indicators_used, backtest_metrics, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    result.task_id, result.stock_code, result.prediction_date, result.prediction_value,
                    result.confidence, result.model_name, result.indicators_used, 
                    result.backtest_metrics, result.created_at
                ))
                result_id = cursor.lastrowid
                conn.commit()
            
            self.logger.info(f"保存任务结果成功: 任务ID={task_id}, 结果ID={result_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"保存任务结果失败 任务ID={task_id}: {e}")
            return False
    
    def get_task_results(self, task_id: int, stock_code: Optional[str] = None) -> List[TaskResult]:
        """
        获取任务结果
        
        Args:
            task_id: 任务ID
            stock_code: 可选的股票代码过滤
        
        Returns:
            List[TaskResult]: 任务结果列表
        """
        try:
            if stock_code:
                sql = "SELECT * FROM task_results WHERE task_id = ? AND stock_code = ? ORDER BY created_at DESC"
                params = (task_id, stock_code)
            else:
                sql = "SELECT * FROM task_results WHERE task_id = ? ORDER BY created_at DESC"
                params = (task_id,)
            
            rows = self.db_manager.fetch_all(sql, params)
            
            results = []
            for row in rows:
                result = TaskResult(
                    id=row['id'],
                    task_id=row['task_id'],
                    stock_code=row['stock_code'],
                    prediction_date=row['prediction_date'],
                    prediction_value=row['prediction_value'],
                    confidence=row['confidence'],
                    model_name=row['model_name'],
                    indicators_used=row['indicators_used'],
                    backtest_metrics=row['backtest_metrics'],
                    created_at=row['created_at']
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"获取任务结果失败 任务ID={task_id}: {e}")
            return []
    
    def get_task_statistics(self) -> Dict[str, Any]:
        """
        获取任务统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        try:
            stats = {
                'total_tasks': 0,
                'pending_tasks': 0,
                'running_tasks': 0,
                'completed_tasks': 0,
                'failed_tasks': 0,
                'total_results': 0,
                'recent_tasks': []
            }
            
            # 任务状态统计
            status_rows = self.db_manager.fetch_all("""
                SELECT status, COUNT(*) as count FROM tasks GROUP BY status
            """)
            
            for row in status_rows:
                status = row['status']
                count = row['count']
                stats['total_tasks'] += count
                
                if status == TaskStatus.PENDING.value:
                    stats['pending_tasks'] = count
                elif status == TaskStatus.RUNNING.value:
                    stats['running_tasks'] = count
                elif status == TaskStatus.COMPLETED.value:
                    stats['completed_tasks'] = count
                elif status == TaskStatus.FAILED.value:
                    stats['failed_tasks'] = count
            
            # 结果总数
            result_row = self.db_manager.fetch_one("SELECT COUNT(*) as count FROM task_results")
            if result_row:
                stats['total_results'] = result_row['count']
            
            # 最近任务
            recent_query = TaskQuery(limit=5)
            recent_tasks = self.query_tasks(recent_query)
            stats['recent_tasks'] = []
            for task in recent_tasks:
                try:
                    stats['recent_tasks'].append(task.to_dict())
                except Exception as e:
                    self.logger.warning(f"转换任务摘要失败: {e}")
                    # 创建简化版本
                    stats['recent_tasks'].append({
                        'id': task.id,
                        'name': task.name,
                        'status': task.status.value,
                        'progress': task.progress,
                        'created_at': task.created_at.isoformat() if hasattr(task.created_at, 'isoformat') else str(task.created_at),
                        'stock_count': task.stock_count,
                        'result_count': task.result_count
                    })
            
            return stats
            
        except Exception as e:
            self.logger.error(f"获取任务统计失败: {e}")
            return {}
    
    def add_status_change_callback(self, callback):
        """添加状态变更回调"""
        self.status_change_callbacks.append(callback)
    
    def add_progress_callback(self, callback):
        """添加进度更新回调"""
        self.progress_callbacks.append(callback)
    
    def _trigger_status_change_callbacks(self, task_id: int, status: TaskStatus):
        """触发状态变更回调"""
        for callback in self.status_change_callbacks:
            try:
                callback(task_id, status)
            except Exception as e:
                self.logger.error(f"状态变更回调执行失败: {e}")
    
    def _trigger_progress_callbacks(self, task_id: int, progress: float):
        """触发进度更新回调"""
        for callback in self.progress_callbacks:
            try:
                callback(task_id, progress)
            except Exception as e:
                self.logger.error(f"进度更新回调执行失败: {e}")
    
    async def run_task_async(self, task_id: int, execution_func):
        """
        异步执行任务
        
        Args:
            task_id: 任务ID
            execution_func: 执行函数，应该接受task_id和progress_callback参数
        """
        try:
            # 更新任务状态为运行中
            self.update_task(TaskUpdateRequest(
                task_id=task_id,
                status=TaskStatus.RUNNING,
                progress=0.0
            ))
            
            # 定义进度回调
            def progress_callback(progress: float):
                self.update_task(TaskUpdateRequest(
                    task_id=task_id,
                    progress=progress
                ))
            
            # 执行任务
            await execution_func(task_id, progress_callback)
            
            # 更新任务状态为完成
            self.update_task(TaskUpdateRequest(
                task_id=task_id,
                status=TaskStatus.COMPLETED,
                progress=100.0
            ))
            
        except Exception as e:
            # 更新任务状态为失败
            self.update_task(TaskUpdateRequest(
                task_id=task_id,
                status=TaskStatus.FAILED,
                error_message=str(e)
            ))
            
            self.logger.error(f"任务执行失败 {task_id}: {e}")
            raise