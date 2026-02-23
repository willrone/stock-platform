#!/usr/bin/env python3
"""
临时脚本：恢复卡住的优化任务
将 pending 状态的优化任务重新提交到进程池执行
"""

import sys
sys.path.insert(0, '/Users/ronghui/Projects/willrone/backend')

from app.core.database import SessionLocal
from app.repositories.task_repository import TaskRepository
from app.services.tasks.process_executor import get_process_executor, start_process_executor
from app.api.v1.optimization import execute_optimization_task_simple
from loguru import logger

def recover_pending_optimization_tasks():
    """恢复 pending 状态的优化任务"""
    
    # 先启动进程池
    logger.info("启动进程池...")
    start_process_executor()
    
    session = SessionLocal()
    try:
        task_repository = TaskRepository(session)
        
        # 查询所有 pending 状态的优化任务（数据库中是 'pending' 字符串）
        from app.models.task_models import Task
        tasks = session.query(Task).filter(
            Task.status == 'pending',
            Task.task_type == 'hyperparameter_optimization'
        ).all()
        
        if not tasks:
            logger.info("没有需要恢复的任务")
            return
        
        logger.info(f"找到 {len(tasks)} 个需要恢复的任务")
        
        process_executor = get_process_executor()
        recovered = 0
        
        for task in tasks:
            try:
                logger.info(f"恢复任务: {task.task_id} - {task.task_name}")
                
                # 提交到进程池
                future = process_executor.submit(
                    execute_optimization_task_simple, 
                    task.task_id
                )
                
                logger.info(f"✅ 任务已提交: {task.task_id}")
                recovered += 1
                
            except Exception as e:
                logger.error(f"❌ 提交任务失���: {task.task_id}, 错误: {e}")
        
        logger.info(f"\n完成！成功恢复 {recovered}/{len(tasks)} 个任务")
        
    except Exception as e:
        logger.error(f"恢复任务失败: {e}", exc_info=True)
    finally:
        session.close()

if __name__ == "__main__":
    recover_pending_optimization_tasks()
