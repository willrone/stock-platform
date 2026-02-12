#!/usr/bin/env python3
"""
测试任务恢复服务
"""

import sys
sys.path.insert(0, '/Users/ronghui/Projects/willrone/backend')

from app.services.tasks.task_recovery_service import task_recovery_service
from app.services.tasks.process_executor import start_process_executor
from loguru import logger

def test_recovery_service():
    """测试恢复服务"""
    
    # 启动进程池
    logger.info("启动进程池...")
    start_process_executor()
    
    # 测试恢复逻辑
    logger.info("测试任务恢复服务...")
    result = task_recovery_service.recover_interrupted_tasks()
    
    logger.info(f"恢复结果: {result}")
    
    if result["recovered"] > 0:
        logger.success(f"✅ 成功恢复 {result['recovered']} 个任务")
        for task in result["tasks"]:
            if task.get("status") == "recovered":
                logger.info(f"  - {task['task_name']}")
    else:
        logger.info("没有需要恢复的任务")
    
    if result["failed"] > 0:
        logger.warning(f"⚠️ {result['failed']} 个任务恢复失败")

if __name__ == "__main__":
    test_recovery_service()
