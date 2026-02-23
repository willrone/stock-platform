#!/usr/bin/env python3
"""独立执行协整优化任务，绕过进程池"""
import sys
import os

# 确保在backend目录下运行
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")

from app.api.v1.optimization import execute_optimization_task_simple

task_id = "9515a132-faba-489e-9a39-7ecae607d762"
print(f"开始执行协整优化任务: {task_id}")
execute_optimization_task_simple(task_id)
print("任务执行完成")
