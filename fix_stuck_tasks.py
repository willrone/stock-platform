#!/usr/bin/env python3
"""
修复卡住的回测任务脚本
解决任务一直在运行中，无法停止和删除的问题
"""

import sqlite3
import requests
import json
from datetime import datetime
import sys
import os

# 添加backend路径到sys.path
sys.path.append('backend')

def check_backend_status():
    """检查后端服务状态"""
    try:
        response = requests.get('http://localhost:8000/api/v1/health', timeout=5)
        return response.status_code == 200
    except:
        return False

def stop_task_via_api(task_id):
    """通过API停止任务"""
    try:
        response = requests.post(f'http://localhost:8000/api/v1/tasks/{task_id}/stop', timeout=10)
        if response.status_code == 200:
            print(f"✓ 成功通过API停止任务: {task_id}")
            return True
        else:
            print(f"✗ API停止任务失败: {task_id}, 状态码: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ API停止任务异常: {task_id}, 错误: {e}")
        return False

def force_update_task_status(task_id, new_status='cancelled'):
    """强制更新数据库中的任务状态"""
    try:
        conn = sqlite3.connect('backend/data/app.db')
        cursor = conn.cursor()
        
        # 更新任务状态
        cursor.execute('''
            UPDATE tasks 
            SET status = ?, completed_at = ?, progress = 100.0
            WHERE task_id = ?
        ''', (new_status, datetime.now().isoformat(), task_id))
        
        if cursor.rowcount > 0:
            conn.commit()
            print(f"✓ 强制更新任务状态成功: {task_id} -> {new_status}")
            return True
        else:
            print(f"✗ 任务不存在: {task_id}")
            return False
            
    except Exception as e:
        print(f"✗ 强制更新任务状态失败: {task_id}, 错误: {e}")
        return False
    finally:
        conn.close()

def delete_task_via_api(task_id):
    """通过API删除任务"""
    try:
        response = requests.delete(f'http://localhost:8000/api/v1/tasks/{task_id}', timeout=10)
        if response.status_code == 200:
            print(f"✓ 成功通过API删除任务: {task_id}")
            return True
        else:
            print(f"✗ API删除任务失败: {task_id}, 状态码: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ API删除任务异常: {task_id}, 错误: {e}")
        return False

def force_delete_task(task_id):
    """强制从数据库删除任务"""
    try:
        conn = sqlite3.connect('backend/data/app.db')
        cursor = conn.cursor()
        
        # 删除任务
        cursor.execute('DELETE FROM tasks WHERE task_id = ?', (task_id,))
        
        if cursor.rowcount > 0:
            conn.commit()
            print(f"✓ 强制删除任务成功: {task_id}")
            return True
        else:
            print(f"✗ 任务不存在: {task_id}")
            return False
            
    except Exception as e:
        print(f"✗ 强制删除任务失败: {task_id}, 错误: {e}")
        return False
    finally:
        conn.close()

def get_stuck_tasks():
    """获取卡住的任务"""
    try:
        conn = sqlite3.connect('backend/data/app.db')
        cursor = conn.cursor()
        
        # 查询运行中的任务
        cursor.execute('''
            SELECT task_id, task_name, task_type, status, created_at, started_at, progress
            FROM tasks 
            WHERE status = 'running'
            ORDER BY created_at DESC
        ''')
        
        tasks = cursor.fetchall()
        return tasks
        
    except Exception as e:
        print(f"✗ 获取任务列表失败: {e}")
        return []
    finally:
        conn.close()

def main():
    print("=== 修复卡住的回测任务 ===")
    print()
    
    # 检查后端服务状态
    backend_running = check_backend_status()
    print(f"后端服务状态: {'运行中' if backend_running else '未运行'}")
    print()
    
    # 获取卡住的任务
    stuck_tasks = get_stuck_tasks()
    
    if not stuck_tasks:
        print("✓ 没有发现卡住的任务")
        return
    
    print(f"发现 {len(stuck_tasks)} 个运行中的任务:")
    for task in stuck_tasks:
        task_id, task_name, task_type, status, created_at, started_at, progress = task
        print(f"  - {task_id[:8]}... ({task_name}) - {task_type} - {progress}%")
    print()
    
    # 处理每个卡住的任务
    for task in stuck_tasks:
        task_id, task_name, task_type, status, created_at, started_at, progress = task
        print(f"处理任务: {task_id} ({task_name})")
        
        success = False
        
        # 方法1: 如果后端运行中，尝试通过API停止
        if backend_running:
            if stop_task_via_api(task_id):
                success = True
        
        # 方法2: 如果API停止失败，强制更新状态
        if not success:
            print("  尝试强制更新任务状态...")
            if force_update_task_status(task_id, 'cancelled'):
                success = True
        
        # 方法3: 尝试删除任务
        if success:
            print("  尝试删除任务...")
            if backend_running:
                delete_task_via_api(task_id)
            else:
                force_delete_task(task_id)
        
        print()
    
    print("=== 修复完成 ===")
    
    # 再次检查
    remaining_tasks = get_stuck_tasks()
    if remaining_tasks:
        print(f"⚠️  仍有 {len(remaining_tasks)} 个任务未处理:")
        for task in remaining_tasks:
            task_id, task_name, task_type, status, created_at, started_at, progress = task
            print(f"  - {task_id[:8]}... ({task_name}) - {task_type} - {progress}%")
    else:
        print("✓ 所有卡住的任务已处理完成")

if __name__ == "__main__":
    main()