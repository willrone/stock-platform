#!/usr/bin/env python3
"""
任务管理工具
用于管理和监控系统中的任务
"""

import argparse
import sys
import os
import requests
import json
from datetime import datetime

# 添加backend路径
sys.path.append('backend')

def check_backend():
    """检查后端服务状态"""
    try:
        response = requests.get('http://localhost:8000/api/v1/health', timeout=5)
        return response.status_code == 200
    except:
        return False

def list_tasks():
    """列出所有任务"""
    try:
        response = requests.get('http://localhost:8000/api/v1/tasks', timeout=10)
        if response.status_code == 200:
            data = response.json()
            tasks = data.get('data', {}).get('tasks', [])
            
            print("=== 任务列表 ===")
            if not tasks:
                print("没有任务")
                return
                
            for task in tasks:
                status_icon = {
                    'running': '🔄',
                    'completed': '✅',
                    'failed': '❌',
                    'cancelled': '⏹️',
                    'queued': '⏳'
                }.get(task['status'], '❓')
                
                print(f"{status_icon} {task['task_id'][:8]}... | {task['task_name']:12} | {task.get('task_type', 'unknown'):10} | {task['status']:10} | {task['progress']:5.1f}%")
        else:
            print(f"获取任务列表失败: {response.status_code}")
    except Exception as e:
        print(f"获取任务列表异常: {e}")

def list_stuck_tasks(timeout_minutes=30):
    """列出卡住的任务"""
    try:
        response = requests.get(f'http://localhost:8000/api/v1/tasks/monitor/stuck?timeout_minutes={timeout_minutes}', timeout=10)
        if response.status_code == 200:
            data = response.json()
            stuck_tasks = data.get('data', {}).get('stuck_tasks', [])
            
            print(f"=== 卡住的任务 (超时 {timeout_minutes} 分钟) ===")
            if not stuck_tasks:
                print("没有发现卡住的任务")
                return
                
            for task in stuck_tasks:
                print(f"🔄 {task['task_id'][:8]}... | {task['task_name']:12} | {task.get('task_type', 'unknown'):10} | {task['status']:10} | {task['progress']:5.1f}%")
                print(f"   创建时间: {task['created_at']}")
                print(f"   开始时间: {task['started_at']}")
                print()
        else:
            print(f"获取卡住任务失败: {response.status_code}")
    except Exception as e:
        print(f"获取卡住任务异常: {e}")

def stop_task(task_id):
    """停止任务"""
    try:
        response = requests.post(f'http://localhost:8000/api/v1/tasks/{task_id}/stop', timeout=10)
        if response.status_code == 200:
            print(f"✅ 任务已停止: {task_id}")
        else:
            print(f"❌ 停止任务失败: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"停止任务异常: {e}")

def delete_task(task_id):
    """删除任务"""
    try:
        response = requests.delete(f'http://localhost:8000/api/v1/tasks/{task_id}', timeout=10)
        if response.status_code == 200:
            print(f"✅ 任务已删除: {task_id}")
        else:
            print(f"❌ 删除任务失败: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"删除任务异常: {e}")

def force_complete_task(task_id, status='cancelled'):
    """强制完成任务"""
    try:
        response = requests.post(f'http://localhost:8000/api/v1/tasks/monitor/force-complete/{task_id}?status={status}', timeout=10)
        if response.status_code == 200:
            print(f"✅ 任务已强制设置为 {status}: {task_id}")
        else:
            print(f"❌ 强制完成任务失败: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"强制完成任务异常: {e}")

def cleanup_stuck_tasks(timeout_minutes=30, auto_fix=False):
    """清理卡住的任务"""
    try:
        response = requests.post(f'http://localhost:8000/api/v1/tasks/monitor/cleanup?timeout_minutes={timeout_minutes}&auto_fix={auto_fix}', timeout=30)
        if response.status_code == 200:
            data = response.json()
            result = data.get('data', {})
            
            print(f"=== 清理结果 ===")
            print(f"发现卡住任务: {result.get('total_stuck', 0)} 个")
            print(f"修复任务: {len(result.get('fixed_tasks', []))} 个")
            print(f"失败任务: {len(result.get('failed_tasks', []))} 个")
            
            if result.get('fixed_tasks'):
                print("\n修复的任务:")
                for task in result['fixed_tasks']:
                    print(f"  ✅ {task['task_id'][:8]}... ({task['task_name']}) -> {task['action']}")
                    
            if result.get('failed_tasks'):
                print("\n失败的任务:")
                for task in result['failed_tasks']:
                    print(f"  ❌ {task['task_id'][:8]}... ({task['task_name']}) - {task['error']}")
        else:
            print(f"清理任务失败: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"清理任务异常: {e}")

def get_statistics():
    """获取任务统计"""
    try:
        response = requests.get('http://localhost:8000/api/v1/tasks/monitor/statistics', timeout=10)
        if response.status_code == 200:
            data = response.json()
            stats = data.get('data', {})
            
            print("=== 任务统计 ===")
            print(f"总任务数: {stats.get('total_tasks', 0)}")
            print(f"最近24小时: {stats.get('recent_24h', 0)}")
            
            status_counts = stats.get('status_counts', {})
            if status_counts:
                print("\n各状态任务数:")
                for status, count in status_counts.items():
                    icon = {
                        'running': '🔄',
                        'completed': '✅',
                        'failed': '❌',
                        'cancelled': '⏹️',
                        'queued': '⏳'
                    }.get(status, '❓')
                    print(f"  {icon} {status}: {count}")
        else:
            print(f"获取统计失败: {response.status_code}")
    except Exception as e:
        print(f"获取统计异常: {e}")

def main():
    parser = argparse.ArgumentParser(description='任务管理工具')
    parser.add_argument('command', choices=[
        'list', 'stuck', 'stop', 'delete', 'force', 'cleanup', 'stats'
    ], help='操作命令')
    parser.add_argument('--task-id', help='任务ID')
    parser.add_argument('--status', default='cancelled', choices=['cancelled', 'failed', 'completed'], help='强制设置的状态')
    parser.add_argument('--timeout', type=int, default=30, help='任务超时时间（分钟）')
    parser.add_argument('--auto-fix', action='store_true', help='自动修复卡住的任务')
    
    args = parser.parse_args()
    
    # 检查后端服务
    if not check_backend():
        print("❌ 后端服务未运行，请先启动后端服务")
        sys.exit(1)
    
    print("✅ 后端服务运行正常")
    print()
    
    if args.command == 'list':
        list_tasks()
    elif args.command == 'stuck':
        list_stuck_tasks(args.timeout)
    elif args.command == 'stop':
        if not args.task_id:
            print("❌ 请提供任务ID: --task-id <task_id>")
            sys.exit(1)
        stop_task(args.task_id)
    elif args.command == 'delete':
        if not args.task_id:
            print("❌ 请提供任务ID: --task-id <task_id>")
            sys.exit(1)
        delete_task(args.task_id)
    elif args.command == 'force':
        if not args.task_id:
            print("❌ 请提供任务ID: --task-id <task_id>")
            sys.exit(1)
        force_complete_task(args.task_id, args.status)
    elif args.command == 'cleanup':
        cleanup_stuck_tasks(args.timeout, args.auto_fix)
    elif args.command == 'stats':
        get_statistics()

if __name__ == "__main__":
    main()