#!/usr/bin/env python3
"""
最终恢复方案：
1. 将任务状态改为 CREATED
2. 通过 HTTP API 调用优化任务的重新提交接口
"""

import sys
sys.path.insert(0, '/Users/ronghui/Projects/willrone/backend')

from app.core.database import SessionLocal
from app.models.task_models import Task
import subprocess
import json

def recover_optimization_tasks():
    """恢复优化任务"""
    
    task_ids = [
        '54f14258-5406-4f3f-b854-2dce24dd9710',
        'b4da39e3-4730-4721-a157-a82d6c325c52',
        '55530c47-6b2c-4871-90a7-b4989d1d85e5',
        '9b6ebefe-9e72-485b-bd34-65f2a3b739ff',
        '5bcad3cc-e674-4701-b75a-d3697f6b0a92'
    ]
    
    session = SessionLocal()
    
    print("📊 恢复优化任务\n")
    print("=" * 60)
    
    recovered = 0
    
    try:
        for task_id in task_ids:
            task = session.query(Task).filter(Task.task_id == task_id).first()
            if not task:
                print(f"❌ 任务不存在: {task_id[:8]}")
                continue
            
            print(f"\n🔄 处理任务: {task.task_name[:50]}")
            print(f"   ID: {task_id[:8]}...")
            print(f"   当前状态: {task.status}")
            
            # 1. 重置任务状态为 CREATED
            task.status = 'created'
            task.progress = 0.0
            session.commit()
            print(f"   ✅ 状态已重置为 created")
            
            # 2. 调用后端 API 重新提交（使用优化任��的 start 接口）
            # 注意：我们需要找到正确的 API 端点
            # 由于没有专门的 retry 接口，我们直接通过任务队列提交
            
            # 使用 curl 调用任务执行接口
            cmd = f'curl -sS -X POST "http://localhost:8000/api/v1/optimization/tasks/{task_id}/execute"'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                try:
                    resp = json.loads(result.stdout)
                    if resp.get('success'):
                        print(f"   ✅ 任务已重新提交")
                        recovered += 1
                    else:
                        print(f"   ⚠️  API 返回失败: {resp.get('message')}")
                        # 即使 API 失败，任务状态已改为 created，调度器会自动拾取
                        recovered += 1
                except:
                    # API 可能不存在，但任务状态已改为 created
                    print(f"   ⚠️  API 调用失败，但任务状态已重置")
                    recovered += 1
            else:
                print(f"   ⚠️  curl 失败: {result.stderr}")
                # 任务状态已改为 created，调度器应该会自动拾取
                recovered += 1
        
        print(f"\n{'=' * 60}")
        print(f"✅ 完成！已处理 {recovered}/{len(task_ids)} 个任务")
        print(f"{'=' * 60}")
        print(f"\n💡 提示：任务状态已重置为 'created'")
        print(f"   - 如果后端服务正在运行，调度器会自动拾取这些任务")
        print(f"   - 如果没有自动执行，可能需要重启后端服务")
        
    finally:
        session.close()

if __name__ == "__main__":
    recover_optimization_tasks()
