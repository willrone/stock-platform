#!/usr/bin/env python3
"""
正确的任务恢复脚本：通过 API 重新提交任务
"""

import requests
import sys

def recover_tasks_via_api():
    """通过 API 恢复卡住的任务"""
    
    # 5 个卡住的任务
    task_ids = [
        '54f14258-5406-4f3f-b854-2dce24dd9710',
        'b4da39e3-4730-4721-a157-a82d6c325c52',
        '55530c47-6b2c-4871-90a7-b4989d1d85e5',
        '9b6ebefe-9e72-485b-bd34-65f2a3b739ff',
        '5bcad3cc-e674-4701-b75a-d3697f6b0a92'
    ]
    
    base_url = "http://localhost:8000/api/v1"
    
    print(f"📊 开始恢复 {len(task_ids)} 个任务...\n")
    
    recovered = 0
    failed = 0
    
    for task_id in task_ids:
        try:
            # 1. 先取消任务（将 running 改为 cancelled）
            print(f"🔄 取消任务: {task_id[:8]}...")
            cancel_resp = requests.post(
                f"{base_url}/tasks/{task_id}/cancel",
                timeout=5
            )
            
            if cancel_resp.status_code != 200:
                print(f"  ⚠️  取消失败: {cancel_resp.status_code}")
                # 继续尝试重试
            else:
                print(f"  ✅ 已取消")
            
            # 2. 重试任务（将 cancelled 改为 created 并重新提交）
            print(f"🚀 重新提交任务: {task_id[:8]}...")
            retry_resp = requests.post(
                f"{base_url}/tasks/{task_id}/retry",
                timeout=5
            )
            
            if retry_resp.status_code == 200:
                print(f"  ✅ 提交成功\n")
                recovered += 1
            else:
                print(f"  ❌ 提交失败: {retry_resp.status_code}")
                print(f"     {retry_resp.text}\n")
                failed += 1
                
        except Exception as e:
            print(f"  ❌ 错误: {e}\n")
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"✅ 成功恢复: {recovered}/{len(task_ids)}")
    print(f"❌ 失败: {failed}/{len(task_ids)}")
    print(f"{'='*60}")
    
    return recovered == len(task_ids)

if __name__ == "__main__":
    success = recover_tasks_via_api()
    sys.exit(0 if success else 1)
