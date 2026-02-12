"""
手动提交 pending 任务到进程池
"""
import sys
from app.services.tasks.process_executor import get_process_executor, start_process_executor
from app.api.v1.optimization import execute_optimization_task_simple
from app.core.database import SessionLocal
from app.models.task_models import Task

def submit_pending_tasks():
    """提交所有 pending 状态的优化任务"""
    session = SessionLocal()
    try:
        # 查询所有 pending 状态的优化任务
        pending_tasks = session.query(Task).filter(
            Task.status == 'pending',
            Task.task_type == 'hyperparameter_optimization'
        ).all()

        if not pending_tasks:
            print("✅ 没有 pending 状态的优化任务")
            return

        print(f"📊 发现 {len(pending_tasks)} 个 pending 任务\n")

        # 启动进程池
        print("🔧 启动进程池...")
        start_process_executor()
        process_executor = get_process_executor()
        print("✅ 进程池已启动\n")

        for task in pending_tasks:
            try:
                print(f"🚀 提交任务: {task.task_name}")
                print(f"   ID: {task.task_id}")
                
                # 提交到进程池
                future = process_executor.submit(
                    execute_optimization_task_simple,
                    task.task_id
                )
                
                print(f"   ✅ 已提交到进程池\n")
                
            except Exception as e:
                print(f"   ❌ 提交失败: {e}\n")

        print(f"✅ 完成！已提交 {len(pending_tasks)} 个任务")

    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        session.close()

if __name__ == "__main__":
    submit_pending_tasks()
