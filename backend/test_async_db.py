"""
测试异步数据库操作
"""
import asyncio
from app.core.database import get_async_session_context
from app.repositories.async_task_repository import AsyncTaskRepository
from app.models.task_models import TaskType, TaskStatus


async def test_async_repository():
    """测试异步 repository"""
    print("开始测试异步数据库操作...")

    async with get_async_session_context() as session:
        repo = AsyncTaskRepository(session)

        # 测试创建任务
        print("\n1. 测试创建任务...")
        task = await repo.create_task(
            task_name="测试任务",
            task_type=TaskType.BACKTEST,
            user_id="test_user",
            config={"test": "data"}
        )
        print(f"✓ 任务创建成功: {task.task_id}")

        # 测试获取任务
        print("\n2. 测试获取任务...")
        retrieved_task = await repo.get_task_by_id(task.task_id)
        assert retrieved_task is not None
        assert retrieved_task.task_id == task.task_id
        print(f"✓ 任务获取成功: {retrieved_task.task_name}")

        # 测试更新任务状态
        print("\n3. 测试更新任务状态...")
        updated_task = await repo.update_task_status(
            task_id=task.task_id,
            status=TaskStatus.RUNNING,
            progress=50.0
        )
        assert updated_task.status == TaskStatus.RUNNING.value
        assert updated_task.progress == 50.0
        print(f"✓ 任务状态更新成功: {updated_task.status}, 进度: {updated_task.progress}%")

        # 测试获取用户任务列表
        print("\n4. 测试获取用户任务列表...")
        tasks = await repo.get_tasks_by_user(user_id="test_user", limit=10)
        assert len(tasks) > 0
        print(f"✓ 获取到 {len(tasks)} 个任务")

        # 测试删除任务
        print("\n5. 测试删除任务...")
        deleted = await repo.delete_task(
            task_id=task.task_id,
            user_id="test_user",
            force=True
        )
        assert deleted is True
        print(f"✓ 任务删除成功")

        # 验证任务已删除
        print("\n6. 验证任务已删除...")
        deleted_task = await repo.get_task_by_id(task.task_id)
        assert deleted_task is None
        print(f"✓ 确认任务已删除")

    print("\n✅ 所有测试通过！")


if __name__ == "__main__":
    asyncio.run(test_async_repository())
