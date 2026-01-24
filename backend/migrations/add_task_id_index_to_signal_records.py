"""
数据库迁移脚本：为 signal_records 表添加 task_id 单独索引

执行方式：
python -m backend.migrations.add_task_id_index_to_signal_records
"""

import asyncio
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import text
from loguru import logger
from app.core.database import get_async_session_context


async def add_task_id_index():
    """为 signal_records 表添加 task_id 单独索引"""
    try:
        async with get_async_session_context() as session:
            # 检查索引是否已存在
            check_sql = """
            SELECT COUNT(*) as count 
            FROM sqlite_master 
            WHERE type='index' AND name='idx_signal_task_id';
            """
            result = await session.execute(text(check_sql))
            count = result.scalar()
            
            if count > 0:
                logger.info("索引 idx_signal_task_id 已存在，跳过迁移")
                return True
            
            # 添加索引
            create_index_sql = """
            CREATE INDEX idx_signal_task_id ON signal_records(task_id);
            """
            
            await session.execute(text(create_index_sql))
            await session.commit()
            
            logger.info("成功为 signal_records 表添加 task_id 索引")
            return True
            
    except Exception as e:
        logger.error(f"添加 task_id 索引失败: {e}", exc_info=True)
        return False


async def main():
    """主函数"""
    logger.info("开始迁移：添加 task_id 索引到 signal_records 表")
    success = await add_task_id_index()
    
    if success:
        logger.info("迁移完成")
        sys.exit(0)
    else:
        logger.error("迁移失败")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
