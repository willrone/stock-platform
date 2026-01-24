"""
数据库迁移脚本：为 signal_records 表添加 execution_reason 字段

执行方式：
python -m backend.migrations.add_execution_reason_to_signal_records
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


async def add_execution_reason_column():
    """为 signal_records 表添加 execution_reason 字段"""
    try:
        async with get_async_session_context() as session:
            # 检查字段是否已存在
            check_sql = """
            SELECT COUNT(*) as count 
            FROM pragma_table_info('signal_records') 
            WHERE name = 'execution_reason';
            """
            result = await session.execute(text(check_sql))
            count = result.scalar()
            
            if count > 0:
                logger.info("字段 execution_reason 已存在，跳过迁移")
                return True
            
            # 添加字段
            alter_sql = """
            ALTER TABLE signal_records 
            ADD COLUMN execution_reason TEXT;
            """
            
            await session.execute(text(alter_sql))
            await session.commit()
            
            logger.info("成功为 signal_records 表添加 execution_reason 字段")
            return True
            
    except Exception as e:
        logger.error(f"添加 execution_reason 字段失败: {e}", exc_info=True)
        return False


async def main():
    """主函数"""
    logger.info("开始迁移：添加 execution_reason 字段到 signal_records 表")
    success = await add_execution_reason_column()
    
    if success:
        logger.info("迁移完成")
        sys.exit(0)
    else:
        logger.error("迁移失败")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
