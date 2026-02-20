"""
数据库迁移脚本：为 backtest_results 表添加 benchmark_comparison JSONB 字段

执行方式：
python -m backend.migrations.add_benchmark_comparison_to_backtest_results
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


async def add_benchmark_comparison_column():
    """为 backtest_results 表添加 benchmark_comparison JSONB 字段"""
    try:
        async with get_async_session_context() as session:
            # 检查字段是否已存在（PostgreSQL）
            check_sql = """
            SELECT COUNT(*)
            FROM information_schema.columns
            WHERE table_name = 'backtest_results'
            AND column_name = 'benchmark_comparison';
            """
            result = await session.execute(text(check_sql))
            count = result.scalar()

            if count > 0:
                logger.info("字段 benchmark_comparison 已存在，跳过迁移")
                return True

            # 添加字段
            alter_sql = """
            ALTER TABLE backtest_results
            ADD COLUMN benchmark_comparison JSONB;
            """

            await session.execute(text(alter_sql))

            # 添加注释
            comment_sql = """
            COMMENT ON COLUMN backtest_results.benchmark_comparison
            IS '基准对比数据';
            """
            await session.execute(text(comment_sql))

            await session.commit()

            logger.info("成功为 backtest_results 表添加 benchmark_comparison 字段")
            return True

    except Exception as e:
        logger.error(f"添加 benchmark_comparison 字段失败: {e}", exc_info=True)
        return False


async def main():
    """主函数"""
    logger.info("开始迁移：添加 benchmark_comparison 字段到 backtest_results 表")
    success = await add_benchmark_comparison_column()

    if success:
        logger.info("迁移完成")
        sys.exit(0)
    else:
        logger.error("迁移失败")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
