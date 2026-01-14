"""
添加策略配置表的迁移脚本
"""

import asyncio
from sqlalchemy import text
from app.core.database import async_engine


async def upgrade():
    """创建策略配置表"""
    async with async_engine.begin() as conn:
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS strategy_configs (
                config_id VARCHAR PRIMARY KEY,
                config_name VARCHAR(255) NOT NULL,
                strategy_name VARCHAR(100) NOT NULL,
                parameters JSON NOT NULL,
                description TEXT,
                user_id VARCHAR(255),
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """))
        
        # 创建索引
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_strategy_configs_strategy_name 
            ON strategy_configs(strategy_name)
        """))
        
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_strategy_configs_user_id 
            ON strategy_configs(user_id)
        """))


async def downgrade():
    """删除策略配置表"""
    async with async_engine.begin() as conn:
        await conn.execute(text("DROP INDEX IF EXISTS idx_strategy_configs_user_id"))
        await conn.execute(text("DROP INDEX IF EXISTS idx_strategy_configs_strategy_name"))
        await conn.execute(text("DROP TABLE IF EXISTS strategy_configs"))


if __name__ == "__main__":
    asyncio.run(upgrade())
    print("策略配置表创建成功")

