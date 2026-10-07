"""Alembic 迁移环境配置。

读取 backend/app/core/config 的 DATABASE_URL，
导入全部 ORM 模型以支持 autogenerate。
"""

import sys
from pathlib import Path
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool
from alembic import context

# 确保 backend 目录在 sys.path 中
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import settings
from app.core.database import Base

# ── 导入全部 ORM 模型，确保 Base.metadata 覆盖所有表 ──
import app.models.user_models          # users
import app.models.subscription_models  # subscription_plans, user_subscriptions, usage_records
import app.models.task_models          # tasks
import app.models.stock                # prediction_results
import app.models.backtest_detailed_models  # backtest_results, backtest_detailed_results, backtest_chart_cache, backtest_statistics, backtest_benchmarks
import app.models.strategy_config_models    # strategy_configs
import app.models.sync_models          # portfolio_snapshots, trade_records, signal_records
import app.models.file_management      # (file-related tables, if any)
import app.models.commerce_models      # 计费相关模型（按使用量付费）

# Alembic Config 对象
config = context.config

# 从 settings 注入同步数据库 URL
db_url = settings.database_url_sync
config.set_main_option("sqlalchemy.url", db_url)

# Python logging 配置
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """离线模式迁移（生成 SQL 脚本）。"""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,  # SQLite 兼容：用 batch 模式支持 ALTER
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """在线模式迁移（直接操作数据库）。"""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
