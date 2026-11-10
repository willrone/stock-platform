"""billing_records.usage_event_id 改为可空

**为什么**：`billing_records` 原本只用于「扣费」记录（每条记录必关联一个
usage_event）。2026-10-25 引入两段式充值订单制后，管理员确认到账时也需要
写一条资金流水（`charge_status='credited'`），但充值**没有对应的用量事件**，
代码里传了 `usage_event_id=None` —— 而该列是 `NOT NULL`：

    sqlalchemy.exc.IntegrityError: NOT NULL constraint failed:
    billing_records.usage_event_id

结果：**管理员确认到账接口在真实数据库上必然 500，充值到账从来没成功过**
（`backend/scripts/test_payment_upgrade_e2e.py` 是把它测出来的那个测试）。

**修法**：把该列改为可空。`usage_event_id` 的外键关系保持不变 ——
扣费记录仍然必填（service 层保证），只有充值流水为 NULL。
SQLite 不支持直接改列，用 batch_alter_table 重建表。

Revision ID: commerce_004_billing_usage_event_nullable
Revises: commerce_003
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "commerce_004_billing_usage_event_nullable"
down_revision = "commerce_003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("billing_records") as batch:
        batch.alter_column(
            "usage_event_id",
            existing_type=sa.String(length=36),
            nullable=True,
        )


def downgrade() -> None:
    # 回滚前必须先清理掉充值流水（usage_event_id 为 NULL 的行），
    # 否则 NOT NULL 约束加回去会失败
    op.execute(
        "DELETE FROM billing_records WHERE usage_event_id IS NULL"
    )
    with op.batch_alter_table("billing_records") as batch:
        batch.alter_column(
            "usage_event_id",
            existing_type=sa.String(length=36),
            nullable=False,
        )
