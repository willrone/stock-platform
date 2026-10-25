"""commerce_003: payment_orders 加 plan_id

2026-10-25：充值订单需要记录「本次充值对应的套餐」，
管理员确认到账时据此自动升级 user.subscription_tier ——
否则出现「收了钱但套餐没变」，用户付费了付费墙照样把他挡住。

Revision ID: commerce_003
Revises: commerce_002
Create Date: 2026-10-25
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "commerce_003"
down_revision = "commerce_002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("payment_orders") as batch_op:
        batch_op.add_column(sa.Column("plan_id", sa.String(length=50), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("payment_orders") as batch_op:
        batch_op.drop_column("plan_id")
