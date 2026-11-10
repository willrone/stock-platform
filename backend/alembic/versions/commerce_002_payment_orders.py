"""充值订单制：payment_orders + payment_configs

2026-10-25。解决「/commerce/deposit 无支付验证即可加余额」的收入漏洞：
原实现任何注册用户都能自助充值无限余额。改为两段式订单制 ——
用户下单（pending）→ 管理员确认到账（paid）才写入余额。

同时新增 payment_configs（单行表）用于维护人工收款方式：
收款方式名 / 收款账号 / 收款码图片 URL / 支付说明 / 是否开启。
没有自动支付渠道（STRIPE_SECRET_KEY 为空）时走人工对账。
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

# revision identifiers, used by Alembic.
revision = "commerce_002"
down_revision = "commerce_001"
branch_labels = None
depends_on = None


def _table_exists(table_name: str) -> bool:
    bind = op.get_bind()
    inspector = inspect(bind)
    return table_name in inspector.get_table_names()


def upgrade() -> None:
    if not _table_exists("payment_orders"):
        op.create_table(
            "payment_orders",
            sa.Column("id", sa.String(36), primary_key=True),
            sa.Column("order_no", sa.String(32), nullable=False, unique=True),
            sa.Column(
                "user_id",
                sa.String(36),
                sa.ForeignKey("users.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column("amount_cents", sa.Integer(), nullable=False),
            sa.Column("status", sa.String(20), nullable=False, server_default="pending"),
            sa.Column("payment_method", sa.String(50), nullable=False, server_default="manual"),
            sa.Column("user_remark", sa.Text(), nullable=True),
            sa.Column("admin_remark", sa.Text(), nullable=True),
            sa.Column("confirmed_at", sa.DateTime(), nullable=True),
            sa.Column("confirmed_by", sa.String(64), nullable=True),
            sa.Column("billing_record_id", sa.String(36), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
        )
        op.create_index("ix_payment_orders_order_no", "payment_orders", ["order_no"])
        op.create_index("ix_payment_orders_user_id", "payment_orders", ["user_id"])

    if not _table_exists("payment_configs"):
        op.create_table(
            "payment_configs",
            sa.Column("id", sa.String(36), primary_key=True),
            sa.Column("method_name", sa.String(50), nullable=False, server_default=""),
            sa.Column("payee_account", sa.String(100), nullable=False, server_default=""),
            sa.Column("qr_image_url", sa.String(500), nullable=False, server_default=""),
            sa.Column("instructions", sa.Text(), nullable=True),
            sa.Column("is_enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
        )


def downgrade() -> None:
    if _table_exists("payment_configs"):
        op.drop_table("payment_configs")
    if _table_exists("payment_orders"):
        op.drop_index("ix_payment_orders_user_id", table_name="payment_orders")
        op.drop_index("ix_payment_orders_order_no", table_name="payment_orders")
        op.drop_table("payment_orders")
