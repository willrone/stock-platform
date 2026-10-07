"""按使用量计费模型初始迁移

创建 commerce 相关的所有表格：
- usage_events: 用量事件记录
- user_balances: 用户钱包余额
- billing_records: 扣费账单
- commerce_pricing_rules: 计费规则配置

注意：如果 usage_events 表已存在（由其他迁移创建），则跳过创建。
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

# revision identifiers, used by Alembic.
revision = "commerce_001"
down_revision = None
branch_labels = None
depends_on = None


def _table_exists(table_name: str) -> bool:
    """检查表是否已存在"""
    bind = op.get_bind()
    inspector = inspect(bind)
    tables = inspector.get_table_names()
    return table_name in tables


def _index_exists(index_name: str) -> bool:
    """检查索引是否已存在"""
    bind = op.get_bind()
    inspector = inspect(bind)
    indexes = inspector.get_indexes("usage_events")  # 默认检查 usage_events
    return any(idx["name"] == index_name for idx in indexes)


def upgrade() -> None:
    """升级：创建 commerce 相关表格"""
    bind = op.get_bind()
    dialect_name = bind.dialect.name
    
    # ── 用量事件表 ──
    if not _table_exists("usage_events"):
        op.create_table(
            "usage_events",
            sa.Column("id", sa.String(36), primary_key=True),
            sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
            sa.Column("event_type", sa.String(50), nullable=False),
            sa.Column("unit_cost", sa.Numeric(10, 2), nullable=False),
            sa.Column("quantity", sa.Integer, nullable=False, default=1),
            sa.Column("total_cost", sa.Numeric(10, 2), nullable=False),
            sa.Column("metadata", sa.Text, nullable=True),
            sa.Column("billing_record_id", sa.String(36), nullable=True),
            sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        )
        op.create_index("ix_usage_events_user_id", "usage_events", ["user_id"])
        op.create_index("ix_usage_events_event_type", "usage_events", ["event_type"])
        op.create_index("ix_usage_events_created_at", "usage_events", ["created_at"])
    else:
        # 表已存在，添加缺失的列（用于从旧版本迁移）
        inspector = inspect(bind)
        columns = {col["name"] for col in inspector.get_columns("usage_events")}
        
        if "billing_record_id" not in columns:
            op.add_column("usage_events", sa.Column("billing_record_id", sa.String(36), nullable=True))
            op.create_foreign_key(
                "fk_usage_events_billing_record_id",
                "usage_events", "billing_records",
                ["billing_record_id"], ["id"],
                ondelete="SET NULL"
            )
        
        if "metadata" not in columns:
            op.add_column("usage_events", sa.Column("metadata", sa.Text, nullable=True))
        
        if "unit_cost" not in columns:
            op.add_column("usage_events", sa.Column("unit_cost", sa.Numeric(10, 2), nullable=False, server_default="0"))
        
        if "total_cost" not in columns:
            op.add_column("usage_events", sa.Column("total_cost", sa.Numeric(10, 2), nullable=False, server_default="0"))
        
        if "quantity" not in columns:
            op.add_column("usage_events", sa.Column("quantity", sa.Integer, nullable=False, default=1))

    # ── 用户余额表 ──
    if not _table_exists("user_balances"):
        op.create_table(
            "user_balances",
            sa.Column("id", sa.String(36), primary_key=True),
            sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False, unique=True),
            sa.Column("balance_cents", sa.Integer, nullable=False, default=0),
            sa.Column("credited_cents", sa.Integer, nullable=False, default=0),
            sa.Column("total_spent_cents", sa.Integer, nullable=False, default=0),
            sa.Column("updated_at", sa.DateTime, nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        )
        op.create_index("ix_user_balances_user_id", "user_balances", ["user_id"])

    # ── 账单记录表 ──
    if not _table_exists("billing_records"):
        op.create_table(
            "billing_records",
            sa.Column("id", sa.String(36), primary_key=True),
            sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
            sa.Column("usage_event_id", sa.String(36), sa.ForeignKey("usage_events.id", ondelete="CASCADE"), nullable=True),
            sa.Column("amount_cents", sa.Integer, nullable=False),
            sa.Column("charge_status", sa.String(20), nullable=False, default="pending"),
            sa.Column("payment_method", sa.String(50), nullable=True),
            sa.Column("balance_before_cents", sa.Integer, nullable=True),
            sa.Column("balance_after_cents", sa.Integer, nullable=True),
            sa.Column("remark", sa.Text, nullable=True),
            sa.Column("charged_at", sa.DateTime, nullable=True),
            sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        )
        op.create_index("ix_billing_records_user_id", "billing_records", ["user_id"])
        op.create_index("ix_billing_records_usage_event_id", "billing_records", ["usage_event_id"])
        op.create_index("ix_billing_records_charge_status", "billing_records", ["charge_status"])
        op.create_index("ix_billing_records_created_at", "billing_records", ["created_at"])

    # ── 计费规则表 ──
    if not _table_exists("commerce_pricing_rules"):
        op.create_table(
            "commerce_pricing_rules",
            sa.Column("id", sa.String(36), primary_key=True),
            sa.Column("event_type", sa.String(50), nullable=False, unique=True),
            sa.Column("base_price_cents", sa.Integer, nullable=False),
            sa.Column("discounted_price_cents", sa.Integer, nullable=True),
            sa.Column("unit_name", sa.String(50), nullable=True),
            sa.Column("description", sa.Text, nullable=True),
            sa.Column("is_active", sa.Boolean, nullable=False, default=True),
            sa.Column("effective_from", sa.DateTime, nullable=True),
            sa.Column("effective_until", sa.DateTime, nullable=True),
            sa.Column("tier_discounts", sa.Text, nullable=True),
            sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
            sa.Column("updated_at", sa.DateTime, nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        )
        op.create_index("ix_commerce_pricing_rules_event_type", "commerce_pricing_rules", ["event_type"])
        op.create_index("ix_commerce_pricing_rules_is_active", "commerce_pricing_rules", ["is_active"])


def downgrade() -> None:
    """降级：删除 commerce 相关表格"""
    op.drop_index("ix_commerce_pricing_rules_event_type", table_name="commerce_pricing_rules")
    op.drop_index("ix_commerce_pricing_rules_is_active", table_name="commerce_pricing_rules")
    op.drop_table("commerce_pricing_rules")

    op.drop_index("ix_billing_records_user_id", table_name="billing_records")
    op.drop_index("ix_billing_records_usage_event_id", table_name="billing_records")
    op.drop_index("ix_billing_records_charge_status", table_name="billing_records")
    op.drop_index("ix_billing_records_created_at", table_name="billing_records")
    op.drop_table("billing_records")

    op.drop_index("ix_user_balances_user_id", table_name="user_balances")
    op.drop_table("user_balances")

    # 注意：usage_events 可能由其他迁移创建，不在此处删除
    op.drop_index("ix_usage_events_user_id", table_name="usage_events")
    op.drop_index("ix_usage_events_event_type", "usage_events")
    op.drop_index("ix_usage_events_created_at", "usage_events")