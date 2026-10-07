"""初始化默认订阅套餐。

用法：
    python scripts/seed_plans.py

应用启动时 init_db 也会执行同样的幂等 seed；本脚本用于手动补齐新环境。
"""

import asyncio
import sys
from pathlib import Path

# 允许从 backend 根目录外调用本脚本。
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.database import SessionLocal, init_db
from app.models.subscription_models import DEFAULT_PLANS, SubscriptionPlan


def seed_plans() -> int:
    """补齐缺失套餐，返回本次新增数量。"""
    session = SessionLocal()
    created = 0
    try:
        for plan_data in DEFAULT_PLANS:
            existing = (
                session.query(SubscriptionPlan)
                .filter(SubscriptionPlan.name == plan_data["name"])
                .first()
            )
            if existing is not None:
                continue
            session.add(SubscriptionPlan(**plan_data))
            created += 1
        session.commit()
        return created
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


if __name__ == "__main__":
    asyncio.run(init_db())
    print(f"已新增 {seed_plans()} 个套餐")
