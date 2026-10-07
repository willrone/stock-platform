"""
按使用量计费 - 单元测试

测试模型、服务、API 的正确性
"""

import asyncio
import sys
from decimal import Decimal

# 添加项目根目录到路径
sys.path.insert(0, "/Users/ronghui/Projects/stock-platform")

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from app.core.database import Base
from app.models.commerce_models import (
    BillingRecord,
    ChargeStatus,
    CommercePricingRule,
    DEFAULT_PRICING_RULES,
    EventType,
    UserBalance,
    UsageEvent,
)
from app.models.user_models import User
from app.services.commerce.commission_service import (
    CommerceService,
    check_can_proceed,
    seed_default_pricing_rules,
)


async def test_models():
    """测试数据库模型创建"""
    print("=" * 60)
    print("测试 1: 模型创建")
    print("=" * 60)
    
    # 创建内存 SQLite 数据库
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    SessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    db = SessionLocal()
    
    try:
        # 测试创建用户
        user = User(
            id="test-user-1",
            email="test@example.com",
            username="testuser",
            hashed_password="hashed_password",
        )
        db.add(user)
        await db.flush()
        print("✅ 用户创建成功")
        
        # 测试创建用户余额
        balance = UserBalance(
            user_id="test-user-1",
            balance_cents=1500,
            credited_cents=1000,
        )
        db.add(balance)
        await db.flush()
        print(f"✅ 用户余额创建成功: {balance.balance_cents} 分")
        
        # 测试创建计费规则
        rule = CommercePricingRule(
            event_type="test_event",
            base_price_cents=100,
            unit_name="次",
            description="测试事件",
            is_active=True,
        )
        db.add(rule)
        await db.flush()
        print(f"✅ 计费规则创建成功: {rule.event_type} @ {rule.base_price_cents} 分")
        
        # 测试创建用量事件
        event = UsageEvent(
            user_id="test-user-1",
            event_type="test_event",
            unit_cost=100,
            quantity=1,
            total_cost=100,
            metadata_json='{"test": true}',
        )
        db.add(event)
        await db.flush()
        print(f"✅ 用量事件创建成功: {event.id}")
        
        # 测试创建账单记录
        record = BillingRecord(
            user_id="test-user-1",
            usage_event_id=event.id,
            amount_cents=100,
            charge_status="charged",
            balance_before_cents=1500,
            balance_after_cents=1400,
        )
        db.add(record)
        await db.flush()
        print(f"✅ 账单记录创建成功: {record.id}")
        
        return True
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        await db.close()
        await engine.dispose()


async def test_commerce_service():
    """测试计费服务"""
    print("\n" + "=" * 60)
    print("测试 2: CommerceService")
    print("=" * 60)
    
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    SessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    db = SessionLocal()
    
    try:
        # 创建测试用户
        user = User(
            id="test-user-2",
            email="commerce@example.com",
            username="commerce_user",
            hashed_password="hashed_password",
        )
        db.add(user)
        await db.flush()
        print("✅ 测试用户创建")
        
        service = CommerceService(db=db)
        
        # 1. 记录用量（余额充足）
        event, billing = await service.record_usage(
            user_id="test-user-2",
            event_type="backtest_basic",
            quantity=1,
            metadata={"symbol": "AAPL", "strategy": "test"},
        )
        print(f"✅ 用量记录: event_id={event.id[:8]}..., total_cost={event.total_cost} 分")
        print(f"   扣费状态: {billing.charge_status if billing else 'None'}")
        
        # 2. 查询余额
        balance = await service.get_balance("test-user-2")
        print(f"✅ 当前余额: {balance.balance_cents} 分 = {balance.balance_cents/100:.2f} 元")
        
        # 3. 充值
        new_balance, record = await service.deposit(
            user_id="test-user-2",
            amount_cents=500,
            remark="测试充值",
        )
        print(f"✅ 充值后余额: {new_balance.balance_cents} 分 = {new_balance.balance_cents/100:.2f} 元")
        
        # 4. 查询计费规则
        rules = await service.get_pricing_rules()
        print(f"✅ 计费规则数量: {len(rules)}")
        
        # 5. 用量汇总
        summary = await service.get_usage_summary("test-user-2")
        print(f"✅ 用量汇总: 总支出 {summary['total_spent_yuan']:.2f} 元")
        
        return True
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        await db.close()
        await engine.dispose()


async def test_default_pricing_rules():
    """测试默认计费规则"""
    print("\n" + "=" * 60)
    print("测试 3: 默认计费规则初始化")
    print("=" * 60)
    
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    db = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)()
    
    try:
        rules = await seed_default_pricing_rules(db)
        print(f"✅ 初始化计费规则: {len(rules)} 条")
        for rule in rules:
            print(f"   - {rule.event_type}: ¥{rule.base_price_cents/100:.2f}/次")
        
        return True
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        await db.close()
        await engine.dispose()


async def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("按使用量计费系统 - 单元测试")
    print("=" * 60 + "\n")
    
    results = []
    
    results.append(await test_models())
    results.append(await test_commerce_service())
    results.append(await test_default_pricing_rules())
    
    print("\n" + "=" * 60)
    print(f"测试结果: {'✅ 全部通过' if all(results) else '❌ 部分失败'}")
    print("=" * 60)
    
    return all(results)


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)