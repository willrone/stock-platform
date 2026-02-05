#!/usr/bin/env python3
"""
迁移脚本：将 tasks.result.trade_history 数据同步到 trade_records 表
用于修复旧任务没有交易记录的问题
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from app.models.backtest_detailed_models import TradeRecord


async def migrate_trade_history(task_id: str = None):
    """迁移交易历史数据"""
    
    # 创建数据库连接
    engine = create_async_engine(
        settings.DATABASE_URL.replace("sqlite:///", "sqlite+aiosqlite:///"),
        echo=False
    )
    
    async_session = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    
    async with async_session() as session:
        # 查询需要迁移的任务
        if task_id:
            query = text("""
                SELECT task_id, result 
                FROM tasks 
                WHERE task_id = :task_id 
                AND task_type = 'backtest'
                AND result IS NOT NULL
            """)
            result = await session.execute(query, {"task_id": task_id})
        else:
            query = text("""
                SELECT task_id, result 
                FROM tasks 
                WHERE task_type = 'backtest'
                AND result IS NOT NULL
            """)
            result = await session.execute(query)
        
        tasks = result.fetchall()
        print(f"找到 {len(tasks)} 个回测任务需要检查")
        
        for task_row in tasks:
            task_id = task_row[0]
            result_json = task_row[1]
            
            if not result_json:
                print(f"  跳过任务 {task_id}: 没有结果数据")
                continue
            
            try:
                result_data = json.loads(result_json)
            except json.JSONDecodeError:
                print(f"  跳过任务 {task_id}: JSON 解析失败")
                continue
            
            trade_history = result_data.get("trade_history", [])
            if not trade_history:
                print(f"  跳过任务 {task_id}: 没有交易历史")
                continue
            
            # 检查是否已有交易记录
            check_query = text("""
                SELECT COUNT(*) FROM trade_records WHERE task_id = :task_id
            """)
            check_result = await session.execute(check_query, {"task_id": task_id})
            existing_count = check_result.scalar()
            
            if existing_count > 0:
                print(f"  跳过任务 {task_id}: 已有 {existing_count} 条交易记录")
                continue
            
            print(f"  迁移任务 {task_id}: {len(trade_history)} 条交易记录...")
            
            # 批量插入交易记录
            backtest_id = f"bt_{task_id[:8]}"
            trades_to_insert = []
            
            for trade in trade_history:
                # 解析时间戳
                timestamp_value = trade.get("timestamp")
                if isinstance(timestamp_value, str):
                    try:
                        timestamp_value = datetime.fromisoformat(
                            timestamp_value.replace("Z", "+00:00")
                        )
                    except:
                        timestamp_value = datetime.now()
                
                trade_record = TradeRecord(
                    task_id=task_id,
                    backtest_id=backtest_id,
                    trade_id=trade.get("trade_id", ""),
                    stock_code=trade.get("stock_code", ""),
                    stock_name=trade.get("stock_code", ""),  # 使用股票代码作为名称
                    action=trade.get("action", ""),
                    quantity=int(trade.get("quantity", 0)),
                    price=float(trade.get("price", 0)),
                    timestamp=timestamp_value,
                    commission=float(trade.get("commission", 0)),
                    pnl=float(trade.get("pnl", 0)) if trade.get("pnl") else None,
                    holding_days=trade.get("holding_days"),
                    technical_indicators={},
                )
                trades_to_insert.append(trade_record)
            
            # 批量添加
            session.add_all(trades_to_insert)
            await session.flush()
            
            print(f"    ✓ 成功插入 {len(trades_to_insert)} 条交易记录")
        
        # 提交所有更改
        await session.commit()
        print("\n迁移完成!")
    
    await engine.dispose()


if __name__ == "__main__":
    # 可以指定特定任务 ID，或者不指定则迁移所有任务
    task_id = sys.argv[1] if len(sys.argv) > 1 else None
    
    if task_id:
        print(f"迁移指定任务: {task_id}")
    else:
        print("迁移所有回测任务的交易历史...")
    
    asyncio.run(migrate_trade_history(task_id))
