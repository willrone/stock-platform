"""
回测统计信息迁移脚本
为现有的回测任务计算并填充统计信息
"""

import asyncio
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import select
from app.core.database import get_async_session
from app.models.backtest_detailed_models import BacktestStatistics, SignalRecord, TradeRecord
from app.services.backtest.statistics import StatisticsCalculator
from loguru import logger


async def migrate_statistics():
    """迁移统计信息"""
    logger.info("开始迁移回测统计信息...")
    
    async for session in get_async_session():
        try:
            # 查找所有有信号记录或交易记录的任务
            # 获取所有唯一的 task_id
            signal_task_stmt = select(SignalRecord.task_id).distinct()
            signal_result = await session.execute(signal_task_stmt)
            signal_task_ids = set(row[0] for row in signal_result.all())
            
            trade_task_stmt = select(TradeRecord.task_id).distinct()
            trade_result = await session.execute(trade_task_stmt)
            trade_task_ids = set(row[0] for row in trade_result.all())
            
            all_task_ids = signal_task_ids | trade_task_ids
            logger.info(f"找到 {len(all_task_ids)} 个需要迁移的任务")
            
            # 检查哪些任务已经有统计信息
            existing_stats_stmt = select(BacktestStatistics.task_id)
            existing_result = await session.execute(existing_stats_stmt)
            existing_task_ids = set(row[0] for row in existing_result.all())
            
            # 需要迁移的任务（有数据但没有统计信息）
            tasks_to_migrate = all_task_ids - existing_task_ids
            logger.info(f"需要迁移的任务数: {len(tasks_to_migrate)}")
            
            if not tasks_to_migrate:
                logger.info("没有需要迁移的任务")
                return
            
            # 计算统计信息
            calculator = StatisticsCalculator(session)
            success_count = 0
            error_count = 0
            
            for task_id in tasks_to_migrate:
                try:
                    # 获取 backtest_id（从信号记录或交易记录中获取）
                    backtest_id = None
                    signal_stmt = select(SignalRecord.backtest_id).where(
                        SignalRecord.task_id == task_id
                    ).limit(1)
                    signal_result = await session.execute(signal_stmt)
                    signal_row = signal_result.first()
                    if signal_row:
                        backtest_id = signal_row[0]
                    else:
                        trade_stmt = select(TradeRecord.backtest_id).where(
                            TradeRecord.task_id == task_id
                        ).limit(1)
                        trade_result = await session.execute(trade_stmt)
                        trade_row = trade_result.first()
                        if trade_row:
                            backtest_id = trade_row[0]
                    
                    if not backtest_id:
                        backtest_id = f"bt_{task_id[:8]}"
                    
                    logger.info(f"计算统计信息: task_id={task_id}, backtest_id={backtest_id}")
                    stats = await calculator.calculate_all_statistics(task_id, backtest_id)
                    await session.flush()
                    success_count += 1
                    
                    if success_count % 10 == 0:
                        logger.info(f"进度: {success_count}/{len(tasks_to_migrate)}")
                        await session.commit()
                        
                except Exception as e:
                    error_count += 1
                    logger.error(f"计算统计信息失败: task_id={task_id}, error={e}", exc_info=True)
                    await session.rollback()
                    continue
            
            # 提交剩余的更改
            await session.commit()
            
            logger.info(f"迁移完成: 成功 {success_count} 个, 失败 {error_count} 个")
            
        except Exception as e:
            await session.rollback()
            logger.error(f"迁移失败: {e}", exc_info=True)
            raise
        finally:
            break


if __name__ == "__main__":
    asyncio.run(migrate_statistics())
