#!/usr/bin/env python3
"""
为已完成的回测任务填充详细数据
"""

import sys
import asyncio
from pathlib import Path

# 添加backend到路径
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from app.core.database import get_async_session
from app.repositories.task_repository import TaskRepository
from app.repositories.backtest_detailed_repository import BacktestDetailedRepository
from app.services.backtest.backtest_data_adapter import BacktestDataAdapter
from loguru import logger


async def fill_task_details(task_id: str):
    """为单个任务填充详细数据"""
    logger.info(f"开始处理任务: {task_id}")
    
    async for session in get_async_session():
        try:
            # 获取任务
            task_repo = TaskRepository(session)
            task = await task_repo.get_task_by_id(task_id)
            
            if not task:
                logger.error(f"任务不存在: {task_id}")
                return False
            
            if task.status != "completed":
                logger.warning(f"任务未完成: {task_id}, 状态: {task.status}")
                return False
            
            if not task.result:
                logger.warning(f"任务没有结果数据: {task_id}")
                return False
            
            # 检查是否已有详细数据
            detail_repo = BacktestDetailedRepository(session)
            existing = await detail_repo.get_detailed_result_by_task_id(task_id)
            if existing:
                logger.info(f"任务已有详细数据，跳过: {task_id}")
                return True
            
            # 转换数据
            logger.info(f"开始转换数据...")
            adapter = BacktestDataAdapter()
            enhanced_result = await adapter.adapt_backtest_result(task.result)
            
            # 保存详细结果
            logger.info(f"保存详细结果...")
            await detail_repo.create_detailed_result(
                task_id=task_id,
                backtest_id=f"bt_{task_id[:8]}",
                sortino_ratio=enhanced_result.extended_risk_metrics.sortino_ratio if enhanced_result.extended_risk_metrics else 0,
                calmar_ratio=enhanced_result.extended_risk_metrics.calmar_ratio if enhanced_result.extended_risk_metrics else 0,
                max_drawdown_duration=enhanced_result.extended_risk_metrics.max_drawdown_duration if enhanced_result.extended_risk_metrics else 0,
                var_95=enhanced_result.extended_risk_metrics.var_95 if enhanced_result.extended_risk_metrics else 0,
                downside_deviation=enhanced_result.extended_risk_metrics.downside_deviation if enhanced_result.extended_risk_metrics else 0,
                drawdown_analysis=enhanced_result.drawdown_analysis.to_dict() if enhanced_result.drawdown_analysis else {},
                monthly_returns=[mr.to_dict() for mr in enhanced_result.monthly_returns] if enhanced_result.monthly_returns else [],
                position_analysis=[pa.to_dict() for pa in enhanced_result.position_analysis] if enhanced_result.position_analysis else [],
                benchmark_comparison=enhanced_result.benchmark_data or {},
                rolling_metrics={}
            )
            
            # 保存组合快照
            if enhanced_result.portfolio_history:
                logger.info(f"保存 {len(enhanced_result.portfolio_history)} 个组合快照...")
                for snapshot in enhanced_result.portfolio_history:
                    await detail_repo.create_portfolio_snapshot(
                        task_id=task_id,
                        backtest_id=f"bt_{task_id[:8]}",
                        snapshot_date=snapshot.get("date"),
                        portfolio_value=snapshot.get("portfolio_value", 0),
                        cash=snapshot.get("cash", 0),
                        positions_count=snapshot.get("positions_count", 0),
                        total_return=snapshot.get("total_return", 0),
                        drawdown=0,
                        positions=snapshot.get("positions", {})
                    )
            
            # 保存交易记录
            if enhanced_result.trade_history:
                logger.info(f"保存 {len(enhanced_result.trade_history)} 条交易记录...")
                for trade in enhanced_result.trade_history:
                    await detail_repo.create_trade_record(
                        task_id=task_id,
                        backtest_id=f"bt_{task_id[:8]}",
                        trade_id=trade.get("trade_id", ""),
                        stock_code=trade.get("stock_code", ""),
                        stock_name=trade.get("stock_code", ""),
                        action=trade.get("action", ""),
                        quantity=trade.get("quantity", 0),
                        price=trade.get("price", 0),
                        timestamp=trade.get("timestamp"),
                        commission=trade.get("commission", 0),
                        pnl=trade.get("pnl", 0),
                        holding_days=trade.get("holding_days", 0),
                        technical_indicators={}
                    )
            
            await session.commit()
            logger.info(f"✅ 任务详细数据填充成功: {task_id}")
            return True
            
        except Exception as e:
            await session.rollback()
            logger.error(f"❌ 填充详细数据失败: {task_id}, 错误: {e}", exc_info=True)
            return False
        finally:
            break


async def fill_all_completed_tasks():
    """为所有已完成的回测任务填充详细数据"""
    logger.info("=" * 60)
    logger.info("开始填充所有已完成回测任务的详细数据")
    logger.info("=" * 60)
    
    async for session in get_async_session():
        try:
            # 获取所有已完成的回测任务
            task_repo = TaskRepository(session)
            
            # 使用原始SQL查询
            from sqlalchemy import text
            query = text("""
                SELECT task_id, task_name, completed_at
                FROM tasks
                WHERE status = 'completed'
                AND task_type = 'backtest'
                AND result IS NOT NULL
                ORDER BY completed_at DESC
            """)
            
            result = await session.execute(query)
            tasks = result.fetchall()
            
            logger.info(f"找到 {len(tasks)} 个已完成的回测任务")
            
            success_count = 0
            skip_count = 0
            fail_count = 0
            
            for task in tasks:
                task_id = task.task_id
                logger.info(f"\n处理任务: {task_id} ({task.task_name})")
                
                result = await fill_task_details(task_id)
                if result:
                    success_count += 1
                elif result is None:
                    skip_count += 1
                else:
                    fail_count += 1
            
            logger.info("\n" + "=" * 60)
            logger.info("填充完成统计:")
            logger.info(f"  总任务数: {len(tasks)}")
            logger.info(f"  成功: {success_count}")
            logger.info(f"  跳过: {skip_count}")
            logger.info(f"  失败: {fail_count}")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"查询任务失败: {e}", exc_info=True)
        finally:
            break


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # 填充指定任务
        task_id = sys.argv[1]
        asyncio.run(fill_task_details(task_id))
    else:
        # 填充所有任务
        asyncio.run(fill_all_completed_tasks())
