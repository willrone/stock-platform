#!/usr/bin/env python3
"""
为已完成的回测任务填充详细数据
"""

import sys
import asyncio
from pathlib import Path

# 添加backend到路径
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from app.core.database import get_async_session, SessionLocal
from app.repositories.task_repository import TaskRepository
from app.repositories.backtest_detailed_repository import BacktestDetailedRepository
from app.services.backtest.backtest_data_adapter import BacktestDataAdapter
from loguru import logger


async def fill_task_details(task_id: str):
    """为单个任务填充详细数据"""
    logger.info(f"开始处理任务: {task_id}")
    
    # 使用同步会话获取任务信息
    sync_session = SessionLocal()
    try:
        # 获取任务（同步）
        task_repo = TaskRepository(sync_session)
        task = task_repo.get_task_by_id(task_id)
        
        if not task:
            logger.error(f"任务不存在: {task_id}")
            return False
        
        if task.status != "completed":
            logger.warning(f"任务未完成: {task_id}, 状态: {task.status}")
            return False
        
        if not task.result:
            logger.warning(f"任务没有结果数据: {task_id}")
            return False
        
        # 获取任务结果数据
        task_result = task.result
    finally:
        sync_session.close()
    
    # 使用异步会话保存详细数据
    async for session in get_async_session():
        try:
            # 检查是否已有详细数据
            detail_repo = BacktestDetailedRepository(session)
            existing = await detail_repo.get_detailed_result_by_task_id(task_id)
            if existing:
                logger.info(f"任务已有详细数据，跳过: {task_id}")
                return True
            
            # 转换数据
            logger.info(f"开始转换数据...")
            adapter = BacktestDataAdapter()
            enhanced_result = await adapter.adapt_backtest_result(task_result)
            
            # 准备扩展指标数据
            extended_metrics = {}
            if enhanced_result.extended_risk_metrics:
                extended_metrics = {
                    'sortino_ratio': enhanced_result.extended_risk_metrics.sortino_ratio,
                    'calmar_ratio': enhanced_result.extended_risk_metrics.calmar_ratio,
                    'max_drawdown_duration': enhanced_result.extended_risk_metrics.max_drawdown_duration,
                    'var_95': enhanced_result.extended_risk_metrics.var_95,
                    'downside_deviation': enhanced_result.extended_risk_metrics.downside_deviation,
                }
            
            # 准备分析数据
            analysis_data = {
                'drawdown_analysis': enhanced_result.drawdown_analysis.to_dict() if enhanced_result.drawdown_analysis else {},
                'monthly_returns': [mr.to_dict() for mr in enhanced_result.monthly_returns] if enhanced_result.monthly_returns else [],
                'position_analysis': [pa.to_dict() for pa in enhanced_result.position_analysis] if enhanced_result.position_analysis else [],
                'benchmark_comparison': enhanced_result.benchmark_data or {},
                'rolling_metrics': {}
            }
            
            # 保存详细结果
            logger.info(f"保存详细结果...")
            await detail_repo.create_detailed_result(
                task_id=task_id,
                backtest_id=f"bt_{task_id[:8]}",
                extended_metrics=extended_metrics,
                analysis_data=analysis_data
            )
            
            # 批量保存组合快照
            if enhanced_result.portfolio_history:
                logger.info(f"保存 {len(enhanced_result.portfolio_history)} 个组合快照...")
                snapshots_data = []
                for snapshot in enhanced_result.portfolio_history:
                    snapshots_data.append({
                        'date': snapshot.get("date"),
                        'portfolio_value': snapshot.get("portfolio_value", 0),
                        'cash': snapshot.get("cash", 0),
                        'positions_count': snapshot.get("positions_count", 0),
                        'total_return': snapshot.get("total_return", 0),
                        'drawdown': 0,
                        'positions': snapshot.get("positions", {})
                    })
                
                if snapshots_data:
                    await detail_repo.batch_create_portfolio_snapshots(
                        task_id=task_id,
                        backtest_id=f"bt_{task_id[:8]}",
                        snapshots_data=snapshots_data
                    )
            
            # 批量保存交易记录
            if enhanced_result.trade_history:
                logger.info(f"保存 {len(enhanced_result.trade_history)} 条交易记录...")
                trades_data = []
                for trade in enhanced_result.trade_history:
                    trades_data.append({
                        'trade_id': trade.get("trade_id", ""),
                        'stock_code': trade.get("stock_code", ""),
                        'stock_name': trade.get("stock_code", ""),
                        'action': trade.get("action", ""),
                        'quantity': trade.get("quantity", 0),
                        'price': trade.get("price", 0),
                        'timestamp': trade.get("timestamp"),
                        'commission': trade.get("commission", 0),
                        'pnl': trade.get("pnl", 0),
                        'holding_days': trade.get("holding_days", 0),
                        'technical_indicators': {}
                    })
                
                if trades_data:
                    await detail_repo.batch_create_trade_records(
                        task_id=task_id,
                        backtest_id=f"bt_{task_id[:8]}",
                        trades_data=trades_data
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
    
    # 使用同步会话查询任务列表
    sync_session = SessionLocal()
    try:
        from sqlalchemy import text
        query = text("""
            SELECT task_id, task_name, completed_at
            FROM tasks
            WHERE status = 'completed'
            AND task_type = 'backtest'
            AND result IS NOT NULL
            ORDER BY completed_at DESC
        """)
        
        result = sync_session.execute(query)
        tasks = result.fetchall()
        
        logger.info(f"找到 {len(tasks)} 个已完成的回测任务")
    finally:
        sync_session.close()
    
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


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # 填充指定任务
        task_id = sys.argv[1]
        asyncio.run(fill_task_details(task_id))
    else:
        # 填充所有任务
        asyncio.run(fill_all_completed_tasks())
