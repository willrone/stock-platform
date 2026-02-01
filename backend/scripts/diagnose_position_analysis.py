"""
诊断持仓分析数据生成问题
检查数据库中存储的数据和生成流程
"""

import asyncio
import json
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select, text
from loguru import logger

from app.core.config import settings
from app.core.database import AsyncSessionLocal
from app.repositories.backtest_detailed_repository import BacktestDetailedRepository
from app.repositories.task_repository import TaskRepository
from app.services.backtest.utils import BacktestDataAdapter


async def diagnose_task(task_id: str):
    """诊断单个任务的持仓分析数据"""
    logger.info(f"开始诊断任务: {task_id}")
    
    # 创建数据库连接
    database_url = settings.DATABASE_URL.replace("sqlite://", "sqlite+aiosqlite://")
    engine = create_async_engine(database_url, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    async with async_session() as session:
        # 1. 检查回测详细结果
        repository = BacktestDetailedRepository(session)
        detailed_result = await repository.get_detailed_result_by_task_id(task_id)
        
        if not detailed_result:
            logger.warning(f"任务 {task_id} 没有回测详细结果记录")
            return
        
        logger.info(f"找到回测详细结果记录: task_id={task_id}")
        
        # 2. 检查 position_analysis 字段
        position_analysis = detailed_result.position_analysis
        logger.info(f"position_analysis 类型: {type(position_analysis)}")
        logger.info(f"position_analysis 值: {position_analysis}")
        
        if position_analysis is None:
            logger.warning("position_analysis 为 None")
        elif isinstance(position_analysis, dict):
            logger.info(f"position_analysis 是字典，键: {list(position_analysis.keys())}")
            if 'stock_performance' in position_analysis:
                stock_perf = position_analysis['stock_performance']
                logger.info(f"stock_performance 类型: {type(stock_perf)}")
                if isinstance(stock_perf, list):
                    logger.info(f"stock_performance 长度: {len(stock_perf)}")
                    if len(stock_perf) > 0:
                        logger.info(f"第一条数据示例: {json.dumps(stock_perf[0], indent=2, ensure_ascii=False)}")
                else:
                    logger.warning(f"stock_performance 不是列表: {type(stock_perf)}")
            else:
                logger.warning("position_analysis 中没有 stock_performance 字段")
        elif isinstance(position_analysis, list):
            logger.info(f"position_analysis 是列表，长度: {len(position_analysis)}")
        else:
            logger.warning(f"position_analysis 是未知类型: {type(position_analysis)}")
        
        # 3. 检查原始任务数据
        task_repo = TaskRepository(session)
        task = task_repo.get_task_by_id(task_id)
        
        if task and task.result:
            logger.info("找到原始任务数据")
            result = task.result
            if isinstance(result, str):
                result = json.loads(result)
            
            trade_history = result.get('trade_history', [])
            portfolio_history = result.get('portfolio_history', [])
            
            logger.info(f"原始 trade_history 长度: {len(trade_history)}")
            logger.info(f"原始 portfolio_history 长度: {len(portfolio_history)}")
            
            if len(trade_history) > 0:
                logger.info(f"第一条交易记录示例: {json.dumps(trade_history[0], indent=2, ensure_ascii=False)}")
            
            # 4. 尝试重新生成持仓分析数据
            logger.info("尝试重新生成持仓分析数据...")
            try:
                adapter = BacktestDataAdapter()
                enhanced_result = await adapter.adapt_backtest_result(result)
                
                if enhanced_result.position_analysis:
                    logger.info("成功生成持仓分析数据")
                    if isinstance(enhanced_result.position_analysis, dict):
                        pos_dict = enhanced_result.position_analysis.to_dict() if hasattr(enhanced_result.position_analysis, 'to_dict') else enhanced_result.position_analysis
                        stock_perf = pos_dict.get('stock_performance', [])
                        logger.info(f"生成的 stock_performance 长度: {len(stock_perf)}")
                    else:
                        logger.info(f"生成的 position_analysis 类型: {type(enhanced_result.position_analysis)}")
                else:
                    logger.warning("生成的 position_analysis 为 None")
            except Exception as e:
                logger.error(f"重新生成持仓分析数据失败: {e}", exc_info=True)
        else:
            logger.warning("未找到原始任务数据")


async def list_all_tasks():
    """列出所有回测任务"""
    from app.core.config import settings
    database_url = settings.DATABASE_URL.replace("sqlite://", "sqlite+aiosqlite://")
    engine = create_async_engine(database_url, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    async with async_session() as session:
        # 查询所有回测详细结果
        from app.models.backtest_detailed_models import BacktestDetailedResult
        stmt = select(BacktestDetailedResult).order_by(BacktestDetailedResult.created_at.desc()).limit(10)
        result = await session.execute(stmt)
        detailed_results = result.scalars().all()
        
        logger.info(f"找到 {len(detailed_results)} 条回测详细结果")
        for dr in detailed_results:
            pos_analysis = dr.position_analysis
            has_data = False
            if pos_analysis:
                if isinstance(pos_analysis, dict) and pos_analysis.get('stock_performance'):
                    has_data = len(pos_analysis.get('stock_performance', [])) > 0
                elif isinstance(pos_analysis, list):
                    has_data = len(pos_analysis) > 0
            
            logger.info(f"任务 {dr.task_id}: position_analysis={'有数据' if has_data else '无数据/None'}")
        
        return [dr.task_id for dr in detailed_results]


async def main():
    """主函数"""
    import sys
    
    if len(sys.argv) > 1:
        task_id = sys.argv[1]
        await diagnose_task(task_id)
    else:
        # 列出所有任务
        task_ids = await list_all_tasks()
        if task_ids:
            logger.info(f"\n请使用以下命令诊断特定任务:")
            logger.info(f"python diagnose_position_analysis.py {task_ids[0]}")


if __name__ == "__main__":
    asyncio.run(main())

