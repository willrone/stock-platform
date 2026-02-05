#!/usr/bin/env python3
"""
迁移脚本：为旧任务创建 backtest_detailed_results 记录
使用 BacktestDataAdapter 从 tasks.result 生成完整的详细分析数据
"""

import asyncio
import json
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from app.core.config import settings


async def migrate_detailed_result(task_id: str):
    """为指定任务创建详细结果记录"""
    
    # 创建数据库连接
    engine = create_async_engine(
        settings.DATABASE_URL.replace("sqlite:///", "sqlite+aiosqlite:///"),
        echo=False
    )
    
    async_session = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    
    async with async_session() as session:
        # 查询任务结果
        query = text("""
            SELECT task_id, result 
            FROM tasks 
            WHERE task_id = :task_id 
            AND task_type = 'backtest'
            AND result IS NOT NULL
        """)
        result = await session.execute(query, {"task_id": task_id})
        task_row = result.fetchone()
        
        if not task_row:
            print(f"未找到任务: {task_id}")
            return
        
        result_json = task_row[1]
        if not result_json:
            print(f"任务没有结果数据: {task_id}")
            return
        
        try:
            backtest_report = json.loads(result_json)
        except json.JSONDecodeError:
            print(f"JSON 解析失败: {task_id}")
            return
        
        # 检查是否已有详细结果
        check_query = text("""
            SELECT COUNT(*) FROM backtest_detailed_results WHERE task_id = :task_id
        """)
        check_result = await session.execute(check_query, {"task_id": task_id})
        existing_count = check_result.scalar()
        
        if existing_count > 0:
            print(f"任务已有详细结果记录，跳过: {task_id}")
            return
        
        print(f"开始为任务 {task_id} 生成详细结果...")
        print(f"  trade_history: {len(backtest_report.get('trade_history', []))} 条")
        print(f"  portfolio_history: {len(backtest_report.get('portfolio_history', []))} 条")
        
        # 使用 BacktestDataAdapter 转换数据
        from app.services.backtest.utils import BacktestDataAdapter
        from app.services.backtest.models import EnhancedPositionAnalysis
        from app.repositories.backtest_detailed_repository import BacktestDetailedRepository
        
        adapter = BacktestDataAdapter()
        enhanced_result = await adapter.adapt_backtest_result(backtest_report)
        
        print(f"  转换完成:")
        print(f"    - extended_risk_metrics: {enhanced_result.extended_risk_metrics is not None}")
        print(f"    - monthly_returns: {len(enhanced_result.monthly_returns) if enhanced_result.monthly_returns else 0} 条")
        print(f"    - position_analysis: {enhanced_result.position_analysis is not None}")
        print(f"    - drawdown_analysis: {enhanced_result.drawdown_analysis is not None}")
        
        # 辅助函数：将numpy类型转换为Python原生类型
        def to_python_type(value):
            """将numpy/pandas类型转换为Python原生类型"""
            import numpy as np
            from datetime import datetime
            
            if isinstance(value, (np.integer, np.floating)):
                return value.item()
            elif isinstance(value, np.ndarray):
                return value.tolist()
            elif hasattr(value, 'to_pydatetime'):
                return value.to_pydatetime().isoformat()
            elif isinstance(value, datetime):
                return value.isoformat()
            elif isinstance(value, dict):
                return {to_python_type(k): to_python_type(v) for k, v in value.items()}
            elif isinstance(value, (list, tuple)):
                return [to_python_type(v) for v in value]
            return value
        
        # 准备扩展指标数据
        extended_metrics = {}
        if enhanced_result.extended_risk_metrics:
            extended_metrics = {
                "sortino_ratio": to_python_type(enhanced_result.extended_risk_metrics.sortino_ratio),
                "calmar_ratio": to_python_type(enhanced_result.extended_risk_metrics.calmar_ratio),
                "max_drawdown_duration": to_python_type(enhanced_result.extended_risk_metrics.max_drawdown_duration),
                "var_95": to_python_type(enhanced_result.extended_risk_metrics.var_95),
                "downside_deviation": to_python_type(enhanced_result.extended_risk_metrics.downside_deviation),
            }
        
        # 处理 position_analysis
        position_analysis_data = None
        if enhanced_result.position_analysis:
            if isinstance(enhanced_result.position_analysis, EnhancedPositionAnalysis):
                position_analysis_data = enhanced_result.position_analysis.to_dict()
            elif isinstance(enhanced_result.position_analysis, list):
                position_analysis_data = [pa.to_dict() for pa in enhanced_result.position_analysis]
            else:
                position_analysis_data = enhanced_result.position_analysis
        
        # 准备分析数据
        analysis_data = {
            "drawdown_analysis": to_python_type(enhanced_result.drawdown_analysis.to_dict())
            if enhanced_result.drawdown_analysis
            else {},
            "monthly_returns": to_python_type([
                mr.to_dict()
                for mr in enhanced_result.monthly_returns
            ])
            if enhanced_result.monthly_returns
            else [],
            "position_analysis": to_python_type(position_analysis_data) if position_analysis_data else None,
            "benchmark_comparison": to_python_type(enhanced_result.benchmark_data)
            if enhanced_result.benchmark_data
            else {},
            "rolling_metrics": {},
        }
        
        # 创建详细结果记录
        repository = BacktestDetailedRepository(session)
        await repository.create_detailed_result(
            task_id=task_id,
            backtest_id=f"bt_{task_id[:8]}",
            extended_metrics=extended_metrics,
            analysis_data=analysis_data,
        )
        
        # 批量创建组合快照记录（如果还没有）
        check_snapshots = text("""
            SELECT COUNT(*) FROM portfolio_snapshots WHERE task_id = :task_id
        """)
        snapshots_result = await session.execute(check_snapshots, {"task_id": task_id})
        snapshots_count = snapshots_result.scalar()
        
        if snapshots_count == 0:
            portfolio_history = enhanced_result.portfolio_history or []
            if portfolio_history:
                from datetime import datetime
                snapshots_data = []
                for snapshot in portfolio_history:
                    date_value = snapshot.get("date")
                    if isinstance(date_value, str):
                        try:
                            date_value = datetime.fromisoformat(date_value.replace("Z", "+00:00"))
                        except:
                            continue
                    
                    snapshots_data.append({
                        "date": date_value,
                        "portfolio_value": to_python_type(snapshot.get("portfolio_value", 0)),
                        "cash": to_python_type(snapshot.get("cash", 0)),
                        "positions_count": to_python_type(snapshot.get("positions_count", 0)),
                        "total_return": to_python_type(snapshot.get("total_return", 0)),
                        "drawdown": 0,
                        "positions": to_python_type(snapshot.get("positions", {})),
                    })
                
                if snapshots_data:
                    await repository.batch_create_portfolio_snapshots(
                        task_id=task_id,
                        backtest_id=f"bt_{task_id[:8]}",
                        snapshots_data=snapshots_data,
                    )
                    print(f"  ✓ 创建 {len(snapshots_data)} 个组合快照")
        else:
            print(f"  组合快照已存在: {snapshots_count} 条")
        
        await session.commit()
        print(f"\n✓ 详细结果创建成功: {task_id}")
    
    await engine.dispose()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python migrate_detailed_result.py <task_id>")
        sys.exit(1)
    
    task_id = sys.argv[1]
    print(f"为任务 {task_id} 创建详细结果...")
    asyncio.run(migrate_detailed_result(task_id))
