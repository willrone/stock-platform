#!/usr/bin/env python3
"""快速填充单个任务的详细数据"""
import sys
sys.path.insert(0, 'backend')

import asyncio
from sqlalchemy import text
from app.core.database import get_async_session
from app.repositories.backtest_detailed_repository import BacktestDetailedRepository
from app.services.backtest.backtest_data_adapter import BacktestDataAdapter
import json

async def fill_task(task_id: str):
    print(f"填充任务: {task_id}")
    
    async for session in get_async_session():
        try:
            # 获取任务结果
            query = text("SELECT result FROM tasks WHERE task_id = :task_id")
            result = await session.execute(query, {"task_id": task_id})
            row = result.fetchone()
            
            if not row or not row.result:
                print("❌ 任务不存在或无结果")
                return
            
            backtest_result = json.loads(row.result)
            print(f"✓ 获取到回测结果")
            
            # 转换数据
            adapter = BacktestDataAdapter()
            enhanced = await adapter.adapt_backtest_result(backtest_result)
            print(f"✓ 数据转换完成")
            
            # 保存
            repo = BacktestDetailedRepository(session)
            
            # 详细结果
            await repo.create_detailed_result(
                task_id=task_id,
                backtest_id=f"bt_{task_id[:8]}",
                sortino_ratio=enhanced.extended_risk_metrics.sortino_ratio if enhanced.extended_risk_metrics else 0,
                calmar_ratio=enhanced.extended_risk_metrics.calmar_ratio if enhanced.extended_risk_metrics else 0,
                max_drawdown_duration=enhanced.extended_risk_metrics.max_drawdown_duration if enhanced.extended_risk_metrics else 0,
                var_95=enhanced.extended_risk_metrics.var_95 if enhanced.extended_risk_metrics else 0,
                downside_deviation=enhanced.extended_risk_metrics.downside_deviation if enhanced.extended_risk_metrics else 0,
                drawdown_analysis=enhanced.drawdown_analysis.to_dict() if enhanced.drawdown_analysis else {},
                monthly_returns=[mr.to_dict() for mr in enhanced.monthly_returns] if enhanced.monthly_returns else [],
                position_analysis=[pa.to_dict() for pa in enhanced.position_analysis] if enhanced.position_analysis else [],
                benchmark_comparison={},
                rolling_metrics={}
            )
            print(f"✓ 详细结果已保存")
            
            # 组合快照
            if enhanced.portfolio_history:
                for snap in enhanced.portfolio_history:
                    await repo.create_portfolio_snapshot(
                        task_id=task_id,
                        backtest_id=f"bt_{task_id[:8]}",
                        snapshot_date=snap.get("date"),
                        portfolio_value=snap.get("portfolio_value", 0),
                        cash=snap.get("cash", 0),
                        positions_count=snap.get("positions_count", 0),
                        total_return=snap.get("total_return", 0),
                        drawdown=0,
                        positions=snap.get("positions", {})
                    )
                print(f"✓ {len(enhanced.portfolio_history)} 个组合快照已保存")
            
            # 交易记录
            if enhanced.trade_history:
                for trade in enhanced.trade_history:
                    await repo.create_trade_record(
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
                        holding_days=0,
                        technical_indicators={}
                    )
                print(f"✓ {len(enhanced.trade_history)} 条交易记录已保存")
            
            await session.commit()
            print(f"✅ 任务 {task_id} 详细数据填充成功！")
            
        except Exception as e:
            await session.rollback()
            print(f"❌ 错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            break

if __name__ == "__main__":
    task_id = sys.argv[1] if len(sys.argv) > 1 else "f35381a7-5e5f-4e0e-8e0f-c8e8e8e8e8e8"
    asyncio.run(fill_task(task_id))
