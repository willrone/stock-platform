#!/usr/bin/env python3
"""
填充回测详细数据脚本
将现有的回测任务结果转换为详细的回测数据并存储到数据库中
"""

import asyncio
import json
from typing import Dict, Any, List
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from app.core.database import get_async_session
from app.repositories.backtest_detailed_repository import BacktestDetailedRepository
from app.services.backtest.utils import BacktestDataAdapter


class BacktestDetailedDataPopulator:
    """回测详细数据填充器"""
    
    def __init__(self):
        self.adapter = BacktestDataAdapter()
        self.logger = logger.bind(module="backtest_data_populator")
    
    async def populate_all_backtest_tasks(self) -> Dict[str, Any]:
        """填充所有回测任务的详细数据"""
        self.logger.info("开始填充回测详细数据...")
        
        results = {
            "total_tasks": 0,
            "processed_tasks": 0,
            "failed_tasks": 0,
            "skipped_tasks": 0,
            "task_details": []
        }
        
        async for session in get_async_session():
            try:
                repository = BacktestDetailedRepository(session)
                
                # 获取所有已完成的回测任务
                backtest_tasks = await self._get_completed_backtest_tasks(session)
                results["total_tasks"] = len(backtest_tasks)
                
                self.logger.info(f"找到 {len(backtest_tasks)} 个已完成的回测任务")
                
                for task in backtest_tasks:
                    task_id = task.get("task_id")
                    task_name = task.get("task_name", "")
                    
                    try:
                        # 检查是否已经处理过
                        existing = await repository.get_detailed_result_by_task_id(task_id)
                        if existing:
                            self.logger.info(f"任务 {task_id} 已存在详细数据，跳过")
                            results["skipped_tasks"] += 1
                            results["task_details"].append({
                                "task_id": task_id,
                                "task_name": task_name,
                                "status": "skipped",
                                "message": "详细数据已存在"
                            })
                            continue
                        
                        # 处理任务数据
                        success = await self._process_task(session, task, repository)
                        
                        if success:
                            results["processed_tasks"] += 1
                            results["task_details"].append({
                                "task_id": task_id,
                                "task_name": task_name,
                                "status": "success",
                                "message": "详细数据创建成功"
                            })
                            self.logger.info(f"任务 {task_id} 处理成功")
                        else:
                            results["failed_tasks"] += 1
                            results["task_details"].append({
                                "task_id": task_id,
                                "task_name": task_name,
                                "status": "failed",
                                "message": "处理失败"
                            })
                            self.logger.error(f"任务 {task_id} 处理失败")
                    
                    except Exception as e:
                        results["failed_tasks"] += 1
                        results["task_details"].append({
                            "task_id": task_id,
                            "task_name": task_name,
                            "status": "error",
                            "message": str(e)
                        })
                        self.logger.error(f"处理任务 {task_id} 时出错: {e}", exc_info=True)
                
                # 提交所有更改
                await session.commit()
                
                self.logger.info(f"填充完成: 总计 {results['total_tasks']} 个任务, "
                               f"成功 {results['processed_tasks']} 个, "
                               f"失败 {results['failed_tasks']} 个, "
                               f"跳过 {results['skipped_tasks']} 个")
                
                return results
                
            except Exception as e:
                await session.rollback()
                self.logger.error(f"填充回测详细数据失败: {e}", exc_info=True)
                raise
    
    async def _get_completed_backtest_tasks(self, session: AsyncSession) -> List[Dict[str, Any]]:
        """获取所有已完成的回测任务"""
        try:
            # 使用原始SQL查询获取回测任务
            from sqlalchemy import text
            
            query = text("""
                SELECT task_id, task_name, status, result, created_at, completed_at
                FROM tasks 
                WHERE status = 'completed' 
                AND task_name IN ('back_test', 'backtest')
                AND result IS NOT NULL
                ORDER BY completed_at DESC
            """)
            
            result = await session.execute(query)
            rows = result.fetchall()
            
            tasks = []
            for row in rows:
                task_data = {
                    "task_id": row.task_id,
                    "task_name": row.task_name,
                    "status": row.status,
                    "results": json.loads(row.result) if row.result else {},
                    "created_at": row.created_at,
                    "completed_at": row.completed_at
                }
                tasks.append(task_data)
            
            return tasks
            
        except Exception as e:
            self.logger.error(f"获取回测任务失败: {e}", exc_info=True)
            return []
    
    async def _process_task(
        self, 
        session: AsyncSession, 
        task: Dict[str, Any], 
        repository: BacktestDetailedRepository
    ) -> bool:
        """处理单个回测任务"""
        try:
            task_id = task["task_id"]
            results = task.get("results", {})
            
            # 数据直接在results中，不是嵌套在backtest_results下
            if not results:
                self.logger.warning(f"任务 {task_id} 没有结果数据")
                return False
            
            # 使用适配器转换数据
            enhanced_result = await self.adapter.adapt_backtest_result(results)
            
            # 创建详细结果记录
            await self._create_detailed_result(session, task_id, enhanced_result, repository)
            
            # 创建组合快照记录
            await self._create_portfolio_snapshots(session, task_id, enhanced_result, repository)
            
            # 创建交易记录
            await self._create_trade_records(session, task_id, enhanced_result, repository)
            
            return True
            
        except Exception as e:
            self.logger.error(f"处理任务数据失败: {e}", exc_info=True)
            return False
    
    async def _create_detailed_result(
        self, 
        session: AsyncSession, 
        task_id: str, 
        enhanced_result, 
        repository: BacktestDetailedRepository
    ):
        """创建详细结果记录"""
        try:
            # 构建详细结果数据
            detailed_data = {
                "task_id": task_id,
                "backtest_id": f"bt_{task_id[:8]}",
                "sortino_ratio": enhanced_result.extended_risk_metrics.sortino_ratio if enhanced_result.extended_risk_metrics else 0,
                "calmar_ratio": enhanced_result.extended_risk_metrics.calmar_ratio if enhanced_result.extended_risk_metrics else 0,
                "max_drawdown_duration": enhanced_result.extended_risk_metrics.max_drawdown_duration if enhanced_result.extended_risk_metrics else 0,
                "var_95": enhanced_result.extended_risk_metrics.var_95 if enhanced_result.extended_risk_metrics else 0,
                "downside_deviation": enhanced_result.extended_risk_metrics.downside_deviation if enhanced_result.extended_risk_metrics else 0,
                "drawdown_analysis": enhanced_result.drawdown_analysis.to_dict() if enhanced_result.drawdown_analysis else {},
                "monthly_returns": [mr.to_dict() for mr in enhanced_result.monthly_returns] if enhanced_result.monthly_returns else [],
                "position_analysis": [pa.to_dict() for pa in enhanced_result.position_analysis] if enhanced_result.position_analysis else [],
                "benchmark_comparison": enhanced_result.benchmark_data or {},
                "rolling_metrics": enhanced_result.rolling_metrics or {}  # 滚动指标时间序列
            }
            
            # 使用原始SQL插入
            from sqlalchemy import text
            
            insert_query = text("""
                INSERT INTO backtest_detailed_results (
                    task_id, backtest_id, sortino_ratio, calmar_ratio, 
                    max_drawdown_duration, var_95, downside_deviation,
                    drawdown_analysis, monthly_returns, position_analysis,
                    benchmark_comparison, rolling_metrics, created_at, updated_at
                ) VALUES (
                    :task_id, :backtest_id, :sortino_ratio, :calmar_ratio,
                    :max_drawdown_duration, :var_95, :downside_deviation,
                    :drawdown_analysis, :monthly_returns, :position_analysis,
                    :benchmark_comparison, :rolling_metrics, :created_at, :updated_at
                )
            """)
            
            await session.execute(insert_query, {
                "task_id": detailed_data["task_id"],
                "backtest_id": detailed_data["backtest_id"],
                "sortino_ratio": detailed_data["sortino_ratio"],
                "calmar_ratio": detailed_data["calmar_ratio"],
                "max_drawdown_duration": detailed_data["max_drawdown_duration"],
                "var_95": detailed_data["var_95"],
                "downside_deviation": detailed_data["downside_deviation"],
                "drawdown_analysis": json.dumps(detailed_data["drawdown_analysis"]),
                "monthly_returns": json.dumps(detailed_data["monthly_returns"]),
                "position_analysis": json.dumps(detailed_data["position_analysis"]),
                "benchmark_comparison": json.dumps(detailed_data["benchmark_comparison"]),
                "rolling_metrics": json.dumps(detailed_data["rolling_metrics"]),
                "created_at": datetime.now(),
                "updated_at": datetime.now()
            })
            
            self.logger.debug(f"创建详细结果记录成功: {task_id}")
            
        except Exception as e:
            self.logger.error(f"创建详细结果记录失败: {e}", exc_info=True)
            raise
    
    async def _create_portfolio_snapshots(
        self, 
        session: AsyncSession, 
        task_id: str, 
        enhanced_result, 
        repository: BacktestDetailedRepository
    ):
        """创建组合快照记录"""
        try:
            if not enhanced_result.portfolio_history:
                return
            
            from sqlalchemy import text
            
            # 批量插入组合快照
            snapshots_data = []
            for snapshot in enhanced_result.portfolio_history:
                snapshot_data = {
                    "task_id": task_id,
                    "backtest_id": f"bt_{task_id[:8]}",
                    "snapshot_date": snapshot.get("date"),
                    "portfolio_value": snapshot.get("portfolio_value", 0),
                    "cash": snapshot.get("cash", 0),
                    "positions_count": snapshot.get("positions_count", 0),
                    "total_return": snapshot.get("total_return", 0),
                    "drawdown": 0,  # 可以后续计算
                    "positions": json.dumps(snapshot.get("positions", {})),
                    "created_at": datetime.now()
                }
                snapshots_data.append(snapshot_data)
            
            # 批量插入
            if snapshots_data:
                insert_query = text("""
                    INSERT INTO portfolio_snapshots (
                        task_id, backtest_id, snapshot_date, portfolio_value, cash,
                        positions_count, total_return, drawdown, positions, created_at
                    ) VALUES (
                        :task_id, :backtest_id, :snapshot_date, :portfolio_value, :cash,
                        :positions_count, :total_return, :drawdown, :positions, :created_at
                    )
                """)
                
                await session.execute(insert_query, snapshots_data)
                
                self.logger.debug(f"创建 {len(snapshots_data)} 个组合快照记录: {task_id}")
            
        except Exception as e:
            self.logger.error(f"创建组合快照记录失败: {e}", exc_info=True)
            raise
    
    async def _create_trade_records(
        self, 
        session: AsyncSession, 
        task_id: str, 
        enhanced_result, 
        repository: BacktestDetailedRepository
    ):
        """创建交易记录"""
        try:
            if not enhanced_result.trade_history:
                return
            
            from sqlalchemy import text
            
            # 批量插入交易记录
            trades_data = []
            for trade in enhanced_result.trade_history:
                trade_data = {
                    "task_id": task_id,
                    "backtest_id": f"bt_{task_id[:8]}",
                    "trade_id": trade.get("trade_id", ""),
                    "stock_code": trade.get("stock_code", ""),
                    "stock_name": trade.get("stock_code", ""),  # 使用股票代码作为名称
                    "action": trade.get("action", ""),
                    "quantity": trade.get("quantity", 0),
                    "price": trade.get("price", 0),
                    "timestamp": trade.get("timestamp"),
                    "commission": trade.get("commission", 0),
                    "pnl": trade.get("pnl", 0),
                    "holding_days": trade.get("holding_days", 0),
                    "technical_indicators": json.dumps({}),
                    "created_at": datetime.now()
                }
                trades_data.append(trade_data)
            
            # 批量插入
            if trades_data:
                insert_query = text("""
                    INSERT INTO trade_records (
                        task_id, backtest_id, trade_id, stock_code, stock_name,
                        action, quantity, price, timestamp, commission, pnl,
                        holding_days, technical_indicators, created_at
                    ) VALUES (
                        :task_id, :backtest_id, :trade_id, :stock_code, :stock_name,
                        :action, :quantity, :price, :timestamp, :commission, :pnl,
                        :holding_days, :technical_indicators, :created_at
                    )
                """)
                
                await session.execute(insert_query, trades_data)
                
                self.logger.debug(f"创建 {len(trades_data)} 个交易记录: {task_id}")
            
        except Exception as e:
            self.logger.error(f"创建交易记录失败: {e}", exc_info=True)
            raise


async def main():
    """主函数"""
    populator = BacktestDetailedDataPopulator()
    
    try:
        results = await populator.populate_all_backtest_tasks()
        
        print("\n" + "="*60)
        print("回测详细数据填充完成")
        print("="*60)
        print(f"总任务数: {results['total_tasks']}")
        print(f"成功处理: {results['processed_tasks']}")
        print(f"处理失败: {results['failed_tasks']}")
        print(f"跳过任务: {results['skipped_tasks']}")
        print("="*60)
        
        # 显示详细信息
        if results['task_details']:
            print("\n任务详情:")
            for detail in results['task_details']:
                status_icon = {
                    'success': '✓',
                    'failed': '✗',
                    'error': '✗',
                    'skipped': '-'
                }.get(detail['status'], '?')
                
                print(f"{status_icon} {detail['task_id'][:8]}... ({detail['task_name']}) - {detail['message']}")
        
        return results['failed_tasks'] == 0
        
    except Exception as e:
        logger.error(f"填充过程失败: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    # 运行填充脚本
    success = asyncio.run(main())
    exit(0 if success else 1)