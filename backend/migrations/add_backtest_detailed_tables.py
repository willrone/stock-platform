"""
数据库迁移脚本：添加回测详细结果相关表
用于支持回测结果可视化功能

执行方式：
python -m backend.migrations.add_backtest_detailed_tables
"""

import asyncio
from datetime import datetime
from typing import Dict
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from app.core.database import async_engine, Base, get_async_session
from app.models.backtest_detailed_models import (
    BacktestDetailedResult,
    BacktestChartCache,
    PortfolioSnapshot,
    TradeRecord,
    SignalRecord,
    BacktestBenchmark,
    BacktestStatistics
)


class BacktestDetailedTablesMigration:
    """回测详细表迁移类"""
    
    def __init__(self):
        self.logger = logger.bind(migration="backtest_detailed_tables")
    
    async def check_table_exists(self, session: AsyncSession, table_name: str) -> bool:
        """检查表是否存在"""
        try:
            # SQLite检查表是否存在的查询
            result = await session.execute(
                text("SELECT name FROM sqlite_master WHERE type='table' AND name=:table_name"),
                {"table_name": table_name}
            )
            return result.scalar() is not None
        except Exception as e:
            self.logger.error(f"检查表 {table_name} 是否存在时出错: {e}")
            return False
    
    async def create_backtest_detailed_result_table(self, session: AsyncSession) -> bool:
        """创建回测详细结果表"""
        try:
            if await self.check_table_exists(session, "backtest_detailed_results"):
                self.logger.info("表 backtest_detailed_results 已存在，跳过创建")
                return True
            
            create_sql = """
            CREATE TABLE backtest_detailed_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id VARCHAR(50) NOT NULL,
                backtest_id VARCHAR(50) NOT NULL,
                sortino_ratio REAL DEFAULT 0.0,
                calmar_ratio REAL DEFAULT 0.0,
                max_drawdown_duration INTEGER DEFAULT 0,
                var_95 REAL DEFAULT 0.0,
                downside_deviation REAL DEFAULT 0.0,
                drawdown_analysis JSON,
                monthly_returns JSON,
                position_analysis JSON,
                benchmark_comparison JSON,
                rolling_metrics JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
            
            await session.execute(text(create_sql))
            
            # 创建索引
            index_sqls = [
                "CREATE INDEX idx_backtest_detailed_task_id ON backtest_detailed_results(task_id);",
                "CREATE INDEX idx_backtest_detailed_backtest_id ON backtest_detailed_results(backtest_id);"
            ]
            
            for index_sql in index_sqls:
                await session.execute(text(index_sql))
            
            self.logger.info("成功创建表 backtest_detailed_results 及其索引")
            return True
            
        except Exception as e:
            self.logger.error(f"创建表 backtest_detailed_results 失败: {e}")
            return False
    
    async def create_backtest_chart_cache_table(self, session: AsyncSession) -> bool:
        """创建回测图表缓存表"""
        try:
            if await self.check_table_exists(session, "backtest_chart_cache"):
                self.logger.info("表 backtest_chart_cache 已存在，跳过创建")
                return True
            
            create_sql = """
            CREATE TABLE backtest_chart_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id VARCHAR(50) NOT NULL,
                chart_type VARCHAR(50) NOT NULL,
                chart_data JSON NOT NULL,
                data_hash VARCHAR(64),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP
            );
            """
            
            await session.execute(text(create_sql))
            
            # 创建索引
            index_sqls = [
                "CREATE UNIQUE INDEX uk_task_chart ON backtest_chart_cache(task_id, chart_type);",
                "CREATE INDEX idx_chart_cache_expires ON backtest_chart_cache(expires_at);",
                "CREATE INDEX idx_chart_cache_task_id ON backtest_chart_cache(task_id);"
            ]
            
            for index_sql in index_sqls:
                await session.execute(text(index_sql))
            
            self.logger.info("成功创建表 backtest_chart_cache 及其索引")
            return True
            
        except Exception as e:
            self.logger.error(f"创建表 backtest_chart_cache 失败: {e}")
            return False
    
    async def create_portfolio_snapshots_table(self, session: AsyncSession) -> bool:
        """创建组合快照表"""
        try:
            if await self.check_table_exists(session, "portfolio_snapshots"):
                self.logger.info("表 portfolio_snapshots 已存在，跳过创建")
                return True
            
            create_sql = """
            CREATE TABLE portfolio_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id VARCHAR(50) NOT NULL,
                backtest_id VARCHAR(50) NOT NULL,
                snapshot_date TIMESTAMP NOT NULL,
                portfolio_value REAL NOT NULL,
                cash REAL NOT NULL,
                positions_count INTEGER DEFAULT 0,
                total_return REAL DEFAULT 0.0,
                drawdown REAL DEFAULT 0.0,
                positions JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
            
            await session.execute(text(create_sql))
            
            # 创建索引
            index_sqls = [
                "CREATE INDEX idx_portfolio_task_date ON portfolio_snapshots(task_id, snapshot_date);",
                "CREATE INDEX idx_portfolio_backtest_date ON portfolio_snapshots(backtest_id, snapshot_date);"
            ]
            
            for index_sql in index_sqls:
                await session.execute(text(index_sql))
            
            self.logger.info("成功创建表 portfolio_snapshots 及其索引")
            return True
            
        except Exception as e:
            self.logger.error(f"创建表 portfolio_snapshots 失败: {e}")
            return False
    
    async def create_trade_records_table(self, session: AsyncSession) -> bool:
        """创建交易记录表"""
        try:
            if await self.check_table_exists(session, "trade_records"):
                self.logger.info("表 trade_records 已存在，跳过创建")
                return True
            
            create_sql = """
            CREATE TABLE trade_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id VARCHAR(50) NOT NULL,
                backtest_id VARCHAR(50) NOT NULL,
                trade_id VARCHAR(50) NOT NULL,
                stock_code VARCHAR(20) NOT NULL,
                stock_name VARCHAR(100),
                action VARCHAR(10) NOT NULL,
                quantity INTEGER NOT NULL,
                price REAL NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                commission REAL DEFAULT 0.0,
                pnl REAL,
                holding_days INTEGER,
                technical_indicators JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
            
            await session.execute(text(create_sql))
            
            # 创建索引
            index_sqls = [
                "CREATE INDEX idx_trade_task_stock ON trade_records(task_id, stock_code);",
                "CREATE INDEX idx_trade_backtest_time ON trade_records(backtest_id, timestamp);",
                "CREATE INDEX idx_trade_stock_time ON trade_records(stock_code, timestamp);"
            ]
            
            for index_sql in index_sqls:
                await session.execute(text(index_sql))
            
            self.logger.info("成功创建表 trade_records 及其索引")
            return True
            
        except Exception as e:
            self.logger.error(f"创建表 trade_records 失败: {e}")
            return False
    
    async def create_signal_records_table(self, session: AsyncSession) -> bool:
        """创建信号记录表"""
        try:
            if await self.check_table_exists(session, "signal_records"):
                self.logger.info("表 signal_records 已存在，跳过创建")
                return True
            
            create_sql = """
            CREATE TABLE signal_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id VARCHAR(50) NOT NULL,
                backtest_id VARCHAR(50) NOT NULL,
                signal_id VARCHAR(50) NOT NULL,
                stock_code VARCHAR(20) NOT NULL,
                stock_name VARCHAR(100),
                signal_type VARCHAR(10) NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                price REAL NOT NULL,
                strength REAL DEFAULT 0.0,
                reason TEXT,
                signal_metadata JSON,
                executed BOOLEAN DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
            
            await session.execute(text(create_sql))
            
            # 创建索引
            index_sqls = [
                "CREATE INDEX idx_signal_task_stock ON signal_records(task_id, stock_code);",
                "CREATE INDEX idx_signal_backtest_time ON signal_records(backtest_id, timestamp);",
                "CREATE INDEX idx_signal_stock_time ON signal_records(stock_code, timestamp);",
                "CREATE INDEX idx_signal_type ON signal_records(signal_type);",
                "CREATE INDEX idx_signal_executed ON signal_records(executed);"
            ]
            
            for index_sql in index_sqls:
                await session.execute(text(index_sql))
            
            self.logger.info("成功创建表 signal_records 及其索引")
            return True
            
        except Exception as e:
            self.logger.error(f"创建表 signal_records 失败: {e}")
            return False
    
    async def create_backtest_benchmarks_table(self, session: AsyncSession) -> bool:
        """创建回测基准表"""
        try:
            if await self.check_table_exists(session, "backtest_benchmarks"):
                self.logger.info("表 backtest_benchmarks 已存在，跳过创建")
                return True
            
            create_sql = """
            CREATE TABLE backtest_benchmarks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id VARCHAR(50) NOT NULL,
                backtest_id VARCHAR(50) NOT NULL,
                benchmark_symbol VARCHAR(20) NOT NULL,
                benchmark_name VARCHAR(100) NOT NULL,
                benchmark_data JSON NOT NULL,
                correlation REAL,
                beta REAL,
                alpha REAL,
                tracking_error REAL,
                information_ratio REAL,
                excess_return REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
            
            await session.execute(text(create_sql))
            
            # 创建索引
            index_sqls = [
                "CREATE INDEX idx_benchmark_task_symbol ON backtest_benchmarks(task_id, benchmark_symbol);",
                "CREATE INDEX idx_benchmark_backtest_id ON backtest_benchmarks(backtest_id);"
            ]
            
            for index_sql in index_sqls:
                await session.execute(text(index_sql))
            
            self.logger.info("成功创建表 backtest_benchmarks 及其索引")
            return True
            
        except Exception as e:
            self.logger.error(f"创建表 backtest_benchmarks 失败: {e}")
            return False
    
    async def create_backtest_statistics_table(self, session: AsyncSession) -> bool:
        """创建回测统计信息表"""
        try:
            if await self.check_table_exists(session, "backtest_statistics"):
                self.logger.info("表 backtest_statistics 已存在，跳过创建")
                return True
            
            create_sql = """
            CREATE TABLE backtest_statistics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id VARCHAR(50) NOT NULL UNIQUE,
                backtest_id VARCHAR(50) NOT NULL,
                total_signals INTEGER DEFAULT 0,
                buy_signals INTEGER DEFAULT 0,
                sell_signals INTEGER DEFAULT 0,
                executed_signals INTEGER DEFAULT 0,
                unexecuted_signals INTEGER DEFAULT 0,
                execution_rate REAL DEFAULT 0.0,
                avg_signal_strength REAL DEFAULT 0.0,
                total_trades INTEGER DEFAULT 0,
                buy_trades INTEGER DEFAULT 0,
                sell_trades INTEGER DEFAULT 0,
                winning_trades INTEGER DEFAULT 0,
                losing_trades INTEGER DEFAULT 0,
                win_rate REAL DEFAULT 0.0,
                avg_profit REAL DEFAULT 0.0,
                avg_loss REAL DEFAULT 0.0,
                profit_factor REAL DEFAULT 0.0,
                total_commission REAL DEFAULT 0.0,
                total_pnl REAL DEFAULT 0.0,
                avg_holding_days REAL DEFAULT 0.0,
                total_stocks INTEGER DEFAULT 0,
                profitable_stocks INTEGER DEFAULT 0,
                avg_stock_return REAL DEFAULT 0.0,
                max_stock_return REAL,
                min_stock_return REAL,
                first_signal_date TIMESTAMP,
                last_signal_date TIMESTAMP,
                first_trade_date TIMESTAMP,
                last_trade_date TIMESTAMP,
                trading_days INTEGER DEFAULT 0,
                unique_stocks_signaled INTEGER DEFAULT 0,
                unique_stocks_traded INTEGER DEFAULT 0,
                most_signaled_stock VARCHAR(20),
                most_traded_stock VARCHAR(20),
                max_single_profit REAL,
                max_single_loss REAL,
                max_consecutive_wins INTEGER DEFAULT 0,
                max_consecutive_losses INTEGER DEFAULT 0,
                largest_position_size REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
            
            await session.execute(text(create_sql))
            
            # 创建索引
            index_sqls = [
                "CREATE UNIQUE INDEX idx_statistics_task_id ON backtest_statistics(task_id);",
                "CREATE INDEX idx_statistics_backtest_id ON backtest_statistics(backtest_id);"
            ]
            
            for index_sql in index_sqls:
                await session.execute(text(index_sql))
            
            self.logger.info("成功创建表 backtest_statistics 及其索引")
            return True
            
        except Exception as e:
            self.logger.error(f"创建表 backtest_statistics 失败: {e}")
            return False
    
    async def run_migration(self) -> bool:
        """执行迁移"""
        self.logger.info("开始执行回测详细表迁移...")
        
        async for session in get_async_session():
            try:
                # 开始事务
                await session.begin()
                
                # 创建各个表
                tables_created = []
                
                # 1. 创建回测详细结果表
                if await self.create_backtest_detailed_result_table(session):
                    tables_created.append("backtest_detailed_results")
                
                # 2. 创建图表缓存表
                if await self.create_backtest_chart_cache_table(session):
                    tables_created.append("backtest_chart_cache")
                
                # 3. 创建组合快照表
                if await self.create_portfolio_snapshots_table(session):
                    tables_created.append("portfolio_snapshots")
                
                # 4. 创建交易记录表
                if await self.create_trade_records_table(session):
                    tables_created.append("trade_records")
                
                # 5. 创建信号记录表
                if await self.create_signal_records_table(session):
                    tables_created.append("signal_records")
                
                # 6. 创建基准对比表
                if await self.create_backtest_benchmarks_table(session):
                    tables_created.append("backtest_benchmarks")
                
                # 7. 创建统计信息表
                if await self.create_backtest_statistics_table(session):
                    tables_created.append("backtest_statistics")
                
                # 提交事务
                await session.commit()
                
                self.logger.info(f"迁移完成！成功创建/验证了以下表: {', '.join(tables_created)}")
                return True
                
            except Exception as e:
                await session.rollback()
                self.logger.error(f"迁移失败: {e}", exc_info=True)
                return False
    
    async def rollback_migration(self) -> bool:
        """回滚迁移（删除创建的表）"""
        self.logger.warning("开始回滚回测详细表迁移...")
        
        tables_to_drop = [
            "backtest_benchmarks",
            "signal_records",
            "trade_records", 
            "portfolio_snapshots",
            "backtest_chart_cache",
            "backtest_detailed_results"
        ]
        
        async for session in get_async_session():
            try:
                await session.begin()
                
                for table_name in tables_to_drop:
                    if await self.check_table_exists(session, table_name):
                        await session.execute(text(f"DROP TABLE {table_name}"))
                        self.logger.info(f"删除表: {table_name}")
                    else:
                        self.logger.info(f"表 {table_name} 不存在，跳过删除")
                
                await session.commit()
                self.logger.info("回滚完成！")
                return True
                
            except Exception as e:
                await session.rollback()
                self.logger.error(f"回滚失败: {e}", exc_info=True)
                return False
    
    async def verify_migration(self) -> Dict[str, bool]:
        """验证迁移结果"""
        self.logger.info("验证迁移结果...")
        
        tables_to_check = [
            "backtest_detailed_results",
            "backtest_chart_cache", 
            "portfolio_snapshots",
            "trade_records",
            "signal_records",
            "backtest_benchmarks"
        ]
        
        verification_results = {}
        
        async for session in get_async_session():
            for table_name in tables_to_check:
                exists = await self.check_table_exists(session, table_name)
                verification_results[table_name] = exists
                
                if exists:
                    self.logger.info(f"✓ 表 {table_name} 存在")
                else:
                    self.logger.error(f"✗ 表 {table_name} 不存在")
        
        return verification_results


async def main():
    """主函数"""
    migration = BacktestDetailedTablesMigration()
    
    # 执行迁移
    success = await migration.run_migration()
    
    if success:
        # 验证迁移结果
        verification_results = await migration.verify_migration()
        
        all_tables_exist = all(verification_results.values())
        if all_tables_exist:
            logger.info("🎉 所有表创建成功，迁移验证通过！")
        else:
            logger.error("❌ 部分表创建失败，请检查日志")
            return False
    else:
        logger.error("❌ 迁移失败")
        return False
    
    return True


if __name__ == "__main__":
    # 运行迁移
    result = asyncio.run(main())
    exit(0 if result else 1)