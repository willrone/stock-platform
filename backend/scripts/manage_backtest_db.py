#!/usr/bin/env python3
"""
回测数据库管理脚本
用于管理回测详细结果相关的数据库操作

使用方法:
python manage_backtest_db.py migrate          # 执行迁移
python manage_backtest_db.py rollback         # 回滚迁移
python manage_backtest_db.py verify           # 验证迁移
python manage_backtest_db.py cleanup          # 清理缓存
python manage_backtest_db.py stats            # 查看统计信息
"""

import asyncio
import sys
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from migrations.add_backtest_detailed_tables import BacktestDetailedTablesMigration
from app.services.backtest.cache_cleanup_service import cache_cleanup_service
from app.services.backtest.chart_cache_service import chart_cache_service
from loguru import logger


async def migrate():
    """执行数据库迁移"""
    print("🚀 开始执行回测详细表迁移...")
    migration = BacktestDetailedTablesMigration()
    
    success = await migration.run_migration()
    if success:
        print("✅ 迁移执行成功！")
        
        # 验证迁移结果
        verification_results = await migration.verify_migration()
        all_tables_exist = all(verification_results.values())
        
        if all_tables_exist:
            print("✅ 迁移验证通过！")
            return True
        else:
            print("❌ 迁移验证失败，部分表未创建成功")
            for table, exists in verification_results.items():
                status = "✅" if exists else "❌"
                print(f"  {status} {table}")
            return False
    else:
        print("❌ 迁移执行失败！")
        return False


async def rollback():
    """回滚数据库迁移"""
    print("⚠️  开始回滚回测详细表迁移...")
    
    # 确认操作
    confirm = input("这将删除所有回测详细数据表，确定要继续吗？(y/N): ")
    if confirm.lower() != 'y':
        print("❌ 操作已取消")
        return False
    
    migration = BacktestDetailedTablesMigration()
    success = await migration.rollback_migration()
    
    if success:
        print("✅ 回滚完成！")
        return True
    else:
        print("❌ 回滚失败！")
        return False


async def verify():
    """验证迁移状态"""
    print("🔍 验证回测详细表迁移状态...")
    
    migration = BacktestDetailedTablesMigration()
    verification_results = await migration.verify_migration()
    
    print("\n📊 表状态检查结果:")
    all_tables_exist = True
    
    for table, exists in verification_results.items():
        status = "✅ 存在" if exists else "❌ 不存在"
        print(f"  {table}: {status}")
        if not exists:
            all_tables_exist = False
    
    if all_tables_exist:
        print("\n✅ 所有表都已正确创建！")
        return True
    else:
        print("\n❌ 部分表缺失，可能需要重新执行迁移")
        return False


async def cleanup():
    """清理缓存和旧数据"""
    print("🧹 开始清理缓存和旧数据...")
    
    try:
        # 手动执行清理任务
        cleanup_results = await cache_cleanup_service.manual_cleanup()
        
        print("\n📊 清理结果:")
        print(f"  过期缓存清理: {cleanup_results.get('expired_cache_cleaned', 0)} 条")
        
        old_data_cleaned = cleanup_results.get('old_data_cleaned', {})
        if old_data_cleaned:
            print("  旧数据清理:")
            for table, count in old_data_cleaned.items():
                if count > 0:
                    print(f"    {table}: {count} 条")
        
        errors = cleanup_results.get('errors', [])
        if errors:
            print("  错误:")
            for error in errors:
                print(f"    ❌ {error}")
        
        if not errors:
            print("\n✅ 清理完成！")
            return True
        else:
            print("\n⚠️  清理完成，但有部分错误")
            return False
            
    except Exception as e:
        print(f"❌ 清理失败: {e}")
        return False


async def stats():
    """查看统计信息"""
    print("📊 获取回测数据库统计信息...")
    
    try:
        # 获取缓存统计
        cache_stats = await chart_cache_service.get_cache_statistics()
        
        print("\n📈 图表缓存统计:")
        print(f"  总缓存记录: {cache_stats.get('total_cache_records', 0)}")
        print(f"  活跃记录: {cache_stats.get('active_records', 0)}")
        print(f"  过期记录: {cache_stats.get('expired_records', 0)}")
        print(f"  默认过期时间: {cache_stats.get('default_expiry_hours', 0)} 小时")
        
        cache_by_type = cache_stats.get('cache_by_type', {})
        if cache_by_type:
            print("  按类型统计:")
            for chart_type, count in cache_by_type.items():
                if count > 0:
                    print(f"    {chart_type}: {count}")
        
        # 获取清理服务统计
        cleanup_stats = await cache_cleanup_service.get_cleanup_statistics()
        
        service_status = cleanup_stats.get('service_status', {})
        print(f"\n🔧 清理服务状态:")
        print(f"  运行状态: {'运行中' if service_status.get('is_running', False) else '已停止'}")
        print(f"  清理间隔: {service_status.get('cleanup_interval_hours', 0)} 小时")
        print(f"  数据保留: {service_status.get('data_retention_days', 0)} 天")
        
        print("\n✅ 统计信息获取完成！")
        return True
        
    except Exception as e:
        print(f"❌ 获取统计信息失败: {e}")
        return False


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="回测数据库管理工具")
    parser.add_argument(
        'command',
        choices=['migrate', 'rollback', 'verify', 'cleanup', 'stats'],
        help='要执行的命令'
    )
    
    args = parser.parse_args()
    
    print(f"🎯 执行命令: {args.command}")
    print("=" * 50)
    
    try:
        if args.command == 'migrate':
            success = await migrate()
        elif args.command == 'rollback':
            success = await rollback()
        elif args.command == 'verify':
            success = await verify()
        elif args.command == 'cleanup':
            success = await cleanup()
        elif args.command == 'stats':
            success = await stats()
        else:
            print(f"❌ 未知命令: {args.command}")
            success = False
        
        print("=" * 50)
        if success:
            print("🎉 操作完成！")
            sys.exit(0)
        else:
            print("💥 操作失败！")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️  操作被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"💥 操作异常: {e}")
        logger.error(f"管理脚本异常: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())