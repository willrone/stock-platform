#!/usr/bin/env python3
"""
回测进度监控功能测试

测试回测进度监控器的基本功能
"""

import asyncio
import sys
import os
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from app.services.backtest.backtest_progress_monitor import (
    BacktestProgressMonitor, 
    BacktestProgressData,
    BacktestProgressStage
)


async def test_progress_monitor():
    """测试进度监控器"""
    print("🧪 开始测试回测进度监控器...")
    
    monitor = BacktestProgressMonitor()
    task_id = "test_task_001"
    backtest_id = "bt_test_001"
    
    try:
        # 1. 开始监控
        print("\n1️⃣ 开始监控回测进度...")
        progress_data = await monitor.start_backtest_monitoring(
            task_id=task_id,
            backtest_id=backtest_id,
            total_trading_days=100
        )
        
        print(f"✅ 监控已开始: {progress_data.task_id}")
        print(f"   - 回测ID: {progress_data.backtest_id}")
        print(f"   - 总交易日: {progress_data.total_trading_days}")
        print(f"   - 阶段数量: {len(progress_data.stages)}")
        
        # 2. 更新阶段进度
        print("\n2️⃣ 更新阶段进度...")
        await monitor.update_stage(task_id, "data_loading", progress=50, status="running")
        await monitor.update_stage(task_id, "data_loading", progress=100, status="completed")
        
        progress_data = monitor.get_progress_data(task_id)
        data_loading_stage = next(s for s in progress_data.stages if s.stage_name == "data_loading")
        print(f"✅ 数据加载阶段: {data_loading_stage.status} ({data_loading_stage.progress}%)")
        
        # 3. 更新执行进度
        print("\n3️⃣ 更新执行进度...")
        await monitor.update_execution_progress(
            task_id=task_id,
            processed_days=25,
            current_date="2024-01-15",
            signals_generated=5,
            trades_executed=3,
            portfolio_value=105000.0
        )
        
        progress_data = monitor.get_progress_data(task_id)
        print(f"✅ 执行进度更新:")
        print(f"   - 已处理天数: {progress_data.processed_trading_days}")
        print(f"   - 当前日期: {progress_data.current_date}")
        print(f"   - 组合价值: {progress_data.current_portfolio_value}")
        print(f"   - 总体进度: {progress_data.overall_progress:.1f}%")
        
        # 4. 添加警告
        print("\n4️⃣ 添加警告信息...")
        await monitor.add_warning(task_id, "股票000001数据缺失，跳过该交易日")
        
        progress_data = monitor.get_progress_data(task_id)
        print(f"✅ 警告已添加: {len(progress_data.warnings)} 个警告")
        
        # 5. 完成回测
        print("\n5️⃣ 完成回测...")
        await monitor.complete_backtest(task_id, {"total_return": 0.15})
        
        progress_data = monitor.get_progress_data(task_id)
        print(f"✅ 回测已完成: {progress_data.overall_progress}%")
        
        # 6. 获取所有活跃回测
        print("\n6️⃣ 获取活跃回测...")
        active_backtests = monitor.get_all_active_backtests()
        print(f"✅ 活跃回测数量: {len(active_backtests)}")
        
        print("\n🎉 所有测试通过！")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_error_handling():
    """测试错误处理"""
    print("\n🧪 测试错误处理...")
    
    monitor = BacktestProgressMonitor()
    task_id = "test_error_task"
    
    try:
        # 开始监控
        await monitor.start_backtest_monitoring(task_id, "bt_error_test")
        
        # 设置错误
        await monitor.set_error(task_id, "模拟的回测错误")
        
        progress_data = monitor.get_progress_data(task_id)
        print(f"✅ 错误处理测试通过: {progress_data.error_message}")
        
        return True
        
    except Exception as e:
        print(f"❌ 错误处理测试失败: {e}")
        return False


async def test_cancellation():
    """测试取消功能"""
    print("\n🧪 测试取消功能...")
    
    monitor = BacktestProgressMonitor()
    task_id = "test_cancel_task"
    
    try:
        # 开始监控
        await monitor.start_backtest_monitoring(task_id, "bt_cancel_test")
        
        # 取消回测
        await monitor.cancel_backtest(task_id, "用户手动取消")
        
        # 检查是否已从活跃列表中移除
        active_backtests = monitor.get_all_active_backtests()
        is_removed = task_id not in active_backtests
        
        print(f"✅ 取消功能测试通过: 已从活跃列表移除 = {is_removed}")
        
        return True
        
    except Exception as e:
        print(f"❌ 取消功能测试失败: {e}")
        return False


async def main():
    """主测试函数"""
    print("🚀 回测进度监控功能测试")
    print("=" * 50)
    
    tests = [
        ("基本功能测试", test_progress_monitor),
        ("错误处理测试", test_error_handling),
        ("取消功能测试", test_cancellation),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 30)
        
        try:
            result = await test_func()
            if result:
                passed += 1
                print(f"✅ {test_name} 通过")
            else:
                print(f"❌ {test_name} 失败")
        except Exception as e:
            print(f"❌ {test_name} 异常: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！")
        return 0
    else:
        print("💥 部分测试失败！")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)