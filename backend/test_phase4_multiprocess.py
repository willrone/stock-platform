#!/usr/bin/env python3
"""
Phase 4 多进程回测性能测试

对比：
1. 单进程回测（基线）
2. 多进程回��（8核并行）

目标：500只股票×3年 < 180秒
"""

import sys
import time
from datetime import datetime
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from loguru import logger

from app.services.backtest.execution.multiprocess_backtest import run_multiprocess_backtest
from app.services.backtest.models import BacktestConfig


def load_baseline_stocks():
    """加载基线任务的股票列表"""
    import sqlite3
    import json
    
    db_path = Path(__file__).parent / "data" / "app.db"
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT config FROM tasks 
        WHERE task_id = '814287d1-202c-4109-a746-c932206bd840'
    """)
    
    row = cursor.fetchone()
    conn.close()
    
    if row:
        config = json.loads(row[0])
        return config.get('stock_codes', [])
    
    return None


def main():
    logger.info("=" * 80)
    logger.info("Phase 4 多进程回测性能测试")
    logger.info("=" * 80)
    
    # 1. 加载基线任务的股票列表
    logger.info("📊 加载基线任务股票列表...")
    stock_codes = load_baseline_stocks()
    
    if not stock_codes:
        logger.error("❌ 无法加载基线任务股票列表")
        return
    
    logger.info(f"✅ 加载完成: {len(stock_codes)} 只股票")
    
    # 2. 配置参数
    start_date = datetime(2023, 2, 4)
    end_date = datetime(2026, 2, 4)
    
    strategy_name = "RSI"
    strategy_config = {
        "rsi_period": 14,
        "oversold": 30,
        "overbought": 70,
    }
    
    backtest_config = BacktestConfig(
        initial_cash=1000000.0,
        commission_rate=0.0003,
    )
    
    # 3. 执行多进程回测
    logger.info("")
    logger.info("🚀 开始多进程回测...")
    logger.info(f"  策略: {strategy_name}")
    logger.info(f"  股票数: {len(stock_codes)}")
    logger.info(f"  日期: {start_date.date()} ~ {end_date.date()}")
    logger.info("")
    
    # 使用绝对路径
    import os
    project_root = "/Users/ronghui/Projects/willrone"
    data_dir = os.path.join(project_root, "data")
    logger.info(f"  数据目录: {data_dir}")
    
    start_time = time.perf_counter()
    
    try:
        # 传递绝对路径的 data_dir
        result = run_multiprocess_backtest(
            strategy_name=strategy_name,
            stock_codes=stock_codes,
            start_date=start_date,
            end_date=end_date,
            strategy_config=strategy_config,
            backtest_config=backtest_config,
            num_workers=8,  # 使用 8 个进程
            data_dir=data_dir,  # 传递数据目录
        )
        
        total_time = time.perf_counter() - start_time
        
        # 4. 输出结果
        logger.info("")
        logger.info("=" * 80)
        logger.info("📊 回测结果")
        logger.info("=" * 80)
        
        perf = result.get('performance_metrics', {})
        breakdown = result.get('perf_breakdown', {})
        
        logger.info(f"✅ 回测完成!")
        logger.info(f"")
        logger.info(f"⏱️  性能指标:")
        logger.info(f"  总耗时: {total_time:.2f} 秒")
        logger.info(f"  数据加载: {breakdown.get('data_loading_s', 0):.2f} 秒")
        logger.info(f"  序列化: {breakdown.get('serialize_s', 0):.2f} 秒")
        logger.info(f"  多进程执行: {breakdown.get('multiprocess_s', 0):.2f} 秒")
        logger.info(f"  结果合并: {breakdown.get('merge_s', 0):.2f} 秒")
        logger.info(f"")
        logger.info(f"📈 回测统计:")
        logger.info(f"  交易日数: {result.get('trading_days', 0)}")
        logger.info(f"  信号数: {result.get('total_signals', 0)}")
        logger.info(f"  交易数: {result.get('executed_trades', 0)}")
        logger.info(f"")
        logger.info(f"💰 收益指标:")
        logger.info(f"  总收益率: {perf.get('total_return', 0):.2%}")
        logger.info(f"  年化收益率: {perf.get('annualized_return', 0):.2%}")
        logger.info(f"  夏普比率: {perf.get('sharpe_ratio', 0):.2f}")
        logger.info(f"  最大回撤: {perf.get('max_drawdown', 0):.2%}")
        logger.info(f"")
        
        # 5. 对比基线
        baseline_time = 345.76
        target_time = 180.0
        
        speedup = baseline_time / total_time
        vs_target = total_time / target_time
        
        logger.info(f"🎯 性能对比:")
        logger.info(f"  基线耗时: {baseline_time:.2f} 秒 (Phase 3)")
        logger.info(f"  当前耗时: {total_time:.2f} 秒")
        logger.info(f"  加速比: {speedup:.2f}x")
        logger.info(f"")
        logger.info(f"  目标耗时: {target_time:.2f} 秒")
        logger.info(f"  距离目标: {vs_target:.2f}x")
        
        if total_time < target_time:
            logger.info(f"  ✅ 已达成目标! 提前 {target_time - total_time:.2f} 秒")
        else:
            logger.info(f"  ⚠️  未达成目标，还需提速 {vs_target:.2f}x")
        
        logger.info("")
        logger.info("=" * 80)
        
        # 6. 保存结果
        import json
        output_file = Path(__file__).parent / "PHASE4_MULTIPROCESS_RESULT.json"
        with open(output_file, 'w') as f:
            # 简化结果（去掉大数组）
            simplified_result = {
                'strategy_name': result['strategy_name'],
                'stock_count': len(result['stock_codes']),
                'start_date': result['start_date'],
                'end_date': result['end_date'],
                'total_signals': result['total_signals'],
                'executed_trades': result['executed_trades'],
                'trading_days': result['trading_days'],
                'performance_metrics': result['performance_metrics'],
                'perf_breakdown': result['perf_breakdown'],
                'num_workers': result['num_workers'],
                'speedup': speedup,
                'vs_target': vs_target,
                'achieved_target': total_time < target_time,
            }
            json.dump(simplified_result, f, indent=2)
        
        logger.info(f"💾 结果已保存到: {output_file}")
        
    except Exception as e:
        logger.error(f"❌ 回测执行失败: {e}", exc_info=True)
        return


if __name__ == "__main__":
    main()
