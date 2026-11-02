#!/usr/bin/env python3
"""
Optuna 优化效率 POC 验证脚本

测试优化后的超参数搜索性能：
1. SQLite 持久化存储
2. HyperbandPruner 激进剪枝
3. 数据预加载缓存
4. 并行执行 (n_jobs)

运行方式：
    cd <项目根>/backend
    source venv/bin/activate
    python scripts/test_optuna_optimization.py
"""

import asyncio
import sys
import os
import time
from datetime import datetime, timedelta

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.backtest.optimization.strategy_hyperparameter_optimizer import (
    StrategyHyperparameterOptimizer,
)


async def run_optimization_test(n_jobs: int = 4, n_trials: int = 20):
    """运行优化测试"""
    
    print("=" * 60)
    print(f"Optuna 优化效率 POC 测试")
    print(f"并行进程数: {n_jobs}")
    print(f"试验次数: {n_trials}")
    print("=" * 60)
    
    # 创建优化器
    optimizer = StrategyHyperparameterOptimizer(
        n_jobs=n_jobs,
        use_persistent_storage=True
    )
    
    # 测试配置
    strategy_name = "ma_crossover"
    stock_codes = ["000001.SZ", "000002.SZ", "600000.SH"]  # 3只股票
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)  # 6个月数据
    
    # 参数空间
    param_space = {
        "short_window": {
            "type": "int",
            "low": 5,
            "high": 30,
            "enabled": True
        },
        "long_window": {
            "type": "int", 
            "low": 20,
            "high": 120,
            "enabled": True
        },
        "stop_loss": {
            "type": "float",
            "low": 0.02,
            "high": 0.10,
            "enabled": True
        },
        "take_profit": {
            "type": "float",
            "low": 0.05,
            "high": 0.30,
            "enabled": True
        }
    }
    
    # 目标配置
    objective_config = {
        "objective_metric": "sharpe",
        "direction": "maximize"
    }
    
    # 回测配置
    backtest_config = {
        "initial_cash": 100000.0,
        "commission_rate": 0.0003,
        "slippage_rate": 0.0001,
    }
    
    # 进度回调
    trial_times = []
    last_report_time = time.time()
    
    def progress_callback(trial_num, total_trials, params, score, report, **kwargs):
        nonlocal last_report_time
        current_time = time.time()
        trial_times.append(current_time)
        
        completed = kwargs.get('completed_trials', trial_num)
        pruned = kwargs.get('pruned_trials', 0)
        best_score = kwargs.get('best_score', score)
        
        # 每5个trial或每30秒报告一次
        if trial_num % 5 == 0 or (current_time - last_report_time) > 30:
            print(f"  Trial {trial_num}/{total_trials}: score={score:.4f if score else 'N/A'}, "
                  f"best={best_score:.4f if best_score else 'N/A'}, "
                  f"completed={completed}, pruned={pruned}")
            last_report_time = current_time
    
    # 运行优化
    print(f"\n开始优化 @ {datetime.now().strftime('%H:%M:%S')}")
    print("-" * 60)
    
    start_time = time.time()
    
    try:
        result = await optimizer.optimize_strategy_parameters(
            strategy_name=strategy_name,
            param_space=param_space,
            stock_codes=stock_codes,
            start_date=start_date,
            end_date=end_date,
            objective_config=objective_config,
            backtest_config=backtest_config,
            n_trials=n_trials,
            optimization_method="tpe",
            timeout=600,  # 10分钟超时
            progress_callback=progress_callback,
        )
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        print("-" * 60)
        print(f"优化完成 @ {datetime.now().strftime('%H:%M:%S')}")
        print("=" * 60)
        
        # 输出结果
        print(f"\n📊 性能统计:")
        print(f"  总耗时: {total_duration:.2f} 秒")
        print(f"  平均每 trial: {total_duration / n_trials:.2f} 秒")
        print(f"  并行效率: {n_jobs}x 理论加速")
        
        if trial_times and len(trial_times) > 1:
            # 计算实际吞吐量
            actual_throughput = len(trial_times) / total_duration
            print(f"  实际吞吐量: {actual_throughput:.2f} trials/秒")
        
        print(f"\n🏆 最优结果:")
        print(f"  最佳得分: {result.get('best_score', 'N/A')}")
        print(f"  最佳参数: {result.get('best_params', {})}")
        
        stats = result.get('statistics', {})
        print(f"\n📈 试验统计:")
        print(f"  完成: {stats.get('completed_trials', 'N/A')}")
        print(f"  剪枝: {stats.get('pruned_trials', 'N/A')}")
        print(f"  失败: {stats.get('failed_trials', 'N/A')}")
        
        # 检查是否使用了持久化存储
        storage_dir = optimizer._storage_dir
        print(f"\n💾 存储位置: {storage_dir}")
        
        return {
            "success": True,
            "duration_seconds": total_duration,
            "avg_trial_seconds": total_duration / n_trials,
            "best_score": result.get('best_score'),
            "best_params": result.get('best_params'),
            "statistics": stats
        }
        
    except Exception as e:
        end_time = time.time()
        print(f"\n❌ 优化失败: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "duration_seconds": end_time - start_time
        }


async def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Optuna 优化效率 POC 测试")
    parser.add_argument("--n-jobs", type=int, default=4, help="并行进程数")
    parser.add_argument("--n-trials", type=int, default=20, help="试验次数")
    parser.add_argument("--compare", action="store_true", help="对比串行 vs 并行")
    
    args = parser.parse_args()
    
    if args.compare:
        # 对比测试
        print("\n" + "=" * 60)
        print("对比测试: 串行 vs 并行")
        print("=" * 60)
        
        # 串行测试 (n_jobs=1)
        print("\n[1/2] 串行测试 (n_jobs=1)...")
        result_serial = await run_optimization_test(n_jobs=1, n_trials=args.n_trials)
        
        # 并行测试
        print(f"\n[2/2] 并行测试 (n_jobs={args.n_jobs})...")
        result_parallel = await run_optimization_test(n_jobs=args.n_jobs, n_trials=args.n_trials)
        
        # 对比结果
        print("\n" + "=" * 60)
        print("📊 对比结果")
        print("=" * 60)
        
        if result_serial["success"] and result_parallel["success"]:
            serial_time = result_serial["duration_seconds"]
            parallel_time = result_parallel["duration_seconds"]
            speedup = serial_time / parallel_time if parallel_time > 0 else 0
            
            print(f"  串行耗时: {serial_time:.2f} 秒")
            print(f"  并行耗时: {parallel_time:.2f} 秒")
            print(f"  实际加速比: {speedup:.2f}x")
            print(f"  理论加速比: {args.n_jobs}x")
            print(f"  并行效率: {(speedup / args.n_jobs) * 100:.1f}%")
    else:
        # 单次测试
        await run_optimization_test(n_jobs=args.n_jobs, n_trials=args.n_trials)


if __name__ == "__main__":
    asyncio.run(main())
