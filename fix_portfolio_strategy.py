#!/usr/bin/env python3
"""
修复协整策略在单股票回测中的问题

问题：协整策略需要配对交易，在单股票场景下无法生成有效信号
解决：移除协整策略，只使用布林带和RSI策略
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from app.database import SessionLocal
from app.models import Task
import json

def fix_portfolio_strategy():
    """修复portfolio策略配置"""
    db = SessionLocal()
    
    try:
        # 查找使用portfolio策略的任务
        tasks = db.query(Task).filter(
            Task.task_type == 'backtest',
            Task.status.in_(['created', 'pending'])
        ).all()
        
        fixed_count = 0
        
        for task in tasks:
            config = json.loads(task.config) if isinstance(task.config, str) else task.config
            
            # 检查是否使用portfolio策略且包含cointegration
            if config.get('strategy_name') == 'portfolio':
                strategy_config = config.get('strategy_config', {})
                strategies = strategy_config.get('strategies', [])
                
                # 检查是否包含cointegration
                has_cointegration = any(s.get('name') == 'cointegration' for s in strategies)
                
                if has_cointegration:
                    print(f"\n修复任务: {task.task_id} ({task.task_name})")
                    print(f"  原策略: {[s['name'] for s in strategies]}")
                    
                    # 移除cointegration策略
                    new_strategies = [s for s in strategies if s.get('name') != 'cointegration']
                    
                    # 重新归一化权重
                    total_weight = sum(s.get('weight', 1) for s in new_strategies)
                    for s in new_strategies:
                        s['weight'] = s.get('weight', 1) / total_weight
                    
                    strategy_config['strategies'] = new_strategies
                    config['strategy_config'] = strategy_config
                    
                    # 更新任务配置
                    task.config = json.dumps(config)
                    
                    print(f"  新策略: {[s['name'] for s in new_strategies]}")
                    print(f"  新权重: {[f\"{s['name']}={s['weight']:.2f}\" for s in new_strategies]}")
                    
                    fixed_count += 1
        
        if fixed_count > 0:
            db.commit()
            print(f"\n✅ 成功修复 {fixed_count} 个任务")
        else:
            print("\n✅ 没有需要修复的任务")
            
    except Exception as e:
        db.rollback()
        print(f"\n❌ 修复失败: {e}")
        raise
    finally:
        db.close()

def create_fixed_task_example():
    """创建一个修复后的示例任务配置"""
    config = {
        "stock_codes": ["000001.SZ", "000002.SZ"],  # 示例股票
        "strategy_name": "portfolio",
        "start_date": "2023-01-01",
        "end_date": "2026-02-05",
        "initial_cash": 100000,
        "commission_rate": 0.0003,
        "slippage_rate": 0.0001,
        "strategy_config": {
            "strategies": [
                {
                    "name": "bollinger",
                    "weight": 0.5,
                    "config": {
                        "period": 20,
                        "std_dev": 2,
                        "entry_threshold": 0.02
                    }
                },
                {
                    "name": "rsi",
                    "weight": 0.5,
                    "config": {
                        "rsi_period": 14,
                        "oversold_threshold": 30,
                        "overbought_threshold": 70,
                        "trend_ma_period": 50,
                        "enable_trend_alignment": True,
                        "enable_divergence": True,
                        "enable_crossover": True,
                        "uptrend_buy_threshold": 40,
                        "downtrend_sell_threshold": 60
                    }
                }
            ],
            "integration_method": "weighted_voting"
        },
        "enable_performance_profiling": False
    }
    
    print("\n📋 修复后的策略配置示例：")
    print(json.dumps(config, indent=2, ensure_ascii=False))
    
    return config

if __name__ == "__main__":
    print("=" * 60)
    print("Portfolio策略修复工具")
    print("=" * 60)
    
    # 显示示例配置
    create_fixed_task_example()
    
    # 修复现有任务
    print("\n" + "=" * 60)
    print("开始修复现有任务...")
    print("=" * 60)
    fix_portfolio_strategy()
