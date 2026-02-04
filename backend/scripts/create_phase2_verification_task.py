#!/usr/bin/env python3
"""
Phase 2 性能验证任务创建脚本
测试多进程优化效果
"""

import requests
import json
from datetime import datetime, timedelta

# API 配置
BASE_URL = "http://localhost:8000"
API_URL = f"{BASE_URL}/api/v1/tasks"

# 任务配置
task_config = {
    "task_name": "Phase2性能验证-多进程优化",
    "task_type": "backtest",
    "config": {
        "strategy_name": "rsi",
        "stock_codes": [],  # 将由后端自动选择500只股票
        "start_date": "2023-02-04",
        "end_date": "2026-02-04",
        "initial_cash": 1000000,
        "strategy_config": {
            "rsi_period": 14,
            "oversold": 30,
            "overbought": 70,
            "commission_rate": 0.0003,
            "slippage_rate": 0.0001
        },
        "enable_performance_profiling": True,  # 启用性能分析
        "stock_count": 500  # 自动选择500只股票
    }
}

def create_task():
    """创建回测任务"""
    print("=" * 60)
    print("Phase 2 性能验证任务创建")
    print("=" * 60)
    print(f"\n配置:")
    print(f"  策略: RSI (period=14, oversold=30, overbought=70)")
    print(f"  股票数: 500只（自动选择）")
    print(f"  时间范围: 2023-02-04 ~ 2026-02-04 (3年)")
    print(f"  初始资金: 1,000,000")
    print(f"  性能分析: 启用")
    print(f"  多进程: 启用 (use_multiprocessing=True, max_workers=6)")
    print(f"\n目标: < 180秒")
    print(f"预期: ~20秒 (基于历史17.44x加速)")
    print()
    
    try:
        response = requests.post(API_URL, json=task_config)
        response.raise_for_status()
        
        result = response.json()
        if result.get("success"):
            task_id = result["data"]["task_id"]
            print(f"✅ 任务创建成功!")
            print(f"   任务ID: {task_id}")
            print(f"   任务名称: {task_config['task_name']}")
            print(f"\n监控命令:")
            print(f"   curl -sS 'http://localhost:8000/api/v1/tasks/{task_id}' | python3 -m json.tool")
            print(f"\n等待任务完成...")
            return task_id
        else:
            print(f"❌ 任务创建失败: {result.get('message')}")
            return None
            
    except Exception as e:
        print(f"❌ 请求失败: {e}")
        return None

if __name__ == "__main__":
    task_id = create_task()
    if task_id:
        print(f"\n任务ID已保存，可用于后续查询")
        print(f"Task ID: {task_id}")
