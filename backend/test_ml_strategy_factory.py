#!/usr/bin/env python3
"""测试 ML 策略通过工厂创建和执行"""

import sys
from pathlib import Path

# 添加项目路径
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

import pandas as pd
from app.services.backtest.strategies.strategy_factory import StrategyFactory
from app.services.backtest.execution.data_loader import DataLoader

def main():
    print("=" * 60)
    print("测试 ML 策略工厂创建和执行")
    print("=" * 60)
    
    # 1. 检查策略是否注册
    print("\n1. 检查可用策略:")
    available = StrategyFactory.get_available_strategies()
    print(f"   可用策略: {available}")
    
    if "ml_ensemble_lgb_xgb_riskctl" in available:
        print("   ✅ ML 策略已注册")
    else:
        print("   ❌ ML 策略未注册")
        return
    
    # 2. 创建策略实例
    print("\n2. 创建策略实例:")
    config = {
        "prob_threshold": 0.45,
        "lgb_weight": 0.5,
        "xgb_weight": 0.5,
        "top_n": 5
    }
    
    try:
        strategy = StrategyFactory.create_strategy("ml_ensemble_lgb_xgb_riskctl", config)
        print(f"   ✅ 策略创建成功: {strategy.name}")
        print(f"   策略类型: {type(strategy).__name__}")
        print(f"   配置: prob_threshold={strategy.prob_threshold}")
    except Exception as e:
        print(f"   ❌ 策略创建失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 3. 加载数据
    print("\n3. 加载测试数据:")
    # DataLoader 需要的是项目根目录下的 data 目录
    data_dir = backend_dir.parent / "data"
    loader = DataLoader(str(data_dir))
    
    stock_code = "000001.SZ"
    start_date = "2024-01-01"
    end_date = "2024-06-30"
    
    try:
        data = loader.load_stock_data(stock_code, start_date, end_date)
        print(f"   ✅ 数据加载成功: {len(data)} 行")
        print(f"   日期范围: {data.index[0]} ~ {data.index[-1]}")
    except Exception as e:
        print(f"   ❌ 数据加载失败: {e}")
        return
    
    # 4. 测试 precompute_all_signals
    print("\n4. 测试预计算信号:")
    try:
        signals = strategy.precompute_all_signals(data)
        
        if signals is None:
            print("   ❌ precompute_all_signals 返回 None")
            return
        
        print(f"   ✅ 信号预计算成功")
        print(f"   信号类型: {type(signals)}")
        print(f"   信号长度: {len(signals)}")
        
        from app.services.backtest.models import SignalType
        buy_count = (signals == SignalType.BUY).sum()
        sell_count = (signals == SignalType.SELL).sum()
        none_count = signals.isna().sum()
        
        print(f"   BUY 信号: {buy_count}")
        print(f"   SELL 信号: {sell_count}")
        print(f"   None: {none_count}")
        
        if buy_count == 0 and sell_count == 0:
            print("   ⚠️  警告: 没有产生任何信号！")
        
    except Exception as e:
        print(f"   ❌ 预计算失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)

if __name__ == "__main__":
    main()
