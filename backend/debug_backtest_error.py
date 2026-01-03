#!/usr/bin/env python3
"""
调试回测任务失败的脚本
"""

import sys
import os
import traceback
from datetime import datetime

def test_backtest_execution():
    """测试回测执行过程"""
    try:
        from app.services.backtest.backtest_executor import BacktestExecutor
        from app.services.backtest.backtest_engine import BacktestConfig
        
        print("=== 测试回测执行 ===")
        
        # 使用失败任务的配置
        config = {
            "stock_codes": ["688807.SH", "002082.SZ", "001322.SZ"],  # 减少股票数量用于测试
            "strategy_name": "rsi",
            "start_date": "2021-01-01",
            "end_date": "2021-12-31",  # 缩短时间范围
            "initial_cash": 100000,
            "commission_rate": 0.0003,
            "slippage_rate": 0.0001
        }
        
        print(f"配置: {config}")
        
        # 解析配置
        stock_codes = config.get('stock_codes', [])
        strategy_name = config.get('strategy_name', 'rsi')
        start_date_str = config.get('start_date')
        end_date_str = config.get('end_date')
        initial_cash = config.get('initial_cash', 100000.0)
        
        print(f"股票代码: {stock_codes}")
        print(f"策略名称: {strategy_name}")
        print(f"开始日期: {start_date_str}")
        print(f"结束日期: {end_date_str}")
        
        # 解析日期
        start_date = datetime.fromisoformat(start_date_str) if isinstance(start_date_str, str) else start_date_str
        end_date = datetime.fromisoformat(end_date_str) if isinstance(end_date_str, str) else end_date_str
        
        print(f"解析后开始日期: {start_date}")
        print(f"解析后结束日期: {end_date}")
        
        # 创建回测执行器
        executor = BacktestExecutor(data_dir="data")
        print("回测执行器创建成功")
        
        # 创建回测配置
        backtest_config = BacktestConfig(
            initial_cash=initial_cash,
            commission_rate=config.get('commission_rate', 0.0003),
            slippage_rate=config.get('slippage_rate', 0.0001)
        )
        print(f"回测配置创建成功: {backtest_config}")
        
        # 验证参数
        print("开始参数验证...")
        executor.validate_backtest_parameters(
            strategy_name=strategy_name,
            stock_codes=stock_codes,
            start_date=start_date,
            end_date=end_date,
            strategy_config={}
        )
        print("参数验证通过")
        
        # 执行回测
        print("开始执行回测...")
        result = executor.run_backtest(
            strategy_name=strategy_name,
            stock_codes=stock_codes,
            start_date=start_date,
            end_date=end_date,
            strategy_config={},
            backtest_config=backtest_config
        )
        
        print("回测执行成功!")
        print(f"结果类型: {type(result)}")
        print(f"结果键: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
        
        return True
        
    except Exception as e:
        print(f"❌ 回测执行失败: {e}")
        print(f"错误类型: {type(e)}")
        print("完整错误堆栈:")
        traceback.print_exc()
        return False

def test_strategy_creation():
    """测试策略创建"""
    try:
        from app.services.backtest.backtest_engine import StrategyFactory
        
        print("\n=== 测试策略创建 ===")
        
        # 获取可用策略
        available_strategies = StrategyFactory.get_available_strategies()
        print(f"可用策略: {available_strategies}")
        
        # 创建RSI策略
        strategy = StrategyFactory.create_strategy('rsi', {})
        print(f"RSI策略创建成功: {strategy}")
        print(f"策略名称: {strategy.name}")
        print(f"策略配置: {strategy.config}")
        
        return True
        
    except Exception as e:
        print(f"❌ 策略创建失败: {e}")
        traceback.print_exc()
        return False

def test_data_loading():
    """测试数据加载"""
    try:
        from app.services.backtest.backtest_executor import DataLoader
        from datetime import datetime
        
        print("\n=== 测试数据加载 ===")
        
        loader = DataLoader(data_dir="data")
        print("数据加载器创建成功")
        
        # 测试加载单只股票
        stock_code = "688807.SH"
        start_date = datetime(2021, 1, 1)
        end_date = datetime(2021, 12, 31)
        
        print(f"尝试加载股票数据: {stock_code}")
        data = loader.load_stock_data(stock_code, start_date, end_date)
        
        print(f"数据加载成功: {len(data)} 行")
        print(f"数据列: {list(data.columns)}")
        print(f"数据索引类型: {type(data.index)}")
        print(f"数据属性: {data.attrs}")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        traceback.print_exc()
        return False

def main():
    print("开始调试回测任务失败问题...")
    
    # 测试各个组件
    tests = [
        ("策略创建", test_strategy_creation),
        ("数据加载", test_data_loading),
        ("回测执行", test_backtest_execution)
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"测试: {test_name}")
        print('='*50)
        
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"测试 {test_name} 出现未捕获异常: {e}")
            traceback.print_exc()
            results[test_name] = False
    
    # 总结结果
    print(f"\n{'='*50}")
    print("测试结果总结")
    print('='*50)
    
    for test_name, success in results.items():
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{test_name}: {status}")
    
    if all(results.values()):
        print("\n🎉 所有测试通过！问题可能在其他地方。")
    else:
        print("\n⚠️  发现问题，请查看上面的错误信息。")

if __name__ == "__main__":
    main()