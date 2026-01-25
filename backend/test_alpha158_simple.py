#!/usr/bin/env python3
"""
简单测试Alpha158表达式引擎 - 只测试前10个因子
"""
import sys
import os
from pathlib import Path

backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

import pandas as pd
import numpy as np
from app.services.qlib.enhanced_qlib_provider import Alpha158Calculator

def main():
    print("初始化Alpha158计算器...")
    calculator = Alpha158Calculator()
    
    if len(calculator.alpha_fields) == 0:
        print("⚠️  Alpha158配置不可用")
        return
    
    print(f"找到 {len(calculator.alpha_fields)} 个因子表达式")
    
    # 创建测试数据
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    test_data = pd.DataFrame({
        '$open': np.random.uniform(10, 20, 100),
        '$high': np.random.uniform(20, 30, 100),
        '$low': np.random.uniform(5, 15, 100),
        '$close': np.random.uniform(10, 20, 100),
        '$volume': np.random.uniform(1000000, 10000000, 100),
    }, index=dates)
    test_data['$close'] = test_data['$close'].cumsum()
    test_data['$high'] = test_data['$close'] * 1.1
    test_data['$low'] = test_data['$close'] * 0.9
    test_data['$open'] = test_data['$close'] * 0.95
    
    # 只测试前10个因子
    print("\n测试前10个因子表达式...")
    success = 0
    failed = []
    
    for idx in range(min(10, len(calculator.alpha_fields))):
        field_expr = calculator.alpha_fields[idx]
        factor_name = calculator.alpha_names[idx]
        print(f"{idx+1}/10: {factor_name} = {field_expr[:60]}...")
        
        try:
            result = calculator._evaluate_qlib_expression(test_data, field_expr)
            if result is not None and len(result) > 0:
                valid_count = result.notna().sum()
                if valid_count > 0:
                    print(f"  ✓ 成功 (有效值: {valid_count})")
                    success += 1
                else:
                    print(f"  ✗ 全部为NaN")
                    failed.append((factor_name, "全部为NaN"))
            else:
                print(f"  ✗ 返回空")
                failed.append((factor_name, "返回空"))
        except Exception as e:
            print(f"  ✗ 错误: {str(e)[:50]}")
            failed.append((factor_name, str(e)[:50]))
    
    print(f"\n结果: 成功 {success}/10, 失败 {len(failed)}/10")
    if failed:
        print("\n失败的因子:")
        for name, error in failed:
            print(f"  - {name}: {error}")

if __name__ == '__main__':
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'app.settings')
    main()
