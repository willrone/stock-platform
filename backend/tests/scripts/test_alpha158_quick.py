#!/usr/bin/env python3
"""
快速测试Alpha158表达式引擎
只测试关键功能，不运行完整测试套件
"""
import sys
import os
from pathlib import Path

# 添加项目路径
backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

import pandas as pd
import numpy as np
from app.services.qlib.enhanced_qlib_provider import Alpha158Calculator

def test_basic_functions():
    """测试基础函数"""
    print("=" * 60)
    print("测试基础函数...")
    print("=" * 60)
    
    calculator = Alpha158Calculator()
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
    
    # 测试Ref函数
    print("1. 测试Ref函数...")
    result = calculator._evaluate_qlib_expression(test_data, "Ref($close, 5)/$close")
    assert result is not None, "Ref函数测试失败"
    print("   ✓ Ref函数测试通过")
    
    # 测试Greater/Less函数
    print("2. 测试Greater/Less函数...")
    result = calculator._evaluate_qlib_expression(test_data, "Greater($close-Ref($close, 1), 0)")
    assert result is not None, "Greater函数测试失败"
    print("   ✓ Greater函数测试通过")
    
    result = calculator._evaluate_qlib_expression(test_data, "Less($close-Ref($close, 1), 0)")
    assert result is not None, "Less函数测试失败"
    print("   ✓ Less函数测试通过")
    
    # 测试Sum函数
    print("3. 测试Sum函数...")
    expr = "Sum(Greater($close-Ref($close, 1), 0), 5)/(Sum(Abs($close-Ref($close, 1)), 5)+1e-12)"
    result = calculator._evaluate_qlib_expression(test_data, expr)
    assert result is not None, "Sum函数测试失败"
    valid_count = result.notna().sum()
    assert valid_count > 0, "Sum函数结果全部为NaN"
    print(f"   ✓ Sum函数测试通过 (有效值: {valid_count})")
    
    # 测试Max/Min函数
    print("4. 测试Max/Min函数...")
    result = calculator._evaluate_qlib_expression(test_data, "Max($high, 5)")
    assert result is not None, "Max函数测试失败"
    print("   ✓ Max函数测试通过")
    
    # 测试IdxMax/IdxMin函数
    print("5. 测试IdxMax/IdxMin函数...")
    result = calculator._evaluate_qlib_expression(test_data, "IdxMax($high, 5)/5")
    assert result is not None, "IdxMax函数测试失败"
    print("   ✓ IdxMax函数测试通过")
    
    result = calculator._evaluate_qlib_expression(test_data, "IdxMin($low, 5)/5")
    assert result is not None, "IdxMin函数测试失败"
    print("   ✓ IdxMin函数测试通过")
    
    print("\n所有基础函数测试通过！\n")

def test_all_factors():
    """测试所有158个因子"""
    print("=" * 60)
    print("测试所有158个Alpha158因子...")
    print("=" * 60)
    
    calculator = Alpha158Calculator()
    
    # 检查是否有Alpha158配置
    if len(calculator.alpha_fields) == 0:
        print("⚠️  警告: Alpha158配置不可用，无法测试所有因子")
        return
    
    print(f"找到 {len(calculator.alpha_fields)} 个因子表达式")
    
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
    
    import time
    start_time = time.time()
    
    print("开始计算所有因子...")
    factors = calculator._calculate_alpha_factors_from_expressions(test_data, 'TEST.SZ')
    
    elapsed_time = time.time() - start_time
    print(f"\n因子计算完成，耗时: {elapsed_time:.2f}秒")
    
    # 检查因子数量
    print(f"\n因子数量: {len(factors.columns)}")
    assert len(factors.columns) == 158, f"应该有158个因子，实际有{len(factors.columns)}个"
    
    # 检查失败的因子（全部为0的）
    zero_cols = [col for col in factors.columns if (factors[col] == 0).all()]
    fail_count = len(zero_cols)
    
    print(f"\n因子计算统计:")
    print(f"  总因子数: {len(factors.columns)}")
    print(f"  成功因子数: {len(factors.columns) - fail_count}")
    print(f"  失败因子数: {fail_count}")
    print(f"  成功率: {(len(factors.columns) - fail_count) / len(factors.columns) * 100:.2f}%")
    
    if zero_cols:
        print(f"\n失败的因子 ({len(zero_cols)}个):")
        for col in zero_cols[:20]:  # 只显示前20个
            print(f"  - {col}")
        if len(zero_cols) > 20:
            print(f"  ... 还有 {len(zero_cols) - 20} 个失败的因子")
    
    # 要求所有因子都能计算出来
    assert fail_count == 0, f"所有158个因子都应该能计算出来，但失败了{fail_count}个"
    
    print("\n✓ 所有158个因子计算成功！")

if __name__ == '__main__':
    try:
        # 设置环境变量
        os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'app.settings')
        
        # 运行基础函数测试
        test_basic_functions()
        
        # 运行所有因子测试
        test_all_factors()
        
        print("\n" + "=" * 60)
        print("所有测试通过！")
        print("=" * 60)
        
    except Exception as e:
        import traceback
        print(f"\n❌ 测试失败: {e}")
        print("\n详细错误信息:")
        traceback.print_exc()
        sys.exit(1)
