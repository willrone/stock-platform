#!/usr/bin/env python3
"""
最终测试：验证所有158个Alpha158因子都能成功计算
"""
import sys
import os
from pathlib import Path
import time

backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

import pandas as pd
import numpy as np
from app.services.qlib.enhanced_qlib_provider import Alpha158Calculator

def main():
    print("=" * 70)
    print("Alpha158因子计算测试 - 验证所有158个因子")
    print("=" * 70)
    
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'app.settings')
    
    print("\n1. 初始化Alpha158计算器...")
    calculator = Alpha158Calculator()
    
    if len(calculator.alpha_fields) == 0:
        print("❌ 错误: Alpha158配置不可用")
        return False
    
    print(f"✓ 找到 {len(calculator.alpha_fields)} 个因子表达式")
    
    print("\n2. 创建测试数据...")
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
    print(f"✓ 测试数据创建完成: {test_data.shape}")
    
    print("\n3. 开始计算所有158个因子...")
    start_time = time.time()
    
    try:
        factors = calculator._calculate_alpha_factors_from_expressions(test_data, 'TEST.SZ')
        elapsed_time = time.time() - start_time
        
        print(f"\n✓ 因子计算完成，耗时: {elapsed_time:.2f}秒")
        
        # 检查因子数量
        print(f"\n4. 检查因子数量...")
        print(f"   期望: 158个")
        print(f"   实际: {len(factors.columns)}个")
        
        if len(factors.columns) != 158:
            print(f"❌ 错误: 因子数量不匹配！")
            return False
        
        print("✓ 因子数量正确")
        
        # 检查失败的因子（全部为0的）
        print(f"\n5. 检查因子计算质量...")
        zero_cols = [col for col in factors.columns if (factors[col] == 0).all()]
        fail_count = len(zero_cols)
        success_count = len(factors.columns) - fail_count
        success_rate = success_count / len(factors.columns) * 100
        
        print(f"   总因子数: {len(factors.columns)}")
        print(f"   成功因子数: {success_count}")
        print(f"   失败因子数: {fail_count}")
        print(f"   成功率: {success_rate:.2f}%")
        
        if zero_cols:
            print(f"\n   失败的因子 ({len(zero_cols)}个):")
            for col in zero_cols[:20]:  # 只显示前20个
                print(f"     - {col}")
            if len(zero_cols) > 20:
                print(f"     ... 还有 {len(zero_cols) - 20} 个失败的因子")
        
        # 要求所有因子都能计算出来
        if fail_count > 0:
            print(f"\n❌ 测试失败: 有 {fail_count} 个因子计算失败")
            return False
        
        print("\n" + "=" * 70)
        print("✅ 所有158个Alpha158因子计算成功！")
        print("=" * 70)
        return True
        
    except Exception as e:
        import traceback
        print(f"\n❌ 测试失败: {e}")
        print("\n详细错误信息:")
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
