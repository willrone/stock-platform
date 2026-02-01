#!/usr/bin/env python3
"""
直接运行Qlib优化测试（不依赖pytest）
用于快速验证测试逻辑是否正确
"""

import sys
import traceback
import numpy as np
import pandas as pd
from datetime import datetime

# 添加项目路径
sys.path.insert(0, '/Users/ronghui/Documents/GitHub/willrone/backend')

try:
    from app.services.qlib.unified_qlib_training_engine import (
        UnifiedQlibTrainingEngine,
        QlibTrainingConfig,
        QlibModelType,
        RobustFeatureScaler,
        OutlierHandler,
    )
    from app.services.qlib.enhanced_qlib_provider import EnhancedQlibDataProvider
    from app.services.qlib.qlib_model_manager import LightGBMAdapter
except ImportError as e:
    print(f"导入错误: {e}")
    sys.exit(1)


def test_label_calculation():
    """测试标签计算"""
    print("\n=== 测试标签计算 ===")
    try:
        engine = UnifiedQlibTrainingEngine()
        
        # 创建样本数据
        dates = pd.date_range(start='2023-01-01', periods=20, freq='D')
        data = pd.DataFrame({
            '$close': 100 + np.cumsum(np.random.randn(20) * 0.5),
            '$open': 100 + np.cumsum(np.random.randn(20) * 0.5),
            '$high': 100 + np.cumsum(np.random.randn(20) * 0.5) + 1,
            '$low': 100 + np.cumsum(np.random.randn(20) * 0.5) - 1,
            '$volume': np.random.randint(1000000, 10000000, 20),
        }, index=dates)
        
        # 测试prediction_horizon=5
        prediction_horizon = 5
        processed = engine._process_stock_data(data.copy(), 'TEST', prediction_horizon=prediction_horizon)
        
        assert 'label' in processed.columns, "应该创建label列"
        assert processed['label'].iloc[-prediction_horizon:].isna().all() or \
               (processed['label'].iloc[-prediction_horizon:] == 0).all(), \
               "最后N行标签应该是NaN或0"
        
        print("✅ 标签计算测试通过")
        return True
    except Exception as e:
        print(f"❌ 标签计算测试失败: {e}")
        traceback.print_exc()
        return False


def test_missing_values():
    """测试缺失值处理"""
    print("\n=== 测试缺失值处理 ===")
    try:
        provider = EnhancedQlibDataProvider()
        
        dates = pd.date_range(start='2023-01-01', periods=20, freq='D')
        data = pd.DataFrame({
            '$close': 100 + np.cumsum(np.random.randn(20) * 0.5),
            '$open': 100 + np.cumsum(np.random.randn(20) * 0.5),
            'indicator1': np.random.randn(20),
        }, index=dates)
        
        # 添加缺失值
        data.loc[data.index[5], '$close'] = np.nan
        data.loc[data.index[:3], 'indicator1'] = np.nan
        
        processed = provider._handle_missing_values(data.copy())
        
        assert not processed['$close'].isna().any(), "价格数据不应该有缺失值"
        assert not processed['indicator1'].isna().any(), "技术指标不应该有缺失值"
        
        print("✅ 缺失值处理测试通过")
        return True
    except Exception as e:
        print(f"❌ 缺失值处理测试失败: {e}")
        traceback.print_exc()
        return False


def test_robust_scaler():
    """测试特征标准化"""
    print("\n=== 测试特征标准化 ===")
    try:
        scaler = RobustFeatureScaler()
        
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'feature1': np.random.randn(100) * 10 + 100,
            'feature2': np.random.randn(100) * 0.1 + 0.5,
            'label': np.random.randn(100) * 0.02,
        }, index=dates)
        
        feature_cols = ['feature1', 'feature2']
        scaled_data = scaler.fit_transform(data.copy(), feature_cols)
        
        assert 'feature1' in scaled_data.columns
        assert 'feature2' in scaled_data.columns
        assert scaler.fitted, "scaler应该被标记为已拟合"
        
        # 验证标签没有被标准化
        assert np.allclose(scaled_data['label'].values, data['label'].values, equal_nan=True), \
               "标签不应该被标准化"
        
        print("✅ 特征标准化测试通过")
        return True
    except Exception as e:
        print(f"❌ 特征标准化测试失败: {e}")
        traceback.print_exc()
        return False


def test_outlier_handler():
    """测试异常值处理"""
    print("\n=== 测试异常值处理 ===")
    try:
        handler = OutlierHandler(method="winsorize", lower_percentile=0.01, upper_percentile=0.99)
        
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        normal_returns = np.random.randn(100) * 0.02
        normal_returns[10] = 0.8  # 极端正值
        normal_returns[20] = -0.7  # 极端负值
        
        data = pd.DataFrame({
            'label': normal_returns,
            'feature1': np.random.randn(100),
        }, index=dates)
        
        processed = handler.handle_label_outliers(data.copy(), label_col="label")
        
        assert 'label' in processed.columns
        assert processed['label'].min() >= -1, "标签最小值应该合理"
        assert processed['label'].max() <= 1, "标签最大值应该合理"
        
        print("✅ 异常值处理测试通过")
        return True
    except Exception as e:
        print(f"❌ 异常值处理测试失败: {e}")
        traceback.print_exc()
        return False


def test_loss_function():
    """测试损失函数配置"""
    print("\n=== 测试损失函数配置 ===")
    try:
        adapter = LightGBMAdapter()
        
        hyperparameters = {
            "learning_rate": 0.1,
            "num_leaves": 31,
            "huber_delta": 0.1,
        }
        
        config = adapter.create_qlib_config(hyperparameters)
        
        assert config["kwargs"]["loss"] == "huber", "应该使用Huber损失"
        assert "huber_delta" in config["kwargs"], "应该包含huber_delta参数"
        assert config["kwargs"]["huber_delta"] == 0.1, "huber_delta应该正确设置"
        
        print("✅ 损失函数配置测试通过")
        return True
    except Exception as e:
        print(f"❌ 损失函数配置测试失败: {e}")
        traceback.print_exc()
        return False


async def test_enhanced_provider_loss():
    """测试EnhancedQlibDataProvider的损失函数"""
    print("\n=== 测试EnhancedQlibDataProvider损失函数 ===")
    try:
        import asyncio
        provider = EnhancedQlibDataProvider()
        
        hyperparameters = {
            "learning_rate": 0.05,
            "huber_delta": 0.15,
        }
        
        config = await provider.create_qlib_model_config("lightgbm", hyperparameters)
        
        assert config["kwargs"]["loss"] == "huber", "应该使用Huber损失"
        assert config["kwargs"]["huber_delta"] == 0.15, "huber_delta应该正确设置"
        
        print("✅ EnhancedQlibDataProvider损失函数测试通过")
        return True
    except Exception as e:
        print(f"❌ EnhancedQlibDataProvider损失函数测试失败: {e}")
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("开始运行Qlib优化功能测试...")
    
    results = []
    
    # 同步测试
    results.append(("标签计算", test_label_calculation()))
    results.append(("缺失值处理", test_missing_values()))
    results.append(("特征标准化", test_robust_scaler()))
    results.append(("异常值处理", test_outlier_handler()))
    results.append(("损失函数配置", test_loss_function()))
    
    # 异步测试
    try:
        import asyncio
        async_result = asyncio.run(test_enhanced_provider_loss())
        results.append(("EnhancedQlibDataProvider损失函数", async_result))
    except Exception as e:
        print(f"⚠️  异步测试跳过: {e}")
        results.append(("EnhancedQlibDataProvider损失函数", False))
    
    # 汇总结果
    print("\n" + "="*50)
    print("测试结果汇总:")
    print("="*50)
    
    passed = 0
    failed = 0
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{name}: {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print("="*50)
    print(f"总计: {len(results)} 个测试")
    print(f"通过: {passed} 个")
    print(f"失败: {failed} 个")
    print("="*50)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
