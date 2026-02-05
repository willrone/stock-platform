#!/usr/bin/env python3
"""
验证模型文件可以被 ml_ensemble_strategy.py 正确加载
"""
import sys
sys.path.insert(0, '/Users/ronghui/Documents/GitHub/willrone/backend')

import pickle
from pathlib import Path
import numpy as np

MODEL_DIR = Path('/Users/ronghui/Documents/GitHub/willrone/data/models')

def test_model_loading():
    """测试模型加载"""
    print("="*60)
    print("验证模型加载")
    print("="*60)
    
    # 1. 直接加载测试
    print("\n1. 直接加载模型文件...")
    
    lgb_path = MODEL_DIR / "lgb_model.pkl"
    xgb_path = MODEL_DIR / "xgb_model.pkl"
    
    with open(lgb_path, 'rb') as f:
        lgb_model = pickle.load(f)
    print(f"   ✅ LightGBM 模型加载成功: {type(lgb_model)}")
    
    with open(xgb_path, 'rb') as f:
        xgb_model = pickle.load(f)
    print(f"   ✅ XGBoost 模型加载成功: {type(xgb_model)}")
    
    # 2. 测试预测
    print("\n2. 测试模型预测...")
    
    # 创建测试特征（62个特征）
    test_features = np.random.randn(1, 62)
    
    lgb_pred = lgb_model.predict(test_features)
    print(f"   ✅ LightGBM 预测: {lgb_pred[0]:.4f}")
    
    import xgboost as xgb
    xgb_pred = xgb_model.predict(xgb.DMatrix(test_features))
    print(f"   ✅ XGBoost 预测: {xgb_pred[0]:.4f}")
    
    ensemble_pred = 0.5 * lgb_pred[0] + 0.5 * xgb_pred[0]
    print(f"   ✅ 集成预测: {ensemble_pred:.4f}")
    
    # 3. 测试策略类加载
    print("\n3. 测试 MLEnsembleLgbXgbRiskCtlStrategy 加载...")
    
    try:
        from app.services.backtest.strategies.ml_ensemble_strategy import MLEnsembleLgbXgbRiskCtlStrategy
        
        config = {
            'model_path': str(MODEL_DIR),
            'lgb_weight': 0.5,
            'xgb_weight': 0.5,
            'top_n': 5,
        }
        
        strategy = MLEnsembleLgbXgbRiskCtlStrategy(config)
        strategy._load_models()
        
        if strategy.lgb_model is not None and strategy.xgb_model is not None:
            print(f"   ✅ 策略成功加载模型")
            print(f"      LGB 模型: {type(strategy.lgb_model)}")
            print(f"      XGB 模型: {type(strategy.xgb_model)}")
        else:
            print(f"   ❌ 策略未能加载模型")
            return False
            
    except Exception as e:
        print(f"   ❌ 策略加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*60)
    print("✅ 所有验证通过！")
    print("="*60)
    return True

if __name__ == '__main__':
    success = test_model_loading()
    sys.exit(0 if success else 1)
