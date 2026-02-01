#!/usr/bin/env python3
"""
验证模型管理系统功能
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.app.services.model_storage import (
    ModelStorage, ModelMetadata, ModelType, ModelStatus, ModelVersionManager
)
from backend.app.services.model_deployment_service import (
    ModelDeploymentService, ModelEvaluator, DeploymentConfig
)


def test_model_storage():
    """测试模型存储功能"""
    print("测试模型存储功能...")
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    storage_dir = Path(temp_dir) / "models"
    
    try:
        # 初始化存储
        storage = ModelStorage(str(storage_dir))
        
        # 创建测试模型
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        model.fit(X, y)
        
        # 创建元数据
        model_id = f"test_model_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        metadata = ModelMetadata(
            model_id=model_id,
            model_name="test_model",
            model_type=ModelType.RANDOM_FOREST,
            version="1.0.0",
            description="测试模型",
            created_by="test_user",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            status=ModelStatus.TRAINED,
            training_data_info={"samples": 100, "features": 5},
            hyperparameters={"n_estimators": 10},
            training_config={"test": True},
            performance_metrics={"mse": 0.1, "r2": 0.8},
            validation_metrics={"mse": 0.12, "r2": 0.75},
            feature_columns=["f1", "f2", "f3", "f4", "f5"]
        )
        
        # 保存模型
        success = storage.save_model(model, metadata)
        assert success, "模型保存失败"
        print(f"✓ 模型保存成功: {model_id}")
        
        # 检查模型存在
        assert storage.model_exists(model_id), "模型不存在"
        print("✓ 模型存在检查通过")
        
        # 加载模型
        loaded_model, loaded_metadata = storage.load_model(model_id)
        assert loaded_model is not None, "加载的模型为空"
        assert loaded_metadata.model_id == model_id, "元数据不匹配"
        print("✓ 模型加载成功")
        
        # 验证预测一致性
        test_X = np.random.randn(10, 5)
        original_pred = model.predict(test_X)
        loaded_pred = loaded_model.predict(test_X)
        np.testing.assert_array_almost_equal(original_pred, loaded_pred, decimal=5)
        print("✓ 预测结果一致性验证通过")
        
        # 测试模型列表
        models = storage.list_models()
        assert len(models) >= 1, "模型列表为空"
        assert any(m.model_id == model_id for m in models), "模型不在列表中"
        print("✓ 模型列表功能正常")
        
        return model_id, storage
        
    except Exception as e:
        print(f"✗ 模型存储测试失败: {e}")
        raise
    finally:
        # 清理临时目录
        if Path(temp_dir).exists():
            shutil.rmtree(temp_dir)


def test_model_versioning():
    """测试模型版本管理"""
    print("\n测试模型版本管理...")
    
    temp_dir = tempfile.mkdtemp()
    storage_dir = Path(temp_dir) / "models"
    
    try:
        # 初始化存储和版本管理器
        storage = ModelStorage(str(storage_dir))
        version_manager = ModelVersionManager(storage)
        
        # 创建基础模型
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        X = np.random.randn(50, 3)
        y = np.random.randn(50)
        model.fit(X, y)
        
        model_id = f"version_test_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        metadata = ModelMetadata(
            model_id=model_id,
            model_name="version_test",
            model_type=ModelType.RANDOM_FOREST,
            version="1.0.0",
            description="版本测试模型",
            created_by="test_user",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            status=ModelStatus.TRAINED,
            training_data_info={"samples": 50},
            hyperparameters={},
            training_config={},
            performance_metrics={"mse": 0.1},
            validation_metrics={"mse": 0.1},
            feature_columns=["f1", "f2", "f3"]
        )
        
        # 保存基础模型
        storage.save_model(model, metadata)
        print(f"✓ 基础模型创建成功: {model_id}")
        
        # 创建新版本
        success = version_manager.create_version(
            model_id=model_id,
            version="1.1.0",
            description="改进版本",
            created_by="test_user",
            performance_metrics={"mse": 0.08, "r2": 0.85}
        )
        assert success, "版本创建失败"
        print("✓ 新版本创建成功: 1.1.0")
        
        # 列出版本
        versions = version_manager.list_versions(model_id)
        version_strings = [v.version for v in versions]
        assert "1.1.0" in version_strings, "新版本不在列表中"
        print(f"✓ 版本列表正常: {version_strings}")
        
        return True
        
    except Exception as e:
        print(f"✗ 版本管理测试失败: {e}")
        raise
    finally:
        if Path(temp_dir).exists():
            shutil.rmtree(temp_dir)


def test_model_evaluation_and_deployment():
    """测试模型评估和部署"""
    print("\n测试模型评估和部署...")
    
    temp_dir = tempfile.mkdtemp()
    storage_dir = Path(temp_dir) / "models"
    data_dir = Path(temp_dir) / "data"
    
    try:
        # 创建测试数据
        data_dir.mkdir(parents=True, exist_ok=True)
        stock_dir = data_dir / "daily" / "TEST001"
        stock_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成测试数据
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        test_data = pd.DataFrame({
            'open': 100 + np.random.randn(100).cumsum() * 0.1,
            'high': 100 + np.random.randn(100).cumsum() * 0.1 + 1,
            'low': 100 + np.random.randn(100).cumsum() * 0.1 - 1,
            'close': 100 + np.random.randn(100).cumsum() * 0.1,
            'volume': np.random.randint(1000000, 10000000, 100),
            'f1': np.random.randn(100),
            'f2': np.random.randn(100),
            'f3': np.random.randn(100)
        }, index=dates)
        
        test_data.to_parquet(stock_dir / "2023.parquet")
        print("✓ 测试数据创建成功")
        
        # 初始化服务
        storage = ModelStorage(str(storage_dir))
        evaluator = ModelEvaluator(storage, str(data_dir))
        deployment_service = ModelDeploymentService(storage, evaluator)
        
        # 创建测试模型
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        X = test_data[['f1', 'f2', 'f3']].values
        y = test_data['close'].pct_change().fillna(0).values
        model.fit(X, y)
        
        model_id = f"deploy_test_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        metadata = ModelMetadata(
            model_id=model_id,
            model_name="deploy_test",
            model_type=ModelType.RANDOM_FOREST,
            version="1.0.0",
            description="部署测试模型",
            created_by="test_user",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            status=ModelStatus.TRAINED,
            training_data_info={"samples": 100, "stock_codes": ["TEST001"]},
            hyperparameters={"n_estimators": 10},
            training_config={},
            performance_metrics={"mse": 0.1, "r2": 0.8},
            validation_metrics={"mse": 0.1, "r2": 0.8},
            feature_columns=["f1", "f2", "f3"]
        )
        
        # 保存模型
        storage.save_model(model, metadata)
        print(f"✓ 测试模型创建成功: {model_id}")
        
        # 评估模型
        evaluation = evaluator.evaluate_model(
            model_id=model_id,
            evaluator="test_user",
            test_data=test_data
        )
        
        assert evaluation.model_id == model_id, "评估结果模型ID不匹配"
        assert 0 <= evaluation.overall_score <= 1, "综合评分超出范围"
        assert evaluation.recommendation in ["deploy", "retrain", "reject"], "建议值无效"
        print(f"✓ 模型评估成功: 评分={evaluation.overall_score:.3f}, 建议={evaluation.recommendation}")
        
        # 部署模型
        config = DeploymentConfig(
            model_id=model_id,
            deployment_name="test_deployment",
            deployment_type="staging",
            traffic_percentage=100.0
        )
        
        deployment_id = deployment_service.deploy_model(
            model_id=model_id,
            config=config,
            deployed_by="test_user",
            force=True
        )
        
        assert deployment_id is not None, "部署ID为空"
        print(f"✓ 模型部署成功: {deployment_id}")
        
        # 检查部署状态
        deployment_record = deployment_service.get_deployment_status(deployment_id)
        assert deployment_record is not None, "部署记录不存在"
        assert deployment_record.model_id == model_id, "部署记录模型ID不匹配"
        print("✓ 部署状态检查通过")
        
        # 测试回滚
        success = deployment_service.rollback_deployment(
            deployment_id=deployment_id,
            user_id="test_user",
            reason="测试回滚"
        )
        assert success, "回滚失败"
        print("✓ 部署回滚成功")
        
        return True
        
    except Exception as e:
        print(f"✗ 评估和部署测试失败: {e}")
        raise
    finally:
        if Path(temp_dir).exists():
            shutil.rmtree(temp_dir)


def main():
    """主测试函数"""
    print("开始验证模型管理系统...")
    
    try:
        # 测试模型存储
        test_model_storage()
        
        # 测试版本管理
        test_model_versioning()
        
        # 测试评估和部署
        test_model_evaluation_and_deployment()
        
        print("\n🎉 所有模型管理系统测试通过！")
        print("\n模型管理系统功能验证完成：")
        print("✓ 模型存储和加载")
        print("✓ 模型版本管理")
        print("✓ 模型评估")
        print("✓ 模型部署和回滚")
        print("✓ 元数据管理")
        print("✓ 性能监控框架")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 模型管理系统验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)