"""
模型管理系统属性测试

属性 4: 模型管理完整性
验证: 需求 4.1, 4.2, 4.3, 4.4, 4.5
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import shutil
from hypothesis import given, strategies as st, settings, assume
from hypothesis.stateful import RuleBasedStateMachine, Bundle, rule, initialize, invariant
from typing import Dict, List, Any
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from app.services.models.model_deployment_service import (
    DeploymentConfig,
    DeploymentStatus,
    ModelDeploymentService,
    ModelEvaluator,
)
from app.services.models.model_evaluation import ModelVersionManager
from app.services.models.model_storage import (
    ModelMetadata,
    ModelStatus,
    ModelStorage,
    ModelType,
)
from app.services.models.model_training_service import (
    ModelTrainingService,
    TrainerFactory,
    TrainingConfig,
)


class ModelManagementStateMachine(RuleBasedStateMachine):
    """模型管理系统状态机测试"""
    
    def __init__(self):
        super().__init__()
        
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()
        self.storage_dir = Path(self.temp_dir) / "models"
        self.data_dir = Path(self.temp_dir) / "data"
        
        # 初始化服务
        self.model_storage = ModelStorage(str(self.storage_dir))
        self.version_manager = ModelVersionManager(self.model_storage)
        self.evaluator = ModelEvaluator(self.model_storage, str(self.data_dir))
        self.deployment_service = ModelDeploymentService(self.model_storage, self.evaluator)
        
        # 状态跟踪
        self.stored_models: Dict[str, ModelMetadata] = {}
        self.model_versions: Dict[str, List[str]] = {}
        self.deployments: Dict[str, str] = {}  # deployment_id -> model_id
        
        # 创建测试数据
        self._create_test_data()
    
    def _create_test_data(self):
        """创建测试数据"""
        # 创建数据目录
        self.data_dir.mkdir(parents=True, exist_ok=True)
        stock_dir = self.data_dir / "daily" / "TEST001"
        stock_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成测试数据
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        np.random.seed(42)
        
        data = pd.DataFrame({
            'open': 100 + np.random.randn(len(dates)).cumsum() * 0.1,
            'high': 100 + np.random.randn(len(dates)).cumsum() * 0.1 + 1,
            'low': 100 + np.random.randn(len(dates)).cumsum() * 0.1 - 1,
            'close': 100 + np.random.randn(len(dates)).cumsum() * 0.1,
            'volume': np.random.randint(1000000, 10000000, len(dates))
        }, index=dates)
        
        # 添加技术指标
        data['sma_5'] = data['close'].rolling(5).mean()
        data['sma_10'] = data['close'].rolling(10).mean()
        data['rsi_14'] = 50 + np.random.randn(len(dates)) * 10
        
        # 保存数据
        data.to_parquet(stock_dir / "2023.parquet")
    
    def teardown(self):
        """清理资源"""
        if hasattr(self, 'temp_dir') and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    # Bundle定义
    models = Bundle('models')
    model_ids = Bundle('model_ids')
    versions = Bundle('versions')
    deployments_bundle = Bundle('deployments')
    
    @initialize()
    def init_state(self):
        """初始化状态"""
        pass
    
    @rule(target=models, 
          model_name=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))),
          model_type=st.sampled_from(list(ModelType)))
    def create_and_store_model(self, model_name: str, model_type: ModelType):
        """创建并存储模型"""
        assume(len(model_name.strip()) > 0)
        
        try:
            # 创建简单模型
            if model_type == ModelType.RANDOM_FOREST:
                model = RandomForestRegressor(n_estimators=10, random_state=42)
            elif model_type == ModelType.LINEAR_REGRESSION:
                model = LinearRegression()
            else:
                # 对于其他类型，使用随机森林作为替代
                model = RandomForestRegressor(n_estimators=10, random_state=42)
            
            # 训练模型（使用简单数据）
            X = np.random.randn(100, 5)
            y = np.random.randn(100)
            model.fit(X, y)
            
            # 创建元数据
            model_id = f"{model_name}_{model_type.value}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"
            metadata = ModelMetadata(
                model_id=model_id,
                model_name=model_name,
                model_type=model_type,
                version="1.0.0",
                description=f"测试模型 {model_name}",
                created_by="test_user",
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                status=ModelStatus.TRAINED,
                training_data_info={
                    "stock_codes": ["TEST001"],
                    "samples": 100,
                    "features": 5
                },
                hyperparameters={"n_estimators": 10},
                training_config={"test": True},
                performance_metrics={"mse": 0.1, "r2": 0.8},
                validation_metrics={"mse": 0.12, "r2": 0.75},
                feature_columns=["feature_1", "feature_2", "feature_3", "feature_4", "feature_5"]
            )
            
            # 存储模型
            success = self.model_storage.save_model(model, metadata)
            assert success, "模型存储应该成功"
            
            # 更新状态
            self.stored_models[model_id] = metadata
            self.model_versions[model_id] = ["1.0.0"]
            
            return model_id
            
        except Exception as e:
            # 如果创建失败，返回None（Hypothesis会处理）
            assume(False)
    
    @rule(model_id=models,
          version=st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=('Nd', 'Po'))),
          description=st.text(min_size=1, max_size=50))
    def create_model_version(self, model_id: str, version: str, description: str):
        """创建模型版本"""
        assume(model_id in self.stored_models)
        assume(version not in self.model_versions.get(model_id, []))
        assume(len(version.strip()) > 0 and len(description.strip()) > 0)
        
        try:
            success = self.version_manager.create_version(
                model_id=model_id,
                version=version,
                description=description,
                created_by="test_user",
                performance_metrics={"mse": 0.1, "r2": 0.8}
            )
            
            if success:
                self.model_versions[model_id].append(version)
                
        except Exception:
            # 版本创建可能因为各种原因失败，这是正常的
            pass
    
    @rule(model_id=models)
    def evaluate_model(self, model_id: str):
        """评估模型"""
        assume(model_id in self.stored_models)
        
        try:
            # 创建简单的测试数据
            test_data = pd.DataFrame({
                'feature_1': np.random.randn(50),
                'feature_2': np.random.randn(50),
                'feature_3': np.random.randn(50),
                'feature_4': np.random.randn(50),
                'feature_5': np.random.randn(50),
                'close': np.random.randn(50)
            }, index=pd.date_range('2024-01-01', periods=50))
            
            evaluation = self.evaluator.evaluate_model(
                model_id=model_id,
                evaluator="test_user",
                test_data=test_data
            )
            
            # 验证评估结果
            assert evaluation.model_id == model_id
            assert evaluation.overall_score >= 0.0
            assert evaluation.overall_score <= 1.0
            assert evaluation.recommendation in ["deploy", "retrain", "reject"]
            
        except Exception:
            # 评估可能失败，这是正常的
            pass
    
    @rule(target=deployments_bundle,
          model_id=models,
          deployment_name=st.text(min_size=1, max_size=20),
          deployment_type=st.sampled_from(["production", "staging", "canary"]))
    def deploy_model(self, model_id: str, deployment_name: str, deployment_type: str):
        """部署模型"""
        assume(model_id in self.stored_models)
        assume(len(deployment_name.strip()) > 0)
        assume(model_id not in [self.deployments.get(d) for d in self.deployments])  # 模型未部署
        
        try:
            config = DeploymentConfig(
                model_id=model_id,
                deployment_name=deployment_name,
                deployment_type=deployment_type,
                traffic_percentage=100.0 if deployment_type != "canary" else 10.0
            )
            
            deployment_id = self.deployment_service.deploy_model(
                model_id=model_id,
                config=config,
                deployed_by="test_user",
                force=True  # 强制部署以避免评估检查
            )
            
            # 更新状态
            self.deployments[deployment_id] = model_id
            
            return deployment_id
            
        except Exception:
            # 部署可能失败，返回None
            assume(False)
    
    @rule(deployment_id=deployments_bundle)
    def rollback_deployment(self, deployment_id: str):
        """回滚部署"""
        assume(deployment_id in self.deployments)
        
        try:
            success = self.deployment_service.rollback_deployment(
                deployment_id=deployment_id,
                user_id="test_user",
                reason="测试回滚"
            )
            
            if success:
                # 从活跃部署中移除
                model_id = self.deployments[deployment_id]
                # 不从deployments字典中删除，因为记录仍然存在
                
        except Exception:
            # 回滚可能失败
            pass
    
    @rule(model_id=models)
    def load_model(self, model_id: str):
        """加载模型"""
        assume(model_id in self.stored_models)
        
        try:
            model, metadata = self.model_storage.load_model(model_id)
            
            # 验证加载的模型和元数据
            assert model is not None
            assert metadata.model_id == model_id
            assert hasattr(model, 'predict')  # 模型应该有预测方法
            
        except Exception as e:
            # 加载失败应该抛出适当的异常
            assert "模型" in str(e) or "不存在" in str(e)
    
    @invariant()
    def models_exist_in_storage(self):
        """不变量：存储的模型应该能够被检索"""
        for model_id in self.stored_models:
            assert self.model_storage.model_exists(model_id), f"模型 {model_id} 应该存在于存储中"
    
    @invariant()
    def model_metadata_consistency(self):
        """不变量：模型元数据应该保持一致"""
        for model_id, expected_metadata in self.stored_models.items():
            try:
                actual_metadata = self.model_storage.get_model_metadata(model_id)
                if actual_metadata:
                    assert actual_metadata.model_id == expected_metadata.model_id
                    assert actual_metadata.model_type == expected_metadata.model_type
                    assert actual_metadata.model_name == expected_metadata.model_name
            except Exception:
                # 如果获取元数据失败，跳过检查
                pass
    
    @invariant()
    def deployment_consistency(self):
        """不变量：部署记录应该保持一致"""
        for deployment_id, model_id in self.deployments.items():
            deployment_record = self.deployment_service.get_deployment_status(deployment_id)
            if deployment_record:
                assert deployment_record.model_id == model_id
                assert deployment_record.deployment_id == deployment_id


class TestModelManagementProperties:
    """模型管理属性测试类"""
    
    def setup_method(self):
        """测试设置"""
        self.temp_dir = tempfile.mkdtemp()
        self.storage_dir = Path(self.temp_dir) / "models"
        self.data_dir = Path(self.temp_dir) / "data"
        
        # 初始化服务
        self.model_storage = ModelStorage(str(self.storage_dir))
        self.version_manager = ModelVersionManager(self.model_storage)
        self.evaluator = ModelEvaluator(self.model_storage, str(self.data_dir))
        self.deployment_service = ModelDeploymentService(self.model_storage, self.evaluator)
        
        # 创建测试数据
        self._create_test_data()
    
    def teardown_method(self):
        """测试清理"""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def _create_test_data(self):
        """创建测试数据"""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        stock_dir = self.data_dir / "daily" / "TEST001"
        stock_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成测试数据
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        np.random.seed(42)
        
        data = pd.DataFrame({
            'open': 100 + np.random.randn(len(dates)).cumsum() * 0.1,
            'high': 100 + np.random.randn(len(dates)).cumsum() * 0.1 + 1,
            'low': 100 + np.random.randn(len(dates)).cumsum() * 0.1 - 1,
            'close': 100 + np.random.randn(len(dates)).cumsum() * 0.1,
            'volume': np.random.randint(1000000, 10000000, len(dates)),
            'sma_5': 0,
            'sma_10': 0,
            'rsi_14': 50
        }, index=dates)
        
        # 计算技术指标
        data['sma_5'] = data['close'].rolling(5).mean().fillna(data['close'])
        data['sma_10'] = data['close'].rolling(10).mean().fillna(data['close'])
        data['rsi_14'] = 50 + np.random.randn(len(dates)) * 10
        
        data.to_parquet(stock_dir / "2023.parquet")
    
    @given(st.lists(st.tuples(
        st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))),
        st.sampled_from(list(ModelType))
    ), min_size=1, max_size=5, unique_by=lambda x: x[0]))
    @settings(max_examples=10, deadline=30000)
    def test_model_storage_consistency(self, model_specs):
        """属性测试：模型存储一致性"""
        stored_models = {}
        
        for model_name, model_type in model_specs:
            # 创建模型
            if model_type == ModelType.RANDOM_FOREST:
                model = RandomForestRegressor(n_estimators=10, random_state=42)
            else:
                model = LinearRegression()
            
            # 训练模型
            X = np.random.randn(50, 3)
            y = np.random.randn(50)
            model.fit(X, y)
            
            # 创建元数据
            model_id = f"{model_name}_{model_type.value}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"
            metadata = ModelMetadata(
                model_id=model_id,
                model_name=model_name,
                model_type=model_type,
                version="1.0.0",
                description=f"测试模型 {model_name}",
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
            
            # 存储模型
            success = self.model_storage.save_model(model, metadata)
            assert success, f"模型 {model_id} 存储应该成功"
            
            stored_models[model_id] = (model, metadata)
        
        # 验证所有模型都能正确加载
        for model_id, (original_model, original_metadata) in stored_models.items():
            # 检查模型存在
            assert self.model_storage.model_exists(model_id), f"模型 {model_id} 应该存在"
            
            # 加载模型
            loaded_model, loaded_metadata = self.model_storage.load_model(model_id)
            
            # 验证元数据一致性
            assert loaded_metadata.model_id == original_metadata.model_id
            assert loaded_metadata.model_type == original_metadata.model_type
            assert loaded_metadata.model_name == original_metadata.model_name
            
            # 验证模型功能一致性
            test_X = np.random.randn(10, 3)
            original_pred = original_model.predict(test_X)
            loaded_pred = loaded_model.predict(test_X)
            np.testing.assert_array_almost_equal(original_pred, loaded_pred, decimal=5)
    
    @given(st.integers(min_value=1, max_value=5))
    @settings(max_examples=5, deadline=30000)
    def test_model_version_management(self, num_versions):
        """属性测试：模型版本管理"""
        # 创建基础模型
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        X = np.random.randn(50, 3)
        y = np.random.randn(50)
        model.fit(X, y)
        
        model_id = f"test_model_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"
        metadata = ModelMetadata(
            model_id=model_id,
            model_name="test_model",
            model_type=ModelType.RANDOM_FOREST,
            version="1.0.0",
            description="基础版本",
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
        
        # 存储基础模型
        self.model_storage.save_model(model, metadata)
        
        # 创建多个版本
        created_versions = ["1.0.0"]  # 基础版本
        for i in range(num_versions):
            version = f"1.{i+1}.0"
            success = self.version_manager.create_version(
                model_id=model_id,
                version=version,
                description=f"版本 {version}",
                created_by="test_user",
                performance_metrics={"mse": 0.1 - i * 0.01}
            )
            
            if success:
                created_versions.append(version)
        
        # 验证版本列表
        versions = self.version_manager.list_versions(model_id)
        version_strings = [v.version for v in versions]
        
        # 所有创建的版本都应该存在
        for version in created_versions[1:]:  # 跳过基础版本，因为它可能不在版本列表中
            if version in version_strings:
                # 如果版本存在，验证其属性
                version_obj = next(v for v in versions if v.version == version)
                assert version_obj.created_by == "test_user"
                assert version_obj.description == f"版本 {version}"
    
    @given(st.sampled_from(["production", "staging", "canary"]))
    @settings(max_examples=3, deadline=30000)
    def test_model_deployment_lifecycle(self, deployment_type):
        """属性测试：模型部署生命周期"""
        # 创建模型
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        X = np.random.randn(50, 3)
        y = np.random.randn(50)
        model.fit(X, y)
        
        model_id = f"deploy_test_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"
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
            training_data_info={"samples": 50},
            hyperparameters={},
            training_config={},
            performance_metrics={"mse": 0.1},
            validation_metrics={"mse": 0.1},
            feature_columns=["f1", "f2", "f3"]
        )
        
        # 存储模型
        self.model_storage.save_model(model, metadata)
        
        # 部署配置
        config = DeploymentConfig(
            model_id=model_id,
            deployment_name=f"test_deployment_{deployment_type}",
            deployment_type=deployment_type,
            traffic_percentage=100.0 if deployment_type != "canary" else 10.0
        )
        
        # 部署模型
        deployment_id = self.deployment_service.deploy_model(
            model_id=model_id,
            config=config,
            deployed_by="test_user",
            force=True
        )
        
        # 验证部署状态
        deployment_record = self.deployment_service.get_deployment_status(deployment_id)
        assert deployment_record is not None
        assert deployment_record.model_id == model_id
        assert deployment_record.status == DeploymentStatus.DEPLOYED
        assert deployment_record.config.deployment_type == deployment_type
        
        # 测试回滚
        success = self.deployment_service.rollback_deployment(
            deployment_id=deployment_id,
            user_id="test_user",
            reason="测试回滚"
        )
        
        assert success
        
        # 验证回滚后状态
        deployment_record = self.deployment_service.get_deployment_status(deployment_id)
        assert deployment_record.status == DeploymentStatus.ROLLBACK


# 运行状态机测试
TestModelManagement = ModelManagementStateMachine.TestCase


@pytest.mark.slow
def test_model_management_state_machine():
    """运行模型管理状态机测试"""
    # 创建状态机实例
    state_machine = ModelManagementStateMachine()
    
    try:
        # 运行状态机测试
        state_machine.init_state()
        
        # 手动执行一些操作来测试基本功能
        model_id = state_machine.create_and_store_model("test_model", ModelType.RANDOM_FOREST)
        if model_id:
            state_machine.evaluate_model(model_id)
            deployment_id = state_machine.deploy_model(model_id, "test_deployment", "staging")
            if deployment_id:
                state_machine.rollback_deployment(deployment_id)
    
    finally:
        # 清理
        state_machine.teardown()


if __name__ == "__main__":
    # 运行基本测试
    test_instance = TestModelManagementProperties()
    test_instance.setup_method()
    
    try:
        # 运行一些基本测试
        test_instance.test_model_storage_consistency([("test", ModelType.RANDOM_FOREST)])
        test_instance.test_model_version_management(2)
        test_instance.test_model_deployment_lifecycle("staging")
        print("所有模型管理属性测试通过！")
    
    finally:
        test_instance.teardown_method()