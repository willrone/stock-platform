"""
高级训练功能测试

测试模型集成（ensemble）、在线学习和增量训练功能。
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app', 'services'))

from advanced_training import (
    AdvancedTrainingService,
    EnsembleModelManager,
    OnlineLearningManager,
    EnsembleConfig,
    OnlineLearningConfig,
    EnsembleMethod
)


class TestAdvancedTrainingService:
    """高级训练服务测试"""
    
    def test_service_creation(self):
        """测试服务创建"""
        service = AdvancedTrainingService()
        assert service is not None
        assert service.model_training_service is None
        assert service.ensemble_manager is None
        assert service.online_learning_manager is None
    
    @pytest.mark.asyncio
    async def test_service_initialization(self):
        """测试服务初始化"""
        service = AdvancedTrainingService()
        
        # 模拟依赖服务
        with patch('advanced_training.ModelTrainingService') as mock_training_class:
            mock_training_instance = Mock()
            mock_training_instance.initialize = AsyncMock()
            mock_training_class.return_value = mock_training_instance
            
            await service.initialize()
            
            assert service.model_training_service is not None
            assert service.ensemble_manager is not None
            assert service.online_learning_manager is not None
            mock_training_instance.initialize.assert_called_once()


class TestEnsembleConfig:
    """集成配置测试"""
    
    def test_ensemble_config_creation(self):
        """测试集成配置创建"""
        config = EnsembleConfig(
            method=EnsembleMethod.VOTING,
            base_models=["model1", "model2"],
            weights=[0.6, 0.4],
            voting_strategy="soft"
        )
        
        assert config.method == EnsembleMethod.VOTING
        assert config.base_models == ["model1", "model2"]
        assert config.weights == [0.6, 0.4]
        assert config.voting_strategy == "soft"
    
    def test_ensemble_config_defaults(self):
        """测试集成配置默认值"""
        config = EnsembleConfig(
            method=EnsembleMethod.WEIGHTED,
            base_models=["model1"]
        )
        
        assert config.weights is None
        assert config.meta_model_type is None
        assert config.voting_strategy == "soft"


class TestOnlineLearningConfig:
    """在线学习配置测试"""
    
    def test_online_learning_config_creation(self):
        """测试在线学习配置创建"""
        config = OnlineLearningConfig(
            update_frequency=10,
            learning_rate_decay=0.9,
            memory_size=500,
            adaptation_threshold=0.15
        )
        
        assert config.update_frequency == 10
        assert config.learning_rate_decay == 0.9
        assert config.memory_size == 500
        assert config.adaptation_threshold == 0.15
    
    def test_online_learning_config_defaults(self):
        """测试在线学习配置默认值"""
        config = OnlineLearningConfig()
        
        assert config.update_frequency == 5
        assert config.learning_rate_decay == 0.95
        assert config.memory_size == 1000
        assert config.adaptation_threshold == 0.1


class TestEnsembleModelManager:
    """集成模型管理器测试"""
    
    def test_ensemble_manager_creation(self):
        """测试集成管理器创建"""
        mock_training_service = Mock()
        manager = EnsembleModelManager(mock_training_service)
        
        assert manager.training_service == mock_training_service
        assert manager.ensemble_dir.name == "ensembles"
    
    @pytest.mark.asyncio
    async def test_load_base_models_empty(self):
        """测试加载空的基础模型列表"""
        mock_training_service = Mock()
        manager = EnsembleModelManager(mock_training_service)
        
        models = await manager._load_base_models([])
        assert models == []
    
    def test_calculate_ensemble_metrics(self):
        """测试集成模型指标计算"""
        mock_training_service = Mock()
        manager = EnsembleModelManager(mock_training_service)
        
        y_true = np.array([1, 0, 1, 0, 1])
        y_pred = np.array([1, 0, 0, 0, 1])
        
        metrics = manager._calculate_ensemble_metrics(y_true, y_pred)
        
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert 0 <= metrics["accuracy"] <= 1


class TestOnlineLearningManager:
    """在线学习管理器测试"""
    
    def test_online_manager_creation(self):
        """测试在线学习管理器创建"""
        mock_training_service = Mock()
        manager = OnlineLearningManager(mock_training_service)
        
        assert manager.training_service == mock_training_service
        assert manager.memory_buffer == {}
        assert manager.model_performance_history == {}
    
    @pytest.mark.asyncio
    async def test_setup_online_learning(self):
        """测试在线学习设置"""
        mock_training_service = Mock()
        manager = OnlineLearningManager(mock_training_service)
        
        config = OnlineLearningConfig(
            update_frequency=5,
            learning_rate_decay=0.95,
            memory_size=1000,
            adaptation_threshold=0.1
        )
        
        result = await manager.setup_online_learning("test_model", config)
        
        assert result["model_id"] == "test_model"
        assert result["status"] == "active"
        assert "test_model" in manager.memory_buffer
        assert "test_model" in manager.model_performance_history
        
        # 检查缓冲区初始化
        buffer = manager.memory_buffer["test_model"]
        assert buffer["max_size"] == 1000
        assert len(buffer["data"]) == 0
    
    @pytest.mark.asyncio
    async def test_update_model_online_insufficient_data(self):
        """测试数据不足时的在线更新"""
        mock_training_service = Mock()
        manager = OnlineLearningManager(mock_training_service)
        
        config = OnlineLearningConfig()
        await manager.setup_online_learning("test_model", config)
        
        # 只提供少量数据
        X = np.random.rand(10, 20, 10)
        y = np.random.randint(0, 2, 10)
        mock_model = Mock()
        
        updated_model, metrics = await manager.update_model_online(
            "test_model", X, y, mock_model
        )
        
        assert updated_model == mock_model  # 模型未更新
        assert metrics["message"] == "insufficient_data"
    
    @pytest.mark.asyncio
    async def test_should_retrain_detection(self):
        """测试重训练检测"""
        mock_training_service = Mock()
        manager = OnlineLearningManager(mock_training_service)
        
        config = OnlineLearningConfig(adaptation_threshold=0.1)
        await manager.setup_online_learning("test_model", config)
        
        # 模拟性能历史
        history = manager.model_performance_history["test_model"]
        history["accuracies"] = [0.9, 0.85, 0.8, 0.75, 0.7]  # 性能下降
        
        current_metrics = {"accuracy": 0.65}
        should_retrain = await manager._should_retrain("test_model", current_metrics)
        
        assert should_retrain == True  # 性能下降超过阈值
        
        # 测试性能稳定的情况
        history["accuracies"] = [0.85, 0.84, 0.86, 0.85, 0.87]  # 性能稳定
        should_retrain = await manager._should_retrain("test_model", current_metrics)
        
        assert should_retrain == False  # 性能稳定，不需要重训练


# 运行测试的示例
if __name__ == "__main__":
    import asyncio
    
    async def run_basic_tests():
        """运行基本测试"""
        print("开始高级训练功能测试...")
        
        # 测试集成配置
        config = EnsembleConfig(
            method=EnsembleMethod.VOTING,
            base_models=["model1", "model2"]
        )
        print(f"✓ 集成配置创建成功: {config.method.value}")
        
        # 测试在线学习配置
        online_config = OnlineLearningConfig()
        print(f"✓ 在线学习配置创建成功: 更新频率={online_config.update_frequency}")
        
        # 测试服务初始化
        service = AdvancedTrainingService()
        print("✓ 高级训练服务创建成功")
        
        # 测试集成管理器
        mock_training_service = Mock()
        ensemble_manager = EnsembleModelManager(mock_training_service)
        print("✓ 集成模型管理器创建成功")
        
        # 测试在线学习管理器
        online_manager = OnlineLearningManager(mock_training_service)
        print("✓ 在线学习管理器创建成功")
        
        print("所有基本测试通过！")
    
    asyncio.run(run_basic_tests())