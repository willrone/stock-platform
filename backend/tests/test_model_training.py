"""
模型训练服务测试
"""

import pytest
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from app.services.model_training import (
    ModelTrainingService, 
    TrainingConfig, 
    ModelType, 
    QlibDataProvider,
    ModelMetrics
)


class TestModelTrainingService:
    """模型训练服务测试类"""
    
    @pytest.fixture
    async def service(self):
        """创建模型训练服务实例"""
        service = ModelTrainingService()
        # 模拟初始化，避免实际的Qlib初始化
        service.data_provider = Mock()
        return service
    
    @pytest.fixture
    def training_config(self):
        """创建训练配置"""
        return TrainingConfig(
            model_type=ModelType.LSTM,
            sequence_length=30,
            prediction_horizon=5,
            batch_size=16,
            epochs=10,
            learning_rate=0.001
        )
    
    @pytest.fixture
    def sample_features_data(self):
        """创建样本特征数据"""
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        data = []
        
        for stock_code in ['000001.SZ', '000002.SZ']:
            for date in dates:
                data.append({
                    'date': date,
                    'stock_code': stock_code,
                    'open': 10.0 + np.random.normal(0, 0.5),
                    'high': 10.5 + np.random.normal(0, 0.5),
                    'low': 9.5 + np.random.normal(0, 0.5),
                    'close': 10.0 + np.random.normal(0, 0.5),
                    'volume': 1000000 + np.random.randint(-100000, 100000),
                    'ma_5': 10.0 + np.random.normal(0, 0.3),
                    'ma_10': 10.0 + np.random.normal(0, 0.3),
                    'ma_20': 10.0 + np.random.normal(0, 0.3),
                    'ma_60': 10.0 + np.random.normal(0, 0.3),
                    'rsi': 50 + np.random.normal(0, 10),
                    'macd': np.random.normal(0, 0.1),
                    'macd_signal': np.random.normal(0, 0.1),
                    'bb_upper': 10.5 + np.random.normal(0, 0.3),
                    'bb_lower': 9.5 + np.random.normal(0, 0.3)
                })
        
        return pd.DataFrame(data)
    
    def test_training_config_creation(self):
        """测试训练配置创建"""
        config = TrainingConfig(model_type=ModelType.XGBOOST)
        
        assert config.model_type == ModelType.XGBOOST
        assert config.sequence_length == 60
        assert config.prediction_horizon == 5
        assert len(config.feature_columns) > 0
        assert 'close' in config.feature_columns
    
    def test_model_metrics_to_dict(self):
        """测试模型指标转换为字典"""
        metrics = ModelMetrics(
            accuracy=0.85,
            precision=0.80,
            recall=0.75,
            sharpe_ratio=1.2,
            max_drawdown=-0.15,
            total_return=0.25,
            win_rate=0.60
        )
        
        metrics_dict = metrics.to_dict()
        
        assert metrics_dict['accuracy'] == 0.85
        assert metrics_dict['sharpe_ratio'] == 1.2
        assert len(metrics_dict) == 7
    
    def test_prepare_training_data(self, service, training_config, sample_features_data):
        """测试训练数据准备"""
        X, y = service._prepare_training_data(sample_features_data, training_config)
        
        # 检查数据形状
        assert len(X.shape) == 3  # (samples, sequence_length, features)
        assert X.shape[1] == training_config.sequence_length
        assert len(y.shape) == 1  # (samples,)
        assert len(X) == len(y)
        
        # 检查标签值
        assert all(label in [0, 1] for label in y)
    
    def test_time_series_split(self, service):
        """测试时间序列数据分割"""
        # 创建测试数据
        X = np.random.rand(100, 30, 10)
        y = np.random.randint(0, 2, 100)
        
        train_X, train_y, val_X, val_y = service._time_series_split(X, y, 0.2)
        
        # 检查分割比例
        assert len(train_X) == 80
        assert len(val_X) == 20
        assert len(train_y) == 80
        assert len(val_y) == 20
        
        # 检查数据形状保持一致
        assert train_X.shape[1:] == X.shape[1:]
        assert val_X.shape[1:] == X.shape[1:]
    
    def test_calculate_metrics(self, service):
        """测试指标计算"""
        # 创建测试预测结果
        y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
        y_pred = np.array([1, 0, 0, 1, 0, 1, 1, 0, 1, 0])
        y_pred_proba = np.array([0.8, 0.2, 0.4, 0.9, 0.1, 0.7, 0.6, 0.3, 0.8, 0.2])
        
        metrics = service._calculate_metrics(y_true, y_pred, y_pred_proba)
        
        # 检查指标范围
        assert 0 <= metrics.accuracy <= 1
        assert 0 <= metrics.precision <= 1
        assert 0 <= metrics.recall <= 1
        assert metrics.max_drawdown <= 0
        assert 0 <= metrics.win_rate <= 1
    
    @patch('app.services.model_training.xgb')
    async def test_train_xgboost_model(self, mock_xgb, service, training_config):
        """测试XGBoost模型训练"""
        # 模拟XGBoost
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0.6, 0.4, 0.8, 0.3, 0.7])
        mock_xgb.train.return_value = mock_model
        mock_xgb.DMatrix = Mock()
        
        # 创建测试数据
        train_X = np.random.rand(50, 30, 10)
        train_y = np.random.randint(0, 2, 50)
        val_X = np.random.rand(20, 30, 10)
        val_y = np.random.randint(0, 2, 20)
        
        training_config.model_type = ModelType.XGBOOST
        
        model, metrics = await service._train_xgboost(
            train_X, train_y, val_X, val_y, training_config
        )
        
        # 验证XGBoost被调用
        assert mock_xgb.train.called
        assert mock_xgb.DMatrix.called
        assert isinstance(metrics, ModelMetrics)
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.get_device_name')
    def test_get_device_rocm(self, mock_get_device_name, mock_cuda_available):
        """测试ROCm设备检测"""
        from app.services.model_training import get_device
        
        mock_cuda_available.return_value = True
        mock_get_device_name.return_value = "AMD Radeon RX 7900 XTX"
        
        device = get_device()
        
        assert device.type == 'cuda'
        mock_get_device_name.assert_called_once_with(0)
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.get_device_name')
    def test_get_device_nvidia(self, mock_get_device_name, mock_cuda_available):
        """测试NVIDIA GPU设备检测"""
        from app.services.model_training import get_device
        
        mock_cuda_available.return_value = True
        mock_get_device_name.return_value = "NVIDIA GeForce RTX 4090"
        
        device = get_device()
        
        assert device.type == 'cuda'
        mock_get_device_name.assert_called_once_with(0)
    
    @patch('app.services.model_training.get_device')
    async def test_train_deep_learning_model(self, mock_get_device, service, training_config):
        """测试深度学习模型训练"""
        mock_get_device.return_value = torch.device('cpu')  # 使用CPU进行测试
        
        # 创建小规模测试数据
        train_X = np.random.rand(20, 10, 5)
        train_y = np.random.randint(0, 2, 20)
        val_X = np.random.rand(10, 10, 5)
        val_y = np.random.randint(0, 2, 10)
        
        training_config.epochs = 2  # 减少训练轮数
        training_config.batch_size = 5
        
        model, metrics = await service._train_deep_learning_model(
            train_X, train_y, val_X, val_y, training_config
        )
        
        # 验证返回结果
        assert model is not None
        assert isinstance(metrics, ModelMetrics)
        assert 0 <= metrics.accuracy <= 1


class TestQlibDataProvider:
    """Qlib数据提供器测试类"""
    
    @pytest.fixture
    def data_provider(self):
        """创建数据提供器实例"""
        mock_data_service = Mock()
        return QlibDataProvider(mock_data_service)
    
    def test_add_fundamental_features(self, data_provider):
        """测试基本面特征添加"""
        # 创建测试数据
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=100),
            'close': np.random.rand(100) * 10 + 10,
            'volume': np.random.randint(1000000, 5000000, 100),
            'high': np.random.rand(100) * 10 + 12,
            'low': np.random.rand(100) * 10 + 8
        })
        
        result_df = data_provider._add_fundamental_features(df)
        
        # 检查新增特征
        expected_features = [
            'price_change', 'price_change_5d', 'price_change_20d',
            'volume_change', 'volume_ma_ratio',
            'volatility_5d', 'volatility_20d', 'price_position'
        ]
        
        for feature in expected_features:
            assert feature in result_df.columns
        
        # 检查数据长度不变
        assert len(result_df) == len(df)


@pytest.mark.asyncio
async def test_model_training_integration():
    """集成测试：模型训练完整流程"""
    # 这是一个简化的集成测试
    service = ModelTrainingService()
    
    # 模拟数据提供器
    mock_data_provider = Mock()
    mock_features_df = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=100),
        'stock_code': ['000001.SZ'] * 100,
        'open': np.random.rand(100) * 10 + 10,
        'high': np.random.rand(100) * 10 + 12,
        'low': np.random.rand(100) * 10 + 8,
        'close': np.random.rand(100) * 10 + 10,
        'volume': np.random.randint(1000000, 5000000, 100),
        'ma_5': np.random.rand(100) * 10 + 10,
        'ma_10': np.random.rand(100) * 10 + 10,
        'ma_20': np.random.rand(100) * 10 + 10,
        'ma_60': np.random.rand(100) * 10 + 10,
        'rsi': np.random.rand(100) * 100,
        'macd': np.random.rand(100) * 0.2 - 0.1,
        'macd_signal': np.random.rand(100) * 0.2 - 0.1,
        'bb_upper': np.random.rand(100) * 10 + 12,
        'bb_lower': np.random.rand(100) * 10 + 8
    })
    
    mock_data_provider.prepare_features = AsyncMock(return_value=mock_features_df)
    service.data_provider = mock_data_provider
    
    # 创建训练配置
    config = TrainingConfig(
        model_type=ModelType.XGBOOST,
        sequence_length=20,
        epochs=5
    )
    
    # 模拟保存模型方法
    service._save_model = AsyncMock(return_value="/path/to/model.json")
    
    # 执行训练（这里会调用实际的XGBoost训练）
    with patch('app.services.model_training.xgb') as mock_xgb:
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0.6, 0.4, 0.8])
        mock_xgb.train.return_value = mock_model
        mock_xgb.DMatrix = Mock()
        
        model_path, metrics = await service.train_model(
            model_id="test_model",
            stock_codes=["000001.SZ"],
            config=config,
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31)
        )
        
        # 验证结果
        assert model_path == "/path/to/model.json"
        assert isinstance(metrics, ModelMetrics)
        assert 0 <= metrics.accuracy <= 1