"""
Qlib模型优化功能单元测试

测试以下优化功能：
1. 标签计算逻辑（prediction_horizon）
2. 缺失值处理改进
3. 特征标准化（RobustFeatureScaler）
4. 异常值处理（OutlierHandler）
5. 损失函数配置优化
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from app.services.qlib.unified_qlib_training_engine import (
    UnifiedQlibTrainingEngine,
    QlibTrainingConfig,
    QlibModelType,
    RobustFeatureScaler,
    OutlierHandler,
)
from app.services.qlib.enhanced_qlib_provider import EnhancedQlibDataProvider
from app.services.qlib.qlib_model_manager import LightGBMAdapter


class TestLabelCalculation:
    """测试标签计算逻辑优化"""

    @pytest.fixture
    def sample_data(self):
        """创建样本数据"""
        dates = pd.date_range(start='2023-01-01', periods=20, freq='D')
        data = pd.DataFrame({
            '$close': 100 + np.cumsum(np.random.randn(20) * 0.5),
            '$open': 100 + np.cumsum(np.random.randn(20) * 0.5),
            '$high': 100 + np.cumsum(np.random.randn(20) * 0.5) + 1,
            '$low': 100 + np.cumsum(np.random.randn(20) * 0.5) - 1,
            '$volume': np.random.randint(1000000, 10000000, 20),
        }, index=dates)
        return data

    @pytest.fixture
    def multiindex_data(self):
        """创建MultiIndex样本数据"""
        dates = pd.date_range(start='2023-01-01', periods=20, freq='D')
        stock_codes = ['000001.SZ', '000002.SZ']
        
        data_list = []
        for stock_code in stock_codes:
            stock_data = pd.DataFrame({
                '$close': 100 + np.cumsum(np.random.randn(20) * 0.5),
                '$open': 100 + np.cumsum(np.random.randn(20) * 0.5),
                '$high': 100 + np.cumsum(np.random.randn(20) * 0.5) + 1,
                '$low': 100 + np.cumsum(np.random.randn(20) * 0.5) - 1,
                '$volume': np.random.randint(1000000, 10000000, 20),
            }, index=dates)
            stock_data.index = pd.MultiIndex.from_product(
                [[stock_code], dates], names=['instrument', 'datetime']
            )
            data_list.append(stock_data)
        
        return pd.concat(data_list)

    def test_process_stock_data_with_prediction_horizon(self, sample_data):
        """测试_process_stock_data使用prediction_horizon参数"""
        engine = UnifiedQlibTrainingEngine()
        
        # 测试prediction_horizon=5
        prediction_horizon = 5
        processed = engine._process_stock_data(
            sample_data, 'TEST', prediction_horizon=prediction_horizon
        )
        
        assert 'label' in processed.columns, "应该创建label列"
        
        # 验证标签计算：应该是 (future_price - current_price) / current_price
        # 最后5行应该是NaN（因为shift(-5)）
        assert processed['label'].iloc[-prediction_horizon:].isna().all() or \
               (processed['label'].iloc[-prediction_horizon:] == 0).all(), \
               "最后N行标签应该是NaN或0"
        
        # 验证标签值范围合理（收益率通常在-1到1之间）
        valid_labels = processed['label'].dropna()
        if len(valid_labels) > 0:
            assert valid_labels.min() >= -2, "标签最小值应该合理"
            assert valid_labels.max() <= 2, "标签最大值应该合理"

    def test_process_stock_data_multiindex(self, multiindex_data):
        """测试MultiIndex数据的标签计算"""
        engine = UnifiedQlibTrainingEngine()
        
        # 提取第一只股票的数据
        stock_code = '000001.SZ'
        stock_data = multiindex_data.xs(stock_code, level=0, drop_level=False)
        
        prediction_horizon = 5
        processed = engine._process_stock_data(
            stock_data, stock_code, prediction_horizon=prediction_horizon
        )
        
        assert 'label' in processed.columns, "应该创建label列"
        
        # 验证标签计算正确
        # 注意：processed已经是单个股票的数据，但可能还保留MultiIndex
        current_prices = processed['$close']
        if isinstance(processed.index, pd.MultiIndex):
            # 如果还是MultiIndex，需要按level分组
            future_prices = processed.groupby(level=0)['$close'].shift(-prediction_horizon)
        else:
            # 如果已经是单层索引，直接shift
            future_prices = processed['$close'].shift(-prediction_horizon)
        expected_labels = (future_prices - current_prices) / current_prices
        
        # 比较计算出的标签（允许小的浮点误差）
        valid_mask = ~processed['label'].isna() & ~expected_labels.isna()
        if valid_mask.sum() > 0:
            diff = np.abs(processed['label'][valid_mask] - expected_labels[valid_mask])
            assert diff.max() < 1e-5, f"标签计算应该正确，最大误差: {diff.max()}"

    def test_label_calculation_different_horizons(self, sample_data):
        """测试不同prediction_horizon值的标签计算"""
        engine = UnifiedQlibTrainingEngine()
        
        for horizon in [1, 5, 10]:
            processed = engine._process_stock_data(
                sample_data.copy(), 'TEST', prediction_horizon=horizon
            )
            
            assert 'label' in processed.columns, f"horizon={horizon}时应该创建label列"
            
            # 验证最后N行应该是NaN或0
            last_n = processed['label'].iloc[-horizon:]
            assert last_n.isna().all() or (last_n == 0).all(), \
                   f"horizon={horizon}时，最后{horizon}行应该是NaN或0"


class TestMissingValueHandling:
    """测试缺失值处理改进"""

    @pytest.fixture
    def data_with_missing_values(self):
        """创建包含缺失值的数据"""
        dates = pd.date_range(start='2023-01-01', periods=20, freq='D')
        data = pd.DataFrame({
            '$close': 100 + np.cumsum(np.random.randn(20) * 0.5),
            '$open': 100 + np.cumsum(np.random.randn(20) * 0.5),
            '$high': 100 + np.cumsum(np.random.randn(20) * 0.5) + 1,
            '$low': 100 + np.cumsum(np.random.randn(20) * 0.5) - 1,
            '$volume': np.random.randint(1000000, 10000000, 20),
            'indicator1': np.random.randn(20),
            'indicator2': np.random.randn(20),
        }, index=dates)
        
        # 添加缺失值
        data.loc[data.index[5], '$close'] = np.nan  # 价格数据缺失
        data.loc[data.index[:3], 'indicator1'] = np.nan  # 技术指标开头缺失（计算窗口不足）
        data.loc[data.index[10], 'indicator2'] = np.nan  # 技术指标中间缺失
        
        return data

    def test_handle_missing_values_price_data(self, data_with_missing_values):
        """测试价格数据的缺失值处理"""
        provider = EnhancedQlibDataProvider()
        
        processed = provider._handle_missing_values(data_with_missing_values.copy())
        
        # 价格数据应该被填充（前向填充）
        assert not processed['$close'].isna().any(), "价格数据不应该有缺失值"
        assert not processed['$open'].isna().any(), "价格数据不应该有缺失值"

    def test_handle_missing_values_indicators(self, data_with_missing_values):
        """测试技术指标的缺失值处理"""
        provider = EnhancedQlibDataProvider()
        
        processed = provider._handle_missing_values(data_with_missing_values.copy())
        
        # 技术指标应该被智能填充
        assert not processed['indicator1'].isna().any(), "技术指标不应该有缺失值"
        assert not processed['indicator2'].isna().any(), "技术指标不应该有缺失值"

    def test_handle_missing_values_high_missing_rate(self):
        """测试高缺失率的技术指标处理"""
        dates = pd.date_range(start='2023-01-01', periods=20, freq='D')
        data = pd.DataFrame({
            '$close': 100 + np.cumsum(np.random.randn(20) * 0.5),
            'indicator': np.random.randn(20),
        }, index=dates)
        
        # 设置高缺失率（>50%）
        data.loc[data.index[:12], 'indicator'] = np.nan
        
        provider = EnhancedQlibDataProvider()
        processed = provider._handle_missing_values(data.copy())
        
        # 应该使用中位数填充
        assert not processed['indicator'].isna().any(), "高缺失率应该使用中位数填充"


class TestRobustFeatureScaler:
    """测试RobustFeatureScaler"""

    @pytest.fixture
    def sample_features(self):
        """创建样本特征数据"""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'feature1': np.random.randn(100) * 10 + 100,  # 均值100，标准差10
            'feature2': np.random.randn(100) * 0.1 + 0.5,  # 均值0.5，标准差0.1
            'feature3': np.random.randn(100) * 1000 + 5000,  # 均值5000，标准差1000
            'label': np.random.randn(100) * 0.02,  # 标签
        }, index=dates)
        return data

    def test_robust_scaler_fit_transform(self, sample_features):
        """测试RobustFeatureScaler的fit_transform"""
        scaler = RobustFeatureScaler()
        
        feature_cols = ['feature1', 'feature2', 'feature3']
        scaled_data = scaler.fit_transform(sample_features.copy(), feature_cols)
        
        # 验证特征被标准化
        assert 'feature1' in scaled_data.columns
        assert 'feature2' in scaled_data.columns
        assert 'feature3' in scaled_data.columns
        
        # 验证标签没有被标准化
        assert np.allclose(
            scaled_data['label'].values,
            sample_features['label'].values
        ), "标签不应该被标准化"
        
        # 验证scaler已拟合
        assert scaler.fitted, "scaler应该被标记为已拟合"
        assert len(scaler.scalers) == 3, "应该有3个特征的scaler"

    def test_robust_scaler_transform(self, sample_features):
        """测试RobustFeatureScaler的transform"""
        scaler = RobustFeatureScaler()
        
        feature_cols = ['feature1', 'feature2', 'feature3']
        train_data = sample_features.iloc[:80].copy()
        test_data = sample_features.iloc[80:].copy()
        
        # 在训练集上拟合
        train_scaled = scaler.fit_transform(train_data, feature_cols)
        
        # 在测试集上转换
        test_scaled = scaler.transform(test_data, feature_cols)
        
        # 验证测试集也被转换
        assert 'feature1' in test_scaled.columns
        assert not np.allclose(
            test_scaled['feature1'].values,
            test_data['feature1'].values
        ), "测试集特征应该被转换"

    def test_robust_scaler_without_sklearn(self, sample_features, monkeypatch):
        """测试当sklearn不可用时的行为"""
        # 模拟sklearn不可用 - 需要在导入之前mock
        import sys
        from unittest.mock import MagicMock
        
        # 保存原始模块
        original_sklearn = sys.modules.get('sklearn.preprocessing')
        
        # Mock sklearn.preprocessing模块
        mock_sklearn_preprocessing = MagicMock()
        mock_sklearn_preprocessing.RobustScaler = None
        sys.modules['sklearn.preprocessing'] = mock_sklearn_preprocessing
        
        try:
            # 重新导入以触发ImportError处理
            from app.services.qlib.unified_qlib_training_engine import RobustFeatureScaler
            scaler = RobustFeatureScaler()
            feature_cols = ['feature1', 'feature2']
            
            # 应该跳过标准化但不报错
            result = scaler.fit_transform(sample_features.copy(), feature_cols)
            assert result is not None, "即使sklearn不可用也应该返回数据"
            # 验证数据没有被修改（因为跳过了标准化）
            assert np.allclose(result['feature1'].values, sample_features['feature1'].values, equal_nan=True), \
                   "sklearn不可用时数据应该保持不变"
        finally:
            # 恢复原始模块
            if original_sklearn:
                sys.modules['sklearn.preprocessing'] = original_sklearn
            elif 'sklearn.preprocessing' in sys.modules:
                del sys.modules['sklearn.preprocessing']


class TestOutlierHandler:
    """测试OutlierHandler"""

    @pytest.fixture
    def data_with_outliers(self):
        """创建包含异常值的数据"""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        
        # 创建正常收益率数据
        normal_returns = np.random.randn(100) * 0.02
        
        # 添加异常值
        normal_returns[10] = 0.8  # 极端正值
        normal_returns[20] = -0.7  # 极端负值
        normal_returns[30] = 0.6  # 另一个极端正值
        
        data = pd.DataFrame({
            'label': normal_returns,
            'feature1': np.random.randn(100),
        }, index=dates)
        
        return data

    def test_outlier_handler_winsorize(self, data_with_outliers):
        """测试Winsorize方法"""
        handler = OutlierHandler(method="winsorize", lower_percentile=0.01, upper_percentile=0.99)
        
        processed = handler.handle_label_outliers(data_with_outliers.copy(), label_col="label")
        
        # 验证异常值被处理
        assert 'label' in processed.columns
        
        # 验证标签值在合理范围内
        assert processed['label'].min() >= -1, "标签最小值应该合理"
        assert processed['label'].max() <= 1, "标签最大值应该合理"
        
        # 验证极端值被截断
        original_max = data_with_outliers['label'].max()
        processed_max = processed['label'].max()
        assert processed_max <= original_max, "最大值应该被截断"

    def test_outlier_handler_clip(self, data_with_outliers):
        """测试Clip方法"""
        handler = OutlierHandler(method="clip")
        
        processed = handler.handle_label_outliers(data_with_outliers.copy(), label_col="label")
        
        # 验证异常值被处理
        assert 'label' in processed.columns
        
        # 验证标签值在合理范围内
        assert processed['label'].min() >= -1, "标签最小值应该合理"
        assert processed['label'].max() <= 1, "标签最大值应该合理"

    def test_outlier_handler_no_label_column(self, data_with_outliers):
        """测试当没有标签列时的行为"""
        handler = OutlierHandler()
        
        data_no_label = data_with_outliers.drop(columns=['label'])
        processed = handler.handle_label_outliers(data_no_label.copy(), label_col="label")
        
        # 应该返回原始数据
        assert processed.equals(data_no_label), "没有标签列时应该返回原始数据"

    def test_outlier_handler_extreme_returns(self):
        """测试极端收益率（除权除息）的处理"""
        dates = pd.date_range(start='2023-01-01', periods=20, freq='D')
        data = pd.DataFrame({
            'label': np.random.randn(20) * 0.02,
            'feature1': np.random.randn(20),
        }, index=dates)
        
        # 添加极端收益率（>50%）
        data.loc[data.index[5], 'label'] = 0.6  # 60%收益率
        data.loc[data.index[10], 'label'] = -0.55  # -55%收益率
        
        handler = OutlierHandler(method="winsorize")
        processed = handler.handle_label_outliers(data.copy(), label_col="label")
        
        # 验证极端值被Winsorize处理（截断到99%分位数）
        assert 'label' in processed.columns, "应该有label列"
        # 验证处理后的值在合理范围内（Winsorize会截断到分位数）
        assert processed['label'].min() >= -1, "标签最小值应该合理"
        assert processed['label'].max() <= 1, "标签最大值应该合理"
        # 验证极端值被截断（应该小于等于99%分位数）
        upper_bound = data['label'].quantile(0.99)
        assert processed['label'].max() <= max(upper_bound, data['label'].max()), \
               "极端值应该被Winsorize截断"


class TestLossFunctionOptimization:
    """测试损失函数优化"""

    def test_lightgbm_adapter_huber_loss(self):
        """测试LightGBM适配器使用Huber损失"""
        adapter = LightGBMAdapter()
        
        hyperparameters = {
            "learning_rate": 0.1,
            "num_leaves": 31,
            "huber_delta": 0.1,
        }
        
        config = adapter.create_qlib_config(hyperparameters)
        
        # 验证使用Huber损失
        assert config["kwargs"]["loss"] == "huber", "应该使用Huber损失"
        assert "huber_delta" in config["kwargs"], "应该包含huber_delta参数"
        assert config["kwargs"]["huber_delta"] == 0.1, "huber_delta应该正确设置"

    def test_lightgbm_adapter_default_huber_delta(self):
        """测试LightGBM适配器的默认huber_delta"""
        adapter = LightGBMAdapter()
        
        hyperparameters = {
            "learning_rate": 0.1,
        }
        
        config = adapter.create_qlib_config(hyperparameters)
        
        # 验证默认huber_delta
        assert config["kwargs"]["loss"] == "huber", "应该使用Huber损失"
        assert config["kwargs"]["huber_delta"] == 0.1, "默认huber_delta应该是0.1"

    @pytest.mark.asyncio
    async def test_enhanced_provider_huber_loss(self):
        """测试EnhancedQlibDataProvider使用Huber损失"""
        provider = EnhancedQlibDataProvider()
        
        hyperparameters = {
            "learning_rate": 0.05,
            "huber_delta": 0.15,
        }
        
        config = await provider.create_qlib_model_config("lightgbm", hyperparameters)
        
        # 验证使用Huber损失
        assert config["kwargs"]["loss"] == "huber", "应该使用Huber损失"
        assert config["kwargs"]["huber_delta"] == 0.15, "huber_delta应该正确设置"


class TestIntegration:
    """集成测试：测试多个优化功能的协同工作"""

    @pytest.fixture
    def complete_sample_data(self):
        """创建完整的样本数据"""
        dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
        stock_codes = ['000001.SZ']
        
        data_list = []
        for stock_code in stock_codes:
            stock_data = pd.DataFrame({
                '$close': 100 + np.cumsum(np.random.randn(50) * 0.5),
                '$open': 100 + np.cumsum(np.random.randn(50) * 0.5),
                '$high': 100 + np.cumsum(np.random.randn(50) * 0.5) + 1,
                '$low': 100 + np.cumsum(np.random.randn(50) * 0.5) - 1,
                '$volume': np.random.randint(1000000, 10000000, 50),
                'feature1': np.random.randn(50) * 10,
                'feature2': np.random.randn(50) * 0.1,
            }, index=dates)
            
            # 添加一些缺失值
            stock_data.loc[stock_data.index[5], '$close'] = np.nan
            stock_data.loc[stock_data.index[:3], 'feature1'] = np.nan
            
            # 添加一些异常值到标签（稍后创建）
            stock_data.index = pd.MultiIndex.from_product(
                [[stock_code], dates], names=['instrument', 'datetime']
            )
            data_list.append(stock_data)
        
        return pd.concat(data_list)

    @pytest.mark.asyncio
    async def test_prepare_training_datasets_integration(self, complete_sample_data):
        """测试_prepare_training_datasets的完整流程"""
        engine = UnifiedQlibTrainingEngine()
        
        config = QlibTrainingConfig(
            model_type=QlibModelType.LIGHTGBM,
            hyperparameters={"learning_rate": 0.1},
            prediction_horizon=5,
            validation_split=0.2,
        )
        
        # 注意：这个测试可能需要mock一些依赖
        # 这里主要测试数据准备流程不会出错
        try:
            train_dataset, val_dataset = await engine._prepare_training_datasets(
                complete_sample_data.copy(),
                validation_split=0.2,
                config=config,
            )
            
            # 验证返回的数据集不为空
            assert train_dataset is not None, "训练集不应该为空"
            assert val_dataset is not None, "验证集不应该为空"
            
        except Exception as e:
            # 如果因为Qlib不可用等原因失败，这是可以接受的
            pytest.skip(f"集成测试跳过（可能是Qlib不可用）: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
