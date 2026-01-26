"""
预测引擎服务测试

测试核心预测逻辑、多时间维度预测和风险评估功能。
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path

from enum import Enum

from app.models.task_models import PredictionResult
from app.services.prediction import (
    ModelLoader,
    PredictionConfig,
    PredictionEngine,
)
from app.services.prediction.prediction_engine import RiskAssessment


class PredictionHorizon(Enum):
    """预测时间范围枚举（测试用）"""
    INTRADAY = "intraday"
    SHORT_TERM = "short_term"
    MEDIUM_TERM = "medium_term"
    LONG_TERM = "long_term"


class TestPredictionConfig:
    """预测配置测试"""
    
    def test_prediction_config_creation(self):
        """测试预测配置创建"""
        config = PredictionConfig(
            horizon=PredictionHorizon.SHORT_TERM,
            confidence_level=0.95,
            prediction_days=5
        )
        
        assert config.horizon == PredictionHorizon.SHORT_TERM
        assert config.confidence_level == 0.95
        assert config.prediction_days == 5
    
    def test_prediction_config_auto_days(self):
        """测试预测配置自动设置天数"""
        # 日内预测
        config_intraday = PredictionConfig(horizon=PredictionHorizon.INTRADAY)
        assert config_intraday.prediction_days == 1
        
        # 短期预测
        config_short = PredictionConfig(horizon=PredictionHorizon.SHORT_TERM)
        assert config_short.prediction_days == 5
        
        # 中期预测
        config_medium = PredictionConfig(horizon=PredictionHorizon.MEDIUM_TERM)
        assert config_medium.prediction_days == 20


class TestPredictionResult:
    """预测结果测试"""
    
    def test_prediction_result_creation(self):
        """测试预测结果创建"""
        result = PredictionResult(
            stock_code="000001.SZ",
            prediction_date=datetime.now(),
            horizon=PredictionHorizon.SHORT_TERM,
            predicted_direction=1,
            predicted_return=0.05,
            confidence_score=0.85,
            confidence_interval_lower=0.02,
            confidence_interval_upper=0.08,
            value_at_risk=-0.03,
            expected_shortfall=-0.05,
            volatility=0.2,
            model_id="test_model",
            model_version="1.0",
            created_at=datetime.now()
        )
        
        assert result.stock_code == "000001.SZ"
        assert result.predicted_direction == 1
        assert result.confidence_score == 0.85
    
    def test_prediction_result_to_dict(self):
        """测试预测结果转换为字典"""
        result = PredictionResult(
            stock_code="000001.SZ",
            prediction_date=datetime.now(),
            horizon=PredictionHorizon.SHORT_TERM,
            predicted_direction=1,
            predicted_return=0.05,
            confidence_score=0.85,
            confidence_interval_lower=0.02,
            confidence_interval_upper=0.08,
            value_at_risk=-0.03,
            expected_shortfall=-0.05,
            volatility=0.2,
            model_id="test_model",
            model_version="1.0",
            created_at=datetime.now()
        )
        
        result_dict = result.to_dict()
        
        assert result_dict["stock_code"] == "000001.SZ"
        assert result_dict["predicted_direction"] == 1
        assert "confidence_interval" in result_dict
        assert "risk_assessment" in result_dict
        assert "model_info" in result_dict


class TestRiskAssessment:
    """风险评估测试"""
    
    def test_calculate_var(self):
        """测试VaR计算"""
        # 创建测试收益率数据
        returns = np.array([-0.05, -0.03, -0.01, 0.01, 0.02, 0.03, 0.04, 0.05])
        
        var = RiskAssessment.calculate_var(returns, confidence_level=0.95)
        
        # VaR应该是负值（损失）
        assert var < 0
        assert isinstance(var, float)
    
    def test_calculate_expected_shortfall(self):
        """测试期望损失计算"""
        returns = np.array([-0.05, -0.03, -0.01, 0.01, 0.02, 0.03, 0.04, 0.05])
        
        es = RiskAssessment.calculate_expected_shortfall(returns, confidence_level=0.95)
        
        # 期望损失应该是负值且小于等于VaR
        var = RiskAssessment.calculate_var(returns, confidence_level=0.95)
        assert es <= var
        assert isinstance(es, float)
    
    def test_calculate_confidence_interval(self):
        """测试置信区间计算"""
        predictions = np.array([0.02, 0.03, 0.04, 0.05, 0.06])
        
        lower, upper = RiskAssessment.calculate_confidence_interval(
            predictions, confidence_level=0.95
        )
        
        assert lower < upper
        assert isinstance(lower, float)
        assert isinstance(upper, float)
    
    def test_empty_data_handling(self):
        """测试空数据处理"""
        empty_returns = np.array([])
        
        var = RiskAssessment.calculate_var(empty_returns)
        es = RiskAssessment.calculate_expected_shortfall(empty_returns)
        lower, upper = RiskAssessment.calculate_confidence_interval(empty_returns)
        
        assert var == 0.0
        assert es == 0.0
        assert lower == 0.0
        assert upper == 0.0


class TestModelLoader:
    """模型加载器测试"""
    
    def test_model_loader_creation(self):
        """测试模型加载器创建"""
        with tempfile.TemporaryDirectory() as temp_dir:
            models_dir = Path(temp_dir)
            loader = ModelLoader(models_dir)
            
            assert loader.models_dir == models_dir
            assert loader.loaded_models == {}
    
    def test_get_model_info(self):
        """测试获取模型信息"""
        with tempfile.TemporaryDirectory() as temp_dir:
            models_dir = Path(temp_dir)
            loader = ModelLoader(models_dir)
            
            model_info = loader.get_model_info("test_model")
            
            assert model_info["model_id"] == "test_model"
            assert "version" in model_info
            assert "created_at" in model_info
    
    def test_load_model_file_not_found(self):
        """测试模型文件不存在的情况"""
        with tempfile.TemporaryDirectory() as temp_dir:
            models_dir = Path(temp_dir)
            loader = ModelLoader(models_dir)
            
            with pytest.raises(FileNotFoundError):
                loader.load_model("nonexistent_model", "xgboost")


class TestPredictionEngine:
    """预测引擎测试"""
    
    def test_prediction_engine_creation(self):
        """测试预测引擎创建"""
        engine = PredictionEngine()
        
        assert engine.models_dir.name == "models"
        assert engine.model_loader is not None
        assert engine.risk_assessment is not None
    
    def test_infer_model_type(self):
        """测试模型类型推断"""
        engine = PredictionEngine()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            engine.models_dir = Path(temp_dir)
            
            # 创建测试模型文件
            xgb_file = engine.models_dir / "test_xgb_model.json"
            xgb_file.touch()
            
            pytorch_file = engine.models_dir / "test_pytorch_model.pth"
            pytorch_file.touch()
            
            # 测试类型推断
            assert engine._infer_model_type("test_xgb_model") == "xgboost"
            assert engine._infer_model_type("test_pytorch_model") == "pytorch"
            assert engine._infer_model_type("nonexistent_model") == "unknown"
    
    def test_calculate_predicted_return(self):
        """测试预测收益率计算"""
        engine = PredictionEngine()
        
        # 测试上涨预测
        return_up = engine._calculate_predicted_return(
            direction=1, 
            confidence=0.8, 
            horizon=PredictionHorizon.SHORT_TERM
        )
        assert return_up > 0
        
        # 测试下跌预测
        return_down = engine._calculate_predicted_return(
            direction=0, 
            confidence=0.8, 
            horizon=PredictionHorizon.SHORT_TERM
        )
        assert return_down < 0
        
        # 测试不同时间维度
        return_intraday = engine._calculate_predicted_return(
            direction=1, 
            confidence=0.8, 
            horizon=PredictionHorizon.INTRADAY
        )
        return_medium = engine._calculate_predicted_return(
            direction=1, 
            confidence=0.8, 
            horizon=PredictionHorizon.MEDIUM_TERM
        )
        
        # 中期预测的收益率应该大于日内预测
        assert abs(return_medium) > abs(return_intraday)
    
    @pytest.mark.asyncio
    async def test_make_prediction_xgboost(self):
        """测试XGBoost模型预测"""
        engine = PredictionEngine()
        
        # 创建模拟的XGBoost模型
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0.75])  # 预测概率
        
        # 创建测试输入
        X = np.random.rand(1, 60, 10)
        
        proba, direction = await engine._make_prediction(mock_model, X, "xgboost")
        
        assert proba == 0.75
        assert direction == 1  # 0.75 > 0.5，所以是上涨
        mock_model.predict.assert_called_once()
    
    def test_calculate_prediction_confidence_interval(self):
        """测试预测置信区间计算"""
        engine = PredictionEngine()
        
        lower, upper = engine._calculate_prediction_confidence_interval(
            predicted_return=0.05,
            volatility=0.2,
            confidence_level=0.95
        )
        
        assert lower < 0.05 < upper
        assert isinstance(lower, float)
        assert isinstance(upper, float)


# 运行测试的示例
if __name__ == "__main__":
    import asyncio
    
    async def run_basic_tests():
        """运行基本测试"""
        print("开始预测引擎功能测试...")
        
        # 测试预测配置
        config = PredictionConfig(horizon=PredictionHorizon.SHORT_TERM)
        print(f"✓ 预测配置创建成功: {config.horizon.value}, 预测天数: {config.prediction_days}")
        
        # 测试风险评估
        returns = np.array([-0.05, -0.03, -0.01, 0.01, 0.02, 0.03, 0.04, 0.05])
        var = RiskAssessment.calculate_var(returns)
        print(f"✓ VaR计算成功: {var:.4f}")
        
        # 测试预测引擎
        engine = PredictionEngine()
        print("✓ 预测引擎创建成功")
        
        # 测试预测收益率计算
        predicted_return = engine._calculate_predicted_return(
            direction=1, confidence=0.8, horizon=PredictionHorizon.SHORT_TERM
        )
        print(f"✓ 预测收益率计算成功: {predicted_return:.4f}")
        
        print("所有基本测试通过！")
    
    asyncio.run(run_basic_tests())