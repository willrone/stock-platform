"""
预测引擎服务

集成训练好的模型进行预测，实现多时间维度预测和风险评估。
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PredictionHorizon(Enum):
    """预测时间维度"""
    INTRADAY = "intraday"      # 日内预测
    SHORT_TERM = "short_term"  # 短期预测（1-5天）
    MEDIUM_TERM = "medium_term" # 中期预测（1-4周）


@dataclass
class PredictionConfig:
    """预测配置"""
    horizon: PredictionHorizon
    confidence_level: float = 0.95  # 置信水平
    risk_free_rate: float = 0.03    # 无风险利率
    prediction_days: int = 5        # 预测天数
    use_ensemble: bool = False      # 是否使用集成模型
    
    def __post_init__(self):
        # 根据预测时间维度设置默认预测天数
        if self.horizon == PredictionHorizon.INTRADAY:
            self.prediction_days = 1
        elif self.horizon == PredictionHorizon.SHORT_TERM:
            self.prediction_days = 5
        elif self.horizon == PredictionHorizon.MEDIUM_TERM:
            self.prediction_days = 20


@dataclass
class PredictionResult:
    """预测结果"""
    stock_code: str
    prediction_date: datetime
    horizon: PredictionHorizon
    
    # 预测值
    predicted_direction: int  # 0=下跌, 1=上涨
    predicted_return: float   # 预测收益率
    confidence_score: float   # 置信度分数
    
    # 置信区间
    confidence_interval_lower: float
    confidence_interval_upper: float
    
    # 风险评估
    value_at_risk: float      # VaR值
    expected_shortfall: float # 期望损失
    volatility: float         # 预测波动率
    
    # 元数据
    model_id: str
    model_version: str
    created_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "stock_code": self.stock_code,
            "prediction_date": self.prediction_date.isoformat(),
            "horizon": self.horizon.value,
            "predicted_direction": self.predicted_direction,
            "predicted_return": self.predicted_return,
            "confidence_score": self.confidence_score,
            "confidence_interval": {
                "lower": self.confidence_interval_lower,
                "upper": self.confidence_interval_upper
            },
            "risk_assessment": {
                "value_at_risk": self.value_at_risk,
                "expected_shortfall": self.expected_shortfall,
                "volatility": self.volatility
            },
            "model_info": {
                "model_id": self.model_id,
                "model_version": self.model_version
            },
            "created_at": self.created_at.isoformat()
        }


class RiskAssessment:
    """风险评估计算器"""
    
    @staticmethod
    def calculate_var(returns: np.ndarray, confidence_level: float = 0.95) -> float:
        """计算风险价值（VaR）"""
        if len(returns) == 0:
            return 0.0
        
        # 使用历史模拟法计算VaR
        sorted_returns = np.sort(returns)
        index = int((1 - confidence_level) * len(sorted_returns))
        var = sorted_returns[index] if index < len(sorted_returns) else sorted_returns[-1]
        
        return float(var)
    
    @staticmethod
    def calculate_expected_shortfall(returns: np.ndarray, confidence_level: float = 0.95) -> float:
        """计算期望损失（Expected Shortfall/CVaR）"""
        if len(returns) == 0:
            return 0.0
        
        var = RiskAssessment.calculate_var(returns, confidence_level)
        # 计算超过VaR的平均损失
        tail_losses = returns[returns <= var]
        
        if len(tail_losses) == 0:
            return var
        
        expected_shortfall = np.mean(tail_losses)
        return float(expected_shortfall)
    
    @staticmethod
    def calculate_confidence_interval(
        predictions: np.ndarray, 
        confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """计算预测的置信区间"""
        if len(predictions) == 0:
            return 0.0, 0.0
        
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bound = np.percentile(predictions, lower_percentile)
        upper_bound = np.percentile(predictions, upper_percentile)
        
        return float(lower_bound), float(upper_bound)


class ModelLoader:
    """模型加载器"""
    
    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self.loaded_models = {}  # 缓存已加载的模型
    
    def load_model(self, model_id: str, model_type: str):
        """加载模型"""
        model_path = self.models_dir / f"{model_id}.{self._get_model_extension(model_type)}"
        
        if not model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # 这里应该根据模型类型加载实际模型
        # 为了测试，返回一个模拟对象
        return {"model_id": model_id, "type": model_type, "path": str(model_path)}
    
    def _get_model_extension(self, model_type: str) -> str:
        """获取模型文件扩展名"""
        extensions = {
            "xgboost": "json",
            "pytorch": "pth",
            "sklearn": "pkl"
        }
        return extensions.get(model_type, "pkl")
    
    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "model_id": model_id,
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "status": "active"
        }


class PredictionEngine:
    """预测引擎主类"""
    
    def __init__(self):
        self.models_dir = Path("backend/data/models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.model_loader = ModelLoader(self.models_dir)
        self.risk_assessment = RiskAssessment()
        
        # 导入相关服务
        self.data_service = None
        self.indicator_calculator = None
    
    async def initialize(self):
        """初始化预测引擎"""
        logger.info("预测引擎初始化完成")
    
    def _infer_model_type(self, model_id: str) -> str:
        """推断模型类型"""
        # 检查模型文件扩展名来推断类型
        for extension, model_type in [
            (".json", "xgboost"),
            (".pth", "pytorch"),
            (".pkl", "sklearn"),
            (".joblib", "sklearn")
        ]:
            model_path = self.models_dir / f"{model_id}{extension}"
            if model_path.exists():
                return model_type
        
        return "unknown"
    
    async def _make_prediction(self, model, X: np.ndarray, model_type: str) -> Tuple[float, int]:
        """使用模型进行预测"""
        if model_type == "xgboost":
            # XGBoost模型预测
            if hasattr(model, 'predict'):
                proba = model.predict(X.reshape(X.shape[0], -1))[0]  # 展平输入
            else:
                proba = 0.5  # 默认值
        elif model_type == "pytorch":
            # PyTorch模型预测
            if hasattr(model, 'forward'):
                import torch
                with torch.no_grad():
                    tensor_input = torch.FloatTensor(X)
                    output = model.forward(tensor_input)
                    proba = torch.sigmoid(output).item()
            else:
                proba = 0.5
        else:
            # 其他模型类型
            proba = 0.5
        
        # 确定方向：概率>0.5为上涨(1)，否则为下跌(0)
        direction = 1 if proba > 0.5 else 0
        
        return float(proba), direction
    
    def _calculate_predicted_return(
        self, 
        direction: int, 
        confidence: float, 
        horizon: PredictionHorizon
    ) -> float:
        """计算预测收益率"""
        # 基础收益率（根据时间维度调整）
        base_return = {
            PredictionHorizon.INTRADAY: 0.02,    # 日内2%
            PredictionHorizon.SHORT_TERM: 0.05,  # 短期5%
            PredictionHorizon.MEDIUM_TERM: 0.10  # 中期10%
        }.get(horizon, 0.05)
        
        # 根据方向和置信度调整
        sign = 1 if direction == 1 else -1
        confidence_factor = (confidence - 0.5) * 2  # 将0.5-1.0映射到0-1.0
        
        predicted_return = sign * base_return * confidence_factor
        return predicted_return
    
    def _calculate_prediction_confidence_interval(
        self, 
        predicted_return: float, 
        volatility: float, 
        confidence_level: float
    ) -> Tuple[float, float]:
        """计算预测的置信区间"""
        # 简化版本：使用固定的标准差
        margin_of_error = 1.96 * volatility / np.sqrt(252)  # 95%置信区间
        
        lower_bound = predicted_return - margin_of_error
        upper_bound = predicted_return + margin_of_error
        
        return lower_bound, upper_bound


# 导出主要类和函数
__all__ = [
    'PredictionEngine',
    'PredictionConfig',
    'PredictionResult',
    'PredictionHorizon',
    'ModelLoader',
    'RiskAssessment']