"""
高级模型训练功能

实现模型集成（ensemble）、在线学习和增量训练功能。
支持多模型组合、动态模型更新和增量数据训练。
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import pickle

import numpy as np
import pandas as pd
import torch

# 避免循环导入，在此处重新定义ModelType
class ModelType(Enum):
    """支持的模型类型"""
    TRANSFORMER = "transformer"
    TIMESNET = "timesnet"
    PATCHTST = "patchtst"
    INFORMER = "informer"
    LSTM = "lstm"
    XGBOOST = "xgboost"
import torch.nn as nn
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import xgboost as xgb

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)


class EnsembleMethod(Enum):
    """集成方法类型"""
    VOTING = "voting"           # 投票集成
    WEIGHTED = "weighted"       # 加权集成
    STACKING = "stacking"       # 堆叠集成
    BAGGING = "bagging"         # 装袋集成


@dataclass
class EnsembleConfig:
    """集成模型配置"""
    method: EnsembleMethod
    base_models: List[str]      # 基础模型ID列表
    weights: Optional[List[float]] = None  # 权重（用于加权集成）
    meta_model_type: Optional[ModelType] = None  # 元模型类型（用于堆叠集成）
    voting_strategy: str = "soft"  # 投票策略：hard或soft


@dataclass
class OnlineLearningConfig:
    """在线学习配置"""
    update_frequency: int = 5   # 更新频率（天）
    learning_rate_decay: float = 0.95  # 学习率衰减
    memory_size: int = 1000     # 记忆缓冲区大小
    adaptation_threshold: float = 0.1  # 性能下降阈值


class EnsembleModelManager:
    """集成模型管理器"""
    
    def __init__(self, model_training_service: "ModelTrainingService"):
        self.training_service = model_training_service
        self.evaluator = ModelEvaluator()
        self.ensemble_dir = Path("backend/data/ensembles")
        self.ensemble_dir.mkdir(parents=True, exist_ok=True)
    
    async def create_ensemble(
        self,
        ensemble_id: str,
        config: EnsembleConfig,
        validation_data: Tuple[np.ndarray, np.ndarray]
    ) -> Dict[str, Any]:
        """
        创建集成模型
        
        Args:
            ensemble_id: 集成模型ID
            config: 集成配置
            validation_data: 验证数据 (X, y)
            
        Returns:
            集成模型信息
        """
        logger.info(f"开始创建集成模型 {ensemble_id}，方法: {config.method.value}")
        
        # 加载基础模型
        base_models = await self._load_base_models(config.base_models)
        
        if not base_models:
            raise ValueError("没有找到有效的基础模型")
        
        # 根据集成方法创建集成模型
        if config.method == EnsembleMethod.VOTING:
            ensemble_model = await self._create_voting_ensemble(
                base_models, config, validation_data
            )
        elif config.method == EnsembleMethod.WEIGHTED:
            ensemble_model = await self._create_weighted_ensemble(
                base_models, config, validation_data
            )
        elif config.method == EnsembleMethod.STACKING:
            ensemble_model = await self._create_stacking_ensemble(
                base_models, config, validation_data
            )
        elif config.method == EnsembleMethod.BAGGING:
            ensemble_model = await self._create_bagging_ensemble(
                base_models, config, validation_data
            )
        else:
            raise ValueError(f"不支持的集成方法: {config.method}")
        
        # 评估集成模型
        X_val, y_val = validation_data
        predictions = await self._predict_ensemble(ensemble_model, X_val)
        metrics = self._calculate_ensemble_metrics(y_val, predictions)
        
        # 保存集成模型
        ensemble_info = {
            "ensemble_id": ensemble_id,
            "method": config.method.value,
            "base_models": config.base_models,
            "config": config.__dict__,
            "metrics": metrics,
            "created_at": datetime.now().isoformat()
        }
        
        ensemble_path = self.ensemble_dir / f"{ensemble_id}.pkl"
        with open(ensemble_path, 'wb') as f:
            pickle.dump({
                "model": ensemble_model,
                "info": ensemble_info
            }, f)
        
        logger.info(f"集成模型 {ensemble_id} 创建完成，准确率: {metrics['accuracy']:.4f}")
        return ensemble_info
    
    async def _load_base_models(self, model_ids: List[str]) -> List[Any]:
        """加载基础模型"""
        models = []
        
        for model_id in model_ids:
            try:
                # 这里应该从模型版本管理器加载模型
                # 为了简化，我们创建模拟模型
                model_path = Path(f"backend/data/models/{model_id}")
                if model_path.exists():
                    # 根据文件扩展名判断模型类型
                    if model_path.suffix == '.json':
                        # XGBoost模型
                        model = xgb.Booster()
                        model.load_model(str(model_path))
                    else:
                        # PyTorch模型
                        model = torch.load(model_path, map_location='cpu')
                    
                    models.append({
                        "id": model_id,
                        "model": model,
                        "type": "xgboost" if model_path.suffix == '.json' else "pytorch"
                    })
                    logger.info(f"成功加载模型: {model_id}")
                else:
                    logger.warning(f"模型文件不存在: {model_id}")
            
            except Exception as e:
                logger.error(f"加载模型 {model_id} 失败: {e}")
                continue
        
        return models
    
    async def _create_voting_ensemble(
        self,
        base_models: List[Dict],
        config: EnsembleConfig,
        validation_data: Tuple[np.ndarray, np.ndarray]
    ) -> Dict[str, Any]:
        """创建投票集成模型"""
        logger.info("创建投票集成模型")
        
        X_val, y_val = validation_data
        
        # 获取每个基础模型的预测
        model_predictions = []
        model_weights = []
        
        for i, model_info in enumerate(base_models):
            try:
                predictions = await self._get_model_predictions(
                    model_info["model"], X_val, model_info["type"]
                )
                
                # 计算模型在验证集上的准确率作为权重
                accuracy = accuracy_score(y_val, predictions)
                
                model_predictions.append(predictions)
                model_weights.append(accuracy)
                
                logger.info(f"模型 {model_info['id']} 验证准确率: {accuracy:.4f}")
                
            except Exception as e:
                logger.error(f"获取模型 {model_info['id']} 预测失败: {e}")
                continue
        
        if not model_predictions:
            raise ValueError("没有获得有效的模型预测")
        
        # 归一化权重
        total_weight = sum(model_weights)
        normalized_weights = [w / total_weight for w in model_weights]
        
        ensemble_model = {
            "type": "voting",
            "models": base_models,
            "weights": normalized_weights,
            "voting_strategy": config.voting_strategy
        }
        
        return ensemble_model
    
    async def _create_weighted_ensemble(
        self,
        base_models: List[Dict],
        config: EnsembleConfig,
        validation_data: Tuple[np.ndarray, np.ndarray]
    ) -> Dict[str, Any]:
        """创建加权集成模型"""
        logger.info("创建加权集成模型")
        
        # 如果没有提供权重，使用验证集性能计算权重
        if config.weights is None:
            X_val, y_val = validation_data
            weights = []
            
            for model_info in base_models:
                predictions = await self._get_model_predictions(
                    model_info["model"], X_val, model_info["type"]
                )
                accuracy = accuracy_score(y_val, predictions)
                weights.append(accuracy)
            
            # 归一化权重
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]
        else:
            weights = config.weights
        
        ensemble_model = {
            "type": "weighted",
            "models": base_models,
            "weights": weights
        }
        
        return ensemble_model
    
    async def _create_stacking_ensemble(
        self,
        base_models: List[Dict],
        config: EnsembleConfig,
        validation_data: Tuple[np.ndarray, np.ndarray]
    ) -> Dict[str, Any]:
        """创建堆叠集成模型"""
        logger.info("创建堆叠集成模型")
        
        X_val, y_val = validation_data
        
        # 获取基础模型的预测作为元特征
        meta_features = []
        
        for model_info in base_models:
            predictions = await self._get_model_predictions(
                model_info["model"], X_val, model_info["type"]
            )
            meta_features.append(predictions)
        
        # 转置以获得正确的形状 (n_samples, n_models)
        meta_X = np.column_stack(meta_features)
        
        # 训练元模型
        meta_model_type = config.meta_model_type or ModelType.XGBOOST
        
        if meta_model_type == ModelType.XGBOOST:
            meta_model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                random_state=42
            )
            meta_model.fit(meta_X, y_val)
        else:
            # 简化的神经网络元模型
            meta_model = nn.Sequential(
                nn.Linear(len(base_models), 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 2)
            )
            
            # 训练元模型（简化版本）
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(meta_model.parameters(), lr=0.001)
            
            meta_X_tensor = torch.FloatTensor(meta_X)
            y_val_tensor = torch.LongTensor(y_val)
            
            for epoch in range(100):
                optimizer.zero_grad()
                outputs = meta_model(meta_X_tensor)
                loss = criterion(outputs, y_val_tensor)
                loss.backward()
                optimizer.step()
        
        ensemble_model = {
            "type": "stacking",
            "base_models": base_models,
            "meta_model": meta_model,
            "meta_model_type": meta_model_type.value
        }
        
        return ensemble_model
    
    async def _create_bagging_ensemble(
        self,
        base_models: List[Dict],
        config: EnsembleConfig,
        validation_data: Tuple[np.ndarray, np.ndarray]
    ) -> Dict[str, Any]:
        """创建装袋集成模型"""
        logger.info("创建装袋集成模型")
        
        # 装袋集成通过对基础模型的预测进行平均
        ensemble_model = {
            "type": "bagging",
            "models": base_models,
            "aggregation": "average"
        }
        
        return ensemble_model
    
    async def _get_model_predictions(
        self, 
        model: Any, 
        X: np.ndarray, 
        model_type: str
    ) -> np.ndarray:
        """获取模型预测"""
        if model_type == "xgboost":
            # XGBoost预测
            if X.ndim == 3:
                X_flat = X.reshape(X.shape[0], -1)
            else:
                X_flat = X
            
            dtest = xgb.DMatrix(X_flat)
            predictions = model.predict(dtest)
            return (predictions > 0.5).astype(int)
        
        else:
            # PyTorch模型预测
            model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X)
                outputs = model(X_tensor)
                predictions = torch.argmax(outputs, dim=1).numpy()
            return predictions
    
    async def _predict_ensemble(
        self, 
        ensemble_model: Dict[str, Any], 
        X: np.ndarray
    ) -> np.ndarray:
        """集成模型预测"""
        ensemble_type = ensemble_model["type"]
        
        if ensemble_type in ["voting", "weighted"]:
            # 获取所有基础模型的预测
            predictions = []
            weights = ensemble_model["weights"]
            
            for i, model_info in enumerate(ensemble_model["models"]):
                model_pred = await self._get_model_predictions(
                    model_info["model"], X, model_info["type"]
                )
                predictions.append(model_pred * weights[i])
            
            # 加权平均
            final_predictions = np.sum(predictions, axis=0)
            return (final_predictions > 0.5).astype(int)
        
        elif ensemble_type == "stacking":
            # 获取基础模型预测作为元特征
            meta_features = []
            for model_info in ensemble_model["base_models"]:
                model_pred = await self._get_model_predictions(
                    model_info["model"], X, model_info["type"]
                )
                meta_features.append(model_pred)
            
            meta_X = np.column_stack(meta_features)
            
            # 使用元模型预测
            meta_model = ensemble_model["meta_model"]
            if ensemble_model["meta_model_type"] == "xgboost":
                return meta_model.predict(meta_X)
            else:
                with torch.no_grad():
                    meta_X_tensor = torch.FloatTensor(meta_X)
                    outputs = meta_model(meta_X_tensor)
                    return torch.argmax(outputs, dim=1).numpy()
        
        elif ensemble_type == "bagging":
            # 获取所有模型预测并平均
            predictions = []
            for model_info in ensemble_model["models"]:
                model_pred = await self._get_model_predictions(
                    model_info["model"], X, model_info["type"]
                )
                predictions.append(model_pred)
            
            # 多数投票
            predictions_array = np.array(predictions)
            final_predictions = np.mean(predictions_array, axis=0)
            return (final_predictions > 0.5).astype(int)
        
        else:
            raise ValueError(f"不支持的集成类型: {ensemble_type}")
    
    def _calculate_ensemble_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """计算集成模型指标"""
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0)
        }


class OnlineLearningManager:
    """在线学习管理器"""
    
    def __init__(self, model_training_service: "ModelTrainingService"):
        self.training_service = model_training_service
        self.memory_buffer = {}  # 存储最近的训练数据
        self.model_performance_history = {}  # 模型性能历史
        
    async def setup_online_learning(
        self,
        model_id: str,
        config: OnlineLearningConfig
    ) -> Dict[str, Any]:
        """
        设置在线学习
        
        Args:
            model_id: 模型ID
            config: 在线学习配置
            
        Returns:
            在线学习设置信息
        """
        logger.info(f"为模型 {model_id} 设置在线学习")
        
        # 初始化记忆缓冲区
        self.memory_buffer[model_id] = {
            "data": [],
            "labels": [],
            "timestamps": [],
            "max_size": config.memory_size
        }
        
        # 初始化性能历史
        self.model_performance_history[model_id] = {
            "accuracies": [],
            "timestamps": [],
            "config": config
        }
        
        setup_info = {
            "model_id": model_id,
            "config": config.__dict__,
            "status": "active",
            "setup_time": datetime.now().isoformat()
        }
        
        logger.info(f"模型 {model_id} 在线学习设置完成")
        return setup_info
    
    async def update_model_online(
        self,
        model_id: str,
        new_data: np.ndarray,
        new_labels: np.ndarray,
        current_model: Any
    ) -> Tuple[Any, Dict[str, float]]:
        """
        在线更新模型
        
        Args:
            model_id: 模型ID
            new_data: 新的训练数据
            new_labels: 新的标签
            current_model: 当前模型
            
        Returns:
            (更新后的模型, 性能指标)
        """
        logger.info(f"开始在线更新模型 {model_id}")
        
        if model_id not in self.memory_buffer:
            raise ValueError(f"模型 {model_id} 未设置在线学习")
        
        # 添加新数据到记忆缓冲区
        buffer = self.memory_buffer[model_id]
        current_time = datetime.now()
        
        # 添加新数据
        for i in range(len(new_data)):
            buffer["data"].append(new_data[i])
            buffer["labels"].append(new_labels[i])
            buffer["timestamps"].append(current_time)
        
        # 维护缓冲区大小
        if len(buffer["data"]) > buffer["max_size"]:
            excess = len(buffer["data"]) - buffer["max_size"]
            buffer["data"] = buffer["data"][excess:]
            buffer["labels"] = buffer["labels"][excess:]
            buffer["timestamps"] = buffer["timestamps"][excess:]
        
        # 使用缓冲区数据进行增量训练
        if len(buffer["data"]) >= 32:  # 最少需要一个批次的数据
            updated_model = await self._incremental_training(
                current_model, 
                np.array(buffer["data"]), 
                np.array(buffer["labels"]),
                model_id
            )
            
            # 评估更新后的模型
            metrics = await self._evaluate_online_model(
                updated_model, 
                np.array(buffer["data"]), 
                np.array(buffer["labels"])
            )
            
            # 更新性能历史
            history = self.model_performance_history[model_id]
            history["accuracies"].append(metrics["accuracy"])
            history["timestamps"].append(current_time)
            
            # 检查是否需要触发模型重训练
            if await self._should_retrain(model_id, metrics):
                logger.info(f"模型 {model_id} 性能下降，建议重新训练")
                metrics["retrain_recommended"] = True
            
            logger.info(f"模型 {model_id} 在线更新完成，准确率: {metrics['accuracy']:.4f}")
            return updated_model, metrics
        
        else:
            logger.info(f"数据不足，跳过模型 {model_id} 的在线更新")
            return current_model, {"accuracy": 0.0, "message": "insufficient_data"}
    
    async def _incremental_training(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        model_id: str
    ) -> Any:
        """增量训练"""
        config = self.model_performance_history[model_id]["config"]
        
        # 检查模型类型并进行相应的增量训练
        if isinstance(model, xgb.Booster):
            # XGBoost增量训练
            dtrain = xgb.DMatrix(X.reshape(X.shape[0], -1), label=y)
            
            # 使用较小的学习率进行增量训练
            updated_model = xgb.train(
                {
                    'objective': 'binary:logistic',
                    'eval_metric': 'logloss',
                    'learning_rate': 0.01,  # 较小的学习率
                    'max_depth': 3
                },
                dtrain,
                num_boost_round=10,  # 较少的轮数
                xgb_model=model  # 基于现有模型继续训练
            )
            
            return updated_model
        
        elif isinstance(model, nn.Module):
            # PyTorch模型增量训练
            model.train()
            
            # 使用衰减的学习率
            current_lr = config.learning_rate * (config.learning_rate_decay ** len(self.model_performance_history[model_id]["accuracies"]))
            optimizer = torch.optim.Adam(model.parameters(), lr=current_lr)
            criterion = nn.CrossEntropyLoss()
            
            # 转换数据
            X_tensor = torch.FloatTensor(X)
            y_tensor = torch.LongTensor(y)
            
            # 小批量增量训练
            batch_size = min(32, len(X))
            for i in range(0, len(X), batch_size):
                batch_X = X_tensor[i:i+batch_size]
                batch_y = y_tensor[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            
            return model
        
        else:
            logger.warning(f"不支持的模型类型进行增量训练: {type(model)}")
            return model
    
    async def _evaluate_online_model(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, float]:
        """评估在线模型"""
        try:
            if isinstance(model, xgb.Booster):
                dtest = xgb.DMatrix(X.reshape(X.shape[0], -1))
                predictions = model.predict(dtest)
                y_pred = (predictions > 0.5).astype(int)
            
            elif isinstance(model, nn.Module):
                model.eval()
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X)
                    outputs = model(X_tensor)
                    y_pred = torch.argmax(outputs, dim=1).numpy()
            
            else:
                return {"accuracy": 0.0, "error": "unsupported_model_type"}
            
            accuracy = accuracy_score(y, y_pred)
            precision = precision_score(y, y_pred, zero_division=0)
            recall = recall_score(y, y_pred, zero_division=0)
            
            return {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall
            }
        
        except Exception as e:
            logger.error(f"在线模型评估失败: {e}")
            return {"accuracy": 0.0, "error": str(e)}
    
    async def _should_retrain(
        self,
        model_id: str,
        current_metrics: Dict[str, float]
    ) -> bool:
        """判断是否需要重新训练模型"""
        history = self.model_performance_history[model_id]
        config = history["config"]
        
        if len(history["accuracies"]) < 5:  # 需要足够的历史数据
            return False
        
        # 计算最近5次的平均准确率
        recent_avg = np.mean(history["accuracies"][-5:])
        
        # 计算历史最佳准确率
        historical_best = np.max(history["accuracies"][:-1]) if len(history["accuracies"]) > 1 else recent_avg
        
        # 如果性能下降超过阈值，建议重训练
        performance_drop = historical_best - recent_avg
        
        return performance_drop > config.adaptation_threshold


class AdvancedTrainingService:
    """高级训练服务主类"""
    
    def __init__(self):
        self.model_training_service = None
        self.ensemble_manager = None
        self.online_learning_manager = None
    
    async def initialize(self):
        """初始化服务"""
        # 延迟导入避免循环导入
        from .model_training import ModelTrainingService
        self.model_training_service = ModelTrainingService()
        await self.model_training_service.initialize()
        
        self.ensemble_manager = EnsembleModelManager(self.model_training_service)
        self.online_learning_manager = OnlineLearningManager(self.model_training_service)
        
        logger.info("高级训练服务初始化完成")
    
    async def create_ensemble_model(
        self,
        ensemble_id: str,
        config: EnsembleConfig,
        validation_data: Tuple[np.ndarray, np.ndarray]
    ) -> Dict[str, Any]:
        """创建集成模型"""
        return await self.ensemble_manager.create_ensemble(
            ensemble_id, config, validation_data
        )
    
    async def setup_online_learning(
        self,
        model_id: str,
        config: OnlineLearningConfig
    ) -> Dict[str, Any]:
        """设置在线学习"""
        return await self.online_learning_manager.setup_online_learning(
            model_id, config
        )
    
    async def update_model_online(
        self,
        model_id: str,
        new_data: np.ndarray,
        new_labels: np.ndarray,
        current_model: Any
    ) -> Tuple[Any, Dict[str, float]]:
        """在线更新模型"""
        return await self.online_learning_manager.update_model_online(
            model_id, new_data, new_labels, current_model
        )


# 导出主要类和函数
__all__ = [
    'AdvancedTrainingService',
    'EnsembleModelManager',
    'OnlineLearningManager',
    'EnsembleConfig',
    'OnlineLearningConfig',
    'EnsembleMethod']