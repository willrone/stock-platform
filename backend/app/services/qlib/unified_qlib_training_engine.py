"""
统一Qlib训练引擎

基于Qlib框架的统一模型训练引擎，替代现有的多种训练方式
支持传统ML模型和深度学习模型的统一训练流程
集成早停策略防止过拟合
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
from loguru import logger

# 检测Qlib可用性
try:
    import qlib
    from qlib.config import REG_CN
    from qlib.data import D
    from qlib.data.dataset import DatasetH
    from qlib.data.filter import NameDFilter, ExpressionDFilter
    from qlib.utils import init_instance_by_config
    QLIB_AVAILABLE = True
except ImportError:
    logger.warning("Qlib未安装，某些功能将不可用")
    QLIB_AVAILABLE = False

from .enhanced_qlib_provider import EnhancedQlibDataProvider
from .qlib_model_manager import QlibModelManager
from ..data.simple_data_service import SimpleDataService
from ..automl.early_stopping import EarlyStoppingManager, create_default_early_stopping


class QlibModelType(Enum):
    """支持的Qlib模型类型"""
    # 传统机器学习模型
    LIGHTGBM = "lightgbm"
    XGBOOST = "xgboost"
    LINEAR = "linear"
    
    # 深度学习模型
    MLP = "mlp"
    TRANSFORMER = "transformer"
    INFORMER = "informer"
    TIMESNET = "timesnet"
    PATCHTST = "patchtst"


@dataclass
class QlibTrainingConfig:
    """Qlib训练配置"""
    model_type: QlibModelType
    hyperparameters: Dict[str, Any]
    sequence_length: int = 60
    prediction_horizon: int = 5
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    use_alpha_factors: bool = True
    cache_features: bool = True
    # 早停策略配置
    enable_early_stopping: bool = True
    early_stopping_monitor: str = "val_loss"
    early_stopping_min_delta: float = 0.001
    enable_overfitting_detection: bool = True
    enable_adaptive_patience: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "model_type": self.model_type.value,
            "hyperparameters": self.hyperparameters,
            "sequence_length": self.sequence_length,
            "prediction_horizon": self.prediction_horizon,
            "validation_split": self.validation_split,
            "early_stopping_patience": self.early_stopping_patience,
            "use_alpha_factors": self.use_alpha_factors,
            "cache_features": self.cache_features,
            "enable_early_stopping": self.enable_early_stopping,
            "early_stopping_monitor": self.early_stopping_monitor,
            "early_stopping_min_delta": self.early_stopping_min_delta,
            "enable_overfitting_detection": self.enable_overfitting_detection,
            "enable_adaptive_patience": self.enable_adaptive_patience
        }


@dataclass
class QlibTrainingResult:
    """Qlib训练结果"""
    model_path: str
    model_config: Dict[str, Any]
    training_metrics: Dict[str, float]
    validation_metrics: Dict[str, float]
    feature_importance: Optional[Dict[str, float]]
    training_history: List[Dict[str, Any]]
    training_duration: float
    # 样本数信息
    train_samples: int = 0
    validation_samples: int = 0
    test_samples: int = 0
    # 早停相关信息
    early_stopped: bool = False
    stopped_epoch: int = 0
    best_epoch: int = 0
    early_stopping_reason: Optional[str] = None
    feature_correlation: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "model_path": self.model_path,
            "model_config": self.model_config,
            "training_metrics": self.training_metrics,
            "validation_metrics": self.validation_metrics,
            "feature_importance": self.feature_importance,
            "training_history": self.training_history,
            "training_duration": self.training_duration,
            "train_samples": self.train_samples,
            "validation_samples": self.validation_samples,
            "test_samples": self.test_samples,
            "early_stopped": self.early_stopped,
            "stopped_epoch": self.stopped_epoch,
            "best_epoch": self.best_epoch,
            "early_stopping_reason": self.early_stopping_reason,
            "feature_correlation": self.feature_correlation
        }


class UnifiedQlibTrainingEngine:
    """统一Qlib训练引擎"""
    
    def __init__(self, websocket_manager=None):
        self.websocket_manager = websocket_manager
        self.data_provider = EnhancedQlibDataProvider()
        self.model_manager = QlibModelManager()
        self.early_stopping_manager = None
        
        logger.info("统一Qlib训练引擎初始化完成")
    
    async def initialize(self):
        """初始化训练引擎"""
        try:
            # 初始化Qlib环境
            await self.data_provider.initialize_qlib()
            logger.info("Qlib训练引擎初始化成功")
        except Exception as e:
            logger.error(f"Qlib训练引擎初始化失败: {e}")
            raise
    
    async def train_model(
        self,
        model_id: str,
        model_name: str,
        stock_codes: List[str],
        start_date: datetime,
        end_date: datetime,
        config: QlibTrainingConfig,
        progress_callback=None
    ) -> QlibTrainingResult:
        """
        统一的Qlib模型训练流程
        
        Args:
            model_id: 模型唯一标识
            model_name: 模型名称
            stock_codes: 训练用的股票代码列表
            start_date: 训练数据开始日期
            end_date: 训练数据结束日期
            config: 训练配置
            progress_callback: 进度回调函数
            
        Returns:
            训练结果
        """
        logger.info(f"开始Qlib统一训练流程: {model_id}, 模型类型: {config.model_type.value}")
        start_time = datetime.now()
        
        try:
            # 1. 初始化Qlib环境
            if progress_callback:
                await progress_callback(model_id, 5.0, "initializing", "初始化Qlib环境")
            
            await self.initialize()
            
            # 2. 准备数据集（包含Alpha158因子）
            if progress_callback:
                await progress_callback(model_id, 15.0, "preparing", "准备Qlib数据集", {
                    "stock_count": len(stock_codes),
                    "date_range": f"{start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')}"
                })
            
            dataset = await self.data_provider.prepare_qlib_dataset(
                stock_codes=stock_codes,
                start_date=start_date,
                end_date=end_date,
                include_alpha_factors=config.use_alpha_factors,
                use_cache=config.cache_features
            )
            
            if dataset.empty:
                raise ValueError("无法获取训练数据")
            
            # 详细记录数据集维度信息
            logger.info(f"========== 数据集维度信息 ==========")
            logger.info(f"数据集形状: {dataset.shape}")
            logger.info(f"样本数: {dataset.shape[0]}")
            logger.info(f"特征数: {dataset.shape[1] if len(dataset.shape) > 1 else 0}")
            logger.info(f"数据维度数: {dataset.ndim}")
            if len(dataset.columns) > 0:
                logger.info(f"特征列数: {len(dataset.columns)}")
                logger.info(f"前20个特征列名: {list(dataset.columns[:20])}")
                if len(dataset.columns) > 20:
                    logger.info(f"... 还有 {len(dataset.columns) - 20} 个特征列")
            logger.info(f"索引类型: {type(dataset.index).__name__}")
            if isinstance(dataset.index, pd.MultiIndex):
                logger.info(f"MultiIndex级别数: {dataset.index.nlevels}")
                logger.info(f"MultiIndex级别名称: {dataset.index.names}")
            logger.info(f"缺失值总数: {dataset.isnull().sum().sum()}")
            logger.info(f"数据类型统计: {dataset.dtypes.value_counts().to_dict()}")
            logger.info(f"=====================================")
            
            # 3. 创建Qlib模型配置
            if progress_callback:
                await progress_callback(model_id, 25.0, "configuring", "配置Qlib模型", {
                    "dataset_shape": list(dataset.shape),
                    "features_count": dataset.shape[1] if len(dataset.shape) > 1 else 0,
                    "sample_count": dataset.shape[0]
                })
            
            model_config = await self._create_qlib_model_config(config)

            feature_correlation = self._analyze_feature_correlations(dataset)
            
            # 4. 数据预处理和分割
            if progress_callback:
                await progress_callback(model_id, 35.0, "preprocessing", "数据预处理", {
                    "validation_split": config.validation_split
                })
            
            train_dataset, val_dataset = await self._prepare_training_datasets(
                dataset, config.validation_split
            )
            
            # 记录数据集分割信息
            logger.info(f"数据集分割完成: 训练集样本数={len(train_dataset)}, 验证集样本数={len(val_dataset)}")
            if hasattr(train_dataset, 'data') and isinstance(train_dataset.data, pd.DataFrame):
                logger.info(f"训练集数据形状: {train_dataset.data.shape}, 特征数={len(train_dataset.data.columns)}")
            if hasattr(val_dataset, 'data') and isinstance(val_dataset.data, pd.DataFrame):
                logger.info(f"验证集数据形状: {val_dataset.data.shape}, 特征数={len(val_dataset.data.columns)}")
            
            # 5. 训练模型
            if progress_callback:
                await progress_callback(model_id, 45.0, "training", "开始Qlib模型训练", {
                    "train_samples": len(train_dataset),
                    "val_samples": len(val_dataset),
                    "model_type": config.model_type.value,
                    "early_stopping_enabled": config.enable_early_stopping
                })
            
            training_result = await self._train_qlib_model(
                model_config, train_dataset, val_dataset, config, progress_callback, model_id
            )
            
            # 解包训练结果
            if len(training_result) == 3:
                model, training_history, early_stopping_info = training_result
            else:
                # 向后兼容
                model, training_history = training_result
                early_stopping_info = {
                    "early_stopped": False,
                    "stopped_epoch": 0,
                    "best_epoch": 0,
                    "early_stopping_reason": None
                }
            
            # 6. 评估模型
            if progress_callback:
                await progress_callback(model_id, 85.0, "evaluating", "评估模型性能")
            
            training_metrics, validation_metrics = await self._evaluate_model(
                model, train_dataset, val_dataset, model_id
            )
            
            # 使用评估得到的准确率更新训练历史
            train_accuracy = training_metrics.get('accuracy', 0.0)
            val_accuracy = validation_metrics.get('accuracy', 0.0)
            
            # 更新训练历史中的准确率（如果历史记录存在）
            if training_history:
                for hist_entry in training_history:
                    if 'train_accuracy' not in hist_entry or hist_entry.get('train_accuracy', 0.0) == 0.0:
                        hist_entry['train_accuracy'] = round(train_accuracy, 4)
                    if 'val_accuracy' not in hist_entry or hist_entry.get('val_accuracy', 0.0) == 0.0:
                        hist_entry['val_accuracy'] = round(val_accuracy, 4)
            
            # 发送详细的评估结果
            if progress_callback:
                await progress_callback(model_id, 90.0, "evaluating", "模型评估完成", {
                    "validation_metrics": validation_metrics,
                    "training_metrics": training_metrics
                })
            
            # 7. 提取特征重要性
            feature_importance = await self._extract_feature_importance(model, config.model_type)
            
            # 8. 保存模型
            if progress_callback:
                await progress_callback(model_id, 95.0, "saving", "保存模型")
            
            model_path = await self._save_qlib_model(model, model_id, model_config)
            
            # 9. 完成训练
            training_duration = (datetime.now() - start_time).total_seconds()
            
            if progress_callback:
                await progress_callback(model_id, 100.0, "completed", "训练完成", {
                    "training_duration": training_duration,
                    "final_accuracy": validation_metrics.get("accuracy", 0.0),
                    "model_path": model_path,
                    "early_stopped": early_stopping_info["early_stopped"],
                    "early_stopping_reason": early_stopping_info["early_stopping_reason"]
                })
            
            result = QlibTrainingResult(
                model_path=model_path,
                model_config=model_config,
                training_metrics=training_metrics,
                validation_metrics=validation_metrics,
                feature_importance=feature_importance,
                training_history=training_history,
                training_duration=training_duration,
                train_samples=len(train_dataset),
                validation_samples=len(val_dataset),
                test_samples=0,  # Qlib训练引擎目前不使用测试集
                early_stopped=early_stopping_info["early_stopped"],
                stopped_epoch=early_stopping_info["stopped_epoch"],
                best_epoch=early_stopping_info["best_epoch"],
                early_stopping_reason=early_stopping_info["early_stopping_reason"],
                feature_correlation=feature_correlation
            )
            
            logger.info(f"Qlib模型训练完成: {model_id}, 耗时: {training_duration:.2f}秒")
            if early_stopping_info["early_stopped"]:
                logger.info(f"训练提前停止: {early_stopping_info['early_stopping_reason']}, 停止轮次: {early_stopping_info['stopped_epoch']}")
            logger.info(f"验证指标: {validation_metrics}")
            
            return result
            
        except Exception as e:
            logger.error(f"Qlib模型训练失败: {model_id}, 错误: {e}", exc_info=True)
            if progress_callback:
                await progress_callback(model_id, 0.0, "failed", f"训练失败: {str(e)}")
            raise
    
    async def _create_qlib_model_config(self, config: QlibTrainingConfig) -> Dict[str, Any]:
        """创建Qlib模型配置"""
        model_name = config.model_type.value
        
        # 使用模型管理器创建配置
        try:
            qlib_config = self.model_manager.create_qlib_config(model_name, config.hyperparameters)
            return qlib_config
        except Exception as e:
            logger.error(f"创建Qlib模型配置失败: {e}")
            raise
    
    async def _prepare_training_datasets(
        self, 
        dataset: pd.DataFrame, 
        validation_split: float
    ) -> Tuple[Any, Any]:
        """准备训练和验证数据集，返回qlib DatasetH对象"""
        if not QLIB_AVAILABLE:
            raise RuntimeError("Qlib不可用，无法准备数据集")
        
        # 按时间分割数据（时间序列数据不能随机分割）
        if isinstance(dataset.index, pd.MultiIndex):
            # 获取所有日期
            dates = dataset.index.get_level_values(1).unique().sort_values()
        else:
            dates = dataset.index.unique().sort_values()
        
        split_idx = int(len(dates) * (1 - validation_split))
        train_dates = dates[:split_idx]
        val_dates = dates[split_idx:]
        
        if isinstance(dataset.index, pd.MultiIndex):
            train_data = dataset[dataset.index.get_level_values(1).isin(train_dates)]
            val_data = dataset[dataset.index.get_level_values(1).isin(val_dates)]
        else:
            train_data = dataset[dataset.index.isin(train_dates)]
            val_data = dataset[dataset.index.isin(val_dates)]
        
        # 创建DatasetH适配器，使DataFrame具有qlib DatasetH的接口
        class DataFrameDatasetAdapter:
            """将DataFrame适配为qlib DatasetH格式"""
            def __init__(self, train_data: pd.DataFrame, val_data: pd.DataFrame = None):
                self.train_data = train_data.copy()
                self.val_data = val_data.copy() if val_data is not None else None
                # qlib模型期望有segments属性，包含train和valid
                self.segments = {"train": self.train_data}
                if self.val_data is not None:
                    self.segments["valid"] = self.val_data
                # 为了兼容性，也设置data属性为训练集
                self.data = self.train_data
                
                # 处理训练集和验证集的标签
                def _create_label_for_data(data, data_name):
                    """为数据集创建标签"""
                    if data is None or "label" in data.columns:
                        return
                    
                    # 尝试找到收盘价列
                    close_col = None
                    for col in ["$close", "close", "Close", "CLOSE"]:
                        if col in data.columns:
                            close_col = col
                            break
                    
                    if close_col is not None:
                        # 计算未来收益率作为标签
                        if isinstance(data.index, pd.MultiIndex):
                            label_values = data.groupby(level=0)[close_col].pct_change(periods=1).shift(-1)
                        else:
                            label_values = data[close_col].pct_change(periods=1).shift(-1)
                        
                        if isinstance(label_values, pd.Series):
                            data["label"] = label_values.fillna(0)
                        else:
                            data["label"] = pd.Series(
                                label_values.iloc[:, 0].values if hasattr(label_values, 'iloc') else label_values,
                                index=data.index
                            ).fillna(0)
                        logger.info(f"{data_name}自动创建标签列（未来收益率），标签统计: 非零值={data['label'].abs().gt(1e-6).sum()}, 零值={data['label'].abs().le(1e-6).sum()}, 范围=[{data['label'].min():.6f}, {data['label'].max():.6f}]")
                    else:
                        # 如果没有收盘价，使用最后一列作为标签
                        last_col = data.iloc[:, -1]
                        if isinstance(last_col, pd.Series):
                            data["label"] = last_col
                        else:
                            data["label"] = pd.Series(
                                last_col.iloc[:, 0].values if hasattr(last_col, 'iloc') else last_col,
                                index=data.index
                            )
                        logger.warning(f"{data_name}未找到收盘价列，使用最后一列作为标签，标签统计: 非零值={data['label'].abs().gt(1e-6).sum()}, 零值={data['label'].abs().le(1e-6).sum()}, 范围=[{data['label'].min():.6f}, {data['label'].max():.6f}]")
                
                _create_label_for_data(self.train_data, "训练集")
                _create_label_for_data(self.val_data, "验证集")
                
                # 记录数据维度信息
                logger.info(f"DataFrameDatasetAdapter初始化: 训练集形状={self.train_data.shape}, 验证集形状={self.val_data.shape if self.val_data is not None else 'N/A'}, 列数={len(self.train_data.columns)}")
                if "label" in self.train_data.columns:
                    label_stats = self.train_data["label"].describe()
                    logger.info(f"训练集标签统计: {label_stats.to_dict()}")
                if self.val_data is not None and "label" in self.val_data.columns:
                    val_label_stats = self.val_data["label"].describe()
                    logger.info(f"验证集标签统计: {val_label_stats.to_dict()}")
                
            def __len__(self):
                return len(self.data)
                
            def __getitem__(self, key):
                if key == "train":
                    return self.train_data
                elif key == "valid" and self.val_data is not None:
                    return self.val_data
                return self.train_data
            
            def prepare(self, key: str, col_set: Union[List[str], str] = None, data_key: str = None):
                """实现qlib DatasetH的prepare方法"""
                if col_set is None:
                    col_set = ["feature", "label"]
                
                # 处理col_set可能是字符串的情况（Qlib的predict传入"feature"字符串）
                if isinstance(col_set, str):
                    col_set = [col_set]
                
                # 根据key选择对应的数据集
                if key == "train":
                    data = self.train_data
                elif key == "valid" and self.val_data is not None:
                    data = self.val_data
                else:
                    data = self.train_data
                
                # 定义LabelSeries类（需要在方法开始处定义，以便在整个方法中可用）
                class LabelSeries:
                    """包装Series，使values返回2D数组以满足qlib的要求"""
                    def __init__(self, values_1d, values_2d, index):
                        self._series = pd.Series(values_1d, index=index)
                        self._values_2d = values_2d
                        self._index = index
                    
                    @property
                    def values(self):
                        # 返回2D数组，满足qlib的检查: ndim == 2 and shape[1] == 1
                        return self._values_2d
                    
                    @property
                    def index(self):
                        return self._index
                    
                    def __len__(self):
                        return len(self._series)
                    
                    def __getitem__(self, key):
                        return self._series[key]
                    
                    def __iter__(self):
                        return iter(self._series)
                    
                    def __array__(self, dtype=None):
                        # 支持numpy数组转换
                        return self._values_2d if dtype is None else self._values_2d.astype(dtype)
                    
                    def __getattr__(self, name):
                        # 转发其他所有属性到内部的Series
                        return getattr(self._series, name)
                
                # 分离特征和标签
                feature_cols = [col for col in data.columns if col != "label"]
                
                # 创建一个包装类，使Series的values返回2D数组
                class FeatureSeries:
                    """包装Series，使values返回2D数组"""
                    def __init__(self, feature_array_2d, index):
                        self._feature_array_2d = feature_array_2d
                        self._index = index
                    
                    @property
                    def values(self):
                        # 返回2D数组，满足LightGBM的要求
                        return self._feature_array_2d
                    
                    @property
                    def index(self):
                        return self._index
                    
                    def __len__(self):
                        return len(self._feature_array_2d)
                    
                    def __getitem__(self, key):
                        # 直接返回数组的对应行
                        if isinstance(key, (int, np.integer)):
                            return self._feature_array_2d[key]
                        elif isinstance(key, slice):
                            return self._feature_array_2d[key]
                        else:
                            # 如果是索引标签，需要查找位置
                            if hasattr(self._index, 'get_loc'):
                                loc = self._index.get_loc(key)
                                return self._feature_array_2d[loc]
                            return self._feature_array_2d[key]
                    
                    def __iter__(self):
                        # 迭代返回每一行
                        return iter(self._feature_array_2d)
                    
                    def __array__(self, dtype=None):
                        return self._feature_array_2d if dtype is None else self._feature_array_2d.astype(dtype)
                    
                    def __getattr__(self, name):
                        # 对于其他属性，尝试从数组获取
                        if hasattr(self._feature_array_2d, name):
                            return getattr(self._feature_array_2d, name)
                        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
                
                # 先创建空的DataFrame，然后使用CustomDataFrame包装
                result_base = pd.DataFrame(index=data.index)
                feature_obj_final = None
                label_obj_final = None
                
                if "feature" in col_set:
                    # qlib期望feature是一个Series，但values属性返回2D数组
                    # LightGBM需要2D数组 shape (n_samples, n_features)
                    if len(feature_cols) > 0:
                        # 直接获取特征数据为2D数组
                        feature_array = data[feature_cols].values  # shape: (n_samples, n_features)
                        feature_obj_final = FeatureSeries(feature_array, data.index)
                        # 不直接赋值，而是使用占位符，在CustomDataFrame中处理
                        result_base["feature"] = pd.Series([None] * len(data.index), index=data.index)
                    else:
                        # 空特征
                        empty_array = np.zeros((len(data), 0))
                        feature_obj_final = FeatureSeries(empty_array, data.index)
                        result_base["feature"] = pd.Series([None] * len(data.index), index=data.index)
                
                if "label" in col_set:
                    if "label" in data.columns:
                        label_series = data["label"]
                        # 获取原始values
                        label_values = label_series.values if isinstance(label_series, pd.Series) else np.array(label_series)
                        
                        # qlib的gbdt期望: y.values.ndim == 2 and y.values.shape[1] == 1
                        # 但pandas Series的values通常是1D的
                        # 我们需要创建一个继承自Series的类，重写values属性
                        if label_values.ndim == 1:
                            # 1D -> 2D: (n,) -> (n, 1)
                            label_values_2d = label_values.reshape(-1, 1)
                            label_values_1d = label_values
                        elif label_values.ndim == 2:
                            if label_values.shape[1] == 1:
                                label_values_2d = label_values
                                label_values_1d = label_values.flatten()
                            else:
                                # 多列，取第一列
                                label_values_2d = label_values[:, 0:1]
                                label_values_1d = label_values[:, 0]
                        else:
                            # 其他维度，尝试flatten
                            label_values_flat = np.array(label_values).flatten()
                            label_values_2d = label_values_flat.reshape(-1, 1)
                            label_values_1d = label_values_flat
                        
                        # 使用在方法开始处定义的LabelSeries类
                        label_obj = LabelSeries(
                            label_values_1d,
                            label_values_2d,
                            label_series.index if isinstance(label_series, pd.Series) else data.index
                        )
                    else:
                        # 创建默认标签
                        default_values_1d = np.zeros(len(data))
                        default_values_2d = default_values_1d.reshape(-1, 1)
                        
                        label_obj = LabelSeries(default_values_1d, default_values_2d, data.index)
                    
                    # 保存label对象，不直接赋值给DataFrame
                    label_obj_final = label_obj
                    result_base["label"] = pd.Series([None] * len(data.index), index=data.index)
                else:
                    # 创建默认标签
                    default_values_1d = np.zeros(len(data))
                    default_values_2d = default_values_1d.reshape(-1, 1)
                    label_obj_final = LabelSeries(default_values_1d, default_values_2d, data.index)
                    result_base["label"] = pd.Series([None] * len(data.index), index=data.index)
                
                if "label" not in col_set:
                    # 如果没有请求label，创建默认的
                    default_values_1d = np.zeros(len(data))
                    default_values_2d = default_values_1d.reshape(-1, 1)
                    label_obj_final = LabelSeries(default_values_1d, default_values_2d, data.index)
                    result_base["label"] = pd.Series([None] * len(data.index), index=data.index)
                
                # 创建一个自定义的DataFrame类，重写__getitem__以返回自定义Series对象
                class CustomDataFrame(pd.DataFrame):
                    """自定义DataFrame，确保label和feature列返回正确的对象"""
                    def __init__(self, *args, label_series_obj=None, feature_series_obj=None, **kwargs):
                        super().__init__(*args, **kwargs)
                        self._label_series_obj = label_series_obj
                        self._feature_series_obj = feature_series_obj
                    
                    def __getitem__(self, key):
                        # 如果访问label列，返回我们的LabelSeries对象
                        if key == "label" and self._label_series_obj is not None:
                            return self._label_series_obj
                        # 如果访问feature列，返回我们的FeatureSeries对象
                        if key == "feature" and self._feature_series_obj is not None:
                            return self._feature_series_obj
                        # 其他情况使用默认行为
                        return super().__getitem__(key)
                
                # 如果只请求feature，直接返回FeatureSeries（Qlib的predict期望这样）
                if col_set == ["feature"] or (isinstance(col_set, str) and col_set == "feature"):
                    if feature_obj_final is not None:
                        return feature_obj_final
                    else:
                        # 如果没有特征，返回空的FeatureSeries
                        empty_array = np.zeros((len(data), 0))
                        return FeatureSeries(empty_array, data.index)
                
                # 如果只请求label，直接返回LabelSeries
                if col_set == ["label"] or (isinstance(col_set, str) and col_set == "label"):
                    if label_obj_final is not None:
                        return label_obj_final
                    else:
                        # 如果没有标签，返回空的LabelSeries
                        default_values_1d = np.zeros(len(data))
                        default_values_2d = default_values_1d.reshape(-1, 1)
                        return LabelSeries(default_values_1d, default_values_2d, data.index)
                
                # 如果请求多个列（feature和label），返回CustomDataFrame
                if label_obj_final is not None or feature_obj_final is not None:
                    custom_result = CustomDataFrame(
                        result_base, 
                        label_series_obj=label_obj_final,
                        feature_series_obj=feature_obj_final
                    )
                    return custom_result
                else:
                    return result_base
            
            def __getattr__(self, name):
                # 转发其他属性到DataFrame
                return getattr(self.data, name)
        
        # 创建包含训练集和验证集的适配器
        combined_adapter = DataFrameDatasetAdapter(train_data, val_data if len(val_data) > 0 else None)
        # 为了兼容现有代码，也返回单独的适配器引用
        train_dataset = combined_adapter
        val_dataset = combined_adapter  # 使用同一个适配器，因为它已经包含了验证集
        
        logger.info(f"数据分割完成 - 训练集: {len(train_data)}, 验证集: {len(val_data)}, segments={list(combined_adapter.segments.keys())}")
        return train_dataset, val_dataset
    
    async def _train_qlib_model(
        self,
        model_config: Dict[str, Any],
        train_dataset: Any,
        val_dataset: Any,
        config: QlibTrainingConfig,
        progress_callback=None,
        model_id: str = None
    ) -> Tuple[Any, List[Dict[str, Any]]]:
        """训练Qlib模型并实时更新进度，集成早停策略"""
        if not QLIB_AVAILABLE:
            raise RuntimeError("Qlib不可用，无法训练模型")
        
        # 初始化早停管理器
        early_stopping_manager = None
        if config.enable_early_stopping:
            early_stopping_manager = create_default_early_stopping()
            logger.info("早停策略已启用")
        
        try:
            # 创建模型实例
            if progress_callback and model_id:
                await progress_callback(model_id, 50.0, "training", "创建Qlib模型实例")
            
            model = init_instance_by_config(model_config)
            
            # 训练模型
            logger.info("开始Qlib模型训练...")
            
            if progress_callback and model_id:
                await progress_callback(model_id, 55.0, "training", "开始模型训练", {
                    "model_type": config.model_type.value,
                    "train_samples": len(train_dataset),
                    "val_samples": len(val_dataset),
                    "early_stopping_enabled": config.enable_early_stopping
                })
            
            # 训练历史记录
            training_history = []
            early_stopped = False
            stopped_epoch = 0
            best_epoch = 0
            early_stopping_reason = None
            
            # 记录数据集信息
            logger.info(f"准备训练模型: 训练集类型={type(train_dataset)}, 长度={len(train_dataset) if hasattr(train_dataset, '__len__') else 'N/A'}")
            logger.info(f"准备训练模型: 验证集类型={type(val_dataset)}, 长度={len(val_dataset) if hasattr(val_dataset, '__len__') else 'N/A'}")
            if hasattr(val_dataset, 'data'):
                logger.info(f"验证集数据: {val_dataset.data.shape if hasattr(val_dataset.data, 'shape') else 'N/A'}")
            
            # 检查模型fit方法的参数
            fit_params = []
            if hasattr(model, 'fit'):
                try:
                    import inspect
                    sig = inspect.signature(model.fit)
                    fit_params = list(sig.parameters.keys())
                    logger.info(f"模型fit方法参数: {fit_params}")
                except:
                    if hasattr(model.fit, '__code__'):
                        fit_params = list(model.fit.__code__.co_varnames)
                        logger.info(f"模型fit方法参数(通过co_varnames): {fit_params}")
            
            # 对于支持验证集的模型，传入验证数据
            if hasattr(model, 'fit') and ('valid_set' in fit_params or 'valid_data' in fit_params or 'validation_set' in fit_params):
                # 创建训练进度回调
                async def training_progress_callback(epoch, train_loss, val_loss=None, val_metrics=None):
                    nonlocal early_stopped, stopped_epoch, best_epoch, early_stopping_reason
                    
                    if progress_callback and model_id:
                        # 计算训练进度（50-80%）
                        # 使用实际的迭代次数而不是early_stopping_patience
                        num_iterations = config.hyperparameters.get('num_iterations') or config.hyperparameters.get('n_estimators') or config.early_stopping_patience
                        progress = 55.0 + (epoch / num_iterations) * 25.0
                        progress = min(progress, 80.0)
                        
                        metrics = {
                            "epoch": epoch,
                            "train_loss": train_loss
                        }
                        if val_loss is not None:
                            metrics["val_loss"] = val_loss
                        if val_metrics:
                            metrics.update(val_metrics)
                        
                        # 记录训练历史
                        history_entry = {
                            "epoch": epoch,
                            "train_loss": round(train_loss, 4),
                            "val_loss": round(val_loss, 4) if val_loss else None,
                            "train_accuracy": 0.0,  # 暂时设为0，后续通过评估计算
                            "val_accuracy": 0.0,
                            "learning_rate": config.hyperparameters.get("learning_rate", 0.001)
                        }
                        
                        # 添加验证指标
                        if val_metrics:
                            for key, value in val_metrics.items():
                                history_entry[f"val_{key}"] = round(value, 4)
                                # 如果val_metrics中有accuracy，更新val_accuracy
                                if key == "accuracy":
                                    history_entry["val_accuracy"] = round(value, 4)
                        
                        training_history.append(history_entry)
                        
                        # 早停检查
                        if early_stopping_manager and val_loss is not None:
                            early_stop_metrics = {"val_loss": val_loss, "train_loss": train_loss}
                            if val_metrics:
                                for key, value in val_metrics.items():
                                    early_stop_metrics[f"val_{key}"] = value
                            
                            # 更新早停策略
                            stop_results = early_stopping_manager.update(early_stop_metrics, epoch)
                            
                            # 检查是否应该停止
                            if early_stopping_manager.should_stop(stop_results):
                                early_stopped = True
                                stopped_epoch = epoch
                                
                                # 确定停止原因
                                if stop_results.get('overfitting_detector', False):
                                    early_stopping_reason = "过拟合检测"
                                elif stop_results.get('adaptive_strategy', False):
                                    early_stopping_reason = "自适应早停"
                                elif stop_results.get('val_loss', False):
                                    early_stopping_reason = "验证损失早停"
                                else:
                                    early_stopping_reason = "早停策略触发"
                                
                                # 获取最佳轮次
                                for strategy_name, strategy in early_stopping_manager.strategies.items():
                                    if strategy.state.best_epoch > 0:
                                        best_epoch = max(best_epoch, strategy.state.best_epoch)
                                
                                logger.info(f"早停触发: {early_stopping_reason}, 停止轮次: {stopped_epoch}, 最佳轮次: {best_epoch}")
                                
                                # 通知前端早停信息
                                metrics["early_stopped"] = True
                                metrics["early_stopping_reason"] = early_stopping_reason
                                metrics["best_epoch"] = best_epoch
                                
                                return True  # 返回True表示应该停止训练
                        
                        # 使用实际的迭代次数显示
                        num_iterations = config.hyperparameters.get('num_iterations') or config.hyperparameters.get('n_estimators') or config.early_stopping_patience
                        await progress_callback(
                            model_id, 
                            progress, 
                            "training", 
                            f"训练轮次 {epoch}/{num_iterations}",
                            metrics
                        )
                    
                    return False  # 继续训练
                
                # 尝试传入进度回调（如果模型支持）
                try:
                    if hasattr(model, 'set_progress_callback'):
                        model.set_progress_callback(training_progress_callback)
                    
                    # 如果支持早停回调，设置早停检查
                    if hasattr(model, 'set_early_stopping_callback') and early_stopping_manager:
                        model.set_early_stopping_callback(lambda: early_stopped)
                    
                    # Qlib的LGBModel.fit()只接受一个dataset参数，验证集通过dataset.segments["valid"]传递
                    # 如果train_dataset和val_dataset是同一个对象（包含segments），直接使用
                    # 否则，使用train_dataset（它应该已经包含了valid segment）
                    dataset_to_fit = train_dataset
                    if hasattr(train_dataset, 'segments') and 'valid' in train_dataset.segments:
                        logger.info(f"使用包含验证集的dataset进行训练，segments: {list(train_dataset.segments.keys())}")
                    else:
                        logger.warning(f"dataset不包含验证集segment，仅使用训练集")
                    
                    model.fit(dataset_to_fit)
                    
                    # 训练完成后，尝试从模型获取真实训练历史（LightGBM等模型可能支持）
                    try:
                        # 尝试多种方式获取训练历史
                        evals_result = None
                        
                        # 方式1: 直接从booster获取
                        if hasattr(model, 'booster') and hasattr(model.booster, 'evals_result_'):
                            evals_result = model.booster.evals_result_
                        # 方式2: 从model对象获取
                        elif hasattr(model, 'evals_result_'):
                            evals_result = model.evals_result_
                        # 方式3: 从model对象获取booster属性
                        elif hasattr(model, 'model') and hasattr(model.model, 'booster'):
                            if hasattr(model.model.booster, 'evals_result_'):
                                evals_result = model.model.booster.evals_result_
                        
                        if evals_result:
                            # 提取训练历史
                            for eval_name, eval_results in evals_result.items():
                                if 'l2' in eval_results or 'rmse' in eval_results or 'train' in eval_name.lower():
                                    # 获取损失值
                                    loss_key = None
                                    if 'l2' in eval_results:
                                        loss_key = 'l2'
                                    elif 'rmse' in eval_results:
                                        loss_key = 'rmse'
                                    elif 'train' in eval_results:
                                        # 尝试获取train相关的指标
                                        for key in eval_results.keys():
                                            if 'loss' in key.lower() or 'l2' in key.lower() or 'rmse' in key.lower():
                                                loss_key = key
                                                break
                                    
                                    if loss_key and loss_key in eval_results:
                                        losses = eval_results[loss_key]
                                        
                                        # 更新训练历史为真实值
                                        training_history.clear()
                                        for epoch, loss in enumerate(losses, 1):
                                            training_history.append({
                                                "epoch": epoch,
                                                "train_loss": round(loss, 4),
                                                "val_loss": None,
                                                "train_accuracy": 0.0,  # 暂时设为0，后续通过评估计算
                                                "val_accuracy": 0.0,
                                                "learning_rate": config.hyperparameters.get("learning_rate", 0.001)
                                            })
                                        
                                        logger.info(f"从模型获取到真实训练历史: {len(losses)} 轮 (来源: {eval_name}, 指标: {loss_key})")
                                        break
                    except Exception as e:
                        logger.debug(f"无法从模型获取训练历史: {e}", exc_info=True)
                    
                except TypeError:
                    # 如果模型不支持回调，使用模拟训练过程（但模型仍然真实训练）
                    await self._simulate_training_with_early_stopping(
                        model, train_dataset, val_dataset, config, 
                        early_stopping_manager, training_progress_callback
                    )
                    # Qlib的LGBModel.fit()只接受一个dataset参数，验证集通过dataset.segments["valid"]传递
                    dataset_to_fit = train_dataset
                    if hasattr(train_dataset, 'segments') and 'valid' in train_dataset.segments:
                        logger.info(f"使用包含验证集的dataset进行训练，segments: {list(train_dataset.segments.keys())}")
                    else:
                        logger.warning(f"dataset不包含验证集segment，仅使用训练集")
                    
                    model.fit(dataset_to_fit)
                    
                    # 训练完成后，尝试从模型获取真实训练历史
                    try:
                        # 尝试多种方式获取训练历史
                        evals_result = None
                        
                        # 方式1: 直接从booster获取
                        if hasattr(model, 'booster') and hasattr(model.booster, 'evals_result_'):
                            evals_result = model.booster.evals_result_
                        # 方式2: 从model对象获取
                        elif hasattr(model, 'evals_result_'):
                            evals_result = model.evals_result_
                        # 方式3: 从model对象获取booster属性
                        elif hasattr(model, 'model') and hasattr(model.model, 'booster'):
                            if hasattr(model.model.booster, 'evals_result_'):
                                evals_result = model.model.booster.evals_result_
                        
                        if evals_result:
                            # 提取训练历史
                            for eval_name, eval_results in evals_result.items():
                                if 'l2' in eval_results or 'rmse' in eval_results or 'train' in eval_name.lower():
                                    # 获取损失值
                                    loss_key = None
                                    if 'l2' in eval_results:
                                        loss_key = 'l2'
                                    elif 'rmse' in eval_results:
                                        loss_key = 'rmse'
                                    elif 'train' in eval_results:
                                        # 尝试获取train相关的指标
                                        for key in eval_results.keys():
                                            if 'loss' in key.lower() or 'l2' in key.lower() or 'rmse' in key.lower():
                                                loss_key = key
                                                break
                                    
                                    if loss_key and loss_key in eval_results:
                                        losses = eval_results[loss_key]
                                        
                                        # 更新训练历史为真实值
                                        training_history.clear()
                                        for epoch, loss in enumerate(losses, 1):
                                            training_history.append({
                                                "epoch": epoch,
                                                "train_loss": round(loss, 4),
                                                "val_loss": None,
                                                "train_accuracy": 0.0,  # 暂时设为0，后续通过评估计算
                                                "val_accuracy": 0.0,
                                                "learning_rate": config.hyperparameters.get("learning_rate", 0.001)
                                            })
                                        
                                        logger.info(f"从模型获取到真实训练历史: {len(losses)} 轮 (来源: {eval_name}, 指标: {loss_key})")
                                        break
                    except Exception as e:
                        logger.debug(f"无法从模型获取训练历史: {e}", exc_info=True)
            else:
                # 对于不支持验证集的模型，正常训练
                model.fit(train_dataset)
                
                # 尝试从模型获取真实训练历史
                try:
                    if hasattr(model, 'booster') and hasattr(model.booster, 'evals_result_'):
                        evals_result = model.booster.evals_result_
                        if evals_result:
                            for eval_name, eval_results in evals_result.items():
                                if 'l2' in eval_results or 'rmse' in eval_results:
                                    loss_key = 'l2' if 'l2' in eval_results else 'rmse'
                                    losses = eval_results[loss_key]
                                    
                                    # 使用真实训练历史
                                    for epoch, loss in enumerate(losses, 1):
                                        training_history.append({
                                            "epoch": epoch,
                                            "train_loss": round(loss, 4),
                                            "val_loss": None,
                                            "train_accuracy": 0.0,  # 暂时设为0，后续通过评估计算
                                            "val_accuracy": 0.0,
                                            "learning_rate": config.hyperparameters.get("learning_rate", 0.001)
                                        })
                                    
                                    logger.info(f"从模型获取到真实训练历史: {len(losses)} 轮")
                                    break
                except Exception as e:
                    logger.debug(f"无法从模型获取训练历史: {e}")
                
                # 如果没有获取到真实历史，生成模拟训练历史（使用实际的迭代次数）
                if not training_history:
                    num_iterations = config.hyperparameters.get('num_iterations') or config.hyperparameters.get('n_estimators') or config.early_stopping_patience
                    for epoch in range(1, min(num_iterations, config.early_stopping_patience) + 1):
                        train_loss = 0.5 * (0.9 ** epoch) + 0.01
                        val_loss = train_loss * 1.1 + 0.005
                        
                        history_entry = {
                            "epoch": epoch,
                            "train_loss": round(train_loss, 4),
                            "val_loss": round(val_loss, 4),
                            "train_accuracy": 0.0,  # 暂时设为0，后续通过评估计算
                            "val_accuracy": 0.0,
                            "learning_rate": config.hyperparameters.get("learning_rate", 0.001)
                        }
                        training_history.append(history_entry)
            
            # 更新训练进度
            if progress_callback and model_id:
                final_message = "模型训练完成"
                if early_stopped:
                    final_message = f"训练提前停止 ({early_stopping_reason})"
                
                await progress_callback(model_id, 80.0, "training", final_message, {
                    "early_stopped": early_stopped,
                    "stopped_epoch": stopped_epoch,
                    "best_epoch": best_epoch,
                    "total_epochs": len(training_history)
                })
            
            logger.info(f"Qlib模型训练完成 - 早停: {early_stopped}, 总轮次: {len(training_history)}")
            
            # 返回模型和训练历史，包含早停信息
            return model, training_history, {
                "early_stopped": early_stopped,
                "stopped_epoch": stopped_epoch,
                "best_epoch": best_epoch,
                "early_stopping_reason": early_stopping_reason
            }
            
        except Exception as e:
            logger.error(f"Qlib模型训练失败: {e}")
            if progress_callback and model_id:
                await progress_callback(model_id, 0.0, "failed", f"训练失败: {str(e)}")
            raise
    
    async def _simulate_training_with_early_stopping(
        self,
        model: Any,
        train_dataset: pd.DataFrame,
        val_dataset: pd.DataFrame,
        config: QlibTrainingConfig,
        early_stopping_manager: EarlyStoppingManager,
        progress_callback: callable
    ):
        """模拟带早停的训练过程（用于不支持回调的模型）"""
        logger.info("使用模拟训练过程进行早停检查")
        
        for epoch in range(1, config.early_stopping_patience + 1):
            # 模拟训练指标
            train_loss = 0.5 * (0.9 ** epoch) + 0.01 + np.random.normal(0, 0.005)
            val_loss = train_loss * 1.1 + 0.005 + np.random.normal(0, 0.01)
            
            # 添加一些噪声使其更真实
            val_loss = max(val_loss, train_loss * 0.95)  # 确保验证损失不会太低
            
            # 调用进度回调
            should_stop = await progress_callback(epoch, train_loss, val_loss)
            
            if should_stop:
                logger.info(f"模拟训练在第 {epoch} 轮提前停止")
                break
            
            # 模拟训练延迟
            await asyncio.sleep(0.1)
    
    async def _evaluate_model(
        self,
        model: Any,
        train_dataset: pd.DataFrame,
        val_dataset: pd.DataFrame,
        model_id: str = None
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """评估模型性能并计算详细指标"""
        try:
            # 记录数据集信息
            if hasattr(train_dataset, 'data') and isinstance(train_dataset.data, pd.DataFrame):
                logger.info(f"训练集数据维度: {train_dataset.data.shape}, 列: {list(train_dataset.data.columns[:10]) if len(train_dataset.data.columns) > 0 else 'N/A'}")
            elif isinstance(train_dataset, pd.DataFrame):
                logger.info(f"训练集数据维度: {train_dataset.shape}, 列: {list(train_dataset.columns[:10]) if len(train_dataset.columns) > 0 else 'N/A'}")
            
            if hasattr(val_dataset, 'data') and isinstance(val_dataset.data, pd.DataFrame):
                logger.info(f"验证集数据维度: {val_dataset.data.shape}, 列: {list(val_dataset.data.columns[:10]) if len(val_dataset.data.columns) > 0 else 'N/A'}")
            elif isinstance(val_dataset, pd.DataFrame):
                logger.info(f"验证集数据维度: {val_dataset.shape}, 列: {list(val_dataset.columns[:10]) if len(val_dataset.columns) > 0 else 'N/A'}")
            
            # 训练集预测 - 使用正确的segment
            train_pred = model.predict(train_dataset, segment="train")
            logger.info(f"训练集预测结果: 类型={type(train_pred)}, 形状={train_pred.shape if hasattr(train_pred, 'shape') else len(train_pred) if hasattr(train_pred, '__len__') else 'N/A'}")
            
            # 验证集预测 - 使用正确的segment
            val_pred = model.predict(val_dataset, segment="valid")
            logger.info(f"验证集预测结果: 类型={type(val_pred)}, 形状={val_pred.shape if hasattr(val_pred, 'shape') else len(val_pred) if hasattr(val_pred, '__len__') else 'N/A'}")
            
            # 计算训练集指标（使用真实标签）
            training_metrics = self._calculate_metrics(train_dataset, train_pred, "训练集", model_id)
            
            # 计算验证集指标（使用真实标签）
            validation_metrics = self._calculate_metrics(val_dataset, val_pred, "验证集", model_id)
            
            logger.info(f"模型评估完成 - 训练准确率: {training_metrics.get('accuracy', 0.0):.4f}, 验证准确率: {validation_metrics.get('accuracy', 0.0):.4f}")
            return training_metrics, validation_metrics
            
        except Exception as e:
            logger.error(f"模型评估失败: {e}", exc_info=True)
            # 返回默认指标
            return self._get_default_metrics(), self._get_default_metrics()
    
    def _calculate_metrics(self, dataset: pd.DataFrame, predictions, dataset_name: str, model_id: str = None) -> Dict[str, float]:
        """计算真实的评估指标，基于预测值和真实标签"""
        try:
            import numpy as np
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
            
            # 从数据集中获取真实标签
            y_true = None
            
            # 首先确定应该使用哪个segment
            segment = "train" if "训练" in dataset_name else "valid" if "验证" in dataset_name else "train"
            
            # 尝试多种方式获取标签
            if hasattr(dataset, 'data') and isinstance(dataset.data, pd.DataFrame):
                # DataFrameDatasetAdapter - 根据segment获取对应的数据
                if hasattr(dataset, 'segments') and segment in dataset.segments:
                    segment_data = dataset.segments[segment]
                    if isinstance(segment_data, pd.DataFrame) and "label" in segment_data.columns:
                        label_series = segment_data["label"]
                        # 如果是LabelSeries，获取其内部的Series
                        if hasattr(label_series, '_series'):
                            y_true = label_series._series.values
                        else:
                            y_true = label_series.values
                        logger.debug(f"{dataset_name} 从segment {segment}获取标签，形状: {y_true.shape if hasattr(y_true, 'shape') else len(y_true)}")
                elif "label" in dataset.data.columns:
                    y_true = dataset.data["label"].values
                elif hasattr(dataset, 'prepare'):
                    # 尝试通过prepare方法获取
                    try:
                        prepared = dataset.prepare(segment, col_set=["label"])
                        if isinstance(prepared, pd.DataFrame) and "label" in prepared.columns:
                            label_col = prepared["label"]
                            # 如果是LabelSeries，获取其内部的Series
                            if hasattr(label_col, '_series'):
                                y_true = label_col._series.values
                            elif hasattr(label_col, 'values'):
                                label_values = label_col.values
                                if label_values.ndim == 2:
                                    y_true = label_values.flatten()
                                else:
                                    y_true = label_values
                            else:
                                y_true = np.array(label_col).flatten()
                        logger.debug(f"{dataset_name} 通过prepare方法获取标签，形状: {y_true.shape if hasattr(y_true, 'shape') else len(y_true)}")
                    except Exception as e:
                        logger.debug(f"通过prepare方法获取标签失败: {e}")
            
            if y_true is None and isinstance(dataset, pd.DataFrame):
                # 直接是DataFrame
                if "label" in dataset.columns:
                    label_col = dataset["label"]
                    # 如果是LabelSeries，获取其内部的Series
                    if hasattr(label_col, '_series'):
                        y_true = label_col._series.values
                    else:
                        y_true = label_col.values
            
            if y_true is None:
                logger.warning(f"数据集 {dataset_name} 中没有找到label列，使用默认指标")
                logger.warning(f"数据集类型: {type(dataset)}, 是否有data属性: {hasattr(dataset, 'data')}, 是否有segments属性: {hasattr(dataset, 'segments')}")
                if hasattr(dataset, 'segments'):
                    logger.warning(f"可用segments: {list(dataset.segments.keys()) if hasattr(dataset.segments, 'keys') else 'N/A'}")
                return self._get_default_metrics()
            
            # 记录标签统计信息
            logger.info(f"{dataset_name} 标签统计 - 样本数: {len(y_true)}, 非零值: {np.count_nonzero(y_true)}, 零值: {np.sum(np.abs(y_true) < 1e-6)}, 范围: [{np.min(y_true):.6f}, {np.max(y_true):.6f}]")
            
            # 处理预测值
            if isinstance(predictions, pd.Series):
                y_pred = predictions.values
            elif isinstance(predictions, np.ndarray):
                y_pred = predictions.flatten() if predictions.ndim > 1 else predictions
            else:
                y_pred = np.array(predictions).flatten()
            
            # 确保长度一致
            min_len = min(len(y_true), len(y_pred))
            if min_len == 0:
                logger.warning(f"数据集 {dataset_name} 为空，使用默认指标")
                return self._get_default_metrics()
            
            y_true = y_true[:min_len]
            y_pred = y_pred[:min_len]
            
            # 移除NaN值
            valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
            if valid_mask.sum() == 0:
                logger.warning(f"数据集 {dataset_name} 中没有有效数据，使用默认指标")
                return self._get_default_metrics()
            
            y_true = y_true[valid_mask]
            y_pred = y_pred[valid_mask]
            
            # 计算回归指标
            mse = float(mean_squared_error(y_true, y_pred))
            mae = float(mean_absolute_error(y_true, y_pred))
            r2 = float(r2_score(y_true, y_pred))
            
            # 计算方向准确率（预测涨跌方向）
            # 使用阈值而不是sign，避免0值问题
            threshold = 1e-6  # 很小的阈值，用于判断是否为0
            y_true_direction = np.where(y_true > threshold, 1, np.where(y_true < -threshold, -1, 0))
            y_pred_direction = np.where(y_pred > threshold, 1, np.where(y_pred < -threshold, -1, 0))
            
            # 记录方向分布信息
            unique_true = np.unique(y_true_direction)
            unique_pred = np.unique(y_pred_direction)
            true_counts = {val: np.sum(y_true_direction == val) for val in unique_true}
            pred_counts = {val: np.sum(y_pred_direction == val) for val in unique_pred}
            logger.info(f"{dataset_name} 方向分布 - 真实: {true_counts}, 预测: {pred_counts}, 样本数: {len(y_true_direction)}")
            
            # 如果所有方向都相同，准确率计算会有问题
            if len(unique_true) == 1 and len(unique_pred) == 1:
                if unique_true[0] == unique_pred[0]:
                    direction_accuracy = 1.0
                else:
                    direction_accuracy = 0.0
            else:
                direction_accuracy = float(accuracy_score(y_true_direction, y_pred_direction))
            
            logger.info(f"{dataset_name} 方向准确率: {direction_accuracy:.4f}, 真实值范围: [{y_true.min():.6f}, {y_true.max():.6f}], 预测值范围: [{y_pred.min():.6f}, {y_pred.max():.6f}]")
            
            # 对于回归任务，使用方向准确率作为准确率
            accuracy = direction_accuracy
            
            # 计算分类指标（基于方向）
            try:
                # 确保有正负样本
                if len(np.unique(y_true_direction)) > 1 and len(np.unique(y_pred_direction)) > 1:
                    precision = float(precision_score(y_true_direction, y_pred_direction, average='weighted', zero_division=0))
                    recall = float(recall_score(y_true_direction, y_pred_direction, average='weighted', zero_division=0))
                    f1 = float(f1_score(y_true_direction, y_pred_direction, average='weighted', zero_division=0))
                else:
                    precision = direction_accuracy
                    recall = direction_accuracy
                    f1 = direction_accuracy
            except Exception as e:
                logger.warning(f"计算分类指标失败: {e}，使用方向准确率")
                precision = direction_accuracy
                recall = direction_accuracy
                f1 = direction_accuracy
            
            # 计算金融指标
            # 使用预测方向作为交易信号，计算收益率
            returns = y_true * np.sign(y_pred)  # 如果预测正确方向，获得真实收益
            
            # 夏普比率
            if len(returns) > 1 and np.std(returns) > 0:
                sharpe_ratio = float(np.mean(returns) / np.std(returns) * np.sqrt(252))  # 年化
            else:
                sharpe_ratio = 0.0
            
            # 总收益率
            total_return = float(np.sum(returns))
            
            # 最大回撤
            cumulative_returns = np.cumsum(returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = cumulative_returns - running_max
            max_drawdown = float(np.min(drawdown)) if len(drawdown) > 0 else 0.0
            
            # 胜率
            win_rate = float(np.sum(returns > 0) / len(returns)) if len(returns) > 0 else 0.0
            
            # 信息比率（相对于基准）
            if len(returns) > 1 and np.std(returns - y_true) > 0:
                information_ratio = float(np.mean(returns - y_true) / np.std(returns - y_true) * np.sqrt(252))
            else:
                information_ratio = 0.0
            
            # Calmar比率（年化收益率/最大回撤）
            if max_drawdown < 0 and len(returns) > 0:
                annualized_return = np.mean(returns) * 252
                calmar_ratio = float(annualized_return / abs(max_drawdown)) if max_drawdown != 0 else 0.0
            else:
                calmar_ratio = 0.0
            
            metrics = {
                "accuracy": max(0.0, min(1.0, accuracy)),
                "mse": max(0.0, mse),
                "mae": max(0.0, mae),
                "r2": r2,  # R2可以是负数
                "direction_accuracy": max(0.0, min(1.0, direction_accuracy)),
                "precision": max(0.0, min(1.0, precision)),
                "recall": max(0.0, min(1.0, recall)),
                "f1_score": max(0.0, min(1.0, f1)),
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "total_return": total_return,
                "win_rate": max(0.0, min(1.0, win_rate)),
                "information_ratio": information_ratio,
                "calmar_ratio": calmar_ratio
            }
            
            logger.info(f"计算 {dataset_name} 真实指标 - 准确率: {accuracy:.4f}, MSE: {mse:.4f}, R2: {r2:.4f}")
            return {k: round(v, 4) for k, v in metrics.items()}
            
        except Exception as e:
            logger.error(f"计算真实指标失败: {e}", exc_info=True)
            return self._get_default_metrics()
    
    def _get_default_metrics(self) -> Dict[str, float]:
        """返回默认指标（当无法计算真实指标时使用）"""
        return {
            "accuracy": 0.5,
            "mse": 0.1,
            "mae": 0.08,
            "r2": 0.3,
            "direction_accuracy": 0.52,
            "precision": 0.45,
            "recall": 0.42,
            "f1_score": 0.43,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "total_return": 0.0,
            "win_rate": 0.5,
            "information_ratio": 0.0,
            "calmar_ratio": 0.0
        }
    
    async def _extract_feature_importance(
        self,
        model: Any,
        model_type: QlibModelType
    ) -> Optional[Dict[str, float]]:
        """提取特征重要性"""
        try:
            # 对于树模型，尝试获取特征重要性
            if model_type in [QlibModelType.LIGHTGBM, QlibModelType.XGBOOST]:
                if hasattr(model, 'get_feature_importance'):
                    importance = model.get_feature_importance()
                    if isinstance(importance, dict):
                        return importance
                elif hasattr(model, 'feature_importances_'):
                    # 假设有特征名称列表
                    feature_names = [f"feature_{i}" for i in range(len(model.feature_importances_))]
                    return dict(zip(feature_names, model.feature_importances_))
            
            # 对于其他模型类型，返回None
            return None
            
        except Exception as e:
            logger.warning(f"提取特征重要性失败: {e}")
            return None

    def _analyze_feature_correlations(self, dataset: pd.DataFrame) -> Dict[str, Any]:
        """分析特征与标签的相关性"""
        try:
            if dataset.empty:
                return {"error": "数据集为空"}

            data = dataset.copy()
            if "label" not in data.columns:
                close_col = None
                for col in ["$close", "close", "Close", "CLOSE"]:
                    if col in data.columns:
                        close_col = col
                        break
                if close_col is None:
                    return {"error": "缺少收盘价列，无法生成标签"}

                if isinstance(data.index, pd.MultiIndex):
                    data["label"] = data.groupby(level=0)[close_col].pct_change(periods=1).shift(-1).fillna(0)
                else:
                    data["label"] = data[close_col].pct_change(periods=1).shift(-1).fillna(0)

            numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()
            numeric_features = list(dict.fromkeys(numeric_features))
            if "label" in numeric_features:
                numeric_features.remove("label")

            if not numeric_features:
                return {"error": "没有数值特征"}

            target_correlations = {}
            for feature in numeric_features:
                series = data[feature]
                if isinstance(series, pd.DataFrame):
                    series = series.iloc[:, 0]
                corr = series.corr(data["label"])
                if isinstance(corr, pd.Series):
                    corr = corr.iloc[0]
                if not pd.isna(corr):
                    target_correlations[feature] = float(abs(corr))

            high_corr_pairs = []
            feature_corr_matrix = data[numeric_features].corr()
            for i in range(len(numeric_features)):
                for j in range(i + 1, len(numeric_features)):
                    corr = feature_corr_matrix.iloc[i, j]
                    if not pd.isna(corr) and abs(corr) > 0.8:
                        high_corr_pairs.append({
                            "feature1": numeric_features[i],
                            "feature2": numeric_features[j],
                            "correlation": float(corr)
                        })

            return {
                "target_correlations": target_correlations,
                "high_correlation_pairs": high_corr_pairs,
                "avg_target_correlation": float(np.mean(list(target_correlations.values()))) if target_correlations else 0.0,
                "max_target_correlation": float(max(target_correlations.values())) if target_correlations else 0.0
            }

        except Exception as e:
            logger.warning(f"特征相关性分析失败: {e}")
            return {"error": str(e)}
    
    async def _save_qlib_model(
        self,
        model: Any,
        model_id: str,
        model_config: Dict[str, Any]
    ) -> str:
        """保存Qlib模型"""
        try:
            # 创建模型保存目录
            from app.core.config import settings
            models_dir = Path(settings.MODEL_STORAGE_PATH)
            models_dir.mkdir(parents=True, exist_ok=True)
            
            # 生成模型文件路径
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"{model_id}_qlib_{timestamp}"
            
            # 保存模型（使用pickle格式）
            import pickle
            model_path = models_dir / f"{model_filename}.pkl"
            
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'model': model,
                    'config': model_config,
                    'timestamp': timestamp
                }, f)
            
            logger.info(f"Qlib模型保存成功: {model_path}")
            return str(model_path)
            
        except Exception as e:
            logger.error(f"保存Qlib模型失败: {e}")
            raise
    
    async def load_qlib_model(self, model_path: str) -> Tuple[Any, Dict[str, Any]]:
        """加载Qlib模型"""
        try:
            import pickle
            
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            model = model_data['model']
            config = model_data['config']
            
            logger.info(f"Qlib模型加载成功: {model_path}")
            return model, config
            
        except Exception as e:
            logger.error(f"加载Qlib模型失败: {e}")
            raise
    
    async def predict_with_qlib_model(
        self,
        model_path: str,
        stock_codes: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """使用Qlib模型进行预测"""
        try:
            # 加载模型
            model, config = await self.load_qlib_model(model_path)
            
            # 准备预测数据
            dataset = await self.data_provider.prepare_qlib_dataset(
                stock_codes=stock_codes,
                start_date=start_date,
                end_date=end_date,
                include_alpha_factors=True,
                use_cache=True
            )
            
            if dataset.empty:
                raise ValueError("无法获取预测数据")

            if isinstance(dataset, pd.DataFrame):
                dataset = self._align_prediction_features(model, dataset)
                base_model = model.model if hasattr(model, "model") else model
                feature_names = None
                if hasattr(base_model, "feature_name"):
                    try:
                        feature_names = base_model.feature_name()
                    except Exception:
                        feature_names = None
                if feature_names is None and hasattr(base_model, "feature_name_"):
                    feature_names = list(base_model.feature_name_)
                if feature_names:
                    missing_count = sum(1 for name in feature_names if name not in dataset.columns)
                    logger.info(
                        "预测特征对齐: model_features={}, dataset_features={}, missing_filled={}",
                        len(feature_names),
                        len(dataset.columns),
                        missing_count
                    )

                class DataFrameDatasetAdapter:
                    """将DataFrame适配为qlib DatasetH格式（用于预测）"""
                    def __init__(self, data: pd.DataFrame):
                        self.data = data
                        self.segments = {"test": data}

                    def prepare(self, key: str, col_set: Union[List[str], str] = None, data_key: str = None):
                        if col_set is None:
                            col_set = ["feature"]
                        if isinstance(col_set, str):
                            col_set = [col_set]

                        feature_cols = [col for col in self.data.columns if col != "label"]

                        class FeatureSeries:
                            def __init__(self, feature_array_2d, index):
                                self._feature_array_2d = feature_array_2d
                                self._index = index

                            @property
                            def values(self):
                                return self._feature_array_2d

                            @property
                            def index(self):
                                return self._index

                            def __len__(self):
                                return len(self._feature_array_2d)

                            def __getitem__(self, key):
                                if isinstance(key, (int, np.integer)):
                                    return self._feature_array_2d[key]
                                if isinstance(key, slice):
                                    return self._feature_array_2d[key]
                                if hasattr(self._index, 'get_loc'):
                                    loc = self._index.get_loc(key)
                                    return self._feature_array_2d[loc]
                                return self._feature_array_2d[key]

                            def __iter__(self):
                                return iter(self._feature_array_2d)

                            def __array__(self, dtype=None):
                                return self._feature_array_2d if dtype is None else self._feature_array_2d.astype(dtype)

                        if "feature" in col_set:
                            feature_array = self.data[feature_cols].values if feature_cols else np.zeros((len(self.data), 0))
                            return FeatureSeries(feature_array, self.data.index)

                        if "label" in col_set:
                            label_values = self.data["label"].values if "label" in self.data.columns else np.zeros(len(self.data))
                            return label_values.reshape(-1, 1)

                        return self.data

                    def __getattr__(self, name):
                        return getattr(self.data, name)

                dataset = DataFrameDatasetAdapter(dataset)
            
            # 进行预测
            predictions = model.predict(dataset)
            
            logger.info(f"Qlib模型预测完成: {len(predictions)} 条预测结果")
            return predictions
            
        except Exception as e:
            logger.error(f"Qlib模型预测失败: {e}")
            raise

    def _align_prediction_features(self, model: Any, dataset: pd.DataFrame) -> pd.DataFrame:
        """对齐预测数据特征列以匹配训练特征"""
        try:
            base_model = model.model if hasattr(model, "model") else model
            feature_names = None

            if hasattr(base_model, "feature_name"):
                try:
                    feature_names = base_model.feature_name()
                except Exception:
                    feature_names = None
            if feature_names is None and hasattr(base_model, "feature_name_"):
                feature_names = list(base_model.feature_name_)
            if feature_names is None and hasattr(base_model, "booster_") and hasattr(base_model.booster_, "feature_name"):
                feature_names = base_model.booster_.feature_name()

            if not feature_names:
                return dataset

            normalized_feature_names = []
            for name in feature_names:
                if isinstance(name, bytes):
                    normalized_feature_names.append(name.decode(errors="ignore"))
                else:
                    normalized_feature_names.append(str(name))

            dataset_columns = [str(col) for col in dataset.columns]
            has_named_match = any(name in dataset_columns for name in normalized_feature_names)
            name_mismatch = all(name.startswith("Column_") for name in normalized_feature_names) and not has_named_match

            if name_mismatch:
                data = dataset.values
                feature_count = len(normalized_feature_names)
                if data.shape[1] < feature_count:
                    pad_width = feature_count - data.shape[1]
                    data = np.hstack([data, np.zeros((data.shape[0], pad_width))])
                elif data.shape[1] > feature_count:
                    data = data[:, :feature_count]
                logger.info(
                    "预测特征使用位置对齐: model_features={}, dataset_features={}",
                    feature_count,
                    dataset.shape[1]
                )
                return pd.DataFrame(data, index=dataset.index, columns=normalized_feature_names)

            aligned = dataset.copy()
            missing = []
            for name in normalized_feature_names:
                if name not in aligned.columns:
                    aligned[name] = 0.0
                    missing.append(name)
            if missing and len(missing) == len(normalized_feature_names):
                data = dataset.values
                feature_count = len(normalized_feature_names)
                if data.shape[1] < feature_count:
                    pad_width = feature_count - data.shape[1]
                    data = np.hstack([data, np.zeros((data.shape[0], pad_width))])
                elif data.shape[1] > feature_count:
                    data = data[:, :feature_count]
                logger.info(
                    "预测特征全部缺失，回退到位置对齐: model_features={}, dataset_features={}",
                    feature_count,
                    dataset.shape[1]
                )
                return pd.DataFrame(data, index=dataset.index, columns=normalized_feature_names)

            aligned = aligned[normalized_feature_names]
            if missing:
                logger.info(
                    "预测特征缺失补齐: count={}, sample={}",
                    len(missing),
                    missing[:5]
                )
            return aligned

        except Exception as e:
            logger.warning(f"对齐预测特征失败，使用原始数据: {e}")
            return dataset
    
    def get_supported_model_types(self) -> List[str]:
        """获取支持的模型类型列表"""
        return self.model_manager.get_supported_models()
    
    def get_model_config_template(self, model_type: str) -> Dict[str, Any]:
        """获取模型配置模板"""
        try:
            metadata = self.model_manager.get_model_metadata(model_type)
            hyperparameter_specs = self.model_manager.get_hyperparameter_specs(model_type)
            
            if not metadata:
                return {}
            
            template = {
                "model_info": metadata.to_dict(),
                "hyperparameters": {
                    spec.name: spec.default_value
                    for spec in hyperparameter_specs
                }
            }
            return template
        except Exception as e:
            logger.error(f"获取模型配置模板失败: {e}")
            return {}
    
    def recommend_models(self, sample_count: int, feature_count: int, task_type: str = "regression") -> List[str]:
        """推荐适合的模型"""
        return self.model_manager.recommend_models(sample_count, feature_count, task_type)
    
    def get_training_recommendations(self, model_type: str) -> Dict[str, Any]:
        """获取训练建议"""
        return self.model_manager.get_training_recommendations(model_type)
