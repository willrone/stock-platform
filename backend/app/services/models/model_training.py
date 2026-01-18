"""
机器学习模型训练服务

集成Qlib框架，实现多模态特征工程和现代深度学习模型训练。
支持Transformer、TimesNet、PatchTST、Informer等现代模型，
以及LSTM、XGBoost等基线模型。
"""

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
from loguru import logger

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb

# 检测可用的计算设备
def get_device():
    """
    检测并返回最佳可用设备
    优先级：ROCm > CUDA > CPU
    """
    if torch.cuda.is_available():
        # 检查是否是AMD GPU (ROCm)
        device_name = torch.cuda.get_device_name(0)
        if 'AMD' in device_name or 'Radeon' in device_name:
            logger.info(f"检测到AMD GPU: {device_name}，使用ROCm")
            return torch.device('cuda')
        else:
            logger.info(f"检测到NVIDIA GPU: {device_name}，使用CUDA")
            return torch.device('cuda')
    else:
        logger.info("未检测到GPU，使用CPU")
        return torch.device('cpu')
try:
    import qlib
    from qlib.config import REG_CN, C
    from qlib.data import D
    from qlib.data.dataset import DatasetH
    from qlib.data.filter import NameDFilter, ExpressionDFilter
    from qlib.utils import init_instance_by_config
    QLIB_AVAILABLE = True
except ImportError as e:
    error_msg = str(e)
    missing_module = None
    
    # 检测缺失的模块
    if "setuptools_scm" in error_msg:
        missing_module = "setuptools_scm"
    elif "ruamel" in error_msg or "ruamel.yaml" in error_msg:
        missing_module = "ruamel.yaml"
    elif "cvxpy" in error_msg:
        missing_module = "cvxpy"
    
    if missing_module:
        logger.warning(
            f"Qlib缺少依赖 {missing_module}: {e}\n"
            f"解决方法: pip install {missing_module}\n"
            f"或运行修复脚本: ./fix_qlib_dependencies.sh"
        )
    else:
        logger.warning(f"Qlib未安装，某些功能将不可用: {e}")
    QLIB_AVAILABLE = False

# 导入其他依赖
from ..data.simple_data_service import SimpleDataService
from ..prediction.technical_indicators import TechnicalIndicatorCalculator
try:
    from .modern_models import TimesNet, PatchTST, Informer
    MODERN_MODELS_AVAILABLE = True
except ImportError:
    MODERN_MODELS_AVAILABLE = False
    TimesNet = None
    PatchTST = None
    Informer = None

try:
    from .model_evaluation import ModelEvaluator, ModelVersionManager, BacktestMetrics
    MODEL_EVALUATION_AVAILABLE = True
except ImportError:
    MODEL_EVALUATION_AVAILABLE = False
    ModelEvaluator = None
    ModelVersionManager = None
    BacktestMetrics = None

try:
    from .advanced_training import AdvancedTrainingService, EnsembleConfig, OnlineLearningConfig, EnsembleMethod
    ADVANCED_TRAINING_AVAILABLE = True
except ImportError:
    ADVANCED_TRAINING_AVAILABLE = False
    AdvancedTrainingService = None
    EnsembleConfig = None
    OnlineLearningConfig = None
    EnsembleMethod = None

# 使用 loguru 日志记录器（已在文件顶部导入）


class ModelType(Enum):
    """支持的模型类型"""
    TRANSFORMER = "transformer"
    TIMESNET = "timesnet"
    PATCHTST = "patchtst"
    INFORMER = "informer"
    LSTM = "lstm"
    XGBOOST = "xgboost"


@dataclass
class TrainingConfig:
    """模型训练配置"""
    model_type: ModelType
    sequence_length: int = 60  # 输入序列长度
    prediction_horizon: int = 5  # 预测天数
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    feature_columns: List[str] = None
    target_column: str = "close"
    
    def __post_init__(self):
        if self.feature_columns is None:
            self.feature_columns = [
                "open", "high", "low", "close", "volume",
                "ma_5", "ma_10", "ma_20", "ma_60",
                "rsi", "macd", "macd_signal", "bb_upper", "bb_lower"
            ]


@dataclass
class ModelMetrics:
    """模型评估指标"""
    accuracy: float
    precision: float
    recall: float
    sharpe_ratio: float
    max_drawdown: float
    total_return: float
    win_rate: float
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "total_return": self.total_return,
            "win_rate": self.win_rate
        }


class QlibDataProvider:
    """Qlib数据提供器，集成本地Parquet数据"""
    
    def __init__(self, data_service: SimpleDataService):
        self.data_service = data_service
        self.indicator_calculator = TechnicalIndicatorCalculator()
        
    async def initialize_qlib(self):
        """初始化Qlib环境"""
        try:
            # 在使用memory://模式时，需要先设置mount_path和provider_uri，否则qlib会报错
            from app.core.config import settings
            
            # 使用配置中的QLIB_DATA_PATH，如果不存在则创建
            qlib_data_path = Path(settings.QLIB_DATA_PATH).resolve()
            qlib_data_path.mkdir(parents=True, exist_ok=True)
            
            # 准备mount_path和provider_uri配置
            # qlib.init()内部会调用C.set()重置配置，所以需要通过参数传递
            mount_path_config = {
                "day": str(qlib_data_path),
                "1min": str(qlib_data_path),
            }
            
            provider_uri_config = {
                "day": "memory://",
                "1min": "memory://",
            }
            
            # 使用内存模式，通过kwargs传递配置，避免被C.set()重置
            # 注意：provider_uri作为字典传递时，会覆盖字符串形式的provider_uri
            qlib.init(
                region=REG_CN,
                provider_uri=provider_uri_config,
                mount_path=mount_path_config
            )
            logger.info("Qlib环境初始化成功")
        except Exception as e:
            logger.error(f"Qlib初始化失败: {e}")
            raise
    
    async def prepare_features(
        self, 
        stock_codes: List[str], 
        start_date: datetime, 
        end_date: datetime
    ) -> pd.DataFrame:
        """
        准备多模态特征集
        
        Args:
            stock_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            包含所有特征的DataFrame
        """
        all_features = []
        
        for stock_code in stock_codes:
            try:
                # 优先从本地文件加载数据
                from app.services.data.stock_data_loader import StockDataLoader
                from app.core.config import settings
                
                loader = StockDataLoader(data_root=settings.DATA_ROOT_PATH)
                stock_data = loader.load_stock_data(stock_code, start_date=start_date, end_date=end_date)
                
                # 如果本地没有数据，尝试从远端服务获取
                if stock_data.empty:
                    logger.info(f"本地无数据，从远端服务获取: {stock_code}")
                    stock_data_list = await self.data_service.get_stock_data(
                        stock_code, start_date, end_date
                    )
                    
                    if not stock_data_list or len(stock_data_list) == 0:
                        logger.warning(f"股票 {stock_code} 无数据")
                        continue
                    
                    # 转换为DataFrame格式
                    stock_data = pd.DataFrame([{
                        'date': item.date,
                        'open': item.open,
                        'high': item.high,
                        'low': item.low,
                        'close': item.close,
                        'volume': item.volume
                    } for item in stock_data_list])
                    stock_data = stock_data.set_index('date')
                
                # 确保数据有正确的列名
                if stock_data.empty:
                    logger.warning(f"股票 {stock_code} 无数据")
                    continue
                
                # 计算技术指标
                indicators = await self.indicator_calculator.calculate_all_indicators(
                    stock_data
                )
                
                # 合并数据（使用索引合并，因为两者都使用日期作为索引）
                if not indicators.empty:
                    features = stock_data.merge(indicators, left_index=True, right_index=True, how='left')
                else:
                    features = stock_data.copy()
                features['stock_code'] = stock_code
                # 确保有date列用于后续排序（如果索引是日期，将其作为列）
                if 'date' not in features.columns and isinstance(features.index, pd.DatetimeIndex):
                    features = features.reset_index()
                    features.rename(columns={'index': 'date'}, inplace=True)
                elif 'date' not in features.columns:
                    # 如果索引不是日期，尝试从索引名称推断
                    features = features.reset_index()
                    if 'date' not in features.columns:
                        # 如果还是没有date列，使用索引作为date
                        features['date'] = features.index
                
                # 添加基本面特征（暂时使用简单的价格衍生特征）
                features = self._add_fundamental_features(features)
                
                all_features.append(features)
                
            except Exception as e:
                logger.error(f"处理股票 {stock_code} 特征时出错: {e}")
                continue
        
        if not all_features:
            raise ValueError("没有成功处理任何股票数据")
        
        # 合并所有股票的特征
        combined_features = pd.concat(all_features, ignore_index=True)
        combined_features = combined_features.sort_values(['stock_code', 'date'])
        
        logger.info(f"成功准备了 {len(stock_codes)} 只股票的特征数据，共 {len(combined_features)} 条记录")
        return combined_features
    
    def _add_fundamental_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加基本面特征（简化版本）"""
        # 价格变化率
        df['price_change'] = df['close'].pct_change()
        df['price_change_5d'] = df['close'].pct_change(periods=5)
        df['price_change_20d'] = df['close'].pct_change(periods=20)
        
        # 成交量变化率
        df['volume_change'] = df['volume'].pct_change()
        df['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        
        # 波动率
        df['volatility_5d'] = df['price_change'].rolling(5).std()
        df['volatility_20d'] = df['price_change'].rolling(20).std()
        
        # 价格位置
        df['price_position'] = (df['close'] - df['low'].rolling(20).min()) / (
            df['high'].rolling(20).max() - df['low'].rolling(20).min()
        )
        
        return df


class ModelTrainingService:
    """模型训练服务主类"""
    
    def __init__(self):
        self.data_provider = None
        # 修复路径问题：使用配置中的路径
        from app.core.config import settings
        models_dir_path = Path(settings.MODEL_STORAGE_PATH)
        if not models_dir_path.is_absolute():
            # 如果是相对路径，从backend目录解析
            backend_dir = Path(__file__).parent.parent.parent
            self.models_dir = (backend_dir / models_dir_path).resolve()
        else:
            self.models_dir = models_dir_path
        try:
            self.models_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning(f"创建模型目录失败: {e}, 使用默认路径")
            # 如果失败，使用默认路径
            backend_dir = Path(__file__).parent.parent.parent
            self.models_dir = (backend_dir / "data" / "models").resolve()
            self.models_dir.mkdir(parents=True, exist_ok=True)
        self.evaluator = ModelEvaluator() if ModelEvaluator is not None else None
        self.version_manager = ModelVersionManager() if ModelVersionManager is not None else None
        self.advanced_training = None  # 高级训练服务
        
    async def initialize(self):
        """初始化服务"""
        from ..data.simple_data_service import SimpleDataService
        data_service = SimpleDataService()
        
        self.data_provider = QlibDataProvider(data_service)
        # QlibDataProvider 的初始化在需要时进行，不需要提前初始化
        
        # 初始化高级训练服务（传入self避免循环依赖）
        if ADVANCED_TRAINING_AVAILABLE and AdvancedTrainingService is not None:
            self.advanced_training = AdvancedTrainingService(self)
            if hasattr(self.advanced_training, 'initialize'):
                await self.advanced_training.initialize()
        
        logger.info("模型训练服务初始化完成")
    
    async def train_model(
        self,
        model_id: str,
        stock_codes: List[str],
        config: TrainingConfig,
        start_date: datetime,
        end_date: datetime
    ) -> Tuple[str, BacktestMetrics]:
        """
        训练模型（增强版，包含完整评估）
        
        Args:
            model_id: 模型唯一标识
            stock_codes: 训练用的股票代码列表
            config: 训练配置
            start_date: 训练数据开始日期
            end_date: 训练数据结束日期
            
        Returns:
            (模型版本信息, 评估指标)
        """
        logger.info(f"开始训练模型 {model_id}，类型: {config.model_type.value}")
        
        # 准备训练数据
        features_df = await self.data_provider.prepare_features(
            stock_codes, start_date, end_date
        )
        
        # 数据预处理
        X, y = self._prepare_training_data(features_df, config)
        
        # 准备价格数据用于评估
        actual_prices = self._extract_prices_for_evaluation(features_df, config)
        
        # 时间序列交叉验证分割
        train_X, train_y, val_X, val_y = self._time_series_split(X, y, config.validation_split)
        
        # 根据模型类型训练
        if config.model_type == ModelType.XGBOOST:
            model = await self._train_xgboost(
                train_X, train_y, val_X, val_y, config
            )
        else:
            model = await self._train_deep_learning_model(
                train_X, train_y, val_X, val_y, config
            )
        
        # 全面评估模型
        logger.info("开始模型评估...")
        if self.evaluator is not None:
            metrics = await self.evaluator.evaluate_model(
                model, X, y, actual_prices, config.model_type.value
            )
        else:
            # 如果评估器不可用，创建基本的评估指标
            logger.warning("ModelEvaluator不可用，使用基本评估指标")
            from dataclasses import dataclass
            @dataclass
            class BasicMetrics:
                accuracy: float = 0.0
                precision: float = 0.0
                recall: float = 0.0
                f1_score: float = 0.0
                total_return: float = 0.0
                sharpe_ratio: float = 0.0
                max_drawdown: float = 0.0
            metrics = BasicMetrics()
        
        # 保存模型版本
        if self.version_manager is not None:
            model_version = self.version_manager.save_model_version(
                model_id=model_id,
                model=model,
                model_type=config.model_type.value,
                parameters=config.__dict__,
                metrics=metrics,
                training_data=(X, y)
            )
        else:
            logger.warning("ModelVersionManager不可用，跳过版本保存")
            model_version = None
        
        logger.info(f"模型 {model_id} 训练完成")
        logger.info(f"评估结果 - 准确率: {metrics.accuracy:.4f}, 夏普比率: {metrics.sharpe_ratio:.4f}")
        logger.info(f"总收益: {metrics.total_return:.4f}, 最大回撤: {metrics.max_drawdown:.4f}")
        
        return model_version.file_path, metrics
    
    def _extract_prices_for_evaluation(
        self, 
        features_df: pd.DataFrame, 
        config: TrainingConfig
    ) -> np.ndarray:
        """提取价格数据用于评估"""
        # 按股票和日期排序
        features_df = features_df.sort_values(['stock_code', 'date'])
        
        # 提取收盘价
        prices = features_df[config.target_column].values
        
        return prices
    
    def _prepare_training_data(
        self, 
        features_df: pd.DataFrame, 
        config: TrainingConfig
    ) -> Tuple[np.ndarray, np.ndarray]:
        """准备训练数据"""
        # 选择特征列
        feature_cols = [col for col in config.feature_columns if col in features_df.columns]
        
        # 填充缺失值
        features_df[feature_cols] = features_df[feature_cols].ffill().fillna(0)
        
        # 为每只股票创建序列数据
        X_list, y_list = [], []
        
        for stock_code in features_df['stock_code'].unique():
            stock_data = features_df[features_df['stock_code'] == stock_code].copy()
            stock_data = stock_data.sort_values('date')
            
            if len(stock_data) < config.sequence_length + config.prediction_horizon:
                continue
            
            # 创建滑动窗口
            for i in range(len(stock_data) - config.sequence_length - config.prediction_horizon + 1):
                # 输入序列
                X_seq = stock_data[feature_cols].iloc[i:i+config.sequence_length].values
                
                # 目标值（未来收益率）
                current_price = stock_data[config.target_column].iloc[i+config.sequence_length-1]
                future_price = stock_data[config.target_column].iloc[i+config.sequence_length+config.prediction_horizon-1]
                
                # 计算收益率并转换为分类标签（上涨=1，下跌=0）
                return_rate = (future_price - current_price) / current_price
                y_label = 1 if return_rate > 0 else 0
                
                X_list.append(X_seq)
                y_list.append(y_label)
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        logger.info(f"准备训练数据完成，样本数: {len(X)}, 特征维度: {X.shape}")
        return X, y
    
    def _time_series_split(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        validation_split: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """时间序列数据分割"""
        split_idx = int(len(X) * (1 - validation_split))
        
        train_X, train_y = X[:split_idx], y[:split_idx]
        val_X, val_y = X[split_idx:], y[split_idx:]
        
        logger.info(f"数据分割完成，训练集: {len(train_X)}, 验证集: {len(val_X)}")
        return train_X, train_y, val_X, val_y
    async def _train_xgboost(
        self,
        train_X: np.ndarray,
        train_y: np.ndarray,
        val_X: np.ndarray,
        val_y: np.ndarray,
        config: TrainingConfig
    ) -> Any:
        """训练XGBoost模型"""
        logger.info("开始训练XGBoost模型")
        
        # 将3D数据展平为2D（XGBoost不支持3D输入）
        train_X_flat = train_X.reshape(train_X.shape[0], -1)
        val_X_flat = val_X.reshape(val_X.shape[0], -1)
        
        # XGBoost参数
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 6,
            'learning_rate': config.learning_rate,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        
        # 创建DMatrix
        dtrain = xgb.DMatrix(train_X_flat, label=train_y)
        dval = xgb.DMatrix(val_X_flat, label=val_y)
        
        # 训练模型
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=config.epochs,
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=config.early_stopping_patience,
            verbose_eval=False
        )
        
        logger.info("XGBoost模型训练完成")
        return model
    
    async def _train_deep_learning_model(
        self,
        train_X: np.ndarray,
        train_y: np.ndarray,
        val_X: np.ndarray,
        val_y: np.ndarray,
        config: TrainingConfig
    ) -> Any:
        """训练深度学习模型（支持ROCm和CUDA）"""
        logger.info(f"开始训练深度学习模型: {config.model_type.value}")
        
        # 获取最佳可用设备
        device = get_device()
        
        # 转换为PyTorch张量
        train_X_tensor = torch.FloatTensor(train_X).to(device)
        train_y_tensor = torch.LongTensor(train_y).to(device)
        val_X_tensor = torch.FloatTensor(val_X).to(device)
        val_y_tensor = torch.LongTensor(val_y).to(device)
        
        # 创建模型
        input_dim = train_X.shape[-1]  # 特征维度
        seq_len = train_X.shape[1]     # 序列长度
        
        if config.model_type == ModelType.LSTM:
            model = LSTMModel(input_dim, hidden_dim=128, num_layers=2, num_classes=2)
        elif config.model_type == ModelType.TRANSFORMER:
            model = TransformerModel(input_dim, d_model=128, nhead=8, num_layers=4, num_classes=2)
        elif config.model_type == ModelType.TIMESNET:
            model = TimesNet(
                input_dim=input_dim,
                seq_len=seq_len,
                num_classes=2,
                d_model=64,
                d_ff=256,
                num_kernels=6,
                top_k=5
            )
        elif config.model_type == ModelType.PATCHTST:
            model = PatchTST(
                input_dim=input_dim,
                seq_len=seq_len,
                num_classes=2,
                patch_len=min(16, seq_len // 4),
                stride=min(8, seq_len // 8),
                d_model=128,
                nhead=8,
                num_layers=3
            )
        elif config.model_type == ModelType.INFORMER:
            model = Informer(
                input_dim=input_dim,
                seq_len=seq_len,
                num_classes=2,
                d_model=min(512, input_dim * 8),  # 根据输入维度调整
                nhead=8,
                num_encoder_layers=2,
                num_decoder_layers=1
            )
        else:
            # 默认使用LSTM
            model = LSTMModel(input_dim, hidden_dim=128, num_layers=2, num_classes=2)
        
        model = model.to(device)
        
        # 检查模型是否成功移动到GPU
        if device.type == 'cuda':
            logger.info(f"模型已移动到GPU: {torch.cuda.get_device_name(device)}")
            logger.info(f"GPU内存使用: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
        
        # 损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        
        # 训练循环
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(config.epochs):
            # 训练阶段
            model.train()
            train_loss = 0
            
            # 简单的批次训练（实际应用中应该使用DataLoader）
            for i in range(0, len(train_X_tensor), config.batch_size):
                batch_X = train_X_tensor[i:i+config.batch_size]
                batch_y = train_y_tensor[i:i+config.batch_size]
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # 验证阶段
            model.eval()
            with torch.no_grad():
                val_outputs = model(val_X_tensor)
                val_loss = criterion(val_outputs, val_y_tensor).item()
                
                # 早停检查
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # 保存最佳模型状态
                    best_model_state = model.state_dict().copy()
                else:
                    patience_counter += 1
                
                if patience_counter >= config.early_stopping_patience:
                    logger.info(f"早停触发，在第 {epoch+1} 轮停止训练")
                    break
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{config.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                if device.type == 'cuda':
                    logger.info(f"GPU内存使用: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
        
        # 恢复最佳模型状态
        model.load_state_dict(best_model_state)
        
        # 清理GPU内存
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            logger.info("GPU内存已清理")
        
        logger.info(f"深度学习模型训练完成: {config.model_type.value}")
        return model
    
    def _calculate_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        y_pred_proba: np.ndarray
    ) -> ModelMetrics:
        """计算模型评估指标"""
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        
        # 计算金融指标（简化版本）
        # 假设预测正确时获得正收益，错误时获得负收益
        returns = []
        for i in range(len(y_true)):
            if y_pred[i] == y_true[i]:
                returns.append(0.01)  # 1%收益
            else:
                returns.append(-0.01)  # -1%损失
        
        returns = np.array(returns)
        
        # 夏普比率
        sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
        
        # 最大回撤
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # 总收益
        total_return = cumulative_returns[-1] - 1
        
        # 胜率
        win_rate = np.sum(returns > 0) / len(returns)
        
        return ModelMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            total_return=total_return,
            win_rate=win_rate
        )
    
    async def _save_model(
        self,
        model_id: str,
        model: Any,
        config: TrainingConfig,
        metrics: ModelMetrics
    ) -> str:
        """保存模型和元数据"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{model_id}_{config.model_type.value}_{timestamp}"
        
        # 保存模型文件
        if config.model_type == ModelType.XGBOOST:
            model_path = self.models_dir / f"{model_filename}.json"
            model.save_model(str(model_path))
        else:
            model_path = self.models_dir / f"{model_filename}.pth"
            torch.save(model.state_dict(), model_path)
        
        # 保存模型元数据到数据库
        async with get_db_session() as session:
            # 这里应该插入到model_metadata表，但为了简化先记录日志
            metadata = {
                "id": model_id,
                "name": model_filename,
                "type": config.model_type.value,
                "version": "1.0",
                "parameters": config.__dict__,
                "performance_metrics": metrics.to_dict(),
                "file_path": str(model_path),
                "created_at": datetime.now().isoformat(),
                "is_active": True
            }
            
            logger.info(f"模型元数据: {json.dumps(metadata, indent=2, ensure_ascii=False)}")
        
        return str(model_path)
    
    async def create_ensemble_model(
        self,
        ensemble_id: str,
        base_model_ids: List[str],
        ensemble_method: str = "voting",
        weights: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        创建集成模型
        
        Args:
            ensemble_id: 集成模型ID
            base_model_ids: 基础模型ID列表
            ensemble_method: 集成方法 (voting, weighted, stacking, bagging)
            weights: 权重（用于加权集成）
            
        Returns:
            集成模型信息
        """
        logger.info(f"开始创建集成模型 {ensemble_id}")
        
        # 创建集成配置
        method_map = {
            "voting": EnsembleMethod.VOTING,
            "weighted": EnsembleMethod.WEIGHTED,
            "stacking": EnsembleMethod.STACKING,
            "bagging": EnsembleMethod.BAGGING
        }
        
        config = EnsembleConfig(
            method=method_map.get(ensemble_method, EnsembleMethod.VOTING),
            base_models=base_model_ids,
            weights=weights
        )
        
        # 准备验证数据（使用最近的数据）
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)
        
        # 获取一些股票数据作为验证集
        stock_codes = ["000001.SZ", "000002.SZ", "600000.SH"]
        features_df = await self.data_provider.prepare_features(
            stock_codes, start_date, end_date
        )
        
        # 准备验证数据
        training_config = TrainingConfig(ModelType.XGBOOST)  # 使用默认配置
        X, y = self._prepare_training_data(features_df, training_config)
        
        # 使用后20%作为验证数据
        split_idx = int(len(X) * 0.8)
        validation_data = (X[split_idx:], y[split_idx:])
        
        # 创建集成模型
        ensemble_info = await self.advanced_training.create_ensemble_model(
            ensemble_id, config, validation_data
        )
        
        logger.info(f"集成模型 {ensemble_id} 创建完成")
        return ensemble_info
    
    async def setup_online_learning(
        self,
        model_id: str,
        update_frequency: int = 5,
        memory_size: int = 1000,
        adaptation_threshold: float = 0.1
    ) -> Dict[str, Any]:
        """
        为模型设置在线学习
        
        Args:
            model_id: 模型ID
            update_frequency: 更新频率（天）
            memory_size: 记忆缓冲区大小
            adaptation_threshold: 性能下降阈值
            
        Returns:
            在线学习设置信息
        """
        logger.info(f"为模型 {model_id} 设置在线学习")
        
        config = OnlineLearningConfig(
            update_frequency=update_frequency,
            memory_size=memory_size,
            adaptation_threshold=adaptation_threshold
        )
        
        setup_info = await self.advanced_training.setup_online_learning(model_id, config)
        
        logger.info(f"模型 {model_id} 在线学习设置完成")
        return setup_info
    
    async def update_model_with_new_data(
        self,
        model_id: str,
        new_stock_codes: List[str],
        start_date: datetime,
        end_date: datetime,
        current_model: Any
    ) -> Tuple[Any, Dict[str, float]]:
        """
        使用新数据更新模型（在线学习）
        
        Args:
            model_id: 模型ID
            new_stock_codes: 新的股票代码列表
            start_date: 新数据开始日期
            end_date: 新数据结束日期
            current_model: 当前模型
            
        Returns:
            (更新后的模型, 性能指标)
        """
        logger.info(f"使用新数据更新模型 {model_id}")
        
        # 准备新的训练数据
        features_df = await self.data_provider.prepare_features(
            new_stock_codes, start_date, end_date
        )
        
        training_config = TrainingConfig(ModelType.XGBOOST)  # 使用默认配置
        new_X, new_y = self._prepare_training_data(features_df, training_config)
        
        # 在线更新模型
        updated_model, metrics = await self.advanced_training.update_model_online(
            model_id, new_X, new_y, current_model
        )
        
        logger.info(f"模型 {model_id} 在线更新完成，准确率: {metrics.get('accuracy', 0):.4f}")
        return updated_model, metrics
    
    async def train_ensemble_models(
        self,
        ensemble_id: str,
        stock_codes: List[str],
        start_date: datetime,
        end_date: datetime,
        model_types: List[str] = None
    ) -> Dict[str, Any]:
        """
        训练多个基础模型并创建集成模型
        
        Args:
            ensemble_id: 集成模型ID
            stock_codes: 股票代码列表
            start_date: 训练数据开始日期
            end_date: 训练数据结束日期
            model_types: 要训练的模型类型列表
            
        Returns:
            集成模型信息和基础模型信息
        """
        logger.info(f"开始训练集成模型 {ensemble_id} 的基础模型")
        
        if model_types is None:
            model_types = ["xgboost", "lstm", "transformer"]
        
        # 训练基础模型
        base_model_ids = []
        base_model_results = []
        
        for i, model_type in enumerate(model_types):
            model_id = f"{ensemble_id}_base_{model_type}_{i}"
            
            try:
                config = TrainingConfig(
                    model_type=ModelType(model_type),
                    epochs=50,  # 减少训练轮数以加快速度
                    batch_size=32
                )
                
                model_path, metrics = await self.train_model(
                    model_id, stock_codes, config, start_date, end_date
                )
                
                base_model_ids.append(model_id)
                base_model_results.append({
                    "model_id": model_id,
                    "model_type": model_type,
                    "model_path": model_path,
                    "metrics": metrics.__dict__ if hasattr(metrics, '__dict__') else metrics
                })
                
                logger.info(f"基础模型 {model_id} 训练完成")
                
            except Exception as e:
                logger.error(f"训练基础模型 {model_id} 失败: {e}")
                continue
        
        if not base_model_ids:
            raise ValueError("没有成功训练任何基础模型")
        
        # 创建集成模型
        ensemble_info = await self.create_ensemble_model(
            ensemble_id, base_model_ids, "voting"
        )
        
        result = {
            "ensemble_info": ensemble_info,
            "base_models": base_model_results,
            "total_base_models": len(base_model_ids)
        }
        
        logger.info(f"集成模型 {ensemble_id} 训练完成，包含 {len(base_model_ids)} 个基础模型")
        return result


# 深度学习模型定义

class LSTMModel(nn.Module):
    """LSTM基线模型"""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, num_classes: int):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        # LSTM前向传播
        lstm_out, _ = self.lstm(x)
        
        # 取最后一个时间步的输出
        last_output = lstm_out[:, -1, :]
        
        # Dropout和全连接层
        output = self.dropout(last_output)
        output = self.fc(output)
        
        return output


class TransformerModel(nn.Module):
    """Transformer模型"""
    
    def __init__(self, input_dim: int, d_model: int, nhead: int, num_layers: int, num_classes: int):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(d_model // 2, num_classes)
        )
    
    def forward(self, x):
        # 输入投影
        x = self.input_projection(x)
        
        # 位置编码
        x = self.pos_encoding(x)
        
        # Transformer编码
        transformer_out = self.transformer(x)
        
        # 全局平均池化
        pooled = torch.mean(transformer_out, dim=1)
        
        # 分类
        output = self.classifier(pooled)
        
        return output


class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(1), :].transpose(0, 1)


# 导出主要类和函数
__all__ = [
    'ModelTrainingService',
    'TrainingConfig', 
    'ModelType',
    'ModelMetrics',
    'QlibDataProvider']