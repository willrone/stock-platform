"""
统一Qlib训练引擎

基于Qlib框架的统一模型训练引擎，替代现有的多种训练方式
支持传统ML模型和深度学习模型的统一训练流程
集成早停策略防止过拟合
"""

import asyncio
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd
from loguru import logger

# 检测Qlib可用性
try:
    from qlib.utils import init_instance_by_config

    QLIB_AVAILABLE = True
    logger.info("Qlib库已成功导入")
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
    elif "lightgbm" in error_msg:
        missing_module = "lightgbm"

    if missing_module:
        logger.warning(
            f"Qlib缺少依赖 {missing_module}。导入错误: {e}\n"
            f"解决方法: pip install {missing_module}\n"
            f"如果还有其他依赖缺失，请运行修复脚本: ./fix_qlib_dependencies.sh\n"
            f"或手动安装所有依赖: pip install setuptools_scm cvxpy dill fire gym jupyter lightgbm matplotlib mlflow nbconvert pymongo python-redis-lock redis 'ruamel.yaml>=0.17.38'\n"
            f"详细说明: 查看 backend/QLIB_INSTALLATION.md"
        )
    else:
        logger.warning(
            f"Qlib未安装或缺少依赖。导入错误: {e}\n"
            f"安装方法: pip install git+https://github.com/microsoft/qlib.git\n"
            f"或使用 Gitee 镜像: pip install git+https://gitee.com/mirrors/qlib.git\n"
            f"如果已安装但缺少依赖，请运行: ./fix_qlib_dependencies.sh\n"
            f"详细说明: 查看 backend/QLIB_INSTALLATION.md"
        )
    QLIB_AVAILABLE = False
except Exception as e:
    logger.error(f"Qlib导入时发生未知错误: {e}")
    QLIB_AVAILABLE = False

from ..automl.early_stopping import EarlyStoppingManager, create_default_early_stopping
from .enhanced_qlib_provider import EnhancedQlibDataProvider
from .performance_monitor import get_performance_monitor
from .qlib_model_manager import QlibModelManager
from .training_engine import (
    QlibTrainingOrchestrator,
    QlibTrainingPipeline,
    QlibTrainingResultAssembler,
    TrainingRequest,
)


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
    # 特征选择配置
    selected_features: Optional[List[str]] = (
        None  # 用户选择的特征列表，None表示使用所有特征
    )
    # 早停策略配置
    enable_early_stopping: bool = True
    early_stopping_monitor: str = "val_loss"
    early_stopping_min_delta: float = 0.001
    enable_overfitting_detection: bool = True
    enable_adaptive_patience: bool = True
    label_normalization: Optional[str] = None
    label_definition: Optional[str] = None

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
            "selected_features": self.selected_features,
            "enable_early_stopping": self.enable_early_stopping,
            "early_stopping_monitor": self.early_stopping_monitor,
            "early_stopping_min_delta": self.early_stopping_min_delta,
            "enable_overfitting_detection": self.enable_overfitting_detection,
            "enable_adaptive_patience": self.enable_adaptive_patience,
            "label_normalization": self.label_normalization,
            "label_definition": self.label_definition,
        }


@dataclass
class QlibTrainingResult:
    """Qlib训练结果"""

    model_path: str
    model_config: Dict[str, Any]
    training_metrics: Dict[str, Any]
    validation_metrics: Dict[str, Any]
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
    signal_quality: Optional[Dict[str, Any]] = None
    segment_evaluation: Optional[Dict[str, Any]] = None

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
            "feature_correlation": self.feature_correlation,
            "signal_quality": self.signal_quality,
            "segment_evaluation": self.segment_evaluation,
        }


class OutlierHandler:
    """异常值处理器 - 对收益率标签进行Winsorize处理"""

    def __init__(
        self,
        method: str = "winsorize",
        lower_percentile: float = 0.01,
        upper_percentile: float = 0.99,
    ):
        """
        初始化异常值处理器

        Args:
            method: 处理方法，'winsorize' 或 'clip'
            lower_percentile: 下分位数（用于Winsorize）
            upper_percentile: 上分位数（用于Winsorize）
        """
        self.method = method
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile

    def handle_label_outliers(
        self, data: pd.DataFrame, label_col: str = "label"
    ) -> pd.DataFrame:
        """
        处理标签中的异常值

        Args:
            data: 数据DataFrame
            label_col: 标签列名

        Returns:
            处理后的DataFrame
        """
        if label_col not in data.columns:
            return data

        data_processed = data.copy()
        label_values = data_processed[label_col]

        # 移除NaN和无穷值
        valid_mask = pd.notna(label_values) & np.isfinite(label_values)
        if not valid_mask.any():
            logger.warning(f"标签列 {label_col} 没有有效值")
            return data_processed

        valid_labels = label_values[valid_mask]

        if self.method == "winsorize":
            # Winsorize方法：将极端值截断到分位数
            lower_bound = valid_labels.quantile(self.lower_percentile)
            upper_bound = valid_labels.quantile(self.upper_percentile)

            # 记录异常值数量
            outliers_lower = (label_values < lower_bound).sum()
            outliers_upper = (label_values > upper_bound).sum()

            if outliers_lower > 0 or outliers_upper > 0:
                logger.info(
                    f"标签异常值处理: 下界={lower_bound:.6f} (异常值={outliers_lower}), "
                    f"上界={upper_bound:.6f} (异常值={outliers_upper})"
                )

            # 截断到分位数
            data_processed[label_col] = label_values.clip(
                lower=lower_bound, upper=upper_bound
            )

        elif self.method == "clip":
            # Clip方法：使用Z-score方法检测异常值
            mean = valid_labels.mean()
            std = valid_labels.std()

            if std > 0:
                z_scores = np.abs((label_values - mean) / std)
                # 使用3倍标准差作为阈值
                threshold = 3.0
                outliers = z_scores > threshold

                if outliers.sum() > 0:
                    logger.info(
                        f"标签异常值处理: 使用Z-score方法，检测到 {outliers.sum()} 个异常值"
                    )
                    # 将异常值截断到阈值
                    data_processed.loc[outliers, label_col] = (
                        np.sign(label_values[outliers] - mean) * threshold * std + mean
                    )

        # 处理极端价格变化（可能是除权除息）
        # 如果收益率超过50%，标记为可疑
        extreme_mask = np.abs(data_processed[label_col]) > 0.5
        if extreme_mask.sum() > 0:
            logger.warning(
                f"检测到 {extreme_mask.sum()} 个极端收益率（>50%），可能是除权除息，已处理"
            )

        return data_processed


class RobustFeatureScaler:
    """鲁棒特征标准化器（时间序列安全）"""

    def __init__(self) -> None:
        try:
            from sklearn.preprocessing import RobustScaler

            self.RobustScaler = RobustScaler
        except ImportError:
            logger.warning("sklearn不可用，特征标准化将使用简单标准化")
            self.RobustScaler = None

        self.scalers: Dict[str, Any] = {}
        self.fitted = False
        self.feature_cols: Optional[List[str]] = None

    def fit_transform(
        self, data: pd.DataFrame, feature_cols: List[str]
    ) -> pd.DataFrame:
        """按时间序列方式标准化（避免未来信息泄漏）"""
        if self.RobustScaler is None:
            logger.warning("sklearn不可用，跳过特征标准化")
            return data

        data_scaled = data.copy()
        self.feature_cols = feature_cols

        # 确保数据按时间排序
        if isinstance(data.index, pd.MultiIndex):
            data_scaled = data_scaled.sort_index()
        elif isinstance(data.index, pd.DatetimeIndex):
            data_scaled = data_scaled.sort_index()

        for col in feature_cols:
            if col not in data_scaled.columns:
                continue

            # 跳过标签列和非数值列
            if col == "label" or not pd.api.types.is_numeric_dtype(data_scaled[col]):
                continue

            try:
                scaler = self.RobustScaler()
                # 只使用历史数据拟合（时间序列安全）
                col_values = data_scaled[col].values.reshape(-1, 1)
                # 移除NaN值进行拟合
                valid_mask = ~np.isnan(col_values.flatten())
                if valid_mask.sum() > 0:
                    scaler.fit(col_values[valid_mask])
                    # 转换所有值（包括NaN，NaN会保持为NaN）
                    data_scaled[col] = scaler.transform(col_values).flatten()
                    self.scalers[col] = scaler
                else:
                    logger.warning(f"列 {col} 全为NaN，跳过标准化")
            except Exception as e:
                logger.warning(f"标准化列 {col} 时出错: {e}，跳过该列")
                continue

        self.fitted = True
        logger.info(f"特征标准化完成，标准化了 {len(self.scalers)} 个特征列")
        return data_scaled

    def transform(
        self, data: pd.DataFrame, feature_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """转换新数据"""
        if not self.fitted:
            raise ValueError("Scaler尚未拟合，请先调用fit_transform")

        if feature_cols is None:
            feature_cols = self.feature_cols

        if feature_cols is None:
            logger.warning("未指定特征列，返回原始数据")
            return data

        if self.RobustScaler is None:
            return data

        data_scaled = data.copy()

        for col in feature_cols:
            if col not in data_scaled.columns or col not in self.scalers:
                continue

            try:
                col_values = data_scaled[col].values.reshape(-1, 1)
                data_scaled[col] = self.scalers[col].transform(col_values).flatten()
            except Exception as e:
                logger.warning(f"转换列 {col} 时出错: {e}，保持原值")
                continue

        return data_scaled


class UnifiedQlibTrainingEngine:
    """统一Qlib训练引擎"""

    def __init__(self, websocket_manager: Any = None) -> None:
        self.websocket_manager = websocket_manager
        self.data_provider = EnhancedQlibDataProvider()
        self.model_manager = QlibModelManager()
        self.early_stopping_manager = None
        self._enable_multiprocessing = False
        self.performance_monitor = get_performance_monitor()
        self.training_pipeline = QlibTrainingPipeline(self)
        self.result_assembler = QlibTrainingResultAssembler(QlibTrainingResult)
        self.training_orchestrator = QlibTrainingOrchestrator(
            engine=self,
            pipeline=self.training_pipeline,
            result_assembler=self.result_assembler,
            qlib_available_getter=lambda: QLIB_AVAILABLE,
        )

        logger.info("统一Qlib训练引擎初始化完成")

    async def initialize(self) -> None:
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
        progress_callback: Any = None,
    ) -> QlibTrainingResult:
        """统一的Qlib模型训练流程。"""
        request = TrainingRequest(
            model_id=model_id,
            model_name=model_name,
            stock_codes=stock_codes,
            start_date=start_date,
            end_date=end_date,
            config=config,
            progress_callback=progress_callback,
        )
        return cast(
            QlibTrainingResult, await self.training_orchestrator.execute(request)
        )

    async def _create_qlib_model_config(
        self, config: QlibTrainingConfig
    ) -> Dict[str, Any]:
        """创建Qlib模型配置"""
        model_name = config.model_type.value

        # 使用模型管理器创建配置
        try:
            qlib_config: Dict[str, Any] = self.model_manager.create_qlib_config(
                model_name, config.hyperparameters
            )
            return qlib_config
        except Exception as e:
            logger.error(f"创建Qlib模型配置失败: {e}")
            raise

    def _process_stock_data(
        self, stock_data: pd.DataFrame, stock_code: str, prediction_horizon: int = 5
    ) -> pd.DataFrame:
        """处理单个股票的数据，包括特征计算和标签生成"""
        try:
            # 复制数据以避免修改原始数据
            processed_data = stock_data.copy()

            # 计算基本特征
            if "$close" in processed_data.columns:
                close = processed_data["$close"]
                # 计算收益率
                processed_data["RET1"] = close.pct_change(1)
                processed_data["RET5"] = close.pct_change(5)
                processed_data["RET20"] = close.pct_change(20)

                # 计算移动平均线
                processed_data["MA5"] = close.rolling(5).mean()
                processed_data["MA20"] = close.rolling(20).mean()

                # 计算标准差
                processed_data["STD5"] = close.rolling(5).std()
                processed_data["STD20"] = close.rolling(20).std()

            if "$volume" in processed_data.columns:
                volume = processed_data["$volume"]
                processed_data["VOL1"] = volume.pct_change(1)
                processed_data["VOL5"] = volume.pct_change(5)

            # 生成标签 - 修复：使用prediction_horizon参数计算未来N天收益率
            if "$close" in processed_data.columns:
                # 正确计算未来N天收益率作为标签
                current_price = processed_data["$close"]
                if isinstance(processed_data.index, pd.MultiIndex):
                    # 按股票分组，计算未来N天的价格
                    future_price = processed_data.groupby(level=0)["$close"].shift(
                        -prediction_horizon
                    )
                else:
                    # 直接计算未来N天的价格
                    future_price = processed_data["$close"].shift(-prediction_horizon)

                # 计算收益率：(未来价格 - 当前价格) / 当前价格
                label_values = (future_price - current_price) / current_price

                if isinstance(label_values, pd.Series):
                    processed_data["label"] = label_values.fillna(0)
                else:
                    processed_data["label"] = pd.Series(
                        (
                            label_values.iloc[:, 0].values
                            if hasattr(label_values, "iloc")
                            else label_values
                        ),
                        index=processed_data.index,
                    ).fillna(0)

                logger.debug(
                    f"股票 {stock_code} 标签创建完成，预测周期={prediction_horizon}天，"
                    f"标签范围=[{processed_data['label'].min():.6f}, {processed_data['label'].max():.6f}]"
                )

            # 填充缺失值
            processed_data = processed_data.fillna(0)

            return processed_data
        except Exception as e:
            logger.error(f"处理股票 {stock_code} 数据时发生错误: {e}")
            return stock_data

    async def _prepare_training_datasets(
        self,
        dataset: pd.DataFrame,
        validation_split: float,
        config: Optional[QlibTrainingConfig] = None,
    ) -> Tuple[Any, Any]:
        """准备训练和验证数据集，返回qlib DatasetH对象"""
        if not QLIB_AVAILABLE:
            error_msg = (
                "Qlib不可用，无法准备数据集。\n"
                "请安装Qlib库：\n"
                "  pip install git+https://github.com/microsoft/qlib.git\n"
                "或者查看安装文档：backend/QLIB_INSTALLATION.md"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        # 确定数据索引类型
        if isinstance(dataset.index, pd.MultiIndex) and dataset.index.nlevels == 2:
            # MultiIndex: (stock_code, date)
            logger.info("使用MultiIndex数据结构，按股票分组并行处理")

            # 按股票分割数据
            stock_groups = {}
            stock_codes = dataset.index.get_level_values(0).unique()

            for stock_code in stock_codes:
                try:
                    stock_data = dataset.xs(stock_code, level=0, drop_level=False)
                    if not stock_data.empty:
                        stock_groups[stock_code] = stock_data
                except KeyError:
                    logger.warning(f"股票 {stock_code} 不在数据中")
                    continue

            # 使用多进程并行处理
            processed_stocks = []

            max_workers = min(mp.cpu_count(), 8)
            # 获取prediction_horizon参数
            prediction_horizon = config.prediction_horizon if config else 5

            # 默认禁用多进程以避免 pickle 序列化问题，可通过实例属性开启
            if (
                len(stock_groups) > 1
                and max_workers > 1
                and self._enable_multiprocessing
            ):
                # 多进程处理
                logger.info(f"使用 {max_workers} 个进程并行处理数据")

                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    futures = {}

                    # 提交任务
                    for stock_code, stock_data in stock_groups.items():
                        future = executor.submit(
                            self._process_stock_data,
                            stock_data,
                            stock_code,
                            prediction_horizon,
                        )
                        futures[future] = stock_code

                    # 收集结果
                    for future in as_completed(futures):
                        stock_code = futures[future]
                        try:
                            processed_data = future.result()
                            if not processed_data.empty:
                                processed_stocks.append(processed_data)
                                logger.debug(f"完成股票 {stock_code} 的数据处理")
                        except Exception as e:
                            logger.error(f"处理股票 {stock_code} 的数据时发生错误: {e}")
            else:
                # 单进程处理
                logger.info("使用单进程处理数据")
                for stock_code, stock_data in stock_groups.items():
                    processed_data = self._process_stock_data(
                        stock_data, stock_code, prediction_horizon
                    )
                    if not processed_data.empty:
                        processed_stocks.append(processed_data)
                        logger.debug(f"完成股票 {stock_code} 的数据处理")

            # 合并处理后的数据
            if processed_stocks:
                dataset = pd.concat(processed_stocks)
                logger.info(f"数据处理完成，合并后数据形状: {dataset.shape}")
            else:
                logger.warning("没有处理任何股票数据")

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

        # 异常值处理（在标签创建后、特征标准化前）
        outlier_handler = OutlierHandler(
            method="winsorize", lower_percentile=0.01, upper_percentile=0.99
        )
        if "label" in train_data.columns:
            logger.info("开始处理标签异常值")
            train_data = outlier_handler.handle_label_outliers(
                train_data, label_col="label"
            )
            if val_data is not None and "label" in val_data.columns:
                val_data = outlier_handler.handle_label_outliers(
                    val_data, label_col="label"
                )
            logger.info("标签异常值处理完成")

        # 特征标准化（时间序列安全）
        feature_scaler = RobustFeatureScaler()
        # 获取特征列（排除标签列）
        feature_cols = [col for col in train_data.columns if col != "label"]

        if feature_cols:
            logger.info(f"开始特征标准化，特征列数: {len(feature_cols)}")
            # 在训练集上拟合并转换
            train_data = feature_scaler.fit_transform(train_data, feature_cols)
            # 在验证集上只转换（使用训练集的统计量）
            if val_data is not None and len(val_data) > 0:
                val_data = feature_scaler.transform(val_data, feature_cols)
            logger.info("特征标准化完成")
        else:
            logger.warning("未找到特征列，跳过特征标准化")

        # 创建DatasetH适配器，使DataFrame具有qlib DatasetH的接口
        class DataFrameDatasetAdapter:
            """将DataFrame适配为qlib DatasetH格式"""

            def __init__(
                self,
                train_data: pd.DataFrame,
                val_data: pd.DataFrame = None,
                prediction_horizon: int = 5,
                primary_segment: str = "train",
            ):
                self.train_data = train_data.copy()
                self.val_data = val_data.copy() if val_data is not None else None
                self.primary_segment = primary_segment
                # qlib模型期望有segments属性，包含train和valid
                self.segments = {"train": self.train_data}
                if self.val_data is not None:
                    self.segments["valid"] = self.val_data
                # data / __len__ 应反映当前适配器代表的主 segment，避免把验证集伪装成训练集
                if self.primary_segment == "valid" and self.val_data is not None:
                    self.data = self.val_data
                else:
                    self.data = self.train_data

                # 处理训练集和验证集的标签
                def _create_label_for_data(
                    data: Any, data_name: Any, horizon: Any
                ) -> Any:
                    """为数据集创建标签 - 修复：使用prediction_horizon参数"""
                    if data is None:
                        return
                    has_label = "label" in data.columns

                    # 尝试找到收盘价列
                    close_col = None
                    for col in ["$close", "close", "Close", "CLOSE"]:
                        if col in data.columns:
                            close_col = col
                            break

                    if not has_label and close_col is not None:
                        # 默认标签: 未来N天收益率
                        current_price = data[close_col]
                        if isinstance(data.index, pd.MultiIndex):
                            future_price = data.groupby(level=0)[close_col].shift(
                                -horizon
                            )
                        else:
                            future_price = data[close_col].shift(-horizon)

                        label_values = (future_price - current_price) / current_price
                        if isinstance(label_values, pd.Series):
                            data["label"] = label_values.fillna(0)
                        else:
                            data["label"] = pd.Series(
                                (
                                    label_values.iloc[:, 0].values
                                    if hasattr(label_values, "iloc")
                                    else label_values
                                ),
                                index=data.index,
                            ).fillna(0)
                    elif not has_label:
                        # 如果没有收盘价，使用最后一列作为标签
                        last_col = data.iloc[:, -1]
                        if isinstance(last_col, pd.Series):
                            data["label"] = last_col
                        else:
                            data["label"] = pd.Series(
                                (
                                    last_col.iloc[:, 0].values
                                    if hasattr(last_col, "iloc")
                                    else last_col
                                ),
                                index=data.index,
                            )
                        logger.warning(
                            f"{data_name}未找到收盘价列，使用最后一列作为标签，标签统计: 非零值={data['label'].abs().gt(1e-6).sum()}, 零值={data['label'].abs().le(1e-6).sum()}, 范围=[{data['label'].min():.6f}, {data['label'].max():.6f}]"
                        )

                    if "label" not in data.columns:
                        return

                    if config and config.label_definition == "future_excess_return_cs":
                        if isinstance(data.index, pd.MultiIndex):
                            date_level = (
                                "datetime"
                                if "datetime" in (data.index.names or [])
                                else data.index.names[-1]
                            )
                            data["label"] = data["label"] - data["label"].groupby(
                                level=date_level
                            ).transform("mean")
                        else:
                            data["label"] = data["label"] - float(data["label"].mean())

                    if config and config.label_normalization == "cs_rank_norm":
                        if isinstance(data.index, pd.MultiIndex):
                            date_level = (
                                "datetime"
                                if "datetime" in (data.index.names or [])
                                else data.index.names[-1]
                            )
                            ranked = (
                                data["label"]
                                .groupby(level=date_level, group_keys=False)
                                .rank(pct=True)
                            )
                        else:
                            ranked = data["label"].rank(pct=True)
                        data["label"] = (ranked - 0.5) * 3.46

                    logger.info(
                        f"{data_name}标签列准备完成，标签统计: 非零值={data['label'].abs().gt(1e-6).sum()}, 零值={data['label'].abs().le(1e-6).sum()}, 范围=[{data['label'].min():.6f}, {data['label'].max():.6f}]"
                    )

                prediction_horizon = config.prediction_horizon if config else 5
                _create_label_for_data(self.train_data, "训练集", prediction_horizon)
                _create_label_for_data(self.val_data, "验证集", prediction_horizon)

                # 记录数据维度信息
                logger.info(
                    f"DataFrameDatasetAdapter初始化: 训练集形状={self.train_data.shape}, 验证集形状={self.val_data.shape if self.val_data is not None else 'N/A'}, 列数={len(self.train_data.columns)}"
                )
                if "label" in self.train_data.columns:
                    label_stats = self.train_data["label"].describe()
                    logger.info(f"训练集标签统计: {label_stats.to_dict()}")
                if self.val_data is not None and "label" in self.val_data.columns:
                    val_label_stats = self.val_data["label"].describe()
                    logger.info(f"验证集标签统计: {val_label_stats.to_dict()}")

            def __len__(self) -> Any:
                return len(self.data)

            def __getitem__(self, key: Any) -> Any:
                if key == "train":
                    return self.train_data
                elif key == "valid" and self.val_data is not None:
                    return self.val_data
                return self.train_data

            def prepare(
                self,
                key: Union[List[str], Tuple[str], str, slice, pd.Index],
                col_set: Optional[Union[List[str], str]] = None,
                data_key: Optional[str] = None,
                **kwargs: Any,
            ) -> Any:
                """实现接近 qlib DatasetH 的 prepare 接口。"""
                if col_set is None:
                    col_set = ["feature", "label"]

                original_col_set = col_set
                if isinstance(col_set, str):
                    col_set = [col_set]

                def _prepare_single(segment_key: Any) -> Any:
                    # 根据key选择对应的数据集
                    if segment_key == "train":
                        data = self.train_data
                    elif segment_key == "valid" and self.val_data is not None:
                        data = self.val_data
                    else:
                        data = self.train_data

                    class LabelSeries:
                        """包装Series，使values返回2D数组以满足qlib的要求"""

                        def __init__(
                            self, values_1d: Any, values_2d: Any, index: Any
                        ) -> None:
                            self._series = pd.Series(values_1d, index=index)
                            self._values_2d = values_2d
                            self._index = index

                        @property
                        def values(self) -> Any:
                            return self._values_2d

                        @property
                        def index(self) -> Any:
                            return self._index

                        def __len__(self) -> Any:
                            return len(self._series)

                        def __getitem__(self, key: Any) -> Any:
                            return self._series[key]

                        def __iter__(self) -> Any:
                            return iter(self._series)

                        def __array__(self, dtype: Any = None) -> Any:
                            return (
                                self._values_2d
                                if dtype is None
                                else self._values_2d.astype(dtype)
                            )

                        def __getattr__(self, name: Any) -> Any:
                            return getattr(self._series, name)

                    # 分离特征和标签
                    all_feature_cols = [col for col in data.columns if col != "label"]
                    if config and config.selected_features:

                        def map_feature_name(feature_name: str) -> List[str]:
                            """将前端特征名称映射到可能的Qlib特征名称"""
                            base_mapping = {
                                "open": ["$open", "open"],
                                "high": ["$high", "high"],
                                "low": ["$low", "low"],
                                "close": ["$close", "close"],
                                "volume": ["$volume", "volume"],
                            }

                            indicator_mapping = {
                                "ma_5": ["MA5", "ma_5", "MA_5"],
                                "ma_10": ["MA10", "ma_10", "MA_10"],
                                "ma_20": ["MA20", "ma_20", "MA_20"],
                                "ma_60": ["MA60", "ma_60", "MA_60"],
                                "sma": ["SMA", "sma"],
                                "ema": ["EMA", "EMA20", "ema"],
                                "rsi": ["RSI14", "RSI", "rsi", "rsi_14"],
                                "macd": ["MACD", "macd"],
                                "macd_signal": [
                                    "MACD_SIGNAL",
                                    "macd_signal",
                                    "MACD_SIGN",
                                ],
                                "macd_histogram": [
                                    "MACD_HIST",
                                    "MACD_HISTOGRAM",
                                    "macd_histogram",
                                ],
                                "bb_upper": [
                                    "BOLL_UPPER",
                                    "BB_UPPER",
                                    "bb_upper",
                                    "bollinger_upper",
                                ],
                                "bb_middle": [
                                    "BOLL_MIDDLE",
                                    "BB_MIDDLE",
                                    "bb_middle",
                                    "bollinger_middle",
                                ],
                                "bb_lower": [
                                    "BOLL_LOWER",
                                    "BB_LOWER",
                                    "bb_lower",
                                    "bollinger_lower",
                                ],
                                "atr": ["ATR14", "ATR", "atr", "atr_14"],
                                "vwap": ["VWAP", "vwap"],
                                "obv": ["OBV", "obv"],
                                "stoch": ["STOCH_K", "STOCH", "stoch", "stoch_k"],
                                "kdj_k": ["KDJ_K", "kdj_k"],
                                "kdj_d": ["KDJ_D", "kdj_d"],
                                "kdj_j": ["KDJ_J", "kdj_j"],
                                "williams_r": ["WILLIAMS_R", "williams_r", "WILLIAMS"],
                                "cci": ["CCI20", "CCI", "cci"],
                                "momentum": ["MOMENTUM", "momentum"],
                                "roc": ["ROC", "roc"],
                                "sar": ["SAR", "sar"],
                                "adx": ["ADX", "adx"],
                                "volume_rsi": ["VOLUME_RSI", "volume_rsi"],
                            }

                            fundamental_mapping = {
                                "price_change": [
                                    "RET1",
                                    "price_change",
                                    "PRICE_CHANGE",
                                ],
                                "price_change_5d": [
                                    "RET5",
                                    "price_change_5d",
                                    "PRICE_CHANGE_5D",
                                ],
                                "price_change_20d": [
                                    "RET20",
                                    "price_change_20d",
                                    "PRICE_CHANGE_20D",
                                ],
                                "volume_change": [
                                    "VOLUME_RET1",
                                    "volume_change",
                                    "VOLUME_CHANGE",
                                ],
                                "volume_ma_ratio": [
                                    "VOLUME_MA_RATIO",
                                    "volume_ma_ratio",
                                ],
                                "volatility_5d": [
                                    "VOLATILITY5",
                                    "volatility_5d",
                                    "VOLATILITY_5D",
                                ],
                                "volatility_20d": [
                                    "VOLATILITY20",
                                    "volatility_20d",
                                    "VOLATILITY_20D",
                                ],
                                "price_position": ["PRICE_POSITION", "price_position"],
                            }

                            all_mapping = {
                                **base_mapping,
                                **indicator_mapping,
                                **fundamental_mapping,
                            }

                            if feature_name in all_mapping:
                                return all_mapping[feature_name]
                            return [feature_name]

                        mapped_features = []
                        for user_feature in config.selected_features:
                            possible_names = map_feature_name(user_feature)
                            found = False
                            for possible_name in possible_names:
                                if possible_name in all_feature_cols:
                                    mapped_features.append(possible_name)
                                    found = True
                                    break
                            if not found:
                                logger.debug(
                                    f"特征 '{user_feature}' 未找到匹配项，尝试的变体: {possible_names}"
                                )

                        feature_cols = [
                            col for col in mapped_features if col in all_feature_cols
                        ]
                        if len(feature_cols) == 0:
                            logger.warning(
                                f"用户指定的特征都不存在，使用所有可用特征。指定特征: {config.selected_features}, 可用特征: {all_feature_cols[:20]}"
                            )
                            feature_cols = all_feature_cols
                        else:
                            missing_features = [
                                col
                                for col in config.selected_features
                                if col
                                not in [
                                    f for f in mapped_features if f in all_feature_cols
                                ]
                            ]
                            if missing_features:
                                logger.warning(
                                    f"以下特征不存在，将被忽略: {missing_features[:10]}"
                                )
                            logger.info(
                                f"使用用户选择的 {len(feature_cols)} 个特征进行训练: {feature_cols[:10]}"
                            )
                    else:
                        feature_cols = all_feature_cols

                    class FeatureSeries:
                        """包装Series，使values返回2D数组"""

                        def __init__(self, feature_array_2d: Any, index: Any) -> None:
                            self._feature_array_2d = feature_array_2d
                            self._index = index

                        @property
                        def values(self) -> Any:
                            return self._feature_array_2d

                        @property
                        def index(self) -> Any:
                            return self._index

                        def __len__(self) -> Any:
                            return len(self._feature_array_2d)

                        def __getitem__(self, key: Any) -> Any:
                            if isinstance(key, (int, np.integer)):
                                return self._feature_array_2d[key]
                            elif isinstance(key, slice):
                                return self._feature_array_2d[key]
                            else:
                                if hasattr(self._index, "get_loc"):
                                    loc = self._index.get_loc(key)
                                    return self._feature_array_2d[loc]
                                return self._feature_array_2d[key]

                        def __iter__(self) -> Any:
                            return iter(self._feature_array_2d)

                        def __array__(self, dtype: Any = None) -> Any:
                            return (
                                self._feature_array_2d
                                if dtype is None
                                else self._feature_array_2d.astype(dtype)
                            )

                        def __getattr__(self, name: Any) -> Any:
                            if hasattr(self._feature_array_2d, name):
                                return getattr(self._feature_array_2d, name)
                            raise AttributeError(
                                f"'{type(self).__name__}' object has no attribute '{name}'"
                            )

                    result_base = pd.DataFrame(index=data.index)
                    feature_obj_final = None
                    label_obj_final = None

                    if "feature" in col_set:
                        if len(feature_cols) > 0:
                            feature_array = data[feature_cols].values
                            feature_obj_final = FeatureSeries(feature_array, data.index)
                            result_base["feature"] = pd.Series(
                                [None] * len(data.index), index=data.index
                            )
                        else:
                            empty_array = np.zeros((len(data), 0))
                            feature_obj_final = FeatureSeries(empty_array, data.index)
                            result_base["feature"] = pd.Series(
                                [None] * len(data.index), index=data.index
                            )

                    if "label" in col_set:
                        if "label" in data.columns:
                            label_series = data["label"]
                            label_values = (
                                label_series.values
                                if isinstance(label_series, pd.Series)
                                else np.array(label_series)
                            )
                            if label_values.ndim == 1:
                                label_values_2d = label_values.reshape(-1, 1)
                                label_values_1d = label_values
                            elif label_values.ndim == 2:
                                if label_values.shape[1] == 1:
                                    label_values_2d = label_values
                                    label_values_1d = label_values.flatten()
                                else:
                                    label_values_2d = label_values[:, 0:1]
                                    label_values_1d = label_values[:, 0]
                            else:
                                label_values_flat = np.array(label_values).flatten()
                                label_values_2d = label_values_flat.reshape(-1, 1)
                                label_values_1d = label_values_flat

                            label_obj = LabelSeries(
                                label_values_1d,
                                label_values_2d,
                                (
                                    label_series.index
                                    if isinstance(label_series, pd.Series)
                                    else data.index
                                ),
                            )
                        else:
                            default_values_1d = np.zeros(len(data))
                            default_values_2d = default_values_1d.reshape(-1, 1)
                            label_obj = LabelSeries(
                                default_values_1d, default_values_2d, data.index
                            )

                        label_obj_final = label_obj
                        result_base["label"] = pd.Series(
                            [None] * len(data.index), index=data.index
                        )
                    else:
                        default_values_1d = np.zeros(len(data))
                        default_values_2d = default_values_1d.reshape(-1, 1)
                        label_obj_final = LabelSeries(
                            default_values_1d, default_values_2d, data.index
                        )
                        result_base["label"] = pd.Series(
                            [None] * len(data.index), index=data.index
                        )

                    if "label" not in col_set:
                        default_values_1d = np.zeros(len(data))
                        default_values_2d = default_values_1d.reshape(-1, 1)
                        label_obj_final = LabelSeries(
                            default_values_1d, default_values_2d, data.index
                        )
                        result_base["label"] = pd.Series(
                            [None] * len(data.index), index=data.index
                        )

                    class CustomDataFrame(pd.DataFrame):
                        """自定义DataFrame，确保label和feature列返回正确的对象"""

                        _metadata = ["_label_series_obj", "_feature_series_obj"]

                        def __init__(
                            self,
                            *args: Any,
                            label_series_obj: Any = None,
                            feature_series_obj: Any = None,
                            **kwargs: Any,
                        ) -> None:
                            super().__init__(*args, **kwargs)
                            object.__setattr__(
                                self, "_label_series_obj", label_series_obj
                            )
                            object.__setattr__(
                                self, "_feature_series_obj", feature_series_obj
                            )

                        def __getitem__(self, key: Any) -> Any:
                            if key == "label" and self._label_series_obj is not None:
                                return self._label_series_obj
                            if (
                                key == "feature"
                                and self._feature_series_obj is not None
                            ):
                                return self._feature_series_obj
                            return super().__getitem__(key)

                    if original_col_set == "feature" or col_set == ["feature"]:
                        if feature_obj_final is not None:
                            return feature_obj_final
                        empty_array = np.zeros((len(data), 0))
                        return FeatureSeries(empty_array, data.index)

                    if original_col_set == "label" or col_set == ["label"]:
                        return label_obj_final

                    return CustomDataFrame(
                        result_base,
                        label_series_obj=label_obj_final,
                        feature_series_obj=feature_obj_final,
                    )

                if isinstance(key, (list, tuple)):
                    return [_prepare_single(str(segment_key)) for segment_key in key]

                return _prepare_single(key)

            def __getattr__(self, name: Any) -> Any:
                # 转发其他属性到DataFrame
                return getattr(self.data, name)

        # 创建包含训练集和验证集的适配器
        prediction_horizon = config.prediction_horizon if config else 5
        combined_adapter = DataFrameDatasetAdapter(
            train_data,
            val_data if len(val_data) > 0 else None,
            prediction_horizon,
            primary_segment="train",
        )
        validation_adapter = DataFrameDatasetAdapter(
            train_data,
            val_data if len(val_data) > 0 else None,
            prediction_horizon,
            primary_segment="valid",
        )
        # 训练仍使用包含 train/valid segments 的主适配器；验证适配器只修正对外暴露的主 segment 语义
        train_dataset = combined_adapter
        val_dataset = validation_adapter

        logger.info(
            f"数据分割完成 - 训练集: {len(train_data)}, 验证集: {len(val_data)}, segments={list(combined_adapter.segments.keys())}"
        )
        return train_dataset, val_dataset

    async def _train_qlib_model(
        self,
        model_config: Dict[str, Any],
        train_dataset: Any,
        val_dataset: Any,
        config: QlibTrainingConfig,
        progress_callback: Any = None,
        model_id: Optional[str] = None,
    ) -> Tuple[Any, List[Dict[str, Any]]]:
        """训练Qlib模型并实时更新进度，集成早停策略"""
        if not QLIB_AVAILABLE:
            raise RuntimeError("Qlib不可用，无法训练模型")

        # 初始化早停管理器
        if config.enable_early_stopping:
            create_default_early_stopping()
            logger.info("早停策略已启用")

        try:
            # 创建模型实例
            if progress_callback and model_id:
                await progress_callback(model_id, 50.0, "training", "创建Qlib模型实例")

            model = init_instance_by_config(model_config)

            # 训练模型
            logger.info("开始Qlib模型训练...")

            if progress_callback and model_id:
                await progress_callback(
                    model_id,
                    55.0,
                    "training",
                    "开始模型训练",
                    {
                        "model_type": config.model_type.value,
                        "train_samples": len(train_dataset),
                        "val_samples": len(val_dataset),
                        "early_stopping_enabled": config.enable_early_stopping,
                    },
                )

            # 训练历史记录
            training_history: Any = []
            early_stopped = False
            stopped_epoch = 0
            best_epoch = 0
            early_stopping_reason = None

            # 记录数据集信息
            logger.info(
                f"准备训练模型: 训练集类型={type(train_dataset)}, 长度={len(train_dataset) if hasattr(train_dataset, '__len__') else 'N/A'}"
            )
            logger.info(
                f"准备训练模型: 验证集类型={type(val_dataset)}, 长度={len(val_dataset) if hasattr(val_dataset, '__len__') else 'N/A'}"
            )
            if hasattr(val_dataset, "data"):
                logger.info(
                    f"验证集数据: {val_dataset.data.shape if hasattr(val_dataset.data, 'shape') else 'N/A'}"
                )

            # 检查模型fit方法的参数
            fit_params = []
            if hasattr(model, "fit"):
                try:
                    import inspect

                    sig = inspect.signature(model.fit)
                    fit_params = list(sig.parameters.keys())
                    logger.info(f"模型fit方法参数: {fit_params}")
                except Exception:
                    if hasattr(model.fit, "__code__"):
                        fit_params = list(model.fit.__code__.co_varnames)
                        logger.info(f"模型fit方法参数(通过co_varnames): {fit_params}")

            # 创建训练进度回调（主要用于非官方模型或回退场景）
            async def training_progress_callback(
                epoch: Any,
                train_loss: Any,
                val_loss: Any = None,
                val_metrics: Any = None,
            ) -> Any:
                if progress_callback and model_id:
                    num_iterations = (
                        config.hyperparameters.get("num_iterations")
                        or config.hyperparameters.get("n_estimators")
                        or config.hyperparameters.get("num_boost_round")
                        or config.early_stopping_patience
                    )
                    progress = 55.0 + (epoch / max(num_iterations, 1)) * 25.0
                    progress = min(progress, 80.0)

                    metrics = {"epoch": epoch, "train_loss": train_loss}
                    if val_loss is not None:
                        metrics["val_loss"] = val_loss
                    if val_metrics:
                        metrics.update(val_metrics)

                    await progress_callback(
                        model_id,
                        progress,
                        "training",
                        f"训练轮次 {epoch}/{num_iterations}",
                        metrics,
                    )

                return False

            # 优先按 Qlib 官方 fit 接口调用：dataset + num_boost_round + early_stopping_rounds + evals_result
            dataset_to_fit = train_dataset
            if hasattr(train_dataset, "segments") and "valid" in train_dataset.segments:
                logger.info(
                    f"使用包含验证集的dataset进行训练，segments: {list(train_dataset.segments.keys())}"
                )
            else:
                logger.warning("dataset不包含验证集segment，仅使用训练集")

            num_boost_round = (
                config.hyperparameters.get("num_iterations")
                or config.hyperparameters.get("n_estimators")
                or config.hyperparameters.get("num_boost_round")
            )
            early_stopping_rounds = (
                config.early_stopping_patience if config.enable_early_stopping else None
            )
            verbose_eval = config.hyperparameters.get("verbose_eval", 20)
            evals_result: Any = {}

            fit_kwargs = {}
            if "num_boost_round" in fit_params and num_boost_round is not None:
                fit_kwargs["num_boost_round"] = num_boost_round
            if (
                "early_stopping_rounds" in fit_params
                and early_stopping_rounds is not None
            ):
                fit_kwargs["early_stopping_rounds"] = early_stopping_rounds
            if "verbose_eval" in fit_params:
                fit_kwargs["verbose_eval"] = verbose_eval
            if "evals_result" in fit_params:
                fit_kwargs["evals_result"] = evals_result

            logger.info(f"按Qlib官方接口调用 model.fit，参数: {fit_kwargs}")
            log_metrics_patched = False
            original_log_metrics = None
            try:
                from qlib.workflow import R

                original_log_metrics = getattr(R, "log_metrics", None)
                if callable(original_log_metrics):
                    R.log_metrics = lambda *args, **kwargs: None  # type: ignore[method-assign]
                    log_metrics_patched = True
            except Exception as e:
                logger.debug(f"禁用 Qlib workflow metric logging 失败，继续训练: {e}")

            try:
                model.fit(dataset_to_fit, **fit_kwargs)
            finally:
                if log_metrics_patched and original_log_metrics is not None:
                    try:
                        from qlib.workflow import R

                        R.log_metrics = original_log_metrics  # type: ignore[method-assign]
                    except Exception:
                        logger.debug(
                            "恢复 Qlib workflow metric logging 失败", exc_info=True
                        )

            # 优先使用官方 evals_result 重建训练历史
            try:
                if not evals_result:
                    if hasattr(model, "evals_result_"):
                        evals_result = model.evals_result_
                    elif hasattr(model, "model") and hasattr(
                        model.model, "best_iteration"
                    ):
                        logger.debug(
                            "模型未直接暴露 evals_result，回退到 best_iteration 信息"
                        )

                if evals_result:
                    train_metrics = evals_result.get("train", {})
                    valid_metrics = evals_result.get("valid", {})
                    preferred_metric = None
                    for candidate in ["l2", "rmse", "loss", "binary_logloss"]:
                        if candidate in train_metrics or candidate in valid_metrics:
                            preferred_metric = candidate
                            break
                    if preferred_metric is None:
                        metric_keys = list(train_metrics.keys() or valid_metrics.keys())
                        preferred_metric = metric_keys[0] if metric_keys else None

                    if preferred_metric:
                        train_losses = train_metrics.get(preferred_metric, [])
                        valid_losses = valid_metrics.get(preferred_metric, [])
                        max_epochs = max(len(train_losses), len(valid_losses))
                        training_history.clear()
                        for epoch in range(max_epochs):
                            train_loss = (
                                train_losses[epoch]
                                if epoch < len(train_losses)
                                else None
                            )
                            val_loss = (
                                valid_losses[epoch]
                                if epoch < len(valid_losses)
                                else None
                            )
                            training_history.append(
                                {
                                    "epoch": int(epoch + 1),
                                    "train_loss": (
                                        float(round(train_loss, 4))
                                        if train_loss is not None
                                        else None
                                    ),
                                    "val_loss": (
                                        float(round(val_loss, 4))
                                        if val_loss is not None
                                        else None
                                    ),
                                    "train_accuracy": 0.0,
                                    "val_accuracy": 0.0,
                                    "learning_rate": float(
                                        config.hyperparameters.get(
                                            "learning_rate", 0.001
                                        )
                                    ),
                                }
                            )

                        logger.info(
                            f"从官方 evals_result 获取训练历史: {len(training_history)} 轮, metric={preferred_metric}"
                        )
            except Exception as e:
                logger.debug(
                    f"无法从官方 evals_result 获取训练历史: {e}", exc_info=True
                )

            # 读取官方模型的真实 best_iteration / early stopping 结果
            actual_best_iteration = None
            if hasattr(model, "model") and hasattr(model.model, "best_iteration"):
                actual_best_iteration = getattr(model.model, "best_iteration", None)
            elif hasattr(model, "best_iteration"):
                actual_best_iteration = getattr(model, "best_iteration", None)

            if actual_best_iteration:
                best_epoch = int(actual_best_iteration)
                if num_boost_round and best_epoch < num_boost_round:
                    early_stopped = True
                    stopped_epoch = best_epoch
                    early_stopping_reason = "Qlib/LightGBM官方早停"

            # 如果没有拿到真实历史，才生成轻量回退历史
            if not training_history:
                fallback_epochs = (
                    best_epoch
                    or num_boost_round
                    or config.hyperparameters.get("num_iterations")
                    or config.hyperparameters.get("n_estimators")
                    or config.early_stopping_patience
                )
                for epoch in range(1, max(int(fallback_epochs or 1), 1) + 1):
                    await training_progress_callback(epoch, 0.0, 0.0)
                    training_history.append(
                        {
                            "epoch": epoch,
                            "train_loss": None,
                            "val_loss": None,
                            "train_accuracy": 0.0,
                            "val_accuracy": 0.0,
                            "learning_rate": config.hyperparameters.get(
                                "learning_rate", 0.001
                            ),
                        }
                    )

            # 更新训练进度
            if progress_callback and model_id:
                final_message = "模型训练完成"
                if early_stopped:
                    final_message = f"训练提前停止 ({early_stopping_reason})"

                await progress_callback(
                    model_id,
                    80.0,
                    "training",
                    final_message,
                    {
                        "early_stopped": early_stopped,
                        "stopped_epoch": stopped_epoch,
                        "best_epoch": best_epoch,
                        "total_epochs": len(training_history),
                    },
                )

            logger.info(
                f"Qlib模型训练完成 - 早停: {early_stopped}, 总轮次: {len(training_history)}"
            )

            # 返回模型和训练历史，包含早停信息
            return cast(
                Tuple[Any, List[Dict[str, Any]]],
                (
                    model,
                    training_history,
                    {
                        "early_stopped": early_stopped,
                        "stopped_epoch": stopped_epoch,
                        "best_epoch": best_epoch,
                        "early_stopping_reason": early_stopping_reason,
                    },
                ),
            )

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
        progress_callback: Callable[..., Any],
    ) -> Any:
        """模拟带早停的训练过程（用于不支持回调的模型）"""
        logger.info("使用模拟训练过程进行早停检查")

        for epoch in range(1, config.early_stopping_patience + 1):
            # 模拟训练指标
            train_loss = 0.5 * (0.9**epoch) + 0.01 + np.random.normal(0, 0.005)
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
        test_dataset: Any = None,
        model_id: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """评估模型性能并计算详细指标"""
        try:
            # 记录数据集信息
            if hasattr(train_dataset, "data") and isinstance(
                train_dataset.data, pd.DataFrame
            ):
                logger.info(
                    f"训练集数据维度: {train_dataset.data.shape}, 列: {list(train_dataset.data.columns[:10]) if len(train_dataset.data.columns) > 0 else 'N/A'}"
                )
            elif isinstance(train_dataset, pd.DataFrame):
                logger.info(
                    f"训练集数据维度: {train_dataset.shape}, 列: {list(train_dataset.columns[:10]) if len(train_dataset.columns) > 0 else 'N/A'}"
                )

            if hasattr(val_dataset, "data") and isinstance(
                val_dataset.data, pd.DataFrame
            ):
                logger.info(
                    f"验证集数据维度: {val_dataset.data.shape}, 列: {list(val_dataset.data.columns[:10]) if len(val_dataset.data.columns) > 0 else 'N/A'}"
                )
            elif isinstance(val_dataset, pd.DataFrame):
                logger.info(
                    f"验证集数据维度: {val_dataset.shape}, 列: {list(val_dataset.columns[:10]) if len(val_dataset.columns) > 0 else 'N/A'}"
                )

            # 训练集预测 - 使用正确的segment
            train_pred = model.predict(train_dataset, segment="train")
            logger.info(
                f"训练集预测结果: 类型={type(train_pred)}, 形状={train_pred.shape if hasattr(train_pred, 'shape') else len(train_pred) if hasattr(train_pred, '__len__') else 'N/A'}"
            )

            # 验证集预测 - 使用正确的segment
            val_pred = model.predict(val_dataset, segment="valid")
            logger.info(
                f"验证集预测结果: 类型={type(val_pred)}, 形状={val_pred.shape if hasattr(val_pred, 'shape') else len(val_pred) if hasattr(val_pred, '__len__') else 'N/A'}"
            )

            # 计算训练集指标（使用真实标签）
            training_metrics = self._calculate_metrics(
                train_dataset, train_pred, "训练集", model_id
            )

            # 计算验证集指标（使用真实标签）
            validation_metrics = self._calculate_metrics(
                val_dataset, val_pred, "验证集", model_id
            )
            validation_signal_quality = self._calculate_signal_quality(
                val_dataset, val_pred, "验证集"
            )

            test_metrics = self._get_default_metrics()
            test_signal_quality = {
                "ic": None,
                "icir": None,
                "rank_ic": None,
                "rank_icir": None,
                "long_short_ann_return": None,
                "long_short_ann_sharpe": None,
                "long_avg_ann_return": None,
                "long_avg_ann_sharpe": None,
                "sample_count": 0,
                "analysis_scope": "test",
            }
            if test_dataset is not None:
                try:
                    test_pred = model.predict(test_dataset, segment="test")
                    test_shape = (
                        test_pred.shape
                        if hasattr(test_pred, "shape")
                        else len(test_pred) if hasattr(test_pred, "__len__") else "N/A"
                    )
                    logger.info(
                        f"测试集预测结果: 类型={type(test_pred)}, 形状={test_shape}"
                    )
                    test_metrics = self._calculate_metrics(
                        test_dataset, test_pred, "测试集", model_id
                    )
                    test_signal_quality = self._calculate_signal_quality(
                        test_dataset, test_pred, "测试集"
                    )
                except Exception as e:
                    logger.warning(
                        f"测试集评估失败，保留默认测试指标: {e}", exc_info=True
                    )

            train_samples = (
                len(train_dataset) if hasattr(train_dataset, "__len__") else 0
            )
            validation_samples = (
                len(val_dataset) if hasattr(val_dataset, "__len__") else 0
            )
            test_samples = (
                len(test_dataset)
                if test_dataset is not None and hasattr(test_dataset, "__len__")
                else 0
            )

            segment_evaluation = {
                "train": {
                    "dataset_samples": train_samples,
                    "evaluated_samples": int(
                        training_metrics.get("sample_count", 0) or 0
                    ),
                    "performance_metrics": training_metrics,
                    "signal_quality": self._calculate_signal_quality(
                        train_dataset, train_pred, "训练集"
                    ),
                },
                "validation": {
                    "dataset_samples": validation_samples,
                    "evaluated_samples": int(
                        validation_metrics.get("sample_count", 0) or 0
                    ),
                    "performance_metrics": validation_metrics,
                    "signal_quality": validation_signal_quality,
                },
                "test": {
                    "dataset_samples": test_samples,
                    "evaluated_samples": int(test_metrics.get("sample_count", 0) or 0),
                    "performance_metrics": test_metrics,
                    "signal_quality": test_signal_quality,
                },
            }

            logger.info(
                "模型评估完成 - "
                f"训练准确率: {training_metrics.get('accuracy', 0.0):.4f}, "
                f"验证准确率: {validation_metrics.get('accuracy', 0.0):.4f}, "
                f"RankIC: {validation_signal_quality.get('rank_ic')}"
            )
            return (
                training_metrics,
                validation_metrics,
                validation_signal_quality,
                segment_evaluation,
            )

        except Exception as e:
            logger.error(f"模型评估失败: {e}", exc_info=True)
            # 返回默认指标
            default_metrics = self._get_default_metrics()
            default_signal_quality = {
                "ic": None,
                "icir": None,
                "rank_ic": None,
                "rank_icir": None,
                "long_short_ann_return": None,
                "long_short_ann_sharpe": None,
                "long_avg_ann_return": None,
                "long_avg_ann_sharpe": None,
                "sample_count": 0,
                "analysis_scope": "validation",
            }
            return (
                default_metrics,
                default_metrics,
                default_signal_quality,
                {
                    "train": {
                        "dataset_samples": 0,
                        "evaluated_samples": 0,
                        "performance_metrics": default_metrics,
                        "signal_quality": {
                            **default_signal_quality,
                            "analysis_scope": "train",
                        },
                    },
                    "validation": {
                        "dataset_samples": 0,
                        "evaluated_samples": 0,
                        "performance_metrics": default_metrics,
                        "signal_quality": default_signal_quality,
                    },
                    "test": {
                        "dataset_samples": 0,
                        "evaluated_samples": 0,
                        "performance_metrics": default_metrics,
                        "signal_quality": {
                            **default_signal_quality,
                            "analysis_scope": "test",
                        },
                    },
                },
            )

    def _extract_evaluation_inputs(
        self,
        dataset: pd.DataFrame,
        predictions: Any,
        dataset_name: str,
    ) -> Optional[Dict[str, Any]]:
        """统一提取评估所需的标签、预测值与索引。"""
        try:
            import numpy as np

            y_true = None
            y_index = None

            lower_name = str(dataset_name).lower()
            if "测试" in dataset_name or "test" in lower_name:
                segment = "test"
            elif "验证" in dataset_name or "valid" in lower_name:
                segment = "valid"
            else:
                segment = "train"

            def _extract_label_from_prepared(prepared_obj: Any) -> Any:
                if prepared_obj is None:
                    return None, None
                if isinstance(prepared_obj, pd.Series):
                    return prepared_obj.values, prepared_obj.index
                if isinstance(prepared_obj, pd.DataFrame):
                    if "label" in prepared_obj.columns:
                        label_series = prepared_obj["label"]
                        return label_series.values, label_series.index
                    if isinstance(prepared_obj.columns, pd.MultiIndex):
                        level0 = prepared_obj.columns.get_level_values(0)
                        label_mask = level0 == "label"
                        if label_mask.any():
                            label_df = prepared_obj.loc[:, label_mask]
                            if (
                                isinstance(label_df, pd.DataFrame)
                                and not label_df.empty
                            ):
                                label_series = label_df.iloc[:, 0]
                                return label_series.values, label_series.index
                    if prepared_obj.shape[1] == 1:
                        label_series = prepared_obj.iloc[:, 0]
                        return label_series.values, label_series.index
                if hasattr(prepared_obj, "_series"):
                    series_obj = prepared_obj._series
                    return series_obj.values, series_obj.index
                if hasattr(prepared_obj, "values"):
                    values = prepared_obj.values
                    values = (
                        values.flatten() if getattr(values, "ndim", 1) == 2 else values
                    )
                    return values, getattr(prepared_obj, "index", None)
                return np.asarray(prepared_obj).flatten(), getattr(
                    prepared_obj, "index", None
                )

            if hasattr(dataset, "data") and isinstance(dataset.data, pd.DataFrame):
                if hasattr(dataset, "segments") and segment in dataset.segments:
                    segment_data = dataset.segments[segment]
                    if (
                        isinstance(segment_data, pd.DataFrame)
                        and "label" in segment_data.columns
                    ):
                        label_series = segment_data["label"]
                        if hasattr(label_series, "_series"):
                            y_true = label_series._series.values
                            y_index = label_series._series.index
                        else:
                            y_true = label_series.values
                            y_index = label_series.index
                elif "label" in dataset.data.columns:
                    y_true = dataset.data["label"].values
                    y_index = dataset.data.index
                elif hasattr(dataset, "prepare"):
                    try:
                        prepared = dataset.prepare(segment, col_set=["label"])
                        if (
                            isinstance(prepared, pd.DataFrame)
                            and "label" in prepared.columns
                        ):
                            label_col = prepared["label"]
                            if hasattr(label_col, "_series"):
                                y_true = label_col._series.values
                                y_index = label_col._series.index
                            elif hasattr(label_col, "values"):
                                label_values = label_col.values
                                y_true = (
                                    label_values.flatten()
                                    if getattr(label_values, "ndim", 1) == 2
                                    else label_values
                                )
                                y_index = label_col.index
                            else:
                                y_true = np.array(label_col).flatten()
                                y_index = prepared.index
                    except Exception as e:
                        logger.debug(f"通过prepare方法获取标签失败: {e}")

            if (
                y_true is None
                and hasattr(dataset, "dataset")
                and hasattr(dataset.dataset, "prepare")
            ):
                try:
                    prepared = dataset.dataset.prepare(segment, col_set="label")
                    y_true, y_index = _extract_label_from_prepared(prepared)
                except Exception as e:
                    logger.debug(
                        f"通过official adapter dataset.prepare获取标签失败: {e}"
                    )

            if y_true is None and hasattr(dataset, "prepare"):
                try:
                    prepared = dataset.prepare(segment, col_set="label")
                    y_true, y_index = _extract_label_from_prepared(prepared)
                except Exception as e:
                    logger.debug(f"通过prepare获取标签失败: {e}")

            if y_true is None and isinstance(dataset, pd.DataFrame):
                if "label" in dataset.columns:
                    label_col = dataset["label"]
                    if hasattr(label_col, "_series"):
                        y_true = label_col._series.values
                        y_index = label_col._series.index
                    else:
                        y_true = label_col.values
                        y_index = label_col.index

            if y_true is None:
                logger.warning(f"数据集 {dataset_name} 中没有找到label列")
                return None

            if isinstance(predictions, pd.Series):
                y_pred = predictions.values
                pred_index = predictions.index
            elif isinstance(predictions, np.ndarray):
                y_pred = predictions.flatten() if predictions.ndim > 1 else predictions
                pred_index = getattr(predictions, "index", None)
            else:
                y_pred = np.array(predictions).flatten()
                pred_index = getattr(predictions, "index", None)

            min_len = min(len(y_true), len(y_pred))
            if min_len == 0:
                logger.warning(f"数据集 {dataset_name} 为空")
                return None

            y_true = np.asarray(y_true[:min_len])
            y_pred = np.asarray(y_pred[:min_len])
            if y_true.ndim > 1:
                y_true = y_true.reshape(-1)
            if y_pred.ndim > 1:
                y_pred = y_pred.reshape(-1)
            if y_index is not None:
                y_index = y_index[:min_len]
            elif pred_index is not None:
                y_index = pred_index[:min_len]

            valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
            if valid_mask.sum() == 0:
                logger.warning(f"数据集 {dataset_name} 中没有有效数据")
                return None

            y_true = y_true[valid_mask]
            y_pred = y_pred[valid_mask]
            if y_index is not None:
                y_index_array = np.asarray(y_index)
                y_index = y_index_array[valid_mask]

            return {
                "y_true": y_true,
                "y_pred": y_pred,
                "y_index": y_index,
            }
        except Exception as e:
            logger.error(f"提取评估输入失败: {e}", exc_info=True)
            return None

    def _calculate_signal_quality(
        self,
        dataset: pd.DataFrame,
        predictions: Any,
        dataset_name: str,
    ) -> Dict[str, Any]:
        """按 Qlib 官方思路计算信号质量指标。"""
        lower_name = str(dataset_name).lower()
        if "测试" in dataset_name or "test" in lower_name:
            analysis_scope = "test"
        elif "验证" in dataset_name or "valid" in lower_name:
            analysis_scope = "validation"
        else:
            analysis_scope = "train"
        evaluation_inputs = self._extract_evaluation_inputs(
            dataset, predictions, dataset_name
        )
        default_result = {
            "ic": None,
            "icir": None,
            "rank_ic": None,
            "rank_icir": None,
            "long_short_ann_return": None,
            "long_short_ann_sharpe": None,
            "long_avg_ann_return": None,
            "long_avg_ann_sharpe": None,
            "sample_count": 0,
            "analysis_scope": analysis_scope,
        }
        if evaluation_inputs is None:
            return default_result

        import numpy as np
        import pandas as pd

        y_true = evaluation_inputs["y_true"]
        y_pred = evaluation_inputs["y_pred"]
        y_index = evaluation_inputs["y_index"]

        df = pd.DataFrame({"pred": y_pred, "label": y_true})
        if y_index is not None:
            df.index = y_index

        if isinstance(df.index, pd.MultiIndex):
            date_level = (
                "datetime"
                if "datetime" in (df.index.names or [])
                else df.index.names[-1]
            )
            grouped = df.groupby(level=date_level, group_keys=False)
        else:
            grouped = [("all", df)]

        ic_values = []
        ric_values = []
        long_short_returns = []
        long_avg_returns = []

        for _, group in grouped:
            group = group.dropna()
            if group.empty:
                continue

            ic = group["pred"].corr(group["label"])
            rank_ic = group["pred"].rank().corr(group["label"].rank())
            if pd.notna(ic):
                ic_values.append(float(ic))
            if pd.notna(rank_ic):
                ric_values.append(float(rank_ic))

            quantile_n = max(1, int(len(group) * 0.2))
            long_group = group.nlargest(quantile_n, columns="pred")
            short_group = group.nsmallest(quantile_n, columns="pred")
            long_return = long_group["label"].mean()
            short_return = short_group["label"].mean()
            avg_return = group["label"].mean()
            if pd.notna(long_return) and pd.notna(short_return):
                long_short_returns.append(float((long_return - short_return) / 2))
            if pd.notna(avg_return):
                long_avg_returns.append(float(avg_return))

        def _safe_mean(values: Any) -> Any:
            return float(np.mean(values)) if values else None

        def _safe_ir(values: Any) -> Any:
            if len(values) <= 1:
                return None
            std = float(np.std(values, ddof=1))
            if std <= 0:
                return None
            return float(np.mean(values) / std)

        def _safe_ann_return(values: Any) -> Any:
            return float(np.mean(values) * 252) if values else None

        def _safe_ann_sharpe(values: Any) -> Any:
            if len(values) <= 1:
                return None
            std = float(np.std(values, ddof=1))
            if std <= 0:
                return None
            return float(np.mean(values) / std * np.sqrt(252))

        result = {
            "ic": _safe_mean(ic_values),
            "icir": _safe_ir(ic_values),
            "rank_ic": _safe_mean(ric_values),
            "rank_icir": _safe_ir(ric_values),
            "long_short_ann_return": _safe_ann_return(long_short_returns),
            "long_short_ann_sharpe": _safe_ann_sharpe(long_short_returns),
            "long_avg_ann_return": _safe_ann_return(long_avg_returns),
            "long_avg_ann_sharpe": _safe_ann_sharpe(long_avg_returns),
            "sample_count": int(len(y_true)),
            "analysis_scope": analysis_scope,
        }
        return result

    def _calculate_metrics(
        self,
        dataset: pd.DataFrame,
        predictions: Any,
        dataset_name: str,
        model_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """计算真实的评估指标，基于预测值和真实标签"""
        try:
            import numpy as np
            from sklearn.metrics import (
                accuracy_score,
                f1_score,
                mean_absolute_error,
                mean_squared_error,
                precision_score,
                r2_score,
                recall_score,
            )

            evaluation_inputs = self._extract_evaluation_inputs(
                dataset, predictions, dataset_name
            )
            if evaluation_inputs is None:
                logger.warning(
                    f"数据集 {dataset_name} 中没有有效评估输入，使用默认指标"
                )
                return self._get_default_metrics()

            y_true = evaluation_inputs["y_true"]
            y_pred = evaluation_inputs["y_pred"]

            # 记录标签统计信息
            logger.info(
                f"{dataset_name} 标签统计 - 样本数: {len(y_true)}, 非零值: {np.count_nonzero(y_true)}, 零值: {np.sum(np.abs(y_true) < 1e-6)}, 范围: [{np.min(y_true):.6f}, {np.max(y_true):.6f}]"
            )

            # 计算回归指标
            mse = float(mean_squared_error(y_true, y_pred))
            mae = float(mean_absolute_error(y_true, y_pred))
            r2 = float(r2_score(y_true, y_pred))

            # 计算方向准确率（预测涨跌方向）
            # 使用阈值而不是sign，避免0值问题
            threshold = 1e-6  # 很小的阈值，用于判断是否为0
            y_true_direction = np.where(
                y_true > threshold, 1, np.where(y_true < -threshold, -1, 0)
            )
            y_pred_direction = np.where(
                y_pred > threshold, 1, np.where(y_pred < -threshold, -1, 0)
            )

            # 记录方向分布信息
            unique_true = np.unique(y_true_direction)
            unique_pred = np.unique(y_pred_direction)
            true_counts: Dict[Any, Any] = {
                val: np.sum(y_true_direction == val) for val in unique_true
            }
            pred_counts: Dict[Any, Any] = {
                val: np.sum(y_pred_direction == val) for val in unique_pred
            }
            logger.info(
                f"{dataset_name} 方向分布 - 真实: {true_counts}, 预测: {pred_counts}, 样本数: {len(y_true_direction)}"
            )

            # 如果所有方向都相同，准确率计算会有问题
            if len(unique_true) == 1 and len(unique_pred) == 1:
                if unique_true[0] == unique_pred[0]:
                    direction_accuracy = 1.0
                else:
                    direction_accuracy = 0.0
            else:
                direction_accuracy = float(
                    accuracy_score(y_true_direction, y_pred_direction)
                )

            logger.info(
                f"{dataset_name} 方向准确率: {direction_accuracy:.4f}, 真实值范围: [{y_true.min():.6f}, {y_true.max():.6f}], 预测值范围: [{y_pred.min():.6f}, {y_pred.max():.6f}]"
            )

            # 对于回归任务，使用方向准确率作为准确率
            accuracy = direction_accuracy

            # 计算分类指标（基于方向）
            try:
                # 确保有正负样本
                if (
                    len(np.unique(y_true_direction)) > 1
                    and len(np.unique(y_pred_direction)) > 1
                ):
                    precision = float(
                        precision_score(
                            y_true_direction,
                            y_pred_direction,
                            average="weighted",
                            zero_division=0,
                        )
                    )
                    recall = float(
                        recall_score(
                            y_true_direction,
                            y_pred_direction,
                            average="weighted",
                            zero_division=0,
                        )
                    )
                    f1 = float(
                        f1_score(
                            y_true_direction,
                            y_pred_direction,
                            average="weighted",
                            zero_division=0,
                        )
                    )
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
                sharpe_ratio = float(
                    np.mean(returns) / np.std(returns) * np.sqrt(252)
                )  # 年化
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
            win_rate = (
                float(np.sum(returns > 0) / len(returns)) if len(returns) > 0 else 0.0
            )

            # 信息比率（相对于基准）
            if len(returns) > 1 and np.std(returns - y_true) > 0:
                information_ratio = float(
                    np.mean(returns - y_true) / np.std(returns - y_true) * np.sqrt(252)
                )
            else:
                information_ratio = 0.0

            # Calmar比率（年化收益率/最大回撤）
            if max_drawdown < 0 and len(returns) > 0:
                annualized_return = np.mean(returns) * 252
                calmar_ratio = (
                    float(annualized_return / abs(max_drawdown))
                    if max_drawdown != 0
                    else 0.0
                )
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
                "calmar_ratio": calmar_ratio,
                "sample_count": int(len(y_true)),
            }

            logger.info(
                f"计算 {dataset_name} 真实指标 - 准确率: {accuracy:.4f}, MSE: {mse:.4f}, R2: {r2:.4f}"
            )
            return {k: round(v, 4) for k, v in metrics.items()}

        except Exception as e:
            logger.error(f"计算真实指标失败: {e}", exc_info=True)
            return self._get_default_metrics()

    def _get_default_metrics(self) -> Dict[str, Any]:
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
            "calmar_ratio": 0.0,
            "sample_count": 0,
        }

    async def _extract_feature_importance(
        self, model: Any, model_type: QlibModelType
    ) -> Optional[Dict[str, float]]:
        """提取特征重要性"""
        try:
            # 对于树模型，尝试获取特征重要性
            if model_type in [QlibModelType.LIGHTGBM, QlibModelType.XGBOOST]:
                if hasattr(model, "get_feature_importance"):
                    importance = model.get_feature_importance()
                    if isinstance(importance, dict):
                        return importance
                elif hasattr(model, "feature_importances_"):
                    # 假设有特征名称列表
                    feature_names = [
                        f"feature_{i}" for i in range(len(model.feature_importances_))
                    ]
                    return dict(zip(feature_names, model.feature_importances_))

            # 对于其他模型类型，返回None
            return None

        except Exception as e:
            logger.warning(f"提取特征重要性失败: {e}")
            return None

    def _analyze_feature_correlations(self, dataset: pd.DataFrame) -> Dict[str, Any]:
        """分析特征与标签的相关性"""
        try:
            if hasattr(dataset, "for_segment"):
                return {
                    "skipped": True,
                    "reason": (
                        "official DatasetH adapter uses qlib handlers; "
                        "skip pandas-only correlation analysis"
                    ),
                }
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
                    data["label"] = (
                        data.groupby(level=0)[close_col]
                        .pct_change(periods=1)
                        .shift(-1)
                        .fillna(0)
                    )
                else:
                    data["label"] = (
                        data[close_col].pct_change(periods=1).shift(-1).fillna(0)
                    )

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
                        high_corr_pairs.append(
                            {
                                "feature1": numeric_features[i],
                                "feature2": numeric_features[j],
                                "correlation": float(corr),
                            }
                        )

            return {
                "target_correlations": target_correlations,
                "high_correlation_pairs": high_corr_pairs,
                "avg_target_correlation": (
                    float(np.mean(list(target_correlations.values())))
                    if target_correlations
                    else 0.0
                ),
                "max_target_correlation": (
                    float(max(target_correlations.values()))
                    if target_correlations
                    else 0.0
                ),
            }

        except Exception as e:
            logger.warning(f"特征相关性分析失败: {e}")
            return {"error": str(e)}

    async def _save_qlib_model(
        self, model: Any, model_id: str, model_config: Dict[str, Any]
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

            with open(model_path, "wb") as f:
                pickle.dump(
                    {"model": model, "config": model_config, "timestamp": timestamp}, f
                )

            logger.info(f"Qlib模型保存成功: {model_path}")
            return str(model_path)

        except Exception as e:
            logger.error(f"保存Qlib模型失败: {e}")
            raise

    async def load_qlib_model(self, model_path: str) -> Tuple[Any, Dict[str, Any]]:
        """加载Qlib模型"""
        try:
            import pickle

            with open(model_path, "rb") as f:
                model_data = pickle.load(f)

            model = model_data["model"]
            config = model_data["config"]

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
        end_date: datetime,
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
                use_cache=True,
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
                    missing_count = sum(
                        1 for name in feature_names if name not in dataset.columns
                    )
                    logger.info(
                        "预测特征对齐: model_features={}, dataset_features={}, missing_filled={}",
                        len(feature_names),
                        len(dataset.columns),
                        missing_count,
                    )

                class DataFrameDatasetAdapter:
                    """将DataFrame适配为qlib DatasetH格式（用于预测）"""

                    def __init__(self, data: pd.DataFrame):
                        self.data = data
                        self.segments = {"test": data}

                    def prepare(
                        self,
                        key: str,
                        col_set: Optional[Union[List[str], str]] = None,
                        data_key: Optional[str] = None,
                    ) -> Any:
                        if col_set is None:
                            col_set = ["feature"]
                        if isinstance(col_set, str):
                            col_set = [col_set]

                        feature_cols = [
                            col for col in self.data.columns if col != "label"
                        ]

                        class FeatureSeries:
                            def __init__(
                                self, feature_array_2d: Any, index: Any
                            ) -> None:
                                self._feature_array_2d = feature_array_2d
                                self._index = index

                            @property
                            def values(self) -> Any:
                                return self._feature_array_2d

                            @property
                            def index(self) -> Any:
                                return self._index

                            def __len__(self) -> Any:
                                return len(self._feature_array_2d)

                            def __getitem__(self, key: Any) -> Any:
                                if isinstance(key, (int, np.integer)):
                                    return self._feature_array_2d[key]
                                if isinstance(key, slice):
                                    return self._feature_array_2d[key]
                                if hasattr(self._index, "get_loc"):
                                    loc = self._index.get_loc(key)
                                    return self._feature_array_2d[loc]
                                return self._feature_array_2d[key]

                            def __iter__(self) -> Any:
                                return iter(self._feature_array_2d)

                            def __array__(self, dtype: Any = None) -> Any:
                                return (
                                    self._feature_array_2d
                                    if dtype is None
                                    else self._feature_array_2d.astype(dtype)
                                )

                        if "feature" in col_set:
                            feature_array = (
                                self.data[feature_cols].values
                                if feature_cols
                                else np.zeros((len(self.data), 0))
                            )
                            return FeatureSeries(feature_array, self.data.index)

                        if "label" in col_set:
                            label_values = (
                                self.data["label"].values
                                if "label" in self.data.columns
                                else np.zeros(len(self.data))
                            )
                            return label_values.reshape(-1, 1)

                        return self.data

                    def __getattr__(self, name: Any) -> Any:
                        return getattr(self.data, name)

                dataset = DataFrameDatasetAdapter(dataset)

            # 进行预测
            predictions = model.predict(dataset)

            logger.info(f"Qlib模型预测完成: {len(predictions)} 条预测结果")
            return predictions

        except Exception as e:
            logger.error(f"Qlib模型预测失败: {e}")
            raise

    def _align_prediction_features(
        self, model: Any, dataset: pd.DataFrame
    ) -> pd.DataFrame:
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
            if (
                feature_names is None
                and hasattr(base_model, "booster_")
                and hasattr(base_model.booster_, "feature_name")
            ):
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
            has_named_match = any(
                name in dataset_columns for name in normalized_feature_names
            )
            name_mismatch = (
                all(name.startswith("Column_") for name in normalized_feature_names)
                and not has_named_match
            )

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
                    dataset.shape[1],
                )
                return pd.DataFrame(
                    data, index=dataset.index, columns=normalized_feature_names
                )

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
                    dataset.shape[1],
                )
                return pd.DataFrame(
                    data, index=dataset.index, columns=normalized_feature_names
                )

            aligned = aligned[normalized_feature_names]
            if missing:
                logger.info(
                    "预测特征缺失补齐: count={}, sample={}", len(missing), missing[:5]
                )
            return aligned

        except Exception as e:
            logger.warning(f"对齐预测特征失败，使用原始数据: {e}")
            return dataset

    def get_supported_model_types(self) -> List[str]:
        """获取支持的模型类型列表"""
        return list(self.model_manager.get_supported_models())

    def get_model_config_template(self, model_type: str) -> Dict[str, Any]:
        """获取模型配置模板"""
        try:
            metadata = self.model_manager.get_model_metadata(model_type)
            hyperparameter_specs = self.model_manager.get_hyperparameter_specs(
                model_type
            )

            if not metadata:
                return {}

            template = {
                "model_info": metadata.to_dict(),
                "hyperparameters": {
                    spec.name: spec.default_value for spec in hyperparameter_specs
                },
            }
            return template
        except Exception as e:
            logger.error(f"获取模型配置模板失败: {e}")
            return {}

    def recommend_models(
        self, sample_count: int, feature_count: int, task_type: str = "regression"
    ) -> List[str]:
        """推荐适合的模型"""
        return list(
            self.model_manager.recommend_models(sample_count, feature_count, task_type)
        )

    def get_training_recommendations(self, model_type: str) -> Dict[str, Any]:
        """获取训练建议"""
        recommendations: Dict[str, Any] = (
            self.model_manager.get_training_recommendations(model_type)
        )
        return recommendations
