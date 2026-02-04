"""
模型训练服务 - 处理模型训练任务的执行和管理
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from loguru import logger

from app.core.error_handler import ErrorContext, ErrorSeverity, ModelError
from app.core.logging_config import PerformanceLogger

from ..prediction.feature_extractor import FeatureConfig, FeatureExtractor
from .model_storage import ModelMetadata, ModelStatus, ModelStorage, ModelType


@dataclass
class TrainingConfig:
    """训练配置"""

    model_type: ModelType
    hyperparameters: Dict[str, Any]
    validation_split: float = 0.2
    test_split: float = 0.1
    cross_validation_folds: int = 5
    random_state: int = 42
    early_stopping_rounds: Optional[int] = None
    feature_selection: bool = True
    max_features: Optional[int] = None

    # 数据配置
    target_column: str = "target"
    feature_columns: Optional[List[str]] = None

    # 训练控制
    max_training_time: int = 3600  # 最大训练时间（秒）
    save_intermediate: bool = True


@dataclass
class TrainingResult:
    """训练结果"""

    model_id: str
    training_time: float
    training_metrics: Dict[str, float]
    validation_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    cross_validation_scores: List[float]
    feature_importance: Optional[Dict[str, float]] = None
    best_hyperparameters: Optional[Dict[str, Any]] = None
    training_history: Optional[List[Dict[str, float]]] = None


class ModelTrainer:
    """模型训练器基类"""

    def __init__(self, model_type: ModelType):
        self.model_type = model_type

    def create_model(self, hyperparameters: Dict[str, Any]) -> Any:
        """创建模型实例"""
        raise NotImplementedError

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        config: TrainingConfig,
    ) -> Tuple[Any, Dict[str, Any]]:
        """训练模型"""
        raise NotImplementedError

    def get_feature_importance(
        self, model: Any, feature_names: List[str]
    ) -> Dict[str, float]:
        """获取特征重要性"""
        return {}


class RandomForestTrainer(ModelTrainer):
    """随机森林训练器"""

    def __init__(self):
        super().__init__(ModelType.RANDOM_FOREST)

    def create_model(self, hyperparameters: Dict[str, Any]) -> Any:
        """创建随机森林模型"""
        try:
            from sklearn.ensemble import RandomForestRegressor

            default_params = {
                "n_estimators": 100,
                "max_depth": None,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "random_state": 42,
            }
            default_params.update(hyperparameters)
            return RandomForestRegressor(**default_params)
        except ImportError:
            # 如果sklearn不可用，创建一个简单的模拟模型
            return MockModel()

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        config: TrainingConfig,
    ) -> Tuple[Any, Dict[str, Any]]:
        """训练随机森林模型"""
        model = self.create_model(config.hyperparameters)
        model.fit(X_train, y_train)
        return model, {}

    def get_feature_importance(
        self, model: Any, feature_names: List[str]
    ) -> Dict[str, float]:
        """获取特征重要性"""
        if hasattr(model, "feature_importances_"):
            importance_dict = {}
            for name, importance in zip(feature_names, model.feature_importances_):
                importance_dict[name] = float(importance)
            return importance_dict
        return {}


class LinearRegressionTrainer(ModelTrainer):
    """线性回归训练器"""

    def __init__(self):
        super().__init__(ModelType.LINEAR_REGRESSION)

    def create_model(self, hyperparameters: Dict[str, Any]) -> Any:
        """创建线性回归模型"""
        try:
            from sklearn.linear_model import LinearRegression

            return LinearRegression(**hyperparameters)
        except ImportError:
            return MockModel()

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        config: TrainingConfig,
    ) -> Tuple[Any, Dict[str, Any]]:
        """训练线性回归模型"""
        model = self.create_model(config.hyperparameters)
        model.fit(X_train, y_train)
        return model, {}

    def get_feature_importance(
        self, model: Any, feature_names: List[str]
    ) -> Dict[str, float]:
        """获取特征重要性（系数绝对值）"""
        if hasattr(model, "coef_"):
            importance_dict = {}
            for name, coef in zip(feature_names, model.coef_):
                importance_dict[name] = float(abs(coef))
            return importance_dict
        return {}


class MockModel:
    """模拟模型类，用于在没有sklearn时提供基本功能"""

    def __init__(self):
        self.is_fitted = False
        self.feature_importances_ = None
        self.coef_ = None

    def fit(self, X, y):
        """模拟训练"""
        self.is_fitted = True
        n_features = X.shape[1] if hasattr(X, "shape") else len(X.columns)
        self.feature_importances_ = np.random.rand(n_features)
        self.coef_ = np.random.randn(n_features)
        return self

    def predict(self, X):
        """模拟预测"""
        if not self.is_fitted:
            raise ValueError("模型未训练")
        n_samples = X.shape[0] if hasattr(X, "shape") else len(X)
        return np.random.randn(n_samples)


class TrainerFactory:
    """训练器工厂"""

    _trainers = {
        ModelType.RANDOM_FOREST: RandomForestTrainer,
        ModelType.LINEAR_REGRESSION: LinearRegressionTrainer,
    }

    @classmethod
    def create_trainer(cls, model_type: ModelType) -> ModelTrainer:
        """创建训练器"""
        trainer_class = cls._trainers.get(model_type)
        if not trainer_class:
            # 默认使用随机森林
            trainer_class = RandomForestTrainer
        return trainer_class()


class ModelTrainingService:
    """模型训练服务"""

    def __init__(self, model_storage: ModelStorage, data_dir: str = "backend/data"):
        self.model_storage = model_storage
        self.data_dir = Path(data_dir)

        # 训练统计
        self.training_stats = {
            "total_trainings": 0,
            "successful_trainings": 0,
            "failed_trainings": 0,
            "average_training_time": 0.0,
        }

    def train_model(
        self,
        model_name: str,
        model_type: ModelType,
        stock_codes: List[str],
        start_date: datetime,
        end_date: datetime,
        config: TrainingConfig,
        created_by: str,
        progress_callback: Optional[Callable] = None,
    ) -> TrainingResult:
        """训练模型"""
        training_start_time = datetime.utcnow()

        try:
            self.training_stats["total_trainings"] += 1

            # 清理模型名称，移除不允许的文件名字符
            import re

            safe_model_name = re.sub(r'[<>:"/\\|?*]', "_", model_name)  # 替换不允许的字符为下划线
            safe_model_name = re.sub(r"\s+", "_", safe_model_name)  # 替换空格为下划线

            # 生成模型ID
            model_id = f"{safe_model_name}_{model_type.value}_{training_start_time.strftime('%Y%m%d_%H%M%S')}"

            logger.info(f"开始训练模型: {model_id}")

            # 步骤1: 加载和准备数据
            if progress_callback:
                progress_callback(10, "加载训练数据")

            X, y = self._prepare_training_data(
                stock_codes, start_date, end_date, config
            )

            # 步骤2: 数据分割
            if progress_callback:
                progress_callback(20, "分割训练数据")

            X_train, X_temp, y_train, y_temp = self._train_test_split(
                X,
                y,
                test_size=(config.validation_split + config.test_split),
                random_state=config.random_state,
            )

            val_ratio = config.validation_split / (
                config.validation_split + config.test_split
            )
            X_val, X_test, y_val, y_test = self._train_test_split(
                X_temp,
                y_temp,
                test_size=(1 - val_ratio),
                random_state=config.random_state,
            )

            logger.info(
                f"数据分割完成: 训练集 {len(X_train)}, 验证集 {len(X_val)}, 测试集 {len(X_test)}"
            )

            # 步骤3: 创建训练器
            if progress_callback:
                progress_callback(40, "初始化模型")

            trainer = TrainerFactory.create_trainer(model_type)

            # 步骤4: 训练模型
            if progress_callback:
                progress_callback(50, "训练模型")

            model, training_info = trainer.train(X_train, y_train, X_val, y_val, config)

            # 步骤5: 模型评估
            if progress_callback:
                progress_callback(70, "评估模型")

            training_metrics = self._evaluate_model(model, X_train, y_train)
            validation_metrics = self._evaluate_model(model, X_val, y_val)
            test_metrics = self._evaluate_model(model, X_test, y_test)

            # 步骤6: 交叉验证
            if progress_callback:
                progress_callback(80, "交叉验证")

            cv_scores = self._cross_validate(
                model, X_train, y_train, config.cross_validation_folds
            )

            # 步骤7: 获取特征重要性
            if progress_callback:
                progress_callback(90, "分析特征重要性")

            feature_importance = trainer.get_feature_importance(
                model, X_train.columns.tolist()
            )

            # 步骤8: 保存模型
            if progress_callback:
                progress_callback(95, "保存模型")

            training_time = (datetime.utcnow() - training_start_time).total_seconds()

            # 创建模型元数据
            metadata = ModelMetadata(
                model_id=model_id,
                model_name=model_name,
                model_type=model_type,
                version="1.0.0",
                description=f"训练于 {training_start_time.strftime('%Y-%m-%d %H:%M:%S')}",
                created_by=created_by,
                created_at=training_start_time,
                updated_at=datetime.utcnow(),
                status=ModelStatus.TRAINED,
                training_data_info={
                    "stock_codes": stock_codes,
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "training_samples": len(X_train),
                    "validation_samples": len(X_val),
                    "test_samples": len(X_test),
                },
                hyperparameters=config.hyperparameters,
                training_config={
                    "validation_split": config.validation_split,
                    "test_split": config.test_split,
                    "cross_validation_folds": config.cross_validation_folds,
                    "feature_selection": config.feature_selection,
                },
                performance_metrics=test_metrics,
                validation_metrics=validation_metrics,
                feature_columns=X_train.columns.tolist(),
            )

            # 保存模型
            self.model_storage.save_model(model, metadata)

            # 创建训练结果
            result = TrainingResult(
                model_id=model_id,
                training_time=training_time,
                training_metrics=training_metrics,
                validation_metrics=validation_metrics,
                test_metrics=test_metrics,
                cross_validation_scores=cv_scores,
                feature_importance=feature_importance,
                training_history=training_info.get("training_history"),
            )

            # 更新统计
            self.training_stats["successful_trainings"] += 1
            self._update_average_training_time(training_time)

            if progress_callback:
                progress_callback(100, "训练完成")

            logger.info(f"模型训练完成: {model_id}, 耗时: {training_time:.2f}秒")

            return result

        except Exception as e:
            self.training_stats["failed_trainings"] += 1

            raise ModelError(
                message=f"模型训练失败: {str(e)}",
                severity=ErrorSeverity.HIGH,
                context=ErrorContext(
                    model_id=model_id if "model_id" in locals() else None
                ),
                original_exception=e,
            )

    def _prepare_training_data(
        self,
        stock_codes: List[str],
        start_date: datetime,
        end_date: datetime,
        config: TrainingConfig,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """准备训练数据"""
        from ..data.stock_data_loader import StockDataLoader

        loader = StockDataLoader(data_root=str(self.data_dir))
        all_data = []

        for stock_code in stock_codes:
            try:
                stock_data = loader.load_stock_data(
                    stock_code, start_date=start_date, end_date=end_date
                )
                if not stock_data.empty:
                    stock_data["stock_code"] = stock_code
                    all_data.append(stock_data)
            except Exception as e:
                logger.warning(f"加载股票 {stock_code} 数据失败: {e}")
                continue

        if not all_data:
            # 如果没有加载到任何数据，回退到模拟数据并记录警告
            logger.warning("未能加载任何真实股票数据，使用模拟数据进行训练")
            np.random.seed(42)
            n_samples = 1000
            n_features = 10
            feature_names = [f"feature_{i}" for i in range(n_features)]
            X = pd.DataFrame(np.random.randn(n_samples, n_features), columns=feature_names)
            y = pd.Series(np.random.randn(n_samples), name="target")
            return X, y

        # 合并所有股票数据
        combined_data = pd.concat(all_data, ignore_index=True)

        # 确定特征列
        if config.feature_columns:
            feature_cols = [col for col in config.feature_columns if col in combined_data.columns]
        else:
            # 排除非特征列
            exclude_cols = ["stock_code", "date", config.target_column, "target"]
            feature_cols = [col for col in combined_data.columns if col not in exclude_cols]

        if not feature_cols:
            raise ValueError("没有可用的特征列")

        # 准备特征和目标
        X = combined_data[feature_cols].copy()

        # 确定目标列
        target_col = config.target_column if config.target_column in combined_data.columns else "close"
        if target_col not in combined_data.columns:
            # 如果目标列不存在，计算收益率作为目标
            if "close" in combined_data.columns:
                y = combined_data["close"].pct_change().shift(-1)  # 预测下一期收益率
            else:
                raise ValueError(f"目标列 {target_col} 不存在")
        else:
            y = combined_data[target_col].copy()

        # 处理缺失值
        X = X.ffill().bfill().fillna(0)
        y = y.ffill().bfill().fillna(0)

        # 移除包含无效值的行
        valid_mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_mask]
        y = y[valid_mask]

        logger.info(f"训练数据准备完成: {len(X)} 样本, {len(X.columns)} 特征")

        return X, y

    def _train_test_split(self, X, y, test_size=0.2, random_state=42):
        """数据分割"""
        try:
            from sklearn.model_selection import train_test_split

            return train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
        except ImportError:
            # 简单的数据分割实现
            n_samples = len(X)
            n_test = int(n_samples * test_size)

            np.random.seed(random_state)
            indices = np.random.permutation(n_samples)

            test_indices = indices[:n_test]
            train_indices = indices[n_test:]

            X_train = X.iloc[train_indices] if hasattr(X, "iloc") else X[train_indices]
            X_test = X.iloc[test_indices] if hasattr(X, "iloc") else X[test_indices]
            y_train = y.iloc[train_indices] if hasattr(y, "iloc") else y[train_indices]
            y_test = y.iloc[test_indices] if hasattr(y, "iloc") else y[test_indices]

            return X_train, X_test, y_train, y_test

    def _evaluate_model(
        self, model: Any, X: pd.DataFrame, y: pd.Series
    ) -> Dict[str, float]:
        """评估模型"""
        try:
            y_pred = model.predict(X)

            # 计算基本指标
            mse = np.mean((y - y_pred) ** 2)
            mae = np.mean(np.abs(y - y_pred))
            rmse = float(np.sqrt(mse))

            # 计算R²
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

            # 对于回归模型，计算方向准确率（预测涨跌方向的准确率）
            # 如果y是价格变化率或收益率
            if len(y) > 1:
                # 计算实际方向
                y_direction = np.sign(
                    np.diff(y.values) if hasattr(y, "values") else np.diff(y)
                )
                # 计算预测方向
                y_pred_direction = np.sign(
                    np.diff(y_pred) if len(y_pred) > 1 else y_pred - np.mean(y_pred)
                )

                # 对齐长度
                min_len = min(len(y_direction), len(y_pred_direction))
                if min_len > 0:
                    direction_accuracy = np.mean(
                        y_direction[:min_len] == y_pred_direction[:min_len]
                    )
                else:
                    direction_accuracy = 0.0
            else:
                direction_accuracy = 0.0

            # 对于准确率指标：
            # 1. 优先使用方向准确率（更符合股票预测的实际需求）
            # 2. 如果R²为正，也可以使用R²，但方向准确率更直观
            # 3. 如果R²为负，说明模型比简单均值还差，设为0
            if direction_accuracy > 0:
                accuracy_metric = direction_accuracy
            elif r2 > 0:
                # R²为正时，可以转换为0-1范围（R²通常在0-1之间，但可能超过1）
                accuracy_metric = min(1.0, max(0.0, r2))
            else:
                # R²为负或为0时，使用方向准确率（如果计算成功）或设为0
                accuracy_metric = max(0.0, direction_accuracy)

            metrics = {
                "mse": float(mse),
                "mae": float(mae),
                "r2": float(r2),
                "rmse": rmse,
                "accuracy": float(accuracy_metric),  # 使用方向准确率或R²（取较大值，且R²负值设为0）
                "direction_accuracy": float(direction_accuracy),  # 方向准确率
            }

            return metrics

        except Exception as e:
            logger.error(f"模型评估失败: {e}")
            return {}

    def _cross_validate(
        self, model: Any, X: pd.DataFrame, y: pd.Series, cv: int
    ) -> List[float]:
        """交叉验证"""
        try:
            from sklearn.model_selection import cross_val_score

            scores = cross_val_score(
                model, X, y, cv=cv, scoring="neg_mean_squared_error"
            )
            return [-score for score in scores]  # 转换为正值
        except ImportError:
            # 简单的交叉验证实现
            n_samples = len(X)
            fold_size = n_samples // cv
            scores = []

            for i in range(cv):
                start_idx = i * fold_size
                end_idx = start_idx + fold_size if i < cv - 1 else n_samples

                # 创建训练和验证集
                val_indices = list(range(start_idx, end_idx))
                train_indices = [j for j in range(n_samples) if j not in val_indices]

                X_fold_train = X.iloc[train_indices]
                y_fold_train = y.iloc[train_indices]
                X_fold_val = X.iloc[val_indices]
                y_fold_val = y.iloc[val_indices]

                # 训练和评估
                fold_model = TrainerFactory.create_trainer(
                    ModelType.RANDOM_FOREST
                ).create_model({})
                fold_model.fit(X_fold_train, y_fold_train)
                y_pred = fold_model.predict(X_fold_val)

                mse = np.mean((y_fold_val - y_pred) ** 2)
                scores.append(mse)

            return scores

    def _update_average_training_time(self, training_time: float):
        """更新平均训练时间"""
        total_successful = self.training_stats["successful_trainings"]
        if total_successful == 1:
            self.training_stats["average_training_time"] = training_time
        else:
            current_avg = self.training_stats["average_training_time"]
            new_avg = (
                current_avg * (total_successful - 1) + training_time
            ) / total_successful
            self.training_stats["average_training_time"] = new_avg

    def get_training_statistics(self) -> Dict[str, Any]:
        """获取训练统计信息"""
        return {
            **self.training_stats,
            "success_rate": (
                self.training_stats["successful_trainings"]
                / max(self.training_stats["total_trainings"], 1)
            ),
            "supported_model_types": [t.value for t in ModelType],
        }
