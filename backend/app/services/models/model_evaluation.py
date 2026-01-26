"""
模型评估和版本管理系统

实现时间序列交叉验证、性能评估指标计算、模型版本管理等功能。
专门针对金融时间序列预测任务进行优化。
"""

import hashlib
import json
import pickle
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit

from app.core.database import SessionLocal
from app.core.logging import logger as app_logger

logger = app_logger

# 导入统一的错误处理机制
try:
    from app.core.error_handler import (
        DataError,
        ErrorContext,
        ErrorSeverity,
        ModelError,
        TaskError,
        handle_async_exception,
    )
except ImportError:
    logger.warning("错误处理模块未找到，使用默认错误处理")
    ModelError = Exception
    DataError = Exception
    TaskError = Exception
    ErrorSeverity = None
    ErrorContext = None
    handle_async_exception = lambda func: func


# 从shared_types.py导入共享类型
try:
    from .shared_types import BacktestMetrics, ModelStatus, ModelVersion

    SHARED_TYPES_AVAILABLE = True
except ImportError:
    SHARED_TYPES_AVAILABLE = False

    # 如果导入失败，使用本地定义作为备选
    class ModelStatus(Enum):
        """模型状态"""

        TRAINING = "training"
        COMPLETED = "completed"
        FAILED = "failed"
        DEPLOYED = "deployed"
        ARCHIVED = "archived"

    @dataclass
    class BacktestMetrics:
        """回测评估指标"""

        # 基础分类指标
        accuracy: float
        precision: float
        recall: float
        f1_score: float

        # 金融指标
        total_return: float
        sharpe_ratio: float
        max_drawdown: float
        win_rate: float
        profit_factor: float

        # 风险指标
        volatility: float
        var_95: float  # 95% VaR
        calmar_ratio: float

        # 交易指标
        total_trades: int
        avg_trade_return: float
        max_consecutive_losses: int

        def to_dict(self) -> Dict[str, float]:
            return asdict(self)

    @dataclass
    class ModelVersion:
        """模型版本信息"""

        model_id: str
        version: str
        model_type: str
        parameters: Dict[str, Any]
        metrics: BacktestMetrics
        file_path: str
        created_at: datetime
        status: ModelStatus
        training_data_hash: str

        def to_dict(self) -> Dict[str, Any]:
            result = asdict(self)
            result["created_at"] = self.created_at.isoformat()
            result["status"] = self.status.value
            result["metrics"] = self.metrics.to_dict()

            # 递归转换parameters中的枚举类型
            def convert_enums(obj):
                if isinstance(obj, dict):
                    return {k: convert_enums(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_enums(item) for item in obj]
                elif hasattr(obj, "value"):  # 枚举类型
                    return obj.value
                else:
                    return obj

            if "parameters" in result:
                result["parameters"] = convert_enums(result["parameters"])

            return result


class TimeSeriesValidator:
    """时间序列交叉验证器"""

    def __init__(self, n_splits: int = 5, test_size: Optional[int] = None):
        self.n_splits = n_splits
        self.test_size = test_size

    def split(
        self, X: np.ndarray, y: np.ndarray
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        时间序列交叉验证分割

        Args:
            X: 特征数据 [samples, seq_len, features]
            y: 标签数据 [samples]

        Returns:
            List of (train_indices, test_indices)
        """
        n_samples = len(X)

        if self.test_size is None:
            test_size = n_samples // (self.n_splits + 1)
        else:
            test_size = self.test_size

        splits = []

        for i in range(self.n_splits):
            # 计算测试集的结束位置
            test_end = n_samples - i * (test_size // 2)
            test_start = test_end - test_size

            # 确保有足够的训练数据
            train_end = test_start
            train_start = max(0, train_end - test_size * 3)  # 训练集是测试集的3倍

            if train_start >= train_end or test_start >= test_end:
                continue

            train_indices = np.arange(train_start, train_end)
            test_indices = np.arange(test_start, test_end)

            splits.append((train_indices, test_indices))

        return splits


class FinancialMetricsCalculator:
    """金融指标计算器"""

    @staticmethod
    def calculate_returns(
        predictions: np.ndarray, actual_prices: np.ndarray
    ) -> np.ndarray:
        """
        根据预测结果计算收益率

        Args:
            predictions: 预测标签 [0, 1]，1表示上涨
            actual_prices: 实际价格序列

        Returns:
            每日收益率
        """
        # 计算实际收益率
        actual_returns = np.diff(actual_prices) / actual_prices[:-1]

        # 根据预测生成交易信号
        # 预测上涨(1)则买入，预测下跌(0)则卖出或持现金
        trading_returns = []

        for i, pred in enumerate(predictions):
            if i < len(actual_returns):
                if pred == 1:  # 预测上涨，买入
                    trading_returns.append(actual_returns[i])
                else:  # 预测下跌，持现金
                    trading_returns.append(0.0)

        return np.array(trading_returns)

    @staticmethod
    def calculate_sharpe_ratio(
        returns: np.ndarray, risk_free_rate: float = 0.02
    ) -> float:
        """计算夏普比率"""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0

        excess_returns = returns - risk_free_rate / 252  # 日化无风险利率
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)

    @staticmethod
    def calculate_max_drawdown(returns: np.ndarray) -> float:
        """计算最大回撤"""
        if len(returns) == 0:
            return 0.0

        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        return np.min(drawdown)

    @staticmethod
    def calculate_var(returns: np.ndarray, confidence_level: float = 0.95) -> float:
        """计算风险价值(VaR)"""
        if len(returns) == 0:
            return 0.0

        return np.percentile(returns, (1 - confidence_level) * 100)

    @staticmethod
    def calculate_calmar_ratio(returns: np.ndarray) -> float:
        """计算卡尔玛比率"""
        if len(returns) == 0:
            return 0.0

        annual_return = np.mean(returns) * 252
        max_dd = abs(FinancialMetricsCalculator.calculate_max_drawdown(returns))

        if max_dd == 0:
            return float("inf") if annual_return > 0 else 0.0

        return annual_return / max_dd


class ModelEvaluator:
    """模型评估器"""

    def __init__(self):
        self.validator = TimeSeriesValidator(n_splits=5)
        self.metrics_calculator = FinancialMetricsCalculator()

    @handle_async_exception
    async def evaluate_model(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        actual_prices: np.ndarray,
        model_type: str = "unknown",
    ) -> BacktestMetrics:
        """
        全面评估模型性能

        Args:
            model: 训练好的模型
            X: 特征数据
            y: 标签数据
            actual_prices: 实际价格数据
            model_type: 模型类型

        Returns:
            评估指标
        """
        logger.info(f"开始评估模型，数据量: {len(X)}")

        # 时间序列交叉验证
        splits = self.validator.split(X, y)

        all_predictions = []
        all_true_labels = []
        all_returns = []

        for fold, (train_idx, test_idx) in enumerate(splits):
            logger.info(f"评估第 {fold + 1}/{len(splits)} 折")

            X_test = X[test_idx]
            y_test = y[test_idx]
            prices_test = actual_prices[test_idx]

            # 模型预测
            if hasattr(model, "predict"):
                # sklearn模型
                if len(X_test.shape) == 3:
                    X_test_flat = X_test.reshape(X_test.shape[0], -1)
                else:
                    X_test_flat = X_test
                predictions = model.predict(X_test_flat)
            else:
                # PyTorch模型
                model.eval()
                with torch.no_grad():
                    # 获取模型所在的设备
                    device = next(model.parameters()).device

                    if isinstance(X_test, np.ndarray):
                        X_test_tensor = torch.FloatTensor(X_test).to(device)
                    else:
                        X_test_tensor = (
                            X_test.to(device) if hasattr(X_test, "to") else X_test
                        )

                    outputs = model(X_test_tensor)
                    predictions = torch.argmax(outputs, dim=1).cpu().numpy()

            # 计算收益率
            fold_returns = self.metrics_calculator.calculate_returns(
                predictions, prices_test
            )

            all_predictions.extend(predictions)
            all_true_labels.extend(y_test)
            all_returns.extend(fold_returns)

        # 转换为numpy数组
        all_predictions = np.array(all_predictions)
        all_true_labels = np.array(all_true_labels)
        all_returns = np.array(all_returns)

        # 计算分类指标
        accuracy = accuracy_score(all_true_labels, all_predictions)
        precision = precision_score(all_true_labels, all_predictions, zero_division=0)
        recall = recall_score(all_true_labels, all_predictions, zero_division=0)
        f1 = f1_score(all_true_labels, all_predictions, zero_division=0)

        # 计算金融指标
        total_return = np.prod(1 + all_returns) - 1
        sharpe_ratio = self.metrics_calculator.calculate_sharpe_ratio(all_returns)
        max_drawdown = self.metrics_calculator.calculate_max_drawdown(all_returns)
        win_rate = (
            np.sum(all_returns > 0) / len(all_returns) if len(all_returns) > 0 else 0
        )

        # 计算盈亏比
        winning_trades = all_returns[all_returns > 0]
        losing_trades = all_returns[all_returns < 0]

        if len(losing_trades) > 0 and np.mean(losing_trades) != 0:
            profit_factor = abs(np.sum(winning_trades) / np.sum(losing_trades))
        else:
            profit_factor = float("inf") if len(winning_trades) > 0 else 0.0

        # 计算风险指标
        volatility = np.std(all_returns) * np.sqrt(252)
        var_95 = self.metrics_calculator.calculate_var(all_returns)
        calmar_ratio = self.metrics_calculator.calculate_calmar_ratio(all_returns)

        # 计算交易指标
        total_trades = len(all_returns)
        avg_trade_return = np.mean(all_returns) if len(all_returns) > 0 else 0

        # 计算最大连续亏损
        consecutive_losses = 0
        max_consecutive_losses = 0
        for ret in all_returns:
            if ret < 0:
                consecutive_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            else:
                consecutive_losses = 0

        metrics = BacktestMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            volatility=volatility,
            var_95=var_95,
            calmar_ratio=calmar_ratio,
            total_trades=total_trades,
            avg_trade_return=avg_trade_return,
            max_consecutive_losses=max_consecutive_losses,
        )

        logger.info(f"模型评估完成，准确率: {accuracy:.4f}, 夏普比率: {sharpe_ratio:.4f}")
        return metrics


class ModelVersionManager:
    """模型版本管理器"""

    def __init__(self, models_dir: str = None, storage: "ModelStorage" = None):
        # 使用配置中的路径，如果没有提供则使用默认配置
        from app.core.config import settings

        if models_dir is None:
            models_dir = settings.MODEL_STORAGE_PATH

        self.models_dir = Path(models_dir)
        # 解析相对路径为绝对路径
        if not self.models_dir.is_absolute():
            backend_dir = Path(__file__).parent.parent.parent
            self.models_dir = (backend_dir / self.models_dir).resolve()

        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.versions_file = self.models_dir / "versions.json"
        self.storage = storage

        # 创建版本目录（用于存储每个模型的版本信息）
        self.versions_dir = self.models_dir / "versions"
        self.versions_dir.mkdir(parents=True, exist_ok=True)

    def _load_versions(self) -> Dict[str, List[Dict]]:
        """加载版本信息"""
        if self.versions_file.exists():
            try:
                with open(self.versions_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"版本文件格式错误，将备份并重新创建: {e}")
                # 备份损坏的文件
                backup_file = self.versions_file.with_suffix(".json.bak")
                try:
                    import shutil

                    shutil.copy2(self.versions_file, backup_file)
                    logger.info(f"已备份损坏的版本文件到: {backup_file}")
                except Exception as backup_error:
                    logger.error(f"备份文件失败: {backup_error}")
                # 返回空字典，让系统重新创建
                return {}
        return {}

    def _save_versions(self, versions: Dict[str, List[Dict]]):
        """保存版本信息"""

        # 递归转换不可序列化的对象（如枚举）
        def convert_to_serializable(obj):
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif hasattr(obj, "value"):  # 枚举类型
                return obj.value
            elif hasattr(obj, "__dict__"):  # 其他对象
                return str(obj)
            else:
                return obj

        serializable_versions = convert_to_serializable(versions)
        with open(self.versions_file, "w", encoding="utf-8") as f:
            json.dump(serializable_versions, f, indent=2, ensure_ascii=False)

    def _generate_data_hash(self, X: np.ndarray, y: np.ndarray) -> str:
        """生成训练数据的哈希值"""
        data_str = f"{X.shape}_{np.sum(X)}_{np.sum(y)}"
        return hashlib.md5(data_str.encode()).hexdigest()[:16]

    def save_model_version(
        self,
        model_id: str,
        model: Any,
        model_type: str,
        parameters: Dict[str, Any],
        metrics: BacktestMetrics,
        training_data: Tuple[np.ndarray, np.ndarray],
    ) -> ModelVersion:
        """
        保存模型版本

        Args:
            model_id: 模型唯一标识
            model: 训练好的模型
            model_type: 模型类型
            parameters: 训练参数
            metrics: 评估指标
            training_data: 训练数据(X, y)

        Returns:
            模型版本信息
        """
        X, y = training_data
        data_hash = self._generate_data_hash(X, y)

        # 生成版本号
        versions = self._load_versions()
        model_versions = versions.get(model_id, [])
        version = f"v{len(model_versions) + 1:03d}"

        # 保存模型文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{model_id}_{version}_{timestamp}"

        if hasattr(model, "save_model"):
            # XGBoost模型
            model_path = self.models_dir / f"{model_filename}.json"
            model.save_model(str(model_path))
        elif hasattr(model, "state_dict"):
            # PyTorch模型
            model_path = self.models_dir / f"{model_filename}.pth"
            torch.save(model.state_dict(), model_path)
        else:
            # 其他模型使用pickle
            model_path = self.models_dir / f"{model_filename}.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(model, f)

        # 创建版本信息
        model_version = ModelVersion(
            model_id=model_id,
            version=version,
            model_type=model_type,
            parameters=parameters,
            metrics=metrics,
            file_path=str(model_path),
            created_at=datetime.now(),
            status=ModelStatus.COMPLETED,
            training_data_hash=data_hash,
        )

        # 保存版本信息
        model_versions.append(model_version.to_dict())
        versions[model_id] = model_versions
        self._save_versions(versions)

        logger.info(f"模型版本已保存: {model_id} {version}")
        return model_version

    def get_model_versions(self, model_id: str) -> List[ModelVersion]:
        """获取模型的所有版本"""
        versions = self._load_versions()
        model_versions = versions.get(model_id, [])

        result = []
        for version_dict in model_versions:
            # 重构ModelVersion对象
            version_dict["created_at"] = datetime.fromisoformat(
                version_dict["created_at"]
            )
            version_dict["status"] = ModelStatus(version_dict["status"])

            # 重构BacktestMetrics对象
            metrics_dict = version_dict["metrics"]
            version_dict["metrics"] = BacktestMetrics(**metrics_dict)

            result.append(ModelVersion(**version_dict))

        return result

    def get_best_model(
        self, model_id: str, metric: str = "sharpe_ratio"
    ) -> Optional[ModelVersion]:
        """
        获取最佳模型版本

        Args:
            model_id: 模型ID
            metric: 评估指标名称

        Returns:
            最佳模型版本
        """
        versions = self.get_model_versions(model_id)

        if not versions:
            return None

        # 根据指标排序
        if metric in ["sharpe_ratio", "total_return", "accuracy", "calmar_ratio"]:
            # 越大越好的指标
            best_version = max(versions, key=lambda v: getattr(v.metrics, metric))
        elif metric in ["max_drawdown", "volatility", "var_95"]:
            # 越小越好的指标
            best_version = min(versions, key=lambda v: abs(getattr(v.metrics, metric)))
        else:
            # 默认使用夏普比率
            best_version = max(versions, key=lambda v: v.metrics.sharpe_ratio)

        return best_version

    def load_model(self, model_version: ModelVersion) -> Any:
        """加载模型"""
        model_path = Path(model_version.file_path)

        if not model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        if model_path.suffix == ".json":
            # XGBoost模型
            import xgboost as xgb

            model = xgb.Booster()
            model.load_model(str(model_path))
            return model
        elif model_path.suffix == ".pth":
            # PyTorch模型 - 需要模型结构信息
            state_dict = torch.load(model_path, map_location="cpu")
            # 这里需要根据model_type重新创建模型结构
            # 实际使用时需要完善这部分逻辑
            logger.warning("PyTorch模型加载需要模型结构信息")
            return state_dict
        else:
            # Pickle模型
            with open(model_path, "rb") as f:
                return pickle.load(f)

    def create_version(
        self,
        model_id: str,
        version: str,
        description: str,
        created_by: str,
        performance_metrics: Dict[str, float],
    ) -> bool:
        """创建新版本"""
        try:
            # 检查模型是否存在
            versions = self._load_versions()
            model_versions = versions.get(model_id, [])

            # 检查版本是否已存在
            for v in model_versions:
                if v.get("version") == version:
                    logger.warning(f"版本已存在: {model_id} v{version}")
                    return False

            # 创建版本目录
            version_dir = self.versions_dir / model_id
            version_dir.mkdir(exist_ok=True)

            # 创建版本信息
            version_info = {
                "version": version,
                "created_at": datetime.utcnow().isoformat(),
                "created_by": created_by,
                "description": description,
                "performance_metrics": performance_metrics,
                "is_active": False,
            }

            # 保存版本信息到文件
            version_file = version_dir / f"{version}.json"
            with open(version_file, "w", encoding="utf-8") as f:
                import json

                json.dump(version_info, f, ensure_ascii=False, indent=2)

            logger.info(f"模型版本创建成功: {model_id} v{version}")
            return True

        except Exception as e:
            logger.error(f"创建模型版本失败: {str(e)}")
            return False

    def list_versions(self, model_id: str) -> List[Dict[str, Any]]:
        """列出模型的所有版本"""
        try:
            version_dir = self.versions_dir / model_id
            if not version_dir.exists():
                return []

            versions = []
            for version_file in version_dir.glob("*.json"):
                try:
                    with open(version_file, "r", encoding="utf-8") as f:
                        import json

                        version_dict = json.load(f)
                    versions.append(version_dict)
                except Exception as e:
                    logger.warning(f"读取版本信息失败: {version_file}, 错误: {e}")
                    continue

            # 按创建时间排序
            versions.sort(key=lambda x: x["created_at"], reverse=True)
            return versions
        except Exception as e:
            logger.error(f"列出模型版本失败: {model_id}, 错误: {e}")
            return []


# 导出主要类
__all__ = [
    "ModelEvaluator",
    "ModelVersionManager",
    "TimeSeriesValidator",
    "FinancialMetricsCalculator",
    "BacktestMetrics",
    "ModelVersion",
    "ModelStatus",
]
