"""
预测计算引擎 - 核心预测功能实现
"""

import json
import pickle
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from loguru import logger

from app.core.database import SessionLocal
from app.core.error_handler import (
    ErrorContext,
    ErrorSeverity,
    PredictionError,
    RecoveryAction,
)
from app.models.task_models import PredictionResult, RiskMetrics
from app.repositories.task_repository import ModelInfoRepository

from .feature_extractor import FeatureConfig, FeatureExtractor


@dataclass
class PredictionConfig:
    """预测配置"""

    model_id: str
    horizon: str = "short_term"  # short_term, medium_term, long_term
    confidence_level: float = 0.95
    features: Optional[List[str]] = None
    use_ensemble: bool = True
    risk_assessment: bool = True


@dataclass
class PredictionOutput:
    """预测输出"""

    stock_code: str
    prediction_date: datetime
    predicted_price: float
    predicted_direction: int  # 1: 上涨, -1: 下跌, 0: 持平
    confidence_score: float
    confidence_interval: Tuple[float, float]
    risk_metrics: RiskMetrics
    model_id: str
    features_used: List[str]
    model_confidence: float
    prediction_horizon: str


class ModelLoader:
    """模型加载器"""

    def __init__(self, model_dir: str = "backend/models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.loaded_models: Dict[str, Any] = {}
        self.model_metadata: Dict[str, Dict[str, Any]] = {}

    def _load_model_file(self, model_path: Path) -> Tuple[Any, Dict[str, Any]]:
        if model_path.suffix == ".pkl":
            with open(model_path, "rb") as f:
                model_data = pickle.load(f)
            if isinstance(model_data, dict) and "model" in model_data:
                return model_data["model"], {
                    "model_format": "qlib_pickle",
                    "model_path": str(model_path),
                    "qlib_config": model_data.get("config", {}),
                    "timestamp": model_data.get("timestamp"),
                }
            return model_data, {
                "model_format": "pickle",
                "model_path": str(model_path),
            }
        model = joblib.load(model_path)
        return model, {
            "model_format": "joblib",
            "model_path": str(model_path),
        }

    def load_model(self, model_id: str) -> Tuple[Any, Dict[str, Any]]:
        """加载模型"""
        if model_id in self.loaded_models:
            return self.loaded_models[model_id], self.model_metadata[model_id]

        try:
            model_path = self.model_dir / f"{model_id}.joblib"

            metadata = {}
            if model_path.exists():
                model, metadata = self._load_model_file(model_path)
            else:
                session = SessionLocal()
                try:
                    model_info_repo = ModelInfoRepository(session)
                    model_info = model_info_repo.get_model_info(model_id)
                finally:
                    session.close()
                if not model_info:
                    raise PredictionError(
                        message=f"模型信息不存在: {model_id}",
                        severity=ErrorSeverity.HIGH,
                        context=ErrorContext(model_id=model_id),
                    )

                model_info_path = Path(model_info.file_path)
                if not model_info_path.is_absolute():
                    model_info_path = Path.cwd() / model_info_path
                if not model_info_path.exists():
                    raise PredictionError(
                        message=f"模型文件不存在: {model_info_path}",
                        severity=ErrorSeverity.HIGH,
                        context=ErrorContext(model_id=model_id),
                    )

                model, metadata = self._load_model_file(model_info_path)
                metadata.update(
                    {
                        "model_id": model_id,
                        "model_type": model_info.model_type,
                        "performance_metrics": model_info.performance_metrics or {},
                    }
                )

            # 缓存模型
            self.loaded_models[model_id] = model
            self.model_metadata[model_id] = metadata

            logger.info(f"模型加载成功: {model_id}")
            return model, metadata

        except Exception as e:
            logger.error(f"模型加载失败: {model_id}, 错误: {e}")
            raise PredictionError(
                message=f"模型加载失败: {str(e)}",
                severity=ErrorSeverity.HIGH,
                context=ErrorContext(model_id=model_id),
                original_exception=e,
            )

    def _create_fallback_model(self, model_id: str) -> Tuple[Any, Dict[str, Any]]:
        """创建备用模型"""
        from sklearn.linear_model import LinearRegression

        # 创建简单的线性回归模型
        model = LinearRegression()

        # 使用虚拟数据训练模型
        X_dummy = np.random.randn(100, 10)
        y_dummy = np.random.randn(100)
        model.fit(X_dummy, y_dummy)

        metadata = {
            "model_id": f"{model_id}_fallback",
            "model_type": "linear_regression",
            "is_fallback": True,
            "created_at": datetime.utcnow().isoformat(),
            "performance_metrics": {"accuracy": 0.5, "mse": 1.0, "mae": 0.8},
        }

        # 缓存备用模型
        self.loaded_models[model_id] = model
        self.model_metadata[model_id] = metadata

        logger.info(f"创建备用模型: {model_id}")
        return model, metadata

    def save_model(self, model_id: str, model: Any, metadata: Dict[str, Any]):
        """保存模型"""
        try:
            model_path = self.model_dir / f"{model_id}.joblib"
            metadata_path = self.model_dir / f"{model_id}_metadata.json"

            # 保存模型
            joblib.dump(model, model_path)

            # 保存元数据
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

            logger.info(f"模型保存成功: {model_id}")

        except Exception as e:
            logger.error(f"模型保存失败: {model_id}, 错误: {e}")
            raise PredictionError(
                message=f"模型保存失败: {str(e)}",
                severity=ErrorSeverity.HIGH,
                context=ErrorContext(model_id=model_id),
            )


class RiskAssessment:
    """风险评估器"""

    @staticmethod
    def calculate_var(returns: pd.Series, confidence_level: float = 0.95) -> float:
        """计算风险价值(VaR)"""
        if len(returns) == 0:
            return 0.0

        return np.percentile(returns.dropna(), (1 - confidence_level) * 100)

    @staticmethod
    def calculate_expected_shortfall(
        returns: pd.Series, confidence_level: float = 0.95
    ) -> float:
        """计算期望损失(ES)"""
        if len(returns) == 0:
            return 0.0

        var = RiskAssessment.calculate_var(returns, confidence_level)
        return returns[returns <= var].mean()

    @staticmethod
    def calculate_volatility(returns: pd.Series, annualize: bool = True) -> float:
        """计算波动率"""
        if len(returns) == 0:
            return 0.0

        vol = returns.std()
        if annualize:
            vol *= np.sqrt(252)  # 年化
        return vol

    @staticmethod
    def calculate_max_drawdown(prices: pd.Series) -> float:
        """计算最大回撤"""
        if len(prices) == 0:
            return 0.0

        cumulative = (1 + prices.pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()

    @staticmethod
    def calculate_sharpe_ratio(
        returns: pd.Series, risk_free_rate: float = 0.02
    ) -> float:
        """计算夏普比率"""
        if len(returns) == 0 or returns.std() == 0:
            return 0.0

        excess_returns = returns.mean() - risk_free_rate / 252
        return excess_returns / returns.std() * np.sqrt(252)

    @staticmethod
    def assess_prediction_risk(
        historical_prices: pd.Series,
        predicted_price: float,
        confidence_level: float = 0.95,
    ) -> RiskMetrics:
        """评估预测风险"""
        returns = historical_prices.pct_change().dropna()

        # 计算各项风险指标
        var = RiskAssessment.calculate_var(returns, confidence_level)
        expected_shortfall = RiskAssessment.calculate_expected_shortfall(
            returns, confidence_level
        )
        volatility = RiskAssessment.calculate_volatility(returns)
        max_drawdown = RiskAssessment.calculate_max_drawdown(historical_prices)
        sharpe_ratio = RiskAssessment.calculate_sharpe_ratio(returns)

        return RiskMetrics(
            value_at_risk=var,
            expected_shortfall=expected_shortfall,
            volatility=volatility,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
        )


class PredictionEngine:
    """预测引擎主类"""

    def __init__(
        self, model_dir: str = "backend/models", data_dir: str = "backend/data"
    ):
        self.model_loader = ModelLoader(model_dir)
        self.feature_extractor = FeatureExtractor(cache_enabled=True)
        self.risk_assessment = RiskAssessment()
        self.data_dir = Path(data_dir)

        # 预测缓存
        self.prediction_cache: Dict[str, PredictionOutput] = {}
        self.cache_ttl = timedelta(hours=1)  # 缓存1小时

        # 性能统计
        self.prediction_stats = {
            "total_predictions": 0,
            "cache_hits": 0,
            "model_fallbacks": 0,
            "errors": 0,
        }

    def predict_single_stock(
        self,
        stock_code: str,
        config: PredictionConfig,
        end_date: Optional[datetime] = None,
    ) -> PredictionOutput:
        """预测单只股票"""
        try:
            if end_date is None:
                end_date = datetime.now()

            # 检查缓存
            cache_key = f"{stock_code}_{config.model_id}_{end_date.date()}"
            if cache_key in self.prediction_cache:
                cached_prediction = self.prediction_cache[cache_key]
                if datetime.now() - cached_prediction.prediction_date < self.cache_ttl:
                    self.prediction_stats["cache_hits"] += 1
                    logger.info(f"使用缓存预测: {stock_code}")
                    return cached_prediction

            # 加载历史数据
            historical_data = self._load_stock_data(stock_code, end_date)

            # 加载模型
            model, model_metadata = self.model_loader.load_model(config.model_id)

            current_price = float(historical_data["close"].iloc[-1])

            if model_metadata.get("model_format") == "qlib_pickle":
                predicted_return = self._predict_with_qlib_model(
                    model_metadata["model_path"], stock_code, end_date
                )
                predicted_price = current_price * (1 + predicted_return)
                prediction_result = {
                    "predicted_price": predicted_price,
                    "predicted_direction": 1
                    if predicted_return > 0.01
                    else -1
                    if predicted_return < -0.01
                    else 0,
                    "confidence_score": model_metadata.get(
                        "performance_metrics", {}
                    ).get("accuracy", 0.5),
                    "confidence_interval": self._calculate_confidence_interval(
                        predicted_price, 0.02, config.confidence_level
                    ),
                    "features_used": [],
                    "model_confidence": model_metadata.get(
                        "performance_metrics", {}
                    ).get("accuracy", 0.5),
                    "predicted_return": float(predicted_return),
                }
            else:
                # 提取特征
                features = self._extract_features(stock_code, historical_data, config)
                # 执行预测
                prediction_result = self._execute_prediction(
                    stock_code, features, model, model_metadata, config, current_price
                )

            # 风险评估
            if config.risk_assessment:
                risk_metrics = self.risk_assessment.assess_prediction_risk(
                    historical_data["close"],
                    prediction_result["predicted_price"],
                    config.confidence_level,
                )
            else:
                risk_metrics = RiskMetrics(0, 0, 0, 0, 0)

            # 构建预测输出
            prediction_output = PredictionOutput(
                stock_code=stock_code,
                prediction_date=datetime.now(),
                predicted_price=prediction_result["predicted_price"],
                predicted_direction=prediction_result["predicted_direction"],
                confidence_score=prediction_result["confidence_score"],
                confidence_interval=prediction_result["confidence_interval"],
                risk_metrics=risk_metrics,
                model_id=config.model_id,
                features_used=prediction_result["features_used"],
                model_confidence=prediction_result["model_confidence"],
                prediction_horizon=config.horizon,
            )

            # 缓存结果
            self.prediction_cache[cache_key] = prediction_output
            self.prediction_stats["total_predictions"] += 1

            logger.info(
                f"预测完成: {stock_code}, 预测价格: {prediction_result['predicted_price']:.2f}"
            )
            return prediction_output

        except Exception as e:
            self.prediction_stats["errors"] += 1
            error_context = ErrorContext(
                stock_code=stock_code, model_id=config.model_id
            )

            # 提供恢复建议
            recovery_actions = [
                RecoveryAction(
                    action_type="use_fallback_model",
                    parameters={"fallback_model_id": "simple_linear"},
                    description="使用备用线性模型",
                ),
                RecoveryAction(
                    action_type="extend_data_window",
                    parameters={"additional_days": 30},
                    description="扩大历史数据窗口",
                ),
            ]

            raise PredictionError(
                message=f"单股预测失败: {str(e)}",
                severity=ErrorSeverity.HIGH,
                context=error_context,
                original_exception=e,
                recovery_actions=recovery_actions,
            )

    def predict_multiple_stocks(
        self,
        stock_codes: List[str],
        config: PredictionConfig,
        end_date: Optional[datetime] = None,
    ) -> List[PredictionOutput]:
        """批量预测多只股票"""
        predictions = []
        failed_stocks = []

        for stock_code in stock_codes:
            try:
                prediction = self.predict_single_stock(stock_code, config, end_date)
                predictions.append(prediction)
            except Exception as e:
                logger.error(f"预测失败: {stock_code}, 错误: {e}")
                failed_stocks.append(stock_code)
                continue

        if failed_stocks:
            logger.warning(f"部分股票预测失败: {failed_stocks}")

        logger.info(f"批量预测完成: 成功 {len(predictions)}, 失败 {len(failed_stocks)}")
        return predictions

    def _load_stock_data(
        self, stock_code: str, end_date: datetime, days_back: int = 252
    ) -> pd.DataFrame:
        """加载股票历史数据"""
        try:
            # 使用统一的数据加载器
            from app.services.data.stock_data_loader import StockDataLoader

            loader = StockDataLoader(data_root=str(self.data_dir))

            # 计算开始日期（确保有足够的历史数据）
            start_date = None  # 不限制开始日期，让加载器返回所有数据

            # 加载数据
            data = loader.load_stock_data(
                stock_code, start_date=start_date, end_date=end_date
            )

            if data.empty:
                raise PredictionError(
                    message=f"未找到股票数据文件: {stock_code}",
                    severity=ErrorSeverity.HIGH,
                    context=ErrorContext(stock_code=stock_code),
                )

            # 获取最近的交易日数据
            data = data.tail(days_back)

            if len(data) < 50:
                raise PredictionError(
                    message=f"历史数据不足: {stock_code}, 需要至少50个交易日，当前: {len(data)}",
                    severity=ErrorSeverity.MEDIUM,
                    context=ErrorContext(stock_code=stock_code),
                )

            logger.info(f"加载股票数据: {stock_code}, 数据量: {len(data)}")
            return data

        except Exception as e:
            if isinstance(e, PredictionError):
                raise

            raise PredictionError(
                message=f"加载股票数据失败: {str(e)}",
                severity=ErrorSeverity.HIGH,
                context=ErrorContext(stock_code=stock_code),
                original_exception=e,
            )

    def _extract_features(
        self, stock_code: str, data: pd.DataFrame, config: PredictionConfig
    ) -> pd.DataFrame:
        """提取特征"""
        try:
            # 构建特征配置
            feature_config = FeatureConfig(
                technical_indicators=[
                    "sma_5",
                    "sma_10",
                    "sma_20",
                    "ema_12",
                    "ema_26",
                    "rsi_14",
                    "macd",
                    "bb_20",
                    "atr_14",
                ],
                statistical_features=[
                    "returns",
                    "volatility",
                    "momentum",
                    "price_ratios",
                ],
                time_windows=[5, 10, 20],
                cache_enabled=True,
            )

            # 提取特征
            features = self.feature_extractor.extract_features(
                stock_code, data, feature_config
            )

            # 如果指定了特定特征，进行过滤
            if config.features:
                available_features = [
                    f for f in config.features if f in features.columns
                ]
                if available_features:
                    features = features[available_features]
                else:
                    logger.warning(f"指定的特征不可用: {config.features}")

            return features

        except Exception as e:
            raise PredictionError(
                message=f"特征提取失败: {str(e)}",
                severity=ErrorSeverity.HIGH,
                context=ErrorContext(stock_code=stock_code),
                original_exception=e,
            )

    def _execute_prediction(
        self,
        stock_code: str,
        features: pd.DataFrame,
        model: Any,
        model_metadata: Dict[str, Any],
        config: PredictionConfig,
        current_price: float,
    ) -> Dict[str, Any]:
        """执行预测计算"""
        try:
            # 获取最新特征
            latest_features = features.iloc[-1:].fillna(0)

            # 执行预测
            if hasattr(model, "predict"):
                predicted_return = model.predict(latest_features)[0]
            else:
                raise PredictionError(
                    message=f"模型不支持predict接口: {type(model)}",
                    severity=ErrorSeverity.HIGH,
                    context=ErrorContext(
                        stock_code=stock_code, model_id=config.model_id
                    ),
                )

            # 计算预测价格
            predicted_price = current_price * (1 + predicted_return)

            # 确定预测方向
            if predicted_return > 0.01:
                predicted_direction = 1  # 上涨
            elif predicted_return < -0.01:
                predicted_direction = -1  # 下跌
            else:
                predicted_direction = 0  # 持平

            # 计算置信度
            model_confidence = model_metadata.get("performance_metrics", {}).get(
                "accuracy", 0.5
            )
            confidence_score = min(0.95, max(0.1, model_confidence))

            # 计算置信区间
            volatility = 0.02  # 默认波动率
            if "volatility_20d" in features.columns:
                volatility = (
                    features["volatility_20d"].iloc[-1]
                    if not pd.isna(features["volatility_20d"].iloc[-1])
                    else 0.02
                )

            confidence_interval = self._calculate_confidence_interval(
                predicted_price, volatility, config.confidence_level
            )

            return {
                "predicted_price": predicted_price,
                "predicted_direction": predicted_direction,
                "confidence_score": confidence_score,
                "confidence_interval": confidence_interval,
                "features_used": features.columns.tolist(),
                "model_confidence": model_confidence,
                "predicted_return": float(predicted_return),
            }

        except Exception as e:
            raise PredictionError(
                message=f"预测计算失败: {str(e)}",
                severity=ErrorSeverity.HIGH,
                context=ErrorContext(stock_code=stock_code),
                original_exception=e,
            )

    def _calculate_confidence_interval(
        self, predicted_price: float, volatility: float, confidence_level: float
    ) -> Tuple[float, float]:
        """计算置信区间"""
        from scipy import stats

        # 计算Z分数
        z_score = stats.norm.ppf((1 + confidence_level) / 2)

        # 计算置信区间
        margin = predicted_price * volatility * z_score
        lower_bound = predicted_price - margin
        upper_bound = predicted_price + margin

        return (lower_bound, upper_bound)

    def _predict_with_qlib_model(
        self, model_path: str, stock_code: str, end_date: datetime
    ) -> float:
        """使用Qlib模型进行预测，返回预测收益率"""
        try:
            import asyncio

            from app.services.qlib.unified_qlib_training_engine import (
                UnifiedQlibTrainingEngine,
            )

            engine = UnifiedQlibTrainingEngine()
            start_date = end_date - timedelta(days=365)

            async def run_predict():
                return await engine.predict_with_qlib_model(
                    model_path=model_path,
                    stock_codes=[stock_code],
                    start_date=start_date,
                    end_date=end_date,
                )

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                predictions = loop.run_until_complete(run_predict())
            finally:
                loop.close()

            if predictions is None or len(predictions) == 0:
                raise PredictionError(
                    message="Qlib模型预测结果为空",
                    severity=ErrorSeverity.HIGH,
                    context=ErrorContext(stock_code=stock_code),
                )

            if isinstance(predictions, pd.DataFrame):
                data = predictions
                if isinstance(data.index, pd.MultiIndex):
                    if "instrument" in data.index.names:
                        data = data.xs(stock_code, level="instrument")
                    else:
                        try:
                            data = data.xs(stock_code, level=-1)
                        except Exception:
                            pass
                last_row = data.iloc[-1]
                if isinstance(last_row, pd.Series):
                    return float(last_row.iloc[0])
                return float(last_row)

            if isinstance(predictions, pd.Series):
                return float(predictions.iloc[-1])

            return float(predictions[-1])

        except Exception as e:
            raise PredictionError(
                message=f"Qlib模型预测失败: {str(e)}",
                severity=ErrorSeverity.HIGH,
                context=ErrorContext(stock_code=stock_code),
                original_exception=e,
            )

    async def predict_return_series(
        self,
        stock_code: str,
        config: PredictionConfig,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.Series:
        """预测区间内的收益率序列"""
        model, model_metadata = self.model_loader.load_model(config.model_id)

        if model_metadata.get("model_format") == "qlib_pickle":
            from app.services.qlib.unified_qlib_training_engine import (
                UnifiedQlibTrainingEngine,
            )

            engine = UnifiedQlibTrainingEngine()

            predictions = await engine.predict_with_qlib_model(
                model_path=model_metadata["model_path"],
                stock_codes=[stock_code],
                start_date=start_date,
                end_date=end_date,
            )

            if isinstance(predictions, pd.DataFrame):
                data = predictions
                if isinstance(data.index, pd.MultiIndex):
                    if "instrument" in data.index.names:
                        data = data.xs(stock_code, level="instrument")
                    else:
                        data = data.xs(stock_code, level=-1)
                series = data.iloc[:, 0] if isinstance(data, pd.DataFrame) else data
                return series.sort_index()

            if isinstance(predictions, pd.Series):
                return predictions.sort_index()

            raise PredictionError(
                message="Qlib预测结果类型不支持",
                severity=ErrorSeverity.HIGH,
                context=ErrorContext(stock_code=stock_code),
            )

        historical_data = self._load_stock_data(stock_code, end_date)
        features = self._extract_features(stock_code, historical_data, config)
        features = features.loc[
            (features.index >= start_date) & (features.index <= end_date)
        ]

        if features.empty:
            raise PredictionError(
                message="预测特征数据为空",
                severity=ErrorSeverity.HIGH,
                context=ErrorContext(stock_code=stock_code),
            )

        if not hasattr(model, "predict"):
            raise PredictionError(
                message=f"模型不支持predict接口: {type(model)}",
                severity=ErrorSeverity.HIGH,
                context=ErrorContext(stock_code=stock_code, model_id=config.model_id),
            )

        predictions = model.predict(features.fillna(0))
        return pd.Series(predictions, index=features.index).sort_index()

    def validate_prediction_inputs(
        self, stock_code: str, config: PredictionConfig
    ) -> bool:
        """验证预测输入参数"""
        try:
            # 验证股票代码
            if not stock_code or len(stock_code) < 6:
                raise PredictionError(
                    message=f"无效的股票代码: {stock_code}", severity=ErrorSeverity.MEDIUM
                )

            # 验证模型ID
            if not config.model_id:
                raise PredictionError(message="模型ID不能为空", severity=ErrorSeverity.MEDIUM)

            # 验证置信水平
            if not 0.5 <= config.confidence_level <= 0.99:
                raise PredictionError(
                    message=f"置信水平必须在0.5-0.99之间: {config.confidence_level}",
                    severity=ErrorSeverity.MEDIUM,
                )

            # 验证预测期限
            valid_horizons = ["short_term", "medium_term", "long_term"]
            if config.horizon not in valid_horizons:
                raise PredictionError(
                    message=f"无效的预测期限: {config.horizon}，有效值: {valid_horizons}",
                    severity=ErrorSeverity.MEDIUM,
                )

            return True

        except PredictionError:
            raise
        except Exception as e:
            raise PredictionError(
                message=f"输入验证失败: {str(e)}",
                severity=ErrorSeverity.MEDIUM,
                original_exception=e,
            )

    def get_prediction_stats(self) -> Dict[str, Any]:
        """获取预测统计信息"""
        return {
            **self.prediction_stats,
            "cache_size": len(self.prediction_cache),
            "cache_hit_rate": (
                self.prediction_stats["cache_hits"]
                / max(self.prediction_stats["total_predictions"], 1)
            ),
            "error_rate": (
                self.prediction_stats["errors"]
                / max(self.prediction_stats["total_predictions"], 1)
            ),
            "feature_cache_stats": self.feature_extractor.get_cache_stats(),
        }

    def clear_cache(self):
        """清空预测缓存"""
        self.prediction_cache.clear()
        if self.feature_extractor.cache:
            self.feature_extractor.cache.clear()
        logger.info("预测缓存已清空")
