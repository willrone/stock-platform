"""
预测引擎降级策略和错误处理
"""

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from app.core.error_handler import (
    ErrorContext,
    ErrorSeverity,
    PredictionError,
    RecoveryAction,
)


class FallbackStrategy(Enum):
    """降级策略类型"""

    SIMPLE_LINEAR = "simple_linear"
    MOVING_AVERAGE = "moving_average"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    RANDOM_WALK = "random_walk"
    HISTORICAL_AVERAGE = "historical_average"


@dataclass
class FallbackConfig:
    """降级配置"""

    strategy: FallbackStrategy
    parameters: Dict[str, Any]
    confidence_penalty: float = 0.3  # 降级时置信度惩罚
    max_retries: int = 3
    retry_delay: float = 1.0  # 秒


@dataclass
class RetryConfig:
    """重试配置"""

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_factor: float = 2.0
    jitter: bool = True


class PredictionFallbackEngine:
    """预测降级引擎"""

    def __init__(self):
        self.fallback_strategies = {
            FallbackStrategy.SIMPLE_LINEAR: self._simple_linear_prediction,
            FallbackStrategy.MOVING_AVERAGE: self._moving_average_prediction,
            FallbackStrategy.MOMENTUM: self._momentum_prediction,
            FallbackStrategy.MEAN_REVERSION: self._mean_reversion_prediction,
            FallbackStrategy.RANDOM_WALK: self._random_walk_prediction,
            FallbackStrategy.HISTORICAL_AVERAGE: self._historical_average_prediction,
        }

        # 默认降级策略优先级
        self.fallback_priority = [
            FallbackStrategy.SIMPLE_LINEAR,
            FallbackStrategy.MOVING_AVERAGE,
            FallbackStrategy.MOMENTUM,
            FallbackStrategy.MEAN_REVERSION,
            FallbackStrategy.HISTORICAL_AVERAGE,
            FallbackStrategy.RANDOM_WALK,
        ]

    def execute_fallback_prediction(
        self,
        stock_code: str,
        historical_data: pd.DataFrame,
        strategy: FallbackStrategy,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """执行降级预测"""
        try:
            if parameters is None:
                parameters = {}

            # 获取降级策略函数
            strategy_func = self.fallback_strategies.get(strategy)
            if not strategy_func:
                raise PredictionError(
                    message=f"未知的降级策略: {strategy}",
                    severity=ErrorSeverity.MEDIUM,
                    context=ErrorContext(stock_code=stock_code),
                )

            # 执行降级预测
            result = strategy_func(historical_data, parameters)

            # 添加降级标记
            result["is_fallback"] = True
            result["fallback_strategy"] = strategy.value
            result["confidence_score"] *= 1 - 0.3  # 降级惩罚

            logger.info(f"降级预测完成: {stock_code}, 策略: {strategy.value}")
            return result

        except Exception as e:
            raise PredictionError(
                message=f"降级预测失败: {str(e)}",
                severity=ErrorSeverity.HIGH,
                context=ErrorContext(stock_code=stock_code),
                original_exception=e,
            )

    def _simple_linear_prediction(
        self, data: pd.DataFrame, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """简单线性回归预测"""
        window = params.get("window", 20)

        if len(data) < window:
            window = len(data)

        prices = data["close"].tail(window)

        # 简单线性回归
        x = np.arange(len(prices))
        y = prices.values

        # 计算斜率和截距
        slope = np.polyfit(x, y, 1)[0]

        # 预测下一个价格
        current_price = prices.iloc[-1]
        predicted_price = current_price + slope

        # 计算预测方向
        predicted_direction = 1 if slope > 0 else (-1 if slope < 0 else 0)

        # 计算置信度（基于拟合度）
        y_pred = np.polyval([slope, prices.iloc[0]], x)
        r_squared = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
        confidence_score = max(0.1, min(0.8, r_squared))

        return {
            "predicted_price": predicted_price,
            "predicted_direction": predicted_direction,
            "confidence_score": confidence_score,
            "method_details": {
                "slope": slope,
                "r_squared": r_squared,
                "window": window,
            },
        }

    def _moving_average_prediction(
        self, data: pd.DataFrame, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """移动平均预测"""
        short_window = params.get("short_window", 5)
        long_window = params.get("long_window", 20)

        prices = data["close"]

        # 计算移动平均
        short_ma = prices.rolling(window=short_window).mean().iloc[-1]
        long_ma = prices.rolling(window=long_window).mean().iloc[-1]
        current_price = prices.iloc[-1]

        # 基于移动平均交叉预测
        if short_ma > long_ma:
            # 短期均线在长期均线之上，预测上涨
            predicted_price = current_price * 1.02
            predicted_direction = 1
        elif short_ma < long_ma:
            # 短期均线在长期均线之下，预测下跌
            predicted_price = current_price * 0.98
            predicted_direction = -1
        else:
            # 均线持平
            predicted_price = current_price
            predicted_direction = 0

        # 计算置信度（基于均线距离）
        ma_diff = abs(short_ma - long_ma) / long_ma
        confidence_score = max(0.1, min(0.7, ma_diff * 10))

        return {
            "predicted_price": predicted_price,
            "predicted_direction": predicted_direction,
            "confidence_score": confidence_score,
            "method_details": {
                "short_ma": short_ma,
                "long_ma": long_ma,
                "ma_diff": ma_diff,
            },
        }

    def _momentum_prediction(
        self, data: pd.DataFrame, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """动量预测"""
        window = params.get("window", 10)

        prices = data["close"]

        # 计算动量
        momentum = prices.iloc[-1] / prices.iloc[-window] - 1
        current_price = prices.iloc[-1]

        # 基于动量预测
        predicted_return = momentum * 0.5  # 动量衰减
        predicted_price = current_price * (1 + predicted_return)

        # 预测方向
        predicted_direction = (
            1 if predicted_return > 0.01 else (-1 if predicted_return < -0.01 else 0)
        )

        # 置信度基于动量强度
        confidence_score = max(0.1, min(0.6, abs(momentum) * 2))

        return {
            "predicted_price": predicted_price,
            "predicted_direction": predicted_direction,
            "confidence_score": confidence_score,
            "method_details": {
                "momentum": momentum,
                "predicted_return": predicted_return,
                "window": window,
            },
        }

    def _mean_reversion_prediction(
        self, data: pd.DataFrame, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """均值回归预测"""
        window = params.get("window", 20)
        reversion_speed = params.get("reversion_speed", 0.1)

        prices = data["close"]

        # 计算历史均值
        historical_mean = prices.tail(window).mean()
        current_price = prices.iloc[-1]

        # 均值回归预测
        deviation = current_price - historical_mean
        predicted_price = current_price - deviation * reversion_speed

        # 预测方向
        if predicted_price > current_price * 1.01:
            predicted_direction = 1
        elif predicted_price < current_price * 0.99:
            predicted_direction = -1
        else:
            predicted_direction = 0

        # 置信度基于偏离程度
        deviation_ratio = abs(deviation) / historical_mean
        confidence_score = max(0.1, min(0.6, deviation_ratio * 5))

        return {
            "predicted_price": predicted_price,
            "predicted_direction": predicted_direction,
            "confidence_score": confidence_score,
            "method_details": {
                "historical_mean": historical_mean,
                "deviation": deviation,
                "deviation_ratio": deviation_ratio,
            },
        }

    def _random_walk_prediction(
        self, data: pd.DataFrame, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """随机游走预测"""
        volatility_window = params.get("volatility_window", 20)

        prices = data["close"]
        returns = prices.pct_change().dropna()

        # 计算历史波动率
        volatility = returns.tail(volatility_window).std()
        current_price = prices.iloc[-1]

        # 随机游走预测（期望收益为0）
        predicted_price = current_price
        predicted_direction = 0

        # 置信度很低，因为是随机预测
        confidence_score = 0.1

        return {
            "predicted_price": predicted_price,
            "predicted_direction": predicted_direction,
            "confidence_score": confidence_score,
            "method_details": {"volatility": volatility, "method": "random_walk"},
        }

    def _historical_average_prediction(
        self, data: pd.DataFrame, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """历史平均预测"""
        window = params.get("window", 50)

        prices = data["close"]
        returns = prices.pct_change().dropna()

        # 计算历史平均收益
        avg_return = returns.tail(window).mean()
        current_price = prices.iloc[-1]

        # 基于历史平均收益预测
        predicted_price = current_price * (1 + avg_return)

        # 预测方向
        predicted_direction = (
            1 if avg_return > 0.001 else (-1 if avg_return < -0.001 else 0)
        )

        # 置信度基于收益稳定性
        return_std = returns.tail(window).std()
        confidence_score = max(
            0.1, min(0.5, abs(avg_return) / return_std if return_std > 0 else 0.1)
        )

        return {
            "predicted_price": predicted_price,
            "predicted_direction": predicted_direction,
            "confidence_score": confidence_score,
            "method_details": {
                "avg_return": avg_return,
                "return_std": return_std,
                "window": window,
            },
        }


class RetryManager:
    """重试管理器"""

    def __init__(self, config: Optional[RetryConfig] = None):
        self.config = config or RetryConfig()

    def execute_with_retry(self, func, *args, **kwargs):
        """带重试的函数执行"""
        last_exception = None

        for attempt in range(self.config.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e

                if attempt < self.config.max_retries:
                    delay = self._calculate_delay(attempt)
                    logger.warning(
                        f"执行失败，{delay:.2f}秒后重试 (尝试 {attempt + 1}/{self.config.max_retries}): {e}"
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"重试次数已用完，最终失败: {e}")

        # 所有重试都失败了
        raise last_exception

    def _calculate_delay(self, attempt: int) -> float:
        """计算重试延迟"""
        delay = min(
            self.config.base_delay * (self.config.backoff_factor**attempt),
            self.config.max_delay,
        )

        # 添加随机抖动
        if self.config.jitter:
            delay *= 0.5 + np.random.random() * 0.5

        return delay


class PredictionErrorHandler:
    """预测错误处理器"""

    def __init__(self):
        self.fallback_engine = PredictionFallbackEngine()
        self.retry_manager = RetryManager()

        # 错误类型到降级策略的映射
        self.error_strategy_mapping = {
            "model_load_error": FallbackStrategy.SIMPLE_LINEAR,
            "data_insufficient": FallbackStrategy.MOVING_AVERAGE,
            "feature_extraction_error": FallbackStrategy.MOMENTUM,
            "prediction_timeout": FallbackStrategy.HISTORICAL_AVERAGE,
            "memory_error": FallbackStrategy.MEAN_REVERSION,
            "unknown_error": FallbackStrategy.RANDOM_WALK,
        }

    def handle_prediction_error(
        self,
        error: Exception,
        stock_code: str,
        historical_data: pd.DataFrame,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """处理预测错误并执行降级策略"""
        try:
            # 分析错误类型
            error_type = self._classify_error(error)

            # 选择降级策略
            fallback_strategy = self._select_fallback_strategy(error_type, context)

            # 执行降级预测
            result = self.fallback_engine.execute_fallback_prediction(
                stock_code, historical_data, fallback_strategy
            )

            # 添加错误处理信息
            result["error_handled"] = True
            result["original_error"] = str(error)
            result["error_type"] = error_type
            result["recovery_strategy"] = fallback_strategy.value

            logger.info(
                f"错误处理完成: {stock_code}, 错误类型: {error_type}, 降级策略: {fallback_strategy.value}"
            )
            return result

        except Exception as fallback_error:
            # 降级策略也失败了，返回最基本的预测
            logger.error(f"降级策略失败: {fallback_error}")
            return self._emergency_fallback(
                stock_code, historical_data, error, fallback_error
            )

    def _classify_error(self, error: Exception) -> str:
        """分类错误类型"""
        error_message = str(error).lower()

        if "model" in error_message and (
            "load" in error_message or "file" in error_message
        ):
            return "model_load_error"
        elif "data" in error_message and (
            "insufficient" in error_message or "not enough" in error_message
        ):
            return "data_insufficient"
        elif "feature" in error_message or "extraction" in error_message:
            return "feature_extraction_error"
        elif "timeout" in error_message or "time" in error_message:
            return "prediction_timeout"
        elif "memory" in error_message or "ram" in error_message:
            return "memory_error"
        else:
            return "unknown_error"

    def _select_fallback_strategy(
        self, error_type: str, context: Optional[Dict[str, Any]] = None
    ) -> FallbackStrategy:
        """选择降级策略"""
        # 基于错误类型选择策略
        strategy = self.error_strategy_mapping.get(
            error_type, FallbackStrategy.SIMPLE_LINEAR
        )

        # 根据上下文调整策略
        if context:
            data_length = context.get("data_length", 0)
            if data_length < 20:
                # 数据太少，使用简单策略
                strategy = FallbackStrategy.HISTORICAL_AVERAGE
            elif data_length < 50:
                # 数据中等，使用移动平均
                strategy = FallbackStrategy.MOVING_AVERAGE

        return strategy

    def _emergency_fallback(
        self,
        stock_code: str,
        historical_data: pd.DataFrame,
        original_error: Exception,
        fallback_error: Exception,
    ) -> Dict[str, Any]:
        """紧急降级策略"""
        try:
            current_price = (
                historical_data["close"].iloc[-1] if len(historical_data) > 0 else 100.0
            )

            # 最简单的预测：当前价格不变
            return {
                "predicted_price": current_price,
                "predicted_direction": 0,
                "confidence_score": 0.05,  # 极低置信度
                "is_fallback": True,
                "is_emergency_fallback": True,
                "fallback_strategy": "emergency",
                "original_error": str(original_error),
                "fallback_error": str(fallback_error),
                "method_details": {
                    "method": "emergency_no_change",
                    "current_price": current_price,
                },
            }
        except Exception as e:
            # 连紧急降级都失败了，返回固定值
            logger.critical(f"紧急降级失败: {e}")
            return {
                "predicted_price": 100.0,
                "predicted_direction": 0,
                "confidence_score": 0.01,
                "is_fallback": True,
                "is_emergency_fallback": True,
                "is_critical_failure": True,
                "fallback_strategy": "critical_failure",
                "original_error": str(original_error),
                "fallback_error": str(fallback_error),
                "critical_error": str(e),
            }

    def get_recovery_suggestions(self, error_type: str) -> List[RecoveryAction]:
        """获取恢复建议"""
        suggestions = []

        if error_type == "model_load_error":
            suggestions.extend(
                [
                    RecoveryAction(
                        action_type="check_model_file",
                        parameters={"verify_integrity": True},
                        description="检查模型文件完整性",
                    ),
                    RecoveryAction(
                        action_type="use_backup_model",
                        parameters={"backup_model_id": "simple_linear"},
                        description="使用备用模型",
                    ),
                ]
            )

        elif error_type == "data_insufficient":
            suggestions.extend(
                [
                    RecoveryAction(
                        action_type="extend_data_period",
                        parameters={"additional_days": 30},
                        description="扩展历史数据期间",
                    ),
                    RecoveryAction(
                        action_type="use_similar_stock_data",
                        parameters={"similarity_threshold": 0.8},
                        description="使用相似股票数据",
                    ),
                ]
            )

        elif error_type == "prediction_timeout":
            suggestions.extend(
                [
                    RecoveryAction(
                        action_type="increase_timeout",
                        parameters={"timeout_multiplier": 2},
                        description="增加预测超时时间",
                    ),
                    RecoveryAction(
                        action_type="use_simpler_model",
                        parameters={"model_complexity": "low"},
                        description="使用更简单的模型",
                    ),
                ]
            )

        return suggestions
