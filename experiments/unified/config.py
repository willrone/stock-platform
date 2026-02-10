"""
训练配置数据类

使用 dataclass 封装训练参数，避免函数参数过多
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional

from .constants import (
    DATA_DIR,
    DEFAULT_END_DATE,
    DEFAULT_N_STOCKS,
    DEFAULT_START_DATE,
    DEFAULT_TOP_N,
    DEFAULT_TRANSACTION_COST,
    EARLY_STOPPING_ROUNDS,
    EMBARGO_DAYS,
    MAX_BOOST_ROUNDS,
    MODEL_DIR,
    RANDOM_SEED,
    TRAIN_END_DATE,
    VAL_END_DATE,
)


@dataclass
class TrainingConfig:
    """统一训练配置"""

    # 数据配置
    data_dir: Path = DATA_DIR
    model_dir: Path = MODEL_DIR
    n_stocks: int = DEFAULT_N_STOCKS
    start_date: str = DEFAULT_START_DATE
    end_date: str = DEFAULT_END_DATE

    # 时间分割
    train_end: str = TRAIN_END_DATE
    val_end: str = VAL_END_DATE
    embargo_days: int = EMBARGO_DAYS

    # 模型配置
    seed: int = RANDOM_SEED
    max_boost_rounds: int = MAX_BOOST_ROUNDS
    early_stopping_rounds: int = EARLY_STOPPING_ROUNDS

    # 市场中性化
    enable_neutralization: bool = True

    # 回测配置
    top_n: int = DEFAULT_TOP_N
    transaction_cost: float = DEFAULT_TRANSACTION_COST


@dataclass
class LGBConfig:
    """LightGBM 回归参数"""

    objective: str = "huber"
    metric: str = "mae"
    boosting_type: str = "gbdt"
    num_leaves: int = 45
    max_depth: int = 6
    learning_rate: float = 0.015
    feature_fraction: float = 0.65
    bagging_fraction: float = 0.65
    bagging_freq: int = 5
    min_child_samples: int = 100
    reg_alpha: float = 0.6
    reg_lambda: float = 0.6
    verbose: int = -1
    seed: int = RANDOM_SEED

    def to_dict(self) -> Dict[str, Any]:
        """转换为 LightGBM 参数字典"""
        return {
            "objective": self.objective,
            "metric": self.metric,
            "boosting_type": self.boosting_type,
            "num_leaves": self.num_leaves,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "feature_fraction": self.feature_fraction,
            "bagging_fraction": self.bagging_fraction,
            "bagging_freq": self.bagging_freq,
            "min_child_samples": self.min_child_samples,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "verbose": self.verbose,
            "seed": self.seed,
        }


@dataclass
class XGBConfig:
    """XGBoost 回归参数"""

    objective: str = "reg:squarederror"
    eval_metric: str = "mae"
    max_depth: int = 5
    learning_rate: float = 0.015
    subsample: float = 0.65
    colsample_bytree: float = 0.65
    min_child_weight: int = 100
    reg_alpha: float = 0.6
    reg_lambda: float = 0.6
    seed: int = RANDOM_SEED

    def to_dict(self) -> Dict[str, Any]:
        """转换为 XGBoost 参数字典"""
        return {
            "objective": self.objective,
            "eval_metric": self.eval_metric,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "min_child_weight": self.min_child_weight,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "seed": self.seed,
        }
