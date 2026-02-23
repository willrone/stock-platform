"""
模型训练模块

LightGBM 和 XGBoost 回归模型训练
"""
from typing import Any, Dict, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from loguru import logger

from .config import LGBConfig, TrainingConfig, XGBConfig
from .constants import EARLY_STOPPING_ROUNDS, LOG_EVAL_PERIOD, MAX_BOOST_ROUNDS
from .features import get_feature_columns


def train_lightgbm(
    train: pd.DataFrame, val: pd.DataFrame, config: TrainingConfig
) -> Tuple[lgb.Booster, Dict[str, float]]:
    """
    训练 LightGBM 回归模型

    Returns:
        (model, metrics_dict)
    """
    feature_cols = get_feature_columns()
    lgb_config = LGBConfig(seed=config.seed)

    X_train, y_train = train[feature_cols], train["future_return"]
    X_val, y_val = val[feature_cols], val["future_return"]

    train_data = lgb.Dataset(X_train, y_train)
    val_data = lgb.Dataset(X_val, y_val, reference=train_data)

    logger.info("开始训练 LightGBM（回归模式）...")
    model = lgb.train(
        lgb_config.to_dict(),
        train_data,
        num_boost_round=config.max_boost_rounds,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(config.early_stopping_rounds),
            lgb.log_evaluation(LOG_EVAL_PERIOD),
        ],
    )

    metrics = _evaluate_regression(model.predict(X_val), y_val, "LightGBM")
    return model, metrics


def train_xgboost(
    train: pd.DataFrame, val: pd.DataFrame, config: TrainingConfig
) -> Tuple[xgb.Booster, Dict[str, float]]:
    """
    训练 XGBoost 回归模型

    Returns:
        (model, metrics_dict)
    """
    feature_cols = get_feature_columns()
    xgb_config = XGBConfig(seed=config.seed)

    X_train, y_train = train[feature_cols], train["future_return"]
    X_val, y_val = val[feature_cols], val["future_return"]

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    logger.info("开始训练 XGBoost（回归模式）...")
    model = xgb.train(
        xgb_config.to_dict(),
        dtrain,
        num_boost_round=config.max_boost_rounds,
        evals=[(dval, "val")],
        early_stopping_rounds=config.early_stopping_rounds,
        verbose_eval=LOG_EVAL_PERIOD,
    )

    metrics = _evaluate_regression(model.predict(dval), y_val, "XGBoost")
    return model, metrics


def predict_ensemble(
    lgb_model: lgb.Booster,
    xgb_model: xgb.Booster,
    data: pd.DataFrame,
) -> np.ndarray:
    """集成预测：等权重平均"""
    feature_cols = get_feature_columns()
    X = data[feature_cols]

    lgb_pred = lgb_model.predict(X)
    xgb_pred = xgb_model.predict(xgb.DMatrix(X))

    return 0.5 * lgb_pred + 0.5 * xgb_pred


def _evaluate_regression(
    predictions: np.ndarray, actuals: pd.Series, model_name: str
) -> Dict[str, float]:
    """评估回归模型指标：MSE, R², IC"""
    actuals_arr = actuals.values

    mse = float(np.mean((predictions - actuals_arr) ** 2))
    r2 = _compute_r2(predictions, actuals_arr)
    ic = _compute_ic(predictions, actuals_arr)

    logger.info(
        f"{model_name} 验证指标: MSE={mse:.6f}, R²={r2:.6f}, IC={ic:.4f}"
    )
    return {"mse": mse, "r2": r2, "ic": ic}


def _compute_r2(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """计算 R² 决定系数"""
    ss_res = np.sum((actuals - predictions) ** 2)
    ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1 - ss_res / ss_tot)


def _compute_ic(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """计算信息系数（Pearson 相关系数）"""
    if len(predictions) < 2:
        return 0.0
    corr = np.corrcoef(predictions, actuals)[0, 1]
    return float(corr) if np.isfinite(corr) else 0.0
