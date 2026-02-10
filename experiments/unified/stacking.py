"""
Stacking 集成模块

用 LightGBM/XGBoost 的 out-of-fold 预测作为 meta-learner 输入，
替代简单等权平均，学习最优组合权重。

架构：
  Layer 1 (Base Learners): LightGBM + XGBoost
  Layer 2 (Meta-Learner):  Ridge Regression

流程：
  1. 用 Purged K-Fold 生成 out-of-fold 预测
  2. 拼接 OOF 预测作为 meta 特征
  3. 训练 Ridge meta-learner
  4. 最终预测 = meta_learner(lgb_pred, xgb_pred)
"""
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from loguru import logger

from .config import LGBConfig, TrainingConfig, XGBConfig
from .constants import (
    EARLY_STOPPING_ROUNDS,
    LOG_EVAL_PERIOD,
    RANDOM_SEED,
)
from .cross_validation import PurgedCVConfig, PurgedGroupTimeSeriesSplit
from .features import get_feature_columns


# === 常量 ===
DEFAULT_RIDGE_ALPHA = 1.0
META_FEATURE_NAMES = ["lgb_pred", "xgb_pred"]


@dataclass
class StackingConfig:
    """Stacking 集成配置"""

    ridge_alpha: float = DEFAULT_RIDGE_ALPHA
    cv_config: PurgedCVConfig = field(default_factory=PurgedCVConfig)
    seed: int = RANDOM_SEED


@dataclass
class StackingResult:
    """Stacking 训练结果"""

    lgb_model: lgb.Booster
    xgb_model: xgb.Booster
    meta_weights: np.ndarray  # Ridge 系数 [w_lgb, w_xgb]
    meta_intercept: float
    oof_metrics: Dict[str, float]
    final_metrics: Dict[str, float]


def train_stacking_ensemble(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    config: TrainingConfig,
    stacking_config: Optional[StackingConfig] = None,
) -> StackingResult:
    """
    训练 Stacking 集成模型

    步骤：
      1. 在训练集上用 Purged CV 生成 OOF 预测
      2. 用 OOF 预测训练 Ridge meta-learner
      3. 在全量训练集上重新训练 base learners
      4. 在验证集上评估最终效果

    Args:
        train_data: 训练集（含 date, feature_cols, future_return）
        val_data: 验证集
        config: 基础训练配置
        stacking_config: Stacking 专用配置

    Returns:
        StackingResult 包含所有模型和指标
    """
    if stacking_config is None:
        stacking_config = StackingConfig()

    feature_cols = get_feature_columns()

    # Step 1: 生成 OOF 预测
    logger.info("=== Step 1: 生成 Out-of-Fold 预测 ===")
    oof_predictions, oof_labels = _generate_oof_predictions(
        train_data, config, stacking_config, feature_cols
    )

    # Step 2: 训练 Meta-Learner
    logger.info("=== Step 2: 训练 Ridge Meta-Learner ===")
    meta_weights, meta_intercept = _train_meta_learner(
        oof_predictions, oof_labels, stacking_config
    )

    # Step 3: 在全量训练集上重新训练 base learners
    logger.info("=== Step 3: 重新训练 Base Learners（全量数据）===")
    lgb_model, xgb_model = _train_base_learners_full(
        train_data, val_data, config, feature_cols
    )

    # Step 4: 评估
    logger.info("=== Step 4: 评估 Stacking 集成 ===")
    oof_metrics = _evaluate_oof(oof_predictions, oof_labels, meta_weights, meta_intercept)
    final_metrics = _evaluate_stacking(
        lgb_model, xgb_model, meta_weights, meta_intercept,
        val_data, feature_cols
    )

    return StackingResult(
        lgb_model=lgb_model,
        xgb_model=xgb_model,
        meta_weights=meta_weights,
        meta_intercept=meta_intercept,
        oof_metrics=oof_metrics,
        final_metrics=final_metrics,
    )


def predict_stacking(
    lgb_model: lgb.Booster,
    xgb_model: xgb.Booster,
    meta_weights: np.ndarray,
    meta_intercept: float,
    data: pd.DataFrame,
) -> np.ndarray:
    """
    Stacking 集成预测

    最终预测 = meta_intercept + w_lgb * lgb_pred + w_xgb * xgb_pred
    """
    feature_cols = get_feature_columns()
    X = data[feature_cols]

    lgb_pred = lgb_model.predict(X)
    xgb_pred = xgb_model.predict(xgb.DMatrix(X))

    meta_features = np.column_stack([lgb_pred, xgb_pred])
    return meta_features @ meta_weights + meta_intercept


def _generate_oof_predictions(
    train_data: pd.DataFrame,
    config: TrainingConfig,
    stacking_config: StackingConfig,
    feature_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    用 Purged K-Fold 生成 out-of-fold 预测

    Returns:
        (oof_predictions [N, 2], oof_labels [N])
        oof_predictions[:, 0] = LGB 预测
        oof_predictions[:, 1] = XGB 预测
    """
    splitter = PurgedGroupTimeSeriesSplit(stacking_config.cv_config)

    all_oof_preds = []
    all_oof_labels = []

    for fold_idx, (fold_train, fold_val) in enumerate(
        splitter.split(train_data, date_col="date")
    ):
        logger.info(f"OOF Fold {fold_idx}: 训练 base learners...")

        lgb_pred, xgb_pred = _train_fold_base_learners(
            fold_train, fold_val, config, feature_cols
        )

        fold_labels = fold_val["future_return"].values
        fold_oof = np.column_stack([lgb_pred, xgb_pred])

        all_oof_preds.append(fold_oof)
        all_oof_labels.append(fold_labels)

        # 打印折指标
        for name, pred in [("LGB", lgb_pred), ("XGB", xgb_pred)]:
            ic = _compute_ic(pred, fold_labels)
            logger.info(f"  Fold {fold_idx} {name} OOF IC: {ic:.4f}")

    oof_predictions = np.vstack(all_oof_preds)
    oof_labels = np.concatenate(all_oof_labels)

    logger.info(
        f"OOF 预测完成: {oof_predictions.shape[0]} 样本, "
        f"{oof_predictions.shape[1]} 个 base learner"
    )
    return oof_predictions, oof_labels


def _train_fold_base_learners(
    fold_train: pd.DataFrame,
    fold_val: pd.DataFrame,
    config: TrainingConfig,
    feature_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """在单折上训练 LGB 和 XGB，返回验证集预测"""
    X_train = fold_train[feature_cols]
    y_train = fold_train["future_return"]
    X_val = fold_val[feature_cols]
    y_val = fold_val["future_return"]

    # LightGBM
    lgb_config = LGBConfig(seed=config.seed)
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_valid = lgb.Dataset(X_val, y_val, reference=lgb_train)

    lgb_model = lgb.train(
        lgb_config.to_dict(),
        lgb_train,
        num_boost_round=config.max_boost_rounds,
        valid_sets=[lgb_valid],
        callbacks=[
            lgb.early_stopping(config.early_stopping_rounds),
            lgb.log_evaluation(LOG_EVAL_PERIOD),
        ],
    )
    lgb_pred = lgb_model.predict(X_val)

    # XGBoost
    xgb_config = XGBConfig(seed=config.seed)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    xgb_model = xgb.train(
        xgb_config.to_dict(),
        dtrain,
        num_boost_round=config.max_boost_rounds,
        evals=[(dval, "val")],
        early_stopping_rounds=config.early_stopping_rounds,
        verbose_eval=LOG_EVAL_PERIOD,
    )
    xgb_pred = xgb_model.predict(dval)

    return lgb_pred, xgb_pred


def _train_meta_learner(
    oof_predictions: np.ndarray,
    oof_labels: np.ndarray,
    stacking_config: StackingConfig,
) -> Tuple[np.ndarray, float]:
    """
    训练 Ridge Regression meta-learner

    Returns:
        (weights [2], intercept)
    """
    from sklearn.linear_model import Ridge

    meta_model = Ridge(
        alpha=stacking_config.ridge_alpha,
        random_state=stacking_config.seed,
    )
    meta_model.fit(oof_predictions, oof_labels)

    weights = meta_model.coef_
    intercept = meta_model.intercept_

    logger.info(
        f"Meta-Learner 权重: LGB={weights[0]:.4f}, XGB={weights[1]:.4f}, "
        f"截距={intercept:.6f}"
    )

    # 与等权对比
    equal_weight_ic = _compute_ic(
        oof_predictions.mean(axis=1), oof_labels
    )
    stacking_pred = oof_predictions @ weights + intercept
    stacking_ic = _compute_ic(stacking_pred, oof_labels)

    logger.info(
        f"OOF IC 对比: 等权={equal_weight_ic:.4f}, "
        f"Stacking={stacking_ic:.4f} "
        f"(提升 {(stacking_ic - equal_weight_ic) / (abs(equal_weight_ic) + 1e-10) * 100:.1f}%)"
    )

    return weights, float(intercept)


def _train_base_learners_full(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    config: TrainingConfig,
    feature_cols: List[str],
) -> Tuple[lgb.Booster, xgb.Booster]:
    """在全量训练集上重新训练 base learners"""
    X_train = train_data[feature_cols]
    y_train = train_data["future_return"]
    X_val = val_data[feature_cols]
    y_val = val_data["future_return"]

    # LightGBM
    lgb_config = LGBConfig(seed=config.seed)
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_valid = lgb.Dataset(X_val, y_val, reference=lgb_train)

    logger.info("训练最终 LightGBM 模型...")
    lgb_model = lgb.train(
        lgb_config.to_dict(),
        lgb_train,
        num_boost_round=config.max_boost_rounds,
        valid_sets=[lgb_valid],
        callbacks=[
            lgb.early_stopping(config.early_stopping_rounds),
            lgb.log_evaluation(LOG_EVAL_PERIOD),
        ],
    )

    # XGBoost
    xgb_config = XGBConfig(seed=config.seed)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    logger.info("训练最终 XGBoost 模型...")
    xgb_model = xgb.train(
        xgb_config.to_dict(),
        dtrain,
        num_boost_round=config.max_boost_rounds,
        evals=[(dval, "val")],
        early_stopping_rounds=config.early_stopping_rounds,
        verbose_eval=LOG_EVAL_PERIOD,
    )

    return lgb_model, xgb_model


def _evaluate_oof(
    oof_predictions: np.ndarray,
    oof_labels: np.ndarray,
    meta_weights: np.ndarray,
    meta_intercept: float,
) -> Dict[str, float]:
    """评估 OOF 预测指标"""
    stacking_pred = oof_predictions @ meta_weights + meta_intercept
    equal_pred = oof_predictions.mean(axis=1)

    metrics = {
        "stacking_ic": _compute_ic(stacking_pred, oof_labels),
        "stacking_mse": float(np.mean((stacking_pred - oof_labels) ** 2)),
        "equal_weight_ic": _compute_ic(equal_pred, oof_labels),
        "equal_weight_mse": float(np.mean((equal_pred - oof_labels) ** 2)),
        "lgb_ic": _compute_ic(oof_predictions[:, 0], oof_labels),
        "xgb_ic": _compute_ic(oof_predictions[:, 1], oof_labels),
    }

    logger.info("=== OOF 指标汇总 ===")
    logger.info(f"  LGB IC:          {metrics['lgb_ic']:.4f}")
    logger.info(f"  XGB IC:          {metrics['xgb_ic']:.4f}")
    logger.info(f"  等权集成 IC:     {metrics['equal_weight_ic']:.4f}")
    logger.info(f"  Stacking IC:     {metrics['stacking_ic']:.4f}")

    return metrics


def _evaluate_stacking(
    lgb_model: lgb.Booster,
    xgb_model: xgb.Booster,
    meta_weights: np.ndarray,
    meta_intercept: float,
    val_data: pd.DataFrame,
    feature_cols: List[str],
) -> Dict[str, float]:
    """在验证集上评估 Stacking 集成"""
    X_val = val_data[feature_cols]
    y_val = val_data["future_return"].values

    lgb_pred = lgb_model.predict(X_val)
    xgb_pred = xgb_model.predict(xgb.DMatrix(X_val))

    # Stacking 预测
    meta_features = np.column_stack([lgb_pred, xgb_pred])
    stacking_pred = meta_features @ meta_weights + meta_intercept

    # 等权预测（对比基线）
    equal_pred = 0.5 * lgb_pred + 0.5 * xgb_pred

    metrics = {
        "stacking_ic": _compute_ic(stacking_pred, y_val),
        "stacking_mse": float(np.mean((stacking_pred - y_val) ** 2)),
        "equal_weight_ic": _compute_ic(equal_pred, y_val),
        "equal_weight_mse": float(np.mean((equal_pred - y_val) ** 2)),
    }

    logger.info("=== 验证集 Stacking vs 等权 ===")
    logger.info(
        f"  Stacking IC={metrics['stacking_ic']:.4f}, "
        f"MSE={metrics['stacking_mse']:.6f}"
    )
    logger.info(
        f"  等权     IC={metrics['equal_weight_ic']:.4f}, "
        f"MSE={metrics['equal_weight_mse']:.6f}"
    )

    return metrics


def save_stacking_model(
    result: StackingResult, model_dir: Path
) -> None:
    """保存 Stacking 模型（base learners + meta weights）"""
    model_dir.mkdir(parents=True, exist_ok=True)

    # 保存 base learners
    with open(model_dir / "lgb_model.pkl", "wb") as f:
        pickle.dump(result.lgb_model, f)
    with open(model_dir / "xgb_model.pkl", "wb") as f:
        pickle.dump(result.xgb_model, f)

    # 保存 meta-learner 参数
    meta_data = {
        "weights": result.meta_weights.tolist(),
        "intercept": result.meta_intercept,
    }
    with open(model_dir / "meta_learner.pkl", "wb") as f:
        pickle.dump(meta_data, f)

    logger.info(f"Stacking 模型已保存到 {model_dir}")
    logger.info(
        f"  Meta 权重: LGB={result.meta_weights[0]:.4f}, "
        f"XGB={result.meta_weights[1]:.4f}"
    )


def load_stacking_meta(model_dir: Path) -> Tuple[np.ndarray, float]:
    """加载 meta-learner 参数"""
    meta_path = model_dir / "meta_learner.pkl"
    if not meta_path.exists():
        logger.warning("meta_learner.pkl 不存在，回退到等权")
        return np.array([0.5, 0.5]), 0.0

    with open(meta_path, "rb") as f:
        meta_data = pickle.load(f)

    weights = np.array(meta_data["weights"])
    intercept = meta_data["intercept"]
    logger.info(
        f"加载 Meta-Learner: LGB={weights[0]:.4f}, "
        f"XGB={weights[1]:.4f}, 截距={intercept:.6f}"
    )
    return weights, intercept


def _compute_ic(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """计算信息系数（Pearson 相关系数）"""
    if len(predictions) < 2:
        return 0.0
    corr = np.corrcoef(predictions, actuals)[0, 1]
    return float(corr) if np.isfinite(corr) else 0.0
