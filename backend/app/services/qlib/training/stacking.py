"""
Stacking 集成模块 — Qlib 训练引擎版

移植自 experiments/unified/stacking.py，适配 Qlib 训练引擎。

架构:
  Layer 1 (Base Learners): LGBModel + XGBModel (Qlib 封装)
  Layer 2 (Meta-Learner):  Ridge Regression

流程:
  1. 用 Purged K-Fold 生成 out-of-fold 预测
  2. 拼接 OOF 预测作为 meta 特征
  3. 训练 Ridge meta-learner
  4. 最终预测 = meta_learner(lgb_pred, xgb_pred)
"""

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from .config import QlibTrainingConfig
from .purged_cv import PurgedCVConfig, PurgedGroupTimeSeriesSplit

# === 常量 ===
DEFAULT_RIDGE_ALPHA = 1.0
RANDOM_SEED = 42


@dataclass
class StackingConfig:
    """Stacking 集成配置"""

    ridge_alpha: float = DEFAULT_RIDGE_ALPHA
    cv_config: PurgedCVConfig = field(default_factory=PurgedCVConfig)
    seed: int = RANDOM_SEED


@dataclass
class StackingResult:
    """Stacking 训��结果"""

    lgb_model: Any  # Qlib LGBModel
    xgb_model: Any  # Qlib XGBModel
    meta_weights: np.ndarray
    meta_intercept: float
    oof_metrics: Dict[str, float]
    final_metrics: Dict[str, float]


def train_stacking_ensemble(
    dataset: Any,
    config: "QlibTrainingConfig",
    stacking_config: Optional[StackingConfig] = None,
) -> StackingResult:
    """
    训练 Stacking 集成模型（Qlib 引擎版）

    Args:
        dataset: DataFrameDatasetAdapter（含 train/valid segments）
        config: Qlib 训练配置
        stacking_config: Stacking 专用配置

    Returns:
        StackingResult
    """
    if stacking_config is None:
        stacking_config = StackingConfig()

    train_data = _extract_dataframe(dataset, "train")
    val_data = _extract_dataframe(dataset, "valid")

    feature_cols = [c for c in train_data.columns if c != "label"]
    label_col = "label"

    # Step 1: OOF 预测
    logger.info("=== Stacking Step 1: 生成 OOF 预测 ===")
    oof_preds, oof_labels = _generate_oof_predictions(
        train_data,
        feature_cols,
        label_col,
        stacking_config,
    )

    # Step 2: Meta-Learner
    logger.info("=== Stacking Step 2: 训练 Ridge Meta-Learner ===")
    weights, intercept = _train_meta_learner(
        oof_preds,
        oof_labels,
        stacking_config,
    )

    # Step 3: 全量训练 base learners
    logger.info("=== Stacking Step 3: 全量训练 Base Learners ===")
    lgb_model, xgb_model = _train_base_learners(
        train_data,
        val_data,
        feature_cols,
        label_col,
    )

    # Step 4: 评估
    logger.info("=== Stacking Step 4: 评估 ===")
    oof_metrics = _evaluate_oof(
        oof_preds,
        oof_labels,
        weights,
        intercept,
    )
    final_metrics = _evaluate_final(
        lgb_model,
        xgb_model,
        weights,
        intercept,
        val_data,
        feature_cols,
        label_col,
    )

    return StackingResult(
        lgb_model=lgb_model,
        xgb_model=xgb_model,
        meta_weights=weights,
        meta_intercept=intercept,
        oof_metrics=oof_metrics,
        final_metrics=final_metrics,
    )


def predict_stacking(
    result: StackingResult,
    data: pd.DataFrame,
) -> np.ndarray:
    """Stacking 集成预测"""
    feature_cols = [c for c in data.columns if c != "label"]
    X = data[feature_cols]

    lgb_pred = result.lgb_model.predict(X)
    xgb_pred = result.xgb_model.predict(X)

    meta_features = np.column_stack([lgb_pred, xgb_pred])
    return meta_features @ result.meta_weights + result.meta_intercept


# === 内部函数 ===


def _extract_dataframe(
    dataset: Any,
    segment: str,
) -> pd.DataFrame:
    """从 DatasetAdapter 中提取 DataFrame"""
    if hasattr(dataset, "data") and isinstance(dataset.data, pd.DataFrame):
        return dataset.data
    if hasattr(dataset, "prepare"):
        return dataset.prepare(segment)
    raise ValueError(f"无法从 dataset 提取 {segment} 数据")


def _generate_oof_predictions(
    train_data: pd.DataFrame,
    feature_cols: List[str],
    label_col: str,
    stacking_config: StackingConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """用 Purged K-Fold 生成 OOF 预测"""
    import lightgbm as lgb
    import xgboost as xgb

    splitter = PurgedGroupTimeSeriesSplit(stacking_config.cv_config)
    all_preds: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    for fold_idx, (fold_train, fold_val) in enumerate(
        splitter.split(train_data),
    ):
        X_train = fold_train[feature_cols].values
        y_train = fold_train[label_col].values
        X_val = fold_val[feature_cols].values
        y_val = fold_val[label_col].values

        # LightGBM
        lgb_ds = lgb.Dataset(X_train, y_train)
        lgb_val_ds = lgb.Dataset(X_val, y_val, reference=lgb_ds)
        lgb_model = lgb.train(
            _lgb_params(stacking_config.seed),
            lgb_ds,
            num_boost_round=1000,
            valid_sets=[lgb_val_ds],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )
        lgb_pred = lgb_model.predict(X_val)

        # XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        xgb_model = xgb.train(
            _xgb_params(stacking_config.seed),
            dtrain,
            num_boost_round=1000,
            evals=[(dval, "val")],
            early_stopping_rounds=50,
            verbose_eval=0,
        )
        xgb_pred = xgb_model.predict(dval)

        fold_oof = np.column_stack([lgb_pred, xgb_pred])
        all_preds.append(fold_oof)
        all_labels.append(y_val)

        lgb_ic = _compute_ic(lgb_pred, y_val)
        xgb_ic = _compute_ic(xgb_pred, y_val)
        logger.info(
            f"  Fold {fold_idx} OOF IC: LGB={lgb_ic:.4f}, XGB={xgb_ic:.4f}",
        )

    return np.vstack(all_preds), np.concatenate(all_labels)


def _train_meta_learner(
    oof_preds: np.ndarray,
    oof_labels: np.ndarray,
    stacking_config: StackingConfig,
) -> Tuple[np.ndarray, float]:
    """训练 Ridge meta-learner"""
    from sklearn.linear_model import Ridge

    model = Ridge(
        alpha=stacking_config.ridge_alpha,
        random_state=stacking_config.seed,
    )
    model.fit(oof_preds, oof_labels)

    weights = model.coef_
    intercept = float(model.intercept_)

    logger.info(
        f"Meta-Learner 权重: LGB={weights[0]:.4f}, "
        f"XGB={weights[1]:.4f}, 截距={intercept:.6f}",
    )
    return weights, intercept


def _train_base_learners(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    feature_cols: List[str],
    label_col: str,
) -> Tuple[Any, Any]:
    """在全量训练集上训练 base learners"""
    import lightgbm as lgb
    import xgboost as xgb

    X_train = train_data[feature_cols].values
    y_train = train_data[label_col].values
    X_val = val_data[feature_cols].values
    y_val = val_data[label_col].values

    # LightGBM
    lgb_ds = lgb.Dataset(X_train, y_train)
    lgb_val_ds = lgb.Dataset(X_val, y_val, reference=lgb_ds)
    lgb_model = lgb.train(
        _lgb_params(RANDOM_SEED),
        lgb_ds,
        num_boost_round=1000,
        valid_sets=[lgb_val_ds],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(200)],
    )

    # XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    xgb_model = xgb.train(
        _xgb_params(RANDOM_SEED),
        dtrain,
        num_boost_round=1000,
        evals=[(dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=200,
    )

    return lgb_model, xgb_model


def _lgb_params(seed: int) -> Dict[str, Any]:
    """Qlib 官方基准 LightGBM 参数"""
    return {
        "objective": "regression",
        "metric": "mse",
        "learning_rate": 0.0421,
        "num_leaves": 210,
        "max_depth": 8,
        "feature_fraction": 0.8879,
        "bagging_fraction": 0.8789,
        "bagging_freq": 1,
        "lambda_l1": 205.6999,
        "lambda_l2": 580.9768,
        "verbose": -1,
        "seed": seed,
    }


def _xgb_params(seed: int) -> Dict[str, Any]:
    """XGBoost 参数"""
    return {
        "objective": "reg:squarederror",
        "eta": 0.1,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": seed,
        "verbosity": 0,
    }


def _evaluate_oof(
    oof_preds: np.ndarray,
    oof_labels: np.ndarray,
    weights: np.ndarray,
    intercept: float,
) -> Dict[str, float]:
    """评估 OOF 指标"""
    stacking_pred = oof_preds @ weights + intercept
    equal_pred = oof_preds.mean(axis=1)

    metrics = {
        "stacking_ic": _compute_ic(stacking_pred, oof_labels),
        "equal_weight_ic": _compute_ic(equal_pred, oof_labels),
        "lgb_ic": _compute_ic(oof_preds[:, 0], oof_labels),
        "xgb_ic": _compute_ic(oof_preds[:, 1], oof_labels),
    }
    logger.info(
        f"OOF IC: LGB={metrics['lgb_ic']:.4f}, "
        f"XGB={metrics['xgb_ic']:.4f}, "
        f"等权={metrics['equal_weight_ic']:.4f}, "
        f"Stacking={metrics['stacking_ic']:.4f}",
    )
    return metrics


def _evaluate_final(
    lgb_model: Any,
    xgb_model: Any,
    weights: np.ndarray,
    intercept: float,
    val_data: pd.DataFrame,
    feature_cols: List[str],
    label_col: str,
) -> Dict[str, float]:
    """在验证集上评估"""
    import xgboost as xgb

    X_val = val_data[feature_cols].values
    y_val = val_data[label_col].values

    lgb_pred = lgb_model.predict(X_val)
    xgb_pred = xgb_model.predict(xgb.DMatrix(X_val))

    meta = np.column_stack([lgb_pred, xgb_pred])
    stacking_pred = meta @ weights + intercept
    equal_pred = 0.5 * lgb_pred + 0.5 * xgb_pred

    metrics = {
        "stacking_ic": _compute_ic(stacking_pred, y_val),
        "equal_weight_ic": _compute_ic(equal_pred, y_val),
    }
    logger.info(
        f"验证集 IC: Stacking={metrics['stacking_ic']:.4f}, "
        f"等权={metrics['equal_weight_ic']:.4f}",
    )
    return metrics


def save_stacking_model(
    result: StackingResult,
    model_dir: Path,
) -> None:
    """保存 Stacking 模型"""
    model_dir.mkdir(parents=True, exist_ok=True)

    with open(model_dir / "lgb_model.pkl", "wb") as f:
        pickle.dump(result.lgb_model, f)
    with open(model_dir / "xgb_model.pkl", "wb") as f:
        pickle.dump(result.xgb_model, f)

    meta = {
        "weights": result.meta_weights.tolist(),
        "intercept": result.meta_intercept,
    }
    with open(model_dir / "meta_learner.pkl", "wb") as f:
        pickle.dump(meta, f)

    logger.info(f"Stacking 模型已保存到 {model_dir}")


def load_stacking_meta(
    model_dir: Path,
) -> Tuple[np.ndarray, float]:
    """加载 meta-learner 参数"""
    meta_path = model_dir / "meta_learner.pkl"
    if not meta_path.exists():
        logger.warning("meta_learner.pkl 不存在，回退到等权")
        return np.array([0.5, 0.5]), 0.0

    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    weights = np.array(meta["weights"])
    intercept = meta["intercept"]
    return weights, intercept


def _compute_ic(
    predictions: np.ndarray,
    actuals: np.ndarray,
) -> float:
    """计算信息系数（Pearson 相关系数）"""
    if len(predictions) < 2:
        return 0.0
    corr = np.corrcoef(predictions, actuals)[0, 1]
    return float(corr) if np.isfinite(corr) else 0.0
