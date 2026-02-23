#!/usr/bin/env python3
"""
统一 ML 训练 Pipeline - 主入口

合并独立训练脚本（62 个手工特征）和 Qlib 引擎（数据预处理流程）

改进内容：
1. 统一训练体系：合并两套 Pipeline
2. 切换回归目标：从二分类改为回归（预测收益率）
3. 增加 Embargo 期：训练/验证集之间 20 天缓冲
4. 市场中性化：截面去均值，降低 market_up_ratio 依赖
"""
import sys
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb
from loguru import logger

from unified.config import LGBConfig, TrainingConfig, XGBConfig
from unified.constants import RANDOM_SEED
from unified.data_loader import prepare_dataset, split_with_embargo
from unified.evaluation import backtest_by_ranking, evaluate_on_test
from unified.features import get_feature_columns
from unified.model_io import save_metadata, save_models, save_training_report
from unified.trainer import predict_ensemble, train_lightgbm, train_xgboost

warnings.filterwarnings("ignore")
np.random.seed(RANDOM_SEED)


def build_test_dataframe(
    test: pd.DataFrame, predictions: np.ndarray
) -> pd.DataFrame:
    """构建回测用的 DataFrame"""
    return pd.DataFrame({
        "date": test["date"].values,
        "date_str": test["date"].astype(str).values,
        "ts_code": test["ts_code"].values,
        "actual_return": test["future_return"].values,
        "pred": predictions,
    })


def build_metadata(
    config: TrainingConfig,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    metrics: dict,
) -> dict:
    """构建训练元数据"""
    return {
        "pipeline": "unified_regression_v1",
        "trained_at": datetime.now().isoformat(),
        "improvements": [
            "unified_pipeline",
            "regression_objective",
            "embargo_period",
            "market_neutralization",
        ],
        "config": {
            "n_stocks": config.n_stocks,
            "embargo_days": config.embargo_days,
            "neutralization": config.enable_neutralization,
            "lgb_objective": LGBConfig().objective,
            "xgb_objective": XGBConfig().objective,
        },
        "data_splits": {
            "train": f"{train['date'].min().date()} ~ {train['date'].max().date()} ({len(train)})",
            "val": f"{val['date'].min().date()} ~ {val['date'].max().date()} ({len(val)})",
            "test": f"{test['date'].min().date()} ~ {test['date'].max().date()} ({len(test)})",
        },
        "feature_cols": get_feature_columns(),
        "n_features": len(get_feature_columns()),
        "metrics": metrics,
    }


def build_training_report(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    lgb_metrics: dict,
    xgb_metrics: dict,
    ensemble_metrics: dict,
    bt_results: dict,
    duration: float,
) -> dict:
    """构建训练报告"""
    return {
        "training_time_seconds": round(duration, 1),
        "data_info": {
            "train_samples": len(train),
            "valid_samples": len(val),
            "test_samples": len(test),
            "feature_count": len(get_feature_columns()),
        },
        "test_metrics": {
            "lightgbm": lgb_metrics,
            "xgboost": xgb_metrics,
            "ensemble": ensemble_metrics,
        },
        "backtest_simulation": bt_results,
    }


def run_pipeline() -> dict:
    """执行完整训练 Pipeline"""
    logger.info("=" * 60)
    logger.info("统一 ML 训练 Pipeline（回归 + Embargo + 中性化）")
    logger.info("=" * 60)

    start_time = time.time()
    config = TrainingConfig()

    # 1. 数据准备
    data = prepare_dataset(config)

    # 2. 带 Embargo 的时间分割
    train, val, test = split_with_embargo(data, config)

    # 3. 训练模型
    lgb_model, lgb_val_metrics = train_lightgbm(train, val, config)
    xgb_model, xgb_val_metrics = train_xgboost(train, val, config)

    # 4. 测试集评估
    feature_cols = get_feature_columns()
    lgb_test_metrics = evaluate_on_test(
        lgb_model.predict(test[feature_cols]),
        test["future_return"],
        "LightGBM",
    )
    xgb_test_metrics = evaluate_on_test(
        xgb_model.predict(xgb.DMatrix(test[feature_cols])),
        test["future_return"],
        "XGBoost",
    )

    ensemble_pred = predict_ensemble(lgb_model, xgb_model, test)
    ensemble_metrics = evaluate_on_test(
        ensemble_pred, test["future_return"], "Ensemble"
    )

    # 5. 回测
    test_df = build_test_dataframe(test, ensemble_pred)
    bt_results = backtest_by_ranking(test_df)

    # 6. 保存
    duration = time.time() - start_time
    save_models(lgb_model, xgb_model, config.model_dir)

    all_metrics = {
        "lgb_test": lgb_test_metrics,
        "xgb_test": xgb_test_metrics,
        "ensemble_test": ensemble_metrics,
        "backtest": bt_results,
    }
    metadata = build_metadata(config, train, val, test, all_metrics)
    save_metadata(metadata, config.model_dir)

    report = build_training_report(
        train, val, test,
        lgb_test_metrics, xgb_test_metrics, ensemble_metrics,
        bt_results, duration,
    )
    save_training_report(report, config.model_dir)

    logger.info(f"\n训练完成！总耗时: {duration:.1f} 秒")
    return report


if __name__ == "__main__":
    run_pipeline()
