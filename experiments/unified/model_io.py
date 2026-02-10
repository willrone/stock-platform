"""
模型保存与加载模块

使用原生格式保存 LightGBM/XGBoost 模型 + pickle 兼容
"""
import json
import pickle
from pathlib import Path
from typing import Any, Dict

import lightgbm as lgb
import xgboost as xgb
from loguru import logger


def save_models(
    lgb_model: lgb.Booster,
    xgb_model: xgb.Booster,
    model_dir: Path,
) -> None:
    """保存 LightGBM 和 XGBoost 模型（pickle 格式，兼容现有策略）"""
    model_dir.mkdir(parents=True, exist_ok=True)

    lgb_path = model_dir / "lgb_model.pkl"
    xgb_path = model_dir / "xgb_model.pkl"

    with open(lgb_path, "wb") as f:
        pickle.dump(lgb_model, f)
    logger.info(f"LightGBM 已保存: {lgb_path} ({lgb_path.stat().st_size / 1024:.1f} KB)")

    with open(xgb_path, "wb") as f:
        pickle.dump(xgb_model, f)
    logger.info(f"XGBoost 已保存: {xgb_path} ({xgb_path.stat().st_size / 1024:.1f} KB)")


def save_metadata(
    metadata: Dict[str, Any], model_dir: Path
) -> None:
    """保存训练元数据"""
    metadata_path = model_dir / "model_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info(f"元数据已保存: {metadata_path}")


def save_training_report(
    report: Dict[str, Any], model_dir: Path
) -> None:
    """保存训练报告"""
    report_path = model_dir / "training_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"训练报告已保存: {report_path}")
