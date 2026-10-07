"""
概率校准模块

为分类模型提供概率校准功能，确保模型输出的概率反映真实可能性。
支持 Platt Scaling (Sigmoid) 和 Isotonic Regression 两种校准方法。

参考：
- sklearn.calibration.CalibratedClassifierCV
- "Predicting Good Probabilities with Supervised Learning" (Platt, 1999)
- De Prado, "Advances in Financial Machine Learning", Ch. 12
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

logger = logging.getLogger(__name__)


def calibrate_model(
    base_model: Any,
    X_calibration: np.ndarray,
    y_calibration: np.ndarray,
    method: str = "isotonic",
    cv: int = 5,
    ensemble: bool = True,
) -> Any:
    """
    对训练好的模型进行概率校准

    Args:
        base_model: 已训练的分类模型（需支持 predict 或 decision_function）
        X_calibration: 校准集特征
        y_calibration: 校准集标签
        method: 校准方法 — 'isotonic'(默认), 'sigmoid'(Platt Scaling), 'none'
        cv: 校准交叉验证折数
        ensemble: 是否使用 CV 集成校准（True=CalibratedClassifierCV, False=单次校准）

    Returns:
        校准后的模型对象（含 predict_proba 方法）
    """
    if method == "none":
        logger.info("跳过概率校准 (method=none)")
        return base_model

    if not hasattr(base_model, "predict") and not hasattr(
        base_model, "decision_function"
    ):
        logger.warning("模型不支持 predict 或 decision_function，无法校准")
        return base_model

    if len(X_calibration) < cv:
        logger.warning(
            f"校准样本数 ({len(X_calibration)}) 小于 cv 折数 ({cv})，回退为单次校准"
        )
        ensemble = False

    sklearn_method = "sigmoid" if method == "sigmoid" else "isotonic"

    try:
        if ensemble:
            # 使用 CalibratedClassifierCV（交叉验证集成校准）
            calibrator = CalibratedClassifierCV(
                base_model, method=sklearn_method, cv=cv if cv > 1 else 3
            )
            calibrator.fit(X_calibration, y_calibration)
            logger.info(
                f"CalibratedClassifierCV 校准完成 (method={sklearn_method}, cv={cv})"
            )
            return calibrator
        else:
            # 单次校准
            if sklearn_method == "sigmoid":
                # Platt Scaling
                cal_probas = base_model.predict_proba(X_calibration)[:, 1:2]
                lr = LogisticRegression(C=1e10, solver="lbfgs")
                lr.fit(cal_probas, y_calibration)
                logger.info("Platt Scaling 校准完成 (单次)")
                return _PlattCalibrator(base_model, lr)
            else:
                # Isotonic Regression
                cal_scores = base_model.predict_proba(X_calibration)[:, 1]
                iso_reg = IsotonicRegression(out_of_bounds="clip")
                iso_reg.fit(cal_scores, y_calibration)
                logger.info("Isotonic Regression 校准完成 (单次)")
                return _IsotonicCalibrator(base_model, iso_reg)

    except Exception as e:
        logger.error(f"概率校准失败: {e}，返回原始模型")
        return base_model


def evaluate_calibration(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    """
    评估校准质量

    Args:
        model: 校准后的模型（需有 predict_proba）
        X_test: 测试特征
        y_test: 测试标签

    Returns:
        校准质量指标字典
    """
    result: Dict[str, float] = {}

    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test)
            # 如果是校准器，取正类概率列
            if proba.shape[1] >= 2:
                y_prob = proba[:, 1]
            else:
                y_prob = proba[:, 0]
        else:
            return result

        # Brier Score (越小越好，完美=0)
        result["brier_score"] = float(brier_score_loss(y_test, y_prob))

        # Log Loss (越小越好)
        result["log_loss"] = float(log_loss(y_test, y_prob))

        # 校准器通过分类准确率
        y_pred = (y_prob >= 0.5).astype(int)
        result["calibrated_accuracy"] = float(np.mean(y_pred == y_test))

        logger.info(
            f"校准评估: Brier={result['brier_score']:.4f}, "
            f"LogLoss={result['log_loss']:.4f}, "
            f"Acc={result['calibrated_accuracy']:.4f}"
        )

    except Exception as e:
        logger.warning(f"校准评估失败: {e}")

    return result


# --- 辅助包装类（单次校准，不依赖 sklearn CalibratedClassifierCV）---


class _PlattCalibrator:
    """Platt Scaling 包装器"""

    def __init__(self, base_model: Any, platt_model: LogisticRegression) -> None:
        self.base_model = base_model
        self.platt_model = platt_model

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.base_model.predict(X)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return self.base_model.decision_function(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        base_proba = self.base_model.predict_proba(X)
        sigmoid_input = (
            base_proba[:, 1:2] if base_proba.shape[1] >= 2 else base_proba[:, 0:1]
        )
        calibrated = self.platt_model.predict_proba(sigmoid_input)
        return calibrated


class _IsotonicCalibrator:
    """Isotonic Regression 包装器"""

    def __init__(self, base_model: Any, iso_reg: IsotonicRegression) -> None:
        self.base_model = base_model
        self.iso_reg = iso_reg

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.base_model.predict(X)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return self.base_model.decision_function(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        base_proba = self.base_model.predict_proba(X)
        raw_scores = base_proba[:, 1] if base_proba.shape[1] >= 2 else base_proba[:, 0]
        calibrated = self.iso_reg.transform(raw_scores)
        # 构造两列输出 [1-p, p]
        result = np.column_stack([1.0 - calibrated, calibrated])
        return result
