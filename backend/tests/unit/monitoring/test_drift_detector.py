"""漂移检测器严重级别映射契约测试。"""

import threading
from datetime import datetime
from typing import Optional
from unittest.mock import MagicMock

import numpy as np
import pytest

from app.services.monitoring.drift_detector import (
    DriftDetectionUnavailableError,
    DriftDetector,
    DriftMethod,
    DriftSeverity,
    StatisticalDriftDetector,
)


@pytest.mark.parametrize(
    ("p_value", "expected"),
    [
        (None, DriftSeverity.NONE),
        (0.11, DriftSeverity.NONE),
        (0.08, DriftSeverity.LOW),
        (0.02, DriftSeverity.MEDIUM),
        (0.005, DriftSeverity.HIGH),
        (0.0005, DriftSeverity.CRITICAL),
    ],
)
def test_determine_severity_for_ks_test(
    p_value: Optional[float], expected: DriftSeverity
) -> None:
    """KS 检验应按 p 值区间映射严重程度。"""
    detector = StatisticalDriftDetector()

    severity = detector._determine_severity(
        drift_score=0.0,
        p_value=p_value,
        method=DriftMethod.KS_TEST,
        threshold=0.05,
    )

    assert severity == expected


@pytest.mark.parametrize(
    ("method", "drift_score", "threshold", "expected"),
    [
        (DriftMethod.PSI, 0.05, 0.05, DriftSeverity.NONE),
        (DriftMethod.PSI, 0.15, 0.05, DriftSeverity.LOW),
        (DriftMethod.PSI, 0.25, 0.05, DriftSeverity.MEDIUM),
        (DriftMethod.PSI, 0.45, 0.05, DriftSeverity.HIGH),
        (DriftMethod.PSI, 0.60, 0.05, DriftSeverity.CRITICAL),
        (DriftMethod.WASSERSTEIN, 0.05, 0.2, DriftSeverity.NONE),
        (DriftMethod.WASSERSTEIN, 0.15, 0.2, DriftSeverity.LOW),
        (DriftMethod.WASSERSTEIN, 0.30, 0.2, DriftSeverity.MEDIUM),
        (DriftMethod.WASSERSTEIN, 0.70, 0.2, DriftSeverity.HIGH),
        (DriftMethod.WASSERSTEIN, 1.20, 0.2, DriftSeverity.CRITICAL),
        (DriftMethod.JENSEN_SHANNON, 0.05, 0.05, DriftSeverity.NONE),
        (DriftMethod.JENSEN_SHANNON, 0.15, 0.05, DriftSeverity.LOW),
        (DriftMethod.JENSEN_SHANNON, 0.30, 0.05, DriftSeverity.MEDIUM),
        (DriftMethod.JENSEN_SHANNON, 0.50, 0.05, DriftSeverity.HIGH),
        (DriftMethod.JENSEN_SHANNON, 0.70, 0.05, DriftSeverity.CRITICAL),
    ],
)
def test_determine_severity_for_score_based_methods(
    method: DriftMethod,
    drift_score: float,
    threshold: float,
    expected: DriftSeverity,
) -> None:
    """分数型漂移方法应按阈值表映射严重程度。"""
    detector = StatisticalDriftDetector()

    severity = detector._determine_severity(
        drift_score=drift_score,
        p_value=None,
        method=method,
        threshold=threshold,
    )

    assert severity == expected


def test_detect_drift_raises_when_all_methods_are_unavailable(monkeypatch) -> None:
    detector = DriftDetector()
    model_id = "model"
    model_version = "v1"
    key = f"{model_id}_{model_version}"
    now = datetime.now()

    detector.reference_data[key] = {
        "features": np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]),
        "feature_names": ["feature_a", "feature_b"],
        "timestamp": now,
    }

    for i in range(25):
        detector.feature_data[key].append(
            {
                "timestamp": now,
                "features": {
                    "feature_a": float(i),
                    "feature_b": float(i + 1),
                },
                "prediction": None,
            }
        )

    monkeypatch.setattr(
        detector.statistical_detector,
        "detect_drift",
        MagicMock(side_effect=ImportError("漂移检测依赖 scipy，请先安装 scipy。")),
    )

    with pytest.raises(DriftDetectionUnavailableError, match="漂移检测不可用"):
        detector.detect_drift(model_id, model_version, methods=[DriftMethod.KS_TEST])


def test_detect_drift_raises_when_only_pca_is_requested_without_fit() -> None:
    detector = DriftDetector()
    model_id = "model"
    model_version = "v1"
    key = f"{model_id}_{model_version}"
    now = datetime.now()

    detector.reference_data[key] = {
        "features": np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]),
        "feature_names": ["feature_a", "feature_b"],
        "timestamp": now,
    }

    for i in range(25):
        detector.feature_data[key].append(
            {
                "timestamp": now,
                "features": {
                    "feature_a": float(i),
                    "feature_b": float(i + 1),
                },
                "prediction": None,
            }
        )

    with pytest.raises(
        DriftDetectionUnavailableError,
        match="PCA 重构检测器尚未完成参考数据拟合",
    ):
        detector.detect_drift(
            model_id,
            model_version,
            methods=[DriftMethod.PCA_RECONSTRUCTION],
        )


def test_detect_drift_raises_when_pca_is_requested_but_unavailable_even_if_ks_works(
    monkeypatch,
) -> None:
    detector = DriftDetector()
    model_id = "model"
    model_version = "v1"
    key = f"{model_id}_{model_version}"
    now = datetime.now()

    detector.reference_data[key] = {
        "features": np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]),
        "feature_names": ["feature_a", "feature_b"],
        "timestamp": now,
    }

    for i in range(25):
        detector.feature_data[key].append(
            {
                "timestamp": now,
                "features": {
                    "feature_a": float(i),
                    "feature_b": float(i + 1),
                },
                "prediction": None,
            }
        )

    monkeypatch.setattr(
        detector.statistical_detector,
        "detect_drift",
        MagicMock(return_value=(0.1, 0.2, DriftSeverity.NONE)),
    )

    with pytest.raises(
        DriftDetectionUnavailableError,
        match="PCA 重构检测器尚未完成参考数据拟合",
    ):
        detector.detect_drift(
            model_id,
            model_version,
            methods=[DriftMethod.KS_TEST, DriftMethod.PCA_RECONSTRUCTION],
        )


def test_detect_drift_does_not_deadlock_when_reference_data_is_missing() -> None:
    detector = DriftDetector()
    model_id = "fresh-model"
    model_version = "v1"
    now = datetime.now()

    for i in range(120):
        detector.add_sample(
            model_id,
            model_version,
            features={
                "feature_a": float(i),
                "feature_b": float(i + 1),
            },
            timestamp=now,
        )

    result_holder: dict[str, object] = {}

    def run_detection() -> None:
        result_holder["report"] = detector.detect_drift(
            model_id,
            model_version,
            methods=[DriftMethod.PSI],
        )

    thread = threading.Thread(target=run_detection, daemon=True)
    thread.start()
    thread.join(timeout=2)

    assert not thread.is_alive()
    assert result_holder["report"] is not None


def test_detect_drift_uses_model_specific_multivariate_state() -> None:
    detector = DriftDetector()
    model_id = "model"
    model_version = "v1"
    key = f"{model_id}_{model_version}"
    now = datetime.now()

    detector.reference_data[key] = {
        "features": np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]),
        "feature_names": ["feature_a", "feature_b"],
        "timestamp": now,
    }

    detector.multivariate_detectors["other_v1"] = MagicMock()

    for i in range(25):
        detector.feature_data[key].append(
            {
                "timestamp": now,
                "features": {
                    "feature_a": float(i),
                    "feature_b": float(i + 1),
                },
                "prediction": None,
            }
        )

    with pytest.raises(
        DriftDetectionUnavailableError,
        match="PCA 重构检测器尚未完成参考数据拟合",
    ):
        detector.detect_drift(
            model_id,
            model_version,
            methods=[DriftMethod.PCA_RECONSTRUCTION],
        )
