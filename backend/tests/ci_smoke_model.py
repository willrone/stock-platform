"""CI 冒烟模型（只在 CI 播种用，不是生产模型）。

背景：`app/core/database.py::_seed_ci_smoke_models_sync` 会在 CI（GITHUB_ACTIONS=true）
往 model_info 里插一行 `xgboost_v1`，file_path 指向 `data/models/xgboost_v1.joblib`。
但 CI 的干净 checkout 里没有这个文件 → prediction_engine 走到
「model_info 有记录 → 文件不存在 → PredictionError」这条分支，
把 `tests/unit/prediction/test_prediction_engine_properties.py` 的属性测试打红。

这个类存在的目的只有一个：让被播种的那行数据库记录**真的有一个可加载的模型文件**，
并且接口对特征宽度不敏感（CI 里没有真实特征面板，宽度说不准）。
它不做任何预测意义上的事情，别把它当模型用，也不许出现在生产路径。
"""

from __future__ import annotations


class CiSmokeModel:
    """predict / predict_proba 都在，且不挑特征宽度。"""

    @staticmethod
    def _n(X: object) -> int:
        try:
            return int(len(X))  # type: ignore[arg-type]
        except TypeError:
            return 1

    def predict(self, X: object):
        return [0.0] * self._n(X)

    def predict_proba(self, X: object):
        return [[0.5, 0.5] for _ in range(self._n(X))]
