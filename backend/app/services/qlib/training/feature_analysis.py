"""
特征分析模块

包含特征重要性提取和特征相关性分析
"""

from typing import Any, Dict, Optional, Tuple, List
from datetime import datetime

import numpy as np
import pandas as pd
from loguru import logger

from .config import QlibModelType


async def _extract_feature_importance(
        self, model: Any, model_type: QlibModelType
    ) -> Optional[Dict[str, float]]:
        """提取特征重要性"""
        try:
            # 对于树模型，尝试获取特征重要性
            if model_type in [QlibModelType.LIGHTGBM, QlibModelType.XGBOOST]:
                if hasattr(model, "get_feature_importance"):
                    importance = model.get_feature_importance()
                    if isinstance(importance, dict):
                        return importance
                elif hasattr(model, "feature_importances_"):
                    # 假设有特征名称列表
                    feature_names = [
                        f"feature_{i}" for i in range(len(model.feature_importances_))
                    ]
                    return dict(zip(feature_names, model.feature_importances_))

            # 对于其他模型类型，返回None
            return None

        except Exception as e:
            logger.warning(f"提取特征重要性失败: {e}")
            return None

def _analyze_feature_correlations(self, dataset: pd.DataFrame) -> Dict[str, Any]:
        """分析特征与标签的相关性"""
        try:
            if dataset.empty:
                return {"error": "数据集为空"}

            data = dataset.copy()
            if "label" not in data.columns:
                close_col = None
                for col in ["$close", "close", "Close", "CLOSE"]:
                    if col in data.columns:
                        close_col = col
                        break
                if close_col is None:
                    return {"error": "缺少收盘价列，无法生成标签"}

                if isinstance(data.index, pd.MultiIndex):
                    data["label"] = (
                        data.groupby(level=0)[close_col]
                        .pct_change(periods=1)
                        .shift(-1)
                        .fillna(0)
                    )
                else:
                    data["label"] = (
                        data[close_col].pct_change(periods=1).shift(-1).fillna(0)
                    )

            numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()
            numeric_features = list(dict.fromkeys(numeric_features))
            if "label" in numeric_features:
                numeric_features.remove("label")

            if not numeric_features:
                return {"error": "没有数值特征"}

            target_correlations = {}
            for feature in numeric_features:
                series = data[feature]
                if isinstance(series, pd.DataFrame):
                    series = series.iloc[:, 0]
                corr = series.corr(data["label"])
                if isinstance(corr, pd.Series):
                    corr = corr.iloc[0]
                if not pd.isna(corr):
                    target_correlations[feature] = float(abs(corr))

            high_corr_pairs = []
            feature_corr_matrix = data[numeric_features].corr()
            for i in range(len(numeric_features)):
                for j in range(i + 1, len(numeric_features)):
                    corr = feature_corr_matrix.iloc[i, j]
                    if not pd.isna(corr) and abs(corr) > 0.8:
                        high_corr_pairs.append(
                            {
                                "feature1": numeric_features[i],
                                "feature2": numeric_features[j],
                                "correlation": float(corr),
                            }
                        )

            return {
                "target_correlations": target_correlations,
                "high_correlation_pairs": high_corr_pairs,
                "avg_target_correlation": float(
                    np.mean(list(target_correlations.values()))
                )
                if target_correlations
                else 0.0,
                "max_target_correlation": float(max(target_correlations.values()))
                if target_correlations
                else 0.0,
            }

        except Exception as e:
            logger.warning(f"特征相关性分析失败: {e}")
            return {"error": str(e)}

async def _save_qlib_model(
        self, model: Any, model_id: str, model_config: Dict[str, Any]
    ) -> str:
        """保存Qlib模型"""
        try:
            # 创建模型保存目录
            from app.core.config import settings

            models_dir = Path(settings.MODEL_STORAGE_PATH)
            models_dir.mkdir(parents=True, exist_ok=True)

            # 生成模型文件路径
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"{model_id}_qlib_{timestamp}"

            # 保存模型（使用pickle格式）
            import pickle

            model_path = models_dir / f"{model_filename}.pkl"

            with open(model_path, "wb") as f:
                pickle.dump(
                    {"model": model, "config": model_config, "timestamp": timestamp}, f
                )

            logger.info(f"Qlib模型保存成功: {model_path}")
            return str(model_path)

        except Exception as e:
            logger.error(f"保存Qlib模型失败: {e}")
            raise

async def load_qlib_model(self, model_path: str) -> Tuple[Any, Dict[str, Any]]:
        """加载Qlib模型"""
        try:
            import pickle

            with open(model_path, "rb") as f:
                model_data = pickle.load(f)

            model = model_data["model"]
            config = model_data["config"]

            logger.info(f"Qlib模型加载成功: {model_path}")
            return model, config

        except Exception as e:
            logger.error(f"加载Qlib模型失败: {e}")
            raise

async def predict_with_qlib_model(
        self,
        model_path: str,
        stock_codes: List[str],
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """使用Qlib模型进行预测"""
        try:
            # 加载模型
            model, config = await self.load_qlib_model(model_path)

            # 准备预测数据
            dataset = await self.data_provider.prepare_qlib_dataset(
                stock_codes=stock_codes,
                start_date=start_date,
                end_date=end_date,
                include_alpha_factors=True,
                use_cache=True,
            )

            if dataset.empty:
                raise ValueError("无法获取预测数据")

            if isinstance(dataset, pd.DataFrame):
                dataset = self._align_prediction_features(model, dataset)
                base_model = model.model if hasattr(model, "model") else model
                feature_names = None
                if hasattr(base_model, "feature_name"):
                    try:
                        feature_names = base_model.feature_name()
                    except Exception:
                        feature_names = None
                if feature_names is None and hasattr(base_model, "feature_name_"):
                    feature_names = list(base_model.feature_name_)
                if feature_names:
                    missing_count = sum(
                        1 for name in feature_names if name not in dataset.columns
                    )
                    logger.info(
                        "预测特征对齐: model_features={}, dataset_features={}, missing_filled={}",
                        len(feature_names),
                        len(dataset.columns),
                        missing_count,
                    )

                class DataFrameDatasetAdapter:
                    """将DataFrame适配为qlib DatasetH格式（用于预测）"""

                    def __init__(self, data: pd.DataFrame):
                        self.data = data
                        self.segments = {"test": data}

                    def prepare(
                        self,
                        key: str,
                        col_set: Union[List[str], str] = None,
                        data_key: str = None,
                    ):
                        if col_set is None:
                            col_set = ["feature"]
                        if isinstance(col_set, str):
                            col_set = [col_set]

                        feature_cols = [
                            col for col in self.data.columns if col != "label"
                        ]

                        class FeatureSeries:
                            def __init__(self, feature_array_2d, index):
                                self._feature_array_2d = feature_array_2d
                                self._index = index

                            @property
                            def values(self):
                                return self._feature_array_2d

                            @property
                            def index(self):
                                return self._index

                            def __len__(self):
                                return len(self._feature_array_2d)

                            def __getitem__(self, key):
                                if isinstance(key, (int, np.integer)):
                                    return self._feature_array_2d[key]
                                if isinstance(key, slice):
                                    return self._feature_array_2d[key]
                                if hasattr(self._index, "get_loc"):
                                    loc = self._index.get_loc(key)
                                    return self._feature_array_2d[loc]
                                return self._feature_array_2d[key]

                            def __iter__(self):
                                return iter(self._feature_array_2d)

                            def __array__(self, dtype=None):
                                return (
                                    self._feature_array_2d
                                    if dtype is None
                                    else self._feature_array_2d.astype(dtype)
                                )

                        if "feature" in col_set:
                            feature_array = (
                                self.data[feature_cols].values
                                if feature_cols
                                else np.zeros((len(self.data), 0))
                            )
                            return FeatureSeries(feature_array, self.data.index)

                        if "label" in col_set:
                            label_values = (
                                self.data["label"].values
                                if "label" in self.data.columns
                                else np.zeros(len(self.data))
                            )
                            return label_values.reshape(-1, 1)

                        return self.data

                    def __getattr__(self, name):
                        return getattr(self.data, name)

                dataset = DataFrameDatasetAdapter(dataset)

            # 进行预测
            predictions = model.predict(dataset)

            logger.info(f"Qlib模型预测完成: {len(predictions)} 条预测结果")
            return predictions

        except Exception as e:
            logger.error(f"Qlib模型预测失败: {e}")
            raise

