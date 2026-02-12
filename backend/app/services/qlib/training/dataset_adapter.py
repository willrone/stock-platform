"""
数据集适配器模块

将DataFrame适配为Qlib DatasetH格式
"""

from typing import Any, List, Union

import numpy as np
import pandas as pd
from loguru import logger

from .config import QlibTrainingConfig

DEFAULT_BINARY_THRESHOLD = 0.003


def _create_label_for_data(
    data: pd.DataFrame,
    data_name: str,
    horizon: int,
    label_type: str = "regression",
    binary_threshold: float = DEFAULT_BINARY_THRESHOLD,
) -> None:
    """为数据集创建标签（原地修改）

    Args:
        data: 数据集 DataFrame
        data_name: 数据集名称（用于日志）
        horizon: 预测周期（天）
        label_type: 标签类型 "regression" 或 "binary"
        binary_threshold: 二分类阈值（仅 binary 时生效）
    """
    if data is None or "label" in data.columns:
        return

    close_col = _find_close_column(data)
    if close_col is not None:
        _create_return_label(data, close_col, horizon)
    else:
        _create_fallback_label(data, data_name)
        return

    # 二分类转换
    if label_type == "binary":
        positive_before = data["label"].mean()
        data["label"] = (data["label"] > binary_threshold).astype(int)
        logger.info(
            f"{data_name}二分类标签: 阈值={binary_threshold}, "
            f"正样本比例={data['label'].mean():.4f}",
        )
    else:
        logger.info(
            f"{data_name}自动创建标签列（未来{horizon}天收益率），"
            f"范围=[{data['label'].min():.6f}, {data['label'].max():.6f}]",
        )


def _find_close_column(data: pd.DataFrame) -> str:
    """查找收盘价列名"""
    for col in ["$close", "close", "Close", "CLOSE"]:
        if col in data.columns:
            return col
    return None


def _create_return_label(
    data: pd.DataFrame, close_col: str, horizon: int,
) -> None:
    """计算未来N天收益率标签（原地修改）"""
    current_price = data[close_col]
    if isinstance(data.index, pd.MultiIndex):
        future_price = data.groupby(level=0)[close_col].shift(-horizon)
    else:
        future_price = data[close_col].shift(-horizon)

    label_values = (future_price - current_price) / current_price
    if isinstance(label_values, pd.Series):
        data["label"] = label_values.fillna(0)
    else:
        data["label"] = pd.Series(
            label_values.iloc[:, 0].values
            if hasattr(label_values, "iloc")
            else label_values,
            index=data.index,
        ).fillna(0)


def _create_fallback_label(
    data: pd.DataFrame, data_name: str,
) -> None:
    """使用最后一列作为标签（原地修改）"""
    last_col = data.iloc[:, -1]
    if isinstance(last_col, pd.Series):
        data["label"] = last_col
    else:
        data["label"] = pd.Series(
            last_col.iloc[:, 0].values
            if hasattr(last_col, "iloc")
            else last_col,
            index=data.index,
        )
    logger.warning(
        f"{data_name}未找到收盘价列，使用最后一列作为标签",
    )


# 创建DatasetH适配器，使DataFrame具有qlib DatasetH的接口
class DataFrameDatasetAdapter:
    """将DataFrame适配为qlib DatasetH格式"""

    def __init__(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame = None,
        prediction_horizon: int = 5,
        config: "QlibTrainingConfig" = None,
    ):
        self.train_data = train_data.copy()
        self.val_data = val_data.copy() if val_data is not None else None
        self.config = config
        # qlib模型期望有segments属性，包含train和valid
        self.segments = {"train": self.train_data}
        if self.val_data is not None:
            self.segments["valid"] = self.val_data
        # 为了兼容性，也设置data属性为训练集
        self.data = self.train_data

        # 处理训练集和验证集的标签
        label_type = config.label_type if config else "regression"
        binary_threshold = config.binary_threshold if config else 0.003

        _create_label_for_data(
            self.train_data, "训练集", prediction_horizon,
            label_type, binary_threshold,
        )
        _create_label_for_data(
            self.val_data, "验证集", prediction_horizon,
            label_type, binary_threshold,
        )

        # 记录数据维度信息
        logger.info(
            f"DataFrameDatasetAdapter初始化: 训练集形状={self.train_data.shape}, 验证集形状={self.val_data.shape if self.val_data is not None else 'N/A'}, 列数={len(self.train_data.columns)}"
        )
        if "label" in self.train_data.columns:
            label_stats = self.train_data["label"].describe()
            logger.info(f"训练集标签统计: {label_stats.to_dict()}")
        if self.val_data is not None and "label" in self.val_data.columns:
            val_label_stats = self.val_data["label"].describe()
            logger.info(f"验证集标签统计: {val_label_stats.to_dict()}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        if key == "train":
            return self.train_data
        elif key == "valid" and self.val_data is not None:
            return self.val_data
        return self.train_data

    def prepare(
        self,
        key,
        col_set: Union[List[str], str] = None,
        data_key: str = None,
    ):
        """实现qlib DatasetH的prepare方法

        支持 key 为字符串（如 "train"）或列表（如 ["train", "valid"]）。
        当 key 为列表时，返回对应数据集的元组，兼容 Qlib 的
        ``df_train, df_valid = dataset.prepare(["train", "valid"], ...)`` 用法。
        """
        if col_set is None:
            col_set = ["feature", "label"]

        # 处理col_set可能是字符串的情况（Qlib的predict传入"feature"字符串）
        if isinstance(col_set, str):
            col_set = [col_set]

        # 如果 key 是列表，递归调用并返回元组
        if isinstance(key, (list, tuple)):
            return tuple(
                self.prepare(k, col_set=col_set, data_key=data_key) for k in key
            )

        # 根据key选择对应的数据集
        if key == "train":
            data = self.train_data
        elif key == "valid" and self.val_data is not None:
            data = self.val_data
        else:
            data = self.train_data

        # 定义LabelSeries类（需要在方法开始处定义，以便在整个方法中可用）
        class LabelSeries:
            """包装Series，使values返回2D数组以满足qlib的要求"""

            def __init__(self, values_1d, values_2d, index):
                self._series = pd.Series(values_1d, index=index)
                self._values_2d = values_2d
                self._index = index

            @property
            def values(self):
                # 返回2D数组，满足qlib的检查: ndim == 2 and shape[1] == 1
                return self._values_2d

            @property
            def index(self):
                return self._index

            def __len__(self):
                return len(self._series)

            def __getitem__(self, key):
                return self._series[key]

            def __iter__(self):
                return iter(self._series)

            def __array__(self, dtype=None):
                # 支持numpy数组转换
                return (
                    self._values_2d
                    if dtype is None
                    else self._values_2d.astype(dtype)
                )

            def __getattr__(self, name):
                # 转发其他所有属性到内部的Series
                return getattr(self._series, name)

        # 分离特征和标签
        # 如果配置中指定了selected_features，则只使用选定的特征
        all_feature_cols = [col for col in data.columns if col != "label"]
        if hasattr(self, 'config') and self.config and self.config.selected_features:
            # 特征名称映射：将前端友好的名称转换为Qlib实际使用的名称
            def map_feature_name(feature_name: str) -> List[str]:
                """将前端特征名称映射到可能的Qlib特征名称"""
                # 基础特征映射（添加$前缀）
                base_mapping = {
                    "open": ["$open", "open"],
                    "high": ["$high", "high"],
                    "low": ["$low", "low"],
                    "close": ["$close", "close"],
                    "volume": ["$volume", "volume"],
                }

                # 技术指标映射（大小写和下划线变体）
                indicator_mapping = {
                    "ma_5": ["MA5", "ma_5", "MA_5"],
                    "ma_10": ["MA10", "ma_10", "MA_10"],
                    "ma_20": ["MA20", "ma_20", "MA_20"],
                    "ma_60": ["MA60", "ma_60", "MA_60"],
                    "sma": ["SMA", "sma"],
                    "ema": ["EMA", "EMA20", "ema"],
                    "rsi": ["RSI14", "RSI", "rsi", "rsi_14"],
                    "macd": ["MACD", "macd"],
                    "macd_signal": ["MACD_SIGNAL", "macd_signal", "MACD_SIGN"],
                    "macd_histogram": [
                        "MACD_HIST",
                        "MACD_HISTOGRAM",
                        "macd_histogram",
                    ],
                    "bb_upper": [
                        "BOLL_UPPER",
                        "BB_UPPER",
                        "bb_upper",
                        "bollinger_upper",
                    ],
                    "bb_middle": [
                        "BOLL_MIDDLE",
                        "BB_MIDDLE",
                        "bb_middle",
                        "bollinger_middle",
                    ],
                    "bb_lower": [
                        "BOLL_LOWER",
                        "BB_LOWER",
                        "bb_lower",
                        "bollinger_lower",
                    ],
                    "atr": ["ATR14", "ATR", "atr", "atr_14"],
                    "vwap": ["VWAP", "vwap"],
                    "obv": ["OBV", "obv"],
                    "stoch": ["STOCH_K", "STOCH", "stoch", "stoch_k"],
                    "kdj_k": ["KDJ_K", "kdj_k"],
                    "kdj_d": ["KDJ_D", "kdj_d"],
                    "kdj_j": ["KDJ_J", "kdj_j"],
                    "williams_r": ["WILLIAMS_R", "williams_r", "WILLIAMS"],
                    "cci": ["CCI20", "CCI", "cci"],
                    "momentum": ["MOMENTUM", "momentum"],
                    "roc": ["ROC", "roc"],
                    "sar": ["SAR", "sar"],
                    "adx": ["ADX", "adx"],
                    "volume_rsi": ["VOLUME_RSI", "volume_rsi"],
                }

                # 基本面特征映射
                fundamental_mapping = {
                    "price_change": ["RET1", "price_change", "PRICE_CHANGE"],
                    "price_change_5d": [
                        "RET5",
                        "price_change_5d",
                        "PRICE_CHANGE_5D",
                    ],
                    "price_change_20d": [
                        "RET20",
                        "price_change_20d",
                        "PRICE_CHANGE_20D",
                    ],
                    "volume_change": [
                        "VOLUME_RET1",
                        "volume_change",
                        "VOLUME_CHANGE",
                    ],
                    "volume_ma_ratio": ["VOLUME_MA_RATIO", "volume_ma_ratio"],
                    "volatility_5d": [
                        "VOLATILITY5",
                        "volatility_5d",
                        "VOLATILITY_5D",
                    ],
                    "volatility_20d": [
                        "VOLATILITY20",
                        "volatility_20d",
                        "VOLATILITY_20D",
                    ],
                    "price_position": ["PRICE_POSITION", "price_position"],
                }

                # 合并所有映射
                all_mapping = {
                    **base_mapping,
                    **indicator_mapping,
                    **fundamental_mapping,
                }

                # 查找映射
                if feature_name in all_mapping:
                    return all_mapping[feature_name]
                # 如果特征名本身已经匹配，直接返回
                return [feature_name]

            # 将前端特征名称映射到实际特征名称
            mapped_features = []
            selected_features = getattr(self, 'config', None) and self.config.selected_features or []
            for user_feature in selected_features:
                possible_names = map_feature_name(user_feature)
                # 查找第一个在数据中存在的特征名称
                found = False
                for possible_name in possible_names:
                    if possible_name in all_feature_cols:
                        mapped_features.append(possible_name)
                        found = True
                        break
                if not found:
                    logger.debug(
                        f"特征 '{user_feature}' 未找到匹配项，尝试的变体: {possible_names}"
                    )

            # 只选择用户指定的特征，且这些特征在数据中存在
            feature_cols = [
                col for col in mapped_features if col in all_feature_cols
            ]
            if len(feature_cols) == 0:
                logger.warning(
                    f"用户指定的特征都不存在，使用所有可用特征。指定特征: {selected_features}, 可用特征: {all_feature_cols[:20]}"
                )
                feature_cols = all_feature_cols
            else:
                missing_features = [
                    col
                    for col in selected_features
                    if col
                    not in [f for f in mapped_features if f in all_feature_cols]
                ]
                if missing_features:
                    logger.warning(f"以下特征不存在，将被忽略: {missing_features[:10]}")
                logger.info(
                    f"使用用户选择的 {len(feature_cols)} 个特征进行训练: {feature_cols[:10]}"
                )
        else:
            feature_cols = all_feature_cols

        # 创建一个包装类，使Series的values返回2D数组
        class FeatureSeries:
            """包装Series，使values返回2D数组"""

            def __init__(self, feature_array_2d, index):
                self._feature_array_2d = feature_array_2d
                self._index = index

            @property
            def values(self):
                # 返回2D数组，满足LightGBM的要求
                return self._feature_array_2d

            @property
            def index(self):
                return self._index

            def __len__(self):
                return len(self._feature_array_2d)

            def __getitem__(self, key):
                # 直接返回数组的对应行
                if isinstance(key, (int, np.integer)):
                    return self._feature_array_2d[key]
                elif isinstance(key, slice):
                    return self._feature_array_2d[key]
                else:
                    # 如果是索引标签，需要查找位置
                    if hasattr(self._index, "get_loc"):
                        loc = self._index.get_loc(key)
                        return self._feature_array_2d[loc]
                    return self._feature_array_2d[key]

            def __iter__(self):
                # 迭代返回每一行
                return iter(self._feature_array_2d)

            def __array__(self, dtype=None):
                return (
                    self._feature_array_2d
                    if dtype is None
                    else self._feature_array_2d.astype(dtype)
                )

            def __getattr__(self, name):
                # 对于其他属性，尝试从数组获取
                if hasattr(self._feature_array_2d, name):
                    return getattr(self._feature_array_2d, name)
                raise AttributeError(
                    f"'{type(self).__name__}' object has no attribute '{name}'"
                )

        # 先创建空的DataFrame，然后使用CustomDataFrame包装
        result_base = pd.DataFrame(index=data.index)
        feature_obj_final = None
        label_obj_final = None

        if "feature" in col_set:
            # qlib期望feature是一个Series，但values属性返回2D数组
            # LightGBM需要2D数组 shape (n_samples, n_features)
            if len(feature_cols) > 0:
                # 直接获取特征数据为2D数组
                feature_array = data[
                    feature_cols
                ].values  # shape: (n_samples, n_features)
                feature_obj_final = FeatureSeries(feature_array, data.index)
                # 不直接赋值，而是使用占位符，在CustomDataFrame中处理
                result_base["feature"] = pd.Series(
                    [None] * len(data.index), index=data.index
                )
            else:
                # 空特征
                empty_array = np.zeros((len(data), 0))
                feature_obj_final = FeatureSeries(empty_array, data.index)
                result_base["feature"] = pd.Series(
                    [None] * len(data.index), index=data.index
                )

        if "label" in col_set:
            if "label" in data.columns:
                label_series = data["label"]
                # 获取原始values
                label_values = (
                    label_series.values
                    if isinstance(label_series, pd.Series)
                    else np.array(label_series)
                )

                # qlib的gbdt期望: y.values.ndim == 2 and y.values.shape[1] == 1
                # 但pandas Series的values通常是1D的
                # 我们需要创建一个继承自Series的类，重写values属性
                if label_values.ndim == 1:
                    # 1D -> 2D: (n,) -> (n, 1)
                    label_values_2d = label_values.reshape(-1, 1)
                    label_values_1d = label_values
                elif label_values.ndim == 2:
                    if label_values.shape[1] == 1:
                        label_values_2d = label_values
                        label_values_1d = label_values.flatten()
                    else:
                        # 多列，取第一列
                        label_values_2d = label_values[:, 0:1]
                        label_values_1d = label_values[:, 0]
                else:
                    # 其他维度，尝试flatten
                    label_values_flat = np.array(label_values).flatten()
                    label_values_2d = label_values_flat.reshape(-1, 1)
                    label_values_1d = label_values_flat

                # 使用在方法开始处定义的LabelSeries类
                label_obj = LabelSeries(
                    label_values_1d,
                    label_values_2d,
                    label_series.index
                    if isinstance(label_series, pd.Series)
                    else data.index,
                )
            else:
                # 创建默认标签
                default_values_1d = np.zeros(len(data))
                default_values_2d = default_values_1d.reshape(-1, 1)

                label_obj = LabelSeries(
                    default_values_1d, default_values_2d, data.index
                )

            # 保存label对象，不直接赋值给DataFrame
            label_obj_final = label_obj
            result_base["label"] = pd.Series(
                [None] * len(data.index), index=data.index
            )
        else:
            # 创建默认标签
            default_values_1d = np.zeros(len(data))
            default_values_2d = default_values_1d.reshape(-1, 1)
            label_obj_final = LabelSeries(
                default_values_1d, default_values_2d, data.index
            )
            result_base["label"] = pd.Series(
                [None] * len(data.index), index=data.index
            )

        if "label" not in col_set:
            # 如果没有请求label，创建默认的
            default_values_1d = np.zeros(len(data))
            default_values_2d = default_values_1d.reshape(-1, 1)
            label_obj_final = LabelSeries(
                default_values_1d, default_values_2d, data.index
            )
            result_base["label"] = pd.Series(
                [None] * len(data.index), index=data.index
            )

        # 创建一个自定义的DataFrame类，重写__getitem__以返回自定义Series对象
        class CustomDataFrame(pd.DataFrame):
            """自定义DataFrame，确保label和feature列返回正确的对象"""

            def __init__(
                self,
                *args,
                label_series_obj=None,
                feature_series_obj=None,
                **kwargs,
            ):
                super().__init__(*args, **kwargs)
                self._label_series_obj = label_series_obj
                self._feature_series_obj = feature_series_obj

            def __getitem__(self, key):
                # 如果访问label列，返回我们的LabelSeries对象
                if key == "label" and self._label_series_obj is not None:
                    return self._label_series_obj
                # 如果访问feature列，返回我们的FeatureSeries对象
                if key == "feature" and self._feature_series_obj is not None:
                    return self._feature_series_obj
                # 其他情况使用默认行为
                return super().__getitem__(key)

        # 如果只请求feature，直接返回FeatureSeries（Qlib的predict期望这样）
        if col_set == ["feature"] or (
            isinstance(col_set, str) and col_set == "feature"
        ):
            if feature_obj_final is not None:
                return feature_obj_final
            else:
                # 如果没有特征，返回空的FeatureSeries
                empty_array = np.zeros((len(data), 0))
                return FeatureSeries(empty_array, data.index)

        # 如果只请求label，直接返回LabelSeries
        if col_set == ["label"] or (
            isinstance(col_set, str) and col_set == "label"
        ):
            if label_obj_final is not None:
                return label_obj_final
            else:
                # 如果没有标签，返回空的LabelSeries
                default_values_1d = np.zeros(len(data))
                default_values_2d = default_values_1d.reshape(-1, 1)
                return LabelSeries(
                    default_values_1d, default_values_2d, data.index
                )

        # 如果请求多个列（feature和label），返回CustomDataFrame
        if label_obj_final is not None or feature_obj_final is not None:
            custom_result = CustomDataFrame(
                result_base,
                label_series_obj=label_obj_final,
                feature_series_obj=feature_obj_final,
            )
            return custom_result
        else:
            return result_base

    def __getattr__(self, name):
        # 转发其他属性到DataFrame
        return getattr(self.data, name)
