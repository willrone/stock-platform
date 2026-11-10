"""Local LightGBM model compatible with Qlib DatasetH.

Qlib's bundled LGBModel logs metrics into qlib.workflow.R after training. In the
current macOS/Python 3.13 environment that MLflow/recorder path can segfault on
shutdown. This local model keeps the official DatasetH contract but avoids the
workflow recorder side effect.
"""

from __future__ import annotations

import os

# ⚠️⚠️ 必须放在 lightgbm 之前：OpenMP 多线程在本机（macOS + Python 3.13 +
# libomp + qlib provider 的 fork/线程）会让 lightgbm 的 C 层**原生崩溃**：
#   OMP_NUM_THREADS=2 → Abort trap (134)
#   OMP_NUM_THREADS=4 → Segmentation fault (139)
#   OMP_NUM_THREADS=1 → 正常
# 症状：训练照常完成，但在 evaluate 阶段 model.predict() 里 SIGSEGV，
# Python 层 catch 不到 → 整个服务进程直接死（pytest 整轮回归也全丢）。
# §12.44
os.environ.setdefault("OMP_NUM_THREADS", "1")

from typing import Any, Optional, Text, Union

import numpy as np
import pandas as pd
from loguru import logger
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset.weight import Reweighter
from qlib.model.base import ModelFT
from qlib.model.interpret.base import LightGBMFInt


class StockPlatformLGBModel(ModelFT, LightGBMFInt):
    """Lazy-loaded LightGBM model for official-style Qlib training."""

    _lgb: Any = None

    @classmethod
    def _lightgbm(cls) -> Any:
        if cls._lgb is None:
            import lightgbm as lgb

            cls._lgb = lgb
        return cls._lgb

    def __init__(
        self,
        loss: str = "mse",
        early_stopping_rounds: int = 50,
        num_boost_round: int = 1000,
        **kwargs: Any,
    ) -> None:
        if loss not in {"mse", "binary"}:
            raise NotImplementedError(f"Unsupported qlib loss: {loss}")
        self.params = {"objective": loss, "verbosity": -1}
        self.params.update(kwargs)
        self.early_stopping_rounds = early_stopping_rounds
        self.num_boost_round = num_boost_round
        self.model: Optional[Any] = None
        self.evals_result_: dict[str, Any] = {}

    def _prepare_data(
        self, dataset: DatasetH, reweighter: Any = None
    ) -> list[tuple[Any, str]]:
        lgb = self._lightgbm()
        ds_l: list[tuple[Any, str]] = []
        if "train" not in dataset.segments:
            raise ValueError("DatasetH must contain train segment")

        for key in ["train", "valid"]:
            if key not in dataset.segments:
                continue
            df = dataset.prepare(
                key,
                col_set=["feature", "label"],
                data_key=DataHandlerLP.DK_L,
            )
            if df.empty:
                raise ValueError("Empty data from dataset, please check dataset config")
            x, y = df["feature"], df["label"]
            if y.values.ndim == 2 and y.values.shape[1] == 1:
                y_values = np.squeeze(y.values)
            else:
                raise ValueError("LightGBM doesn't support multi-label training")

            if reweighter is None:
                w = None
            elif isinstance(reweighter, Reweighter):
                w = reweighter.reweight(df)
            else:
                raise ValueError("Unsupported reweighter type")
            # ⚠️ 数据门禁：特征/标签含 NaN/Inf 时 lightgbm 会训出 NaN 分裂阈值，
            # 预测时**原生段错误**（SIGSEGV，Python 层 catch 不到）——§12.44
            x_values = np.ascontiguousarray(x.values, dtype=np.float64)
            if not np.isfinite(x_values).all():
                bad = int((~np.isfinite(x_values)).sum())
                raise ValueError(
                    f"训练特征含 NaN/Inf（{bad} 个），lightgbm 会训出损坏模型 → 拒绝训练"
                )
            if not np.isfinite(np.asarray(y_values, dtype="float64")).all():
                raise ValueError("训练标签含 NaN/Inf，拒绝训练")

            train_data = lgb.Dataset(
                x_values,
                label=y_values,
                weight=w,
                free_raw_data=False,
            )
            # 记录训练时的特征数：predict 时必须校验，否则特征数不一致会让
            # lightgbm 的 C 层**原生段错误**（SIGSEGV，Python 层 catch 不到，
            # 整个服务进程直接死掉）——§12.44
            self._train_feature_count = int(x_values.shape[1])
            if key == "train":
                self._train_y = np.asarray(y_values, dtype="float64")
                self._n_train_rows = int(x_values.shape[0])
            self._train_feature_names = (
                list(x.columns) if hasattr(x, "columns") else None
            )
            ds_l.append((train_data, key))
        return ds_l

    def fit(
        self,
        dataset: DatasetH,
        num_boost_round: Optional[int] = None,
        early_stopping_rounds: Optional[int] = None,
        verbose_eval: int = 20,
        evals_result: Optional[dict[str, Any]] = None,
        reweighter: Any = None,
        **kwargs: Any,
    ) -> None:
        lgb = self._lightgbm()
        if evals_result is None:
            evals_result = {}
        ds_l = self._prepare_data(dataset, reweighter)
        ds, names = list(zip(*ds_l))
        callbacks = [
            lgb.early_stopping(
                self.early_stopping_rounds
                if early_stopping_rounds is None
                else early_stopping_rounds
            ),
            lgb.log_evaluation(period=verbose_eval),
            lgb.record_evaluation(evals_result),
        ]
        # ⚠️ 样本量自适应：极端宽表（42 行 × 数百列）配 min_data_in_leaf=20 +
        # num_leaves=31 会生成退化树，lightgbm 预测时原生段错误（§12.44）
        params = dict(self.params)
        n_rows = int(getattr(self, "_n_train_rows", 0) or 0)
        if n_rows and params.get("min_data_in_leaf", 20) * 2 > n_rows:
            new_min = max(1, n_rows // 10)
            new_leaves = min(params.get("num_leaves", 31), max(2, n_rows // 10))
            logger.warning(
                "训练样本 {} 行偏少，自动收缩树参数: min_data_in_leaf {}→{}, num_leaves {}→{}",
                n_rows,
                params.get("min_data_in_leaf", 20),
                new_min,
                params.get("num_leaves", 31),
                new_leaves,
            )
            params["min_data_in_leaf"] = new_min
            params["num_leaves"] = new_leaves

        self.model = lgb.train(
            params,
            ds[0],
            num_boost_round=(
                self.num_boost_round if num_boost_round is None else num_boost_round
            ),
            valid_sets=ds,
            valid_names=names,
            callbacks=callbacks,
            **kwargs,
        )
        self.evals_result_ = evals_result

    def predict(
        self, dataset: DatasetH, segment: Union[Text, slice] = "test"
    ) -> pd.Series:
        if self.model is None:
            raise ValueError("model is not fitted yet")
        x_test = dataset.prepare(
            segment,
            col_set="feature",
            data_key=DataHandlerLP.DK_I,
        )
        # ⚠️ 按**训练时的特征名**对齐（顺序 + 内容）：训练/预测特征集不一致时
        # lightgbm 的 C 层会原生段错误（SIGSEGV，Python 层 catch 不到 → 进程死），
        # 且特征错位会让预测结果完全失真。§12.44
        names = getattr(self, "_train_feature_names", None)
        if names and hasattr(x_test, "columns"):
            have = set(x_test.columns)
            missing = [n for n in names if n not in have]
            if missing:
                raise ValueError(
                    f"预测缺失训练特征 {len(missing)} 个（segment={segment!r}），"
                    f"例如 {missing[:5]}"
                )
            x_test = x_test.loc[:, names]

        values = np.ascontiguousarray(x_test.values, dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"预测特征含 NaN/Inf（segment={segment!r}），拒绝预测")
        n_train = getattr(self, "_train_feature_count", None)
        if n_train is not None and values.shape[1] != n_train:
            raise ValueError(
                f"特征数不匹配: 训练 {n_train} 列 vs 预测 {values.shape[1]} 列 "
                f"(segment={segment!r})"
            )
        pred = self.model.predict(values)
        return pd.Series(pred, index=x_test.index)

    def finetune(
        self,
        dataset: DatasetH,
        num_boost_round: int = 10,
        verbose_eval: int = 20,
        reweighter: Any = None,
    ) -> None:
        lgb = self._lightgbm()
        if self.model is None:
            raise ValueError("model is not fitted yet")
        ds_l = self._prepare_data(dataset, reweighter)
        dtrain, _ = ds_l[0]
        self.model = lgb.train(
            self.params,
            dtrain,
            num_boost_round=num_boost_round,
            init_model=self.model,
            callbacks=[lgb.log_evaluation(period=verbose_eval)],
        )
