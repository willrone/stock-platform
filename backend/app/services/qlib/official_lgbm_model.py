"""Local LightGBM model compatible with Qlib DatasetH.

Qlib's bundled LGBModel logs metrics into qlib.workflow.R after training. In the
current macOS/Python 3.13 environment that MLflow/recorder path can segfault on
shutdown. This local model keeps the official DatasetH contract but avoids the
workflow recorder side effect.
"""

from __future__ import annotations

from typing import Any, Optional, Text, Union

import numpy as np
import pandas as pd
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
            train_data = lgb.Dataset(
                x.values,
                label=y_values,
                weight=w,
                free_raw_data=False,
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
        self.model = lgb.train(
            self.params,
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
        values = np.ascontiguousarray(x_test.values, dtype=np.float64)
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
