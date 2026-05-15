"""Official Qlib benchmark workflow presets.

This module captures the canonical defaults used by the public Qlib
LightGBM Alpha158/Alpha360 benchmark workflows so stock-platform can
reproduce them explicitly instead of approximating them through the
local enhanced pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union


class OfficialDataset(str, Enum):
    ALPHA158 = "alpha158"
    ALPHA360 = "alpha360"


class OfficialMarket(str, Enum):
    CSI300 = "csi300"
    CSI500 = "csi500"


@dataclass(frozen=True)
class OfficialSegments:
    train: Tuple[str, str]
    valid: Tuple[str, str]
    test: Tuple[str, str]


@dataclass(frozen=True)
class OfficialWorkflowConfig:
    dataset: OfficialDataset
    market: str
    benchmark: str
    handler_class: str
    feature_count: int
    label_expression: str
    infer_processors: List[Dict[str, Any]]
    learn_processors: List[Dict[str, Any]]
    segments: OfficialSegments
    topk: int
    n_drop: int
    account: int
    open_cost: float
    close_cost: float
    min_cost: float
    limit_threshold: float = 0.095
    deal_price: str = "close"
    ann_scaler: int = 252


DEFAULT_OFFICIAL_SEGMENTS = OfficialSegments(
    train=("2008-01-01", "2014-12-31"),
    valid=("2015-01-01", "2016-12-31"),
    test=("2017-01-01", "2020-08-01"),
)

_DEFAULT_ALPHA158_LEARN_PROCESSORS: List[Dict[str, Any]] = [
    {"class": "DropnaLabel"},
    {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},
]

_DEFAULT_ALPHA360_INFER_PROCESSORS: List[Dict[str, Any]] = []
_DEFAULT_ALPHA360_LEARN_PROCESSORS: List[Dict[str, Any]] = [
    {"class": "DropnaLabel"},
    {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}},
]

_DEFAULT_LABEL_EXPRESSION = "Ref($close, -2) / Ref($close, -1) - 1"


@dataclass(frozen=True)
class OfficialDatasetAdapter:
    dataset: Any
    workflow_config: OfficialWorkflowConfig
    segment_lengths: Dict[str, int]
    primary_segment: str = "train"

    @property
    def shape(self) -> Tuple[int, int]:
        return (sum(self.segment_lengths.values()), self.workflow_config.feature_count)

    @property
    def empty(self) -> bool:
        return self.shape[0] == 0

    def __len__(self) -> int:
        return self.segment_lengths.get(self.primary_segment, 0)

    def for_segment(self, segment: str) -> "OfficialDatasetAdapter":
        return OfficialDatasetAdapter(
            dataset=self.dataset,
            workflow_config=self.workflow_config,
            segment_lengths=self.segment_lengths,
            primary_segment=segment,
        )

    def prepare(self, segments: Any = None, *args: Any, **kwargs: Any) -> Any:
        if segments is None:
            segments = self.primary_segment
        return self.dataset.prepare(segments, *args, **kwargs)

    def __getattr__(self, item: str) -> Any:
        return getattr(self.dataset, item)


def _resolve_benchmark(market: OfficialMarket) -> str:
    if market == OfficialMarket.CSI500:
        return "SH000905"
    return "SH000300"


def build_official_lightgbm_workflow_config(
    *,
    dataset: OfficialDataset,
    market: OfficialMarket = OfficialMarket.CSI300,
) -> OfficialWorkflowConfig:
    """Return the public Qlib LightGBM benchmark defaults for a dataset."""

    if dataset == OfficialDataset.ALPHA360:
        return OfficialWorkflowConfig(
            dataset=dataset,
            market=market.value,
            benchmark=_resolve_benchmark(market),
            handler_class="Alpha360",
            feature_count=360,
            label_expression=_DEFAULT_LABEL_EXPRESSION,
            infer_processors=_DEFAULT_ALPHA360_INFER_PROCESSORS,
            learn_processors=_DEFAULT_ALPHA360_LEARN_PROCESSORS,
            segments=DEFAULT_OFFICIAL_SEGMENTS,
            topk=50,
            n_drop=5,
            account=100000000,
            open_cost=0.0005,
            close_cost=0.0015,
            min_cost=5.0,
        )

    return OfficialWorkflowConfig(
        dataset=OfficialDataset.ALPHA158,
        market=market.value,
        benchmark=_resolve_benchmark(market),
        handler_class="Alpha158",
        feature_count=158,
        label_expression=_DEFAULT_LABEL_EXPRESSION,
        infer_processors=[],
        learn_processors=_DEFAULT_ALPHA158_LEARN_PROCESSORS,
        segments=DEFAULT_OFFICIAL_SEGMENTS,
        topk=50,
        n_drop=5,
        account=100000000,
        open_cost=0.0005,
        close_cost=0.0015,
        min_cost=5.0,
    )


def _normalize_instruments_override(
    instruments_override: Optional[Sequence[str]],
) -> Optional[List[str]]:
    if not instruments_override:
        return None

    normalized: List[str] = []
    for instrument in instruments_override:
        value = (instrument or "").strip()
        if not value:
            continue
        normalized.append(value.replace(".", "_").lower())
    return normalized or None


def build_official_dataset_config(
    workflow: OfficialWorkflowConfig,
    instruments_override: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Build a DatasetH config mirroring the public Qlib benchmark workflow."""

    normalized_instruments = _normalize_instruments_override(instruments_override)

    return {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class": workflow.handler_class,
                "module_path": "qlib.contrib.data.handler",
                "kwargs": {
                    "start_time": workflow.segments.train[0],
                    "end_time": workflow.segments.test[1],
                    "fit_start_time": workflow.segments.train[0],
                    "fit_end_time": workflow.segments.train[1],
                    "instruments": normalized_instruments or workflow.market,
                    "infer_processors": workflow.infer_processors,
                    "learn_processors": workflow.learn_processors,
                    "label": [workflow.label_expression],
                },
            },
            "segments": {
                "train": workflow.segments.train,
                "valid": workflow.segments.valid,
                "test": workflow.segments.test,
            },
        },
    }


def _count_rows(segment_data: Any) -> int:
    if hasattr(segment_data, "shape"):
        shape = segment_data.shape
        if shape:
            return int(shape[0])
    if hasattr(segment_data, "__len__"):
        return int(len(segment_data))
    return 0


def create_official_dataset_adapter(
    workflow: OfficialWorkflowConfig,
    stock_codes: Optional[List[str]] = None,
    provider_uri: Optional[Union[str, Path]] = None,
    dataset_factory: Optional[Callable[[Dict[str, Any]], Any]] = None,
    qlib_initializer: Optional[Callable[[Union[str, Path]], None]] = None,
) -> OfficialDatasetAdapter:
    """Instantiate the official DatasetH and wrap it with lightweight metadata."""

    dataset_config = build_official_dataset_config(
        workflow, instruments_override=stock_codes
    )
    if provider_uri is not None:
        if qlib_initializer is None:
            import qlib
            from qlib.config import REG_CN

            def qlib_initializer(uri: Union[str, Path]) -> None:
                qlib.init(
                    provider_uri=str(uri),
                    region=REG_CN,
                    auto_mount=False,
                    joblib_backend="threading",
                )

        qlib_initializer(provider_uri)

    if dataset_factory is None:
        from qlib.utils import init_instance_by_config

        dataset_factory = init_instance_by_config

    dataset = dataset_factory(dataset_config)
    segment_lengths: Dict[str, int] = {}
    for segment in ("train", "valid", "test"):
        segment_lengths[segment] = _count_rows(
            dataset.prepare(segment, col_set="label")
        )

    return OfficialDatasetAdapter(
        dataset=dataset,
        workflow_config=workflow,
        segment_lengths=segment_lengths,
    )
