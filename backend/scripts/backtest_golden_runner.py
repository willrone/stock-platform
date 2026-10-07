#!/usr/bin/env python
"""Generate and verify golden backtest baselines.

This is the first safety layer before performance optimization. It captures a
normalized, deterministic result payload and can compare future runs against it.

Usage:
  cd backend
  ./venv/bin/python scripts/backtest_golden_runner.py generate --case ma_tiny
  ./venv/bin/python scripts/backtest_golden_runner.py verify --case ma_tiny
  ./venv/bin/python scripts/backtest_golden_runner.py list
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.config import settings
from app.services.backtest import BacktestExecutor
from app.services.backtest.models import BacktestConfig
from scripts.backtest_result_compare import compare_results
from scripts.verify_signal_integrity import verify_result

try:
    from loguru import logger as _logger

    _logger.remove()
    _logger.add(sys.stderr, level=os.environ.get("GOLDEN_LOG_LEVEL", "WARNING"))
except Exception:
    pass

DEFAULT_BASELINE_DIR = BACKEND_ROOT / "tests" / "fixtures" / "backtest_golden"
RUNTIME_FIELDS = {
    "perf_breakdown",
    "performance_analysis",
    "timestamp",
    "execution_time",
}


@dataclass(frozen=True)
class GoldenCase:
    name: str
    description: str
    strategy_name: str
    stock_codes: list[str]
    start: str
    end: str
    strategy_config: dict[str, Any]
    backtest_config: dict[str, Any]
    data_source: str = "synthetic"
    prediction_source: str | None = None


CASES: dict[str, GoldenCase] = {
    "ma_tiny": GoldenCase(
        name="ma_tiny",
        description="Tiny moving-average semantics guard: quick local diff with trades and equity curve.",
        strategy_name="moving_average",
        stock_codes=["000001.SZ", "000002.SZ", "600000.SH"],
        start="2024-01-01",
        end="2024-06-30",
        strategy_config={"short_window": 3, "long_window": 10},
        backtest_config={
            "initial_cash": 100000.0,
            "commission_rate": 0.001,
            "slippage_rate": 0.001,
            "open_cost": 0.0,
            "close_cost": 0.0,
            "min_cost": 0.0,
            "max_position_size": 0.2,
            "cash_reserve_ratio": 0.05,
            "board_lot_size": 100,
            "record_portfolio_history": True,
            "portfolio_history_stride": 1,
            "record_positions_in_history": True,
        },
    ),
    "ma_small": GoldenCase(
        name="ma_small",
        description="Small moving-average guard over a wider stock set/date range.",
        strategy_name="moving_average",
        stock_codes=[
            "000001.SZ",
            "000002.SZ",
            "000333.SZ",
            "600000.SH",
            "600519.SH",
            "601318.SH",
            "000858.SZ",
            "002415.SZ",
            "300750.SZ",
            "601398.SH",
        ],
        start="2023-01-01",
        end="2024-12-31",
        strategy_config={"short_window": 5, "long_window": 20},
        backtest_config={
            "initial_cash": 1000000.0,
            "commission_rate": 0.001,
            "slippage_rate": 0.001,
            "open_cost": 0.0,
            "close_cost": 0.0,
            "min_cost": 0.0,
            "max_position_size": 0.2,
            "cash_reserve_ratio": 0.05,
            "board_lot_size": 100,
            "record_portfolio_history": True,
            "portfolio_history_stride": 1,
            "record_positions_in_history": True,
        },
    ),
    "topk_dropout_tiny": GoldenCase(
        name="topk_dropout_tiny",
        description="Tiny official-style TopK/Dropout ranking guard with synthetic predictions.",
        strategy_name="model_topk_dropout",
        stock_codes=[
            "000001.SZ",
            "000002.SZ",
            "000333.SZ",
            "600000.SH",
            "600519.SH",
            "601318.SH",
        ],
        start="2024-01-01",
        end="2024-04-30",
        strategy_config={
            "model_id": "golden_synthetic_model",
            "topk": 2,
            "n_drop": 1,
            "hold_thresh": 1,
            "score_scale": 20.0,
        },
        backtest_config={
            "initial_cash": 200000.0,
            "commission_rate": 0.001,
            "slippage_rate": 0.001,
            "open_cost": 0.0,
            "close_cost": 0.0,
            "min_cost": 0.0,
            "max_position_size": 0.45,
            "cash_reserve_ratio": 0.02,
            "board_lot_size": 100,
            "record_portfolio_history": True,
            "portfolio_history_stride": 1,
            "record_positions_in_history": True,
        },
        prediction_source="synthetic_rank_rotation",
    ),
    "rsi_small": GoldenCase(
        name="rsi_small",
        description="RSI strategy guard: trend reversal pattern triggering overbought/oversold signals.",
        strategy_name="rsi",
        stock_codes=["SYNTH_1001.SZ", "SYNTH_1002.SZ", "SYNTH_1003.SZ"],
        start="2024-01-02",
        end="2024-04-30",
        strategy_config={
            "rsi_period": 14,
            "oversold_threshold": 30,
            "overbought_threshold": 70,
        },
        backtest_config={
            "initial_cash": 100000.0,
            "commission_rate": 0.001,
            "slippage_rate": 0.001,
            "open_cost": 0.0,
            "close_cost": 0.0,
            "min_cost": 0.0,
            "max_position_size": 0.25,
            "cash_reserve_ratio": 0.05,
            "board_lot_size": 100,
            "record_portfolio_history": True,
            "portfolio_history_stride": 1,
            "record_positions_in_history": True,
        },
    ),
    "macd_small": GoldenCase(
        name="macd_small",
        description="MACD strategy guard: oscillating market generating golden cross / death cross signals.",
        strategy_name="macd",
        stock_codes=["SYNTH_2001.SZ", "SYNTH_2002.SZ", "SYNTH_2003.SZ"],
        start="2024-01-02",
        end="2024-07-31",
        strategy_config={
            "fast_period": 12,
            "slow_period": 26,
            "signal_period": 9,
        },
        backtest_config={
            "initial_cash": 100000.0,
            "commission_rate": 0.001,
            "slippage_rate": 0.001,
            "open_cost": 0.0,
            "close_cost": 0.0,
            "min_cost": 0.0,
            "max_position_size": 0.25,
            "cash_reserve_ratio": 0.05,
            "board_lot_size": 100,
            "record_portfolio_history": True,
            "portfolio_history_stride": 1,
            "record_positions_in_history": True,
        },
    ),
    "multi_factor_small": GoldenCase(
        name="multi_factor_small",
        description="Multi-factor strategy guard: factor combination scoring with zero-cross signals.",
        strategy_name="multi_factor",
        stock_codes=["SYNTH_3001.SZ", "SYNTH_3002.SZ", "SYNTH_3003.SZ"],
        start="2024-01-02",
        end="2024-12-31",
        strategy_config={
            "factors": ["value", "momentum", "low_volatility"],
            "factor_weights": [0.34, 0.33, 0.33],
            "weighting_method": "equal",
        },
        backtest_config={
            "initial_cash": 200000.0,
            "commission_rate": 0.001,
            "slippage_rate": 0.001,
            "open_cost": 0.0,
            "close_cost": 0.0,
            "min_cost": 0.0,
            "max_position_size": 0.15,
            "cash_reserve_ratio": 0.05,
            "board_lot_size": 100,
            "record_portfolio_history": True,
            "portfolio_history_stride": 1,
            "record_positions_in_history": True,
        },
    ),
    "model_topk_dropout_medium": GoldenCase(
        name="model_topk_dropout_medium",
        description="Medium TopK/Dropout ranking guard: more stocks and longer date range.",
        strategy_name="model_topk_dropout",
        stock_codes=[
            "SYNTH_4001.SZ", "SYNTH_4002.SZ", "SYNTH_4003.SZ",
            "SYNTH_4004.SZ", "SYNTH_4005.SZ", "SYNTH_4006.SZ",
            "SYNTH_4007.SZ", "SYNTH_4008.SZ", "SYNTH_4009.SZ",
            "SYNTH_4010.SZ",
        ],
        start="2024-01-02",
        end="2024-06-28",
        strategy_config={
            "model_id": "golden_synthetic_model",
            "topk": 3,
            "n_drop": 2,
            "hold_thresh": 1,
            "score_scale": 20.0,
        },
        backtest_config={
            "initial_cash": 500000.0,
            "commission_rate": 0.001,
            "slippage_rate": 0.001,
            "open_cost": 0.0,
            "close_cost": 0.0,
            "min_cost": 0.0,
            "max_position_size": 0.3,
            "cash_reserve_ratio": 0.02,
            "board_lot_size": 100,
            "record_portfolio_history": True,
            "portfolio_history_stride": 1,
            "record_positions_in_history": True,
        },
        prediction_source="synthetic_rank_rotation",
    ),
}


def _case_path(case_name: str, baseline_dir: Path) -> Path:
    return baseline_dir / case_name / "baseline.json"


def _candidate_path(case_name: str, baseline_dir: Path) -> Path:
    return baseline_dir / case_name / "candidate.json"


def _json_default(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _stable_json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(
            payload,
            f,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=_json_default,
        )
        f.write("\n")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise SystemExit(f"Expected JSON object: {path}")
    return data


def _strip_runtime_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(k): _strip_runtime_fields(v)
            for k, v in value.items()
            if str(k) not in RUNTIME_FIELDS
        }
    if isinstance(value, list):
        return [_strip_runtime_fields(v) for v in value]
    return value


def _normalize_result(result: dict[str, Any], case: GoldenCase) -> dict[str, Any]:
    """Keep only correctness-relevant fields and stable metadata."""
    normalized = _strip_runtime_fields(result)
    normalized["golden_case"] = asdict(case)
    return normalized


def _make_config(case: GoldenCase) -> BacktestConfig:
    return BacktestConfig(**case.backtest_config)


def _synthetic_stock_frame(
    *,
    code: str,
    dates: pd.DatetimeIndex,
    base: float,
    amplitude: float,
    drift: float,
    phase: float,
) -> pd.DataFrame:
    """Build deterministic OHLCV data that creates MA crossovers.

    The shape is intentionally simple and local. Golden guard runs must not rely
    on mutable external market data files, otherwise a data refresh could look
    like an engine regression.
    """
    x = np.arange(len(dates), dtype=np.float64)
    close = base + drift * x + amplitude * np.sin((x + phase) / 6.0)
    open_ = close * (1.0 + 0.002 * np.cos((x + phase) / 5.0))
    high = np.maximum(open_, close) * 1.01
    low = np.minimum(open_, close) * 0.99
    volume = (
        1_000_000 + (x.astype(np.int64) * 137 + len(code) * 997) % 200_000
    ).astype(np.int64)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=dates,
    )


def _synthetic_prediction_series(
    *, code_index: int, dates: pd.DatetimeIndex
) -> pd.Series:
    x = np.arange(len(dates), dtype=np.float64)
    # Deterministic rank rotation: each stock has a different phase and slope.
    values = (
        0.04 * np.sin((x / 8.0) + code_index * 0.9)
        + 0.015 * np.cos((x / 17.0) - code_index * 0.4)
        + 0.002 * ((code_index % 3) - 1)
    )
    return pd.Series(values, index=dates)


def _rsi_trend_frame(
    *,
    code: str,
    dates: pd.DatetimeIndex,
    base: float,
    amplitude: float,
) -> pd.DataFrame:
    """Deterministic price with a clear trend-reversal pattern for RSI strategy.

    Phase 1 (0-40%): moderate decline to push RSI below oversold (30)
    Phase 2 (40-70%): strong rise to push RSI above overbought (70), triggering BUY
    Phase 3 (70-100%): decline to push RSI back below overbought (70), triggering SELL
    """
    n = len(dates)
    x = np.arange(n, dtype=np.float64)
    mid1 = int(n * 0.4)
    mid2 = int(n * 0.7)
    close = np.empty(n, dtype=np.float64)
    # Phase 1: decline
    close[:mid1] = base * (1.0 - 0.20 * x[:mid1] / mid1)
    # Phase 2: strong recovery
    t2 = (x[mid1:mid2] - mid1) / (mid2 - mid1)
    close[mid1:mid2] = close[mid1 - 1] * (1.0 + 0.30 * t2)
    # Phase 3: decline again
    t3 = (x[mid2:] - mid2) / (n - mid2)
    close[mid2:] = close[mid2 - 1] * (1.0 - 0.15 * t3)
    # Add small sine noise for variation
    noise = amplitude * 0.3 * np.sin(x / 5.0 + float(len(code)))
    close = close + noise
    close = np.maximum(close, base * 0.3)
    open_ = close * (1.0 + 0.002 * np.cos(x / 5.0))
    high = np.maximum(open_, close) * 1.01
    low = np.minimum(open_, close) * 0.99
    volume = (1_000_000 + (x.astype(np.int64) * 137 + len(code) * 997) % 200_000).astype(np.int64)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


def _macd_oscillation_frame(
    *,
    code: str,
    dates: pd.DatetimeIndex,
    base: float,
    amplitude: float,
    drift: float,
    phase: float,
) -> pd.DataFrame:
    """Deterministic oscillating price for MACD golden/death cross generation.

    Uses two superimposed sine waves to create clear MACD crossovers.
    """
    x = np.arange(len(dates), dtype=np.float64)
    close = (
        base
        + drift * x
        + amplitude * np.sin((x + phase) / 8.0)
        + 0.5 * amplitude * np.sin((x + phase) / 15.0)
    )
    open_ = close * (1.0 + 0.002 * np.cos((x + phase) / 5.0))
    high = np.maximum(open_, close) * 1.01
    low = np.minimum(open_, close) * 0.99
    volume = (1_000_000 + (x.astype(np.int64) * 137 + len(code) * 997) % 200_000).astype(np.int64)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


def _multi_factor_frame(
    *,
    code: str,
    dates: pd.DatetimeIndex,
    base: float,
    _amplitude: float,  # unused; kept for API compatibility
    drift: float,
    phase: float,
) -> pd.DataFrame:
    """Deterministic price series for MultiFactor strategy.

    Creates a flat-low-volatility pattern, followed by a sharp +30% jump,
    then a -30% crash. The abrupt changes force combined_score zero-crossings
    with non-trivial strength after the 130-day warmup.
    """
    n = len(dates)
    x = np.arange(n, dtype=np.float64)
    close = np.full(n, base, dtype=np.float64)
    # Tiny noise throughout for determinism
    noise = 0.05 * np.sin(x / 3.0 + phase)
    close = close + noise
    # Days ≈ 60-64: spike +30% over 5 days
    spike_start = 60
    spike_end = min(spike_start + 5, n)
    close_spike_pre = close[spike_end - 1] if spike_end <= n else base
    for i in range(spike_start, spike_end):
        t = (i - spike_start) / 5.0
        close[i] = close[spike_start - 1] * (1.0 + 0.30 * t)
    # Days 65–164: hold elevated, small drift
    hold_start = spike_end
    hold_end = min(hold_start + 100, n)
    if hold_end > hold_start:
        hold_val = close[hold_start - 1] if hold_start > 0 else base
        for i in range(hold_start, hold_end):
            close[i] = hold_val + drift * (i - hold_start) + 0.1 * np.sin(i / 5.0 + phase)
    # Days ≈ 165-169: crash -30% over 5 days
    crash_start = hold_end
    crash_end = min(crash_start + 5, n)
    if crash_end > crash_start:
        for i in range(crash_start, crash_end):
            t = (i - crash_start) / 5.0
            close[i] = close[crash_start - 1] * (1.0 - 0.30 * t)
    # Remaining days: low drift
    for i in range(crash_end, n):
        close[i] = close[i - 1] + drift * 0.5
    close = np.maximum(close, base * 0.2)
    open_ = close * (1.0 + 0.002 * np.cos((x + phase) / 5.0))
    high = np.maximum(open_, close) * 1.015
    low = np.minimum(open_, close) * 0.985
    volume = (1_000_000 + (x.astype(np.int64) * 137 + len(code) * 997) % 200_000).astype(np.int64)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )
    open_ = close * (1.0 + 0.002 * np.cos((x + phase) / 5.0))
    high = np.maximum(open_, close) * 1.01
    low = np.minimum(open_, close) * 0.99
    volume = (1_000_000 + (x.astype(np.int64) * 137 + len(code) * 997) % 200_000).astype(np.int64)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


def _synthetic_stock_data(case: GoldenCase) -> dict[str, pd.DataFrame]:
    dates = pd.bdate_range(case.start, case.end)
    stock_data: dict[str, pd.DataFrame] = {}

    if case.name == "rsi_small":
        for i, code in enumerate(case.stock_codes):
            frame = _rsi_trend_frame(
                code=code, dates=dates, base=10.0 + i * 2.0, amplitude=1.5 + (i % 3) * 0.5,
            )
            frame.attrs["stock_code"] = code
            stock_data[code] = frame
        return stock_data

    if case.name == "macd_small":
        for i, code in enumerate(case.stock_codes):
            frame = _macd_oscillation_frame(
                code=code, dates=dates, base=10.0 + i * 3.0, amplitude=1.5 + (i % 3) * 0.4,
                drift=0.005 * ((i % 2) * 2 - 1), phase=float(i * 2),
            )
            frame.attrs["stock_code"] = code
            stock_data[code] = frame
        return stock_data

    if case.name == "multi_factor_small":
        for i, code in enumerate(case.stock_codes):
            frame = _multi_factor_frame(
                code=code, dates=dates, base=10.0 + i * 3.0, _amplitude=0.5,
                drift=0.002 * ((i % 3) - 1), phase=float(i * 4),
            )
            frame.attrs["stock_code"] = code
            stock_data[code] = frame
        return stock_data

    for i, code in enumerate(case.stock_codes):
        frame = _synthetic_stock_frame(
            code=code, dates=dates, base=9.0 + i * 3.5, amplitude=1.2 + (i % 4) * 0.35,
            drift=0.006 * ((i % 3) - 1), phase=float(i * 3),
        )
        frame.attrs["stock_code"] = code
        if case.prediction_source == "synthetic_rank_rotation":
            cache = frame.attrs.setdefault("_model_prediction_returns", {})
            cache["ModelTopkDropout"] = _synthetic_prediction_series(code_index=i, dates=dates)
        stock_data[code] = frame
    return stock_data


async def _run_case(case: GoldenCase) -> dict[str, Any]:
    # Make accidental random usage deterministic for guard runs.
    random.seed(0)

    executor = BacktestExecutor(
        data_dir=str(settings.DATA_ROOT_PATH),
        enable_parallel=False,
        enable_performance_profiling=False,
        use_multiprocessing=False,
    )

    if case.data_source != "synthetic":
        raise SystemExit(f"Unsupported golden case data_source={case.data_source!r}")

    stock_data = _synthetic_stock_data(case)
    executor.data_loader.load_multiple_stocks = lambda *args, **kwargs: stock_data

    strategy_config = dict(case.strategy_config)
    if case.prediction_source == "synthetic_rank_rotation":
        # Predictions are already injected into DataFrame attrs, so keep the
        # strategy preparation hook from calling the real PredictionEngine.
        from app.services.backtest.strategies.model_prediction_base import (
            BaseModelPredictionStrategy,
        )

        async def _skip_prediction_prepare(*args: Any, **kwargs: Any) -> None:
            return None

        BaseModelPredictionStrategy.prepare_backtest_data = _skip_prediction_prepare

    result = await executor.run_backtest(
        strategy_name=case.strategy_name,
        stock_codes=list(case.stock_codes),
        start_date=datetime.fromisoformat(case.start),
        end_date=datetime.fromisoformat(case.end),
        strategy_config=strategy_config,
        backtest_config=_make_config(case),
    )
    return _normalize_result(result, case)


def _print_case(case: GoldenCase) -> None:
    print(f"{case.name}: {case.description}")
    print(
        f"  strategy={case.strategy_name} stocks={len(case.stock_codes)} "
        f"range={case.start}..{case.end}"
    )


async def _generate(case: GoldenCase, baseline_dir: Path, *, overwrite: bool) -> int:
    path = _case_path(case.name, baseline_dir)
    if path.exists() and not overwrite:
        print(f"Refusing to overwrite existing baseline: {path}")
        print("Pass --overwrite if this is an intentional baseline refresh.")
        return 2

    result = await _run_case(case)
    _stable_json_dump(path, result)
    print(f"Generated baseline: {path}")
    _print_summary(result)
    return 0


async def _verify(case: GoldenCase, baseline_dir: Path, *, keep_candidate: bool) -> int:
    baseline_path = _case_path(case.name, baseline_dir)
    if not baseline_path.exists():
        print(f"Missing baseline: {baseline_path}")
        print("Run generate first.")
        return 2

    candidate = await _run_case(case)

    # Step 1: Baseline comparison (existing guard)
    baseline = _load_json(baseline_path)
    diffs = compare_results(baseline, candidate)

    if diffs:
        print(f"FAIL: {case.name} differs from baseline ({len(diffs)} diffs)")
        for diff in diffs[:50]:
            print(diff.format())
        if len(diffs) > 50:
            print(f"... truncated {len(diffs) - 50} more differences")

    # Step 2: Signal integrity verification (new intermediate-data guard)
    integrity_report = verify_result(candidate)
    integrity_ok = integrity_report.get("passed", False)
    if not integrity_ok:
        print(f"FAIL: {case.name} signal integrity check failed")

    # Save candidate for debugging if anything failed
    if diffs or not integrity_ok:
        candidate_path = _candidate_path(case.name, baseline_dir)
        _stable_json_dump(candidate_path, candidate)
        if keep_candidate:
            print(f"Candidate saved: {candidate_path}")
        elif not integrity_ok:
            print(f"Candidate saved for debugging: {candidate_path}")

    if not diffs and integrity_ok:
        print(f"PASS: {case.name} matches baseline")
        print(f"INTEGRITY: signal integrity check passed")
        _print_summary(candidate)
        return 0

    return 1


def _print_summary(result: dict[str, Any]) -> None:
    print(
        "summary: "
        f"final_value={float(result.get('final_value', 0.0)):.6f} "
        f"total_return={float(result.get('total_return', 0.0)):.8f} "
        f"trades={int(result.get('total_trades') or 0)} "
        f"signals={int(result.get('total_signals') or 0)} "
        f"days={int(result.get('trading_days') or 0)}"
    )


def _resolve_cases(case_name: str) -> list[GoldenCase]:
    if case_name == "all":
        return list(CASES.values())
    case = CASES.get(case_name)
    if case is None:
        raise SystemExit(
            f"Unknown case: {case_name}. Use list to inspect available cases."
        )
    return [case]


async def async_main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    list_parser = sub.add_parser("list")
    list_parser.set_defaults(_unused=True)

    gen_parser = sub.add_parser("generate")
    gen_parser.add_argument("--case", default="ma_tiny", help="case name or all")
    gen_parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    gen_parser.add_argument("--overwrite", action="store_true")

    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument("--case", default="ma_tiny", help="case name or all")
    verify_parser.add_argument(
        "--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR
    )
    verify_parser.add_argument("--keep-candidate", action="store_true")

    args = parser.parse_args()

    if args.command == "list":
        for case in CASES.values():
            _print_case(case)
        return 0

    cases = _resolve_cases(args.case)
    exit_code = 0
    for case in cases:
        if args.command == "generate":
            code = await _generate(case, args.baseline_dir, overwrite=args.overwrite)
        elif args.command == "verify":
            code = await _verify(
                case, args.baseline_dir, keep_candidate=args.keep_candidate
            )
        else:
            raise AssertionError(args.command)
        exit_code = max(exit_code, code)
    return exit_code


def main() -> int:
    return asyncio.run(async_main())


if __name__ == "__main__":
    raise SystemExit(main())
