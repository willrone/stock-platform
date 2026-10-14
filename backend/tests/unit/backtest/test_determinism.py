"""Determinism regression test: parallel data loading must not fork results.

Root cause (fixed in backtest_executor.py): DataLoader.load_multiple_stocks
collected futures via as_completed(), so the stock_data dict insertion order
depended on thread completion timing. Downstream code iterates stock_data in
order to generate/execute same-day signals, so differing order forked the
trade sequence (observed: same backtest gave 77% / 79% / 107% total return
across three runs with enable_parallel=True).

The fix rebuilds stock_data in input stock_codes order. This test asserts:
  1. enable_parallel=True repeated 3x -> identical trade sequences
  2. enable_parallel=True == enable_parallel=False (both paths identical)

Universe: first 30 codes of the baseline 500-universe file (same source the
experiments use), window 2023-05-12 -> 2026-05-08, strategy rsi.
"""

import asyncio
import hashlib
import importlib.util
import os
import sys
from datetime import datetime

import pytest

BACKEND_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)


@pytest.fixture(scope="module")
def exp_module():
    """Load scripts/sharpe_overlay_experiment.py for make_executor/base_config."""
    if str(BACKEND_ROOT) not in sys.path:
        sys.path.insert(0, str(BACKEND_ROOT))
    spec = importlib.util.spec_from_file_location(
        "sharpe_overlay_experiment",
        os.path.join(BACKEND_ROOT, "scripts", "sharpe_overlay_experiment.py"),
    )
    assert spec is not None, "cannot create module spec for sharpe_overlay_experiment"
    m = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(m)
    return m


def _mkexp(exp_module, parallel: bool):
    return exp_module.BacktestExecutor(
        data_dir=str(exp_module.settings.DATA_ROOT_PATH),
        enable_parallel=parallel,
        max_workers=8,
        enable_performance_profiling=False,
        use_multiprocessing=False,
    )


def _hash_trades(pm) -> str:
    trades = [
        (t["timestamp"].date().isoformat(), t["stock_code"], t["action"])
        for t in pm.trades
    ]
    return hashlib.md5(str(trades).encode()).hexdigest()[:12]


@pytest.mark.slow
@pytest.mark.asyncio
async def test_parallel_serial_determinism(exp_module):
    uni_file = os.path.join(
        exp_module.DATA_ROOT, "strategy_baseline", "baseline_universe_500.txt"
    )
    if not os.path.exists(uni_file):
        pytest.skip(
            f"baseline universe file missing ({uni_file}) — data/ is gitignored, "
            "local-machine only"
        )
    codes = exp_module.load_universe()[:30]
    assert len(codes) >= 20, "universe file too small to be meaningful"

    start = exp_module.START
    end = exp_module.END
    if isinstance(start, str):
        start = datetime.fromisoformat(start)
    if isinstance(end, str):
        end = datetime.fromisoformat(end)
    cfg = exp_module.base_config()

    ex_par = _mkexp(exp_module, True)
    par_hashes = []
    for _ in range(3):
        o = exp_module.Overlay()
        exp_module.install(o)
        await ex_par.run_backtest(
            strategy_name="rsi",
            stock_codes=codes,
            start_date=start,
            end_date=end,
            strategy_config={},
            backtest_config=cfg,
        )
        pm = o.last_instance
        assert pm is not None, "portfolio manager not captured"
        par_hashes.append(_hash_trades(pm))

    assert len(set(par_hashes)) == 1, f"parallel path non-deterministic: {par_hashes}"

    ex_ser = _mkexp(exp_module, False)
    o = exp_module.Overlay()
    exp_module.install(o)
    await ex_ser.run_backtest(
        strategy_name="rsi",
        stock_codes=codes,
        start_date=start,
        end_date=end,
        strategy_config={},
        backtest_config=cfg,
    )
    pm = o.last_instance
    assert pm is not None
    assert _hash_trades(pm) == par_hashes[0], (
        f"parallel/serial path diverge: par={par_hashes[0]} ser={_hash_trades(pm)}"
    )
