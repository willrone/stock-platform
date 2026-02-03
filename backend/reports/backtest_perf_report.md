# Backtest Performance Report (500 stocks × ~3y)

**Date**: 2026-02-03
**Branch**: `feat/backtest-process-signal-poc`

## Acceptance
- ✅ **500 stocks × ~3y < 180s**

## Benchmark setup
- Script: `scripts/benchmark_batch_signals.py`
- Strategy: `MACD`
- Window: ~750 calendar days (includes warm-up); actual trading days reported by executor: **563**
- Mode: `enable_batch=True` (use precompute + fast signal query)

> Note: `trading_days` is the authoritative key returned by executor; the script now falls back correctly.

## Results

### 50 stocks (sanity)
- **Total time**: **10.52s**
- Trading days: **563**
- Signals: **2123**
- Trades: **1628**
- Throughput: **53.49 days/s**

### 500 stocks (target)
- **Total time**: **~99.9s**
- Trading days: **563**
- Signals: **21637**
- Trades: **11937**
- Throughput: **~5.64 days/s**

## Baseline & speedup
- Baseline reference used by benchmark:
  - 50 stocks: **143.96s**
  - 500 stocks (linear scale): **1439.60s**

- Observed speedups:
  - 50 stocks: **~13.7×**
  - 500 stocks: **~14.4×**

## Key changes landed

### 1) BacktestExecutor bugfix (profiling vars)
- Initialize `slice_time_total/gen_time_total/gen_time_max` **at the top of each trading-day loop** to prevent `UnboundLocalError` on exceptional branches.

### 2) Portfolio snapshot log gating
- `PortfolioManager.record_portfolio_snapshot()` previously logged `ERROR` whenever positions > 10, which caused massive I/O overhead for large universes.
- Added config switch (default OFF):
  - `settings.ENABLE_PORTFOLIO_SNAPSHOT_SANITY_LOG = False`

### 3) Qlib DataLoader improvements
- Normalize qlib stock code to `safe_code = stock_code.replace('.', '_')` when loading single-file qlib parquet.
- Add config switch for optional `all_stocks.parquet` path (default OFF to avoid heavy I/O on misses):
  - `settings.QLIB_USE_ALL_STOCKS_FILE = False`

### 4) Benchmark script correctness
- `scripts/benchmark_batch_signals.py` now reads `trading_days` (fallback from `total_trading_days`) so returned dict is correct.

## Remaining follow-ups (optional)
- Reduce `FutureWarning` spam from pandas resample frequency ("M"→"ME", "Y"→"YE").
- Further reduce report generation cost (currently ~15s at 500 stocks) if needed.
- Consider persisting precomputed signal cache across runs to remove warmup cost in repeated benchmarks.
