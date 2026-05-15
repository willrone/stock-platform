# Backtest Golden Baselines

These fixtures protect backtest trading semantics before performance optimization.

Run from the repository root:

```bash
# Main performance-optimization guard. Run this before and after performance-only changes.
backend/scripts/backtest_optimization_guard.sh

# Same guard via the backend quality entrypoint.
backend/scripts/quality.sh backtest-guard

# Lower-level commands when debugging individual cases.
backend/.venv-py313/bin/python backend/scripts/backtest_golden_runner.py list
backend/.venv-py313/bin/python backend/scripts/backtest_golden_runner.py verify --case ma_tiny
backend/.venv-py313/bin/python backend/scripts/backtest_golden_runner.py verify --case ma_small
backend/.venv-py313/bin/python backend/scripts/backtest_golden_runner.py verify --case all
```

Refresh a baseline only when the intended business/trading semantics changed:

```bash
backend/.venv-py313/bin/python backend/scripts/backtest_golden_runner.py generate --case ma_tiny --overwrite
```

Performance-only PRs must not refresh baselines. They must pass `backtest_optimization_guard.sh` against the committed fixtures.

Golden cases use deterministic synthetic OHLCV data. Do not switch them to mutable local market data; otherwise a data refresh could look like an engine regression.

The comparator intentionally ignores runtime-only timing fields and checks:

- scalar return/risk metrics
- trade ledger order, side, quantity, price, commission, slippage, pnl
- daily equity curve and positions
- signal/trade counters
- signal rejection reason distribution when available
