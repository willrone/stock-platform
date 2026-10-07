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

## Golden Case Catalog

| Case | Strategy | Stocks | Period | Trades | Purpose |
|------|----------|--------|--------|--------|---------|
| `ma_tiny` | moving_average | 3 | 2024-01 ~ 2024-06 | 6 | Tiny MA semantics guard (quick diff) |
| `ma_small` | moving_average | 10 | 2023-01 ~ 2024-12 | 119 | Wider MA guard over more stocks/date range |
| `topk_dropout_tiny` | model_topk_dropout | 6 | 2024-01 ~ 2024-04 | 16 | Tiny TopK/Dropout ranking guard |
| `rsi_small` | rsi | 3 | 2024-01 ~ 2024-04 | 6 | RSI trend-reversal (overbought/oversold) |
| `macd_small` | macd | 3 | 2024-01 ~ 2024-07 | 15 | MACD golden/death cross oscillation |
| `multi_factor_small` | multi_factor | 3 | 2024-01 ~ 2024-12 | 1 | Multi-factor zero-cross with spike/crash data |
| `model_topk_dropout_medium` | model_topk_dropout | 10 | 2024-01 ~ 2024-06 | 43 | Larger TopK/Dropout with more stocks |

### Case Details

**`rsi_small`** — RSI strategy using a trend-reversal synthetic price pattern (decline → rise → decline) to trigger RSI crossing the 30 oversold threshold (BUY) and the 70 overbought threshold (SELL). Stocks use `SYNTH_1xxx.SZ` synthetic codes.

**`macd_small`** — MACD strategy using dual-sine oscillating price data to generate golden cross (MACD line crosses above signal line = BUY) and death cross (below = SELL) signals. Stocks use `SYNTH_2xxx.SZ` synthetic codes.

**`multi_factor_small`** — MultiFactor strategy combining value, momentum, and low-volatility factors. Synthetic data uses a spike/crash pattern (flat → +30% jump → flat → -30% crash) to create combined_score zero-crossings with non-trivial strength. Stocks use `SYNTH_3xxx.SZ` synthetic codes.

**`model_topk_dropout_medium`** — TopK/Dropout ranking strategy with 10 stocks over 6 months, using synthetic rank-rotation predictions. Larger scale than `topk_dropout_tiny` (topk=3, n_drop=2). Stocks use `SYNTH_4xxx.SZ` synthetic codes.
