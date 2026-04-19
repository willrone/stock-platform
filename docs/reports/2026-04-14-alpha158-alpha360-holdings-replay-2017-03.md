# alpha158 vs alpha360 holdings-aware replay (2017-03-01 ~ 2017-03-03)

日期：2026-04-14

## Scope / sources
- DB: `backend/data/app.db`
  - `trade_records`
  - `portfolio_snapshots`
  - `tasks.result.trade_history` (for per-trade slippage)
- Executor logic: `backend/app/services/backtest/execution/trade_modes.py:171-279`
- Prior ranking evidence: `docs/reports/2026-04-14-alpha158-alpha360-divergence-event-slicing.md:113-151`

注：`portfolio_snapshots.positions` 在这几天是空 `{}`，所以“实际持仓”这里按 `trade_records` 累积重建；NAV / cash 仍直接取 `portfolio_snapshots`。

## 1. Actual replay: holdings and trades

### Entering 2017-03-01
- alpha158 held `000651.SZ + 600519.SH`
- alpha360 held `000651.SZ + 601288.SH`

### Date-by-date executed path
| Date | alpha158 executed trades | alpha158 EOD holdings | alpha360 executed trades | alpha360 EOD holdings |
|---|---|---|---|---|
| 2017-03-01 | SELL `600519.SH` 400 @ `356.431680` (pnl `+423.879670`); BUY `601288.SH` 51,700 @ `3.261630` | `000651.SZ + 601288.SH` | SELL `601288.SH` 48,700 @ `3.258370` (pnl `-1371.271998`); BUY `600036.SH` 8,700 @ `19.349670` | `000651.SZ + 600036.SH` |
| 2017-03-02 | SELL `601288.SH` 51,700 @ `3.258370` (pnl `-421.228592`); BUY `600519.SH` 400 @ `356.998417` | `000651.SZ + 600519.SH` | SELL `600036.SH` 8,700 @ `19.090450` (pnl `-2504.342389`); BUY `601288.SH` 51,800 @ `3.261630` | `000651.SZ + 601288.SH` |
| 2017-03-03 | SELL `000651.SZ` 6,100 @ `27.286349` (pnl `+621.318875`); BUY `601288.SH` 53,500 @ `3.241620` | `600519.SH + 601288.SH` | SELL `000651.SZ` 6,200 @ `27.286349` (pnl `-1291.465089`); BUY `000001.SZ` 17,900 @ `9.404700` | `601288.SH + 000001.SZ` |

Most important concrete divergence:
- alpha360 did a one-day `601288.SH -> 600036.SH -> 601288.SH` round-trip on 03-01/03-02, realizing `-1371.271998 - 2504.342389 = -3875.614387` before costs.
- alpha158 did not touch `600036.SH` in this window; its realized sells over the same three days summed to `+623.969952`.

## 2. Realized portfolio path and costs

### NAV path from `portfolio_snapshots`
- alpha158
  - `2017-03-01`: `1,011,922.557484`
  - `2017-03-02`: `1,010,527.838533`
  - `2017-03-03`: `1,009,848.519197`
  - 03-01 -> 03-03 return: `-0.204960%`
- alpha360
  - `2017-03-01`: `1,017,459.605141`
  - `2017-03-02`: `1,013,940.483902`
  - `2017-03-03`: `1,012,335.788367`
  - 03-01 -> 03-03 return: `-0.503589%`

So within this exact 3-day replay, alpha158 realized the better path: it lost about `20.5 bps`, while alpha360 lost about `50.4 bps`.

### Window trading friction (`tasks.result.trade_history`)
- alpha158, 2017-03-01~2017-03-03
  - commission = `958.641850`
  - slippage = `481.162996`
  - realized sell pnl = `+623.969952`
- alpha360, 2017-03-01~2017-03-03
  - commission = `993.736697`
  - slippage = `499.788996`
  - realized sell pnl = `-5167.079475`

alpha360 paid slightly more friction (`+35.094846` commission, `+18.626000` slippage) and, more importantly, realized much worse sells.

## 3. Why the local alpha158 edge does not cleanly flip the formal March result

Prior divergence slicing already showed that `2017-03` is locally alpha158-favored, and specifically on `2017-03-03`:
- alpha158 top2: `601288.SH`, `600519.SH`
- alpha360 top2: `601288.SH`, `000001.SZ`
- diff-stock 5d: `600519.SH = +0.037331`, `000001.SZ = 0.000000`

That ranking evidence is real, and this replay shows it did help locally: by 03-03 close, actual holdings had converged to exactly those two different baskets:
- alpha158 EOD 03-03: `600519.SH + 601288.SH`
- alpha360 EOD 03-03: `601288.SH + 000001.SZ`

But it does not immediately become a full realized-month win for two concrete, mechanical reasons:

1. `topk=2, n_drop=1` only allows one replacement per day.
   - The executor literally sells only `n_drop` worst held names and buys only `n_drop` replacements (`trade_modes.py:171-172, 227-230, 252-279`).
   - Entering 03-03, both models still carried `000651.SZ`, so alpha158 only reached its preferred `601288.SH + 600519.SH` basket at the end of 03-03, not before. The 5-day forward edge cited above therefore mostly starts accruing after this replay window.

2. Alpha360 entered the window with a cushion and remained ahead in absolute NAV despite underperforming inside the window.
   - NAV gap on 03-01: `1,017,459.605141 - 1,011,922.557484 = 5,537.047657`
   - NAV gap on 03-03: `1,012,335.788367 - 1,009,848.519197 = 2,487.269171`
   - So alpha158 narrowed the gap by about `3,049.778486`, but did not erase it inside 03-01~03-03.

Bottom line:
- The local 2017-03 divergence advantage for alpha158 does show up in realized replay over 03-01~03-03.
- But the formal March result still stayed in alpha360's favor at month level: `2017-03` return was `-0.005427` for alpha158 vs `-0.000712` for alpha360.
- The replay evidence above shows why that can happen without contradiction: the better alpha158 basket is only fully installed at 03-03 close under `n_drop=1`, while alpha360 still carried a pre-existing NAV lead into the window.
