# alpha158 vs alpha360 holdings-aware replay (2019-07-01 ~ 2019-07-03)

日期：2026-04-14

## Scope / sources
- DB: `backend/data/app.db`
  - `trade_records`
  - `portfolio_snapshots`
  - `tasks.result.trade_history`
  - `tasks.result.monthly_returns_detail`
- Executor logic: `backend/app/services/backtest/execution/trade_modes.py:171-172, 227-230, 252-279`
- Prior local slice reference: `docs/reports/2026-04-14-alpha158-alpha360-divergence-event-slicing.md` section `2.4 2019-07`

注：这几天 `portfolio_snapshots.positions` 仍是空 `{}`，所以“实际持仓”按 `trade_records` 累积重建；NAV / cash 直接取 `portfolio_snapshots`。

## 1. Actual replay: holdings and trades

### Entering 2019-07-01
两边在 2019-06-28 收盘后其实拿的是同一组名字，只是仓位大小不同：
- alpha158: `601288.SH` 46,800 + `600036.SH` 4,300
- alpha360: `601288.SH` 42,800 + `600036.SH` 4,000

### Date-by-date executed path
| Date | alpha158 executed trades | alpha158 EOD holdings | alpha360 executed trades | alpha360 EOD holdings |
|---|---|---|---|---|
| 2019-07-01 | SELL `600036.SH` 4,300 @ `36.881552` (pnl `+3561.430520`); BUY `000001.SZ` 11,100 @ `13.936965` | `601288.SH + 000001.SZ` | SELL `601288.SH` 42,800 @ `3.618190` (pnl `+469.203397`); BUY `600519.SH` 100 @ `1032.375915` | `600036.SH + 600519.SH` |
| 2019-07-02 | SELL `000001.SZ` 11,100 @ `14.172910` (pnl `+2383.010540`); BUY `600036.SH` 4,200 @ `36.538260` | `601288.SH + 600036.SH` | SELL `600036.SH` 4,000 @ `36.501740` (pnl `-7808.812887`); BUY `000001.SZ` 11,400 @ `14.187090` | `000001.SZ + 600519.SH` |
| 2019-07-03 | SELL `600036.SH` 4,200 @ `36.131927` (pnl `-1934.233655`); BUY `600519.SH` 100 @ `988.393974` | `601288.SH + 600519.SH` | SELL `600519.SH` 100 @ `987.406074` (pnl `-4645.095005`); BUY `600036.SH` 4,100 @ `36.168077` | `000001.SZ + 600036.SH` |

Most important replay fact:
- 两边是从同一组持仓名字 (`601288.SH + 600036.SH`) 进入窗口，但 3 天里走出了完全相反的轮动路径。
- alpha158 在窗口内一直保留 `601288.SH`，只轮换另一个名额；alpha360 则先卖 `601288.SH` 换 `600519.SH`，再卖 `600036.SH` 换 `000001.SZ`，再卖 `600519.SH` 换回 `600036.SH`。

## 2. Realized portfolio path and costs

### NAV path from `portfolio_snapshots`
- alpha158
  - `2019-06-28`: `944,749.803217`
  - `2019-07-01`: `949,169.928184`
  - `2019-07-02`: `952,412.838143`
  - `2019-07-03`: `951,392.480897`
  - `2019-06-28 -> 2019-07-03`: `+0.703115%`
- alpha360
  - `2019-06-28`: `915,798.830881`
  - `2019-07-01`: `919,921.870416`
  - `2019-07-02`: `917,262.124748`
  - `2019-07-03`: `911,268.368347`
  - `2019-06-28 -> 2019-07-03`: `-0.494701%`

So this exact 3-day cluster favored alpha158 by about `+1.197816%` relative return.

### Window trading friction and realized sells (`tasks.result.trade_history`)
- alpha158, 2019-07-01~2019-07-03
  - trades = `6`
  - commission = `904.996304`
  - slippage = `437.347512`
  - realized sell pnl = `+4010.207405`
- alpha360, 2019-07-01~2019-07-03
  - trades = `6`
  - commission = `806.038912`
  - slippage = `406.429504`
  - realized sell pnl = `-11984.704495`

This matters because alpha360 did **not** lose this window due to higher trading friction:
- it actually paid slightly lower commission/slippage than alpha158,
- but its realized sells were worse by `-15994.911900`.

## 3. Why the 2019-07 local divergence slices do or do not translate

Prior event slicing for the whole month said:
- divergence-day avg 1d basket return: alpha158 `-0.001470`, alpha360 `-0.002286`
- divergence-day avg 5d basket return: alpha158 `-0.011188`, alpha360 `-0.007245`
- example on `2019-07-02`: alpha158 top2 = `601288.SH + 600036.SH`, alpha360 top2 = `000001.SZ + 600519.SH`
- full-month realized return: alpha158 `-0.002506`, alpha360 `-0.017555`

Holdings-aware replay makes the bridge much clearer:

1. On this exact cluster, the local divergence **did translate** into realized PnL, and strongly favored alpha158.
   - By `2019-07-02` close, actual holdings had fully diverged into the same baskets highlighted by the slice:
     - alpha158: `601288.SH + 600036.SH`
     - alpha360: `000001.SZ + 600519.SH`
   - After that, alpha360 immediately had to unwind `600519.SH` on `2019-07-03` for another realized loss (`-4645.095005`).

2. The month-average 5d slice looked ambiguous, but the executed path was not ambiguous.
   - `topk=2, n_drop=1` means each model can replace only one held name per day.
   - Here that one-name-at-a-time path mattered a lot: alpha360 rotated through two losing sells (`600036.SH`, then `600519.SH`) while alpha158 realized two profitable sells before one smaller loss.
   - So the realized gap came from path-dependent basket installation/unwind, not from signal count or trading-cost inflation.

3. This 3-day cluster explains most of the July month gap.
   - Window relative gap (`2019-06-28 -> 2019-07-03`) = about `1.197816%`
   - Full July relative gap (`-0.250552% - (-1.755487%)`) = `1.504934%`
   - So roughly `79.6%` of the July alpha158-over-alpha360 gap was already created inside `2019-07-01 ~ 2019-07-03`.

## Bottom line
- For `2019-07-01 ~ 2019-07-03`, the holdings-aware replay is decisively alpha158-favored.
- The earlier 2019-07 month-level forward-slice ambiguity does **not** contradict the realized month result: the actual executed path in this cluster already generated most of the eventual July performance gap.
- The key mechanism was not extra cost or extra trade count; it was alpha360's realized path through the wrong names (`600036.SH`, then `600519.SH`) after both models entered the window from the same starting basket.
