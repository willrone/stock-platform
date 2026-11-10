# 策略赛马数据口径修正记录

发现时间：2026-10-28
范围：`backend/app/services/trading/strategy_race.py` + `frontend/src/app/paper-portfolio/page.tsx`

---

## 一、结论先说

策略赛马页原本展示的「年换手率」**全部是错的**，且页面上没有任何地方能看出它是错的。
三个错误叠加，把换手率放大了约 1 倍，并让 multi_factor 得出「年换手 2619%」的荒谬值。

同时发现一个**更值得警惕的问题**：主线策略 3 年累计 +35.5%，而基准（全市场等权买入持有）
+56.0% —— **主线跑输买入持有 20.5 个百分点**。原页面把两者并排列出，却没有任何一句话
点明这个事实，读者只会看到「主线 +35.5%」这一行。

---

## 二、换手率的三个错误

### 错误 1：把「单边成交笔数」当成「完整买卖回合」

引擎实现（`backend/app/services/backtest/core/portfolio_manager_array.py`）：

- L398/L404：买入成交 → `self.trades.append(...)`，`action="BUY"`
- L518/L523：卖出成交 → `self.trades.append(...)`，`action="SELL"`

即**买入记 1 次、卖出记 1 次**，`total_trades = len(trades)` 是**单边成交笔数**。
`win_rate` 用的是 `sell_trades`（按平仓笔算胜率），两者分母不同。

而 `strategy_race.py` 的旧代码注释写死了：

> 注意：`total_trades` 是**完整买卖回合**（一买一卖记 1 笔），
> 所以除以持仓只数已经天然是「双边」口径，不再乘 2。

这段注释与引擎实现直接矛盾。**实际影响**：把单边笔数直接当回合数除以持仓数，
换手率被高估约 1 倍。实测校验（multi_factor，v2 产物逐位复现）：

| | 旧口径（错） | 新口径（对） |
|---|---|---|
| 卖出笔数 | — | 1164（实测） |
| 年化平仓笔数 | 2357/3 = 785.7 | 1164/3 = 388.0 |
| 真实平均持仓 | 28.22（实测） | 28.22 |
| 年换手率 | 2786% | **1375%** |

### 错误 2：分母用 topk 猜值，不是实测持仓数

旧代码用 `STRATEGY_TOPK`（策略类的 topk **目标**持仓数）作分母。topk 是目标，
不是回测里**实际**持有的只数 —— 现金预留、建仓未满、整手约束都会让实际持仓低于 topk。

实测：multi_factor 的 topk=30，但回测里真实平均持仓 **28.22** 只（最大 30）。
honest_composite 的 topk=60，真实平均持仓 **58.01** 只。

### 错误 3：披露文案与事实不符

旧披露文案：

> 年换手率 = 年化交易笔数 / 持仓只数（topk 取策略类默认值）；一笔 = 完整买卖回合。

两处都错（见错误 1、2），且「一笔 = 完整买卖回合」与引擎实现矛盾。

---

## 三、修正方案

### 后端

1. 新增 `backend/scripts/recompute_turnover_truth.py` —— 用与基准产物**逐位一致**的
   口径重跑 5 条主策略，导出实测的 `sell_trades` / `buy_trades` / `avg_positions`。
2. 产物落盘 `data/backtest/turnover_truth.json`。
3. `strategy_race.py` 改为从该产物读实测值；实测数据缺失时降级到 topk 兜底，
   并把 `positions_source` 标成 `topk_fallback`，前端会明确标注「topk 估算」。

### 口径复现的关键（踩过的坑）

`data/backtest/ALL_real_lag1_v2.json` 的**生成脚本已不在仓库里**（是临时脚本），
所以参数必须逆向复原。已逐位校验复原正确的口径：

| 策略 | rebalance_days | total_return | sharpe | max_dd | trades |
|---|---|---|---|---|---|
| honest_composite | **weekly** | 0.35505797455182114 | 0.8795668740640868 | -0.1337121157482497 | 1190 |
| multi_factor | **daily** | -0.3600687175588425 | -0.4523391124517872 | -0.6024189532939825 | 2357 |

两个要点：

1. **`rebalance_days` 是策略侧参数**，引擎不消费 `BacktestConfig.rebalance_frequency`
   —— 必须在 `strategy_config` 里传。策略类默认值都是 `"daily"`。
2. **v2 产物用的是混合频率**：honest_composite 是 weekly，multi_factor 是 daily。
   一开始我按「全 daily」跑，4 个策略笔数与 v2 差 4~5 倍（与 daily/weekly
   交易日数之比 757/156≈4.85 吻合），据此定位到其余 3 个也是 weekly。

### 防回归

`recompute_turnover_truth.py` 末尾会与 v2 基准**逐位比对**
（total_return / sharpe_ratio / max_drawdown / total_trades，容差 1e-12），
不一致直接 `SystemExit(1)`。口径一旦漂移立刻失败，不会静默产出错误数字。

---

## 四、新增：跑输基准必须说出口

`strategy_race.py` 新增 `_benchmark_comparison()`，把结论直接算出来返回：

```
主线 3 年累计        +35.5%
基准（等权买入持有）  +56.0%
超额                -20.5 个百分点
```

但诚实结论要说清另一半：

| | 主线 | 基准 | 谁赢 |
|---|---|---|---|
| 3 年累计 | +35.5% | +56.0% | **基准**（差 20.5 点） |
| 年化 | +10.6% | +16.8% | **基准** |
| 夏普 | 0.880 | 0.737 | **主线** |
| 最大回撤 | -13.4% | -29.9% | **主线**（好 16.5 点） |

结论：**风险调整后收益与回撤优于基准，但绝对收益跑输基准 —— 这是「少亏」而不是「多赚」。**

前端在赛马表上方加了这个诚实对照 Alert（`severity=warning`，
跑赢时才用 `success`），免费档也可见（`benchmark_comparison` 走 `**result` 保留，
不在裁剪列表里）。

---

## 五、前端同步改动

`frontend/src/app/paper-portfolio/page.tsx`：

1. 新增「与基准的诚实对照」Alert（见上）。
2. 年换手率列头补副标题「持仓位年换手次数」，避免被误读成 A 股标准的成交额换手率。
3. 换手率单元格下方显示真实持仓只数；`positions_source === 'topk_fallback'`
   时以 warning 色标注「topk 估算」。
4. 交易笔数列下方显示「买 X / 卖 Y」，让单边计数口径可见。

---

## 六、遗留 / 后续

- **v2 产物的生成脚本已丢失**。本轮靠逐位复现逆向复原了参数，并把该过程固化到
  `recompute_turnover_truth.py`。但 v2 本身（`ALL_real_lag1_v2.json`）如果需要
  重新生成，仍需恢复生成脚本 —— 建议后续把它并入 `recompute_turnover_truth.py`
  的能力（一个脚本同时产出 v2 与 truth，保证两者永远同源）。
- `STRATEGY_TOPK` 表目前只在实测数据缺失时作为降级兜底。若
  `turnover_truth.json` 长期缺失，页面会退化为「topk 估算」，需在运维层面保证
  该产物的生成（可挂到 CI 或数据刷新流程）。
- 换手频率分档阈值（低频 <300% / 中频 <800% / 高频 ≥800%）在口径修正后
  重新评估过：honest_composite 约 325%（中频）、multi_factor 约 1375%（高频），
  分档仍然能区分开，无需调整。

---

## 七、验证方式

```bash
cd backend
.venv/bin/python scripts/recompute_turnover_truth.py
# 末尾输出：✓ 与 v2 基准逐位一致（校验 5 个策略），不一致则退出码 1

.venv/bin/python -c "from app.services.trading.strategy_race import strategy_race_service; ..."
# 检查 turnover.positions_source == 'measured'、benchmark_comparison.verdict 非空
```

浏览器：登录后访问 `/paper-portfolio`，确认
- 赛马表上方出现「与基准的诚实对照」黄色提示
- 换手率列下方显示「持仓 58.01 只」这类实测值
- 交易笔数列下方显示「买 625 / 卖 565」
