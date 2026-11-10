# 2026-10-19 研究日报：执行层成本、因子重选与 22f 候选

> 本文是 2026-10-19 晚间研究的完整落盘。结论优先：**注册版 42f 的真实 10 年 Sharpe 是 0.8193，不是 0.976；日频被实盘成本打死，周频全面更优；22f 基础版是第一个通过多年 OOS 的候选，但只够纸面跟，不够上真钱。**

## 0. TL;DR

| # | 结论 | 关键数字 |
|---|---|---|
| 1 | **0.976 不可复现** | 注册版 42f topk60 = **0.8193**；0.976 来自已丢失的 31 因子临时映射 |
| 2 | **topk30 优于 topk60** | 0.8865 vs 0.8193，收益 +208.6% vs +161.2%，回撤深 2pp |
| 3 | **日频被成本打死** | lag1 + 印花税 + 0.1% 滑点 = **6.18% 年化 / -46.83% 回撤 / 0.486 夏普** |
| 4 | **周频全面更优** | **7.29% / -37.99% / 0.602**，交易 2766 笔 vs 日频 13860 笔 |
| 5 | **池子卫生证伪** | topk30 夏普 0.8865 → **0.2714**；alpha 就在被剔的小盘/低价/低流动性票里 |
| 6 | **42f 的 OOS 是平的** | 2016-2021（真 OOS）夏普仅 **0.041**；10 年数字由 2021-2026（in-sample）扛着 |
| 7 | **22f 是首个过多年 OOS 的候选** | 2021-2026 OOS **9.38% / -27.13% / 0.685**（周频 + 实盘成本） |
| 8 | **滚动重选 ≈ 固定 22f** | 0.6854 vs 0.6736；edge 不是滚动重选，是 **MAX 符号修正** |
| 9 | **ST 污染回测** | 9.5% 涨停掩码对 ST（5%）失效，历史收益可能高估；实盘信号已剔 ST |
| 10 | **当前动作** | 22f 周频纸面信号 + 账本已产出；先纸面 4-8 周，不上真钱 |

---

## 1. 口径与数据

- **窗口**：2016-05-12 → 2026-05-08（10 年，2424 交易日）
- **宇宙**：point-in-time 4329 只（2021-05 时点已上市；避免用当前 5595 只全池引入 1266 只 2021 后上市的票）
- **因子**：
  - `42f`：23 基础公式（MIN/MAX/STD/CORR/CORD × 5/10/20/30/60）+ 17 Alpha158 面板因子
  - `22f`：23 基础公式中剔除 MAX5/MAX10/MAX20，保留 22 个；符号按 2016-2021 训练窗重估
- **执行**：
  - 日频：每天用当日分（lag0）或前一日分（lag1）
  - 周频：周一调仓，用上一交易日（通常周五）的分；对应"周五收盘后出信号 → 周一执行"
- **成本口径**：
  - 回测：佣金 0.03% 双边 + 滑点 0.01% 双边，**无印花税**
  - 实盘：买入 0.03%，卖出 0.08%（佣金 0.03% + 印花税 0.05%），滑点 0.1% 双边
- **数据**：`data/parquet/stock_data`（tdx_pipeline 合并；不复权价 + 除权事件，volume 单位=手）；qlib panel 仅 2021-05 起

---

## 2. TopK 集中度扫描（42f / PIT 4329 / 日频 lag0 / 回测成本）

来源：`data/strategy_baseline/ic_composite_10y_registered_topk.json`

| topk | n_drop | Sharpe | 总收益 | 年化 | 最大回撤 | OOS 16-21 | IN 21-26 | 交易 |
|---|---|---|---|---|---|---|---|---|
| 60 | 5 | 0.8193 | +161.2% | 10.50% | -36.67% | 0.4957 | 1.0821 | 22130 |
| 40 | 4 | 0.7643 | +151.4% | 10.06% | -37.08% | 0.4315 | 1.0318 | 18142 |
| **30** | **3** | **0.8865** | **+208.6%** | **12.43%** | **-38.63%** | **0.5258** | **1.1683** | **13846** |
| 20 | 2 | 0.8191 | +195.4% | 11.92% | **-41.30%** | 0.3587 | 1.1704 | 9388 |

**结论**：
- topk30 在夏普、总收益、OOS 上全面优于 topk60；代价是回撤深 2pp。
- topk20 是风险悬崖：回撤 -41.3%，OOS 0.359 最差。**30 是下限，不要再往下。**
- 持仓数量不是收益来源，是风险刻度；真正赚钱的是 42f 的选股信号。

---

## 3. 0.976 的纠正（重要）

### 3.1 事实

- `ic_composite_10y_expand2.py` 在 2026-10-15 跑出的 `42_topk60` = **0.976**，依赖 `/tmp/factor_formula_map.json`（**31 个已验证因子** = 25 基础 + 6 个额外面板因子）。
- 该临时文件已丢失。重建的 25 项版本只得到 **0.8206**；注册版 42f 的精确实现（`daily_signal.compute_composite`，PIT 4329）得到 **0.8193**。
- **0.976 不是注册策略的真实数字**，是一个已不可复现的 31 因子配置。

### 3.2 影响

- 对外讲故事、写文档、做商用定价，**不能用 0.976**；只能用 **0.819**（或 22f 的 2021-2026 OOS 0.685）。
- `docs/backtest/FACTOR_IC_STUDY.md` 中所有"42f topk60 0.976"的引用已过时；本文与其顶部的更正说明为准。
- 那 6 个额外因子如果真能把 0.82 拉到 0.976，值得找回来；但它们也可能是内盘/涨停污染因子。**在验证之前，0.82 是唯一能信的数字。**

---

## 4. 执行层 A/B：日频 vs 周频，成本 vs lag

来源：`data/strategy_baseline/ic_composite_10y_execution_ab.json`

### 4.1 总表（topk30 / n_drop3）

| 单元 | 口径 | Sharpe | 总收益 | 年化 | 最大回撤 | 交易 |
|---|---|---|---|---|---|---|
| `daily_lag0_origcost` | 日频 lag0，回测成本 | 0.8865 | +208.6% | 12.43% | -38.63% | 13846 |
| `daily_lag0_livecost` | 日频 lag0，实盘成本 | 0.5289 | +88.6% | 6.82% | -43.99% | 13816 |
| `daily_lag1_livecost` | 日频 lag1，实盘成本 | **0.4865** | +78.0% | **6.18%** | **-46.83%** | 13860 |
| `weekly_lag1_livecost` | 周频 lag1，实盘成本 | **0.6022** | +96.7% | **7.29%** | **-37.99%** | **2766** |

### 4.2 成本数学

- 日频 13846 笔 / 2424 天 = **5.7 笔/天**；单边换手约 18%/天；年化约 44x；round-trip 约 21.9x。
- 实盘比回测多：印花税 0.05% + 滑点差 0.09%×2 = **0.23%/round-trip**。
- 额外成本 ≈ 21.9 × 0.23% = **5.0%/年**。12.43% − 5.0% ≈ 7.4%，与实测 6.82% 同量级（复利/取整差异）。
- 周频 2766 笔 / 2424 天 = 1.14 笔/天；成本拖累降到约 **1.2%/年**。

### 4.3 分年（lag1 + 实盘成本）

| 年份 | 日频 | 周频 |
|---|---|---|
| 2016 | +12.9% | +10.7% |
| 2017 | -3.2% | -14.2% |
| **2018** | **-41.0%** | **-24.8%** |
| 2019 | +20.8% | +17.0% |
| 2020 | +26.9% | +8.0% |
| 2021 | +78.3% | +53.2% |
| 2022 | -22.9% | **+3.8%** |
| 2023 | -2.2% | +0.9% |
| 2024 | +18.5% | +28.5% |
| 2025 | +7.7% | +3.1% |
| 2026 | +1.1% | +1.5% |

**结论**：
- **日频死了**：6.18% 年化配 -46.83% 回撤，夏普 0.486，不值得做。
- **周频全面更优**：成本低、回撤浅、2018 和 2022 都明显更好。要交易就交易周频。
- lag1 本身也伤：日频 lag0 实盘成本 0.529 → lag1 0.486，2018 从 -35.0% 恶化到 -41.0%。

---

## 5. 池子卫生（证伪）

来源：`data/strategy_baseline/ic_composite_10y_hygiene.json`

过滤规则（point-in-time）：
- 剔除北交所（.BJ）
- 收盘价 < 2 元
- 20 日均成交额 < 5000 万元
- 上市 < 60 个交易日
- **ST 未纳入**（项目内无历史 ST 标记；qlib 面板无 isST，tdx 原始包仅 108 天）

| 单元 | Sharpe | 总收益 | 年化 | 最大回撤 | 交易 |
|---|---|---|---|---|---|
| `base_topk60` | 0.8193 | +161.2% | 10.50% | -36.67% | 22130 |
| `base_topk30` | 0.8865 | +208.6% | 12.43% | -38.63% | 13846 |
| `hygiene_topk60` | 0.3735 | +50.1% | 4.31% | -37.61% | 19264 |
| `hygiene_topk30` | 0.2714 | +31.3% | 2.87% | -39.67% | 12758 |

**结论**：
- 过滤只留下 23.3% 的股票-日（砍掉 77% 池子），夏普腰斩、收益砍掉 70-85%，**回撤几乎没降**。
- **alpha 就在被剔的小盘/低价/低流动性票里**。所谓"垃圾股"是收益来源，不是回撤来源。
- 选股层没有免费午餐；不要再试池子过滤。

---

## 6. Walk-forward 单次 split

来源：`data/strategy_baseline/ic_composite_walkforward.json`

- 训练：2016-05-12 → 2021-05-11（23 基础公式；剔除训练窗最后 5 天，避免 fwd5 边界前视）
- 测试：2021-05-12 → 2026-05-08
- 选因子：|IC| > 0.01、|t| > 2、训练窗前后两半同号；选中 **22/25**（只剔除 MAX5/MAX10/MAX20）

| 单元 | 期间 | Sharpe | 总收益 | 年化 | 最大回撤 | 交易 |
|---|---|---|---|---|---|---|
| `selected_train` | 2016-2021 | 0.309 | +17.0% | 3.32% | -44.45% | 1392 |
| `selected_test` | 2021-2026 | **0.5276** | +38.1% | 6.97% | -27.54% | 1374 |
| `all23_train` | 2016-2021 | -0.0362 | -10.7% | -2.32% | -56.42% | 1386 |
| `all23_test` | 2021-2026 | 0.3439 | +24.7% | 4.71% | -34.10% | 1392 |

**结论**：
- 选择有效：`selected_test` 0.528 > `all23_test` 0.344；剔除 MAX5/10/20 贡献 +0.18 OOS 夏普。
- 但 edge 很弱：`selected_train` 只有 0.309（年化 3.3%、回撤 -44.5%）。训练期本身就勉强，测试期反而更好——不像稳定 edge，更像 2021-2026 小盘/反转顺风。
- **符号翻转**：MAX30/MAX60 在 2016-2021 的 IC 是**正的**（+0.0123/+0.0101），而 42f 硬编码 MAX = **-1**（来自 2021-2026 的 IC）。42f 是"用未来定过去"，OOS 当然平。

---

## 7. 滚动 Walk-forward

来源：`data/strategy_baseline/ic_composite_rolling_walkforward.json`

滚动规则：每年用过去 5 年重估 IC + 符号，只对下一年打分；拼成 2021-2026 连续 OOS 曲线。
执行：周频周一 + lag1 + 实盘成本（印花税 0.05% + 滑点 0.1%）。

| 单元 | Sharpe | 总收益 | 年化 | 最大回撤 | 交易 |
|---|---|---|---|---|---|
| `rolling_wf` | 0.6854 | +58.3% | 9.38% | -27.13% | 1482 |
| `fixed_22f` | 0.6736 | +58.4% | 9.40% | -29.07% | 1482 |
| `all23` | 0.3480 | +26.9% | 4.76% | -34.65% | 1464 |

### 分年

| 年份 | rolling_wf | fixed_22f | all23 |
|---|---|---|---|
| 2021 | **+41.7%** | +41.4% | +37.8% |
| 2022 | -10.4% | -10.1% | -11.4% |
| 2023 | -6.3% | -9.5% | -8.9% |
| 2024 | +14.9% | +11.8% | +2.1% |
| 2025 | +7.4% | +14.3% | +15.8% |
| 2026 | +6.4% | +7.1% | -4.2% |

**结论**：
- **rolling ≈ fixed**：0.6854 vs 0.6736。年度重选几乎没用；不用搞复杂滚动，固定 22f 就够。
- **all23 明显更差**：0.348 vs 0.685。选择和符号修正才是真增量。
- **22f 是首个通过多年 OOS 的候选**：2021-2026 年化 9.38%、回撤 -27.13%、夏普 0.685。
- **但别高兴太早**：2022-2023 是负的（-10.4%、-6.3%），2021 +41.7% 贡献太大；22f 在 2016-2021 训练期只有 0.309，全 10 年大概 0.5 夏普。**它是纸面候选，不是重仓策略。**

---

## 8. ST 污染（新发现的回测漏洞）

- 引擎的涨停掩码是**固定 9.5%**（`honest_composite_strategy.py` 默认 `limit_up=0.095`）。
- A 股实际涨跌停：主板 10%、创业板/科创板 20%、北交所 30%、**ST 5%**。
- 对 ST：9.5% 掩码**太松**——ST 在 +5% 就涨停，回测却以为还能在收盘买入 → 可能买进不可买的涨停 ST 票，收益被高估。
- 对创业板/科创板/北交所：9.5% 掩码**太紧**——会误剔一些可买票，但不影响"不可买"的安全性。
- 实盘周频信号已用 tdx 当日 `name` 剔除 ST；**历史 ST 标记缺失**，42f/22f 的回测数字可能仍含 ST 涨停污染。
- 待办：用 Baostock `isST` 或历史名称表补 ST 标记，重跑 22f 量化污染幅度。

---

## 9. 当前候选与结论

### 9.1 候选

**22f 基础版 + 周频 + 实盘成本**（MIN +1、STD/CORR/CORD -1、MAX30/60 +1；剔除 MAX5/10/20）：

- 2021-2026 OOS：**9.38% 年化 / -27.13% 回撤 / 0.685 夏普**（1482 笔）
- 2016-2021 训练：3.32% / -44.45% / 0.309
- 全 10 年粗估：约 0.5 夏普

### 9.2 结论

- **42f 暂不用于实盘**：0.976 不可复现，17 个面板因子 pre-2021 无法验证，OOS 仅 0.041。
- **22f 可以纸面跟**：它是第一个通过多年 OOS 的候选，但 edge 弱、年份不稳（2022-2023 负）。
- **不要上真钱**：先纸面 4-8 周，和回测对得上，再谈小仓真钱。
- **不要再试**：池子过滤、MA200/vol/regime 择时、止损、按分数加权、自适应 topk——全部证伪。

---

## 10. 产物与复现

### 10.1 脚本

| 脚本 | 用途 |
|---|---|
| `backend/scripts/daily_signal.py` | 42f 信号核心（打分/买卖/持有） |
| `backend/scripts/weekly_signal.py` | **22f 周频纸面信号 + 账本**（当前产品路径） |
| `backend/scripts/ic_composite_10y_registered_topk.py` | topk 集中度扫描（42f/PIT） |
| `backend/scripts/ic_composite_10y_execution_ab.py` | 日频 vs 周频、lag0 vs lag1、回测 vs 实盘成本 |
| `backend/scripts/ic_composite_10y_hygiene.py` | 池子卫生 A/B |
| `backend/scripts/ic_composite_walkforward.py` | 单次 split walk-forward |
| `backend/scripts/ic_composite_rolling_walkforward.py` | 滚动 walk-forward |

### 10.2 结果 JSON

- `data/strategy_baseline/ic_composite_10y_registered_topk.json`
- `data/strategy_baseline/ic_composite_10y_execution_ab.json`
- `data/strategy_baseline/ic_composite_10y_hygiene.json`
- `data/strategy_baseline/ic_composite_walkforward.json`
- `data/strategy_baseline/ic_composite_rolling_walkforward.json`

### 10.3 纸面信号

- `data/signals/paper_holdings.json`：22f 纸面组合（30 只，现金 ¥72,893，总值 ¥998,796）
- `data/signals/ledger.csv`：22f 逐笔账本
- `data/signals/signal_2026-10-16.json`：首期信号快照
- 旧 42f 纸面：`data/signals/*_42f.*`

### 10.4 复现命令

```bash
cd ~/Projects/stock-platform

# topk 扫描
env -u PYTHONPATH PYTHONHASHSEED=0 BASELINE_LOG_LEVEL=ERROR \
  backend/.venv-py313/bin/python3 backend/scripts/ic_composite_10y_registered_topk.py

# 执行层 A/B
env -u PYTHONPATH PYTHONHASHSEED=0 BASELINE_LOG_LEVEL=ERROR \
  backend/.venv-py313/bin/python3 backend/scripts/ic_composite_10y_execution_ab.py

# 池子卫生
env -u PYTHONPATH PYTHONHASHSEED=0 BASELINE_LOG_LEVEL=ERROR \
  backend/.venv-py313/bin/python3 backend/scripts/ic_composite_10y_hygiene.py

# walk-forward（单次 + 滚动）
env -u PYTHONPATH PYTHONHASHSEED=0 BASELINE_LOG_LEVEL=ERROR \
  backend/.venv-py313/bin/python3 backend/scripts/ic_composite_walkforward.py
env -u PYTHONPATH PYTHONHASHSEED=0 BASELINE_LOG_LEVEL=ERROR \
  backend/.venv-py313/bin/python3 backend/scripts/ic_composite_rolling_walkforward.py

# 周频 22f 纸面信号（周五收盘后）
env -u PYTHONPATH PYTHONHASHSEED=0 BASELINE_LOG_LEVEL=ERROR \
  backend/.venv-py313/bin/python3 backend/scripts/weekly_signal.py
```

---

## 11. 下一步

1. **纸面跟 22f 周频 4-8 周**：每周五跑信号，记账；每周复盘组合 vs 中证500/中证1000。
2. **修 fill 口径**：当前纸面按周五收盘成交（比回测 lag1 乐观一个隔夜跳空）；加 `--fill` 用周一收盘价重记，和回测对齐。
3. **ST 历史标记**：Baostock `isST` 或历史名称表；重跑 22f 量化收益高估幅度。
4. **面板因子验证**：2021-2023 训练 → 2024-2026 测试，看 17 个 Alpha158 因子有没有 OOS 价值；没有就退役。
5. **滚动 walk-forward 扩展**：expanding window + 多年测试 + DSR/PBO，确认 22f 的 0.685 不是单一路径的运气。
6. **cron（暂缓）**：用户暂不启用；将来可设周五 16:05 自动跑数据 + 信号 + 微信推送。

---

## 12. 备注：Hermes 后台任务通知机制

- `delegate_task` 后台跑完**不会主动推送到微信**：
  - `gateway/wake.py`：async-delegation completion 写入 durable DELIVERY row，由 **TUI/desktop 的 poller** 读取；不主动 POST 新 turn。
  - 微信 adapter 是入站消息驱动，没有 poller；只在下一条用户消息时读取。
- **TUI/desktop 自带窗口没有这个问题**（有 live poller）。
- 要在微信主动推送：① 同回合同步等（foreground 等进程结束）；② cronjob 定时任务，deliver 到当前会话。
- `terminal(background=true, notify=true)` 只是系统通知，不是 agent 消息。

---

*生成时间：2026-10-19 22:20 CST。所有数字均取自 `data/strategy_baseline/*.json` 与 `data/signals/*.json` 原件；未做二次估算（成本数学小节标注为近似）。*
