# 策略好用可信专项实施路线图

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** 把 stock-platform 的回测策略从“能跑”提升到“可解释、可复现、可横向比较、可做样本外验证”的策略研究平台。

**Architecture:** 不先追求更复杂的 Alpha，而是先补齐可信回测的基础设施：数据质量门、交易仿真规则、样本内/样本外验证、统一 Benchmark、策略配置资产化、报告解释层。技术策略和模型策略统一走同一套可信度评分卡。

**Tech Stack:** FastAPI + SQLite/SQLAlchemy + Pandas/NumPy + 现有 BacktestExecutor / PortfolioManagerArray / StrategyFactory / Qlib 数据目录。

---

## 0. 当前状态快照（2026-05-24）

### 0.1 已具备能力

- `/api/v1/backtest/strategies` 正常返回策略列表。
- 当前暴露 23 个策略 key：
  - 技术类：`moving_average`, `rsi`, `macd`, `bollinger`, `stochastic`, `cci`, `kdj`, `obv`
  - 统计套利：`pairs_trading`, `mean_reversion`, `cointegration`
  - 因子类：`value_factor`, `momentum_factor`, `low_volatility`, `multi_factor`
  - 模型/排序类：`ml_ensemble_lgb_xgb_riskctl`, `model_signal`, `model`, `signal`, `model_topk_dropout`, `topk_dropout`, `model_ranking`, `ranking`
- `/api/v1/backtest` 同步回测链路可以跑通。
- 数据底座存在：
  - `data/qlib_official_data`: 约 27567 个文件
  - `data/qlib_data`: 约 33080 个文件
  - `data/parquet`: 约 5523 个文件
- `BacktestConfig` 已有基础成本参数：
  - `commission_rate`
  - `slippage_rate`
  - `open_cost`
  - `close_cost`
  - `min_cost`
  - `board_lot_size`
  - `max_position_size`
  - `cash_reserve_ratio`
- `PortfolioManagerArray` 已支持：
  - 佣金
  - 滑点
  - 100 股手数取整
  - 最大单股仓位
  - 现金保留
- 历史全市场 MA 任务可产出大量信号和交易。

### 0.2 最新历史回测证据

任务：`7228187e full-market-ma-available-5y-20210511-20251231`

配置：

```json
{
  "strategy_name": "moving_average",
  "stock_count": 5513,
  "initial_cash": 1000000,
  "strategy_config": {
    "short_window": 5,
    "long_window": 20,
    "signal_threshold": 0.005
  }
}
```

结果摘要：

```text
total_signals: 127329
buy_signals: 64402
sell_signals: 62927
executed_signals: 26066
total_trades: 26066
execution_rate: 20.47%
win_rate: 17.30%
total_pnl: 192467.44
commission: 14508.02
stocks traded: 3478
profitable_stocks: 1358
trading_days: 1097
```

判断：

- 平台能跑大规模回测。
- MA 不再是 0 信号。
- 但策略胜率低，收益来源可能高度集中，必须补解释、稳健性和样本外验证。

---

## 1. 研究结论：可信策略必须过哪些关

参考方向：

- QuantStart 的 backtesting pitfalls：重点强调 optimization bias、look-ahead bias、survivorship bias、transaction cost、psychological tolerance 等风险。
- Qlib 的报告体系：强调组合收益、基准收益、超额收益、最大回撤、IR、月度表现、position/report 分析。
- López de Prado 金融机器学习方法：时间序列交叉验证不能直接用普通 k-fold，需要 purging / embargo 防泄漏。
- 常见 Walk-forward Analysis：参数优化只在训练窗口做，随后在未来窗口验证，滚动推进，避免只在全样本里调参。

落到本项目，策略“好用可信”至少要满足 6 层：

1. **数据可信**：股票池、日期、OHLCV、停牌、缺失、复权、幸存者偏差有记录。
2. **交易仿真可信**：A 股交易规则、费用、滑点、涨跌停、T+1、成交量约束可配置。
3. **策略信号可信**：不能偷看未来；每个信号能解释来源、指标值、参数、拒绝/执行原因。
4. **样本外可信**：固定 train/validation/test 或 walk-forward，不允许只看全样本最优。
5. **稳健性可信**：参数扰动、成本压力、股票池扰动、时间窗口扰动后结果不能崩。
6. **报告可信**：相对基准、收益分解、成本分解、个股贡献、交易分布、失败原因一眼可见。

---

## 2. 当前主要差距

### 2.1 API 参数说明和策略实现不一致

文件：

- `backend/app/services/backtest/strategies/technical/basic_strategies.py`
- `backend/app/api/v1/backtest.py`

现状：

- `MovingAverageStrategy` 默认 `signal_threshold=0.005`
- `/api/v1/backtest/strategies` 文档仍显示 `signal_threshold.default=0.02`

影响：

- 前端和用户看到的是旧阈值。
- 手动配置很容易复现“信号过少”的历史问题。

### 2.2 策略配置还没有资产化

接口：

- `/api/v1/strategy-configs`

现状：

```text
total_count: 0
```

影响：

- 平台里没有“默认可信策略配置”。
- 用户只能临时填参数，不知道哪个配置是 smoke、benchmark、research、production-like。

### 2.3 异步任务执行链路有进程池生命周期问题

最近任务里大量失败：

```text
任务提交失败: 进程池未启动，请先调用start()
```

影响：

- 同步 `/api/v1/backtest` 可用，但 `/api/v1/tasks` 异步回测/预测链路不够可信。
- 长回测依赖异步任务，所以这是策略研究平台化的前置问题。

### 2.4 A 股交易真实性仍不完整

已有：佣金、滑点、最低费用、手数、仓位、现金保留。

待补：

- T+1 卖出限制
- 涨跌停不可成交或成交价约束
- 停牌/无成交量日过滤
- ST/退市/新股上市天数过滤
- 成交量/成交额容量约束
- 印花税和券商佣金分项展示
- 股票池快照，避免幸存者偏差

### 2.5 成本前后对照不完整

文件：

- `backend/app/services/backtest/core/portfolio_manager_array.py`

现状：

- `PortfolioManagerArray` 维护了 without-cost 变量。
- 但 `get_performance_metrics_without_cost()` 目前返回 0 占位值。

影响：

- 没法可靠回答“策略毛收益还行，但被成本吃掉了吗？”
- 也没法做成本压力测试的可信报告。

### 2.6 缺标准化策略评测矩阵

当前有历史 MA 回测结果，但缺统一矩阵：

- 不同策略横向比较
- 不同成本假设比较
- 不同市场区间比较
- 不同股票池比较
- 样本内/样本外比较
- 相对基准比较

---

## 3. 专项目标定义

本专项不以“马上找到赚钱策略”为目标，而以“平台能诚实地区分好策略、坏策略、过拟合策略、数据幻觉策略”为目标。

### 3.1 策略可信度评分卡

每次策略评估输出一个 `trust_score`，由以下维度组成：

| 维度 | 分数 | 说明 |
|---|---:|---|
| data_quality | 0-20 | 缺失率、覆盖率、股票池一致性、异常价格 |
| execution_realism | 0-20 | 成本、滑点、T+1、涨跌停、容量约束 |
| robustness | 0-20 | 参数扰动、时间窗口扰动、成本压力 |
| out_of_sample | 0-20 | 样本外/WFO 是否仍有效 |
| explainability | 0-10 | 信号、交易、个股贡献是否可解释 |
| usability | 0-10 | 是否有默认配置、可复现命令、报告入口 |

建议评级：

- `A`: 85+，可以进入 paper/live shadow 观察
- `B`: 70-84，可以继续研究
- `C`: 50-69，只能作为实验候选
- `D`: <50，不建议使用

---

## 4. 分阶段实施计划

## Phase 1：可信回测最小闭环（P0）

目标：先让每个策略都有标准 smoke、参数配置、报告结构和一致口径。

### Task 1: 修正 MA 策略 API 默认参数

**Objective:** 消除 API 文档与策略实现不一致。

**Files:**

- Modify: `backend/app/api/v1/backtest.py`
- Test: `backend/tests` 或新增 `tests/scripts` 轻量 API 检查

**Steps:**

1. 将 `moving_average.parameters.signal_threshold.default` 从 `0.02` 改为 `0.005`。
2. 检查描述里明确写：默认 0.5%，历史上 2% 会过滤大部分金叉/死叉。
3. 跑：

```bash
curl -s http://127.0.0.1:18082/api/v1/backtest/strategies | python -m json.tool
```

Expected：`moving_average.signal_threshold.default == 0.005`。

### Task 2: 新增 make smoke-backtest

**Objective:** 固化“回测链路可用”的最小质量门。

**Files:**

- Create: `scripts/smoke-backtest.js`
- Create: `tests/scripts/smoke-backtest.test.js`
- Modify: `Makefile`
- Modify: `STARTUP.md`

**Smoke 必须检查：**

1. `GET /api/v1/backtest/strategies` 返回 200 且包含 `moving_average/rsi/macd/model_topk_dropout`。
2. `GET /api/v1/strategy-configs` 返回 200。
3. `POST /api/v1/backtest` 用 30 天本地样本跑通。
4. 响应字段必须包含：
   - `portfolio.initial_cash`
   - `portfolio.final_value`
   - `trading_stats.total_trades`
   - `risk_metrics.max_drawdown`
   - `dates`
5. 对短样本 0 交易不失败，但必须输出 warning：`短样本仅验证链路，不评价策略收益`。

Command：

```bash
make smoke-backtest
```

### Task 3: 建立默认策略配置种子

**Objective:** 把策略从“代码里有”变成“平台里有可复用配置”。

**Files:**

- Create: `backend/scripts/seed_strategy_configs.py`
- Possibly modify: `backend/app/api/v1/strategy_configs.py`
- Docs: `docs/backtest/strategy-config-presets.md`

**默认配置分层：**

1. `smoke/*`：保证有信号、跑得快，不评价收益。
2. `benchmark/*`：稳定基准，参数固定。
3. `research/*`：可参与优化和 WFO。
4. `model/*`：模型排序类策略。

建议初始配置：

- `benchmark/moving_average_5_20_threshold_005`
- `benchmark/rsi_optimized_default`
- `benchmark/macd_default`
- `benchmark/bollinger_20_2`
- `research/portfolio_technical_vote_v1`
- `model/topk_dropout_k10_drop2`

### Task 4: 补策略评估结果 schema

**Objective:** 为可信度报告准备稳定存储结构。

**Files:**

- Create migration: `backend/migrations/add_strategy_evaluation_tables.py`
- Create model: `backend/app/models/strategy_evaluation_models.py`

**Tables:**

1. `strategy_evaluations`
   - `evaluation_id`
   - `strategy_name`
   - `strategy_config_hash`
   - `universe_id`
   - `period_start`
   - `period_end`
   - `benchmark`
   - `trust_score`
   - `rating`
   - `summary_json`

2. `strategy_evaluation_slices`
   - `evaluation_id`
   - `slice_name`
   - `start_date`
   - `end_date`
   - `total_return`
   - `annualized_return`
   - `sharpe`
   - `max_drawdown`
   - `turnover`
   - `cost_drag`
   - `trade_count`

3. `strategy_robustness_results`
   - `evaluation_id`
   - `test_type`
   - `variant_name`
   - `params_json`
   - `metrics_json`
   - `pass_fail`

---

## Phase 2：交易仿真可信化（P0/P1）

目标：让回测更像 A 股，而不是理想化撮合。

### Task 5: 补 A 股交易规则配置

**Files:**

- Modify: `backend/app/services/backtest/models/data_models.py`
- Modify: `backend/app/services/backtest/core/portfolio_manager_array.py`
- Modify: `backend/app/services/backtest/core/portfolio_manager.py`

**BacktestConfig 新增：**

```python
enable_t1_rule: bool = True
enable_limit_up_down_check: bool = True
enable_volume_capacity_check: bool = False
max_volume_participation: float = 0.05
stamp_tax_rate: float = 0.001
commission_min_cost: float = 5.0
limit_up_down_pct: float = 0.10
st_limit_up_down_pct: float = 0.05
```

**验收：**

- 当天买入股票当天不能卖出。
- 涨停日不能以超过可成交规则买入。
- 跌停日不能卖出或至少标记为 rejected。
- 所有拒绝必须进入 `execution_reason`。

### Task 6: 实现成本前后对照

**Files:**

- Modify: `backend/app/services/backtest/core/portfolio_manager_array.py`
- Modify: `backend/app/services/backtest/reporting/backtest_report_builder.py`

**Output:**

```json
{
  "cost_analysis": {
    "gross_return": 0.123,
    "net_return": 0.098,
    "cost_drag": 0.025,
    "commission": 1234.5,
    "stamp_tax": 888.0,
    "slippage": 456.7
  }
}
```

**注意：** 当前 `get_performance_metrics_without_cost()` 返回占位 0，必须替换成真实计算。

### Task 7: 增加 benchmark 对照

**Files:**

- Modify: `backend/app/services/backtest/reporting/backtest_report_builder.py`
- Add helper: `backend/app/services/backtest/analysis/benchmark_analysis.py`

**Metrics:**

- benchmark_return
- excess_return
- tracking_error
- information_ratio
- beta
- alpha
- max_drawdown_vs_benchmark

默认 benchmark：

- 沪深 300：`000300.SH` 或项目当前统一代码格式

---

## Phase 3：样本外与稳健性验证（P1）

目标：不再只看“全样本收益”，而看策略能否跨时间、跨参数、跨成本存活。

### Task 8: Walk-forward evaluation runner

**Files:**

- Create: `backend/app/services/backtest/evaluation/walk_forward.py`
- Create: `backend/scripts/evaluate_strategy_walk_forward.py`
- Test: `backend/tests/backtest/evaluation/test_walk_forward.py`

**Window examples:**

```text
train: 2021-2022 -> test: 2023
train: 2022-2023 -> test: 2024
train: 2023-2024 -> test: 2025
```

**Output:**

- 每个 test window 的收益、回撤、Sharpe、交易次数、成本拖累。
- 汇总稳定性：正收益窗口占比、最大坏窗口、参数漂移。

### Task 9: 参数扰动稳健性测试

**Files:**

- Create: `backend/app/services/backtest/evaluation/parameter_sensitivity.py`
- Create: `backend/scripts/evaluate_strategy_sensitivity.py`

以 MA 为例：

```text
short_window: [3, 5, 8, 10]
long_window: [15, 20, 30, 60]
signal_threshold: [0, 0.0025, 0.005, 0.01]
```

评估原则：

- 不追求单个最优点。
- 看参数邻域是否整体可接受。
- 如果只有一个尖峰参数赚钱，判定为过拟合风险高。

### Task 10: 成本压力测试

**Files:**

- Create: `backend/app/services/backtest/evaluation/cost_stress.py`

测试组：

```text
base: 当前成本
low_cost: 成本减半
high_cost: 成本翻倍
stress: 成本三倍 + 滑点三倍
```

策略可信要求：

- base 不能只靠忽略成本才赚钱。
- high_cost 下不能完全崩溃。
- 报告必须显示 `cost_drag`。

### Task 11: 股票池扰动测试

**Objective:** 判断策略是不是只靠少数个股撑起来。

**Tests:**

1. 排除贡献最大 Top 10 个股后重跑。
2. 随机抽 80% 股票池重复 N 次。
3. 按行业/市值分层抽样。

**Output:**

- return distribution
- max drawdown distribution
- top contributors dependency
- concentration risk

---

## Phase 4：报告解释层（P1）

目标：让用户看到的不只是收益数字，而是“为什么赚/亏、哪里不可信”。

### Task 12: Strategy scorecard API

**Files:**

- Create: `backend/app/api/v1/strategy_evaluations.py`
- Modify: `backend/app/api/v1/api.py`

Endpoints：

```text
POST /api/v1/strategy-evaluations
GET  /api/v1/strategy-evaluations/{evaluation_id}
GET  /api/v1/strategy-evaluations/{evaluation_id}/scorecard
GET  /api/v1/strategy-evaluations/compare
```

### Task 13: 个股贡献与收益归因标准化

已有相关报告：

- `docs/reports/2026-04-14-official-robust-per-stock-contribution-analysis.md`

产品化字段：

```json
{
  "contribution_analysis": {
    "top_positive_contributors": [],
    "top_negative_contributors": [],
    "concentration_ratio_top10": 0.0,
    "profit_without_top10": 0.0
  }
}
```

### Task 14: 信号质量报告

Metrics：

- total_signals
- buy/sell ratio
- execution_rate
- rejection reasons
- avg signal strength
- next_1d/5d/20d forward return after signal
- signal decay curve
- hit rate by signal strength bucket

**重要：** forward return 只用于事后评估，不得进入策略生成逻辑。

---

## Phase 5：前端好用化（P2）

目标：让用户能在页面上判断策略可信度。

### Task 15: 策略配置库页面

Frontend files likely under：

- `frontend/src/app/backtest`
- `frontend/src/components`

功能：

- 策略配置列表
- 标记：smoke / benchmark / research / model
- 一键运行 benchmark
- 一键运行 WFO

### Task 16: 策略评分卡组件

显示：

- trust_score
- rating
- data quality
- execution realism
- robustness
- out-of-sample
- explainability
- usability

### Task 17: 回测报告页增加解释区

新增模块：

- 成本拖累
- 相对 benchmark
- 个股贡献 Top/Bottom
- 参数敏感性热力图
- WFO 窗口表现
- 策略风险提示

---

## 5. 推荐的最小落地顺序

如果只做最值钱的一条线，建议按这个顺序：

1. 修 MA API 参数不一致。
2. 加 `make smoke-backtest`。
3. seed 6 个默认策略配置。
4. 补 cost_analysis，尤其修 without-cost 占位。
5. 做 MA / RSI / MACD / Bollinger 的统一 benchmark runner。
6. 加 walk-forward runner。
7. 产出第一版 strategy scorecard。

这样 1-2 轮后，平台就可以回答：

- 这个策略能不能跑？
- 是不是因为成本没算才赚钱？
- 样本外还行不行？
- 参数稍微变一下会不会崩？
- 是不是只靠几只股票赚钱？
- 和沪深 300 比到底有没有超额？

---

## 6. 第一批验收命令

```bash
# 基础平台仍然可用
make smoke-local

# 新增回测 smoke
make smoke-backtest

# 策略配置种子
cd backend
../backend/.venv-py313/bin/python scripts/seed_strategy_configs.py --dry-run
../backend/.venv-py313/bin/python scripts/seed_strategy_configs.py

# MA benchmark
PYTHONPATH=backend backend/.venv-py313/bin/python backend/scripts/evaluate_strategy_benchmark.py \
  --strategy moving_average \
  --config benchmark/moving_average_5_20_threshold_005 \
  --universe available_full_market \
  --start 2021-05-11 \
  --end 2025-12-31 \
  --benchmark 000300.SH

# Walk-forward
PYTHONPATH=backend backend/.venv-py313/bin/python backend/scripts/evaluate_strategy_walk_forward.py \
  --strategy moving_average \
  --config benchmark/moving_average_5_20_threshold_005
```

---

## 7. Definition of Done

本专项第一阶段完成标准：

- `make smoke-local` 通过。
- `make smoke-backtest` 通过。
- `/api/v1/backtest/strategies` 参数与实现一致。
- `/api/v1/strategy-configs` 至少有 6 个默认配置。
- 至少 4 个技术策略完成统一 benchmark。
- MA 策略有 cost_analysis、benchmark_analysis、walk-forward 结果。
- strategy scorecard 能给出评分和降级原因。
- docs 中有一份“哪些策略当前可信/不可信/为什么”的报告。

---

## 8. 当前建议结论

短期不要急着新增更多策略。当前最优先的是把策略评估体系做可信。

推荐第一刀：

> 先把 `moving_average` 打造成第一个“可信策略样板”：参数一致、配置可复用、成本前后可对照、可样本外验证、可解释贡献来源。然后用同一套框架横向评估 RSI / MACD / Bollinger / MultiFactor。

这样后续每加一个策略，都不是“又多一个能跑的按钮”，而是“多一个能被统一审判的候选策略”。
