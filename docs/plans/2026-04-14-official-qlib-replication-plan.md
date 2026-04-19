# Official Qlib Workflow Replication Plan

> For Hermes: use this plan before continuing any stock-platform ranking/model-quality work. The first objective is not to beat current local results, but to reproduce the official Qlib data/training/evaluation shape closely enough that deviations are intentional rather than accidental.

Goal: 在 stock-platform 里增加一条“官方 Qlib 对齐模式”，先让数据、切分、标签、信号评估、组合评估尽量贴近官方 LightGBM Alpha158/Alpha360 workflow，再基于这条基线讨论我们当前做法是否正确。

Architecture: 保留现有本地增强训练链路不动，新增一条显式的 official-replication path。它必须绕开当前本地预计算特征拼装逻辑，直接使用 Qlib 官方 Alpha158/Alpha360 handler/loader、显式 train/valid/test 分段、官方 TopkDropout 回测参数、以及 SignalRecord/SigAna/PortAna 对应的评估口径。最终让“官方对齐模式”和“本地增强模式”可以并存、可比较。

Tech Stack: FastAPI, stock-platform backend, Qlib (installed in backend/.venv), LightGBM, SQLite model_info/evaluation_report, pytest.

---

## 0. 已确认的官方参考口径

直接从 Qlib 官方 workflow 配置和已安装包核对到：

- 官方 LightGBM benchmark 配置文件：
  - examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml
  - examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha360.yaml
- 官方默认基线：
  - market: `csi300`
  - benchmark: `SH000300`
  - train: `2008-01-01 ~ 2014-12-31`
  - valid: `2015-01-01 ~ 2016-12-31`
  - test: `2017-01-01 ~ 2020-08-01`
- 官方组合评估：
  - `TopkDropoutStrategy`
  - `topk=50`
  - `n_drop=5`
  - `account=100000000`
  - `open_cost=0.0005`
  - `close_cost=0.0015`
  - `min_cost=5`
- 官方记录链：
  - `SignalRecord`
  - `SigAnaRecord`
  - `PortAnaRecord`
- 官方数据形态：
  - Alpha158 = 158 特征
  - Alpha360 = 360 特征
- 官方标签：
  - `Ref($close, -2) / Ref($close, -1) - 1`
- 官方处理器：
  - Alpha158 默认 learn processors: `DropnaLabel + CSZScoreNorm(label)`
  - Alpha360 benchmark yaml 显式用了 `DropnaLabel + CSRankNorm(label)`

---

## 1. 当前 stock-platform 与官方的关键偏差

### 偏差 A：数据源不是“纯官方 Alpha158/Alpha360”

当前文件：
- `backend/app/services/qlib/enhanced_qlib_provider.py`
- `backend/app/services/qlib/qlib_data_adapter.py`
- `backend/app/services/data/stock_data_loader.py`

现状：
- 先从本地 parquet / 预计算特征读 OHLCV + 技术指标
- 再拼接 Alpha158 因子
- 当前训练输入实际常是约 210 列，而不是纯 158/360
- `QlibDataAdapter` 还会把本地技术指标列映射成 `RET1/MA5/RSI14/...`

结论：
- 这条链路更像“本地增强版特征工程”
- 不能拿它直接宣称“已复刻官方数据”

### 偏差 B：切分方式不是官方显式 train/valid/test

当前文件：
- `backend/app/services/qlib/unified_qlib_training_engine.py`
- `backend/app/api/v1/models.py`

现状：
- 当前主要是 `validation_split=0.2` 的时间切分
- 没有官方那种显式 train / valid / test 三段作为训练输入契约
- `test_samples` 常为 0，只在正式任务阶段另行验证

结论：
- 当前更像“工程化训练 + 平台正式任务二次验证”
- 不是官方 benchmark workflow 的原始训练结构

### 偏差 C：标签与归一化是可实验化的，但默认不等于官方

当前文件：
- `backend/app/services/qlib/unified_qlib_training_engine.py`
- `backend/app/api/v1/models.py`

现状：
- 默认 `label_definition = future_return`
- 默认 `label_normalization = none`
- 还支持 `future_excess_return_cs`、`cs_rank_norm`

结论：
- 这很适合研究，但不适合当“官方复刻”的默认入口
- 官方复刻模式必须把 label / processors 锁死成官方配置

### 偏差 D：评估口径相似，但还不是“官方 workflow 原样复用”

当前文件：
- `backend/app/services/qlib/unified_qlib_training_engine.py`
- `backend/app/services/models/evaluation_report.py`
- `backend/app/services/backtest/reporting/backtest_report_builder.py`

现状：
- 我们现在会计算 signal_quality 和正式任务组合指标
- 但不是直接按 `SignalRecord -> SigAnaRecord -> PortAnaRecord` 这条对象链跑
- 组合评估目前已经接近官方 with-cost / without-cost excess return 口径，但还混着平台自己的任务系统抽象

结论：
- 下一步应该把“官方对齐版评估”显式落成一条单独路径，而不是继续混在增强链路里

---

## 2. 设计原则

1. 不破坏现有增强链路
2. 先做“官方对齐模式”，再做与本地增强链路的 A/B 对比
3. 优先让数据形态和切分方式正确，再谈收益表现
4. 任何“官方复刻成功”的结论都必须满足：
   - 特征数/标签/处理器与官方一致
   - train/valid/test 分段与官方一致
   - signal metrics 与 portfolio metrics 都能产出
   - 报告里明确写清是 `official_replication` 而不是 `enhanced_local`

---

## 3. 实现任务

### Task 1: 增加 official replication 配置模型

Objective: 给训练请求增加一组显式的“官方复刻模式”配置，而不是再靠散落超参隐式拼装。

Files:
- Modify: `backend/app/api/v1/schemas.py`
- Modify: `backend/app/api/v1/models.py`
- Modify: `backend/app/services/qlib/unified_qlib_training_engine.py`
- Test: `backend/tests/unit/api/test_model_contract_api.py`

Steps:
1. 在 schema 中新增官方复刻配置结构，例如：
   - `workflow_mode: official_replication | enhanced_local`
   - `official_dataset: alpha158 | alpha360`
   - `official_market: csi300 | csi500`
   - `official_benchmark: SH000300 | SH000905`
   - `official_segments: {train, valid, test}`
2. 默认保持老接口兼容：不传时仍走现有增强路径。
3. 若 `workflow_mode=official_replication`，强制要求：
   - 不接受自由散装 `label_definition`
   - 不接受自由散装 `label_normalization`
   - 不接受当前本地自定义 `validation_split` 作为唯一切分依据。
4. 补 contract test，断言 official 模式的配置会被正确解析并写入训练配置。

Verification:
- `./backend/.venv/bin/pytest backend/tests/unit/api/test_model_contract_api.py -q`

---

### Task 2: 新增“纯官方数据集”准备路径

Objective: 让 official_replication 模式直接从 Qlib 官方 Alpha158/Alpha360 handler 产生数据，而不是从本地预计算技术指标拼接出来。

Files:
- Create: `backend/app/services/qlib/official_workflow.py`
- Modify: `backend/app/services/qlib/training_engine/pipeline.py`
- Modify: `backend/app/services/qlib/unified_qlib_training_engine.py`
- Test: `backend/tests/unit/qlib/test_official_workflow_dataset_shape.py`

Steps:
1. 在新模块里封装：
   - Alpha158 official dataset builder
   - Alpha360 official dataset builder
2. 直接使用 Qlib 官方对象：
   - `qlib.contrib.data.handler.Alpha158`
   - `qlib.contrib.data.handler.Alpha360`
   - `qlib.data.dataset.DatasetH`
3. 允许 market 走 `csi300/csi500` 这类官方 instrument，而不是 stock_codes 小列表。
4. 产出时记录：
   - feature_count
   - label expression
   - processors
   - segments
5. 写回归测试：
   - Alpha158 feature count = 158
   - Alpha360 feature count = 360
   - 官方 label 与 processors 被正确挂载

Verification:
- `./backend/.venv/bin/pytest backend/tests/unit/qlib/test_official_workflow_dataset_shape.py -q`

---

### Task 3: 把切分从 validation_split 升级为显式 train/valid/test

Objective: 让 official_replication 模式完全按官方三段切分工作，并把 sample counts/report 结构对齐。

Files:
- Modify: `backend/app/services/qlib/unified_qlib_training_engine.py`
- Modify: `backend/app/services/qlib/training_engine/orchestrator.py`
- Modify: `backend/app/services/qlib/training_engine/result_assembler.py`
- Test: `backend/tests/unit/models/test_unified_training_engine_split.py`

Steps:
1. 为 official 模式新增显式 segment 支持，不再只按 `validation_split` 拆。
2. 在结果中真实填充：
   - `train_samples`
   - `validation_samples`
   - `test_samples`
3. evaluation_report 中保留官方 segment 边界。
4. 回归测试锁住：
   - official mode 不会把 valid 伪装成 train
   - test segment 样本数非 0

Verification:
- `./backend/.venv/bin/pytest backend/tests/unit/models/test_unified_training_engine_split.py -q`

---

### Task 4: 新增官方信号评估链

Objective: 让 official_replication 模式的 signal_quality 尽量按 Qlib SigAnaRecord 语义产生，而不是只靠本地近似函数。

Files:
- Create: `backend/app/services/qlib/official_signal_analysis.py`
- Modify: `backend/app/services/qlib/training_engine/orchestrator.py`
- Modify: `backend/app/services/models/evaluation_report.py`
- Test: `backend/tests/unit/models/test_training_report_contracts.py`

Steps:
1. 把以下字段作为 official signal metrics 的标准输出：
   - IC
   - ICIR
   - Rank IC
   - Rank ICIR
   - Long-Short Ann Return
   - Long-Short Ann Sharpe
   - Long-Avg Ann Return
   - Long-Avg Ann Sharpe
2. official 模式下，在 validation / test 至少保留一组明确 scope。
3. 在 report 里增加 provenance，例如：
   - `signal_quality_source: official_replication`
   - `analysis_scope: valid/test`
4. 保持老报告兼容，不影响 enhanced_local 模式。

Verification:
- `./backend/.venv/bin/pytest backend/tests/unit/models/test_training_report_contracts.py -q`

---

### Task 5: 新增官方组合评估链

Objective: 对 official_replication 模式增加一条显式的官方 TopkDropout 组合评估，不先依赖当前平台任务页面的小池实验参数。

Files:
- Create: `backend/app/services/qlib/official_portfolio_analysis.py`
- Modify: `backend/app/services/backtest/reporting/backtest_report_builder.py`
- Modify: `backend/app/api/v1/models.py`
- Test: `backend/tests/unit/backtest/test_backtest_report_builder.py`

Steps:
1. 固定官方默认参数：
   - `TopkDropoutStrategy`
   - `topk=50`
   - `n_drop=5`
   - `benchmark=SH000300`（或 csi500 模式下 SH000905）
   - `account=100000000`
   - `open_cost=0.0005`
   - `close_cost=0.0015`
   - `min_cost=5`
2. 报告里明确保存：
   - `excess_return_without_cost`
   - `excess_return_with_cost`
   - `information_ratio`
   - `max_drawdown`
3. 与当前平台正式任务结果区分命名，避免把 official replication benchmark 与本地正式任务混为一谈。

Verification:
- `./backend/.venv/bin/pytest backend/tests/unit/backtest/test_backtest_report_builder.py -q`

---

### Task 6: 做一条端到端官方复刻 smoke

Objective: 至少让 LightGBM + Alpha158 或 Alpha360 有一条完整可跑通的 official baseline。

Files:
- Create: `scripts/run_official_qlib_replication.py`
- Create: `docs/reports/2026-04-14-official-replication-smoke.md`

Steps:
1. 先跑最小 smoke：
   - market=`csi300`
   - dataset=`alpha158`
   - model=`lightgbm`
   - 官方默认 segments
2. 记录：
   - feature_count
   - sample_count(train/valid/test)
   - IC / Rank IC
   - Annualized Return / IR / Max Drawdown
3. 再跑 Alpha360 一次，做最小对照。
4. 输出结论：
   - 是否已达到“数据形态对齐”
   - 是否已达到“评估口径对齐”
   - 还有哪些与官方 benchmark 表格仍不可直接比

Verification:
- `cd /home/willrone/Projects/stock-platform && ./backend/.venv/bin/python scripts/run_official_qlib_replication.py`

---

## 4. 本轮最先落地的优先顺序

先做这 3 步，不要同时大改：

1. Task 1：加 official_replication 配置入口
2. Task 2：做纯官方 Alpha158/Alpha360 数据路径
3. Task 3：把显式 train/valid/test 切分接通

原因：
- 只有先把数据和切分做对，后面的 signal / portfolio 结果才有解释力
- 现在最危险的误判，是用“本地增强特征 + 小池切分”去证明“官方做法没问题”

---

## 5. 完成标准

达到以下条件，才算“复刻官方做法”进入可讨论阶段：

1. 能明确跑出 `official_replication(alpha158)` 与 `official_replication(alpha360)`
2. feature_count 与官方一致：158 / 360
3. train/valid/test 分段是显式的，不是 ratio-only
4. label / processors 与官方配置一致
5. 能产出 official-style 的 signal metrics 和 portfolio metrics
6. 输出报告时能清楚区分：
   - official_replication
   - enhanced_local

只有到这一步，才能比较有底气地说：
- “如果结果和官方差很大，是我们实现偏了，还是数据版本/市场环境不同”
- 而不是现在这种“本地增强链路和官方口径混在一起”的状态
