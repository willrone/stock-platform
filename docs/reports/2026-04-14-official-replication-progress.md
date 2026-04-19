# Official Qlib replication progress note

日期：2026-04-14

本轮推进内容：

## 已完成

1. 后端训练请求已支持官方复刻模式字段
- `workflow_mode`
- `official_dataset`
- `official_market`

2. `/models/train` 已把官方复刻 preset 透传到后台训练任务
- 请求 → create_training_task → executor.submit → train_model_task → `QlibTrainingConfig`

3. `official_workflow.py` 已沉淀出可复用的官方 baseline 结构
- `OfficialDataset`
- `OfficialMarket`
- `OfficialWorkflowConfig`
- `build_official_lightgbm_workflow_config(...)`
- `build_official_dataset_config(...)`
- `create_official_dataset_adapter(...)`

4. `QlibTrainingPipeline` 已增加官方数据路径入口
- official mode 下不再走本地增强 `data_provider.prepare_qlib_dataset`
- 会先调用 `OfficialQlibDataBuilder`
  - 从原始 tushare/parquet OHLCV 读取
  - 只导出纯 OHLCV 到独立的 `OFFICIAL_QLIB_DATA_PATH`
  - 不混入本地增强技术指标
- 然后再基于这个独立 provider_uri 构造官方 DatasetH 配置与 adapter
- `prepare_training_datasets(...)` 已能把官方 adapter 拆成 train/valid 视图
- `log_dataset_overview(...)` / `analyze_feature_correlations(...)` 已支持官方 adapter 元信息

5. 训练/预测核心已做最小兼容
- `_train_qlib_model(...)` 训练时会对 official adapter 解包到底层 dataset
- `_evaluate_model(...)` 预测时会按 adapter 的 `primary_segment` 选择 `train` / `valid`

## 当前仍未完成

1. 还没有把官方 SignalRecord / SigAna / PortAna 评估链彻底接到平台报告里
- 当前已经把 train / validation / test 的 signal-quality 评估接进统一训练结果和 evaluation report
- 但 portfolio 侧仍主要依赖正式任务回测结果单独看，还没有在模型报告里形成统一的官方 PortAna 风格总览

## 本轮新增验证与修正

### 1. 修复 official instruments override 与本地 qlib bin 命名不一致

问题现象：
- `create_official_dataset_adapter(..., stock_codes=["600036.SH", ...])` 会返回空 dataset
- `segment_lengths` 全为 0
- 训练流程在 dataset stage 报 `无法获取训练数据`

根因：
- API / 请求层传入的是 `600036.SH` 这种点分大写代码
- `QlibBinConverter` 写入到 `OFFICIAL_QLIB_DATA_PATH` 时，instrument 名是 `600036_sh`
- 官方 DatasetH 在 handler 里按 instruments 过滤时，点分代码与 bin 树内 instrument 名不匹配

修复：
- `build_official_dataset_config(...)` 现在会把 instruments override 统一规范成 Qlib bin 风格：
  - `600036.SH -> 600036_sh`
  - `000001.SZ -> 000001_sz`

### 2. 修复 official adapter 评估阶段读不到标签

问题现象：
- smoke 虽然能训练，但 `_extract_evaluation_inputs(...)` 读不到 official DatasetH 的 label
- 导致 validation metrics 落成默认值
- `signal_quality.rank_ic / ic / icir` 都是 `null`

根因：
- 官方 DatasetH 的 `prepare(segment, col_set="label")` 返回的是单列表 DataFrame
- 列名是官方 label expression，而不是项目本地适配器里的固定 `label`
- 原先评估提取逻辑只识别本地 `label` 列或本地 adapter 结构

修复：
- 增加 official adapter 专用标签提取路径
- 支持从以下结构里统一抽出标签序列：
  - 单列表 DataFrame
  - MultiIndex columns 中 `("label", expr)` 形式
  - 本地 adapter / pandas DataFrame 原有路径

## 已完成的真实 smoke

### A. 直接引擎 smoke（backend venv）

配置：
- `workflow_mode=official_replication`
- `official_dataset=alpha158`
- `official_market=csi300`
- LightGBM
- 股票子集：`600036.SH / 601288.SH / 600519.SH / 000001.SZ / 000651.SZ`
- `num_iterations=20`

结果：
- DatasetH 成功消费独立 `OFFICIAL_QLIB_DATA_PATH`
- `feature_count = 158`
- `segment_lengths = {train: 8510, valid: 2440, test: 4355}`
- LightGBM 官方 fit 路径成功训练并早停
- `best_epoch = 2`
- validation signal quality 已不再是空值

关键 validation 指标：
- `accuracy = 0.5024`
- `mse = 0.0016`
- `rank_ic = 0.025823863129000035`
- `ic = 0.011977700086373314`
- `sample_count = 2281`

### B. 通过平台 API 的正式训练链验证

第一条 API smoke：
- `model_id = a310c10b-3239-4655-82db-3fcfb03d5225`
- 已验证 official hyperparameters / benchmark / early_stopping / validation signal_quality 能落到 report

本轮继续补做“test 段入报告”的正式 API 验证：
- `POST /api/v1/models/train`
- `workflow_mode=official_replication`
- `official_dataset=alpha158`
- `official_market=csi300`

真实模型：
- `model_id = 1c9176e0-4d60-4873-bbd9-d19a421bdde6`
- `status = ready`
- `training_stage = completed`
- 已生成模型文件与 evaluation report

通过 API 确认：
- `GET /api/v1/models/{model_id}/evaluation-report` 已返回：
  - `training_summary.train_samples = 8510`
  - `training_summary.validation_samples = 2440`
  - `training_summary.test_samples = 4355`
  - `signal_quality.rank_ic = 0.025823863129000035`（validation）
  - `segment_evaluation.train` / `segment_evaluation.validation` / `segment_evaluation.test` 三段都存在
  - `segment_evaluation.test.signal_quality.rank_ic = 0.018446725462808563`
  - `segment_evaluation.test.performance_metrics.accuracy = 0.4957`

结论：
- official replication 已不只是“代码接通”
- 而是已经能通过平台正式训练 API 生成：
  - 真实模型记录
  - 真实 evaluation_report
  - validation signal_quality
  - test segment evaluation

### C. 模型报告层已能回带 formal-task portfolio bridge summary

本轮新增：
- `GET /api/v1/models/{model_id}/evaluation-report` 现在会动态补上 `portfolio_bridge_summary`
- 数据来源是该 `model_id` 关联的正式 backtest tasks + `signal_records`

字段内容：
- `task_count`
- `tasks[]`
  - task_name / window_label / period
  - portfolio_metrics
  - signal_summary
- `best_by_total_return`
- `best_by_sharpe`
- `smallest_drawdown`

已验证：
- official 主候选 `53d9e8ad-e134-4b53-ba50-39a9c91f23df`
  - `portfolio_bridge_summary.task_count = 32`
- robust 主候选 `33b2fd75-af83-4d5f-bc2c-28dbad9fffa2`
  - `portfolio_bridge_summary.task_count = 28`
- 新的 official API smoke 模型 `1c9176e0-4d60-4873-bbd9-d19a421bdde6`
  - `portfolio_bridge_summary.task_count = 0`
  - 说明该字段对无正式任务模型也有稳定空结构

### D. 模型报告层已增加 official-style record summary

本轮继续收口：
- `GET /api/v1/models/{model_id}/evaluation-report` 现在会额外动态补上 `official_record_summary`
- 目标是把当前平台已有信息重新组织成更接近官方 workflow record stack 的形状：
  - `signal_record`
  - `sig_ana_record`
  - `port_ana_record`

字段含义：
- `signal_record`
  - 每段 train / validation / test 的 dataset_samples / evaluated_samples / has_signal_quality
- `sig_ana_record`
  - 每段 train / validation / test 的信号分析指标（IC / RankIC / ICIR / sample_count 等）
- `port_ana_record`
  - 来自正式任务 bridge summary 的组合层摘要（task_count / best return / best sharpe / smallest drawdown / tasks）

已验证：
- 对历史 official / robust 主候选：
  - `official_record_summary.port_ana_record` 已能挂出真实正式任务结果摘要
- 对新的 official smoke 模型 `c7cdd57c-2b2d-4386-a8e8-dd1bb2bdc1b9`：
  - `sig_ana_record.train.analysis_scope = train`
  - `sig_ana_record.validation.analysis_scope = validation`
  - `sig_ana_record.test.analysis_scope = test`
  - `sig_ana_record.test.rank_ic = 0.018446725462808563`

这一步意味着：
- 虽然我们还没有直接原生落 Qlib 的 SignalRecord / SigAnaRecord / PortAnaRecord 对象
- 但模型报告接口已经开始提供一层“官方 record 视图”的稳定输出

### E. alpha360 已补做同等级正式闭环 smoke

本轮继续把“alpha360 还没做同等级闭环验证”的缺口往前推了一步，实际完成了两层验证：

1. 平台正式训练 API smoke
- 请求：
  - `workflow_mode=official_replication`
  - `official_dataset=alpha360`
  - `official_market=csi300`
  - LightGBM
  - 股票子集：`600036.SH / 601288.SH / 600519.SH / 000001.SZ / 000651.SZ`
  - `num_iterations=20`
- 真实模型：`cea581a2-695e-41e2-bf3c-7809501360fe`
- 已确认：
  - `GET /api/v1/models/{model_id}/evaluation-report` 返回完整 `segment_evaluation`
  - `official_record_summary` / `portfolio_bridge_summary` 结构都存在
  - `hyperparameters.official_dataset = alpha360`
  - `hyperparameters.official_market = csi300`
  - `hyperparameters.official_benchmark = SH000300`

关键结果：
- validation:
  - `accuracy = 0.4809`
  - `ic = 0.06757324250967796`
  - `rank_ic = 0.06447484877000852`
  - `icir = 0.11848117716150199`
  - `rank_icir = 0.11626622099478366`
- test:
  - `accuracy = 0.4856`
  - `ic = 0.009336708826747895`
  - `rank_ic = -0.0014520950970494076`
  - `icir = 0.017937594577275544`
  - `rank_icir = -0.002797232760778084`

对比同配置 alpha158 smoke：
- alpha360 在 validation 的 IC / RankIC 明显更强
- 但 test 段已经接近持平甚至略转弱，说明 alpha360 不能只看 validation，需要继续依赖正式任务窗口判断

2. 平台正式任务 `/tasks` 闭环 smoke
- 任务：`8d720316-1eb6-4633-a03a-b2aa5caf63a9`
- 名称：`hermes-alpha360-official-ranking-smoke-20260414`
- 走的是正式任务链路，不是直调 `/backtest`
- 配置：
  - `strategy_name=topk_dropout`（执行时规范成 `model_topk_dropout`）
  - `model_id=cea581a2-695e-41e2-bf3c-7809501360fe`
  - `topk=2`
  - `n_drop=1`
  - `benchmark=SH000300`
  - `open_cost=0.0005`
  - `close_cost=0.0015`
  - `min_cost=5.0`
  - `window=2020-01-01 → 2020-08-01`
- 真实结果：
  - `status = completed`
  - `strategy_name = model_topk_dropout`
  - `total_return = -0.02632061368198006`
  - `annualized_return = -0.04509254267334806`
  - `sharpe_ratio = -0.5306608384626071`
  - `max_drawdown = -0.05734119189739118`
  - `total_trades = 237`
  - `win_rate = 0.4491525423728814`
  - `raw_signal_count = 239`
  - `executed_signal_count = 237`
  - `rejected_signal_count = 2`

同时确认：
- `GET /api/v1/models/cea581a2-695e-41e2-bf3c-7809501360fe/evaluation-report`
  - `portfolio_bridge_summary.task_count = 1`
  - `official_record_summary.port_ana_record.task_count = 1`
- 说明 alpha360 现在已经不是“只能训练、不能正式任务收口”，而是：
  - 可训练
  - 可正式任务回测
  - 可被 bridge / official-style record summary 吸收到模型报告里

本轮新增 contract coverage：
- `backend/tests/unit/api/test_model_contract_api.py`
  - 新增 alpha360 断言，锁住：
    - `official_dataset=alpha360`
    - `official_market=csi500`
    - `official_benchmark=SH000905`
    - `official_segments` 持久化到模型超参数

### F. alpha158 vs alpha360 同窗 formal-task A/B 已完成

本轮继续按“同股票池 / 同 topk / 同成本口径 / 同窗口”的方式，补做了 alpha158 vs alpha360 的正式任务 A/B。

对比对象：
- alpha158: `c7cdd57c-2b2d-4386-a8e8-dd1bb2bdc1b9`
- alpha360: `cea581a2-695e-41e2-bf3c-7809501360fe`

窗口与任务：
- `2020-01-01 → 2020-08-01`
  - alpha158: `603e36db-ee45-4587-bc38-65d4038bacc3`
  - alpha360: `8d720316-1eb6-4633-a03a-b2aa5caf63a9`
- `2017-01-01 → 2020-08-01`
  - alpha158: `1119782d-d65f-42b7-a073-325edb86a42b`
  - alpha360: `e056ff14-0346-4e8c-ac68-891dedd1ab3f`

结果摘要：
- `2020-short`
  - alpha158: `total_return=-0.02339`, `max_drawdown=-0.03972`
  - alpha360: `total_return=-0.02632`, `max_drawdown=-0.05734`
- `2017-2020 testfull`
  - alpha158: `total_return=-0.15496`, `max_drawdown=-0.23367`
  - alpha360: `total_return=-0.17796`, `max_drawdown=-0.33508`

当前可下的判断：
- alpha360 虽然 validation IC / RankIC 更亮眼
- 但正式任务并没有转成更好的 portfolio 结果
- 在当前这组 official smoke 里，alpha158 反而是更稳的默认参考

详细 A/B 记录见：
- `docs/reports/2026-04-14-alpha158-alpha360-formal-ab.md`

### G. alpha158 vs alpha360 的 ranking drift 也已拆开

本轮进一步把“为什么 alpha360 validation 更亮，但 formal-task 仍输给 alpha158”拆到 daily basket 层。

新增报告：
- `docs/reports/2026-04-14-alpha158-alpha360-ranking-drift.md`

核心发现：
- `2020-short`
  - `same_top2_days = 15 / 140` (`10.71%`)
- `2017-2020-testfull`
  - `same_top2_days = 98 / 864` (`11.34%`)
- 说明两者不是“相近模型”，而是长期在做非常不同的 top2 篮子选择

风格差异：
- alpha158 长期更偏 `601288.SH + 600519.SH`
- alpha360 更愿意抬高 `600036.SH` / `000651.SH`
- 这与 formal-task 的个股归因是一致的：
  - alpha360 能抓到 `600519.SH` 的强赢家
  - 但会被 `600036.SH` 等拖累股反噬

因此当前更准确的判断是：
- alpha360 的问题不是“没 alpha”
- 而是 basket allocation 更激进、更分散，当前还没有转化成更稳的组合结果

### H. alpha158 vs alpha360 的 divergence-date event slicing 也已补完

本轮继续往下把几段最典型的分歧月份切开看：
- `2020-02`
- `2020-07`
- `2017-03`
- `2019-07`

新增报告：
- `docs/reports/2026-04-14-alpha158-alpha360-divergence-event-slicing.md`

核心新增结论：
- alpha360 不是“所有分歧月都选错”
- 在 `2020-07` 这类月份里，它把 `600036.SH` 拉进 top2 的局部选择，forward 1d / 5d 切片确实更强
- 但在 `2017-03` 这类月份里，alpha158 的保守篮子局部上明显更优
- 也就是说：
  - alpha360 的风格更 regime-sensitive
  - 它有局部 alpha，但跨 regime 稳定性不够

同时也确认了一个分析边界：
- divergence-day 的 forward slice 足以回答“局部换篮子值不值”
- 但还不足以完整解释月度 realized PnL
- 下一步若要继续深挖，应该进入 holdings-aware replay / execution path attribution

### I. holdings-aware replay 已补到 4 组代表性事件簇里的 4 组中的 4 组

本轮进一步派子代理补完了 4 个代表性事件簇里的剩余 2 组，并把前一轮 2 组一起收口成完整 replay 证据链：
- `2020-02-03 ~ 2020-02-05`
- `2020-07-01 ~ 2020-07-03`
- `2017-03-01 ~ 2017-03-03`
- `2019-07-01 ~ 2019-07-03`

新增报告：
- `docs/reports/2026-04-14-alpha158-alpha360-holdings-replay-2020-02.md`
- `docs/reports/2026-04-14-alpha158-alpha360-holdings-replay-2020-07.md`
- `docs/reports/2026-04-14-alpha158-alpha360-holdings-replay-2017-03.md`
- `docs/reports/2026-04-14-alpha158-alpha360-holdings-replay-2019-07.md`

这轮最重要的新增判断：
1. `2020-02-03 ~ 2020-02-05` 在正式任务里根本没有真实持仓分歧
- 两边都空仓
- 没有 signal_records
- 没有 trade_records
- 所以它只能解释 ranking / forward-slice 层，不能直接解释 realized 月收益

2. `2020-07-01 ~ 2020-07-03` 里 alpha360 的局部优势是真实落地的
- 它在这几天的真实净值追赶更强
- 到 7 月末 gross 口径甚至略微领先 alpha158
- 但全窗累计成本更高，最终把 net 结果吃回去

3. `2017-03-01 ~ 2017-03-03` 与 `2019-07-01 ~ 2019-07-03` 里，alpha158 的 replay 优势都是真实 realized path
- `2017-03`：alpha360 做了更差的 `601288 -> 600036 -> 601288` round-trip
- `2019-07`：alpha360 从和 alpha158 相同的起始篮子出发，但很快轮到错误路径，且这 3 天已经解释了 7 月绝大部分月度差距

因此现在可以把结论收得更硬：
- alpha360 有局部 alpha，尤其在 `2020-07` 这类 regime 下是真会落到真实持仓里的
- 但 alpha158 在更多关键事件簇里，execution path 更稳、错误轮换更少
- 这进一步支持 alpha158 仍应是当前默认 official reference，而 alpha360 更适合作为 research candidate

### J. evaluation-report 已开始产品化承载 bridge 证据

本轮继续把这条研究线的一部分低风险结论正式沉到 report API：
- 新增报告：
  - `docs/reports/2026-04-14-evaluation-report-productization.md`

当前 `GET /api/v1/models/{model_id}/evaluation-report` 已新增：
- `cost_vs_gross_gap_summary`
- `per_stock_ranking_preference`
- `ranking_overlap_summary`
- `event_replay_summary`

其中：
- `cost_vs_gross_gap_summary`
  - 已接真实 formal-task 的 `total_cost` / gross-vs-net gap / gross return / net return
- `per_stock_ranking_preference`
  - 已接真实 formal-task 的 `stock_performance_detail` + signal counts 聚合
- `ranking_overlap_summary`
  - 先给稳定空结构（`available=false`）
- `event_replay_summary`
  - 先给稳定空结构（`available=false`）

同时 `portfolio_bridge_summary.tasks[]` 也已经增强，当前每个任务摘要会额外带：
- `cost_metrics`
- `monthly_return_summary`
- `stock_contribution_summary`

这一步的意义是：
- 以后看模型，不用先翻 tasks 表，report API 已经能直接告诉我们：
  - gross 看起来好不好、net 为什么被成本吃掉
  - 哪只股票整体最赚钱 / 最拖后腿
- overlap / replay 这种更重的分析，先保留为离线分析产物，等后续再决定是否预计算持久化

## 当前最合理的下一步

优先继续把官方 PortAna 收口往“更原生”推进：
- 决定是否把 `official_record_summary` 从动态注入进一步升级为训练完成后的持久化报告字段
- 再评估是否补做 alpha360 的同等级闭环验证
- 如果继续研究解释层，则做关键 divergence dates 的 event slicing
