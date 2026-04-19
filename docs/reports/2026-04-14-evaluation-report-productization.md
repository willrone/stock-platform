# evaluation-report productization update

日期：2026-04-14

目标：
- 把这轮 alpha158 / alpha360 研究里反复用到的 bridge 证据，开始沉到模型 `evaluation-report` 接口结构里
- 先做“低风险、可直接从 formal-task 结果动态回带”的部分
- 对暂时还没有稳定线上计算链的部分，先给稳定空结构，避免前端/导出契约以后再震荡

## 1. 本轮已上线到 report API 的新字段

`GET /api/v1/models/{model_id}/evaluation-report`
现在除了原有：
- `portfolio_bridge_summary`
- `official_record_summary`

还会新增：
- `cost_vs_gross_gap_summary`
- `per_stock_ranking_preference`
- `ranking_overlap_summary`
- `event_replay_summary`

其中：

### A. cost_vs_gross_gap_summary
来源：
- formal backtest `result.cost_statistics`
- `portfolio_history[-1].portfolio_value_without_cost`
- `total_return_without_cost`

当前结构：
- `task_count`
- `tasks[]`
  - `task_id`
  - `task_name`
  - `window_label`
  - `total_cost`
  - `gross_minus_net_value_gap`
  - `total_return`
  - `total_return_without_cost`
- `largest_cost_gap`
- `best_gross_return`
- `best_net_return`

价值：
- 直接回答“这个模型 gross 看起来好不好、net 为什么被吃回去”
- 很适合承接 alpha360 在 `2020-07` 这种：
  - gross 有局部优势
  - net 被累计成本拉回去
  的场景

### B. per_stock_ranking_preference
来源：
- formal backtest `stock_performance_detail`
- `signal_records` 聚合出的 `stock_signal_counts`

当前结构：
- `stocks[]`
  - `stock_code`
  - `task_mentions`
  - `positive_task_count`
  - `negative_task_count`
  - `total_pnl`
  - `signal_count`
- `best_overall`
- `worst_overall`

价值：
- 先给出一个“模型在正式任务里长期更偏哪些股票、结果好坏如何”的紧凑视角
- 它不是完整 daily ranking overlap，但已经能回答：
  - 哪只股票整体最帮忙
  - 哪只股票整体最拖后腿
  - 某只股票到底是频繁出现但不赚钱，还是高频且高贡献

### C. ranking_overlap_summary
当前状态：
- 先上线稳定空结构：
  - `available = false`
  - `windows = []`

原因：
- 这部分需要真实 prediction-series 级别的日频排序对比
- 当前还没把那条计算链做成 API 内可稳定复用的低成本路径

### D. event_replay_summary
当前状态：
- 先上线稳定空结构：
  - `available = false`
  - `events = []`

原因：
- 这部分需要 holdings-aware replay / execution-path attribution
- 当前仍属于分析级产物，不适合直接在在线接口里现算

## 2. 本轮同时增强了 portfolio_bridge_summary 的 task entry

每个 `portfolio_bridge_summary.tasks[]` 现在会额外带：
- `cost_metrics`
- `monthly_return_summary`
- `stock_contribution_summary`

这意味着：
- API 不只是“任务列表 + 几个组合指标”
- 而是已经开始带：
  - 成本 / gross-vs-net 差距
  - 月度收益轮廓
  - 个股贡献摘要

## 3. 当前实测

对真实 smoke 模型查 API，已确认：

### alpha158
- model_id: `c7cdd57c-2b2d-4386-a8e8-dd1bb2bdc1b9`
- `cost_vs_gross_gap_summary.task_count = 2`
- `per_stock_ranking_preference.best_overall.stock_code = 000651.SZ`
- `ranking_overlap_summary.available = false`
- `event_replay_summary.available = false`

### alpha360
- model_id: `cea581a2-695e-41e2-bf3c-7809501360fe`
- `cost_vs_gross_gap_summary.task_count = 2`
- `per_stock_ranking_preference.best_overall.stock_code = 600519.SH`
- `ranking_overlap_summary.available = false`
- `event_replay_summary.available = false`

并且 `cost_vs_gross_gap_summary.tasks[]` 已能真实回带：
- `total_cost`
- `gross_minus_net_value_gap`
- `total_return`
- `total_return_without_cost`

## 4. 这次的产品化边界

本轮我故意没有把下面两类直接在线重算塞进 API：
- daily ranking overlap
- holdings-aware replay event slicing

原因不是不会做，而是：
1. 它们计算更重
2. 依赖 prediction-series / replay 级细粒度路径
3. 现在更适合先作为离线分析报告存在
4. 先把低风险、可稳定动态回带的结构上线，更利于保持接口稳定

所以当前的策略是：
- 先把“formal-task 已有结果里可直接提炼的结构化证据”上线
- 再决定后面是否把 overlap / replay 做成：
  - 离线预计算并持久化
  - 或显式重计算接口

## 5. 下一步建议

如果继续产品化，推荐顺序：

1. ranking_overlap_summary
- 做离线预计算/缓存化
- 不要在普通 report 请求里直接现算全量 prediction series

2. event_replay_summary
- 先只支持少量代表性事件簇
- 更适合做“analysis snapshot persisted on demand”，不是每次 API 请求都重跑

3. 前端展示
- 在模型详情页新增两个轻量区块：
  - `成本 vs gross gap`
  - `个股偏好 / 个股贡献`

这样以后看一个模型，至少不用再手工翻任务结果，先在 report API 就能看到：
- 这个模型是不是 gross 好看但 net 被成本吃掉
- 这个模型主要靠哪几只股票在赚钱或拖后腿