# alpha158 / alpha360 official smoke formal-task A/B

日期：2026-04-14

目标：
- 对同一批 official-replication smoke 模型，做同配置、同股票池、同成本口径的正式任务 A/B
- 回答一个具体问题：
  - alpha360 在 validation signal quality 上更强，这个优势有没有稳定传导到正式任务组合结果？

## 1. 对比对象

模型：
- alpha158
  - model_id: `c7cdd57c-2b2d-4386-a8e8-dd1bb2bdc1b9`
  - model_name: `hermes-official-record-summary-smoke-20260414`
- alpha360
  - model_id: `cea581a2-695e-41e2-bf3c-7809501360fe`
  - model_name: `hermes-official-alpha360-smoke-20260414`

共同条件：
- workflow_mode = `official_replication`
- official_market = `csi300`
- LightGBM
- 股票子集：
  - `600036.SH`
  - `601288.SH`
  - `600519.SH`
  - `000001.SZ`
  - `000651.SZ`
- ranking 正式任务参数：
  - `strategy_name = topk_dropout`（执行时规范成 `model_topk_dropout`）
  - `topk = 2`
  - `n_drop = 1`
  - `benchmark = SH000300`
  - `open_cost = 0.0005`
  - `close_cost = 0.0015`
  - `min_cost = 5.0`

## 2. 先看模型报告里的 signal quality

### alpha158
- validation
  - ic = `0.011977700086373314`
  - rank_ic = `0.025823863129000035`
- test
  - ic = `0.020769709811150908`
  - rank_ic = `0.018446725462808563`

### alpha360
- validation
  - ic = `0.06757324250967796`
  - rank_ic = `0.06447484877000852`
- test
  - ic = `0.009336708826747895`
  - rank_ic = `-0.0014520950970494076`

第一层结论：
- alpha360 的 validation IC / RankIC 明显强于 alpha158
- 但 test 段反而没有维持这个优势：
  - alpha158 的 test IC / RankIC 更稳
  - alpha360 的 test RankIC 已经略为负值

这说明：
- alpha360 不能只靠 validation signal-quality 判优
- 必须继续看 formal-task portfolio 结果

## 3. 正式任务 A/B

## 3.1 2020-short（2020-01-01 → 2020-08-01）

任务：
- alpha158: `603e36db-ee45-4587-bc38-65d4038bacc3`
- alpha360: `8d720316-1eb6-4633-a03a-b2aa5caf63a9`

结果：
- alpha158
  - total_return = `-0.023393007985733218`
  - annualized_return = `-0.04012039016849733`
  - sharpe_ratio = `-0.6206208875433078`
  - max_drawdown = `-0.039719283495219734`
  - volatility = `0.0646455686132505`
  - total_trades = `232`
  - raw_signal_count = `236`
  - executed_signal_count = `232`
  - rejection = `4`（可买数量不足）
  - best_stock = `600519.SH` (`13665.73`)
  - worst_stock = `601288.SH` (`-19776.06`)
- alpha360
  - total_return = `-0.02632061368198006`
  - annualized_return = `-0.04509254267334806`
  - sharpe_ratio = `-0.5306608384626071`
  - max_drawdown = `-0.05734119189739118`
  - volatility = `0.08497431769034806`
  - total_trades = `237`
  - raw_signal_count = `239`
  - executed_signal_count = `237`
  - rejection = `2`（可买数量不足）
  - best_stock = `600519.SH` (`34730.30`)
  - worst_stock = `600036.SH` (`-28747.62`)

结论：
- 短窗 2020-short 里：
  - alpha158 的 total_return 更好（亏得更少）
  - alpha158 的 max_drawdown 明显更小
  - alpha158 的 volatility 更低
- alpha360 的 Sharpe 略好，但这是在更高波动与更深回撤下得到的
- 整体看，2020-short 仍更偏向 alpha158

## 3.2 2017-2020-testfull（2017-01-01 → 2020-08-01）

任务：
- alpha158: `1119782d-d65f-42b7-a073-325edb86a42b`
- alpha360: `e056ff14-0346-4e8c-ac68-891dedd1ab3f`

结果：
- alpha158
  - total_return = `-0.15496096201356838`
  - annualized_return = `-0.046001021867782654`
  - sharpe_ratio = `-0.6427148972403383`
  - max_drawdown = `-0.23367360980601487`
  - volatility = `0.07157298215009467`
  - total_trades = `1678`
  - raw_signal_count = `1690`
  - executed_signal_count = `1678`
  - rejection = `12`（可买数量不足）
  - best_stock = `000651.SZ` (`53915.78`)
  - worst_stock = `000001.SZ` (`-70751.35`)
- alpha360
  - total_return = `-0.17795678539998852`
  - annualized_return = `-0.05333441822480356`
  - sharpe_ratio = `-0.3136431792949552`
  - max_drawdown = `-0.33507688756514104`
  - volatility = `0.17004807292380808`
  - total_trades = `1661`
  - raw_signal_count = `1682`
  - executed_signal_count = `1661`
  - rejection = `21`（可买数量不足）
  - best_stock = `600519.SH` (`55899.86`)
  - worst_stock = `000001.SZ` (`-103028.87`)

结论：
- 拉到完整 test 窗后，alpha158 优势更清楚：
  - total_return 更好
  - annualized_return 更好
  - max_drawdown 小很多
  - volatility 小很多
  - rejection 更少
- alpha360 的 Sharpe 数值虽然没那么差，但它是建立在显著更高波动和更深回撤上
- 从可上线稳定性视角，这个窗口更明显支持 alpha158

## 4. per-stock attribution（先看最主要差异源）

### 4.1 2020-short

alpha360 相对 alpha158 的主要个股差分：
- `600036.SH`: `-34571.46`
  - alpha158: `+5823.84`
  - alpha360: `-28747.62`
- `600519.SH`: `+21064.57`
  - alpha158: `+13665.73`
  - alpha360: `+34730.30`
- `601288.SH`: `+6957.82`
  - alpha158: `-19776.06`
  - alpha360: `-12818.24`

解释：
- alpha360 虽然在 `600519.SH` 上赚得更多
- 但它在 `600036.SH` 上的大幅回撤，足以把这部分优势吃掉
- 所以短窗里 alpha360 的问题，不是“没有亮点”，而是收益集中在少数股票，但被单一拖累股放大反噬

### 4.2 2017-2020-testfull

alpha360 相对 alpha158 的主要个股差分：
- `600519.SH`: `+52132.37`
  - alpha158: `+3767.48`
  - alpha360: `+55899.86`
- `000001.SZ`: `-32277.52`
  - alpha158: `-70751.35`
  - alpha360: `-103028.87`
- `601288.SH`: `-25166.51`
  - alpha158: `-52132.96`
  - alpha360: `-77299.47`
- `600036.SH`: `-17576.56`
  - alpha158: `-26036.62`
  - alpha360: `-43613.18`

解释：
- alpha360 仍然有一个非常强的正贡献源：`600519.SH`
- 但它同时在：
  - `000001.SZ`
  - `601288.SH`
  - `600036.SH`
  上承担了更大的亏损
- 所以长窗里 alpha360 的问题更像是：
  - 选中了一只很亮眼的赢家
  - 但对多个拖累股的暴露控制更差

## 5. bridge 结论

当前这组 alpha158 / alpha360 A/B 可以先下一个比较明确的判断：

1. alpha360 的 validation signal-quality 提升，不等于 portfolio 结果会同步更好
- validation 里 alpha360 看起来更强
- 但 test 段已经开始衰减
- 到正式任务窗口里，并没有转成更好的收益 / 回撤表现

2. alpha158 在当前这组 official smoke 对比里更稳
- 两个正式窗口里：
  - alpha158 都取得了更好的 total_return
  - alpha158 都有更小的回撤
- 尤其完整 test 窗里，alpha158 的稳定性优势更明显

3. 差异并不是“信号数量完全不同”导致的
- 两者 raw / executed signal counts 很接近
- 两者都主要被同一种 rejection 影响：
  - `可买数量不足`
- 所以主差异更像是：
  - 排序落点
  - 持仓分配
  - 个股贡献结构

4. 当前最准确的表述是：
- alpha360 在 validation ranking 质量上更亮眼
- 但 alpha158 在 test + formal-task 上更稳、更接近当前主候选形态

## 5. 模型报告层现状

目前这两只模型的 `evaluation-report` 已经都能直接回带：
- `portfolio_bridge_summary.task_count = 2`
- `official_record_summary.port_ana_record.task_count = 2`

也就是说，这轮 A/B 已经不只是实验记录，而是正式沉到了模型报告桥接层里。

## 6. 当前建议的下一步

与其继续只做更多“同类窗口重复 A/B”，更值钱的下一步是：

1. 做 alpha158 vs alpha360 的 ranking attribution
- daily topk overlap
- 哪些日期篮子分歧最大
- 分歧主要落在哪几只股票上

2. 做 per-stock contribution bridge
- 为什么 alpha360 的 `600519.SH` 收益更强，但组合整体反而更差
- 为什么 alpha158 能把 drawdown 控得更小
- `000001.SZ` / `600036.SH` / `601288.SH` 分别在两个模型里贡献了多少拖累

3. 如果只问“当前官方 preset 谁更值得继续当默认参考”
- 这轮证据更支持 alpha158
- alpha360 目前更像研究候选，不像默认升级版
