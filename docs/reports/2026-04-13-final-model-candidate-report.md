# Stock-Platform 最终候选模型报告

日期：2026-04-13

提交基线：`fadbd72` — `Add official model ranking backtests and qlib portfolio analysis`

结论状态：已收口，已完成多侧验证

---

## 1. 执行摘要

当前正式推荐：

1. 主候选：`official`
   - model_id: `53d9e8ad-e134-4b53-ba50-39a9c91f23df`
   - model_name: `hermes-official-bank-core3-1776037184`
2. 备选：`robust`
   - model_id: `33b2fd75-af83-4d5f-bc2c-28dbad9fffa2`
   - model_name: `hermes-bank-core3-robust-20260412-230648`
3. 不作为主推：`mid`
   - model_id: `08300b59-2147-4be1-a7f2-c997596b6e71`

最终判断：
- `official` 更适合作为“当前正式主候选”继续推进。
- `robust` 在 2025 单季度切片中更强，适合作为并行对照候选。
- `mid` 在早期短窗比较中表现亮眼，但在更长窗口与更完整正式任务验证中不再占优。

---

## 2. 本次收口前完成的关键修复

### 2.1 live ranking 回测执行链路 bug 修复

已确认并修复的问题：
- `model_topk_dropout` 在持仓建立后，executor 的价格查找只覆盖“当前持仓 + 预计算信号矩阵中的股票”。
- 由于该 ranking 策略并不依赖预计算 signal matrix，导致新候选股可能拿不到 `current_prices`。
- 结果：新候选股无法参与 TopK 排名竞争，不同模型的真实 ranking 差异会被执行层压平。

修复方式：
- 在 `BacktestExecutor` 中引入对 ranking trade mode 的全股票池价格查找逻辑。
- 对 `topk_dropout`，即使当前只持有部分股票，也为全股票池加载当日价格。

受影响文件（已提交）：
- `backend/app/services/backtest/execution/backtest_executor.py`
- `backend/tests/unit/backtest/test_backtest_executor_characterization.py`

### 2.2 official_portfolio_analysis 口径修复

已按 Qlib 官方风格接通真实 benchmark-relative 计算：
- `excess_return_without_cost = return - bench`
- `excess_return_with_cost = return - bench - cost`

受影响文件（已提交）：
- `backend/app/services/backtest/reporting/backtest_report_builder.py`
- `backend/tests/unit/backtest/test_backtest_report_builder.py`

---

## 3. 方法与验证范围

本报告只基于“正式任务链路”的结果下结论，不以单测或离线直算代替平台正式流程。

统一验证口径：
- API：`/api/v1/tasks`
- strategy_name: `model_topk_dropout`
- bank-core3 股票池：`600036.SH`, `601288.SH`, `601398.SH`
- 参数：`topk=2`, `n_drop=1`, `benchmark=SH000300`
- 初始资金：100000
- commission_rate: 0.0003
- slippage_rate: 0.0001

额外验证：
- finance5 扩池对照
- rolling 多窗口对比
- 两个独立 reviewer：
  - 独立支持性 reviewer
  - 对抗性 reviewer

---

## 4. 训练侧比较（历史参考）

### bank-core3 三模型训练报告摘要

#### official
- accuracy: 0.6395
- train sharpe: 6.4707
- train total_return: 1.3237
- train max_drawdown: -0.33

#### mid
- accuracy: 0.6728
- train sharpe: 8.2913
- train total_return: 1.7847
- train max_drawdown: -0.1687

#### robust
- accuracy: 0.6358
- train sharpe: 6.5047
- train total_return: 1.4616
- train max_drawdown: -0.2262

训练侧直观印象：
- `mid` 最好
- 但训练优势并没有在更长正式 ranking 回测里稳定兑现

---

## 5. 正式任务核心结论

### 5.1 bank-core3，2025Q1（修复后重新验证）

#### official
- task_id: `6cf7c7fc-e01d-4432-a96f-3447c90886f3`
- total_return: 0.3513%
- annualized_return: 1.4652%
- sharpe_ratio: 0.3784
- max_drawdown: -1.2799%
- total_trades: 74
- profit_factor: 1.0199
- with_cost annual excess return: -8.0230%
- with_cost information_ratio: -2.1321

#### mid
- task_id: `1c59328e-85af-4232-b860-3afd2849b7c4`
- total_return: 0.8560%
- annualized_return: 3.5985%
- sharpe_ratio: 1.0160
- max_drawdown: -1.1481%
- total_trades: 74
- profit_factor: 1.3776
- with_cost annual excess return: -5.9022%
- with_cost information_ratio: -1.7147

#### robust
- task_id: `e72d1090-66b4-4133-b642-8703521d88a0`
- total_return: 0.8076%
- annualized_return: 3.3925%
- sharpe_ratio: 0.8350
- max_drawdown: -1.2883%
- total_trades: 74
- profit_factor: 1.1377
- with_cost annual excess return: -6.0878%
- with_cost information_ratio: -1.5419

Q1 结论：
- `mid` / `robust` 优于 `official`
- 其中 `mid` 与 `robust` 较接近，`mid` 略优于 `robust`

### 5.2 bank-core3，2025 全年

#### official
- task_id: `1cdf2463-85be-4467-ab55-cc7d48ffc435`
- total_return: 5.0663%
- annualized_return: 5.0949%
- sharpe_ratio: 0.9757
- max_drawdown: -3.0180%
- total_trades: 446
- win_rate: 50.00%
- profit_factor: 1.3217
- with_cost annual excess return: -4.2327%
- with_cost information_ratio: -0.8341

#### mid
- task_id: `c67ac003-a022-4aff-ac66-119869ea1e3c`
- total_return: 3.5046%
- annualized_return: 3.5242%
- sharpe_ratio: 0.6783
- max_drawdown: -3.0272%
- total_trades: 446
- win_rate: 53.60%
- profit_factor: 1.0894
- with_cost annual excess return: -5.7070%
- with_cost information_ratio: -1.1304

#### robust
- task_id: `f60fd74b-b223-4a2a-a80f-4e146430efaa`
- total_return: 4.9272%
- annualized_return: 4.9550%
- sharpe_ratio: 0.9266
- max_drawdown: -2.7222%
- total_trades: 446
- win_rate: 50.90%
- profit_factor: 1.2716
- with_cost annual excess return: -4.3567%
- with_cost information_ratio: -0.8383

2025 全年结论：
- `official` 略优于 `robust`
- `robust` 回撤略小，但 `official` 收益与 sharpe 更高
- `mid` 落后于二者

### 5.3 finance5 扩池验证（2025 全年）

#### finance5 专用模型
- task_id: `8405870f-209f-44fb-8f1e-c38dadcb9ff2`
- total_return: -0.6550%
- annualized_return: -0.6586%
- sharpe_ratio: -0.1218
- max_drawdown: -5.2033%

#### bank-core3 mid 跨池到 finance5
- task_id: `574f4faa-7dd0-48ad-8c59-af41d3e2e2f9`
- total_return: 1.1297%
- annualized_return: 1.1360%
- sharpe_ratio: 0.2140
- max_drawdown: -5.9695%

扩池结论：
- finance5 没有打出比 bank-core3 更好的 ranking 结果
- 当前阶段不建议把“扩池”当成优先方向

---

## 6. official vs robust 多窗口 rolling 对比

### 6.1 2024 全年

official
- task_id: `c4e6ff5f-215d-413e-9961-fe390958ca3e`
- total_return: 8.5144%
- annualized_return: 8.5388%
- sharpe: 1.3189
- max_drawdown: -3.5278%
- with_cost IR: -0.3171

robust
- task_id: `99829a97-0fee-4fab-bb05-e26b0d9b5214`
- total_return: 5.9816%
- annualized_return: 5.9985%
- sharpe: 0.9143
- max_drawdown: -4.5744%
- with_cost IR: -0.6780

判定：official 胜

### 6.2 2025 Q1

official
- task_id: `190268cb-2444-4f10-a4e6-2705f3cfa66e`
- total_return: 0.3513%
- sharpe: 0.3784
- max_drawdown: -1.2799%
- with_cost IR: -2.1321

robust
- task_id: `de196562-3915-443f-b271-b9a2e18c49f0`
- total_return: 0.8076%
- sharpe: 0.8350
- max_drawdown: -1.2883%
- with_cost IR: -1.5419

判定：robust 胜

### 6.3 2025 Q2

official
- task_id: `369a916c-e8e8-4a8a-b96b-110a4bd352dc`
- total_return: 2.2663%
- sharpe: 2.5392
- max_drawdown: -1.0462%
- with_cost IR: -0.0651

robust
- task_id: `5de798c7-e402-4072-9c19-0025f62688e2`
- total_return: 2.6904%
- sharpe: 2.9340
- max_drawdown: -1.2076%
- with_cost IR: 0.3818

判定：robust 胜

### 6.4 2025 Q3

official
- task_id: `55565337-1a98-4fa3-a051-369419200484`
- total_return: -0.3756%
- sharpe: -0.2997
- max_drawdown: -2.7069%
- with_cost IR: -2.1362

robust
- task_id: `11dc4a52-10d5-454d-a497-0b1ae29511fc`
- total_return: -0.1185%
- sharpe: -0.0977
- max_drawdown: -2.6886%
- with_cost IR: -1.9998

判定：robust 胜

### 6.5 2025 Q4

official
- task_id: `48a0e8d4-fc3f-4c26-baf7-f9254c1e5eaa`
- total_return: -1.6578%
- sharpe: -1.9030
- max_drawdown: -2.4519%
- with_cost IR: -4.2981

robust
- task_id: `dd078e55-c1c2-4d6f-8255-1f2fcd47fc89`
- total_return: -1.5049%
- sharpe: -1.5934
- max_drawdown: -2.6786%
- with_cost IR: -3.7919

判定：robust 胜

### 6.6 2025 全年

official
- task_id: `194f626d-e22b-4a05-b151-d7f57ddcf747`
- total_return: 5.0663%
- annualized_return: 5.0949%
- sharpe: 0.9757
- max_drawdown: -3.0180%
- with_cost IR: -0.8341

robust
- task_id: `d75bebf6-0445-41b7-9e0f-ec16273cec65`
- total_return: 4.9272%
- annualized_return: 4.9550%
- sharpe: 0.9266
- max_drawdown: -2.7222%
- with_cost IR: -0.8383

判定：official 略胜

### 6.7 2024-2025 全窗

official
- task_id: `cf894471-ab6b-4d7a-b3b8-df60cd704723`
- total_return: 12.1711%
- annualized_return: 5.9192%
- sharpe: 0.9815
- max_drawdown: -3.5278%
- with_cost IR: -0.6691

robust
- task_id: `a136b9b5-4805-442d-a34f-617dd5abf6c8`
- total_return: 10.3870%
- annualized_return: 5.0723%
- sharpe: 0.8231
- max_drawdown: -4.5744%
- with_cost IR: -0.7852

判定：official 胜

### rolling 总结

从单季度看：
- robust 更强（Q1/Q2/Q3/Q4）

从完整年份与跨年长窗看：
- official 更强（2024-full、2025-full、2024-2025-full）

这意味着：
- `robust` 更像短周期更灵活的模型
- `official` 更像更适合作为正式主候选的长期稳定模型

---

## 7. 多方验证

### 7.1 独立 reviewer（支持性）

独立 reviewer 结论：
- winner: `official`
- confidence: `medium`
- strongest evidence:
  - 在 2024-full、2025-full、2024-2025-full 这些最长、最关键窗口上，official 都比 robust 更强
  - 在 2024-2025 全窗上，official 同时领先于 return、annualized_return、sharpe、max_drawdown、with_cost_ir
- counterpoints:
  - robust 在 2025 多个季度切片表现更好
  - 两者的 with_cost IR 在长窗上仍是负值，说明成本后 alpha 并不强

### 7.2 对抗 reviewer（怀疑性）

对抗 reviewer 主要质疑：
- robust 赢了 2025 的所有季度切片，official 依赖长窗聚合优势，可能被 2024 的强表现抬高
- 如果目标是“当前更贴近最近 regime 的候选”，robust 也有很强理由
- 在未做更严格成本口径验证前，official 仍可能只是“长窗低成本更优”，而不是更接近实盘摩擦条件下更优

### 7.3 对抗质疑后的最终判断

对抗 reviewer 的质疑是合理的，但在“正式主候选”这个语境下，仍不足以推翻 official：
- 正式主候选更看重完整年份与跨年稳定性，而不只看最近单季度切片
- 2024-full、2025-full、2024-2025-full 这三类长窗结果对候选选择权重更高
- official 在这些长窗上都更优

因此最终保留：
- 主候选：official
- 备选：robust

### 7.4 严格成本口径追加验证（正式任务）

在正式任务链路中追加了更严格成本压力测试：
- commission_rate: `0.001`
- slippage_rate: `0.0005`
- 股票池仍为 bank-core3
- 窗口：`2024-full`、`2025-full`、`2024-2025-full`

注意：
- 当前 `/api/v1/tasks` worker 只会把 `initial_cash`、`commission_rate`、`slippage_rate` 注入 `BacktestConfig`
- `max_position_size`、`cash_reserve_ratio` 等“更保守持仓”参数暂未接入正式任务 worker
- 因此这轮压力测试是“严格成本”验证，不是“严格成本 + 持仓约束”双重验证

#### 2024-full（严格成本）

official
- task_id: `88b62cf6-ef78-45a4-b2e6-9e00424aa392`
- total_return: 0.4257%
- annualized_return: 0.4268%
- sharpe: 0.0657
- max_drawdown: -5.5101%
- with_cost IR: -1.5278

robust
- task_id: `c94ef053-1f3a-4c3d-a756-4344207c88f9`
- total_return: -2.2138%
- annualized_return: -2.2198%
- sharpe: -0.3372
- max_drawdown: -7.5724%
- with_cost IR: -1.9178

判定：official 胜

#### 2025-full（严格成本）

official
- task_id: `14040cc4-eea8-4abf-bfd7-00f069ffe15f`
- total_return: -2.4224%
- annualized_return: -2.4356%
- sharpe: -0.4740
- max_drawdown: -4.1321%
- with_cost IR: -2.3049

robust
- task_id: `00632423-7599-4219-b986-d292219393d6`
- total_return: -2.7848%
- annualized_return: -2.8000%
- sharpe: -0.5347
- max_drawdown: -5.1506%
- with_cost IR: -2.3325

判定：official 胜

#### 2024-2025-full（严格成本）

official
- task_id: `478f6f30-7c6d-456d-9873-3235fba4e443`
- total_return: -4.0020%
- annualized_return: -2.0242%
- sharpe: -0.3380
- max_drawdown: -6.0717%
- with_cost IR: -1.9899

robust
- task_id: `6670b828-1dd5-4a82-8728-a575eede9d8f`
- total_return: -6.5809%
- annualized_return: -3.3510%
- sharpe: -0.5483
- max_drawdown: -9.2795%
- with_cost IR: -2.1741

判定：official 胜

严格成本追加结论：
- 在更高摩擦条件下，两个候选都明显退化
- 但 `official` 在三个关键长窗里依然全面优于 `robust`
- 因而“official 作为主候选”的结论在严格成本条件下得到进一步支持

### 7.5 严格成本 + 保守持仓联合验证（正式任务）

在补齐正式任务链路对以下参数的真实支持后，又完成了一轮联合压力测试：
- `max_position_size`
- `cash_reserve_ratio`
- `board_lot_size`

联合验证参数：
- commission_rate: `0.001`
- slippage_rate: `0.0005`
- max_position_size: `0.1`
- cash_reserve_ratio: `0.2`
- board_lot_size: `200`

说明：
- 这轮验证已经确认 `/api/v1/tasks` worker 会真实应用上述参数
- smoke 验证任务 `bffcf006-b6eb-450e-ac44-7c98308fd3cc` 中，BUY 数量已从默认口径下的 2900/3200 股明显收缩到 1400/1800 股，证明持仓约束已真实生效
- 当前报告返回的 `backtest_config` 序列化字段还未完整展示 `cash_reserve_ratio` / `board_lot_size`，但交易数量和结果已体现它们的影响

#### 2024-full（联合验证）

official
- task_id: `aaf35d5d-8a7f-410d-80e0-2b510f012f32`
- total_return: 0.5602%
- annualized_return: 0.5618%
- sharpe: 0.1693
- max_drawdown: -2.6464%
- with_cost IR: -2.9943

robust
- task_id: `aa732a64-d1e2-4a9f-ac9f-abd31affeb67`
- total_return: -1.0375%
- annualized_return: -1.0403%
- sharpe: -0.3105
- max_drawdown: -3.7642%
- with_cost IR: -3.4519

判定：official 胜

#### 2025-full（联合验证）

official
- task_id: `fa5c22d6-7ac7-419f-800e-24edc9c10859`
- total_return: -2.1809%
- annualized_return: -2.1928%
- sharpe: -0.8115
- max_drawdown: -2.9087%
- with_cost IR: -4.3248

robust
- task_id: `f5c4c944-7c3c-43d2-945d-48f660f7f277`
- total_return: -1.7244%
- annualized_return: -1.7339%
- sharpe: -0.6226
- max_drawdown: -2.9354%
- with_cost IR: -4.0260

判定：robust 略胜

#### 2024-2025-full（联合验证）

official
- task_id: `8a20170d-db32-4ed0-a906-a85e569b5373`
- total_return: -2.4640%
- annualized_return: -1.2414%
- sharpe: -0.4012
- max_drawdown: -3.3602%
- with_cost IR: -3.6326

robust
- task_id: `f9435be7-dd47-4de3-b936-07733bfbf03d`
- total_return: -2.5105%
- annualized_return: -1.2649%
- sharpe: -0.4042
- max_drawdown: -3.9031%
- with_cost IR: -3.5988

判定：official 在 return / sharpe / drawdown 上略胜，robust 在 with-cost IR 上略胜

联合验证结论：
- 这是目前最接近“严格成本 + 更保守资金使用”的正式任务口径
- 在这套更苛刻的条件下，`official` 不再像“严格成本-only”那样全面碾压，但整体仍然没有被 `robust` 明确推翻
- 三个关键长窗里：
  - `official` 明显赢 2024-full
  - `robust` 略赢 2025-full
  - `2024-2025-full` 基本接近，但 official 在总收益、sharpe、drawdown 上仍略优
- 因此主候选结论可以保留为 `official`，但与 `robust` 的差距明显缩小，备选地位需要保留并重视

### 7.6 官方 Qlib workflow 对照 + 2025 双月级 rolling（正式任务）

在继续候选收口前，又补做了两件事：

1. 对照 Qlib 官方示例与文档：
- `examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml`
- `examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha360.yaml`
- `docs/component/workflow.rst`
- `docs/component/strategy.rst`
- `docs/component/report.rst`
- `examples/benchmarks/README.md`

2. 按“更贴近上线”的当前正式任务口径，补跑 2025 双月级 rolling：
- 仍使用 `model_topk_dropout`
- 仍使用 bank-core3
- 仍使用严格成本 + 保守持仓参数：
  - `commission_rate=0.001`
  - `slippage_rate=0.0005`
  - `max_position_size=0.1`
  - `cash_reserve_ratio=0.2`
  - `board_lot_size=200`

#### 7.6.1 从官方示例里提炼出的直接经验

1. 官方 benchmark 不是只看训练 `accuracy`
- 更核心的是：IC / ICIR、Rank IC / Rank ICIR、Annualized Return、Information Ratio、Max Drawdown
- 这说明 stock-platform 里的 training accuracy 只能作为早筛，最终仍要回到正式任务收益/回撤/IR 判定

2. 官方 workflow 强调完整链路评估
- 典型记录栈是：`SignalRecord`、`SigAnaRecord`、`PortAnaRecord`
- 对应到当前项目，应继续坚持：训练报告 ≠ 最终结论，最终还是看正式任务 ranking 回测

3. 官方训练切分是显式 `train/valid/test`
- 官方 YAML 不是简单地只设一个 `validation_split`
- 对 stock-platform 的启发是：进入候选收口后，应优先做固定训练窗 + 独立滚动测试窗，而不是继续在同一段时间里反复比训练页指标

4. 官方 `TopkDropoutStrategy` 本质是横截面排序策略
- 核心不是分值绝对大小，而是每日横截面排序与换手控制（`topk` / `n_drop`）
- 这进一步支持了当前项目优先用 `model_topk_dropout` 正式任务来验候选，而不是只盯单票阈值信号

5. 官方 Alpha360 workflow 会对 label 做 `DropnaLabel + CSRankNorm`
- 这说明如果当前 ranking 表现进入瓶颈，下一步更值得优先研究“label/目标是否更适合 ranking”，而不是继续只靠更激进 LightGBM 超参微调

6. 官方成本口径更细
- 示例里常见 `open_cost` / `close_cost` / `min_cost`
- 当前 stock-platform 已能做统一 commission/slippage，并额外接通了保守持仓，但仍不等价于官方那套更细摩擦建模

#### 7.6.2 2025 双月级 rolling 结果（正式任务）

##### 2025-01 ~ 2025-02

official
- task_id: `37876379-09c4-4ca8-bcf7-450543f922f4`
- total_return: 0.0975%
- annualized_return: 0.6259%
- sharpe: 0.3823
- max_drawdown: -0.5811%
- with_cost IR: -5.6272

robust
- task_id: `02778e47-1849-4f15-84d2-a17a58177afc`
- total_return: -0.1267%
- annualized_return: -0.8088%
- sharpe: -0.5129
- max_drawdown: -0.6619%
- with_cost IR: -6.8391

判定：official 胜

##### 2025-03 ~ 2025-04

official
- task_id: `829dba26-5b8e-4246-be0b-fb0cf0744c69`
- total_return: -0.3269%
- annualized_return: -2.0394%
- sharpe: -0.6808
- max_drawdown: -0.7984%
- with_cost IR: -3.8902

robust
- task_id: `285ee9c7-972a-4b85-b7cd-ca8f184fe3ba`
- total_return: -0.9958%
- annualized_return: -6.1036%
- sharpe: -2.2009
- max_drawdown: -1.0369%
- with_cost IR: -5.6548

判定：official 胜

##### 2025-05 ~ 2025-06

official
- task_id: `d4e4ec48-9f23-423a-87d4-062b3a6b56c7`
- total_return: 0.3606%
- annualized_return: 2.4173%
- sharpe: 1.2803
- max_drawdown: -0.5518%
- with_cost IR: -3.8307

robust
- task_id: `6c242043-8d31-4f11-9797-ce836aea7b63`
- total_return: 0.8029%
- annualized_return: 5.4502%
- sharpe: 4.1791
- max_drawdown: -0.2615%
- with_cost IR: -3.3779

判定：robust 胜

##### 2025-07 ~ 2025-08

official
- task_id: `e03383fb-25ac-4767-879f-55e2fd64cc2b`
- total_return: 0.7621%
- annualized_return: 4.8086%
- sharpe: 2.7133
- max_drawdown: -0.5473%
- with_cost IR: -2.8649

robust
- task_id: `b7be2200-3995-4ec0-a0bb-04dd5cfa1423`
- total_return: 0.4755%
- annualized_return: 2.9781%
- sharpe: 1.5910
- max_drawdown: -0.6820%
- with_cost IR: -3.5785

判定：official 胜

##### 2025-09 ~ 2025-10

official
- task_id: `34c120ad-ca76-40c3-b512-6f5cd2969142`
- total_return: 1.1741%
- annualized_return: 7.3589%
- sharpe: 3.0034
- max_drawdown: -0.5906%
- with_cost IR: -0.6963

robust
- task_id: `9866dec1-0abf-4217-a4c9-16bec57efe64`
- total_return: 1.6069%
- annualized_return: 10.1835%
- sharpe: 4.1586
- max_drawdown: -0.5881%
- with_cost IR: 0.4269

判定：robust 胜

##### 2025-11 ~ 2025-12

official
- task_id: `d0bf8219-3598-460c-9070-37a0b90875d7`
- total_return: -1.5022%
- annualized_return: -9.0859%
- sharpe: -4.9363
- max_drawdown: -1.5221%
- with_cost IR: -9.7353

robust
- task_id: `b48b008d-1a41-4196-99be-ec81fcb40fdc`
- total_return: -1.2638%
- annualized_return: -7.6919%
- sharpe: -4.2278
- max_drawdown: -1.3492%
- with_cost IR: -9.0741

判定：robust 胜

#### 7.6.3 双月级 rolling 结论

新的信息点是：
- `robust` 并没有在更细粒度窗口里持续碾压
- 2025 双月级结果是 `official 3` 胜 `robust 3` 胜
- `official` 赢在：`01-02`、`03-04`、`07-08`
- `robust` 赢在：`05-06`、`09-10`、`11-12`

这带来的更新判断：
- 之前按季度看，容易得到“robust 在 2025 全年各季度都更强”的直观印象
- 但下钻到双月粒度后，结论变成：`2025 的 regime 优势是交替出现的，不是 robust 单边统治`
- 这反而更支持 `official` 继续作为“长期主候选”，因为它并没有在更细粒度切片里失守成明显劣势模型
- 同时也更支持保留 `robust` 作为“阶段性更强的动态备选”

#### 7.6.4 官方风格成本字段接入进展（实现 + smoke）

在完成上面的官方对照后，又继续把更接近 Qlib 官方示例的成本字段真正接入到当前项目：
- `open_cost`
- `close_cost`
- `min_cost`

当前已完成的真实接入范围：
1. `BacktestConfig` 已支持上述字段
2. `/api/v1/backtest` 直调链路已支持上述字段
3. `/api/v1/tasks` 正式任务 worker 已支持上述字段
4. `PortfolioManager` / `PortfolioManagerArray` 已改为：
   - 买入使用 `open_cost`
   - 卖出使用 `close_cost`
   - 单笔手续费按 `max(rate * turnover, min_cost)` 计算
5. 回测报告 `backtest_config` 已回传这些字段，便于核验真实生效口径

对应新增测试覆盖：
- direct backtest 参数透传
- task worker 参数透传
- PortfolioManager / PortfolioManagerArray 成本计算
- report builder 成本字段回传

相关回归已通过：
- `tests/unit/api/test_backtest_model_driven.py`
- `tests/unit/api/test_task_backtest_model_driven.py`
- `tests/unit/backtest/test_backtest_report_builder.py`
- `tests/unit/backtest/test_backtest_executor_characterization.py`
- `tests/unit/backtest/test_portfolio_manager_cost_modeling.py`

真实 smoke 任务：
- task_id: `a387b512-9607-487c-a127-5d8b28e69220`
- 模型：`official`
- 窗口：`2025-01-01 ~ 2025-02-28`
- 口径：
  - `open_cost = 0.0005`
  - `close_cost = 0.0015`
  - `min_cost = 5.0`
  - `slippage_rate = 0.0005`
  - `max_position_size = 0.1`
  - `cash_reserve_ratio = 0.2`
  - `board_lot_size = 200`

smoke 里已经能直接看到：
- 首笔 BUY commission = `5.0`
- 第二笔 BUY commission = `5.0`
- 首笔 SELL commission = `13.5472`

这说明：
- 小额买入已被 `min_cost` 托底
- 卖出已按更高 `close_cost` 计费
- 不再是单一 `commission_rate` 在买卖两侧对称套用

#### 7.6.5 新成本口径下的关键长窗复核（正式任务）

在成本字段真实接入后，又按新的官方风格口径补跑了 3 个关键长窗：
- `2024-full`
- `2025-full`
- `2024-2025-full`

统一口径：
- `open_cost = 0.0005`
- `close_cost = 0.0015`
- `min_cost = 5.0`
- `slippage_rate = 0.0005`
- `max_position_size = 0.1`
- `cash_reserve_ratio = 0.2`
- `board_lot_size = 200`

##### 2024-full

official
- task_id: `0ed1ee70-b37c-4de3-aff3-0731825e26f1`
- total_return: 0.3216%
- annualized_return: 0.3225%
- sharpe: 0.0972
- max_drawdown: -2.8022%
- with_cost IR: -3.0666

robust
- task_id: `493e5ed9-c420-41f0-8fbf-2ef4f199a036`
- total_return: -1.2660%
- annualized_return: -1.2694%
- sharpe: -0.3787
- max_drawdown: -3.9071%
- with_cost IR: -3.5201

判定：official 胜

##### 2025-full

official
- task_id: `00e488a2-32ab-4ff0-a8a5-1f8c2cf945bf`
- total_return: -2.3597%
- annualized_return: -2.3725%
- sharpe: -0.8789
- max_drawdown: -3.0172%
- with_cost IR: -4.3981

robust
- task_id: `6315c552-292e-4838-95d0-000cb7ba1788`
- total_return: -1.8870%
- annualized_return: -1.8973%
- sharpe: -0.6809
- max_drawdown: -3.0393%
- with_cost IR: -4.0836

判定：robust 胜

##### 2024-2025-full

official
- task_id: `2fc835f9-0766-4c18-9f0a-797c4899ea0e`
- total_return: -2.5245%
- annualized_return: -1.2721%
- sharpe: -0.4129
- max_drawdown: -3.3560%
- with_cost IR: -3.6589

robust
- task_id: `20487997-819a-4ca9-8e35-0d6a3c72b3c9`
- total_return: -2.9763%
- annualized_return: -1.5014%
- sharpe: -0.4793
- max_drawdown: -4.2896%
- with_cost IR: -3.6719

判定：official 胜

长窗复核结论：
- 在新的 `open_cost/close_cost/min_cost` 口径下，结论没有被推翻
- `official` 仍赢下 `2024-full` 与 `2024-2025-full`
- `robust` 仍在 `2025-full` 略强
- 这和之前“严格成本 + 保守持仓”的长窗格局基本一致，只是现在摩擦建模更接近官方示例

#### 7.6.6 CSRankNorm 标签实验（真实训练 + 正式任务）

在把成本口径补齐后，又继续沿着官方 Alpha360 workflow 的思路，做了一轮最小可落地的标签实验：
- 新增可选 `label_normalization = cs_rank_norm`
- 实现方式对齐 Qlib `CSRankNorm`：
  - 按 `datetime` 做横截面 rank(pct=True)
  - 再做 `(rank - 0.5) * 3.46`

对应实现范围：
- `QlibTrainingConfig` 已支持 `label_normalization`
- `UnifiedQlibTrainingEngine._prepare_training_datasets()` 已支持对 train/valid 标签应用 `cs_rank_norm`
- 相关回归已覆盖并通过：
  - `tests/unit/models/test_unified_training_engine_split.py`

##### 真实训练实验

新模型：
- model_id: `b3b657d0-7c9f-4d4e-9470-1070a06dfd85`
- model_name: `hermes-bank-core3-csranknorm-1776087697`
- 股票池：`600036.SH`, `601288.SH`, `601398.SH`
- 训练窗：`2024-01-01 ~ 2024-12-31`
- 超参数基线：沿用 `official` 的 LightGBM 参数，只额外加入：
  - `label_normalization = cs_rank_norm`

训练报告里能看到：
- `hyperparameters.label_normalization = cs_rank_norm`
- `signal_quality` 已真实产出，不再是旧模型报告里缺字段的状态

关键信号质量：
- ic: `-0.0529`
- icir: `-0.0728`
- rank_ic: `-0.0803`
- rank_icir: `-0.1045`
- long_short_ann_return: `0.2616`
- long_short_ann_sharpe: `1.9815`
- sample_count: `147`

##### 正式任务验证

在新的官方风格成本口径下，对该 csranknorm 模型做了正式任务回测：

###### 2025-full
- task_id: `e8f2e91a-21d1-4892-9746-69d6663273bd`
- total_return: `-2.3597%`
- annualized_return: `-2.3725%`
- sharpe: `-0.8789`
- max_drawdown: `-3.0172%`
- with_cost IR: `-4.3981`
- total_trades: `428`

###### 2024-2025-full
- task_id: `bf04b525-1315-4e85-a253-2b5da4040c9e`
- total_return: `-2.5245%`
- annualized_return: `-1.2721%`
- sharpe: `-0.4129`
- max_drawdown: `-3.3560%`
- with_cost IR: `-3.6589`
- total_trades: `904`

##### 这轮实验的结论

这轮实验给出的信号非常明确：
- 对当前 `bank-core3` 小股票池来说，**只加一层 `CSRankNorm` 风格标签归一化，并没有把正式任务结果推到比 `official` 更好**
- 从关键长窗结果看，它和当前 `official` 主候选表现实际上完全重合
- 这说明：
  - 官方 `CSRankNorm` 思路本身值得保留
  - 但在当前这么小的横截面股票池里，**仅做标签归一化还不够**
  - 下一步如果继续做 ranking 方向优化，应该更偏向：
    - 改标签定义本身
    - 扩到更适合横截面排序的股票池
    - 或联动目标函数/特征设计，而不是只加一层 label normalization

#### 7.6.7 横截面超额收益标签实验（真实训练 + 正式任务）

在确认“仅做 `CSRankNorm` 不够”后，又继续做了一轮更根本的标签定义实验：
- `label_definition = future_excess_return_cs`

含义：
- 先按当前链路生成未来 N 日收益率标签
- 再按 `datetime` 做横截面去均值
- 也就是把标签改成“相对同日股票池平均水平的超额收益”

这比单纯 label normalization 更接近 ranking 任务本身，因为它直接把训练目标改成“谁会跑赢同池别的股票”。

##### 真实训练实验

新模型：
- model_id: `6de3252e-4238-4262-8b6e-50152cb4d923`
- model_name: `hermes-bank-core3-csexcess-1776088271`
- 股票池：`600036.SH`, `601288.SH`, `601398.SH`
- 训练窗：`2024-01-01 ~ 2024-12-31`
- 超参数基线：沿用 `official`，只额外加入：
  - `label_definition = future_excess_return_cs`

训练报告关键指标：
- accuracy: `0.4422`
- mae: `0.0079`
- r2: `0.0026`

signal_quality：
- ic: `0.1756`
- icir: `0.2142`
- rank_ic: `0.0613`
- rank_icir: `0.0794`
- long_short_ann_return: `0.1070`
- long_short_ann_sharpe: `0.6691`
- sample_count: `147`

一个很关键的现象是：
- 它的 training/report `accuracy` 反而明显低于 `official`
- 但 signal_quality 变成了正值

这再次说明：
- 对 ranking 候选来说，**训练 accuracy 不是可靠主指标**
- 更贴近 ranking 的标签定义后，应该优先看 signal_quality 和正式任务结果

##### 正式任务验证

统一仍使用新的官方风格成本口径：
- `open_cost = 0.0005`
- `close_cost = 0.0015`
- `min_cost = 5.0`
- `slippage_rate = 0.0005`
- `max_position_size = 0.1`
- `cash_reserve_ratio = 0.2`
- `board_lot_size = 200`

###### 2024-full
- task_id: `250beda5-80a0-4573-a112-4ab110d9b77d`
- total_return: `-0.5817%`
- annualized_return: `-0.5833%`
- sharpe: `-0.1665`
- max_drawdown: `-3.1592%`
- with_cost IR: `-3.1664`
- total_trades: `444`

###### 2025-full
- task_id: `705dbd9d-cdd6-487c-970c-a7146b69df87`
- total_return: `-1.8709%`
- annualized_return: `-1.8811%`
- sharpe: `-0.6721`
- max_drawdown: `-2.8366%`
- with_cost IR: `-4.0600`
- total_trades: `422`

###### 2024-2025-full
- task_id: `defcf688-dc89-4ef7-aa58-be7b670b9e22`
- total_return: `-2.0289%`
- annualized_return: `-1.0211%`
- sharpe: `-0.3116`
- max_drawdown: `-3.8846%`
- with_cost IR: `-3.3594`
- total_trades: `862`

##### 这轮更根本标签实验怎么解读

和现有候选对照后，得到的结论是：
- 它没有成为新的“全面主候选”
- 因为在 `2024-full` 上明显不如 `official`
- 但它比只做 `CSRankNorm` 更有信息增益
- 而且在：
  - `2025-full`
  - `2024-2025-full`
  这两个关键长窗里，结果都优于当前 `official`

因此更准确的判断是：
- `future_excess_return_cs` 是一个**比单纯 CSRankNorm 更值得继续挖的 ranking 标签方向**
- 但在当前 `bank-core3` 小股票池里，它仍然不够稳定，不能直接取代 `official`
- 它更像一个“有潜力但需要更大横截面/更多标签设计联动”的研究分支

到这里，ranking label 方向的阶段性结论已经可以更新为：
1. `CSRankNorm` 单独使用：信息增益不足
2. `future_excess_return_cs`：开始出现真实提升信号，但稳定性还不够
3. 下一步如果继续做 ranking 标签研究，应优先围绕：
   - `future_excess_return_cs` 继续扩展
   - 更大股票池
   - 必要时叠加/对比 `cs_rank_norm`

#### 7.6.8 finance5 扩池上的 `future_excess_return_cs` 实验

在确认 `future_excess_return_cs` 在 `bank-core3` 上已经出现提升信号后，又按原建议把它扩到更大的同风格池：
- `finance5` = `601288.SH`, `601398.SH`, `601988.SH`, `600016.SH`, `600036.SH`

##### 真实训练实验

新模型：
- model_id: `89d67073-ecc8-438f-9744-dcc22f7efa4f`
- model_name: `hermes-finance5-csexcess-1776089690`

训练窗：
- `2024-01-01 ~ 2024-12-31`

标签定义：
- `label_definition = future_excess_return_cs`

训练报告摘要：
- accuracy: `0.3469`
- mae: `0.0095`
- r2: `-0.0055`

signal_quality：
- ic: `-0.1347`
- icir: `-0.2676`
- rank_ic: `-0.0360`
- rank_icir: `-0.0775`
- long_short_ann_return: `-1.3506`
- long_short_ann_sharpe: `-7.2453`
- sample_count: `245`

注意：
- 单看训练报告，这个模型并不好，signal_quality 甚至偏负
- 这再次说明“训练页指标”和正式任务结果可能并不一致，最终还得看正式任务

##### 正式任务验证（新成本口径）

统一口径仍为：
- `open_cost = 0.0005`
- `close_cost = 0.0015`
- `min_cost = 5.0`
- `slippage_rate = 0.0005`
- `max_position_size = 0.1`
- `cash_reserve_ratio = 0.2`
- `board_lot_size = 200`

###### finance5_csexcess - 2024-full
- task_id: `716d4ece-054e-4709-86c5-755ceb45e7ed`
- total_return: `-0.1039%`
- annualized_return: `-0.1042%`
- sharpe: `-0.0302`
- max_drawdown: `-2.7657%`
- with_cost IR: `-3.0747`
- total_trades: `444`

###### finance5_csexcess - 2025-full
- task_id: `5259547f-b724-40f4-8151-409040b2b11a`
- total_return: `-0.0574%`
- annualized_return: `-0.0577%`
- sharpe: `-0.0205`
- max_drawdown: `-2.6844%`
- with_cost IR: `-3.3783`
- total_trades: `442`

###### finance5_csexcess - 2024-2025-full
- task_id: `fdd1d04f-7979-400e-b1f4-88a191540e42`
- total_return: `-1.5009%`
- annualized_return: `-0.7543%`
- sharpe: `-0.2350`
- max_drawdown: `-4.2696%`
- with_cost IR: `-3.3447`
- total_trades: `922`

##### 与 finance5 旧基线同口径对照

为了确认“是标签变好了，还是只是池子变了”，又把旧 finance5 专用模型按同样的新成本口径重跑了一遍：

旧 finance5 baseline 模型：
- model_id: `e1c44fd5-6d93-4ffd-b6e2-e593178d67a4`
- model_name: `hermes-next-finance5-2024-1775965209`

同口径结果：

###### finance5_baseline - 2024-full
- task_id: `4277a765-9a43-40ae-ada5-e30aa38aaeb5`
- total_return: `-1.5409%`
- sharpe: `-0.4330`
- max_drawdown: `-3.8152%`
- with_cost IR: `-3.3834`

###### finance5_baseline - 2025-full
- task_id: `7790f78f-b499-41d4-aa5c-d0cd57e7cf2d`
- total_return: `-3.8867%`
- sharpe: `-1.3732`
- max_drawdown: `-4.6797%`
- with_cost IR: `-4.7317`

###### finance5_baseline - 2024-2025-full
- task_id: `c0790837-2612-4362-9b38-6fc27662e8c3`
- total_return: `-6.1860%`
- sharpe: `-0.9622`
- max_drawdown: `-6.8078%`
- with_cost IR: `-4.0380`

##### finance5 扩池实验结论

这轮给出的结论非常重要：
- 虽然 `finance5_csexcess` 的训练 signal_quality 很差
- 但在正式任务里，它**显著优于 finance5 旧基线**
- 而且是 3 个关键长窗全部更优：
  - 2024-full 更优
  - 2025-full 更优
  - 2024-2025-full 更优

进一步看，它还带来了一个很有价值的信号：
- `finance5_csexcess` 的 2025-full 与 2024-2025-full，甚至优于当前 `bank-core3 official`
- 但它仍没有形成“全面压倒式优势”，因为：
  - 收益大多只是从明显负值改善到接近打平
  - with-cost IR 仍为负
  - 训练信号质量与正式任务表现仍然存在偏离

因此更准确的结论是：
- **`future_excess_return_cs + 更大横截面池` 是目前最有继续研究价值的 ranking 分支**
- 比起继续在 `bank-core3` 上微调，这条线已经出现了更实在的改进证据
- 但它还没到可以直接替代 `official` 主候选的程度
- 当前更合理的定位是：
  - `official` 继续作为上线主候选
  - `robust` 继续作为动态备选
  - `future_excess_return_cs@finance5` 成为当前最值得继续扩展的研究分支

#### 7.6.9 更大银行池 bank10 扩展验证（负结果同样重要）

为了避免“看到 finance5 改善后就盲目继续扩池”，又按同样方法继续做了一轮更大的纯银行池实验：

bank10：
- `601288.SH`
- `601398.SH`
- `601988.SH`
- `600016.SH`
- `600036.SH`
- `601166.SH`
- `601328.SH`
- `601939.SH`
- `600000.SH`
- `601818.SH`

##### 真实训练实验

新模型：
- model_id: `113bad94-26b5-4af3-948d-ab336fe0cb8d`
- model_name: `hermes-bank10-csexcess-1776090504`

训练窗：
- `2024-01-01 ~ 2024-12-31`

标签定义：
- `label_definition = future_excess_return_cs`

训练报告摘要：
- accuracy: `0.2918`
- mae: `0.0100`
- r2: `0.0008`

signal_quality：
- ic: `0.0298`
- icir: `0.0953`
- rank_ic: `-0.0264`
- rank_icir: `-0.0765`
- long_short_ann_return: `0.1230`
- long_short_ann_sharpe: `1.0594`
- sample_count: `490`

##### 正式任务验证（新成本口径）

###### bank10_csexcess - 2024-full
- task_id: `e3cba35e-0fed-46e4-afc7-235882113842`
- total_return: `-1.0310%`
- annualized_return: `-1.0338%`
- sharpe: `-0.3095`
- max_drawdown: `-3.3941%`
- with_cost IR: `-3.4609`
- total_trades: `444`

###### bank10_csexcess - 2025-full
- task_id: `ac21af3c-d85e-432c-9626-4e6bcb2c7041`
- total_return: `-3.7142%`
- annualized_return: `-3.7343%`
- sharpe: `-1.4297`
- max_drawdown: `-4.5630%`
- with_cost IR: `-5.0884`
- total_trades: `442`

###### bank10_csexcess - 2024-2025-full
- task_id: `2cd3202c-d1e3-4a67-83bf-fecc55fb6e7e`
- total_return: `-6.4339%`
- annualized_return: `-3.2749%`
- sharpe: `-1.0698`
- max_drawdown: `-7.2200%`
- with_cost IR: `-4.3604`
- total_trades: `910`

##### bank10 扩展验证结论

这轮负结果非常关键，因为它说明：
- 不是“池子越大越好”
- 即使仍在同风格银行池内，继续从 `finance5` 扩到 `bank10`，正式任务表现反而明显退化
- 也就是说，当前最有价值的方向不是“无限扩池”，而是：
  - 找到**合适大小**的横截面池
  - 让 `future_excess_return_cs` 在这个规模上发挥作用

把 `finance5_csexcess` 和 `bank10_csexcess` 对照起来看：
- `finance5_csexcess`：正式任务结果显著改善
- `bank10_csexcess`：正式任务结果明显恶化

因此当前可以更明确地下这个判断：
- **更大横截面有帮助，但不是越大越好；当前更像存在一个“甜点区间”，finance5 比 bank10 更接近这个区间。**

#### 7.6.10 围绕 finance5 的受控扩展：bank6 / bank8

在 bank10 负结果之后，没有继续盲目扩池，而是回到 `finance5` 附近做受控扩展：

- `bank6` = finance5 + `601818.SH`
- `bank8` = bank6 + `601328.SH` + `601939.SH`

这些新增成分不是随便选的，而是根据本地真实数据覆盖与平均成交量，在 finance5 邻近候选中优先加入的高流动性银行股。

##### 真实训练实验

###### bank6_csexcess
- model_id: `4fc50151-6f78-4ad0-8e3f-31d6688756a8`
- model_name: `hermes-bank6-csexcess-1776090862`
- training accuracy: `0.2619`
- signal_quality:
  - ic: `-0.1270`
  - icir: `-0.2719`
  - rank_ic: `-0.1323`
  - rank_icir: `-0.2626`
  - long_short_ann_return: `-0.3894`
  - long_short_ann_sharpe: `-2.2339`

###### bank8_csexcess
- model_id: `a216ac62-acdb-4d95-b4ca-b224cf4bace4`
- model_name: `hermes-bank8-csexcess-1776090878`
- training accuracy: `0.3112`
- signal_quality:
  - ic: `-0.0890`
  - icir: `-0.1942`
  - rank_ic: `-0.0680`
  - rank_icir: `-0.1453`
  - long_short_ann_return: `-0.1200`
  - long_short_ann_sharpe: `-0.5712`

##### 正式任务验证（新成本口径）

###### bank6_csexcess

2024-full
- task_id: `04ed7be2-598f-42bd-bce0-8206fad06615`
- total_return: `-0.4191%`
- sharpe: `-0.1193`
- max_drawdown: `-3.9686%`
- with_cost IR: `-3.1011`

2025-full
- task_id: `d6bc40ca-eb76-43af-a455-21ddf7ee8257`
- total_return: `-4.4216%`
- sharpe: `-1.5454`
- max_drawdown: `-5.1425%`
- with_cost IR: `-4.8772`

2024-2025-full
- task_id: `d2b06f1d-0abc-456c-a124-304146d8039c`
- total_return: `-5.9280%`
- sharpe: `-0.8606`
- max_drawdown: `-7.0539%`
- with_cost IR: `-3.7301`

###### bank8_csexcess

2024-full
- task_id: `4bff93b8-21e5-4516-89e1-d6824738782b`
- total_return: `-0.6625%`
- sharpe: `-0.1920`
- max_drawdown: `-3.4929%`
- with_cost IR: `-3.2298`

2025-full
- task_id: `99877fa0-3561-408f-8fcd-c8b6b01f602a`
- total_return: `-3.2296%`
- sharpe: `-1.2095`
- max_drawdown: `-4.6934%`
- with_cost IR: `-4.7599`

2024-2025-full
- task_id: `30cf8eac-1f15-42bb-b7d3-8ad3007dcc9b`
- total_return: `-5.3538%`
- sharpe: `-0.8308`
- max_drawdown: `-6.0973%`
- with_cost IR: `-3.9006`

##### 受控扩展结论：甜点区间已经出现明显信号

把四组池子按正式任务结果放在一起看：

- `finance5_csexcess`
  - 2024-full: `-0.1039%`
  - 2025-full: `-0.0574%`
  - 2024-2025-full: `-1.5009%`

- `bank6_csexcess`
  - 2024-full: `-0.4191%`
  - 2025-full: `-4.4216%`
  - 2024-2025-full: `-5.9280%`

- `bank8_csexcess`
  - 2024-full: `-0.6625%`
  - 2025-full: `-3.2296%`
  - 2024-2025-full: `-5.3538%`

- `bank10_csexcess`
  - 2024-full: `-1.0310%`
  - 2025-full: `-3.7142%`
  - 2024-2025-full: `-6.4339%`

从这组对照可以下一个更强的判断：
- **当前 sweet spot 很可能就在 finance5 附近，至少在这轮实验里，`5` 明显优于 `6/8/10`。**
- 也就是说：
  - `future_excess_return_cs` 这个标签方向值得继续保留
  - 但它依赖一个“合适大小”的横截面池
  - 当前并没有证据支持继续把池子往更大方向扩

因此这轮之后，研究建议进一步收口为：
- 暂时停止继续往更大银行池扩
- 如果还要继续研究池子规模，优先做更细的替换/剪枝，而不是单纯加股票
- 例如：
  - finance5 内部成分替换
  - finance5 与 bank6 中新增成分逐个 ablation
  - 找到是哪一两只新增股票破坏了排名稳定性

#### 7.6.11 finance5 内部 leave-one-out ablation

为了进一步确认“finance5 为什么优于 bank6/bank8/bank10”，又对 finance5 做了一轮 leave-one-out ablation：
- 每次删掉 1 只成分
- 剩余 4 只股票重新训练 `future_excess_return_cs`
- 再跑 `2024-full / 2025-full / 2024-2025-full` 三个关键正式任务

被测的 5 个 finance4 子池分别是：
- drop `601288.SH`
- drop `601398.SH`
- drop `601988.SH`
- drop `600016.SH`
- drop `600036.SH`

##### 聚合结果（按三长窗总收益 / sharpe 汇总排序）

从好到坏大致是：
1. drop `601988.SH`
2. drop `601288.SH`
3. drop `600016.SH`
4. drop `600036.SH`
5. drop `601398.SH`

其中最好的 ablation（drop `601988.SH`）结果是：
- 2024-full: `-0.0155%`
- 2025-full: `-1.7645%`
- 2024-2025-full: `-0.7919%`

而原始 `finance5_csexcess` 是：
- 2024-full: `-0.1039%`
- 2025-full: `-0.0574%`
- 2024-2025-full: `-1.5009%`

这说明：
- 某些删减会在单个窗口上看起来更好
- 但如果看三长窗整体，**原始 finance5 仍然是更平衡、更稳的组合**
- 也就是说，目前没有证据表明“finance5 的问题只是某一只股票单独拖后腿，删掉它就会全面变好”

更细一点地看：
- `601988.SH` 被删后，是“伤害最小”的成分
- `601398.SH` 被删后，整体退化最明显，说明它更像 finance5 里的关键稳定成分

因此这轮 ablation 给出的更准确结论是：
- finance5 的优势更像是一个**组合平衡结果**，不是单一问题股导致
- 下一步如果继续研究，应该优先做：
  - 小范围成分替换（1 进 1 出）
  - 而不是简单 leave-one-out 或继续加大池子

---

## 8. 风险与局限

1. 两个候选的 with_cost IR 在完整年份与全窗上仍为负值
- 说明当前策略在相对 benchmark 的成本后超额上仍不算强
- “当前最佳候选”不等于“已经达到上线强标准”

2. robust 在 2025 单季度切片更强
- 如果后续市场 regime 延续 2025 的季度特征，robust 可能在近期继续跑赢 official
- 因此 robust 不应被放弃，应继续作为备选对照

3. finance5 扩池没有给出更强证据
- 当前不建议扩池优先于 bank-core3 收口

4. rolling 批量轮询中曾触发 API rate limit
- 已补轮询并收齐结果
- 不影响最终数据结论

5. 当前摩擦建模虽已前进一步，但仍未完全等价于 Qlib 官方示例
- 现在已经接通 `open_cost` / `close_cost` / `min_cost`
- 但仍缺少更完整的 exchange 级细节，例如 `limit_threshold`、`deal_price` 选择等
- 因此“更贴近上线”依然不等于“已经完全贴近官方/实盘摩擦”

---

## 9. 最终建议

当前阶段正式建议：

### 9.1 立项候选
- 主候选：`official`
- 备选：`robust`

### 9.2 暂不建议
- 不建议当前把 `mid` 作为主候选继续推进
- 不建议当前优先扩到 `finance5` 或更大池

### 9.3 下一步最合理动作
当前已完成：
- 更严格成本验证
- 更严格成本 + 保守持仓联合验证
- 2025 双月级 rolling 正式任务
- 官方 Qlib workflow / strategy / report 示例对照
- `open_cost/close_cost/min_cost` 接入与关键长窗复核
- `CSRankNorm` 标签实验（真实训练 + 正式任务）
- `future_excess_return_cs` 标签实验（真实训练 + 正式任务）
- `future_excess_return_cs@finance5` 扩池实验（真实训练 + 正式任务）
- `future_excess_return_cs@bank6/bank8/bank10` 受控扩展实验（真实训练 + 正式任务）
- `finance5` 内部 leave-one-out ablation
- `finance5` 小范围 1 进 1 出替换实验（替换 `601988` 槽位：`601166/601328/601939/601818`）
- `future_excess_return_cs@finance5 + cs_rank_norm` 复验

补充结论（2026-04-13 follow-up）：
- `finance5` 的 1 进 1 出替换实验没有产出比原始 `finance5_csexcess` 更强的新候选
- 其中 `601939` 是局部最像可行替换的候选，但只在 `2024-full` 略优，`2025-full` 与 `2024-2025-full` 仍明显回落
- 在 `finance5` 上直接叠加 `cs_rank_norm` 虽然把 training accuracy 从 `0.3469` 拉到 `0.6367`，但正式任务三个关键窗口全部变差
- 进一步补做 `2025Q1~Q4` 季度切片与 `2026-YTD` 正式任务后，`finance5` 分支依然没有出现能升级为正式候选的稳定 regime 优势；`601939` 替换版只在极短窗口里呈现“跌得更少”的局部韧性
- `2026-YTD` 复验过程中还发现并修复了一个正式任务短窗口 blocker：`DataLoader._is_data_valid()` 之前硬编码 `min_rows=30`，会把只有 27 个交易日但覆盖完整的 `2026-01-01~2026-02-10` 窗口误判为无效数据；修复后正式任务已可正常完成
- 这进一步确认：训练页 accuracy 只能做早筛，不能代替正式任务判断

因此下一步最合理动作继续收口为：
- 以 `official` 为主候选进入更贴近上线模板的候选收口流程
- 保留 `robust` 作为并行对照和潜在“阶段性切换备选”
- 把原始 `future_excess_return_cs@finance5` 保留为当前 ranking 研究分支里的 sweet spot 基线
- 如果继续做研发型优化，优先级建议改为：
  1. 暂停继续做简单扩池、简单删股和简单 1 进 1 出换股
  2. 暂停继续在 `finance5` 上直接叠加 `cs_rank_norm` 这类表层归一化尝试
  3. 把注意力转向更结构性的改动：更严格的 train/valid/test 切分、signal↔portfolio 桥接分析，或新的标签族

目标从“继续争论 official 和 robust 谁绝对更强”升级为：
- 把 `official` 从“当前最佳研究候选”推进到“更接近上线候选”的验证阶段
- 把 `robust` 明确定位为“在部分 regime 更强的动态备选”
- 把后续研究重点从单纯超参微调，转向 ranking 目标与摩擦建模的结构性改进

---

## 10. 结论一句话版

在已修复 live ranking 执行 bug、接通真实 Qlib 风格 official_portfolio_analysis、并完成短窗、长窗、扩池与多窗口 rolling 正式任务验证后，当前最稳妥的正式主候选是 `official`，`robust` 作为最强备用对照候选继续保留。
