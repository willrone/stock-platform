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

因此下一步最合理动作变为：
- 以 `official` 为主候选进入更贴近上线前的候选收口流程
- 保留 `robust` 作为并行对照
- 如果还要继续压测，优先做：
  - 月级 / 双月级 rolling 正式任务
  - 或更接近最终上线模板的正式任务复刻

目标从“确认 official 是否仍是主候选”升级为：
- 把 `official` 从“当前最佳研究候选”推进到“更接近上线候选”的验证阶段
- 同时验证 `robust` 是否在近期 regime 下值得作为动态切换备选

---

## 10. 结论一句话版

在已修复 live ranking 执行 bug、接通真实 Qlib 风格 official_portfolio_analysis、并完成短窗、长窗、扩池与多窗口 rolling 正式任务验证后，当前最稳妥的正式主候选是 `official`，`robust` 作为最强备用对照候选继续保留。
