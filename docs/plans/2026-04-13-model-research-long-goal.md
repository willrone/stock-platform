# Stock-Platform 模型训练与正式回测长期目标

> For Hermes: treat this as the default self-directed objective for the current stock-platform model research thread. Do not stop to ask whether to continue after each small result; advance to the next milestone unless blocked by destructive changes, missing credentials, or an explicit user redirect.

日期：2026-04-13

## 长目标

以 Qlib 官方 benchmark / workflow 为参考，在 stock-platform 里产出一个可正式上线的 model-driven ranking 候选：
- 必须在平台正式任务链路下验证，不以训练页分数单独定胜负
- 必须相对 current `official` 基线体现出更稳健、可复现、跨窗口不过拟合的改进
- 必须形成清晰的选型标准、实验边界和最终研究报告

## 当前已知事实

1. `official` 仍是当前正式主候选，`robust` 是备选，而且 `robust` 在季度 / 超短 YTD 窗口里更灵活。
2. `mid` 已经从训练页优势回落，不再是主推。
3. `future_excess_return_cs` 仍是当前 ranking 研究里最值得保留的标签方向。
4. `finance5` 证实过 sweet spot 特征，但后续 `leave-one-out`、`601988` 槽位 1进1出替换、以及直接叠 `cs_rank_norm`，都没有带来比原始 `finance5_csexcess` 更稳健的正式任务改进。
5. `2026-YTD` 这类短窗正式任务此前被 `DataLoader._is_data_valid()` 的硬编码 `min_rows=30` 误杀；当前已修复为随窗口长度动态缩放，并补了回归测试。
6. 当前更值得继续的方向，已经从“简单换股 / 简单归一化”转向更结构性的工程与研究动作：
   - 更严格的 train/valid/test 切分与独立验证
   - signal-quality ↔ portfolio-quality 的桥接分析
   - 历史主候选 `official` / `robust` 的 `signal_quality` 回填能力
   - 更贴近官方口径的成本建模与正式任务比较

## 默认推进路线

### Milestone 1: 官方经验映射收口
目标：把 Qlib 官方 workflow / strategy / report 的关键口径映射到 stock-platform 当前训练、信号分析与正式任务回测链路。
完成标准：
- 明确训练页指标只作为早筛，不作为最终裁判
- 正式任务比较口径固定为收益、回撤、IR、信号质量、rejection reasons
- 形成可复用的评估模板

### Milestone 2: finance5 小范围 1进1出替换实验
目标：围绕 `finance5` 做有限而高信息增益的成分替换，不再盲目扩池。
默认优先级：
1. 先把 `601988.SH` 视为最可替换槽位
2. 与邻近银行/金融候选做 1进1出 A/B 实验
3. 每个候选都跑训练 + 正式任务长窗验证
完成标准：
- 得到一组优于原始 `finance5` 或明确不优于 `finance5` 的证据
- 若无显著提升，则停止继续在扩池/换股上消耗过多轮次

### Milestone 3: ranking 标签/归一化增强
目标：在较优成分池上验证更贴近官方 ranking 的标签与归一化方案。
默认顺序：
1. `future_excess_return_cs`
2. `future_excess_return_cs + cs_rank_norm`
3. 只在确有增益时再继续更复杂设定
完成标准：
- 明确哪些设定只改善训练页，哪些设定真正改善正式任务结果

### Milestone 4: 独立测试窗与正式任务复验
目标：避免把局部时间窗偶然优势误判成可上线优势。
完成标准：
- 对入围候选统一跑独立窗口与全流程正式任务
- 汇总 with-cost / without-cost、drawdown、IR、trade stats、signal stats
- 判断其是否真的超越 `official` / `robust`

### Milestone 5: 收口与建议
目标：给出最终研究结论，而不是无限继续探索。
完成标准：
- 明确上线主候选、备选、淘汰项
- 明确下一步值得继续的研究方向和停止线
- 更新最终报告与可复用 skill

## 停止线

出现以下情况时，应停止当前分支并切到下一分支，而不是无休止细调：
- 连续多轮仅改善训练页指标，正式任务无改善
- 候选方案在 2024 / 2025 / 2024-2025 长窗下没有稳定优势
- 信号数量膨胀、rejection reasons 异常、或收益依赖单一窗口偶然性
- 新方案复杂度明显增加，但没有带来更稳健的正式任务收益/IR 改善

## 当前默认下一步

优先进入结构性收口阶段，而不是继续简单扩池/换股：

1. 工程侧
   - 继续按 Qlib 官方口径补齐正式任务成本建模（`open_cost` / `close_cost` / `min_cost`）的验证与比较
   - 保持短窗正式任务可用性回归（特别是 YTD / 月度 / 更短 out-of-sample）
   - 继续把训练链路拆成更清晰的 orchestrator / pipeline / result-assembler，并锁住 train/valid/test 适配回归

2. 研究侧
   - 对 `official` / `robust` 补做或回填 `signal_quality`
   - 做 signal-quality ↔ portfolio-quality 的桥接对比，重点看 IC / Rank IC / Long-Short 指标与正式任务收益、回撤、IR、rejection reasons 是否一致
   - 若桥接分析仍证明当前标签族上限有限，再进入下一轮新标签族探索

一句话：
- 下一步不是继续在 `finance5` 上做表层换股或叠归一化，
- 而是先把“官方口径成本 + 严格切分 + signal/portfolio 桥接”这条基础设施补稳，再决定下一轮新标签实验。
