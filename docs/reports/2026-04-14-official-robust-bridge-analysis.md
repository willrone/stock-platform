# official / robust signal-quality ↔ portfolio-quality bridge analysis

日期：2026-04-14

目标：
- 把历史主候选 `official` / `robust` 的 validation signal quality，与正式任务 portfolio 结果放到同一张分析表里
- 先回答一个核心问题：
  - `robust` validation RankIC 明显更强，这个优势有没有稳定传导到正式任务收益 / Sharpe / 回撤？

## 1. validation signal quality（回填结果）

模型：
- official
  - model_id: `53d9e8ad-e134-4b53-ba50-39a9c91f23df`
  - model_name: `hermes-official-bank-core3-1776037184`
- robust
  - model_id: `33b2fd75-af83-4d5f-bc2c-28dbad9fffa2`
  - model_name: `hermes-bank-core3-robust-20260412-230648`

validation 指标：
- official
  - accuracy = `0.6395`
  - ic = `-0.0529`
  - rank_ic = `-0.0803`
  - icir = `-0.0728`
  - rank_icir = `-0.1045`
  - sample_count = `147`
- robust
  - accuracy = `0.6358`
  - ic = `0.3264`
  - rank_ic = `0.3646`
  - icir = `0.4143`
  - rank_icir = `0.4932`
  - sample_count = `162`

第一层结论：
- 两者 validation accuracy 几乎一样
- 但 ranking 质量完全不是一个量级
- `robust` 的 validation RankIC / IC 明显更强
- `official` 甚至是负 RankIC

如果只看 signal quality，会自然预期：
- `robust` 应该更适合后续 `model_topk_dropout` / ranking 正式任务

## 2. 正式任务 portfolio 结果对照

### 2.1 2024-full
任务：
- official: `0ed1ee70-b37c-4de3-aff3-0731825e26f1`
- robust: `493e5ed9-c420-41f0-8fbf-2ef4f199a036`

结果：
- official
  - total_return = `0.003216`
  - sharpe = `0.097157`
  - max_drawdown = `-0.028022`
  - total_trades = `444`
  - raw_signal_count = `444`
  - rejected_signal_count = `0`
- robust
  - total_return = `-0.012660`
  - sharpe = `-0.378699`
  - max_drawdown = `-0.039071`
  - total_trades = `444`
  - raw_signal_count = `444`
  - rejected_signal_count = `0`

结论：
- 2024-full 是 `official` 明显更强
- 而且这里两者：
  - raw signals 一样
  - executed signals 一样
  - trades 一样
  - rejection 也一样是 0
- 所以差异不是“信号条数更多”，而是“排序/持仓分配不同导致的收益差异”

### 2.2 2025-full
任务：
- official: `00e488a2-32ab-4ff0-a8a5-1f8c2cf945bf`
- robust: `6315c552-292e-4838-95d0-000cb7ba1788`

结果：
- official
  - total_return = `-0.023597`
  - sharpe = `-0.878912`
  - max_drawdown = `-0.030172`
  - total_trades = `428`
  - raw_signal_count = `437`
  - rejected_signal_count = `9`
- robust
  - total_return = `-0.018870`
  - sharpe = `-0.680870`
  - max_drawdown = `-0.030393`
  - total_trades = `422`
  - raw_signal_count = `434`
  - rejected_signal_count = `12`

结论：
- 2025-full 则变成 `robust` 更强
- 它的收益损失更小，Sharpe 更好
- 但它并不是“更少 rejection”才赢：
  - `robust` rejection 反而更多（12 vs 9）
- 所以这里更像是：
  - ranking 顺序 / 换仓选择改善，抵消了更多 buy-lot rejection 的负面影响

### 2.3 2024-2025-full
任务：
- official: `2fc835f9-0766-4c18-9f0a-797c4899ea0e`
- robust: `20487997-819a-4ca9-8e35-0d6a3c72b3c9`

结果：
- official
  - total_return = `-0.025245`
  - sharpe = `-0.412903`
  - max_drawdown = `-0.033560`
  - total_trades = `904`
  - raw_signal_count = `917`
  - rejected_signal_count = `13`
- robust
  - total_return = `-0.029763`
  - sharpe = `-0.479255`
  - max_drawdown = `-0.042896`
  - total_trades = `880`
  - raw_signal_count = `905`
  - rejected_signal_count = `25`

结论：
- 拉长到 2024-2025 全窗后，又是 `official` 更强
- `robust`：
  - return 更差
  - Sharpe 更差
  - drawdown 更差
  - rejection 也更多（25 vs 13）
- 这里 `robust` 的 validation ranking 优势没有稳定转成 portfolio 优势

### 2.4 2026-ytd
任务：
- official: `1e172dce-15fa-40dd-88ea-65f047b85881`
- robust: `8835de8f-15ec-4d24-9c13-202e05cd8375`

结果：
- official
  - total_return = `-0.002114`
  - sharpe = `-1.643642`
  - max_drawdown = `-0.003553`
  - total_trades = `14`
  - raw_signal_count = `14`
  - rejected_signal_count = `0`
- robust
  - total_return = `-0.001596`
  - sharpe = `-1.234408`
  - max_drawdown = `-0.003553`
  - total_trades = `14`
  - raw_signal_count = `14`
  - rejected_signal_count = `0`

结论：
- 短窗 2026-ytd 里，`robust` 再次略优于 `official`
- 但优势是“边际更好”，不是压倒性更强
- 且这里两者 signal/trade/rejection 几乎完全同形，差异依然主要来自排序与持仓细节

## 3. bridge 结论：不是线性传导，而是 regime-dependent transmission

可以先下一个更准确的判断：

1. `robust` 的 validation ranking 质量，确实不是假信号
- 它在 `2025-full` 和 `2026-ytd` 里，都对应了更好的 portfolio 结果
- 说明 validation RankIC 的提升，至少在“较后期 / 较短窗”里是能传导到正式任务的

2. 但这种传导不是稳定线性的
- `2024-full` 和 `2024-2025-full` 反而是 `official` 更强
- 所以不能直接用一句：
  - “validation RankIC 更高，所以 portfolio 一定更好”
- 当前更像是：
  - `robust` 对近端 regime 更灵活
  - `official` 在包含 2024 的更长窗上更稳

3. 当前 portfolio 差异并不是由“信号量”主导
- 多个窗口里：
  - raw_signal_count 很接近
  - total_trades 很接近
  - rejection reasons 也高度同质
- 当前最主要的差异源，更像是：
  - 每日 topk 排序顺序
  - 实际落到哪只股票上的持仓分配
  - 不同 regime 下各股票贡献结构不同

4. rejection reasons 暂时不是主解释变量
- 当前长窗里最常见 rejection 基本都是：
  - `可买数量不足: 无法买入200股`
- 这更像是交易最小手数 / 资金分配约束带来的执行层摩擦
- 它会影响结果，但不足以单独解释 official vs robust 的主差异
- 甚至在 `2025-full` 里，`robust` rejection 更多，但组合结果仍更好

## 4. 当前最值得继续看的不是“更多窗口”，而是“排序落点”

因为现在已经能回答两件事：
- 不能只看 accuracy
- 也不能把 validation RankIC 当成 portfolio 的线性代理变量

所以下一步最值钱的分析，不是再重复跑更多同类窗口，而是：

1. 做 daily topk overlap / ranking drift analysis
- official vs robust 每天 top2/topk 的重合度是多少
- 分歧最大的日期集中在哪些月份
- 分歧落在哪只股票上

2. 做 per-stock contribution bridge
- `600036.SH / 601288.SH / 601398.SH` 各自对收益、回撤、换手、rejections 的贡献拆开看
- 当前已有信号表统计已经提示：
  - 两个模型在长窗里对各股票的 signal allocation 明显不同
  - 这比“总 signal 数”更可能解释组合差异
- 本轮已先补完前置排序层分析，见：
  - `docs/reports/2026-04-14-official-robust-ranking-drift-analysis.md`
- 新增结论是：
  - 当前几乎没有 order-only drift
  - 分歧主要是实际 top2 篮子分歧
  - official 长期更偏 `601398.SH`
  - robust 长期更愿意把 `600036.SH` 放进 top2

3. 把新接通的 official test 段评估纳入同一张桥接图
- train / validation / test 的 signal quality 并排
- 再与正式任务 windows 对比
- 这样才能分辨：
  - 是 validation 本身失真
  - 还是 test → portfolio 之间还有第二层衰减

## 5. 报告层落点

这轮分析不再只是研究文档结论，已经有对应的报告层承载：
- `GET /api/v1/models/{model_id}/evaluation-report` 现在会动态回带 `portfolio_bridge_summary`
- official / robust 的历史主候选都已经能从同一个模型报告接口里看到：
  - validation / test signal quality
  - formal backtest tasks rollup
  - best return / best sharpe / smallest drawdown task
  - signal summary（raw/executed/rejected/top stocks）

这意味着：
- 后续再看某个模型，不需要先手工翻 tasks 表，已经能先从 report API 直接拿到 bridge 概览
- 研究报告继续负责“解释为什么”，而模型报告接口已经开始负责“提供结构化证据”

## 6. 一句话总结

一句话收口：
- `robust` 的 validation RankIC 优势是真实的，但它向 portfolio 的传导是“分 regime 的”，不是稳定单调关系；当前 official vs robust 的主要差异更像来自每日排序落点与个股权重分配，而不是信号数量或 rejection 数量本身。
