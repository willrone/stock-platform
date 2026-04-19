# official / robust per-stock contribution bridge analysis

日期：2026-04-14

目标：
- 在 daily topk overlap / ranking drift 之后，继续回答一个更具体的问题：
  - official / robust 的组合差异，最后到底是由哪只股票贡献出来的？
- 这次直接用正式任务 `trade_history` 的真实成交记录来做归因

## 1. 方法说明

数据来源：
- `tasks.result.trade_history`
- 关键窗口：
  - `2024-full`
  - `2025-full`
  - `2024-2025-full`
  - `2026-ytd`
- 股票池：
  - `600036.SH`
  - `601288.SH`
  - `601398.SH`

本次按每只股票聚合：
- buy_trades / sell_trades
- buy_notional / sell_notional
- commission
- slippage_cost
- realized_pnl
- net_contribution
- winning_sell_trades / losing_sell_trades

口径说明：
- 当前 `trade_history` 可直接拿到每笔成交的 `pnl`
- 因此这次的 per-stock contribution 主要代表：
  - 已实现盈亏归因
  - 再结合该股票相关的交易次数、成交额、成本支出看风格差异
- 它不是严格意义上的完整持仓归因模型，但已经足够解释当前 official / robust 的主要分叉来源

## 2. 结果总览

### 2.1 2024-full
组合层结果：
- official total_return = `0.003216`
- robust total_return = `-0.012660`

per-stock net contribution：
- official
  - `601398.SH = +2847.726`
  - `601288.SH = +451.589`
  - `600036.SH = -1806.524`
- robust
  - `601398.SH = +1561.526`
  - `601288.SH = -98.956`
  - `600036.SH = -1653.346`

关键差异（robust - official）：
- `601398.SH = -1286.200`
- `601288.SH = -550.545`
- `600036.SH = +153.178`

解释：
- 2024 年 official 的优势，最主要来自：
  - `601398.SH` 明显赚得更多
  - `601288.SH` 在 official 里还能保持正贡献，但 robust 已经转负
- 虽然 robust 在 `600036.SH` 上亏得略少，但远不足以抵消：
  - `601398.SH` 少赚的这大块利润
- 这和上一份 ranking drift 结论是吻合的：
  - official 更偏 `601398.SH`
  - 在 2024 这个偏好是有效的

### 2.2 2025-full
组合层结果：
- official total_return = `-0.023597`
- robust total_return = `-0.018870`

per-stock net contribution：
- official
  - `601398.SH = +125.153`
  - `601288.SH = -5.483`
  - `600036.SH = -1456.568`
- robust
  - `601288.SH = +751.080`
  - `601398.SH = -283.372`
  - `600036.SH = -1356.968`

关键差异（robust - official）：
- `601288.SH = +756.563`
- `601398.SH = -408.526`
- `600036.SH = +99.600`

解释：
- 2025 年 robust 略优，最关键的贡献源不是 `600036.SH`
- 真正最大的胜负手是：
  - `601288.SH` 在 robust 里从接近 0 / 小亏，翻成了明显正贡献
- 同时：
  - robust 在 `600036.SH` 上也确实亏得更少
  - 但这只是次要补充项
- 换句话说，2025 年 robust 的优势是：
  - 一部分来自更少踩 `600036.SH`
  - 但更核心的是它把 `601288.SH` 做成了赚钱来源

### 2.3 2024-2025-full
组合层结果：
- official total_return = `-0.025245`
- robust total_return = `-0.029763`

per-stock net contribution：
- official
  - `601398.SH = +2348.145`
  - `601288.SH = -247.204`
  - `600036.SH = -2412.690`
- robust
  - `601398.SH = +901.896`
  - `601288.SH = +566.028`
  - `600036.SH = -2301.432`

关键差异（robust - official）：
- `601398.SH = -1446.249`
- `601288.SH = +813.232`
- `600036.SH = +111.257`

解释：
- 这个长窗最有意思：
  - robust 在 `601288.SH` 上明显更好
  - 在 `600036.SH` 上也略好
  - 但它在 `601398.SH` 上少赚太多
- 最终结果就是：
  - `601398.SH` 的大幅劣化，压过了其它两只股票带来的改善
- 所以 2024-2025-full 里 official 能赢，不是“整体都更优”，而是：
  - 对 `601398.SH` 的偏好赚到了决定性的那一笔大钱

### 2.4 2026-ytd
组合层结果：
- official total_return = `-0.002114`
- robust total_return = `-0.001596`

per-stock net contribution：
- official
  - `600036.SH = +122.572`
  - `601398.SH = -91.691`
  - `601288.SH = -120.223`
- robust
  - `600036.SH = +33.089`
  - `601398.SH = 0.000`
  - `601288.SH = -120.223`

关键差异（robust - official）：
- `601398.SH = +91.691`
- `600036.SH = -89.483`
- `601288.SH = 0.000`

解释：
- 2026-ytd 的边际优势，本质上来自：
  - robust 没有在 `601398.SH` 上实现那笔负贡献
- 代价是：
  - 它在 `600036.SH` 上的正贡献也比 official 小
- 两边几乎刚好互相抵消，但 robust 略占优
- 这和 ranking drift 里看到的 2 月 4 个关键分歧日高度一致：
  - official 在最后几天把 `600036.SH` 拉进 top2
  - 但这并没有产生足够大的净好处

## 3. 把 per-stock contribution 和 ranking drift 连起来看

现在两份报告已经能拼起来：
- ranking drift report：解释“谁更常被放进 top2 / 谁更常被排除”
- contribution report：解释“这些排序差异最后赚没赚钱”

当前可以更清楚地下这个判断：

### A. official 的核心风格
- 长期更偏 `601398.SH`
- 更容易排除 `600036.SH`
- 在 2024 和 2024-2025 全窗里，这个风格总体更赚钱
- 其中最关键的 alpha 来源就是 `601398.SH`

### B. robust 的核心风格
- 更愿意把 `600036.SH` 留在 top2
- 同时在 2025 窗口里，把 `601288.SH` 做成了真正的正贡献来源
- 但在包含 2024 的长窗里，它对 `601398.SH` 的把握明显不如 official

### C. 当前最本质的 regime 解释
- `601398.SH` 更像是 official 风格在 2024 / 长窗上的主胜负手
- `601288.SH` 更像是 robust 风格在 2025 上的主胜负手
- `600036.SH` 则更像一个“robust 更愿意给机会、但收益效果依窗口变化”的弹性因子

所以，之前那句“robust 对近端 regime 更灵活、official 在长窗更稳”，现在已经能具体到股票层：
- official 的稳，主要是押中了 `601398.SH`
- robust 的灵活，主要是 2025 年更好地利用了 `601288.SH`，并适度降低了 `600036.SH` 的拖累

## 4. 这次最重要的新增结论

### 结论 1：2024 / 长窗的胜负关键是 `601398.SH`
- 2024-full：official 在 `601398.SH` 上多赚约 `1286`
- 2024-2025-full：official 在 `601398.SH` 上多赚约 `1446`

这已经足以解释为什么：
- robust validation RankIC 更高
- 但 official 在长窗 portfolio 上仍然更强

### 结论 2：2025 的 robust 优势，主要不是来自 `600036.SH`，而是来自 `601288.SH`
- `601288.SH` 的 delta 约 `+756.6`
- `600036.SH` 只是额外补了约 `+99.6`

所以如果接下来只盯着“robust 更喜欢 600036”来解释 2025 优势，是不完整的。
更准确的说法是：
- robust 在 2025 里，一边少踩了 `600036.SH`
- 更重要的是，它把 `601288.SH` 做成了正贡献

### 结论 3：短窗里边际差异很小，但能看出“少做错”比“多做对”更重要
2026-ytd 里：
- official 在 `600036.SH` 上赚得更多
- 但它在 `601398.SH` 上多亏了一笔
- robust 靠“不在 601398 上实现那笔负贡献”略胜一筹

这说明短窗里：
- 边际控制失误，可能比额外捕捉一点正收益更重要

## 5. 下一步最值钱的动作

现在我建议的下一刀已经更具体了：

1. divergence-date event slicing
重点拆：
- `2024-11`
- `2025-09`
- `2026-02-05 ~ 2026-02-10`

看这些日期附近：
- 三只股票的真实价格路径
- 模型分数排序
- 实际持仓变化
- 对组合收益的边际影响

2. 如果还想继续产品化 bridge summary
可以在 report API 里继续补更可解释的字段：
- per_stock_contribution_summary
- ranking_overlap_summary
- divergence_dates_preview

这样以后查看模型，就不只是看到：
- “收益谁高”
而是直接看到：
- “哪只股票把结果拉开了”

## 6. 一句话总结

一句话收口：
- official / robust 的 portfolio 分叉，最终已经可以落到个股层解释：official 在长窗里主要靠 `601398.SH` 取得优势，而 robust 在 2025 里的领先主要来自把 `601288.SH` 做成正贡献、并略微减轻 `600036.SH` 的拖累；这说明 current regime difference 不是抽象的 RankIC 现象，而是明确的个股贡献结构差异。
