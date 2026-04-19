# official / robust daily topk overlap & ranking drift analysis

日期：2026-04-14

目标：
- 沿着上一版 bridge analysis 继续往下拆，直接回答：
  - official / robust 每天的 topk 排序到底有多像？
  - 差异主要是“同一篮子内顺序不同”，还是“连 top2 篮子都不同”？
  - 这些差异集中在哪些窗口、哪些月份、哪些股票上？

## 1. 方法说明

本次分析没有用“手写近似排序”，而是尽量贴近真实执行链：

1. 使用 `PredictionEngine.predict_return_series(...)`
2. 读取 official / robust 真实模型文件：
   - official: `53d9e8ad-e134-4b53-ba50-39a9c91f23df`
   - robust: `33b2fd75-af83-4d5f-bc2c-28dbad9fffa2`
3. 对银行 core3 股票池逐日生成预测分数：
   - `600036.SH`
   - `601288.SH`
   - `601398.SH`
4. 排序时复用 live executor 的真实 tie-break 规则：
   - `sorted(scores.items(), key=lambda item: (item[1], item[0]), reverse=True)`
5. 统计：
   - same_top2_days
   - same_top1_days
   - same_full_ranking_days
   - basket_divergence_days
   - monthly_basket_divergence
   - per-stock top1 / top2 / excluded counts

关键口径：
- 因为这里只有 3 只股票，`topk=2`
- 所以一旦 top2 篮子不同，就意味着“被排除的那只股票不同”
- 这比一般大股票池更容易直接映射到组合差异

## 2. 结果总览

### 2.1 2024-full
- common_days = `242`
- same_top2_days = `123` (`50.83%`)
- same_top1_days = `153` (`63.22%`)
- same_full_ranking_days = `123` (`50.83%`)
- basket_divergence_days = `119`
- order_only_divergence_days = `0`

解读：
- 2024 年里，两个模型只有一半左右的交易日给出相同 top2 篮子
- 没有出现“top2 相同但顺序不同”的 order-only divergence
- 也就是说，在这个 3 股票 + top2 的小池里：
  - 只要两边不一致，基本就是“实际持仓篮子不同”，不是表面排序差异

### 2.2 2025-full
- common_days = `243`
- same_top2_days = `122` (`50.21%`)
- same_top1_days = `152` (`62.55%`)
- same_full_ranking_days = `122` (`50.21%`)
- basket_divergence_days = `121`
- order_only_divergence_days = `0`

解读：
- 2025 年和 2024 年几乎一样，依然接近“五五开”
- 所以 official / robust 不是“偶尔有几个漂移日”，而是长期持续存在篮子分歧

### 2.3 2024-2025-full
- common_days = `485`
- same_top2_days = `234` (`48.25%`)
- same_top1_days = `299` (`61.65%`)
- same_full_ranking_days = `234` (`48.25%`)
- basket_divergence_days = `251`
- order_only_divergence_days = `0`

解读：
- 拉长到两年全窗后，分歧反而更明显
- same_top2 比例已经跌到 50% 以下
- 这进一步支持之前那条判断：
  - official vs robust 的 portfolio 差异，核心不是信号数量，而是 daily basket allocation 差异

### 2.4 2026-ytd
- common_days = `27`
- same_top2_days = `23` (`85.19%`)
- same_top1_days = `23` (`85.19%`)
- same_full_ranking_days = `23` (`85.19%`)
- basket_divergence_days = `4`
- order_only_divergence_days = `0`

解读：
- 2026-ytd 里两者已经明显更接近
- 只有 4 个分歧日
- 也就是说，短窗里 portfolio 差异更多是由少数关键日期决定，而不是全年那种持续性大漂移

## 3. 分歧月份分布

### 2024-full
分歧最高的月份：
- `2024-11`: `17` 天
- `2024-06`: `12` 天
- `2024-10`: `11` 天
- `2024-01` / `2024-05` / `2024-09`: 各 `10` 天

结论：
- 2024 的 ranking drift 不是集中在开头 warmup 阶段
- 到年末（尤其 11 月）仍然持续明显

### 2025-full
分歧最高的月份：
- `2025-09`: `16` 天
- `2025-03`: `13` 天
- `2025-05` / `2025-11` / `2025-12`: 各 `12` 天

结论：
- 2025 的分歧峰值出现在下半年，但并不是单月孤立现象
- 仍然说明这不是偶发抖动，而是 regime 下的持续性排序偏好差异

### 2026-ytd
- 分歧只集中在 `2026-02`
- 具体日期：
  - `2026-02-05`
  - `2026-02-06`
  - `2026-02-09`
  - `2026-02-10`

结论：
- 2026-ytd 的差异几乎就是一段很短的 2 月初微型 regime 分叉

## 4. 个股层面的 ranking drift

### 4.1 2024-full
official：
- top1 counts
  - `601398.SH`: `181`
  - `601288.SH`: `41`
  - `600036.SH`: `20`
- excluded counts
  - `600036.SH`: `201`
  - `601288.SH`: `35`
  - `601398.SH`: `6`

robust：
- top1 counts
  - `601398.SH`: `129`
  - `601288.SH`: `60`
  - `600036.SH`: `53`
- excluded counts
  - `600036.SH`: `146`
  - `601288.SH`: `60`
  - `601398.SH`: `36`

关键差异：
- official 明显更偏向 `601398.SH` 当 leader
- official 也更频繁把 `600036.SH` 排除在 top2 外
- robust 相比之下：
  - 更常把 `600036.SH` 拉进 top2
  - 也更愿意让 `600036.SH` / `601288.SH` 抢到 top1

### 4.2 2025-full
official：
- top1 counts
  - `601398.SH`: `149`
  - `601288.SH`: `69`
  - `600036.SH`: `25`
- excluded counts
  - `600036.SH`: `172`
  - `601288.SH`: `40`
  - `601398.SH`: `31`

robust：
- top1 counts
  - `601398.SH`: `104`
  - `601288.SH`: `71`
  - `600036.SH`: `68`
- excluded counts
  - `600036.SH`: `122`
  - `601288.SH`: `68`
  - `601398.SH`: `53`

关键差异：
- 2025 年这种倾向更明显：
  - official 继续偏 `601398.SH`
  - robust 明显抬高 `600036.SH`
- 这与上一版组合结果里 `2025-full` robust 略优，是一致的信号

### 4.3 2026-ytd
official：
- top1 counts
  - `601398.SH`: `23`
  - `600036.SH`: `4`
- excluded counts
  - `600036.SH`: `23`
  - `601288.SH`: `4`

robust：
- top1 counts
  - `601398.SH`: `27`
- excluded counts
  - `600036.SH`: `27`

关键差异：
- robust 在 2026-ytd 的 27 天里，始终把 `600036.SH` 排除在 top2 之外
- official 只在最后 4 个分歧日里，把 `600036.SH` 提升进 top2，转而排除 `601288.SH`
- 但即便如此，robust 的 YTD portfolio 仍略优
- 所以这 4 天的 `600036.SH` 倾斜，并没有给 official 带来净优势

## 5. 这次最重要的新发现

### 发现 1：当前没有“order-only drift”，几乎全是“basket drift”
四个窗口里：
- order_only_divergence_days 全部是 `0`

这意味着：
- 在当前 bank-core3 + topk=2 场景里
- official / robust 的差异不是“篮子一样，只是 leader 次序不同”
- 而是“连被纳入 top2 的股票集合都不同”

这是很关键的，因为它直接解释了：
- 为什么 signal 数量差不多
- 但 portfolio 结果还能持续分叉

### 发现 2：official 长期更偏 `601398.SH`，robust 长期更愿意给 `600036.SH` 权重
这条规律在 2024 / 2025 都稳定存在：
- official 更常把 `601398.SH` 放到 top1
- official 更常排除 `600036.SH`
- robust 更常把 `600036.SH` 留在 top2 里

这说明两模型的“排序风格”不是随机噪声，而是稳定的结构性偏差。

### 发现 3：2025 和短窗里，robust 的 ranking 风格更贴近有效组合；但在 2024 长窗里，这个风格反而吃亏
这和上一版 bridge report 是一致的：
- `robust` validation RankIC 更强
- 但 portfolio 传导是 regime-dependent
- 现在通过 daily basket drift 可以更具体地说：
  - 不是 RankIC 假
  - 而是它偏好的“进/出哪只股票”在不同年份收益贡献不同

## 6. 对下一步的直接建议

基于这轮结果，最值钱的下一刀不是继续泛泛跑更多窗口，而是：

1. 做 per-stock contribution bridge
- 把 `600036.SH / 601288.SH / 601398.SH` 的收益贡献、回撤贡献、换手贡献拆开
- 优先验证：
  - robust 在 2025 更好，是否主要因为提高了 `600036.SH` 的持仓机会
  - official 在 2024 更好，是否主要因为更偏向 `601398.SH`
- 本轮这一步已经补完，见：
  - `docs/reports/2026-04-14-official-robust-per-stock-contribution-analysis.md`
- 新增结论：
  - official 在长窗里主要靠 `601398.SH` 拉开优势
  - robust 在 2025 的领先，核心更来自 `601288.SH` 转正，而不只是 `600036.SH`

2. 做 divergence-date event slicing
- 重点看：
  - `2024-11`
  - `2025-09`
  - `2026-02-05 ~ 2026-02-10`
- 看这些日期附近三只股票各自的真实后续收益和交易影响

3. 如果后面要把 bridge summary 再产品化
- 可以考虑给 report 层继续补：
  - `ranking_overlap_summary`
  - `topk_divergence_monthly`
  - `per_stock_rank_counts`
- 这样就不只知道“哪个模型收益更好”，还能知道“它是怎么排出来的”

## 7. 一句话总结

一句话收口：
- official / robust 在 bank-core3 的 daily ranking 差异，不是轻微顺序波动，而是接近一半交易日都会出现的真实 top2 篮子分歧；其中 official 长期更偏向 `601398.SH`、更常排除 `600036.SH`，而 robust 更愿意把 `600036.SH` 放进 top2，这种稳定的选股风格差异正是 portfolio 分 regime 分叉的更直接解释。
