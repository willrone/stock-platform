# official-style 对齐（保留自有股票池）推进记录

日期：2026-04-15

目标：
- 不强制切到完整 csi300/csi500 股票池
- 保留 stock-platform 自己的股票池
- 但把训练、信号、组合、执行这几层尽量向官方 Qlib workflow 对齐

## 1. 重新定义目标

在这个口径下，当前最重要的不是“股票池是不是官方全市场”，而是：
1. 训练侧是否使用官方段划分 / label / processors
2. 信号侧是否按官方 signal_quality 看 IC / RankIC / ICIR / RankICIR
3. 组合侧是否尽量贴近官方 TopkDropout 的持仓与换手语义
4. 执行侧是否把平台自有约束和官方组合语义区分清楚

## 2. 当前已确认对齐的部分

### A. 训练与数据契约
official replication 当前已具备：
- official_segments
  - train: 2008-01-01 ~ 2014-12-31
  - valid: 2015-01-01 ~ 2016-12-31
  - test: 2017-01-01 ~ 2020-08-01
- official_benchmark
  - csi300 -> SH000300
- official label expression
  - `Ref($close, -2) / Ref($close, -1) - 1`
- official learn processors
  - alpha158: DropnaLabel + CSZScoreNorm(label)
  - alpha360: DropnaLabel + CSRankNorm(label)

### B. signal/report 层
已能稳定产出：
- accuracy
- IC / RankIC
- ICIR / RankICIR
- train / validation / test 三段 evaluation
- portfolio_bridge_summary / official_record_summary

## 3. 当前最主要的非股票池差距

### 差距 1：TopkDropout 的“组合语义”还不够官方
官方默认语义：
- topk = 50
- n_drop = 5
- 在 csi300 上对应：
  - 持仓比例约 16.67%
  - 单次轮换比例约 10%

而当前小股票池里常见配置是：
- 3 股票池: topk=2, n_drop=1
  - 持仓比例 66.67%
  - 单次轮换比例 50%
- 5 股票池: topk=2, n_drop=1
  - 持仓比例 40%
  - 单次轮换比例 50%

结论：
- 即使不换股票池，当前组合层也仍明显比官方更集中、更高换手
- 这会放大单票暴露与 regime 波动，不利于把 signal-quality 稳定传导到 portfolio-quality

### 差距 2：训练强度仍偏 smoke
当前 official replication smoke 模型仍以轻量训练为主：
- num_iterations = 20
- alpha360 report metadata 中仍可见 validation_split 这类工程侧配置痕迹

结论：
- 这足以证明链路打通
- 但还不足以拿来对“官方风格表现”做强结论

### 差距 3：validation signal → portfolio 的传导仍不稳
alpha360 在 validation 的 IC / RankIC 明显优于 alpha158，
但 test 与 formal-task 没有稳定兑现。

结论：
- 当前主要问题不是“完全没有 alpha”
- 而是 alpha 不能稳定跨到 test / portfolio

## 4. 这次落实的一个具体改进

### TopkDropout 现在真正开始使用 hold_thresh / buffer 语义
之前问题：
- `ModelTopkDropoutStrategy` 会把 `hold_thresh` 透传到 trade_mode_config
- 但 `TopkDropoutTradeModeExecutor` 实际完全没有消费这个字段
- 导致我们没有办法在小股票池里用 buffer 降低不必要轮换

本轮修复：
- 在 `backend/app/services/backtest/execution/trade_modes.py`
- TopkDropout executor 现在会把：
  - `hold_thresh = 0` 视为无缓冲
  - `hold_thresh > 0` 视为允许持仓在 `topk + hold_thresh` 范围内继续保留
- 只有当持仓排名跌出这个 buffer 之外，才进入 sell_candidates

这意味着：
- 对保留自有股票池的 official-style 对齐来说，终于有了一个降低小池过度换手的可用杠杆
- 后续可以不只靠死板的 `topk / n_drop`，还可以通过 buffer 更接近官方“低一点换手”的组合语义

## 5. 当前推荐的下一步

### 第一优先：定义“自有股票池下的 official-style 参数映射”
建议不要再直接把：
- `topk=50`
- `n_drop=5`
生搬硬套到小池子。

更合理的下一步是明确一套映射原则：
1. 先固定股票池
2. 再按股票池规模推导：
   - 基础 topk
   - 基础 n_drop
   - buffer/hold_thresh
3. 让小池下的组合语义尽量更接近：
   - 更低换手
   - 更少无意义篮子翻转
   - 更贴近 official-style TopkDropout 的“保留赢家 + 温和轮换”

#### 第一版建议映射（保守）

定义：
- `target_hold_ratio = 50 / 300 ≈ 16.67%`
- `target_drop_ratio = 5 / 50 = 10%`

对自有股票池大小 `pool_size`，先用：
- `topk = min(pool_size - 1, max(2, ceil(pool_size * target_hold_ratio)))`
- `n_drop = max(1, ceil(topk * target_drop_ratio))`
- `hold_thresh = 0 if pool_size <= 3 else max(1, min(2, ceil((pool_size - topk) * 0.15)))`

这不是“完全复刻官方数值”，而是尽量保留官方语义：
- 持仓数不要过少到退化成单票
- 单次轮换尽量保持很小
- 对小池额外引入 buffer，弥补整数化后 `n_drop=1` 带来的高换手

按这套第一版映射，典型结果是：
- `pool=3` -> `topk=2`, `n_drop=1`, `hold_thresh=0`
- `pool=5` -> `topk=2`, `n_drop=1`, `hold_thresh=1`
- `pool=10` -> `topk=2`, `n_drop=1`, `hold_thresh=2`
- `pool=15` -> `topk=3`, `n_drop=1`, `hold_thresh=2`
- `pool=20` -> `topk=4`, `n_drop=1`, `hold_thresh=2`
- `pool=30` -> `topk=5`, `n_drop=1`, `hold_thresh=2`

这套参数最适合作为：
- official-style small-pool baseline
- 后续 alpha158 / alpha360 / official / robust 的统一 formal-task A/B 起点

### 第二优先：用新 buffer 机制重新跑一轮 formal-task A/B
优先对象：
- alpha158 / alpha360
- official / robust

目标：
- 验证在保留自有股票池的前提下
- 降低无意义换手后
- signal_quality 是否能更稳定传导到 portfolio-quality

#### 本轮已先完成的第一组真实 A/B（2020-short, 5 股票池）

新任务（`official_style=true`，其余 ranking 参数交给映射 + buffer 逻辑补齐）：
- alpha158: `15b07421-54a5-4d4e-8ecf-f629161ffaff`
- alpha360: `19381274-8258-4f66-8db5-262d46145b5a`

对照旧 smoke：
- alpha158 old: `603e36db-ee45-4587-bc38-65d4038bacc3`
- alpha360 old: `8d720316-1eb6-4633-a03a-b2aa5caf63a9`

结果：
- alpha158
  - old total_return: `-2.3393%`
  - new total_return: `-2.7833%`
  - old annualized_return: `-4.0120%`
  - new annualized_return: `-4.7657%`
  - old sharpe: `-0.6206`
  - new sharpe: `-1.0001`
  - trades: `232 -> 54`
- alpha360
  - old total_return: `-2.6321%`
  - new total_return: `-2.1668%`
  - old annualized_return: `-4.5093%`
  - new annualized_return: `-3.7185%`
  - old sharpe: `-0.5307`
  - new sharpe: `-0.5181`
  - trades: `237 -> 56`

第一轮结论：
- buffer 生效后，两边交易次数都大幅下降（约 `-180` 笔）
- 这说明我们新补的 `hold_thresh` 执行逻辑确实改变了组合行为，不是死代码
- 但收益层并没有统一改善：
  - alpha158 变差
  - alpha360 略有改善
- 这说明“减少换手”本身不是充分条件，真正关键仍然是：
  - 哪些股票被保留下来
  - 哪些股票被替换掉
  - 不同模型的 basket style 是否适合当前 regime

#### 新旧任务的 retained-basket 差异（最关键发现）

##### alpha158
新 own-pool official-style 版本，相比旧 smoke：
- `600519.SH`
  - pnl: `+13665.73 -> 0`
  - trades: `76 -> 0`
- `601288.SH`
  - pnl: `-19776.06 -> -996.59`
  - trades: `35 -> 9`
- `000651.SZ`
  - pnl: `-7181.03 -> -365.93`
  - trades: `28 -> 16`
- `000001.SZ`
  - pnl: `-6874.03 -> -1072.37`
  - trades: `17 -> 9`
- `600036.SH`
  - pnl: `+5823.84 -> -144.42`
  - trades: `76 -> 20`

解释：
- alpha158 新版本的最大问题，不是 `601288.SH` 还亏，而是：
  - 它几乎把旧版最强盈利源 `600519.SH` 完全拿掉了
- 虽然它同时显著减少了：
  - `601288.SH`
  - `000651.SZ`
  - `000001.SZ`
  这些亏损来源
- 但少掉的最大赢家 `600519.SH` 利润，最终仍大于这些止损收益
- 所以 alpha158 这轮“降换手后反而变差”的核心原因可以明确定位为：
  - buffer 让它少做错了很多次
  - 但也把真正的赚钱来源挡掉了

##### alpha360
新 own-pool official-style 版本，相比旧 smoke：
- `600519.SH`
  - pnl: `+34730.30 -> 0`
  - trades: `44 -> 0`
- `600036.SH`
  - pnl: `-28747.62 -> +1457.95`
  - trades: `48 -> 14`
- `601288.SH`
  - pnl: `-12818.24 -> +1202.67`
  - trades: `58 -> 24`
- `000001.SZ`
  - pnl: `-6749.64 -> -797.74`
  - trades: `58 -> 7`
- `000651.SZ`
  - pnl: `-1663.24 -> -1618.68`
  - trades: `29 -> 11`

解释：
- alpha360 新版本同样失去了 `600519.SH` 这个旧版最强盈利源
- 但它比 alpha158 更幸运 / 更贴合当前 regime 的地方在于：
  - 它把原本最大的拖累项 `600036.SH` 从巨大亏损，压成了小幅盈利
  - 同时把 `601288.SH` 从明显负贡献，拉成了正贡献
- 所以即使失去 `600519.SH`，alpha360 仍然靠：
  - 少踩 `600036.SH`
  - 修复 `601288.SH`
  抵消掉了大部分损失
- 这就是为什么 alpha360 在这轮里可以小幅改善，而 alpha158 不行

##### 当前最关键的新判断
- 新 official-style own-pool baseline 已经把问题进一步缩小到了“股票保留结构”这一层：
  - 不是简单的换手多/少
  - 而是 buffer 把哪些候选拦在了组合外面
- 这轮最值得记住的事实是：
  - alpha158 输在“把 `600519.SH` 这个赢家一起过滤掉了”
  - alpha360 赢在“虽然也失去 `600519.SH`，但同时大幅减少了 `600036.SH` 和 `601288.SH` 的错误暴露”

因此下一轮更值得做的，不是只继续压低换手，而是：
- 在 official-style own-pool baseline 上继续看 alpha158 / alpha360 / official / robust 的 retained-basket 差异
- 特别检查 buffer 生效后哪些股票留存变多，以及这些留存到底创造了还是损害了收益

#### 第二组真实 A/B（2024-2025-full, bank-core3）

新任务：
- official: `dfb1f4f4-3cc4-4b12-8368-afc40bf27ae7`
- robust: `cba296ee-e443-47e2-b6f0-b64a98245553`

对照旧任务：
- official old: `cf894471-ab6b-4d7a-b3b8-df60cd704723`
- robust old: `a136b9b5-4805-442d-a34f-617dd5abf6c8`

先说一个非常关键的事实：
- 对 `bank-core3`（3 股票池）来说，当前映射结果是：
  - `topk=2`
  - `n_drop=1`
  - `hold_thresh=0`
- 也就是说，这个池子太小，official-style 映射在组合参数上会退化回原始语义
- 因此这组 A/B 的改善，不应解释成“buffer 在 3 股票池里发挥了作用”，而更可能是：
  - 当前 runtime 其它修复累积后的净效果
  - 以及最新正式任务执行链与旧任务运行时状态的差异

结果：
- official
  - old total_return: `12.1711%`
  - new total_return: `22.6604%`
  - old annualized_return: `5.9192%`
  - new annualized_return: `10.7677%`
  - old sharpe: `0.9815`
  - new sharpe: `1.6012`
  - trades: `930 -> 238`
- robust
  - old total_return: `10.3870%`
  - new total_return: `15.6438%`
  - old annualized_return: `5.0723%`
  - new annualized_return: `7.5485%`
  - old sharpe: `0.8231`
  - new sharpe: `1.1851`
  - trades: `930 -> 400`

##### official 的个股变化
- `601288.SH`
  - pnl: `4643.58 -> 11507.37`
  - trades: `347 -> 75`
- `601398.SH`
  - pnl: `7252.95 -> 10093.54`
  - trades: `159 -> 53`
- `600036.SH`
  - pnl: `2488.39 -> 1383.34`
  - trades: `424 -> 110`

解释：
- official 新版本最大的提升，不是来自更多交易，而是：
  - 用更少交易，把 `601288.SH` 和 `601398.SH` 做得更好
- 这表明当前 official 在 bank-core3 上的强项，依然是：
  - `601288.SH`
  - `601398.SH`
- 而 `600036.SH` 并不是它的主要 alpha 来源

##### robust 的个股变化
- `600036.SH`
  - pnl: `-1062.00 -> +162.68`
  - trades: `354 -> 168`
- `601288.SH`
  - pnl: `7328.79 -> 9320.81`
  - trades: `311 -> 119`
- `601398.SH`
  - pnl: `6307.42 -> 7236.92`
  - trades: `265 -> 113`

解释：
- robust 新版本的提升更均衡：
  - `600036.SH` 从负贡献转正
  - `601288.SH` 和 `601398.SH` 也都同步提高
- 这说明 robust 的问题并不是“必须靠更高换手才有收益”，反而更像：
  - 旧 runtime 下存在过度交易 / 无效轮换
  - 当前链路修复后，robust 的真实排序质量释放得更完整了

##### 这组 bank-core3 A/B 的真正含义
- 对 3 股票池来说，当前 official-style 映射并没有真正引入新的组合语义自由度
  - 因为 `hold_thresh=0`
  - `topk=2/n_drop=1` 也与原始任务一致
- 所以这组提升说明的不是“映射本身已经解决问题”，而是：
  - 当前主瓶颈已经从 simple topk/n_drop 参数，转向更底层的执行/排序链一致性
  - 同时也说明：在极小股票池里，想继续逼近官方风格，只靠 rank-based integer TopK 映射空间已经不大

##### 当前新的瓶颈判断
- 5 股票池里：
  - buffer / retained-basket 仍然能显著改变行为
- 3 股票池里：
  - official-style 参数映射已经接近离散极限
  - 下一步如果还想更贴近官方风格，需要考虑：
    - 扩到更大的“自有股票池”
    - 或引入比 rank-cutoff 更细的保留语义（例如 score-margin / hysteresis / weight-based retention）

## 6. 本轮验证

通过：
- `./.venv/bin/pytest backend/tests/unit/backtest/test_topk_dropout_trade_mode.py -q`
- `./.venv/bin/pytest backend/tests/unit/backtest/test_model_topk_dropout_strategy.py -q`
- `./.venv/bin/python -m py_compile backend/app/services/backtest/execution/trade_modes.py backend/tests/unit/backtest/test_topk_dropout_trade_mode.py`

说明：
- 另有 `test_task_backtest_model_driven.py` 在当前环境里因缺少 `nest_asyncio` 导致导入型失败，和本轮改动无关。

## 7. 当前一句话结论

在“股票池继续用我们自己的”这个前提下，当前离官方风格最远的，不再是股票池本身，而是：
- TopkDropout 的组合语义仍然过于高集中 / 高换手
- 以及 validation 信号还没有稳定传导到 portfolio。

而这轮已经把第一步必要杠杆补上了：
- `hold_thresh / buffer` 终于在执行层真正生效。
