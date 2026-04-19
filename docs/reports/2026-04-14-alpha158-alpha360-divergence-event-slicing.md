# alpha158 / alpha360 divergence-date event slicing

日期：2026-04-14

目标：
- 接着 alpha158 / alpha360 formal A/B 与 ranking drift 分析，继续回答：
  - 分歧最集中的月份里，alpha360 到底是在“选到更好的局部机会”，还是在“系统性选错”？
  - 为什么有些月份 daily basket 看起来不差，但 formal-task 最终结果还是更弱？

## 1. 方法说明

这次不是再看整窗总收益，而是切到“分歧月份 × 分歧日”的局部切片。

分析方法：
- 继续使用 `PredictionEngine.predict_return_series(...)`
- 对真实模型逐日生成分数并复用 runtime 排序规则
- 只抽样分歧最集中的月份：
  - 短窗：`2020-02`, `2020-07`
  - 长窗：`2017-03`, `2019-07`, `2020-07`
- 对每个分歧日，记录：
  - alpha158 / alpha360 的 top2 篮子
  - alpha360 相比 alpha158 新加入 / 移除的股票
  - top2 篮子的 forward 1d / 5d 平均收益
  - 差分股票的 forward 1d / 5d 收益

重要限制：
- 这次的 forward 1d / 5d 只是“局部择股有效性”切片
- 它不是完整的已实现交易归因
- 不能直接替代 formal-task 的真实 trade path / cost / 持仓 carry 分析
- 所以它更适合回答：
  - “这天换篮子局部上看值不值”
- 但不等于：
  - “这个月最终组合一定因此更好”

## 2. 月度切片结果

## 2.1 2020-02（短窗）
- divergence_days = `20`
- divergence-day 平均 basket forward return：
  - 1d
    - alpha158 = `0.002980`
    - alpha360 = `0.002761`
  - 5d
    - alpha158 = `0.011261`
    - alpha360 = `0.011566`

局部观察：
- 两边非常接近
- alpha360 在 5d 上只略优一点点
- 典型模式是：
  - alpha360 用 `000651.SH` 替换 alpha158 的 `601288.SH`

例子：`2020-02-03`
- alpha158 top2: `601288.SH`, `600519.SH`
- alpha360 top2: `600519.SH`, `000651.SH`
- 差分：
  - 加入 `000651.SH`
  - 移除 `601288.SH`
- diff stock 5d：
  - `000651.SH = +0.056662`
  - `601288.SH = +0.005935`

但 formal-task 月收益却是：
- alpha158 `2020-02 = -0.005494`
- alpha360 `2020-02 = -0.013641`

这说明：
- 2020-02 里，alpha360 的局部换篮子并不明显差
- 但这些局部正确性没有转成更好的整月 realized result
- 推测 execution path / 持仓延续 / 成本摩擦 在这个月里更重要

## 2.2 2020-07（短窗）
- divergence_days = `22`
- divergence-day 平均 basket forward return：
  - 1d
    - alpha158 = `0.001863`
    - alpha360 = `0.004385`
  - 5d
    - alpha158 = `-0.006153`
    - alpha360 = `-0.003964`

局部观察：
- 这是一个明显更偏向 alpha360 的分歧月
- alpha360 在分歧日局部 basket 表现更好
- 典型模式是：
  - alpha360 更愿意把 `600036.SH` 拉进 top2

例子：`2020-07-01`
- alpha158 top2: `000651.SH`, `601288.SH`
- alpha360 top2: `601288.SH`, `600036.SH`
- 差分：
  - 加入 `600036.SH`
  - 移除 `000651.SH`
- diff stock 5d：
  - `600036.SH = +0.164128`
  - `000651.SH = +0.036231`

例子：`2020-07-02`
- alpha158 用 `600519.SH`
- alpha360 换成 `600036.SH`
- diff stock 5d：
  - `600036.SH = +0.130595`
  - `600519.SH = +0.104922`

formal-task 月收益也支持这个方向：
- alpha158 `2020-07 = +0.003406`
- alpha360 `2020-07 = +0.015461`

这说明：
- 2020-07 是一个相对“支持 alpha360 风格”的 regime
- 它更激进地抬高 `600036.SH`，在这个月局部上确实抓到了更强的后续弹性

## 2.3 2017-03（长窗）
- divergence_days = `23`
- divergence-day 平均 basket forward return：
  - 1d
    - alpha158 = `0.001810`
    - alpha360 = `-0.000755`
  - 5d
    - alpha158 = `0.015194`
    - alpha360 = `0.000812`

局部观察：
- 这是一个明显更偏向 alpha158 的分歧月
- alpha360 常把：
  - `600036.SH`
  - `000001.SZ`
  拉进来
- alpha158 则更常保留：
  - `601288.SH`
  - `600519.SH`

例子：`2017-03-03`
- alpha158 top2: `601288.SH`, `600519.SH`
- alpha360 top2: `601288.SH`, `000001.SZ`
- diff stock 5d：
  - `000001.SZ = 0.000000`
  - `600519.SH = +0.037331`

局部判断：
- 在 2017-03 的分歧日上，alpha158 的换篮子逻辑更优
- 也就是说，alpha360 在这个月更容易把“强势的 defensive winner”换成回报更弱的票

但 formal-task 月收益却是：
- alpha158 `2017-03 = -0.005427`
- alpha360 `2017-03 = -0.000712`

这再次提醒：
- divergence-day forward slices 能说明局部选股对错
- 但未必直接等于 realized month result
- 整月 trade path、成本、持仓 carry 仍然能改写最终盈亏

## 2.4 2019-07（长窗）
- divergence_days = `23`
- divergence-day 平均 basket forward return：
  - 1d
    - alpha158 = `-0.001470`
    - alpha360 = `-0.002286`
  - 5d
    - alpha158 = `-0.011188`
    - alpha360 = `-0.007245`

局部观察：
- 这是一个“方向不够一致”的月份
- 1d 上 alpha158 略好
- 5d 上 alpha360 略好
- 典型分歧是：
  - alpha360 有时把 `600519.SH` 拉进 top2
  - 有时也会把 `000001.SZ` 拉进来，效果参差不齐

例子：`2019-07-02`
- alpha158 top2: `601288.SH`, `600036.SH`
- alpha360 top2: `000001.SZ`, `600519.SH`
- 这天 alpha360 的 diff stocks 5d 都很差：
  - `000001.SZ = -0.041608`
  - `600519.SH = -0.047746`

formal-task 月收益：
- alpha158 `2019-07 = -0.002506`
- alpha360 `2019-07 = -0.017555`

所以：
- 虽然月内平均 5d 切片没完全压死 alpha360
- 但实际 realized result 明显更差
- 这说明这个月的坏结果并不是靠少数一天决定，而更像是多次局部分歧 + execution path 累积出来的

## 3. 把 event slicing 和前面两份报告合起来看

现在三层证据已经可以拼起来：

1. formal-task A/B
- alpha158 组合结果整体更稳
- alpha360 validation 更亮，但 formal-task 没赢

2. ranking drift
- 两边 same_top2 只有约一成
- alpha158 和 alpha360 本质上在长期选择不同篮子

3. event slicing
- alpha360 不是每个分歧月都错
- 它在 `2020-07` 这种 regime 下，局部换篮子是有真实优势的
- 但在 `2017-03` 这种月份里，alpha158 的局部择股明显更稳
- 还有一些月份（如 `2020-02`, `2019-07`）说明：
  - 仅靠 divergence-day forward return 还不足以解释月度 realized PnL

## 4. 当前最有价值的新判断

### 判断 1：alpha360 的 basket 风格并非全局错误，而是 regime-sensitive
- 在 `2020-07` 这类月份：
  - alpha360 的局部分歧选择更像有效 alpha
- 在 `2017-03` 这类月份：
  - alpha158 的保守篮子更有效

所以更准确的话术是：
- alpha360 不是“普遍选错”
- 而是它的 basket style 更吃 regime
- 当前问题是跨 regime 稳定性不够

### 判断 2：event slicing 已经解释了“为什么 alpha360 有时看起来真有料”
之前最容易困惑的点是：
- validation 指标这么亮
- 为什么 formal-task 却没赢？

现在可以更具体地回答：
- 因为 alpha360 确实会在某些月份做出更强的局部选择
- 但这种局部优势没有稳定覆盖更长窗口里的所有 regime
- 一旦遇到不适配月份，拖累股暴露与 execution path 会把收益吃回去

### 判断 3：下一层解释已经不能只靠 forward slices 了
现在已经到一个边界：
- divergence-day forward return 能说明局部选股值不值
- 但如果要解释“为什么 2017-03 局部切片偏向 alpha158，而月度结果却偏向 alpha360”，
- 或者“为什么 2020-02 局部差不大，但 alpha360 整月更差”，
- 那下一步必须进入：
  - 持仓状态
  - 实际成交序列
  - 调仓路径
  - 成本累计
的 replay 级分析

## 5. 建议的下一步

如果继续推进，最值钱的下一刀应该是：

1. holdings-aware replay / window attribution
- 不是只看 daily ranking
- 而是把分歧日附近 3~10 个交易日内：
  - 持仓变化
  - 买卖顺序
  - 成本消耗
  - realized / unrealized PnL 变化
  连起来看

2. 优先切 4 个代表性事件簇
- `2020-07-01 ~ 2020-07-03`
- `2020-02-03 ~ 2020-02-05`
- `2017-03-01 ~ 2017-03-03`
- `2019-07-01 ~ 2019-07-03`

这样就能真正回答：
- alpha360 的局部“对”是怎么在真实组合里被放大或被抵消的
- alpha158 的稳定性，到底来自排序本身，还是来自更温和的执行路径