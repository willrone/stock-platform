# alpha158 / alpha360 daily topk overlap & ranking drift analysis

日期：2026-04-14

目标：
- 沿着上一轮 alpha158 vs alpha360 formal-task A/B 继续往下拆
- 回答两个更具体的问题：
  - 两个模型每天的 top2 篮子到底有多像？
  - alpha360 为什么 validation 更亮眼，但 formal-task 结果反而输给 alpha158？

## 1. 方法说明

这次分析尽量贴近正式任务执行链，不是手写一个脱离 runtime 的近似排序。

使用路径：
- `PredictionEngine.predict_return_series(...)`
- 也就是 `model_topk_dropout` 正式任务实际会走的预测序列加载路径
- 排序 tie-break 规则复用 live executor 口径：
  - `sorted(scores.items(), key=lambda item: (item[1], item[0]), reverse=True)`

对比模型：
- alpha158
  - `c7cdd57c-2b2d-4386-a8e8-dd1bb2bdc1b9`
- alpha360
  - `cea581a2-695e-41e2-bf3c-7809501360fe`

股票池：
- `600036.SH`
- `601288.SH`
- `600519.SH`
- `000001.SZ`
- `000651.SZ`

窗口：
- `2020-01-01 → 2020-08-01`
- `2017-01-01 → 2020-08-01`

统计项：
- `same_top2_days`
- `same_top1_days`
- `same_full_ranking_days`
- `basket_divergence_days`
- `order_only_divergence_days`
- `monthly_basket_divergence`
- per-stock `top1 / top2 / excluded counts`

说明：
- 和之前 bank-core3 的 3 股票 + top2 场景不同
- 这里是 5 股票 + top2
- 所以这次会自然出现更多：
  - 篮子不同
  - 篮子相同但顺序不同（order-only divergence）

## 2. 结果总览

### 2.1 2020-short
- common_days = `140`
- same_top2_days = `15` (`10.71%`)
- same_top1_days = `25` (`17.86%`)
- same_full_ranking_days = `7` (`5.00%`)
- basket_divergence_days = `125`
- order_only_divergence_days = `8`

解读：
- 在 2020 短窗里，两边每天给出同样 top2 篮子的比例只有一成左右
- 这不是“偶尔分歧”，而是大多数交易日都在选不同篮子
- 同时已经出现 `order_only_divergence_days = 8`
  - 说明这里和 bank-core3 不一样
  - 有一部分日子两边篮子相同，但 leader / runner-up 顺序不同

分歧最集中的月份：
- `2020-07`: `21`
- `2020-02`: `20`
- `2020-03`: `19`
- `2020-06`: `19`
- `2020-05`: `18`
- `2020-04`: `17`

### 2.2 2017-2020-testfull
- common_days = `864`
- same_top2_days = `98` (`11.34%`)
- same_top1_days = `162` (`18.75%`)
- same_full_ranking_days = `7` (`0.81%`)
- basket_divergence_days = `766`
- order_only_divergence_days = `91`

解读：
- 拉到完整 test 窗后，两个模型仍然高度不一致
- same_top2 依旧只有约一成
- 而 same_full_ranking 几乎可以忽略
- 这说明 alpha158 / alpha360 不是“轻微排序偏好不同”，而是长期在做非常不同的 basket allocation

分歧最集中的月份（节选）：
- `2019-07`: `22`
- `2017-03`: `21`
- `2018-07`: `21`
- `2018-08`: `21`
- `2019-01`: `21`
- `2019-08`: `21`
- `2020-07`: `21`

## 3. 风格差异：谁更常被放进 top2？

## 3.1 2020-short

alpha158：
- top1 counts
  - `601288.SH`: `92`
  - `600519.SH`: `19`
  - `000651.SZ`: `19`
- top2 counts
  - `601288.SH`: `113`
  - `600519.SH`: `95`
  - `600036.SH`: `32`
- excluded counts
  - `000001.SZ`: `129`
  - `000651.SZ`: `111`
  - `600036.SH`: `108`

alpha360：
- top1 counts
  - `600036.SH`: `41`
  - `600519.SH`: `39`
  - `601288.SH`: `29`
  - `000651.SZ`: `21`
- top2 counts
  - `601288.SH`: `69`
  - `600036.SH`: `69`
  - `600519.SH`: `61`
- excluded counts
  - `000001.SZ`: `103`
  - `000651.SZ`: `96`
  - `600519.SH`: `79`
  - `600036.SH`: `71`
  - `601288.SH`: `71`

短窗风格结论：
- alpha158 的 leader 明显更集中，强烈偏 `601288.SH`
- alpha158 同时也更稳定地把 `600519.SH` 留在 top2
- alpha360 更分散，也更愿意把 `600036.SH` 顶到 top1 / top2
- 这和 formal-task 归因很一致：
  - alpha360 在 `600036.SH` 上的额外暴露，最后成为主要拖累源

## 3.2 2017-2020-testfull

alpha158：
- top1 counts
  - `601288.SH`: `478`
  - `600519.SH`: `172`
  - `000651.SZ`: `88`
  - `600036.SH`: `72`
- top2 counts
  - `601288.SH`: `603`
  - `600519.SH`: `556`
  - `600036.SH`: `262`
- excluded counts
  - `000001.SZ`: `745`
  - `000651.SZ`: `676`
  - `600036.SH`: `602`

alpha360：
- top1 counts
  - `600036.SH`: `255`
  - `600519.SH`: `208`
  - `000651.SZ`: `187`
  - `601288.SH`: `142`
- top2 counts
  - `600036.SH`: `458`
  - `601288.SH`: `385`
  - `000651.SZ`: `337`
  - `600519.SH`: `312`
- excluded counts
  - `000001.SZ`: `628`
  - `600519.SH`: `552`
  - `000651.SZ`: `527`
  - `601288.SH`: `479`

长窗风格结论：
- alpha158 的长期风格非常清楚：
  - 明显偏 `601288.SH + 600519.SH`
  - 同时长期压低 `600036.SH`
- alpha360 则明显更愿意提升：
  - `600036.SH`
  - `000651.SZ`
- 也就是说，这两个模型在 5 股票池里其实不是“相似模型微调”，而是两种非常不同的 ranking 风格

## 4. 把 ranking drift 和 formal-task 结果连起来看

上一轮 formal A/B 的主要结论是：
- alpha158 在两个正式窗口里都更稳
- alpha360 虽然 validation IC / RankIC 更亮眼，但 formal-task 没赢

现在可以把原因说得更具体：

### 结论 1：alpha360 的问题不是没 alpha，而是 basket allocation 太激进
从 per-stock contribution 看：
- 2020-short 里：
  - alpha360 在 `600519.SH` 上多赚了约 `+21064.57`
  - 但在 `600036.SH` 上多亏了约 `-34571.46`
- 2017-2020 长窗里：
  - alpha360 在 `600519.SH` 上多赚了约 `+52132.37`
  - 但在 `000001.SH/601288.SH/600036.SH` 上合计吃了更大拖累

所以：
- alpha360 不是没有抓到强赢家
- 它的问题是：
  - 对赢家的高弹性
  - 没有转成更稳的组合控制
  - 多个拖累股同时暴露，最终把组合压垮

### 结论 2：alpha158 更像“收敛型篮子”，alpha360 更像“扩散型篮子”
alpha158：
- 领导股票高度集中
- 长期更偏 `601288.SH` 和 `600519.SH`
- 组合结构更像“少数强偏好，少犯错”

alpha360：
- leader 分布明显更分散
- 更频繁把 `600036.SH` / `000651.SH` 拉进 top2
- 组合结构更像“更活跃、更分散、更容易同时吃到正负两边”

### 结论 3：这里不仅有 basket drift，也有 order-only drift
这点和之前 official / robust 的 bank-core3 报告不一样：
- 2020-short: `order_only_divergence_days = 8`
- 2017-2020-testfull: `order_only_divergence_days = 91`

说明：
- 在 5 股票池下，alpha158 / alpha360 不仅经常选不同篮子
- 即使篮子相同，也有不少日子在 leader 次序上不同
- 这会进一步影响：
  - 持仓权重分布
  - 调仓顺序
  - 实际交易成本与回撤路径

## 5. 当前最准确的工程判断

如果只问一句：
- “当前 official preset 里，alpha158 和 alpha360 谁更适合继续当默认参考？”

那么这轮证据仍然支持：
- alpha158 更适合当当前默认参考
- alpha360 更像研究候选，而不是默认升级版

更细一点说：
- alpha360 的 validation 信号质量更亮
- 但 runtime ranking 风格明显更激进、更分散
- 它抓赢家的能力有，但当前控制拖累股暴露的能力还不够
- alpha158 则更像一个虽然不华丽、但组合层更可控的基线

## 6. 下一步最值钱的动作

继续往下，最值钱的不是再机械加更多窗口，而是：

1. 做 divergence-date event slicing
- 优先看：
  - `2020-02`
  - `2020-07`
  - `2019-07`
  - `2017-03`
- 拆这些月份里 basket 分歧最大的日期，看看：
  - 哪只股票被 alpha360 拉进 top2
  - 后续真实收益如何
  - 是否构成主要回撤来源

2. 若后面要把这条能力产品化
- 可以考虑在模型报告 bridge 层补：
  - `ranking_overlap_summary`
  - `basket_divergence_examples`
  - `per_stock_ranking_preference`

这样以后看模型，不用再靠手工离线脚本才能发现“validation 好看但 basket 风格太激进”这种问题。