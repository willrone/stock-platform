# official Qlib 文档 / 源码机制复盘：为什么官方能做到，而我们还没到

日期：2026-04-15

目标：
- 不再只看我们自己这几轮实验现象
- 回到 official Qlib 文档、benchmark 配置、源码实现本身
- 搞清楚官方 workflow 真正依赖的“机制”是什么
- 再反推 stock-platform 当前为什么仍离官方结果有距离

## 1. 这次实际查看的官方材料

### 官方 benchmark 配置（GitHub raw）
- `examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml`
- `examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha360.yaml`
- `examples/benchmarks/README.md`

### 官方文档（GitHub raw）
- `docs/component/workflow.rst`
- `docs/component/strategy.rst`
- `docs/component/report.rst`

### 本机已安装 qlib 源码（backend venv）
- `qlib/contrib/strategy/signal_strategy.py`
- `qlib/workflow/record_temp.py`
- `qlib/contrib/evaluate.py`

## 2. 官方 workflow 到底怎么做

### A. 官方 benchmark 是“完整 workflow”，不是单次训练 + 一个回测按钮
从 `workflow.rst` 与 `workflow_config_lightgbm_Alpha158/360.yaml` 可以确认：

官方一次 execution 包含：
1. 数据加载与处理
2. 模型训练与预测
3. 评估：
   - `SignalRecord`
   - `SigAnaRecord`
   - `PortAnaRecord`

也就是说，官方并不是：
- 训练出一个模型
- 看一个 accuracy
- 再随手跑一次策略

而是：
- 先保存 `pred.pkl` / `label.pkl`
- 再做信号层分析
- 再做组合层回测与风险分析
- 最终看的是整条工作流的结果

### B. 官方 benchmark 默认根本不看 training accuracy
从 `examples/benchmarks/README.md` 可以确认，官方表格核心列是：
- IC
- ICIR
- Rank IC
- Rank ICIR
- Annualized Return
- Information Ratio
- Max Drawdown

这和我们现在很多直觉最大的不同是：
- 官方不是靠“训练准确率高”来判强弱
- 它本质上是在评估：
  - 排序信号和未来收益的相关性
  - 基于信号生成组合后的真实收益 / 回撤 / IR

### C. 官方 TopkDropout 不是简单“每天卖 n_drop 买 n_drop”这么粗糙
从 `docs/component/strategy.rst` 与 `qlib/contrib/strategy/signal_strategy.py` 可以确认：

TopkDropout 的几个关键语义：
- `Topk`: 持仓股票数
- `Drop`: 每天替换数量
- 典型换手语义：`2 * Drop / K`
- 默认 `hold_thresh = 1`
  - 至少持有 1 个 bar 才允许卖出
- 文档里明确说：
  - 当股票池足够大、`K` 足够大、`Drop` 足够小时，`d ≈ Drop`
  - 也就是官方算法依赖“大横截面 + 小比例轮换”的条件

源码细节进一步说明：
- 它先拿“当前持仓 + 今日候选”合并成 `comb`
- 再基于排序决定真实 sell/buy 名单
- sell 数量不是我们现在这种固定、简单、无状态的近似逻辑
- `hold_thresh` 也直接参与卖出过滤

### D. 官方组合评估看的是 benchmark-relative return，不是单纯策略净值
从 `record_temp.py` 和 `evaluate.py` 可以确认：

`PortAnaRecord` 会产出：
- `report_normal.pkl`
- `positions_normal.pkl`
- `port_analysis_<freq>.pkl`

其中重点分析：
- `excess_return_without_cost = return - bench`
- `excess_return_with_cost = return - bench - cost`
- 再用 `risk_analysis(...)` 算：
  - mean
  - std
  - annualized_return
  - information_ratio
  - max_drawdown

这意味着：
- 官方最后要回答的问题不是“模型有没有赚到钱”
- 而是“相对 benchmark 的超额收益，在考虑成本后是否仍成立”

### E. 官方 Long-Short / Long-Avg 指标是可选增强，不是主结论替代品
从 `SigAnaRecord` 源码可以确认：
- 默认核心还是 IC / RankIC / ICIR / RankICIR
- 如果 `ana_long_short=True`，再补：
  - Long-Short Ann Return
  - Long-Short Ann Sharpe
  - Long-Avg Ann Return
  - Long-Avg Ann Sharpe

所以：
- Long-Short 指标是有用的辅助信号
- 但官方 workflow 的“主闭环”仍然是：
  - 信号相关性
  - 组合超额收益与 IR

## 3. 官方 benchmark 的 LightGBM 参考值到底什么量级

从 `examples/benchmarks/README.md` 可直接看到 LightGBM 行：

### Alpha158 / CSI300
- IC: `0.0448`
- ICIR: `0.3660`
- Rank IC: `0.0469`
- Rank ICIR: `0.3877`
- Annualized Return: `0.0901`
- Information Ratio: `1.0164`
- Max Drawdown: `-0.1038`

### Alpha360 / CSI300
- IC: `0.0400`
- ICIR: `0.3037`
- Rank IC: `0.0499`
- Rank ICIR: `0.4042`
- Annualized Return: `0.0558`
- Information Ratio: `0.7632`
- Max Drawdown: `-0.0659`

这两行很重要，因为它给出了官方 workflow 的真实量级。

## 4. 回头看 stock-platform，差距到底在哪

### 差距 1：我们之前一直在“太小的横截面”里逼近一个“大横截面算法”
这是最根本的机制差距。

官方 TopkDropout 文档默认成立的前提是：
- 股票池足够大
- `K` 足够大
- `Drop` 足够小

而我们最近很多 own-pool 实验其实在：
- 3 股票池
- 5 股票池
- `topk=2`
- `n_drop=1`

这种设定下：
- `Topk/Drop` 的整数化误差极大
- daily basket 很容易发生离散化跳变
- 组合风格会被放大成“少数股票是否进 top2”的问题
- 这和官方 300 股票池里 `topk=50, n_drop=5` 的语义完全不是一个稳定性级别

也就是说：
- 即使名字一样叫 TopkDropout
- 小池里的组合行为也可能已经严重变形

### 差距 2：我们很多轮结果还没真正对 benchmark 做 excess-return 视角校验
官方组合评估真正关心的是：
- `return - bench`
- `return - bench - cost`
- 对应 IR / annualized_return / MDD

而我们很多轮讨论虽然已经开始引入：
- with-cost excess return
- information_ratio

但仍然大量被：
- total_return
- sharpe
- training accuracy
牵着走。

这会导致一个偏差：
- 你可能觉得“这个模型赚钱了”
- 但官方会继续问：
  - 赚钱是因为 beta，还是因为 alpha？
  - 扣 benchmark 和 cost 后还剩多少？

### 差距 3：我们的执行器只是“近似官方”，还不是官方原实现
从源码对照看，当前 stock-platform 的 `model_topk_dropout` 执行层，仍然只是接近 official semantics：
- 也有 topk / n_drop
- 也补了 hold_thresh
- 也能做 ranking-style 轮换

但它仍不是 Qlib 原生那套 `generate_trade_decision` + 订单生成逻辑。

特别是：
- 我们的 trade_mode 仍然是平台自己的执行抽象
- lot size / cash reserve / position cap / validation / rejection 等机制都更偏平台侧
- 这些都会改变“同一套分数”如何落到真实 trade path 上

这意味着：
- 当前结果并不只是“模型能力”和官方有差距
- 还有一部分差距来自执行器实现语义差异

### 差距 4：我们对 signal → portfolio 的桥还没完全按官方方式闭环
官方 workflow 的理想闭环是：
- `pred.pkl`
- `label.pkl`
- `SigAnaRecord`
- `PortAnaRecord`

而我们现在虽然已经补了：
- `signal_quality`
- `portfolio_bridge_summary`
- `official_record_summary`

但很多分析还是：
- 先跑任务
- 再离线拆原因

而不是完全原生地把：
- score signal
- benchmark-relative report
- indicator analysis
统一固化在同一条官方式 record 栈里。

所以现在其实已经不是“完全不会”，而是：
- 机制上已经接近
- 但还没完全原生同构

## 5. 这次研究后，最值得更新的判断

### 判断 A：我们离官方远，不只是“结果差”，而是“运行条件和官方适用条件不一样”
最关键不是一句“还差很多”，而是：
- 官方那套方法成立，强依赖大横截面 + 小比例轮换
- 我们最近很多实验是在极小股票池里试图逼近它
- 这天然会放大离散化和 regime 敏感性

### 判断 B：如果坚持用自有股票池，越小的池子，越不该迷信 rank-cutoff 参数本身
对 3~5 股票池来说：
- `topk=2 / n_drop=1 / hold_thresh=0/1/2`
这些参数的空间其实非常小
- 它不足以稳定模拟官方 `50/5` 的组合语义

所以：
- 5 股票池还可以靠 retained-basket / buffer 再挖
- 3 股票池则已经接近离散极限
- 再往下，需要比 rank-cutoff 更细的机制：
  - score margin
  - hysteresis
  - weight-based retention
  - 或直接更大一点的自有股票池

### 判断 C：官方真正做到的，不是“某个神奇模型”，而是“完整 workflow + 合适市场结构 + 合适组合语义”
这次回看官方 docs/source 后，更该避免的误区是：
- 以为官方强，是因为 LightGBM 某个参数很神

更准确的是：
- 它把
  - 数据处理
  - 标签处理
  - 显式切分
  - 排序策略
  - benchmark-relative 回测
  - 记录系统
  全部放在一条自洽 workflow 里
- 而且默认环境就是适合 TopkDropout 发挥的大横截面市场

## 6. 对 stock-platform 下一步的启示

如果用户继续坚持：
- 股票池先用我们自己的

那最优先的方向应该是：

1. 不要再把“对齐官方”理解成只调 topk/n_drop
2. 先按股票池规模分层：
   - 3 股票池：承认已接近离散极限
   - 5~15 股票池：继续做 retained-basket / buffer / hysteresis
   - 更大自有股票池：才更接近官方 TopkDropout 的适用区间
3. 把评估主轴坚定切到：
   - Rank IC / Rank ICIR
   - excess return with/without cost
   - information_ratio
4. 如果还要继续逼近官方机制，而不是只逼近结果：
   - 需要继续补更原生的 strategy/execution semantics
   - 尤其是 small-pool 下的保留与轮换逻辑

## 7. 当前一句话结论

这次重新看官方文档和源码后，可以更明确地说：

我们离官方远，不只是因为“模型还不够强”，
更因为官方 workflow 的成功依赖：
- 大横截面市场结构
- 小比例轮换的 TopkDropout 语义
- benchmark-relative 的完整 record 栈

而我们当前最容易失真的地方，恰恰就是：
- 小股票池
- 整数化 topk/drop
- 平台自有执行抽象

所以以后如果继续说“向官方对齐”，最应该优先对齐的不是嘴上的 Alpha158/360 名字，
而是：
- 适用条件
- 组合语义
- 评估主轴
- 记录闭环
