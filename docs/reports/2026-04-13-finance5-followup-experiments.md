# finance5 后续实验记录：1进1出替换 + cs_rank_norm 复验

日期：2026-04-13

目的：
- 在已确认 `finance5` 是 `future_excess_return_cs` 当前 sweet spot 之后，继续验证两件事：
  1. `finance5` 是否可以通过小范围 1 进 1 出成分替换进一步变好
  2. 在 `finance5` 上叠加 `cs_rank_norm` 是否能把“官方 ranking 风格标签”进一步兑现到正式任务结果

统一正式任务口径：
- API：`/api/v1/tasks`
- strategy_name: `model_topk_dropout`
- 参数：`topk=2`, `n_drop=1`, `benchmark=SH000300`
- 初始资金：`100000`
- 成本：`commission_rate=0.001`, `open_cost=0.0005`, `close_cost=0.0015`, `min_cost=5.0`, `slippage_rate=0.0005`
- 窗口：`2024-full`, `2025-full`, `2024-2025-full`

---

## 1. baseline 回顾

原始 `finance5_csexcess`
- model_id: `89d67073-ecc8-438f-9744-dcc22f7efa4f`
- model_name: `hermes-finance5-csexcess-1776089690`
- training accuracy: `0.3469`

正式任务：
- 2024-full
  - task_id: `716d4ece-054e-4709-86c5-755ceb45e7ed`
  - total_return: `-0.1039%`
  - sharpe: `-0.0302`
  - max_drawdown: `-2.7657%`
  - with_cost IR: `-3.0747`
- 2025-full
  - task_id: `5259547f-b724-40f4-8151-409040b2b11a`
  - total_return: `-0.0574%`
  - sharpe: `-0.0205`
  - max_drawdown: `-2.6844%`
  - with_cost IR: `-3.3783`
- 2024-2025-full
  - task_id: `fdd1d04f-7979-400e-b1f4-88a191540e42`
  - total_return: `-1.5009%`
  - sharpe: `-0.2350`
  - max_drawdown: `-4.2696%`
  - with_cost IR: `-3.3447`

---

## 2. finance5 小范围 1进1出替换实验（替换 601988 槽位）

固定保留：
- `601288.SH`
- `601398.SH`
- `600016.SH`
- `600036.SH`

分别替换入：
- `601166.SH`
- `601328.SH`
- `601939.SH`
- `601818.SH`

### 2.1 训练结果

1) `601166.SH`
- model_id: `f6de9ad1-1271-4ee7-97c9-649a9e88a8e3`
- model_name: `hermes-finance5-swap-601988-to-601166-SH-1776092897`
- accuracy: `0.2082`

2) `601328.SH`
- model_id: `598583ae-4570-4f40-81c1-15774bb5bb01`
- model_name: `hermes-finance5-swap-601988-to-601328-SH-1776092897`
- accuracy: `0.1347`

3) `601939.SH`
- model_id: `c472070b-07c3-45c8-96f0-fd91f7e54d1f`
- model_name: `hermes-finance5-swap-601988-to-601939-SH-1776092897`
- accuracy: `0.2204`

4) `601818.SH`
- model_id: `a262d2fa-6d3a-488e-9b4b-d27a0727b03e`
- model_name: `hermes-finance5-swap-601988-to-601818-SH-1776092897`
- accuracy: `0.1592`

结论（训练页）：
- 四个替换候选的 training accuracy 全部低于原始 `finance5_csexcess` 的 `0.3469`
- 训练侧已经没有出现“明显更优替换”的信号

### 2.2 正式任务结果

#### A. swap → `601166.SH`
- 2024-full
  - task_id: `76a145c8-bff1-441d-a3c9-97caf962a8a1`
  - total_return: `-1.0874%`
  - sharpe: `-0.3375`
  - max_drawdown: `-3.8977%`
  - with_cost IR: `-3.5973`
- 2025-full
  - task_id: `c5ba9835-14bd-4fef-b319-0c9da5a15ce4`
  - total_return: `-2.0738%`
  - sharpe: `-0.7214`
  - max_drawdown: `-3.2282%`
  - with_cost IR: `-4.0035`
- 2024-2025-full
  - task_id: `200dcb00-4104-4968-8f0e-b0ea779ff11d`
  - total_return: `-4.0758%`
  - sharpe: `-0.6613`
  - max_drawdown: `-5.2022%`
  - with_cost IR: `-3.8756`

#### B. swap → `601328.SH`
- 2024-full
  - task_id: `ec91a9f1-a215-40e8-9afd-64a0e5d7fef7`
  - total_return: `-0.8268%`
  - sharpe: `-0.2440`
  - max_drawdown: `-3.2665%`
  - with_cost IR: `-3.3398`
- 2025-full
  - task_id: `849d98db-44f4-4f2e-9f4c-69538f7caed9`
  - total_return: `-1.5366%`
  - sharpe: `-0.5663`
  - max_drawdown: `-3.0417%`
  - with_cost IR: `-4.0406`
- 2024-2025-full
  - task_id: `ba017d29-9cf7-4131-8fd0-14ebd8e557bd`
  - total_return: `-3.4678%`
  - sharpe: `-0.5567`
  - max_drawdown: `-4.5671%`
  - with_cost IR: `-3.7381`

#### C. swap → `601939.SH`
- 2024-full
  - task_id: `b9e4da81-d2a7-466b-b382-09ce4771c26d`
  - total_return: `0.2024%`
  - sharpe: `0.0542`
  - max_drawdown: `-2.8647%`
  - with_cost IR: `-2.7462`
- 2025-full
  - task_id: `6c2efe0f-14d9-4f17-aa99-b2eb5e79c395`
  - total_return: `-1.7299%`
  - sharpe: `-0.5584`
  - max_drawdown: `-3.2213%`
  - with_cost IR: `-3.5992`
- 2024-2025-full
  - task_id: `814ec6e4-4e5e-4b47-835a-5bd0a8556c69`
  - total_return: `-3.1125%`
  - sharpe: `-0.4252`
  - max_drawdown: `-4.7375%`
  - with_cost IR: `-3.1289`

#### D. swap → `601818.SH`
- 2024-full
  - task_id: `ee0824f1-5030-4d76-889b-3cce95921662`
  - total_return: `-3.7375%`
  - sharpe: `-1.1404`
  - max_drawdown: `-5.5511%`
  - with_cost IR: `-4.3754`
- 2025-full
  - task_id: `80492e55-7be1-46dd-8e02-35bde237b7b5`
  - total_return: `-2.8083%`
  - sharpe: `-1.0149`
  - max_drawdown: `-4.5514%`
  - with_cost IR: `-4.4351`
- 2024-2025-full
  - task_id: `e791619e-74bd-401a-9721-dcd420e537d4`
  - total_return: `-7.7460%`
  - sharpe: `-1.2685`
  - max_drawdown: `-8.4602%`
  - with_cost IR: `-4.5065`

### 2.3 替换实验结论

这批结果给出的结论已经很直接：

1. 没有任何一个替换候选在三长窗上整体打赢原始 `finance5`
2. `601939.SH` 是这批替换里最像“局部可行”的候选
   - 它在 `2024-full` 上优于原始 `finance5`
   - 但在 `2025-full` 和 `2024-2025-full` 上又明显回落
   - 所以它更像“窗口局部改善”，不是稳定替代
3. `601818.SH` 明显最差，和之前扩池到 `bank6` / `bank8` 的退化方向一致
4. 因此当前没有证据支持继续围绕 `601988` 槽位做更多简单 1 进 1 出扩展

换句话说：
- `finance5` 的优势目前仍更像一个稳定的组合平衡结果
- 不是“把 601988 随便换成邻近银行股就会更好”的结构

---

## 3. finance5 上叠加 `cs_rank_norm` 复验

实验目的：
- 验证 `future_excess_return_cs + cs_rank_norm` 在 `finance5` 上是否能把训练页的 ranking 风格增强，真正兑现到正式任务结果

模型：
- model_id: `80e96583-212e-497f-ad1d-1ba669e60ca7`
- model_name: `hermes-finance5-csexcess-csranknorm-1776093163`
- training accuracy: `0.6367`

对比基线：
- 原始 `finance5_csexcess` accuracy: `0.3469`
- 也就是说，训练页 accuracy 看起来几乎翻倍

正式任务：
- 2024-full
  - task_id: `22b64339-c1fe-4613-bc03-9ab013651938`
  - total_return: `-1.4650%`
  - sharpe: `-0.4121`
  - max_drawdown: `-4.1143%`
  - with_cost IR: `-3.3660`
- 2025-full
  - task_id: `a580299c-1229-4877-a064-70eae6ac606d`
  - total_return: `-2.1408%`
  - sharpe: `-0.7783`
  - max_drawdown: `-3.5918%`
  - with_cost IR: `-4.2100`
- 2024-2025-full
  - task_id: `653039aa-ce4b-49d9-a11b-a48db9bf3f35`
  - total_return: `-4.7083%`
  - sharpe: `-0.7277`
  - max_drawdown: `-6.2973%`
  - with_cost IR: `-3.7858`

### 3.1 cs_rank_norm 复验结论

这是一个非常关键的反例：

- 训练页 accuracy 从 `0.3469` 提升到 `0.6367`
- 但正式任务三个关键窗口全部明显变差

说明：
1. 在 `finance5` 这个 sweet spot 上，直接给 `future_excess_return_cs` 叠 `cs_rank_norm`，不能改善正式任务 ranking 收益
2. 这再次证明：
   - 训练页 accuracy 只能当早筛指标
   - 不能拿它替代正式任务判断
3. 这个实验也帮助我们收掉一个高概率误判分支：
   - “看起来更像官方 ranking 风格的标签处理”，不等于在当前 stock-platform 链路里就会得到更好的正式任务结果

---

## 4. 本轮后更新的研究判断

到这一步，当前 research branch 已经更明确：

1. `future_excess_return_cs@finance5` 仍然是当前最值得保留的 ranking 研究分支
2. 但它的下一步不该再是：
   - 简单 1 进 1 出换股
   - 直接叠 `cs_rank_norm`
3. 目前这两条低门槛 follow-up 分支都没有带来正式任务改善
4. 因此如果继续做 ranking 研发，更合理的方向应该从“简单池子调整 / 简单归一化”转向更结构性的变化，例如：
   - 更明确的 train/valid/test 切分与独立验证
   - 更强的 signal-quality ↔ portfolio-quality 桥接分析
   - 新标签族，而不是继续在当前标签上做表层处理

一句话结论：
- 这轮 follow-up 实验没有产出比原始 `finance5_csexcess` 更强的新候选
- 原始 `finance5_csexcess` 仍然是这一研究分支里最稳的 sweet spot 基线
- 而 `official` / `robust` 作为当前正式上线主候选与备选的地位，没有被这轮 follow-up 推翻
