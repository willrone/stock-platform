# 独立窗口复验：2025 季度切片 + 2026-YTD 正式任务

日期：2026-04-13

目标：
- 在已完成 `finance5` 替换实验与 `cs_rank_norm` 复验后，继续做更独立的窗口复验
- 明确这些候选在更细窗口和真正未纳入此前长窗结论的短窗口里，是否存在被低估的 regime 优势
- 对照当前正式主候选 `official` / 备选 `robust`

---

## 1. 先踩到的真实阻塞：2026-YTD 正式任务最初全部失败

第一次尝试对以下模型跑 `2026-01-01 ~ 2026-02-10` 正式任务时：
- `official-bank-core3`
- `robust-bank-core3`
- `finance5-csexcess`
- `finance5-csexcess-csranknorm`
- `finance5-swap-601939`

全部失败，错误统一为：
- `TaskError: 回测执行失败: 所有股票数据加载失败`

### 根因定位

不是因为数据真的不存在。

我进一步检查后确认：
- `data/qlib_data/features/day/600036_SH.parquet` 等单股票文件存在
- `all_stocks.parquet` 中这些股票的最大日期是 `2026-02-10`
- 但 `2026-01-01 ~ 2026-02-10` 这个窗口只有 `27` 个真实交易日

而 `DataLoader._is_data_valid()` 之前有一个硬门槛：
- `min_rows = 30`

这会导致：
- 即便一个短窗口的数据覆盖已经完整
- 只要真实交易日少于 30
- 就会被错误判定为“无效数据”
- 最终触发“所有股票数据加载失败”

### 已完成修复

文件：
- `backend/app/services/backtest/execution/data_loader.py`

修复思路：
- 不再对短窗口强制要求绝对 `30` 行
- 把最小行数要求改成随窗口长度和覆盖率动态缩放
- 让“短但覆盖完整”的窗口可以合法通过

新增回归测试：
- `backend/tests/unit/backtest/test_data_loader_validity.py`

跑过的回归：
- `tests/unit/backtest/test_data_loader_validity.py`
- `tests/unit/api/test_task_backtest_model_driven.py`
- `tests/unit/api/test_backtest_model_driven.py`

结果：
- 9 个测试通过

这意味着：
- 以后再做短窗口正式任务（特别是 YTD / 月度 / 很短的 out-of-sample 窗）不会再被这个硬编码门槛误杀

---

## 2. 2026-YTD 正式任务复验（修复后）

统一窗口：
- `2026-01-01 ~ 2026-02-10`

### 2.1 bank-core3 主线对照

#### official-bank-core3
- task_id: `1e172dce-15fa-40dd-88ea-65f047b85881`
- total_return: `-0.2114%`
- annualized_return: `-2.1223%`
- sharpe: `-1.6436`
- max_drawdown: `-0.3553%`
- with_cost IR: `-8.4949`
- total_trades: `14`

#### robust-bank-core3
- task_id: `8835de8f-15ec-4d24-9c13-202e05cd8375`
- total_return: `-0.1596%`
- annualized_return: `-1.6061%`
- sharpe: `-1.2344`
- max_drawdown: `-0.3553%`
- with_cost IR: `-8.0544`
- total_trades: `14`

结论：
- 在 2026-YTD 这个很短的最新窗口里，`robust` 略优于 `official`
- 这和之前“季度切片里 robust 更灵活”的结论是一致的
- 但两者在这个超短窗里都不强，且成本后 IR 都非常差

### 2.2 finance5 研究分支对照

#### finance5-csexcess
- task_id: `07e7f9c7-1990-4be3-b308-95989b73f67b`
- total_return: `-0.2647%`
- annualized_return: `-2.6520%`
- sharpe: `-1.6489`
- max_drawdown: `-0.4655%`
- with_cost IR: `-7.1302`
- total_trades: `14`

#### finance5-csexcess-csranknorm
- task_id: `a8547319-1f83-4552-acf6-715426a3b884`
- total_return: `-0.3014%`
- annualized_return: `-3.0137%`
- sharpe: `-2.0458`
- max_drawdown: `-0.4366%`
- with_cost IR: `-8.0211`
- total_trades: `14`

#### finance5-swap-601939
- task_id: `9abd4e10-4876-41cc-9f86-2377350f9e6d`
- total_return: `-0.0663%`
- annualized_return: `-0.6698%`
- sharpe: `-0.5372`
- max_drawdown: `-0.3530%`
- with_cost IR: `-7.6983`
- total_trades: `14`

结论：
- 在 2026-YTD 这个短最新窗口里：
  - `finance5-swap-601939` 是 finance5 分支里最好的
  - `finance5-csexcess-csranknorm` 仍然最差
- 但即便是 `601939` 替换版，也没有形成真正强势的正式任务收益，只能说“短窗跌得更少”

### 2.3 2026-YTD 总结

这个窗口进一步强化了两个已有判断：

1. `robust` 的确更像“短周期更灵活”的备选模型
2. `finance5 + cs_rank_norm` 仍然没有兑现成正式任务优势
3. `601939` 替换版在超短最新窗口里有一定韧性，但不足以推翻其在 `2025-full` 与 `2024-2025-full` 上的退化结论

因此：
- `official` 主候选、`robust` 备选的正式定位不变
- `601939` 替换版最多只能保留为“finance5 分支里值得记住的局部 regime 观察”，不能升级成主研究候选

---

## 3. finance5 分支的 2025 季度切片复验

为了更细地看 regime，我又把这三个 finance5 分支候选跑了 2025Q1~Q4：
- `finance5-csexcess`
- `finance5-csexcess-csranknorm`
- `finance5-swap-601939`

### 3.1 finance5-csexcess
- 2025Q1
  - task_id: `83bc533b-7af3-46dd-a27e-685ff637156d`
  - total_return: `-0.8818%`
  - sharpe: `-1.8006`
- 2025Q2
  - task_id: `56580031-690d-416f-827e-071466af51b9`
  - total_return: `0.5715%`
  - sharpe: `1.1007`
- 2025Q3
  - task_id: `b9b83083-19da-4fb8-807d-250f2a3a60b4`
  - total_return: `-1.3468%`
  - sharpe: `-2.0340`
- 2025Q4
  - task_id: `15eb1f33-7899-4af2-b72b-56f6d95be84e`
  - total_return: `-0.4755%`
  - sharpe: `-0.8807`

### 3.2 finance5-csexcess-csranknorm
- 2025Q1
  - task_id: `5a5a86b2-1fa1-45a8-b2f7-2949fc0b7701`
  - total_return: `-0.2768%`
  - sharpe: `-0.6098`
- 2025Q2
  - task_id: `cff0390f-2ecd-46e1-8b66-bdfc3ca0de00`
  - total_return: `-0.1621%`
  - sharpe: `-0.3108`
- 2025Q3
  - task_id: `aad05970-898a-4652-8a90-21b18d2903be`
  - total_return: `-1.1236%`
  - sharpe: `-1.9190`
- 2025Q4
  - task_id: `69ee6938-36b0-4617-a1d4-38a8673272ee`
  - total_return: `-1.1230%`
  - sharpe: `-1.8812`

### 3.3 finance5-swap-601939
- 2025Q1
  - task_id: `6ee01e3b-bf80-4a02-8688-7263ae90cdee`
  - total_return: `0.1087%`
  - sharpe: `0.1958`
- 2025Q2
  - task_id: `5463c838-74fe-4410-8ed3-10bb636c817b`
  - total_return: `-0.0423%`
  - sharpe: `-0.0887`
- 2025Q3
  - task_id: `4f0e714f-f741-4a7a-aec4-3f489c9e1ada`
  - total_return: `-1.3112%`
  - sharpe: `-2.0820`
- 2025Q4
  - task_id: `985e51b3-93b1-4133-8a48-ff0b5aed135e`
  - total_return: `-1.8212%`
  - sharpe: `-3.4781`

### 3.4 季度切片结论

1. 原始 `finance5-csexcess` 只在 `2025Q2` 有明显正收益
2. `finance5-csexcess-csranknorm` 虽然训练页好看，但四个季度里没有打出任何一个明显优势季度
3. `601939` 替换版只在 `2025Q1` 勉强略正，随后快速退化
4. 整体看，finance5 分支没有出现像 `robust` 那样“2025 各季度持续更灵活”的形态

因此：
- finance5 研究分支目前更像“长窗里有一定 sweet spot 研究价值，但稳定性不足”
- 它还不能升级到和 `official` / `robust` 同等级的正式候选讨论层级

---

## 4. signal-quality ↔ portfolio-quality 桥接观察

我顺手对比了几个模型的 evaluation-report 与正式任务表现：

### 4.1 finance5-csexcess
- training accuracy: `0.3469`
- signal_quality:
  - IC: `-0.1347`
  - Rank IC: `-0.0360`
  - Long-Short Ann Return: `-1.3506`
- 正式任务长窗仍弱，但至少没有训练页和正式任务严重背离到离谱程度

### 4.2 finance5-csexcess-csranknorm
- training accuracy: `0.6367`
- signal_quality:
  - IC: `-0.0123`
  - Rank IC: `-0.0051`
  - Long-Short Ann Return: `-42.7063`
- 正式任务却在 `2024-full / 2025-full / 2024-2025-full / 2026-YTD` 全部退化

这说明：
- `cs_rank_norm` 至少在当前实现/当前标签组合下，确实可能把训练页某些统计“美化”掉
- 但这些统计并没有转化为真正的组合收益优势

### 4.3 official / robust
- 这两个历史主候选的 evaluation-report 中 `signal_quality` 仍是空块（历史模型兼容结果）
- 因此目前 signal ↔ portfolio 的桥接分析，对新 finance5 分支比对更有信息量，对 official/robust 暂时还不完全对称

这也提示了一个后续工程方向：
- 如果要继续做更严肃的官方风格桥接评估，最好补齐对历史主候选模型的 signal_quality 再计算 / 回填能力

---

## 5. 本轮后更新的最终判断

1. `official` 仍是正式主候选
2. `robust` 仍是正式备选，而且在短周期/季度/超短 YTD 窗口里更灵活
3. `finance5-csexcess` 仍是 ranking 研究分支里的 sweet spot 基线
4. `finance5-csexcess-csranknorm` 可以明确降级为“不值得继续挖”的分支
5. `finance5-swap-601939` 只保留为局部 regime 观察，不升级为主研究候选
6. 如果继续做下一轮 ranking 研究，更值得做的不是继续简单换股或继续叠表层归一化，而是：
   - 更严格的 train/valid/test 切分
   - 历史主候选的 signal_quality 回填与桥接分析
   - 新标签族 / 新结构，而不是继续围绕当前 finance5 小修小补
