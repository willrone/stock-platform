# Stacking 堆叠泛化融合方式 - 实现方案设计

## 文档说明

本文档对现有组合策略代码进行分析，设计 Stacking（堆叠泛化）融合方式的实现方案。**仅做方案设计，不修改任何代码**。

---

## 一、现有架构分析

### 1.1 核心组件与数据流

| 组件 | 职责 | 关键接口 |
|------|------|----------|
| **StrategyPortfolio** | 管理子策略列表、权重、信号整合 | `precompute_all_signals()`, `generate_signals()`, `integrator` |
| **SignalIntegrator** | 多策略信号融合，支持 weighted_voting, rank_sum, borda, consensus_topk | `integrate(signals, weights)` |
| **StrategyFactory** | 根据 config 创建策略，解析 `integration_method` | `_create_portfolio_strategy()` |
| **BacktestExecutor** | 回测执行：预计算→提取→主循环 | `_precompute_strategy_signals()`, `_extract_precomputed_signals_to_dict()` |

### 1.2 当前 weighted_voting 流程

1. **预计算阶段**：`_precompute_strategy_signals` 递归对每个子策略预计算，子策略在各自 `data.attrs['_precomputed_signals']` 中缓存 `pd.Series`
2. **提取阶段**：`_extract_precomputed_signals_to_dict` 对 Portfolio 策略：
   - 递归提取所有子策略信号到 `all_sub_signals`
   - 按 `(stock_code, date)` 分组，构造 `TradingSignal` 列表
   - 对每个日期的信号调用 `strategy.integrator.integrate(signals, weights)` 得到最终信号
3. **主循环**：通过 `get_precomputed_signal_fast(stock_code, date)` 直接从字典取最终信号

### 1.3 配置结构

```python
# strategy_config
{
    "strategies": [
        {"name": "rsi", "weight": 0.4, "config": {"rsi_period": 14}},
        {"name": "macd", "weight": 0.6, "config": {...}}
    ],
    "integration_method": "weighted_voting",  # 当前唯一选项
    "trade_mode": "topk", "topk": 5, ...
}
```

---

## 二、Stacking 架构设计

### 2.1 两层结构在 PortfolioStrategy 中的位置

```
                    ┌─────────────────────────────────────────────────────────┐
                    │              StrategyPortfolio                            │
                    │  integration_method: "stacking" | "weighted_voting"       │
                    └─────────────────────────────────────────────────────────┘
                                          │
           ┌──────────────────────────────┼──────────────────────────────┐
           │                              │                              │
           ▼                              ▼                              ▼
    ┌──────────────┐              ┌──────────────┐              ┌──────────────┐
    │ 第一层基学习器 │              │ 第一层基学习器 │              │ 第一层基学习器 │
    │ RSI/MACD/... │              │ RSI/MACD/... │              │ RSI/MACD/... │
    └──────┬───────┘              └──────┬───────┘              └──────┬───────┘
           │                              │                              │
           │ 输出 BUY/SELL/strength        │                             │
           └──────────────────────────────┼─────────────────────────────┘
                                          │
                                          ▼
                    ┌─────────────────────────────────────────────────────────┐
                    │  第二层元模型 (Meta Model)                                │
                    │  输入: [base_1_pred, base_2_pred, ..., base_n_pred]       │
                    │  输出: 最终 BUY/SELL 或 概率/得分                          │
                    └─────────────────────────────────────────────────────────┘
```

**融入方式**：

- **不改变** `StrategyPortfolio` 的对外接口，继续使用 `integration_method` 驱动行为
- **扩展** `SignalIntegrator`：新增 `stacking` 方法，或为 Stacking 创建独立组件 `StackingIntegrator`
- **关键差异**：Stacking 需要「训练阶段」，而 weighted_voting 为无状态规则。因此 Stacking 的整合逻辑不能完全放在 `SignalIntegrator.integrate()` 中，需要前置训练流程

**推荐架构**：

```
StrategyPortfolio
    ├── integration_method == "weighted_voting"  → 使用现有 SignalIntegrator
    └── integration_method == "stacking"        → 使用 StackingFusion（持有元模型）
```

`StackingFusion` 职责：

1. **训练阶段**：在回测开始前，利用历史数据训练元模型
2. **推理阶段**：接收基策略信号，输出融合后的 `TradingSignal` 列表

### 2.2 与 BacktestExecutor 的协作流程

```
BacktestExecutor.run_backtest()
    │
    ├── 1. 创建 strategy（含 integration_method=stacking）
    │
    ├── 2. 加载 stock_data（全时间段）
    │
    ├── 3. [新增] 若 integration_method == "stacking"：
    │        StackingTrainer.train(
    │            stock_data, strategies, config.stacking_train_ratio
    │        ) → meta_model, meta_config
    │        将 meta_model 挂载到 strategy 或 integrator
    │
    ├── 4. _precompute_strategy_signals() 递归预计算子策略
    │        （与现有逻辑一致，每只股票得到各子策略的 Series）
    │
    ├── 5. [修改] _extract_precomputed_signals_to_dict()：
    │        若 integration_method == "stacking"：
    │            - 不再调用 integrator.integrate()
    │            - 调用 StackingFusion.predict(sub_signals_matrix) → 最终信号
    │        否则保持现有 weighted_voting 逻辑
    │
    └── 6. 主循环（不变）
```

---

## 三、元模型选择

### 3.1 推荐模型及原因

| 模型 | 推荐程度 | 理由 |
|------|----------|------|
| **逻辑回归 (Logistic Regression)** | ⭐⭐⭐⭐⭐ | 结构简单、可解释强、不易过拟合、训练快，适合作为第一选择 |
| **LightGBM** | ⭐⭐⭐⭐ | 非线性能力强、对缺失值友好、训练快，适合特征较多或非线性明显时 |
| **简单线性回归** | ⭐⭐⭐ | 若将标签设为连续收益，可用线性回归；实现简单，但表达能力有限 |
| **XGBoost** | ⭐⭐⭐ | 与 LightGBM 类似，通常略慢，可选 |
| **神经网络** | ⭐⭐ | 样本量小时易过拟合，一般不推荐 |
| **随机森林** | ⭐⭐ | 容易过拟合，不如 LightGBM 可控 |

### 3.2 首选：逻辑回归

- **可解释性**：系数可直接反映各基策略的贡献
- **稳定性**：L2 正则可有效抑制过拟合
- **效率**：训练和预测都很快，适合回测
- **适配性**：基策略数量通常 3–10 个，特征维度不高，逻辑回归足够

### 3.3 备选：LightGBM

- 基策略多（>10）或希望捕捉非线性组合时使用
- 建议：`max_depth=2~3`、`num_leaves=8~16`、`min_data_in_leaf>=50`，控制过拟合

### 3.4 元模型配置建议

```python
# 建议在 strategy_config 中支持
"stacking_config": {
    "meta_model": "logistic",  # logistic | lightgbm | linear
    "meta_params": {},         # 如 lightgbm 的 n_estimators, max_depth 等
    "label_type": "forward_return",  # forward_return | binary
    "forward_days": 5,
    "use_strength": True,     # 是否将 strength 作为特征
}
```

---

## 四、训练数据构造

### 4.1 数据来源

- **股票数据**：`stock_data: Dict[str, pd.DataFrame]`，与现有回测一致
- **基策略预测**：各子策略的 `precompute_all_signals()` 输出
- **标签**：需从价格序列计算，不能使用未来数据

### 4.2 特征矩阵构造

对每个 `(stock_code, date)` 样本：

| 特征类型 | 说明 | 维度 |
|----------|------|------|
| 基策略信号编码 | BUY=1, SELL=-1, HOLD=0（或 NaN） | n_strategies |
| 基策略强度 | `signal.strength`（0–1） | n_strategies |
| 可选：策略一致性 | 同向信号占比 | 1 |
| 可选：原始权重 | `weights[strategy_name]` | n_strategies |

**推荐基础特征**：`[s1_signal, s2_signal, ..., sn_signal, s1_strength, s2_strength, ..., sn_strength]`，共 `2 * n_strategies` 维。

### 4.3 标签构造

| 标签类型 | 计算方式 | 用途 |
|----------|----------|------|
| **forward_return** | `close[t+forward_days] / close[t] - 1` | 回归，预测收益 |
| **binary** | return > 0 → 1, else → 0 | 二分类 BUY/不买 |
| **ternary** | 按收益分档 → BUY/HOLD/SELL | 三分类 |

推荐使用 **forward_return**，元模型输出为连续值，再按阈值（如 >0）或 TopK 转为 BUY。

### 4.4 时间序列与数据泄露

- **严格避免未来信息**：标签只能用 `close[t+1:t+forward_days]`，不能使用 `close[t]` 之后任何时刻的信息
- **训练/验证划分**：
  - **方案 A（固定划分）**：前 70% 日期训练，后 30% 测试
  - **方案 B（滚动窗口）**：每日用过去 N 天训练，预测下一天（更贴近实盘，计算量大）
- **初始建议**：先实现方案 A，通过 `stacking_train_ratio`（如 0.7）控制

### 4.5 样本构造伪代码（逻辑）

```
for each stock_code in stock_data:
    sub_signals = {s.name: s.precompute_all_signals(data) for s in strategies}
    for date in train_date_range:
        X_row = []
        for s in strategies:
            sig = sub_signals[s.name].loc[date]
            X_row += [encode(sig), strength(sig)]  # 信号编码 + 强度
        y = forward_return(data, date, forward_days)
        samples.append((X_row, y, stock_code, date))
```

---

## 五、需要修改的文件清单与改动概要

### 5.1 后端

| 文件 | 改动类型 | 改动概要 |
|------|----------|----------|
| `backend/app/services/backtest/utils/signal_integrator.py` | 扩展 | 在 `SUPPORTED_METHODS` 中加入 `"stacking"`；或对 stacking 做分支，委托给 `StackingFusion`，自身不实现具体逻辑 |
| **新建** `backend/app/services/backtest/utils/stacking_fusion.py` | 新增 | 定义 `StackingFusion`：`train(stock_data, strategies, config)`、`predict(features_matrix)`；内部封装元模型训练与预测 |
| `backend/app/services/backtest/core/strategy_portfolio.py` | 修改 | 当 `integration_method=="stacking"` 时，创建/持有 `StackingFusion` 而非普通 `SignalIntegrator`；或在 `SignalIntegrator` 内根据 method 选择实现 |
| `backend/app/services/backtest/strategies/strategy_factory.py` | 修改 | 解析 `integration_method`、`stacking_config`，传给 `StrategyPortfolio`；若为 stacking，可延迟训练（由 Executor 触发） |
| `backend/app/services/backtest/execution/backtest_executor.py` | 修改 | 在 `_precompute_strategy_signals` 之后、`_extract_precomputed_signals_to_dict` 之前：若为 stacking，调用 `StackingTrainer.train()`，将训练好的元模型挂到 strategy；修改 `_extract_precomputed_signals_to_dict` 中 Portfolio 分支，对 stacking 使用 `StackingFusion.predict()` 生成最终信号 |
| `backend/app/services/backtest/optimization/portfolio_hyperparameter_optimizer.py` | 修改 | 若后续支持 stacking 超参优化，在 `suggest_categorical("integration_method", ...)` 中加入 `"stacking"`，并增加 stacking 相关超参 |
| `backend/app/services/backtest/optimization/strategy_hyperparameter_optimizer.py` | 修改 | 与 portfolio 类似，如需优化 integration_method，需支持 stacking |

### 5.2 模型层

| 文件 | 改动类型 | 改动概要 |
|------|----------|----------|
| `backend/app/models/strategy_config_models.py` | 可选 | 若将 stacking 配置持久化，可在 `parameters` JSON 中约定 `stacking_config` 结构，无需改表结构 |

### 5.3 API 层

| 文件 | 改动类型 | 改动概要 |
|------|----------|----------|
| `backend/app/api/v1/backtest.py` | 修改 | 接受并向下传递 `integration_method: "stacking"` 及 `stacking_config`，确保与现有 config 兼容 |

### 5.4 前端

| 文件 | 改动类型 | 改动概要 |
|------|----------|----------|
| `frontend/src/components/backtest/PortfolioStrategyConfig.tsx` | 修改 | 在信号整合方法的 `Select` 中新增 `<MenuItem value="stacking">Stacking 堆叠泛化</MenuItem>`；可选：当选择 stacking 时显示 `stacking_config` 配置区（元模型类型、训练比例等） |
| `frontend/src/app/tasks/create/page.tsx` | 修改 | 保存/提交时包含 `integration_method: "stacking"` 及可选的 `stacking_config` |
| `frontend/src/app/signals/page.tsx` | 修改 | 同上，确保 signals 任务能正确传递 stacking 配置 |
| `frontend/src/components/optimization/CreateOptimizationTaskForm.tsx` | 修改 | 若优化任务支持组合策略，在 `integration_method.choices` 中加入 `'stacking'` |

---

## 六、前端配置：用户如何选择 Stacking

### 6.1 基础选择

在「信号整合方法」下拉框中增加一项：

```
加权投票 (weighted_voting)
Stacking 堆叠泛化 (stacking)
```

用户选择 `stacking` 即启用 Stacking 融合。

### 6.2 可选高级配置

当选择 `stacking` 时，可展开「Stacking 配置」面板：

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| 元模型类型 | 下拉 | logistic | logistic / lightgbm / linear |
| 训练数据比例 | 滑块 0.5–0.9 | 0.7 | 用于训练元模型的历史数据比例 |
| 标签 lookahead 天数 | 数字 1–20 | 5 | 用于计算未来收益的天数 |
| 是否使用信号强度 | 开关 | true | 是否将 strength 作为特征 |

### 6.3 配置数据结构示例

```typescript
// 前端提交的 strategy_config
{
  strategies: [...],
  integration_method: "stacking",
  stacking_config: {
    meta_model: "logistic",
    train_ratio: 0.7,
    forward_days: 5,
    use_strength: true,
  }
}
```

### 6.4 交互提示

- 选择 stacking 时，可提示：「需要一定历史数据用于训练元模型，建议回测区间不少于 1 年」
- 训练完成后，可在回测结果中展示「元模型训练样本数、验证指标（如 AUC）」等简要信息

---

## 七、潜在风险与注意事项

### 7.1 过拟合

- **表现**：训练期表现好，测试期明显变差
- **对策**：
  - 使用逻辑回归 + L2 正则
  - LightGBM 限制 `max_depth`、`min_data_in_leaf`
  - 训练样本要求：建议至少 5000+（股票×日期）
  - 可考虑使用时间序列交叉验证（如 Purged K-Fold）

### 7.2 数据泄露

- **表现**：训练时使用了未来信息
- **对策**：
  - 标签仅用 `close[t+1:t+k]` 计算
  - 特征只用 `t` 及之前的数据
  - 元模型训练时严格按时间划分，不混用测试区间数据

### 7.3 训练时间

- **表现**：Stacking 比 weighted_voting 慢，因为需先跑一遍基策略再训练元模型
- **对策**：
  - 复用已有预计算：基策略信号与 weighted_voting 共用
  - 元模型训练本身较快（逻辑回归通常 <1 秒）
  - 可异步或后台执行训练，并提供进度反馈

### 7.4 冷启动与数据不足

- **表现**：回测区间过短，训练样本不足
- **对策**：
  - 设置最小训练天数（如 252 个交易日）
  - 若不足，自动回退到 `weighted_voting` 并提示用户
  - 或拒绝启动 stacking，明确报错

### 7.5 子策略变更

- **表现**：修改子策略组合或参数后，原元模型不再适用
- **对策**：每次 `strategy_config` 变化都重新训练元模型，不缓存跨配置的模型（或按 config 哈希缓存）

### 7.6 多股票扩展

- **表现**：不同股票特性不同，单一元模型可能不适用
- **对策**：
  - 第一版：全市场共用一个元模型（简单可解释）
  - 后续可探索：按行业/市值分组建模，或引入股票 ID 作为特征（需防过拟合）

---

## 八、与 weighted_voting 的兼容性

### 8.1 共存方式

- `integration_method` 为唯一开关：`"weighted_voting"` 与 `"stacking"` 互斥
- `StrategyPortfolio` 根据 `integration_method` 选择整合实现：
  - `weighted_voting`（及 rank_sum 等）：继续使用现有 `SignalIntegrator`
  - `stacking`：使用 `StackingFusion`

### 8.2 权重字段的语义

- **weighted_voting**：`weights` 直接参与加权投票
- **stacking**：`weights` 可（1）作为可选特征输入元模型，或（2）暂不使用，由元模型完全学习；建议第一版保留传入，作为可选特征，便于后续扩展

### 8.3 配置兼容性

- 同一 `strategy_config` 结构，仅 `integration_method` 不同
- 未传 `stacking_config` 时使用默认值
- API、前端、优化器只需在原有分支上增加对 `stacking` 的处理，不影响现有 weighted_voting 逻辑

### 8.4 优化器扩展

- `portfolio_hyperparameter_optimizer` 的 `integration_method` 可选值可扩展为 `["weighted_voting", "stacking"]`
- 选择 stacking 时，可额外优化 `meta_model`、`train_ratio`、`forward_days` 等，与现有权重优化并存

---

## 九、实施优先级建议

1. **Phase 1**：实现 `StackingFusion` 与训练流程，固定逻辑回归 + 前 70% 训练
2. **Phase 2**：接入 `BacktestExecutor`，完成 `_extract_precomputed_signals_to_dict` 的 stacking 分支
3. **Phase 3**：前端增加 stacking 选项及基础配置
4. **Phase 4**：支持 LightGBM、可配置 `train_ratio`、`forward_days`
5. **Phase 5**：超参优化支持 stacking，以及可能的滚动训练

---

## 十、总结

| 维度 | 设计要点 |
|------|----------|
| **架构** | Stacking 作为 `integration_method` 的新取值，通过 `StackingFusion` 实现，与 `SignalIntegrator` 并列 |
| **元模型** | 首选逻辑回归，备选 LightGBM |
| **训练数据** | 基策略信号+强度作特征，前 N 日未来收益作标签，严格按时间划分防泄露 |
| **兼容性** | 与 weighted_voting 共用配置结构，通过 `integration_method` 切换，互不干扰 |
| **风险** | 重点关注过拟合、泄露与样本量，通过正则、时间划分和最小样本要求控制 |
