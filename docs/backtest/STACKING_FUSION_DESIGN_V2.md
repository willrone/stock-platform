# Stacking 堆叠泛化融合方式 - V2 实现方案设计

## 文档说明

本文档在 V1 方案基础上，**重新设计 Stacking 堆叠泛化方案**，核心变更是：**将 Stacking 元模型训练从回测执行中抽离，放到模型管理模块，作为新的训练类型 `stacking_ensemble`**。

**仅做方案设计，不修改任何代码**。

---

## 一、V2 与 V1 的核心差异

| 维度 | V1 方案 | V2 方案 |
|------|---------|---------|
| **训练时机** | 回测执行时，在 `_precompute_strategy_signals` 之后、`_extract_precomputed_signals_to_dict` 之前 | 独立训练任务，在模型管理页面发起，与回测解耦 |
| **训练入口** | 回测 API 传入 `integration_method: "stacking"` 时自动触发 | 模型管理 API `/models/train` 选择 `model_type: "stacking_ensemble"` |
| **模型存储** | 训练后挂载到 strategy/StackingFusion 实例，不持久化 | 存入 `model_info` 表，与 lightgbm/xgboost 等同等对待 |
| **回测使用** | 每次回测重新训练 | 回测时通过 `stacking_model_id` 加载已训练模型 |
| **可复用性** | 同一 stacking 模型不能跨回测任务使用 | 一次训练、多次回测，支持版本管理 |

---

## 二、架构设计：训练与回测解耦

### 2.1 整体数据流

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           Stacking V2 整体数据流                                  │
└─────────────────────────────────────────────────────────────────────────────────┘

  【训练路径】独立于回测
  ────────────────────────────────────────────────────────────────────────────────
  用户 → 模型管理页面 → POST /models/train
         {
           model_type: "stacking_ensemble",
           strategy_config: { strategies, ... },   // 基策略配置
           stacking_config: { meta_model, forward_days, ... },
           stock_codes, start_date, end_date
         }
         │
         ▼
  StackingEnsembleTrainer.train()
         │
         ├── 1. 加载股票数据
         ├── 2. 用 strategy_config 创建基策略，预计算所有基策略信号
         ├── 3. 构造特征矩阵 (信号编码 + strength) + 标签 (forward_return)
         ├── 4. 按股票 70/30 划分 train/val
         ├── 5. 训练元模型 (LogisticRegression / LightGBM)
         ├── 6. 保存到 file_path，写入 model_info 表
         │
         ▼
  model_info 表新增一条记录 (model_type=stacking_ensemble)


  【回测路径】使用已训练模型
  ────────────────────────────────────────────────────────────────────────────────
  用户 → 任务创建页面 → 选择「组合策略」+「Stacking 融合」+ 选择已训练 stacking 模型
         │
         ▼
  POST /tasks (或 POST /backtest)
  backtest_config: {
    strategy_name: "portfolio",
    strategy_config: {
      strategies: [...],
      integration_method: "stacking",
      stacking_model_id: "uuid-xxx"   // ← 关键：指定已训练模型
    },
    ...
  }
         │
         ▼
  BacktestExecutor.run_backtest()
         │
         ├── 1. 创建 StrategyPortfolio (integration_method=stacking)
         ├── 2. 若 stacking_model_id 存在：
         │      └── 从 model_info 加载元模型 → 挂到 strategy.integrator (或 StackingFusion)
         ├── 3. 若 stacking_model_id 不存在：
         │      └── Fallback: 回退到 weighted_voting 并警告
         ├── 4. _precompute_strategy_signals() 预计算基策略
         ├── 5. _extract_precomputed_signals_to_dict() 中：
         │      若 stacking → 用元模型 predict(sub_signals_matrix) 融合
         └── 6. 主循环执行
```

### 2.2 训练与回测职责边界

| 模块 | 职责 |
|------|------|
| **模型管理** | 接收 stacking_ensemble 训练请求；加载数据、运行基策略、构造样本、训练元模型、保存到 model_info |
| **回测执行** | 接收 `stacking_model_id`；加载元模型；在信号提取阶段调用元模型做融合 |
| **策略组合** | 持有 StackingFusion（或扩展的 integrator），在 `integration_method=stacking` 时使用 |

---

## 三、模型管理集成：复用 model_info 与训练流程

### 3.1 model_info 表兼容性

现有 `model_info` 表结构已足够，无需新增列：

| 字段 | stacking_ensemble 用途 |
|------|------------------------|
| model_id | 唯一标识 |
| model_name | 用户命名，如 "RSI+MACD Stacking v1" |
| model_type | 固定为 `"stacking_ensemble"` |
| file_path | 元模型序列化文件路径（joblib/pickle） |
| training_data_start / training_data_end | 训练数据时间范围 |
| hyperparameters | 存放 `strategy_config`（基策略配置）、`stacking_config`（元模型类型、forward_days 等） |
| performance_metrics | 训练/验证指标：train_mse, val_mse, train_auc, val_auc, strategy_weights 等 |
| evaluation_report | 可选：策略权重、特征重要性、过拟合检测结果 |
| status | training | ready | failed |

### 3.2 训练请求 Schema 扩展

在 `ModelTrainingRequest` 中，对 `model_type == "stacking_ensemble"` 时，需要额外字段：

```python
# 扩展 ModelTrainingRequest（或新建 StackingEnsembleTrainingRequest）
class ModelTrainingRequest(BaseModel):
    model_name: str
    model_type: str  # 新增支持 "stacking_ensemble"
    stock_codes: List[str]
    start_date: str
    end_date: str
    hyperparameters: Dict[str, Any] = {}

    # stacking_ensemble 专用（当 model_type=stacking_ensemble 时必需）
    strategy_config: Optional[Dict[str, Any]] = None  # 基策略配置，同回测
    stacking_config: Optional[Dict[str, Any]] = None  # 元模型配置
```

`strategy_config` 结构（与回测一致）：

```json
{
  "strategies": [
    {"name": "rsi", "weight": 0.4, "config": {"rsi_period": 14}},
    {"name": "macd", "weight": 0.6, "config": {...}}
  ],
  "integration_method": "stacking"
}
```

`stacking_config` 结构：

```json
{
  "meta_model": "logistic",
  "forward_days": 5,
  "use_strength": true,
  "train_val_split_by_stock_ratio": 0.7
}
```

### 3.3 训练流程与现有模型训练的统一

- 复用 `/models/train` 接口：当 `model_type == "stacking_ensemble"` 时，路由到 `StackingEnsembleTrainer`
- 复用 ModelInfo 创建、WebSocket 进度通知、线程池执行
- 复用 `ModelInfoRepository.save_model_info`、`get_model_info`
- 训练完成后 `status="ready"`，与 lightgbm 等一致

---

## 四、训练流程详细设计

### 4.1 训练数据构造

#### 4.1.1 特征矩阵

对每个 `(stock_code, date)` 样本，特征向量：

| 特征类型 | 说明 | 维度 |
|----------|------|------|
| 信号编码 | BUY=1, SELL=-1, HOLD=0（或 NaN 填充为 0） | n_strategies |
| 信号强度 | `signal.strength`（0–1），无则 0.5 | n_strategies |

**推荐基础特征**：`[s1_signal, s2_signal, ..., sn_signal, s1_strength, s2_strength, ..., sn_strength]`，共 `2 * n_strategies` 维。

可选扩展：策略权重作为特征、一致性占比等。

#### 4.1.2 标签构造

- **标签类型**：`forward_return`
- **计算方式**：`close[t+forward_days] / close[t] - 1`（严格使用未来数据，防止泄露）
- **回归任务**：元模型输出连续值，再按阈值（如 >0）或 TopK 转为 BUY
- **二分类任务**（可选）：return > 0 → 1, else → 0，用于 AUC 等指标

### 4.2 股票维度 70/30 划分（V2 新增重点）

**与 V1 的「按时间 70/30」不同**：V2 建议**按股票划分**，避免同一股票在训练集和验证集同时出现，减少「同一股票不同时期」的泄露。

| 划分方式 | 说明 | 优点 | 缺点 |
|----------|------|------|------|
| 按时间 | 前 70% 日期 train，后 30% val | 简单，符合时间序列 | 同一股票可能两边都有 |
| **按股票** | 70% 股票 train，30% 股票 val | 泛化到新股票更好，无股票级泄露 | 需保证每只股票样本量充足 |

**实现逻辑**：

1. 所有股票代码 shuffle 后按 70/30 划分 → `train_stocks`, `val_stocks`
2. 对 `(stock_code, date)` 样本：若 `stock_code in train_stocks` → 训练集，否则验证集
3. 若某股票样本过少（如 <10 条），可归入训练集或单独过滤

### 4.3 元模型训练

| 选项 | 推荐度 | 说明 |
|------|--------|------|
| **逻辑回归** | ⭐⭐⭐⭐⭐ | 首选：可解释、稳定、训练快 |
| **LightGBM** | ⭐⭐⭐⭐ | 备选：非线性、max_depth=2~3 控过拟合 |

**逻辑回归配置建议**：

- `sklearn.linear_model.LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs')`
- 系数可输出为「策略权重」，便于可解释性

**LightGBM 配置建议**（备选）：

- `max_depth=2`, `num_leaves=8`, `min_data_in_leaf=50`

### 4.4 训练结果保存与指标

**保存内容**（序列化到 file_path）：

- 元模型对象（fit 后的 LogisticRegression 或 LGBMRegressor）
- 特征名列表（用于预测时对齐顺序）
- `stacking_config`、`strategy_config` 快照（用于回测时校验兼容性）

**performance_metrics** 写入 model_info：

```json
{
  "train_mse": 0.012,
  "val_mse": 0.015,
  "train_auc": 0.62,
  "val_auc": 0.58,
  "strategy_weights": {"rsi": 0.35, "macd": 0.65},
  "n_train_samples": 15230,
  "n_val_samples": 6530
}
```

**evaluation_report**（可选）：策略权重、学习曲线、过拟合检测摘要。

---

## 五、回测集成设计

### 5.1 回测配置中的 stacking_model_id

在 `strategy_config` 中新增可选字段：

```json
{
  "strategies": [...],
  "integration_method": "stacking",
  "stacking_model_id": "uuid-xxx"
}
```

- 当 `integration_method == "stacking"` 且 `stacking_model_id` 存在时：加载该模型
- 当 `integration_method == "stacking"` 且 `stacking_model_id` 缺失：Fallback

### 5.2 BacktestExecutor 中的模型加载与融合

**流程**：

1. 在 `run_backtest` 早期（创建 strategy 之后）：
   - 若 `strategy_config.get("integration_method") == "stacking"` 且 `stacking_model_id` 存在：
   - 调用 `ModelInfoRepository.get_model_info(stacking_model_id)`
   - 校验 `model_info.model_type == "stacking_ensemble"`
   - 加载 `file_path` 中的元模型及特征名
   - 校验 `strategy_config.strategies` 与模型内保存的 `strategy_config` 兼容（策略名、数量一致）
   - 将元模型注入到 `strategy.integrator` 或新建 `StackingFusion` 并挂到 strategy

2. 在 `_extract_precomputed_signals_to_dict` 的 Portfolio 分支：
   - 若 `integration_method == "stacking"` 且 strategy 已挂载元模型：
   - 对每个 date 的 `signals_by_date[date]`，构造特征矩阵（与训练时一致）
   - 调用 `meta_model.predict(X)` 得到分数/概率
   - 按阈值或 TopK 转为 BUY/SELL 信号
   - 否则走现有 `integrator.integrate()`（weighted_voting）

### 5.3 Fallback 策略

当 `integration_method == "stacking"` 但 `stacking_model_id` 缺失或模型加载失败时：

| 场景 | Fallback 行为 |
|------|---------------|
| stacking_model_id 未传 | 回退到 `weighted_voting`，记录警告日志，可选在 result 中标记 `stacking_fallback: true` |
| 模型不存在 | 同上 |
| 策略配置不兼容 | 报错并终止回测，提示用户重新训练或选择匹配的模型 |

---

## 六、需要修改的文件清单与改动概要

### 6.1 后端

| 文件 | 改动类型 | 改动概要 |
|------|----------|----------|
| **新建** `backend/app/services/stacking/stacking_ensemble_trainer.py` | 新增 | `StackingEnsembleTrainer`：接收 strategy_config、stacking_config、stock_codes、日期；加载数据、预计算基策略、构造样本、训练元模型、保存 |
| **新建** `backend/app/services/stacking/stacking_fusion.py` | 新增 | `StackingFusion`：持有元模型，提供 `predict(features_matrix) -> signals` |
| **新建** `backend/app/services/stacking/__init__.py` | 新增 | 导出上述模块 |
| `backend/app/services/backtest/utils/signal_integrator.py` | 扩展 | `SUPPORTED_METHODS` 加入 `"stacking"`；stacking 时委托给 StackingFusion |
| `backend/app/services/backtest/core/strategy_portfolio.py` | 修改 | 当 `integration_method=="stacking"` 时，创建 `StackingFusion`（可先为空，由 Executor 注入元模型） |
| `backend/app/services/backtest/strategies/strategy_factory.py` | 修改 | 解析 `stacking_model_id`（可选），传入 StrategyPortfolio |
| `backend/app/services/backtest/execution/backtest_executor.py` | 修改 | 1) 若 stacking 且 stacking_model_id 存在，加载元模型并注入 strategy；2) `_extract_precomputed_signals_to_dict` 的 Portfolio 分支增加 stacking 分支，使用 StackingFusion.predict |
| `backend/app/api/v1/models.py` | 修改 | `create_training_task`：`valid_model_types` 加入 `"stacking_ensemble"`；当 `model_type=="stacking_ensemble"` 时，校验 `strategy_config`/`stacking_config`，调用 `StackingEnsembleTrainer` |
| `backend/app/api/v1/schemas.py` | 修改 | `ModelTrainingRequest` 增加 `strategy_config`、`stacking_config` 可选字段 |
| `backend/app/api/v1/backtest.py` | 修改 | 接受并向下传递 `strategy_config.stacking_model_id` |
| `backend/app/api/v1/schemas.py` (BacktestRequest) | 无需改 | strategy_config 为 Dict，已支持任意字段 |

### 6.2 数据层

| 文件 | 改动类型 | 改动概要 |
|------|----------|----------|
| `backend/app/models/task_models.py` | 无需改 | model_info 表已满足 |
| `backend/app/repositories/task_repository.py` | 无需改 | 已有 `get_model_info`、`save_model_info` |

### 6.3 前端

| 文件 | 改动类型 | 改动概要 |
|------|----------|----------|
| `frontend/src/app/models/page.tsx` | 修改 | 模型类型下拉增加 `stacking_ensemble`；选中时展示 `strategy_config`、`stacking_config` 配置区 |
| `frontend/src/components/models/CreateModelForm.tsx` | 修改 | 当 `model_type==stacking_ensemble` 时，渲染策略组合选择器、stacking_config 表单 |
| `frontend/src/app/tasks/create/page.tsx` | 修改 | 回测任务：当组合策略且 `integration_method==stacking` 时，增加「选择 Stacking 模型」下拉（调用 GET /models?model_type=stacking_ensemble） |
| `frontend/src/components/backtest/PortfolioStrategyConfig.tsx` | 修改 | 信号整合方法增加 `stacking`；选中 stacking 时显示「已训练 Stacking 模型」选择器，并传 `stacking_model_id` |
| `frontend/src/services/dataService.ts` | 修改 | `getModels` 支持 `model_type` 过滤参数 |
| `frontend/src/components/models/ModelListTable.tsx` | 修改 | 对 `stacking_ensemble` 类型展示策略权重、训练/验证指标 |
| `frontend/src/components/models/TrainingReportModal.tsx` | 修改 | stacking 模型详情：策略权重、train/val 对比 |

---

## 七、前端设计

### 7.1 模型管理页面：新增 stacking 训练入口

**入口**：在「创建模型」表单中，模型类型增加：

```
- LightGBM (推荐)
- XGBoost
- 线性回归
- Transformer
- Stacking 组合策略融合  ← 新增
```

**当选择 Stacking 组合策略融合时**：

1. **策略配置**：复用或内嵌 `PortfolioStrategyConfig` 组件，配置基策略列表、权重
2. **Stacking 配置**：
   - 元模型类型：logistic / lightgbm
   - forward_days：1–20，默认 5
   - use_strength：是否使用强度特征
   - train_val_split_ratio：0.5–0.9，默认 0.7
3. **数据范围**：stock_codes、start_date、end_date（与现有一致）

**提交**：调用 `POST /models/train`，body 包含 `strategy_config`、`stacking_config`。

### 7.2 回测创建页面：选择已训练的 Stacking 模型

**流程**：

1. 用户选择「组合策略」
2. 在「信号整合方法」中选择「Stacking 堆叠泛化」
3. 显示「Stacking 模型」下拉框，选项来自 `GET /models?model_type=stacking_ensemble&status=ready`
4. 用户选择后，将 `stacking_model_id` 写入 `strategy_config`
5. 若未选择模型，提交时校验并提示「请先训练或选择 Stacking 模型」，或走 Fallback 并提示

### 7.3 训练结果展示

**模型列表**（stacking_ensemble 类型）：

- 展示策略权重（如 RSI: 0.35, MACD: 0.65）
- 展示 train_mse、val_mse、train_auc、val_auc
- 展示基策略列表（来自 hyperparameters.strategy_config）

**训练报告弹窗**：

- 策略权重柱状图
- 训练/验证指标对比表
- 过拟合提示（val 明显差于 train 时）

---

## 八、潜在风险与注意事项

### 8.1 策略配置兼容性

- **问题**：回测时的 `strategy_config` 与训练时不一致（策略名、数量、参数不同），元模型不适用
- **对策**：将训练时的 `strategy_config` 存到 model_info.hyperparameters；回测加载时做校验，不一致则报错

### 8.2 特征顺序一致性

- **问题**：预测时特征顺序与训练时不一致，导致错误
- **对策**：序列化时保存 `feature_names` 列表；预测时按该顺序构造 X

### 8.3 过拟合

- **对策**：股票维度划分、逻辑回归 L2、LightGBM 限制深度；在报告中展示 train/val 对比

### 8.4 数据泄露

- **对策**：标签严格用 `close[t+1:t+forward_days]`；特征只用 t 及之前

### 8.5 冷启动

- **问题**：用户未训练 stacking 模型就选 stacking 融合
- **对策**：Fallback 到 weighted_voting + 明确提示；或强制用户先选择/训练模型

### 8.6 训练数据与回测数据重叠

- **建议**：训练数据 end_date 应早于回测 start_date，避免未来信息；可在 UI 或文档中说明

---

## 九、实施优先级（分 Phase）

### Phase 1：训练链路（约 2–3 天）

1. 新建 `StackingEnsembleTrainer`，实现特征构造、股票维度划分、元模型训练、保存
2. 扩展 `ModelTrainingRequest`、`/models/train`，支持 `stacking_ensemble`
3. 联调：从模型管理页面发起 stacking 训练，验证 model_info 写入正确

### Phase 2：回测集成（约 2 天）

1. 新建 `StackingFusion`，实现 `predict`
2. 修改 `BacktestExecutor`：加载 stacking 模型、注入 strategy、在 `_extract_precomputed_signals_to_dict` 中走 stacking 分支
3. 修改 `StrategyPortfolio`、`StrategyFactory` 支持 stacking
4. 联调：用已训练 stacking 模型跑回测，验证信号正确

### Phase 3：前端模型管理（约 1–2 天）

1. `CreateModelForm` 增加 stacking_ensemble 类型及配置区
2. `ModelListTable`、`TrainingReportModal` 展示 stacking 专属信息

### Phase 4：前端回测创建（约 1 天）

1. `PortfolioStrategyConfig` 增加 stacking 选项及模型选择器
2. 任务创建页面传递 `stacking_model_id`

### Phase 5：增强与优化（可选）

- Fallback 时的用户提示优化
- 策略配置兼容性校验增强
- 超参优化支持 stacking

---

## 十、总结

| 维度 | V2 设计要点 |
|------|-------------|
| **架构** | 训练与回测解耦；stacking 作为独立训练类型 `stacking_ensemble`，存入 model_info |
| **训练** | 模型管理 API 发起；特征=信号编码+强度，标签=forward_return；股票维度 70/30 划分 |
| **元模型** | 首选逻辑回归，备选 LightGBM |
| **回测** | 通过 `stacking_model_id` 加载模型；无模型时 Fallback 到 weighted_voting |
| **前端** | 模型管理新增 stacking 训练；回测创建新增 stacking 模型选择 |
| **风险** | 策略兼容性、特征顺序、过拟合、数据泄露需严格把控 |

---

## 附录：与 V1 方案对照

| V1 设计点 | V2 对应 |
|-----------|---------|
| StackingFusion 在回测中训练 | StackingFusion 仅负责推理；训练在模型管理 |
| StackingTrainer.train() 在 Executor 内调用 | StackingEnsembleTrainer 在 `/models/train` 中调用 |
| 元模型不持久化 | 元模型存入 model_info.file_path |
| 时间 70/30 划分 | 股票 70/30 划分（可选保留时间划分为配置项） |
| stacking_config 在 strategy_config 内 | 训练时单独 stacking_config；回测只需 stacking_model_id |
