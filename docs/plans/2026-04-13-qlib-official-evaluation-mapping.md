# Qlib 官方评估口径映射到 stock-platform 的实施方案

> For Hermes: use subagent-driven-development / TDD to implement this plan incrementally. Do not bundle all phases into one giant patch.

目标
- 把 stock-platform 当前“训练 accuracy + 自定义近似 sharpe/return + 正式任务回测”的评估方式，升级为更贴近 Qlib 官方三层评估的结构：
  1. 信号层：IC / Rank IC / ICIR / Rank ICIR / Long-Short / Long-Avg
  2. 组合层：TopkDropout + excess return(with/without cost) + information ratio + max drawdown
  3. 执行层：turnover + pa / pos / ffr（或现阶段可落地的简化代理指标）

架构结论
- 当前 stock-platform 已经具备 ranking 正式回测链路，且 ranking warmup bug 已修复。
- 当前缺口主要在“评估结构”而不是“能不能跑回测”。
- 最优落地方向不是替换现有报告，而是：
  - 在模型评估报告里补 `signal_quality` / `official_signal_analysis`
  - 在回测结果里补 `official_portfolio_analysis` / `benchmark_relative_analysis`
  - 在详细结果与任务 API 中补 `execution_quality_analysis`

官方对照依据（摘要）
- SignalRecord: 保存 `pred.pkl`、`label.pkl`
- SigAnaRecord: 计算并记录
  - IC
  - ICIR
  - Rank IC
  - Rank ICIR
  - 可选：Long-Short Ann Return / Sharpe
  - 可选：Long-Avg Ann Return / Sharpe
- PortAnaRecord: 记录
  - `report_normal.pkl`
  - `positions_normal.pkl`
  - `port_analysis_<freq>.pkl`
  - 风险分析核心项：mean / std / annualized_return / information_ratio / max_drawdown
  - 基于 `return - bench` 和 `return - bench - cost`
- indicator_analysis: 交易执行指标
  - pa
  - pos
  - ffr
  - mean / amount_weighted / value_weighted

当前本地链路现状（和官方差异）
1. `backend/app/services/qlib/unified_qlib_training_engine.py`
   - 训练期 financial metrics 目前是 `returns = y_true * sign(y_pred)` 这类近似方向收益口径
   - 这不是官方的 IC/RankIC 层，也不是官方的组合层
2. `backend/app/services/models/evaluation_report.py`
   - 现有 report schema 有 `training_summary` / `performance_metrics` / `feature_importance` 等
   - 但没有官方风格的 `signal_quality` 分层块
3. `backend/app/services/backtest/reporting/backtest_report_builder.py`
   - 已有 `excess_return_without_cost` / `excess_return_with_cost`
   - 但 with-cost 当前 `information_ratio` 直接复用了 `sharpe_ratio`，不够官方
4. `backend/app/repositories/backtest_detailed_repository.py`
   - 已有详细数据表和 signal stats 聚合入口
   - 适合承载 benchmark-relative 与 execution-quality 新分析块

---

## Phase 1: 先补官方风格“信号质量”层（模型评估报告）

### 目标
让 `GET /api/v1/models/{model_id}/evaluation-report` 除现有训练指标外，再返回一块官方风格信号分析：
- IC
- ICIR
- Rank IC
- Rank ICIR
- Long-Short Ann Return
- Long-Short Ann Sharpe
- Long-Avg Ann Return
- Long-Avg Ann Sharpe
- 可选：对应时间序列摘要（如均值、标准差、样本数）

### 建议新增结构
在 report JSON 顶层新增：

```json
{
  "signal_quality": {
    "ic": 0.0123,
    "icir": 0.74,
    "rank_ic": 0.0181,
    "rank_icir": 0.92,
    "long_short_ann_return": 0.134,
    "long_short_ann_sharpe": 1.21,
    "long_avg_ann_return": 0.082,
    "long_avg_ann_sharpe": 0.88,
    "sample_count": 12345,
    "analysis_scope": "validation"
  }
}
```

### 文件
- 修改：`backend/app/services/qlib/unified_qlib_training_engine.py`
- 修改：`backend/app/services/models/evaluation_report.py`
- 修改：`backend/app/api/v1/models.py`
- 测试：
  - `backend/tests/unit/models/test_training_report_contracts.py`
  - `backend/tests/unit/api/test_model_contract_api.py`
  - 新增建议：`backend/tests/unit/models/test_signal_quality_metrics.py`

### 实施要点
1. 在训练结束后，用验证集 `pred` + `label` 计算官方风格信号指标
   - 优先实现一个纯本地 helper，不强耦合 qlib recorder
   - 指标定义尽量对齐 Qlib：
     - IC: Pearson corr(score, label) 按截面/按日后取均值
     - Rank IC: Spearman corr(score, label) 按截面/按日后取均值
     - ICIR / Rank ICIR: mean / std
     - Long-Short / Long-Avg: 先按 score 排名切头尾，再按日聚合
2. 不要删除现有 `performance_metrics`
   - 保持向后兼容
   - 新增 `signal_quality` 作为“官方风格层”
3. 在 `normalize_report_payload()` 里为旧模型补默认空块
   - 避免历史数据没有该字段时前端崩掉

### 完成标准
- evaluation-report 稳定返回 `signal_quality`
- 老模型缺字段时不报错
- 新模型训练后能真实写入该块

---

## Phase 2: 补官方风格“组合质量”层（任务 / 回测结果）

### 目标
把当前回测报告里的 benchmark-relative 信息整理成明确的官方风格组合分析块，而不是散落字段。

### 建议新增结构
在 `backtest_report` / `/tasks/{id}` / `/tasks/{id}/detailed` 中新增：

```json
{
  "official_portfolio_analysis": {
    "benchmark": "SH000300",
    "excess_return_without_cost": {
      "mean": 0.0003,
      "std": 0.012,
      "annualized_return": 0.08,
      "information_ratio": 0.66,
      "max_drawdown": -0.09
    },
    "excess_return_with_cost": {
      "mean": 0.0002,
      "std": 0.013,
      "annualized_return": 0.06,
      "information_ratio": 0.48,
      "max_drawdown": -0.11
    }
  }
}
```

### 文件
- 修改：`backend/app/services/backtest/reporting/backtest_report_builder.py`
- 修改：`backend/app/repositories/backtest_detailed_repository.py`
- 修改：`backend/app/models/backtest_detailed_models.py`（如果要持久化新增 JSON 块）
- 修改：`backend/app/api/v1/tasks.py` / 详细结果读取逻辑（如必要）
- 测试：
  - `backend/tests/unit/api/test_task_backtest_model_driven.py`
  - 新增建议：`backend/tests/unit/backtest/test_official_portfolio_analysis.py`

### 实施要点
1. 不要把 with-cost information_ratio 继续偷用 sharpe_ratio
   - 用真正的 excess return series 做 `risk_analysis`
2. benchmark 缺失时要明确回退策略
   - 如果没有 benchmark 数据，返回空块或 `benchmark_missing=true`
3. 保持现有 `excess_return_without_cost` / `excess_return_with_cost` 字段兼容
   - 新增 `official_portfolio_analysis` 作为正式统一出口

### 完成标准
- `/tasks/{id}` 和 `/tasks/{id}/detailed` 都能拿到统一组合分析块
- excess return with/without cost 信息结构化且可被前端消费

---

## Phase 3: 补“信号质量 × 回测质量”桥接层

### 目标
让用户能直接看见：
- 某个模型的 IC/RankIC 好不好
- 在某个 ranking 策略参数下回测结果好不好
- 两者是否一致

### 建议新增结构
在 task detailed 响应中新增：

```json
{
  "signal_portfolio_bridge": {
    "model_id": "...",
    "signal_quality_snapshot": {
      "rank_ic": 0.0181,
      "rank_icir": 0.92,
      "long_short_ann_return": 0.134
    },
    "portfolio_quality_snapshot": {
      "excess_return_with_cost_ir": 0.48,
      "annualized_return_with_cost": 0.06,
      "max_drawdown_with_cost": -0.11
    },
    "consistency_hint": "rank_ic_positive_but_portfolio_weak"
  }
}
```

### 文件
- 修改：`backend/app/api/v1/tasks.py`
- 修改：`backend/app/repositories/task_repository.py`（如需拼装）
- 可选：`backend/app/services/models/evaluation_report.py`

### 实施要点
- 先做 lightweight snapshot，不要一开始做复杂因果分析
- `consistency_hint` 用规则生成即可：
  - rank_ic > 0 且组合 IR < 0 -> `signal_positive_portfolio_negative`
  - rank_ic > 阈值 且组合也强 -> `signal_and_portfolio_consistent`

---

## Phase 4: 补官方风格“执行质量”层

### 目标
让回测不只看收益，也看交易执行质量。

### 官方对照
Qlib `indicator_analysis` 关注：
- pa: price advantage
- pos: positive rate
- ffr: fulfill rate
- 支持 mean / amount_weighted / value_weighted

### 在 stock-platform 的现实落地
如果底层执行器暂时没有完整撮合细节，先分两层实现：

#### 4A. 先做可落地代理指标
新增：
- turnover_rate
- average_holding_days
- signal_to_trade_conversion
- rejected_signal_rate
- actionable_signal_rate
- average_trade_cost_bps

#### 4B. 再逐步逼近 Qlib 风格指标
如果后续能拿到更细粒度执行数据，再补：
- pa proxy（相对参考价成交优势）
- ffr（挂单/目标交易量完成率）
- pos（盈利成交比例或正向执行比例，需定义清晰）

### 文件
- 修改：`backend/app/repositories/backtest_detailed_repository.py`
- 修改：`backend/app/services/backtest/reporting/backtest_report_builder.py`
- 可选：`backend/app/models/backtest_detailed_models.py`
- 测试：新增 `backend/tests/unit/backtest/test_execution_quality_analysis.py`

### 建议新增结构
```json
{
  "execution_quality_analysis": {
    "turnover_rate": 0.18,
    "avg_holding_days": 12.4,
    "signal_to_trade_conversion": 0.63,
    "actionable_signal_rate": 0.41,
    "rejected_signal_rate": 0.37,
    "average_trade_cost_bps": 8.5,
    "top_rejection_reasons": [...] 
  }
}
```

---

## Phase 5: 前端展示与排序建议

### 目标
让用户能在 models 和 tasks 页面里直接按官方评估思路看模型，而不是被单一 accuracy 误导。

### 前端建议优先展示顺序
1. 信号层
   - Rank IC
   - Rank ICIR
   - Long-Short Ann Return
2. 组合层
   - excess return with cost annualized return
   - information ratio
   - max drawdown
3. 执行层
   - turnover
   - signal_to_trade_conversion
   - top rejection reasons

### 文件
- 前端可能涉及：
  - `frontend/src/components/models/TrainingReportModal.tsx`
  - `frontend/src/app/models/page.tsx`
  - `frontend/src/app/tasks/[taskId]/...`（按实际路径）
  - `frontend/src/services/dataService.ts`
  - `frontend/src/types/model.ts`
  - `frontend/src/types/task.ts`

### UI 原则
- 不要把 `accuracy` 放在唯一主位
- ranking 模型优先显示 `Rank IC / Rank ICIR`
- 回测页优先显示 with-cost excess return + IR + drawdown
- threshold 和 ranking 要有明显模式标签，避免误读

---

## 推荐实现顺序（务实版）

### 第 1 批：最该先做
1. `signal_quality`（IC / RankIC / ICIR / RankICIR）
2. `official_portfolio_analysis`（with/without cost excess return block）
3. 旧数据兼容归一化

### 第 2 批：很值得做
4. Long-Short / Long-Avg 年化指标
5. execution_quality_analysis 基础代理指标
6. 前端展示块

### 第 3 批：更完整官方化
7. 交易指标分析更细项（pa / pos / ffr proxy）
8. task/model 之间的 signal-portfolio bridge

---

## 风险与注意点

1. 不要把当前 training `sharpe_ratio` 误命名成官方组合 Sharpe
- 它只是近似方向收益指标
- 应保留，但降权解释

2. ranking 与 threshold 的评估要分开展示
- 同一个模型在两种策略范式下的结论可能完全不同

3. 旧 evaluation_report / 旧 task detailed 数据必须兼容
- 新字段缺失时要给默认空块
- API 契约不能被历史数据打崩

4. information_ratio 必须基于 excess return 重新算
- 不要直接复用 sharpe_ratio

---

## 建议新增字段清单

### 模型评估报告（evaluation-report）
- `signal_quality.ic`
- `signal_quality.icir`
- `signal_quality.rank_ic`
- `signal_quality.rank_icir`
- `signal_quality.long_short_ann_return`
- `signal_quality.long_short_ann_sharpe`
- `signal_quality.long_avg_ann_return`
- `signal_quality.long_avg_ann_sharpe`
- `signal_quality.sample_count`
- `signal_quality.analysis_scope`

### 回测 / 任务结果
- `official_portfolio_analysis.benchmark`
- `official_portfolio_analysis.excess_return_without_cost.*`
- `official_portfolio_analysis.excess_return_with_cost.*`
- `execution_quality_analysis.turnover_rate`
- `execution_quality_analysis.avg_holding_days`
- `execution_quality_analysis.signal_to_trade_conversion`
- `execution_quality_analysis.actionable_signal_rate`
- `execution_quality_analysis.rejected_signal_rate`
- `execution_quality_analysis.average_trade_cost_bps`

---

## 验证标准

### 后端
- `GET /api/v1/models/{id}/evaluation-report` 返回 `signal_quality`
- `GET /api/v1/tasks/{id}` 与 `/detailed` 返回 `official_portfolio_analysis`
- ranking / threshold 两种任务都不报错
- 旧模型 / 旧任务缺字段时 API 仍稳定

### 数据正确性
- Rank IC / Rank ICIR 对不同模型能体现差异
- with-cost IR 不再等于简单 sharpe 复用
- execution_quality_analysis 与现有 signal summary 不冲突

### 测试命令
```bash
cd /home/willrone/Projects/stock-platform/backend
./.venv/bin/pytest \
  tests/unit/backtest/test_topk_dropout_trade_mode.py \
  tests/unit/backtest/test_model_topk_dropout_strategy.py \
  tests/unit/api/test_backtest_model_driven.py \
  tests/unit/api/test_task_backtest_model_driven.py \
  tests/unit/models/test_training_report_contracts.py \
  tests/unit/api/test_model_contract_api.py -q
```

---

## 当前最务实的下一步

如果只做一批最值的实现，我建议按这个顺序：
1. 先在训练报告里补 `signal_quality`（尤其 Rank IC / Rank ICIR）
2. 再在回测报告里补 `official_portfolio_analysis`
3. 然后做一个简版前端展示

这样就能最快把 Qlib 官方评估思想真正映射进 stock-platform。
