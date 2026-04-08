# strict-baseline golden files（15 策略）

## 目标

把 15 个非 ML baseline 策略的基线快照固化成可机读、可回归、可审计的 golden artifacts，作为后续重构的裁判线。

本次固化只记录现状与证据，不调整策略语义、参数或时间窗口。

## 工件位置

- Manifest：`backend/tests/golden/strict_baseline/manifest.json`
- 单策略 golden：`backend/tests/golden/strict_baseline/strategies/*.json`
- 导出脚本：`backend/tests/scripts/export_strict_baseline_goldens.py`
- 校验脚本：`backend/tests/scripts/verify_strict_baseline_golden.py`

## 任务 ID 映射表

| 策略 | task_id |
|---|---|
| stochastic | `87dca06e-84c4-4d09-b4d8-3c8cf22e75a1` |
| cci | `5650fcfd-b429-4d83-b641-5fe0e14447ef` |
| cointegration | `1846214a-881d-464d-af0c-5864f82771a5` |
| multi_factor | `1c88160e-2b4d-47b3-a74d-cbf3dc20bc20` |
| obv | `ca493750-1107-4b6f-9e38-6f8126c1bb55` |
| low_volatility | `8ec31dac-3a89-4587-8e6e-957f5abe2f0a` |
| momentum_factor | `86e9aa33-ae67-4889-8ab7-b439f4904c8c` |
| rsi | `34ab0a39-d54b-4127-ab09-77ef03619dc1` |
| bollinger | `eebddc55-1734-4b38-a072-fa2682af994d` |
| pairs_trading | `63b92d02-130b-49e8-936a-fbb68f9e9597` |
| kdj | `e78a5ab8-5b0b-4bb6-8e30-26b31c26fd54` |
| value_factor | `77b6111d-706e-490b-99a7-a079da46c1b9` |
| mean_reversion | `dcb1b357-26b7-4d63-ab2a-cef8176ae977` |
| moving_average | `d6408834-1c25-4815-b5b8-253be25ebd1a` |
| macd | `149c1268-7893-41d2-ab5a-6134cf8c9c3e` |

## 单策略 golden 内容

每个 `strategies/<strategy>.json` 包含：

- `source_task`：源 task_id、task_name、创建/完成时间
- `config_snapshot`：
  - 时间区间
  - 初始资金 / 手续费 / 滑点
  - 股票池数量与 `sha256`
  - `strategy_config`
  - 完整 config 与 backtest_config 的 `sha256`
- `metric_snapshot`：
  - 顶层关键指标（收益、回撤、Sharpe、成交次数等）
  - `metrics`
  - `cost_statistics`
  - `signal_execution_summary`
- `fingerprints`：
  - `portfolio_history` / `trade_history` / `monthly_returns_detail`
  - `performance_analysis` / `perf_breakdown`
  - `metrics` / `backtest_config`
  - 对应长度与 `sha256`

## 容忍阈值说明

默认校验规则如下：

- **整数计数类字段**：必须精确相等
- **哈希指纹字段**：必须精确相等
- **资金类浮点字段**：绝对误差 `<= 1e-6`
  - `final_value`
  - `initial_cash`
  - `total_commission`
  - `total_slippage`
  - `total_cost`
- **其余浮点指标**：绝对误差 `<= 1e-9`

这套阈值用于容忍极小浮点序列化噪声；若哈希或计数发生变化，默认视为 baseline 语义漂移。

## 使用方式

### 1. 重新导出 golden

```bash
cd backend
python tests/scripts/export_strict_baseline_goldens.py
```

### 2. 自检当前固化工件

```bash
cd backend
python tests/scripts/verify_strict_baseline_golden.py
```

### 3. 校验某个回测任务是否与 baseline 一致

```bash
cd backend
python tests/scripts/verify_strict_baseline_golden.py \
  --task-id 34ab0a39-d54b-4127-ab09-77ef03619dc1
```

如需只看配置/指标，不比较哈希指纹：

```bash
cd backend
python tests/scripts/verify_strict_baseline_golden.py \
  --task-id <task_id> \
  --strategy <strategy_name> \
  --no-strict-hashes
```

## 本次基线口径

- 时间区间固定：`2021-01-01` 到 `2026-02-23`
- 股票池固定：1000 只
- 只固化现状，不借机修正历史任务里已有的策略配置/性能画像
- 后续若要更新 baseline，必须明确说明“为什么 baseline 需要重置”，并重新生成 manifest 与 15 份 strategy golden
