# official / robust 历史主候选 signal_quality 回填可行性检查

日期：2026-04-14

目标：
- 验证当前历史主候选 `official` / `robust` 虽然数据库 `evaluation_report.signal_quality` 为空，是否已经具备“事后回填”的工程可行性
- 顺手拿到一版真实 validation signal_quality，作为下一步 bridge analysis 的起点

## 1. 检查结果：当前 DB 里确实还是空块

目标模型：
- `official`
  - model_id: `53d9e8ad-e134-4b53-ba50-39a9c91f23df`
  - model_name: `hermes-official-bank-core3-1776037184`
- `robust`
  - model_id: `33b2fd75-af83-4d5f-bc2c-28dbad9fffa2`
  - model_name: `hermes-bank-core3-robust-20260412-230648`

直接查 `backend/data/app.db -> model_info.evaluation_report`：
- 两个模型都还没有 `signal_quality` 顶层字段
- 但都保留了：
  - `file_path`
  - `hyperparameters`
  - `training_data_info.stock_codes/start_date/end_date`

这意味着：
- 回填所需的三件核心输入都在：
  - 模型文件
  - 训练配置/超参
  - 原训练数据窗口与股票池

## 2. 可行性验证方法

使用 `backend/.venv/bin/python`：
1. 从 `model_info` 读取 `file_path` / `hyperparameters` / `evaluation_report.training_data_info`
2. 用 `UnifiedQlibTrainingEngine` 初始化 Qlib 环境
3. 重新准备训练期数据集
4. 调 `_prepare_training_datasets(...)` 复原 train / valid 分段
5. 调 `load_qlib_model(file_path)` 加载历史模型
6. 调 `_evaluate_model(...)` 重新计算 validation `signal_quality`

注意点：
- `load_qlib_model()` 返回的是 `(model, config)` 二元组，不能把整个 tuple 直接传给 `_evaluate_model()`
- 正确做法是先解包出 `model`

## 3. 实测回填结果

### 3.1 official
- model_id: `53d9e8ad-e134-4b53-ba50-39a9c91f23df`
- validation accuracy: `0.6395`
- signal_quality:
  - `ic = -0.0529`
  - `rank_ic = -0.0803`
  - `icir = -0.0728`
  - `rank_icir = -0.1045`
  - `long_short_ann_return = 0.2616`
  - `long_short_ann_sharpe = 1.9815`
  - `long_avg_ann_return = 2.2692`
  - `long_avg_ann_sharpe = 7.2630`
  - `sample_count = 147`
  - `analysis_scope = validation`

解读：
- `official` 的 validation 方向准确率不差
- 但 validation IC / RankIC 是负的
- 这提示它在当前训练切分下，方向命中与横截面排序质量并不一致

### 3.2 robust
- model_id: `33b2fd75-af83-4d5f-bc2c-28dbad9fffa2`
- validation accuracy: `0.6358`
- signal_quality:
  - `ic = 0.3264`
  - `rank_ic = 0.3646`
  - `icir = 0.4143`
  - `rank_icir = 0.4932`
  - `long_short_ann_return = 1.2234`
  - `long_short_ann_sharpe = 7.5013`
  - `long_avg_ann_return = 1.6511`
  - `long_avg_ann_sharpe = 5.0876`
  - `sample_count = 162`
  - `analysis_scope = validation`

解读：
- `robust` 的 validation accuracy 与 `official` 接近
- 但 validation IC / RankIC 明显更强，而且是显著正值
- 这与它在季度 / 短窗里更灵活的实测结论是相互呼应的

## 4. 当前结论

1. 历史主候选 `official` / `robust` 的 `signal_quality` 回填在工程上是可行的
2. 当前不是“缺原始输入”，而只是“还没把回填流程产品化 / API 化”
3. 回填后的第一手结果已经有信息量：
   - `official`：accuracy 尚可，但 validation ranking 质量偏弱甚至为负
   - `robust`：validation ranking 质量明显更强
4. 这说明下一步 bridge analysis 值得继续：
   - 不能只看 accuracy
   - 应把 validation `signal_quality` 与正式任务窗口收益 / 回撤 / IR / rejection reasons 并排看

## 5. 推荐的下一步工程动作

优先级建议：
1. 提供一个可复用的“历史模型 signal_quality 回填”服务函数或脚本
   - 输入：`model_id`
   - 输出：归一化后的 `signal_quality`
   - 可选：直接写回 `model_info.evaluation_report`
2. 给 `/api/v1/models/{model_id}/evaluation-report` 增加按需补算/补齐能力
   - 对旧模型若 `signal_quality` 缺失，可选择：
     - 在线补算后返回
     - 或离线回填后持久化
3. 在正式研究报告里，把以下两组指标并排展示：
   - validation `signal_quality`
   - 正式任务 `with_cost / without_cost` 收益、回撤、IR、trade stats、rejection reasons

一句话：
- 这一步已经证明：`official` / `robust` 的 signal ↔ portfolio bridge analysis 现在可以真正开始做了，不再只是方向建议。
