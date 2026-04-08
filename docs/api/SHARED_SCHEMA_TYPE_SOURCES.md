# Shared API Schema / 前端类型源统一

对应工单：#457 `P1 子单：共享 API schema / 前端类型源统一（backtest + task + model + data）`

## 目标

把 `backtest + task + model + data` 四类核心接口的**前端类型入口**收敛到固定目录，避免继续在 service / store / 组件里散落重复接口定义。

## 统一后的前端类型入口

| 领域 | 前端唯一入口 | 说明 |
|---|---|---|
| task | `frontend/src/types/task.ts` | 任务 DTO 已较集中，继续保留为任务唯一入口 |
| model | `frontend/src/types/model.ts` | 从 `useDataStore.ts` 抽离模型 DTO / 训练进度相关类型 |
| data | `frontend/src/types/data.ts` | 从 `useDataStore.ts`、`dataService.ts` 抽离股票数据、监控、信号、预测类型 |
| backtest | `frontend/src/types/backtest.ts` | 从 `dataService.ts`、`backtestService.ts` 抽离回测请求/结果/详细结果类型 |

## 后端对应来源

| 领域 | 后端来源 | 备注 |
|---|---|---|
| task | `backend/app/api/v1/schemas.py` + `build_task_detail_dto` / `build_task_list_dto` / `build_task_mutation_dto` | 任务 DTO 构建入口 |
| model | `backend/app/api/v1/model_dto.py` | 模型列表/详情/训练进度 DTO |
| data | `backend/app/api/v1/data.py` / `stocks.py` / `signals.py` / `system.py` / `monitoring.py` | 数据、监控、信号相关接口 |
| backtest | `backend/app/api/v1/backtest.py` / `backend/app/api/v1/backtest_detailed.py` | 回测请求、回测详细结果接口 |

## 本次迁移策略

### 1. 最小迁移，不改业务语义
- 只做类型来源收敛与导入路径调整。
- 不主动改变接口字段名、后端响应结构、业务流程。

### 2. 兼容旧调用点
- `dataService.ts` / `backtestService.ts` / `useDataStore.ts` 仍保留 `export type` 转发，避免一次性改爆所有页面。
- 新代码默认直接从 `frontend/src/types/*` 导入。

### 3. 组件侧去掉重复本地接口
- `MobileTaskCard.tsx` / `MobileBacktestCard.tsx` 改为基于 `Task` 组合子集类型。
- 模型页面和模型组件改为直接依赖 `types/model.ts`，不再从 store 反向拿类型。

## 后续约束

1. 新增接口类型时，优先落在 `frontend/src/types/` 对应领域文件。
2. service/store 只消费或转发类型，不再定义同名 DTO。
3. 组件层除非是纯展示态 ViewModel，否则不要再手写 API DTO 结构。
4. 若后端 DTO 发生变化，优先同步这里列出的来源文件，再更新前端统一类型入口。
