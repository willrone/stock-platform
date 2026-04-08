# Task Detail Page 拆分说明

## 结构

- `page.tsx`
  - 仅作为 Next.js 路由入口。
  - 负责装配 `useTaskDetailPage()` 与 `TaskDetailPageView`。
- `useTaskDetailPage.ts`
  - 承载任务详情页的数据获取、WebSocket 同步、页面状态、动作处理。
  - 统一输出 `TaskDetailPageModel`，给展示层消费。
- `TaskDetailPageView.tsx`
  - 负责页面骨架：标题区、顶部动作、内容区、删除/保存配置对话框。
- `TaskDetailContent.tsx`
  - 负责主展示区域与侧边栏渲染。
  - 包含回测 / 预测两类任务详情的展示分支。
- `TaskDetailActionPanel.tsx`
  - 负责顶部动作按钮组：刷新、重试、导出、重建、删除。
- `taskDetailUtils.tsx`
  - 负责页面展示所需的纯函数与轻量 UI helper：策略配置提取、状态/方向展示、策略参数渲染。
- `types.ts`
  - 统一声明页面容器输出的 `TaskDetailPageModel` 类型。

## 依赖关系

```text
page.tsx
  ├─ useTaskDetailPage.ts
  │   ├─ stores/useTaskStore
  │   ├─ services/taskService
  │   ├─ services/backtestService
  │   ├─ services/backtestDataAdapter
  │   ├─ services/strategyConfigService
  │   └─ services/websocket
  └─ TaskDetailPageView.tsx
      ├─ TaskDetailActionPanel.tsx
      ├─ TaskDetailContent.tsx
      ├─ taskDetailUtils.tsx
      └─ backtest/common components
```

## 保持不变的约束

- 路由保持为 `tasks/[id]`
- 原有任务详情主要交互保持不变
- 原有数据来源与回测/预测展示 contract 保持不变
- 本次仅做职责拆分；若后续需要继续细拆，可优先从 `TaskDetailContent.tsx` 内部 tab 区块继续下沉
